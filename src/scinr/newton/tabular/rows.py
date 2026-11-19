"""tabular/rows.py — Cypher-free row logic of the tabular pipeline.

Routing of a row's values into the composite model, normalization keys and the
helpers that instantiate composites. The write path that uses them lives in
``tabular/graph_writer.py`` (the Cypher, in ``graph/neo4j/tabular.py``).
"""

from __future__ import annotations

import logging
import re
import types as _builtin_types
import typing
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel

from scinr.newton.entity_extraction.schema_composer import _to_snake_case
from scinr.newton.tabular.normalization.detector import (
    extract_source_values_from_dict,
    get_normalization_specs,
    instance_has_normalizable_fields,
)
from scinr.newton.tabular.normalization.engine import NormalizationEngine
from scinr.newton.tabular.normalization.models import NormalizationEntry
from scinr.newton.utils.uid import make_uid

if TYPE_CHECKING:
    from scinr.newton.annotation.models import AnnotationDecision
    from scinr.newton.tabular.models import ColumnMapping


logger = logging.getLogger(__name__)

_ROW_BATCH_SIZE = 500
_PARALLEL_ROW_WRITES = 10

# Factory of a NEW streaming pass over one sheet's data rows: ``row_batches(n)``
# yields lists of at most *n* rows, in file order, one batch alive at a time.
# Every call re-reads the file (see tabular.reader.iter_sheet_batches).
RowBatchFactory = Callable[[int], Iterable[list[list[str]]]]

def _build_row_dict(headers: list[str], row_values: list[str]) -> dict[str, str]:
    """Build {header: value} dict for a single row."""
    return {h: v for h, v in zip(headers, row_values) if h}  # noqa: B905


def _merge_values(raw_values: list[str]) -> list[str]:
    """Merge raw column values mapped to the same target field, deduplicating
    by containment (substring OR word-token subset), processed sequentially in
    the given order. Preserves original formatting (trim only) of surviving values.

    Sequential semantics:
    - If a bigger/superset value was already accepted, a later smaller/subset value
      is skipped entirely (not inserted).
    - If a smaller/subset value was already accepted and a later bigger/superset
      value arrives, the smaller survivor(s) are removed and replaced by the new one.
    - Values with no containment relation to any existing survivor are kept as
      distinct entries, in order of first appearance.

    100% deterministic: same input list always produces the same output list.
    """

    def _normalize(s: str) -> str:
        return " ".join(s.split()).lower()

    survivors: list[str] = []  # original strings (trimmed only), in order
    survivors_norm: list[str] = []  # parallel normalized forms, for comparison only

    for raw in raw_values:
        original = raw.strip()
        if not original:
            continue
        norm = _normalize(original)
        if not norm:
            continue
        norm_tokens = set(norm.split())

        # 1. Is this new value redundant vs. an existing survivor?
        redundant = False
        for s_norm in survivors_norm:
            s_tokens = set(s_norm.split())
            if norm in s_norm or norm_tokens <= s_tokens:
                redundant = True
                break
        if redundant:
            continue

        # 2. Does this new value make existing survivors redundant? (bigger arrives later)
        kept_pairs = []
        for orig_s, s_norm in zip(survivors, survivors_norm, strict=True):
            s_tokens = set(s_norm.split())
            if s_norm in norm or s_tokens <= norm_tokens:
                continue  # drop the smaller/subset survivor
            kept_pairs.append((orig_s, s_norm))
        survivors = [p[0] for p in kept_pairs]
        survivors_norm = [p[1] for p in kept_pairs]

        # 3. Insert the new value
        survivors.append(original)
        survivors_norm.append(norm)

    return survivors


def _merge_and_join(raw_values: list[str], field_name: str, context: str) -> str:
    """Merge raw_values via _merge_values() and join the survivors with "; ".

    Shared by the str-typed assembly paths in _route_row_values() (primary
    str fields, supplementary fields, complementary fields) to avoid
    repeating the merge -> join -> debug-log pattern. Logs a debug message
    when more than one raw value was received, describing the merge outcome.

    `context` is a short human-readable description of where the field lives
    (e.g. "primary", "supplementary", or "complementary 'ClassName'") and is
    interpolated directly before " field %r received ..." in the log message.
    """
    merged = _merge_values(raw_values)
    final_str = "; ".join(merged)
    if len(raw_values) > 1:
        logger.debug(
            "_route_row_values: %s field %r received %d value(s), "
            "%d survived merge -> %r",
            context,
            field_name,
            len(raw_values),
            len(merged),
            final_str,
        )
    return final_str


def _classify_merge_type(annotation: object) -> str:
    """Classify a Pydantic field annotation for _route_row_values() merging.

    Unwraps Optional[X]/Union[X, None] to find the real inner type, then
    classifies it as one of:
    - "str": the inner type is `str`, or the type cannot be determined
      confidently (no annotation, `Any`, complex multi-member Union). Fail-safe
      default.
    - "list_str": the inner type is `list[str]`, or a bare unparametrized
      `list`. Fail-safe default for list-like fields.
    - "other": any other type (int, float, bool, date, datetime, Enum, a
      nested BaseModel, etc.) — not eligible for containment-based merging.
    """
    if annotation is None or annotation is Any:
        return "str"

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Union[...] / Optional[...] (incl. PEP 604 `X | None`) → unwrap
    is_union = origin is typing.Union
    if not is_union and hasattr(_builtin_types, "UnionType"):
        is_union = isinstance(annotation, _builtin_types.UnionType)
    if is_union:
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _classify_merge_type(non_none_args[0])
        # Complex multi-member Union — can't determine confidently.
        return "str"

    if annotation is str:
        return "str"

    if origin is list:
        if not args or args[0] is str:
            return "list_str"
        return "other"

    if annotation is list:
        # Bare `list` with no subscript at all.
        return "list_str"

    return "other"


def _route_row_values(
    row_dict: dict[str, str],
    mapping: ColumnMapping,
    primary_cls: type,
) -> tuple[dict[str, str | list[str]], dict[str, str], dict[str, dict[str, str]]]:
    """Route row dict values to primary, supplementary, and complementary kwargs.

    Returns (primary_kwargs, supplementary_kwargs, comp_kwargs) where comp_kwargs
    is {CamelCaseClassName: {field_name: value}}.

    When multiple columns map to the same (target, field_name), their raw
    values are combined via containment-based deduplication (see
    _merge_values()) instead of the last column silently overwriting the
    others. Columns are processed in mapping.mappings order (their order of
    appearance), which keeps the whole pipeline deterministic: the same
    (row_dict, mapping, primary_cls) always yields the same output.

    Type-aware assembly (primary target only, since only there do we have the
    real Pydantic class to introspect via primary_cls.model_fields):
    - str fields               → "; ".join(merged_values)   (str)
    - list[str] fields         → merged_values               (list[str])
    - any other type           → last raw value wins, unmerged (str), with a
      warning suggesting str/list[str] or a model_validator for type coercion.

    supplementary_kwargs and comp_kwargs entries have no accessible Pydantic
    class at this point, so they are always treated as str: merged via
    _merge_values() and joined with "; ".

    Same routing logic as _instantiate_composite_from_row but returns raw dicts
    instead of instantiating models. Reusable for both pre-scan and instantiation.
    """
    primary_field_set = set(primary_cls.model_fields.keys())

    # Pass 1: collect candidate raw values per (target, field_name), preserving
    # mapping.mappings order (insertion order of dict keys == first-seen order).
    primary_raw: dict[str, list[str]] = {}
    supplementary_raw: dict[str, list[str]] = {}
    comp_raw: dict[str, dict[str, list[str]]] = {}

    for col_mapping in mapping.mappings:
        if col_mapping.confidence == "low":
            continue
        field_name = col_mapping.model_field_name
        if field_name == "__extra__":
            continue

        value = row_dict.get(col_mapping.column_name, "")
        if not value:
            continue

        target = col_mapping.target_model

        if target == "primary":
            if field_name in primary_field_set:
                primary_raw.setdefault(field_name, []).append(value)
            else:
                logger.warning(
                    "_route_row_values: field %r not found in primary model %r — skipping",
                    field_name,
                    primary_cls.__name__,
                )
        elif target == "supplementary":
            supplementary_raw.setdefault(field_name, []).append(value)
        else:
            comp_raw.setdefault(target, {}).setdefault(field_name, []).append(value)

    # Pass 2: assemble final kwargs, merging/deduping and applying type-aware
    # join semantics for primary; str-only merge for supplementary/complementary.
    primary_kwargs: dict[str, str | list[str]] = {}
    supplementary_kwargs: dict[str, str] = {}
    comp_kwargs: dict[str, dict[str, str]] = {}

    for field_name, raw_values in primary_raw.items():
        field_info = primary_cls.model_fields.get(field_name)
        annotation = field_info.annotation if field_info is not None else None
        category = _classify_merge_type(annotation)

        if category == "other":
            final_value: str | list[str] = raw_values[-1]
            if len(raw_values) > 1:
                logger.warning(
                    "_route_row_values: field %r of model %r has an unsupported "
                    "type for value merging (%r) — %d values received, using "
                    "the last one (%r) without combining. Consider using str "
                    "or list[str] for this field if it will receive multiple "
                    "mapped columns in the tabular pipeline, or add a Pydantic "
                    "model_validator to coerce/merge the value after construction.",
                    field_name,
                    primary_cls.__name__,
                    annotation,
                    len(raw_values),
                    final_value,
                )
            primary_kwargs[field_name] = final_value
            continue

        if category == "str":
            primary_kwargs[field_name] = _merge_and_join(raw_values, field_name, "primary")
        else:
            merged = _merge_values(raw_values)
            if len(raw_values) > 1:
                logger.debug(
                    "_route_row_values: primary field %r received %d value(s), "
                    "%d survived merge -> %r",
                    field_name,
                    len(raw_values),
                    len(merged),
                    merged,
                )
            primary_kwargs[field_name] = merged

    for field_name, raw_values in supplementary_raw.items():
        supplementary_kwargs[field_name] = _merge_and_join(raw_values, field_name, "supplementary")

    for target, field_map in comp_raw.items():
        comp_kwargs[target] = {}
        for field_name, raw_values in field_map.items():
            comp_kwargs[target][field_name] = _merge_and_join(
                raw_values, field_name, f"complementary {target!r}"
            )

    return primary_kwargs, supplementary_kwargs, comp_kwargs


def _instantiate_composite_from_row(
    primary_cls: type,
    composite_cls: type,
    primary_field_name: str,
    mapping: ColumnMapping,
    row_dict: dict[str, str],
    comp_cls_map: dict[str, type] | None = None,
) -> BaseModel | None:
    """Build a composite model instance from row data using the column mapping.

    Routing (per ColumnFieldMapping entry):
    - confidence == 'low' OR model_field_name == '__extra__'  → skip
    - target_model == 'primary'       → primary_kwargs[field_name] = value
    - target_model == 'supplementary' → supplementary_kwargs[field_name] = value
    - any other str                   → comp_kwargs[target_model][field_name] = value

    After building primary_instance, for each (class_name, kwargs) in comp_kwargs:
    - Look up cls from comp_cls_map; if found, instantiate with model_construct
      and set on composite via the snake_case field name.

    Empty values are skipped (model uses its own defaults).
    """
    comp_cls_map = comp_cls_map or {}

    primary_kwargs, supplementary_kwargs, comp_kwargs = _route_row_values(
        row_dict, mapping, primary_cls
    )

    # ── Instantiate primary model ──────────────────────────────────────────────
    try:
        primary_instance = primary_cls.model_construct(**primary_kwargs)
    except Exception:
        logger.warning(
            "tabular: could not instantiate '%s' from row data (kwargs: %s)",
            primary_cls.__name__,
            list(primary_kwargs.keys()),
        )
        return None

    # ── Build composite with primary + supplementary + complementary fields ────
    composite_init = {primary_field_name: primary_instance}
    composite_field_names = set(composite_cls.model_fields.keys())

    # Supplementary fields
    for k, v in supplementary_kwargs.items():
        if k in composite_field_names:
            composite_init[k] = v
        else:
            logger.debug("tabular: supplementary field '%s' not in composite schema, skipping", k)

    # Complementary model instances
    for class_name, kwargs in comp_kwargs.items():
        comp_cls = comp_cls_map.get(class_name)
        if comp_cls is None:
            logger.debug(
                "tabular: complementary class '%s' not in comp_cls_map, skipping", class_name
            )
            continue
        try:
            comp_instance = comp_cls.model_construct(**kwargs)
            snake_name = _to_snake_case(class_name)
            if snake_name in composite_field_names:
                composite_init[snake_name] = comp_instance
            else:
                logger.debug(
                    "tabular: complementary field '%s' not in composite schema, skipping",
                    snake_name,
                )
        except Exception as exc:
            logger.warning(
                "tabular: could not build complementary instance '%s': %s", class_name, exc
            )

    try:
        composite_instance = composite_cls.model_construct(**composite_init)
        return composite_instance
    except Exception as exc:
        logger.warning("tabular: could not build composite instance: %s", exc)
        return None


class RowNormalizationKey(NamedTuple):
    """One normalizable field of one row, with its normalization unique key."""

    unique_key: str               # "{target_type_name}:{md5_hash}" — key in engine.result_cache
    is_primary: bool              # True: field of the primary model; False: of a complementary one
    class_name: str               # complementary class name as in comp_class_names (primary: its __name__)
    model_class_name: str         # cls.__name__ of the model owning the field
    field_name: str
    target_type: type[BaseModel]
    source_values: dict[str, Any]


def compute_row_normalization_keys(
    headers: list[str],
    row_values: list[str],
    mapping: ColumnMapping,
    primary_cls: type,
    comp_cls_map: dict[str, type],
    comp_class_names: list[str],
    get_specs_fn: Callable[[type], list] | None = None,
) -> list[RowNormalizationKey]:
    """Normalization keys of ONE row: primary model specs first, then each complementary model.

    Pure function of the row and the mapping — no LLM calls, no instantiation.
    It is the single source of the unique keys: the streaming scan pass (dedup map)
    and the write pass (cache lookups in ``engine.result_cache``) must both use it,
    otherwise a row would not find the result the LLM produced for its key.
    """
    fn = get_specs_fn or get_normalization_specs

    row_dict = _build_row_dict(headers, row_values)
    primary_kwargs, _supp_kwargs, comp_kwargs = _route_row_values(row_dict, mapping, primary_cls)

    keys: list[RowNormalizationKey] = []

    for spec in fn(primary_cls):
        source_values = extract_source_values_from_dict(spec, primary_kwargs, primary_cls)
        if not source_values:
            continue
        keys.append(
            RowNormalizationKey(
                unique_key=(
                    f"{spec.target_type.__name__}:"
                    f"{NormalizationEngine._hash_source_values(source_values)}"
                ),
                is_primary=True,
                class_name=primary_cls.__name__,
                model_class_name=primary_cls.__name__,
                field_name=spec.field_name,
                target_type=spec.target_type,
                source_values=source_values,
            )
        )

    for class_name in comp_class_names:
        comp_cls = comp_cls_map.get(class_name)
        if comp_cls is None:
            continue
        comp_model_kwargs = comp_kwargs.get(class_name, {})
        for spec in fn(comp_cls):
            source_values = extract_source_values_from_dict(spec, comp_model_kwargs, comp_cls)
            if not source_values:
                continue
            keys.append(
                RowNormalizationKey(
                    unique_key=(
                        f"{spec.target_type.__name__}:"
                        f"{NormalizationEngine._hash_source_values(source_values)}"
                    ),
                    is_primary=False,
                    class_name=class_name,
                    model_class_name=comp_cls.__name__,
                    field_name=spec.field_name,
                    target_type=spec.target_type,
                    source_values=source_values,
                )
            )

    return keys


def _build_normalization_dedup_map(
    headers: list[str],
    row_batches: Iterable[list[list[str]]],
    primary_cls: type,
    comp_cls_map: dict[str, type],
    comp_class_names: list[str],
    mapping: ColumnMapping,
    get_specs_fn: Callable[[type], list] | None = None,
) -> tuple[dict[str, NormalizationEntry], int]:
    """Streaming pre-scan: build the global dedup map of unique normalization keys.

    For each row (consumed batch by batch), computes its keys with
    :func:`compute_row_normalization_keys` and records the first occurrence of
    each unique key. No LLM calls, no composite instantiation, and no per-row
    bookkeeping: memory is O(unique keys).

    Returns ``({unique_key: NormalizationEntry}, number_of_rows_scanned)``.
    """
    dedup_map: dict[str, NormalizationEntry] = {}
    rows_scanned = 0

    for batch in row_batches:
        for row_values in batch:
            for key in compute_row_normalization_keys(
                headers, row_values, mapping, primary_cls, comp_cls_map, comp_class_names,
                get_specs_fn,
            ):
                if key.unique_key not in dedup_map:
                    dedup_map[key.unique_key] = NormalizationEntry(
                        instance_id=0,
                        model_class_name=key.model_class_name,
                        field_name=key.field_name,
                        target_type=key.target_type,
                        source_values=key.source_values,
                        unique_key=key.unique_key,
                    )
        rows_scanned += len(batch)

    # Validate comp_kwargs coverage: warn if any comp_class_names have no rows
    # with mapped values (possible mismatch between col_mapping.target_model
    # and decision.complementary_models.model_class)
    for class_name in comp_class_names:
        comp_cls = comp_cls_map.get(class_name)
        if comp_cls is None:
            continue
        if not instance_has_normalizable_fields(comp_cls):
            continue
        # Check if any entry in dedup_map references this class
        has_entries = any(e.model_class_name == class_name for e in dedup_map.values())
        if not has_entries:
            logger.warning(
                "tabular: complementary model '%s' has normalizable fields "
                "but no rows were mapped to it (possible target_model mismatch "
                "between column mapping and decision)",
                class_name,
            )

    return dedup_map, rows_scanned


def _instantiate_composite_plain(
    table_composite_id: str,
    row_index: int,
    headers: list[str],
    row_values: list[str],
    mapping: ColumnMapping,
    decision: AnnotationDecision,
    primary_cls: type,
    composite_cls: type,
    primary_field_name: str,
    comp_cls_map: dict[str, type],
) -> tuple[int, list[str], BaseModel | None, str]:
    """Instantiate a composite model WITHOUT normalization.

    Used for remaining rows where normalization failed or is not applicable.
    """
    row_composite_id = f"{table_composite_id}/row_{row_index + 1}"
    row_dict = _build_row_dict(headers, row_values)
    extraction_uid = make_uid(
        "tabular_extraction",
        row_composite_id,
        decision.matched_model_class or "raw",
    )

    composite_instance = _instantiate_composite_from_row(
        primary_cls=primary_cls,
        composite_cls=composite_cls,
        primary_field_name=primary_field_name,
        mapping=mapping,
        row_dict=row_dict,
        comp_cls_map=comp_cls_map,
    )

    return (row_index, row_values, composite_instance, extraction_uid)


def _instantiate_composite_with_normalization(
    table_composite_id: str,
    row_index: int,
    headers: list[str],
    row_values: list[str],
    mapping: ColumnMapping,
    decision: AnnotationDecision,
    primary_cls: type,
    composite_cls: type,
    primary_field_name: str,
    comp_cls_map: dict[str, type],
    comp_class_names: list[str],
    engine: NormalizationEngine,
    get_specs_fn: Callable[[type], list] | None = None,
    row_keys: list[RowNormalizationKey] | None = None,
) -> tuple[int, list[str], BaseModel | None, str]:
    """Instantiate a composite model with cached normalization applied.

    Builds the composite from raw row values, then applies cached normalization
    results from the engine for all normalizable fields.

    The unique keys come from :func:`compute_row_normalization_keys` (the same
    function as the pre-scan), so the cache lookups are consistent. Pass
    *row_keys* to reuse keys the caller already computed for this row.
    """
    row_composite_id = f"{table_composite_id}/row_{row_index + 1}"
    row_dict = _build_row_dict(headers, row_values)
    extraction_uid = make_uid(
        "tabular_extraction",
        row_composite_id,
        decision.matched_model_class or "raw",
    )

    if row_keys is None:
        row_keys = compute_row_normalization_keys(
            headers, row_values, mapping, primary_cls, comp_cls_map, comp_class_names,
            get_specs_fn,
        )

    composite_instance = _instantiate_composite_from_row(
        primary_cls=primary_cls,
        composite_cls=composite_cls,
        primary_field_name=primary_field_name,
        mapping=mapping,
        row_dict=row_dict,
        comp_cls_map=comp_cls_map,
    )

    if composite_instance is None:
        return (row_index, row_values, None, extraction_uid)

    for key in row_keys:
        if key.is_primary:
            # Apply cached normalization to the primary instance
            target_instance = getattr(composite_instance, primary_field_name, None)
        else:
            # Apply cached normalization to a complementary instance
            target_instance = getattr(composite_instance, _to_snake_case(key.class_name), None)
        if target_instance is None:
            continue
        applied = engine.apply_cached_to_instance(target_instance, key.field_name, key.unique_key)
        if not applied:
            if key.is_primary:
                logger.warning(
                    "tabular: cache miss for row %d, field '%s', key '%s' "
                    "(normalization LLM may have failed)",
                    row_index, key.field_name, key.unique_key,
                )
            else:
                logger.warning(
                    "tabular: cache miss for row %d, comp '%s' field '%s', key '%s'",
                    row_index, key.class_name, key.field_name, key.unique_key,
                )

    return (row_index, row_values, composite_instance, extraction_uid)


def _sanitize_rel_type(col_name: str) -> str:
    """Convert a column name to a valid graph relationship type."""
    sanitized = re.sub(r"[^A-Z0-9_]", "_", col_name.upper().strip())
    if not sanitized or sanitized[0].isdigit():
        sanitized = "COL_" + sanitized
    return f"HAS_{sanitized}"
