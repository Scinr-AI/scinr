"""
Detector de campos marcados para normalización automática.

Inspecciona el schema de un modelo Pydantic y retorna los campos
que tienen `normalization_model: True` en su `json_schema_extra`.
"""

from __future__ import annotations

import logging
import types as _builtin_types
import typing
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from scinr.newton.tabular.normalization.models import NormalizationSpec

logger = logging.getLogger(__name__)


def get_normalization_specs(model_class: type[BaseModel]) -> list[NormalizationSpec]:
    """
    Dada una clase Pydantic, retorna los campos marcados para normalización.

    Un campo se considera marcable si:
    - Su `json_schema_extra` contiene `normalization_model: True`
    - Su tipo es (o contiene) un modelo Pydantic (no str, int, etc.)

    Si `normalization_source_fields` no se especifica o está vacío,
    se usan TODOS los campos del modelo como fuente de datos.
    """
    specs: list[NormalizationSpec] = []

    for field_name, field_info in model_class.model_fields.items():
        extra = _get_json_schema_extra(field_info)
        if not extra.get("normalization_model", False):
            continue

        # Extraer el tipo objetivo (el modelo normalizado)
        target_type = _extract_target_type(field_info.annotation)
        if target_type is None:
            logger.warning(
                "Field '%s' in %s has normalization_model=True but no "
                "Pydantic type found. Skipping.",
                field_name,
                model_class.__name__,
            )
            continue

        # Extraer campos fuente
        source_fields = extra.get("normalization_source_fields", []) or []

        specs.append(
            NormalizationSpec(
                field_name=field_name,
                target_type=target_type,
                source_fields=source_fields if source_fields else None,
            )
        )

    return specs


def instance_has_normalizable_fields(model_class: type[BaseModel]) -> bool:
    """Retorna True si el modelo tiene al menos un campo normalizable."""
    return len(get_normalization_specs(model_class)) > 0


def extract_source_values(
    spec: NormalizationSpec,
    instance: BaseModel,
) -> dict[str, Any]:
    """
    Extrae los valores de los campos fuente de una instancia.

    Si spec.source_fields es None, usa TODOS los campos del modelo
    (excepto los que ya son modelos normalizados o el propio campo).
    """
    values: dict[str, Any] = {}

    if spec.source_fields:
        # Campos fuente explícitos
        for src_field in spec.source_fields:
            val = getattr(instance, src_field, None)
            if val is not None and val != "":
                values[src_field] = val
    else:
        # Todos los campos del modelo (fuente implícita)
        for field_name, field_info in instance.model_fields.items():
            # Saltar el propio campo normalizado
            if field_name == spec.field_name:
                continue
            # Saltar campos que ya son modelos Pydantic anidados
            ann_type = _extract_target_type(field_info.annotation)
            if ann_type is not None:
                continue
            val = getattr(instance, field_name, None)
            if val is not None and val != "":
                values[field_name] = val

    return values


# ─── Helpers internos ────────────────────────────────────────────────────────


def _get_json_schema_extra(field_info: FieldInfo) -> dict:
    """Extrae json_schema_extra de forma segura."""
    extra = getattr(field_info, "json_schema_extra", None) or {}
    if isinstance(extra, dict):
        return extra
    return {}


def _extract_target_type(annotation: object) -> type[BaseModel] | None:
    """
    Extrae el tipo Pydantic de una anotación.

    Sigue el mismo patrón que model_resolver._extract_model_classes().

    Maneja:
    - NormalizedSubstance → NormalizedSubstance
    - NormalizedSubstance | None → NormalizedSubstance
    - list[NormalizedSubstance] → NormalizedSubstance
    - Annotated[NormalizedSubstance, ...] → NormalizedSubstance
    """
    if annotation is None:
        return None

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Annotated[T, ...] → desempaquetar
    if origin is typing.Annotated:
        return _extract_target_type(args[0]) if args else None

    # list[T] → desempaquetar
    if origin is list:
        return _extract_target_type(args[0]) if args else None

    # Union (X | None o typing.Union[X, None])
    is_union = origin is typing.Union
    if not is_union and hasattr(_builtin_types, "UnionType"):
        is_union = isinstance(annotation, _builtin_types.UnionType)
    if is_union:
        for arg in args:
            if arg is not type(None):
                result = _extract_target_type(arg)
                if result is not None:
                    return result
        return None

    # Tipo directo
    if _is_pydantic_model(annotation):
        return annotation

    return None


def _is_pydantic_model(annotation: object) -> bool:
    """Retorna True si la anotación es una clase Pydantic BaseModel."""
    try:
        return isinstance(annotation, type) and issubclass(annotation, BaseModel)
    except TypeError:
        return False
