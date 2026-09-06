# Implementation Plan — Graph Navigation API (`scinr.newton.navigation`)

Status: **ready to build** · Companion to [`implement-navigation.md`](implement-navigation.md) (the design) ·
Target release: **`0.3.8`** · Suggested branch: `feature/0.3.8`

This document turns the finalized design into an ordered, file-by-file build plan,
**including every docs deliverable for the mkdocs site** (user guides + API
reference + nav wiring). Read the design first; this plan does not restate the
API surface, only how to construct it.

---

## 0. Corrections & deltas vs. the design doc

Small factual adjustments discovered while grounding the plan in the current tree.
Apply these; the rest of the design stands unchanged.

| # | Design doc says | Reality / adjustment |
|---|---|---|
| 0.1 | "Target release `0.3.7`+" | `0.3.7` is already taken by the provenance-metadata work (see `CHANGELOG.md` `## [0.3.7]`, `pyproject.toml version = "0.3.7"`). **Navigation ships in `0.3.8`.** |
| 0.2 | Implies a Pydantic config model (`ConfigDict`, `Literal` on the model) | `ScinrConfig` is a plain `@dataclass` (`config.py:97`). `graph_backend` is a dataclass field with a default; validation happens **inside `configure()`** exactly like `storage_backend` (`config.py:406-411`), not via a model validator. |
| 0.3 | `navigation/errors.py` holds `NavigationError` | Put the new exception classes in the existing hierarchy file `src/scinr/newton/exceptions.py` (next to `ScinrError`, `StorageError`, …) and re-export them from `navigation/__init__.py`. Keep `navigation/errors.py` as a thin re-export module so design references still resolve. |
| 0.4 | "holds the async driver from `ingest/config.py::get_async_driver()` (or its own)" | Default path: **reuse** `get_async_driver()` (module-level singleton, already reset by `configure()` via `_reset_async_driver_singleton()`). `Neo4jGraphNavigator.close()` must **not** close that shared singleton — only close a driver the navigator constructed itself (opt-in `own_driver=True`). |
| 0.5 | `mkdocstrings` docstring style | Site is configured for **Google-style** docstrings (`mkdocs.yml` → `docstring_style: google`). Every public class/method/function gets a Google-style docstring or it renders empty in the API reference. |
| 0.6 | "`describe_node` … include ancestors" etc. | No change, just note: the API-reference page is auto-generated, so **docstring quality is a shipping requirement**, tracked in the per-phase DoD below. |

---

## 1. Deliverable inventory

### 1.1 New source modules (`src/scinr/newton/navigation/`)

| File | Responsibility | Depends on |
|---|---|---|
| `__init__.py` | Public surface: re-export `get_graph_navigator`, `graph_navigator`, `GraphNavigator`, all `navigation.models` types, all `filters` operators, the exception classes. | everything below |
| `base.py` | `GraphNavigator` ABC — ~50 `@abstractmethod async def` + the non-abstract `execute_raw` / `execute_raw_one` raisers + `dialect: ClassVar[str] = "none"` + async context-manager dunders. | `models`, `filters`, `exceptions` |
| `models.py` | Engine-neutral Pydantic v2 return types (§5 of the design). `_Base(frozen=True, extra="ignore")` with `raw: dict`. | `pydantic` |
| `filters.py` | `Op` frozen base + the 14 operator classes. `normalize_where(where) -> dict[str, Op]` (bare value → `Eq`). Property-key validator `^[A-Za-z_][A-Za-z0-9_]*$`. | `pydantic` |
| `factory.py` | `async def get_graph_navigator() -> GraphNavigator` (reads `cfg.graph_backend`), `@asynccontextmanager graph_navigator()`. | `config`, `neo4j/navigator` |
| `errors.py` | Thin re-export: `from scinr.newton.exceptions import NavigationError, GraphConnectionError, UnsupportedOperationError`. | `exceptions` |
| `pages.py` | Source-text bridge (Group I). Free async functions taking `nav` + id; uses `storage.factory.get_storage()`. | `storage`, `base` |
| `neo4j/__init__.py` | Re-export `Neo4jGraphNavigator`. | |
| `neo4j/navigator.py` | `Neo4jGraphNavigator(GraphNavigator)` — the only backend. Implements every abstract method; `dialect = "cypher"`; overrides `execute_raw*`. Uses `execute_read` sessions + `with_neo4j_retry`. | `ingest/config`, `utils/neo4j_retry`, `queries`, `_safe`, `_translate`, `models` |
| `neo4j/queries.py` | Cypher constant strings, one per method (or grouped). No f-string interpolation except the `_safe`-validated depth token. | — |
| `neo4j/_safe.py` | `safe_ident(s)`, `resolve_depth(d) -> int` (`None → DEFAULT_MAX_DEPTH = 10`, explicit verbatim, must be `> 0`), `assert_read_only(cypher)` (comment-strip + write-keyword regex from design §3.6). | — |
| `neo4j/_translate.py` | `translate_where(where: dict[str, Op]) -> tuple[str, dict]` — each `Op` → `(predicate_fragment, params)`; property key via `safe_ident`; values parameterised. | `filters`, `_safe` |

### 1.2 Edited source files

| File | Edit |
|---|---|
| `src/scinr/newton/exceptions.py` | Add `NavigationError(ScinrError)`, `GraphConnectionError(NavigationError)`, `UnsupportedOperationError(NavigationError)` with docstrings. |
| `src/scinr/newton/config.py` | `ScinrConfig`: add `graph_backend: str = "neo4j"` (in a new `# Graph navigation` section near Neo4j fields). `configure()`: add param `graph_backend: Literal["neo4j"] | None = None`; add resolution `resolved_graph_backend = graph_backend or os.getenv("GRAPH_BACKEND", "neo4j")` + validation block mirroring `config.py:406-411`; pass into `ScinrConfig(...)`; add to the startup log line (`config.py:~639`). Update the `configure()` docstring Args. |
| `src/scinr/newton/__init__.py` | Import + `__all__` the navigation public surface (`get_graph_navigator`, `graph_navigator`, `GraphNavigator`, `NavigationError`, `GraphConnectionError`, `UnsupportedOperationError`, and the model/filter namespaces — see §5.3). Bump `__version__` string here if it is used (currently `"0.2.0"`, stale; leave unless the release process touches it). |
| `pyproject.toml` | `version = "0.3.8"`. |
| `CHANGELOG.md` | New `## [0.3.8]` section — see §7.3. |

### 1.3 New test files

| File | Contents |
|---|---|
| `tests/unit/test_navigation_filters.py` | `normalize_where` sugar, key-regex rejection, each `Op` shape, frozen-ness. |
| `tests/unit/test_navigation_safe.py` | `safe_ident` accept/reject, `resolve_depth` (`None→10`, `3→3`, `0`/negative → error, `100→100` no cap), `assert_read_only` (comment strip, every write keyword, `CALL {} IN TRANSACTIONS`). |
| `tests/unit/test_navigation_translate.py` | `translate_where` → predicate + params for every operator; params never interpolated. |
| `tests/unit/test_navigation_base.py` | `execute_raw` / `execute_raw_one` base raise `UnsupportedOperationError`; `dialect="none"`; ABC cannot be instantiated; `FakeGraphNavigator` (test double) satisfies the ABC. |
| `tests/unit/test_navigation_factory.py` | Default → `Neo4jGraphNavigator`; unknown `graph_backend` → `ConfigurationError`; `graph_navigator()` CM closes; unreachable engine → `GraphConnectionError` (mock `ping`). |
| `tests/unit/test_navigation_neo4j_queries.py` | Per-method: mock `AsyncDriver`/session, assert the Cypher constant used + params dict + result mapping to the right `*Ref`. Covers Groups A–H. `execute_raw` write-guard + `dialect` mismatch. |
| `tests/unit/test_navigation_models.py` | Model construction, `frozen`, `.raw` default, tree recursion types, `model_dump()` round-trip. |
| `tests/unit/test_navigation_pages.py` | `get_storage` unset → `StorageError`; node with no `source_page_ids` → `[]`; happy path with a fake storage pair. |
| `tests/integration/test_navigation_neo4j.py` | `@pytest.mark.integration`. Seed a tiny graph (folder→doc→structure→info_unit→extraction→instances→entities + a 2nd version), then exercise one method per group end-to-end. Skips if `NEO4J_*` env not set. |

`FakeGraphNavigator` lives in `tests/unit/_navigation_fakes.py` (importable helper, not a test module) so both `test_navigation_base.py` and any future backend-agnostic suite reuse it.

### 1.4 Documentation deliverables (mkdocs) — full list

| # | Path | Type | Action |
|---|---|---|---|
| D1 | `docs/user-guides/graph-navigation.md` | User guide (hand-written) | **New.** Outline in §6.1. |
| D2 | `docs/api/navigation.md` | API reference (mkdocstrings `:::`) | **New.** Outline in §6.2. |
| D3 | `mkdocs.yml` | Nav wiring | Add D1 under **User Guides**, D2 under **API Reference**. §6.3. |
| D4 | `docs/api/index.md` | API index | Add a `- [Navigation](navigation.md): …` bullet to **Core Modules**. |
| D5 | `docs/configuration.md` | Config reference | Add `GRAPH_BACKEND` row under **### Neo4j** (Environment Variables), a `graph_backend` entry under **### Neo4j Parameters** (Programmatic Configuration), and a row in **## Complete Reference: All Settings**. §6.4. |
| D6 | `docs/architecture.md` | Architecture | New subsection **"Graph Navigation Layer"** (after §8 Storage Backends, or a new §12); add `graph_backend` to §4 "Key Configuration Parameters"; add `navigation/` to the §5 module-structure tree. §6.5. |
| D7 | `docs/user-guides/neo4j-graph.md` | Existing graph guide | Add a short **"Navigating from Python"** callout near **## Query Patterns** pointing to D1 (the raw-Cypher patterns and the typed API are complementary). |
| D8 | `docs/index.md` | Landing page | Add one line to **## Key Features** and to the **## Documentation** list. |
| D9 | `docs/user-guides/quick-start.md` | Existing quick start | Add a closing pointer: "To read the graph back, see [Graph Navigation](graph-navigation.md)." (1–2 lines, optional but recommended.) |
| D10 | `README.md` (repo root) | Repo readme | One bullet under the feature list + a link to the published guide. Keep minimal. |

No new `stylesheets` or plugins are required — `mkdocstrings` + `material` already cover it.

---

## 2. Build order (phased)

Each phase is independently mergeable and leaves `main` green. Phases map 1:1 to
the design's §7 but with concrete task lists and the docs work pulled earlier
(D1/D2 get **stubs** in Phase 1 and grow each phase, instead of a big-bang doc in
Phase 4).

### Phase 1 — Abstraction + MVP (Groups A, B, C + `where=` + `execute_raw`)

**Config & scaffolding**
1. `exceptions.py`: add the three exception classes (+ Google docstrings).
2. `config.py`: `graph_backend` field + `configure()` param + env resolution + validation + startup log + docstring. Unit-test in `tests/unit/test_config.py` (extend, don't replace): default `"neo4j"`, env override, invalid value → `ConfigurationError`.
3. Create `navigation/` package tree (all files from §1.1) with real `models.py`, `filters.py`, `base.py`, `errors.py`, `factory.py`, and `neo4j/{_safe,_translate,queries}.py`. `neo4j/navigator.py` implements **only** Group A/B/C methods; every other abstract method may `raise NotImplementedError("navigation phase 2")` **temporarily** — but the ABC still declares them so the surface is frozen from day one. (Alternative: split the ABC into mixins per group and add them phase by phase. Decision: **one flat ABC, declared complete in Phase 1**, bodies filled in later — keeps `FakeGraphNavigator` and typing stable.)
4. `_safe.py` + `_translate.py` fully implemented and unit-tested now (they are used by every later phase).

**Methods (design §4)**
- Group A: `list_root_documents` / `count_root_documents`, `get_one_document`, `get_documents`, `document_exists`, `get_document_children`, `get_document_tree`, `get_document_parent`, `get_document_ancestors`, `get_document_leaves`, `iter_document_descendants`, `list_document_versions`, `get_latest_version`, `get_version_chain`. (`get_document_stats` → Phase 2.)
- Group B: `get_structure_nodes` / `count_structure_nodes`, `get_root_structure_nodes`, `get_structure_node`, `get_child_nodes`, `get_structure_subtree`, `get_parent_node`, `get_node_ancestors`, `get_node_path`, `get_document_of_node`, `get_sibling_nodes`, `find_structure_nodes`, `get_nodes_by_theme`. (`describe_node` → Phase 2, needs extraction/decision joins.)
- Group C: `get_info_units`, `get_document_info_units` / `count_info_units`, `get_info_unit`, `get_node_for_info_unit`. (`search_info_units` → Phase 3.)
- `execute_raw` / `execute_raw_one`: base raisers in `base.py`; Neo4j override in `navigator.py` with `assert_read_only` + READ tx + `dialect=` guard.

**Docs (Phase 1 slice)**
- D1 `graph-navigation.md`: sections *Overview*, *Configuration (`graph_backend`)*, *Quick start*, *Documents & folders*, *Structure nodes*, *InfoUnits*, *Filtering with `where=`*, *Raw queries (escape hatch)*. Mark instance/entity/introspection sections "coming in 0.3.8" placeholders or omit until Phase 2/3.
- D2 `navigation/navigation.md`: `:::` blocks for `factory`, `base`, `models`, `filters`, and `neo4j.navigator`. (mkdocstrings renders whatever exists; unimplemented method bodies still have signatures + docstrings.)
- D3 nav wiring (both entries).
- D4, D5, D7 (callout), D8, D9, D10.

**Definition of done (Phase 1)**
- `tests/unit/test_navigation_{filters,safe,translate,base,factory,models,neo4j_queries}.py` pass; coverage of the shipped methods.
- `FakeGraphNavigator` implements the full ABC.
- `factory` returns `Neo4jGraphNavigator` by default; invalid `graph_backend` → `ConfigurationError`.
- `execute_raw` rejects a `MERGE`/`CALL {} IN TRANSACTIONS` string and a wrong `dialect=`.
- `mkdocs build --strict` passes (no broken links / missing nav targets).
- `ruff` + `mypy` clean on `navigation/`.
- Public surface importable: `from scinr.newton.navigation import get_graph_navigator, GraphNavigator, Eq, Gte, ...`.

### Phase 2 — Instances & annotation (Groups D, E + `get_document_stats`, `describe_node`)

**Methods**
- Group E in full: `get_extraction_result`, `get_document_extraction_results`, `get_node_instances`, `get_document_instances` / `count_document_instances`, `get_instances_by_class` / `count_instances_by_class`, `get_instance`, `get_instance_by_key` (+ exposed `normalize_key()` reusing `utils/uid.py::make_instance_uid`), `get_structure_nodes_for_instance`, `get_documents_for_instance`, `get_extraction_results_for_instance`, `get_instance_parents`, `get_instance_children`, `get_instance_subtree`, `get_instance_relationships`, `get_related_instances`, `find_shell_instances`, `list_instance_relationship_types`.
- Group D: `get_model_decision`, `get_document_model_decisions`, `get_nodes_by_annotated_model`, `get_unannotated_nodes`, `get_proposed_models`, `get_annotation_coverage`.
- `get_document_stats`, `describe_node`.
- Containment-only traversal guard (`type(r) STARTS WITH 'HAS_' OR type(r) = 'REFERENCES'`) centralised as a query fragment constant in `queries.py` and reused.

**Tests**
- Extend `test_navigation_neo4j_queries.py`; add `tests/integration/test_navigation_neo4j.py` (seeded graph) — exercises the instance→owner list semantics (dedup instance reachable from 2 nodes; shell instance → `[]`).

**Docs**
- D1: fill *Extraction & ModelInstances* (the core-request section — mirror the design's Spanish examples: instances of a document, of a structure node, by `model_class` + property filter, jump back to owning `StructureNode`(s)), *Annotation decisions*, *Document statistics*, *`describe_node`*.
- D2: add `:::` for any new public helpers (e.g. `normalize_key`).
- D6 architecture subsection first draft.

**DoD**: integration test green against a real Neo4j; `describe_node` / `get_document_stats` return fully-populated models; docs updated; `mkdocs build --strict` green.

### Phase 3 — Entities, introspection, power tools (Groups F, G, H + `search_info_units`)

**Methods**
- Group F: `get_instance_entities`, `get_node_entities`, `get_document_entities`, `list_entity_labels`, `get_labeled_entities`, `get_labeled_entity`, `get_instances_referencing_entity`, `get_nodes_referencing_entity`, `get_entity_relationships`, `get_related_entities`, `get_triples`, `get_document_triples`, `get_entity_triples`.
- Group G: `list_catalog_models`, `list_model_classes_in_use`, `get_model_properties`, `list_node_roles`, `list_themes`, `list_relationship_types`, `list_node_labels`, `get_graph_summary`.
- Group H: `neighbors`, `shortest_path`, `subgraph` (+ `NodeSelector` handling in `_safe`/`_translate`).
- `search_info_units`: Neo4j fulltext (`db.index.fulltext.queryNodes` over `infoUnitTitle` / `infoUnitDescription`), `field ∈ {"title","description","both"}`; documented generic contract (substring fallback + `score=1.0` for a backend without FTS — N/A today but stated).

**Tests**: query-level for F/G/H; integration adds an entity + triple + fulltext assertion.

**Docs**: D1 gains *Entities & triples*, *Schema introspection*, *Power tools (`neighbors`/`shortest_path`/`subgraph`)*, *Full-text search*. D2 stable.

### Phase 4 — Source-text bridge + polish

**Methods**: Group I (`navigation/pages.py`) — `get_node_source_page_ids`, `get_node_source_text`, `get_document_source_text`, `get_info_unit_source_text`. Fold `utils/document_resolver.py::resolve_leaf_document_names_async` usage into `get_document_leaves` if it de-dups logic.

**Optional**: `SyncGraphNavigator` façade (`asyncio.run` per call) — only if requested; not blocking the release.

**Docs**: D1 *Reading source text* section (+ the storage-backend prerequisite admonition). Final pass on D1/D2/D6. D10 README. Cross-link check.

**Release**: `pyproject.toml` → `0.3.8`; `CHANGELOG.md` `## [0.3.8]`; `__init__.py` surface final; `mkdocs build --strict`; tag per repo process.

---

## 3. Key implementation details to get right

### 3.1 Connection lifecycle (`Neo4jGraphNavigator`)

```python
class Neo4jGraphNavigator(GraphNavigator):
    dialect = "cypher"

    def __init__(self, *, driver: AsyncDriver | None = None, database: str | None = None):
        self._external_driver = driver
        self._driver = driver
        self._owns_driver = driver is None
        self._database = database  # None → cfg.neo4j_database or server default

    async def connect(self) -> None:
        if self._driver is None:
            from scinr.newton.ingest.config import get_async_driver
            self._driver = get_async_driver()   # shared singleton — do NOT close
            self._owns_driver = False
        await self.ping()

    async def ping(self) -> bool:
        try:
            async with self._session() as s:
                await s.run("RETURN 1")
            return True
        except Exception as exc:
            raise GraphConnectionError(str(exc)) from exc

    async def close(self) -> None:
        if self._owns_driver and self._driver is not None:
            await self._driver.close()
        self._driver = None

    def _session(self):
        cfg = get_config()
        return self._driver.session(
            database=self._database or cfg.neo4j_database or None,
            default_access_mode=neo4j.READ_ACCESS,
        )
```

- Every read runs inside `session.execute_read(lambda tx: tx.run(CYPHER, **params).data())` wrapped by `with_neo4j_retry`.
- `configure()` resets the shared async-driver singleton, so a long-lived navigator that took the singleton in `connect()` should re-fetch it per operation **or** document that callers must re-create the navigator after a re-`configure()`. Decision: **re-fetch `get_async_driver()` per session acquisition** when `not self._owns_driver` (cheap, singleton) so re-config is transparent.

### 3.2 `execute_raw` override (Neo4j)

```python
async def execute_raw(self, query, params=None, *, dialect=None):
    if dialect is not None and dialect != self.dialect:
        raise NavigationError(f"execute_raw called with dialect={dialect!r} on a {self.dialect!r} backend")
    assert_read_only(query)              # _safe.py — comment strip + write-keyword regex → NavigationError
    async with self._session() as s:
        res = await s.execute_read(lambda tx: tx.run(query, **(params or {})).data())
    return res
```

`execute_raw_one` = same, returns `res[0] if res else None`.

### 3.3 Depth interpolation

Only place a value is ever formatted into Cypher. `resolve_depth()` returns a
validated positive `int`; `queries.py` templates use `f"...*1..{n}..."` where `n`
is that int. Nothing else is interpolated — all identifiers via `safe_ident`, all
values via params.

### 3.4 `where=` translation

`translate_where({"confidence": Gte(0.8), "code": In(["A","B"])})` →
`("mi.`confidence` >= $w_confidence AND mi.`code` IN $w_code", {"w_confidence": 0.8, "w_code": ["A","B"]})`.
Param names are `w_<safe_ident(key)>`; collisions impossible because keys are unique in a dict.
`IsNull` → `mi.`x` IS NULL` (no param). `Regex` → `mi.`x` =~ $w_x`.

### 3.5 Return-model mapping

One private `_row_to_<Model>(record: dict) -> Model` helper per model in
`navigator.py` (or a small `_map.py`). Each sets `raw=record` (or the relevant
sub-dict). Trees built by recursive assembly from a flat `path`/`collect` query,
not N+1.

### 3.6 Reuse checklist (design §3.8)

- `utils/uid.py::make_instance_uid` → `get_instance_by_key`.
- `utils/neo4j_retry.with_neo4j_retry` → every read.
- `ingest/config.py::get_async_driver` → default driver.
- `storage/factory.py::get_storage` → `navigation/pages.py`.
- `config.get_config()` → backend selection, database name.
- `utils/document_resolver.py` → `get_document_leaves` (Phase 4).

---

## 4. Testing strategy

| Layer | Tooling | What it locks down |
|---|---|---|
| Pure units (`filters`, `_safe`, `_translate`, `models`) | plain `pytest` | operator sugar, key/ident regexes, depth resolution, read-only guard, model invariants |
| Backend-agnostic contract | `FakeGraphNavigator` (in-memory dict graph) in `tests/unit/_navigation_fakes.py` | the ABC is implementable without Neo4j; return-type shapes; arity rules (list vs `| None`) |
| Neo4j query units | `unittest.mock` `AsyncDriver` → `AsyncSession` → `AsyncResult` returning canned `.data()` rows | correct Cypher constant chosen, correct params, correct row→model mapping, `execute_raw` guard |
| Integration | `@pytest.mark.integration`, real Neo4j from `NEO4J_*` env, `setup_schema()` + a seed fixture | end-to-end traversal correctness, dedup/shell instance list semantics, version scoping, fulltext |

- Integration test **skips** (not fails) when `NEO4J_USER`/`NEO4J_PASSWORD` unset — match the existing `tests/integration/` convention.
- Seed fixture builds: 1 folder → 2 child docs (`v1`+`v2` on one) → 3 structure nodes (section/table/row) → 2 info units → 1 model decision → 1 extraction result → 2 model instances (one with `instance_key`, reachable from both docs) → 1 shell instance → 2 labeled entities + 1 `REFERENCES` + 1 level-2 rel → 1 triple.
- Add `navigation` to any coverage config; target ≥ 90 % line coverage on `navigation/` excluding `queries.py` constants.

---

## 5. Public API surface

### 5.1 New exceptions (`exceptions.py`)

```python
class NavigationError(ScinrError):
    """Raised for invalid navigation input: bad identifier, malformed selector,
    a write attempted through execute_raw, a dialect mismatch, or a strict=True miss."""

class GraphConnectionError(NavigationError):
    """Raised when the configured graph engine is unreachable (factory / ping)."""

class UnsupportedOperationError(NavigationError):
    """Raised when an optional capability (e.g. execute_raw) is called on a
    backend that does not implement it."""
```

### 5.2 Config (`config.py`)

- `ScinrConfig.graph_backend: str = "neo4j"`.
- `configure(..., graph_backend: Literal["neo4j"] | None = None)`.
- Resolution: `resolved_graph_backend = graph_backend or os.getenv("GRAPH_BACKEND", "neo4j")`; `if resolved_graph_backend not in ("neo4j",): raise ConfigurationError(...)`.
- Startup log line gains `graph_backend=%s`.

### 5.3 `scinr.newton` re-exports (`__init__.py`)

Add to imports and `__all__`:
`get_graph_navigator`, `graph_navigator`, `GraphNavigator`,
`NavigationError`, `GraphConnectionError`, `UnsupportedOperationError`.
The model classes and filter operators are reachable via
`scinr.newton.navigation.models` / `scinr.newton.navigation.filters` and
re-exported from `scinr.newton.navigation` — **not** hoisted to the top-level
`scinr.newton` namespace (keeps it uncluttered; matches how `storage` types are
not hoisted).

---

## 6. Documentation deliverables — detailed

### 6.1 D1 · `docs/user-guides/graph-navigation.md` (new user guide)

Audience: a developer who has run the pipeline and now wants to read the graph
back from Python without writing Cypher. Google-voice, task-oriented, runnable
snippets. Proposed outline:

```
# Graph Navigation

## Overview
  - what the module is (read-only, async, engine-abstracted)
  - relationship to raw Cypher (link to neo4j-graph.md) and to the storage layer
  - "nothing here mutates the graph"

## Configuration
  - graph_backend (default "neo4j"), GRAPH_BACKEND env
  - configure() example; reuses neo4j_* connection settings

## Quick start
  - `async with graph_navigator() as nav:` … list roots → walk tree → pull nodes
  - get_graph_navigator() vs graph_navigator() context manager

## Working with documents and folders
  - list_root_documents ("documentos padre")
  - get_one_document(path, version) — composite key, both mandatory
  - get_documents(...) — always a list; every filter
  - children with depth; get_document_tree; ancestors / parent / leaves
  - versions: list_document_versions / get_latest_version / get_version_chain

## Structure nodes
  - get_structure_nodes(document, roles=...), get_root_structure_nodes
  - child nodes with depth, subtree, ancestors, node path
  - find_structure_nodes (cross-document), get_nodes_by_theme
  - describe_node (aggregate view)

## InfoUnits
  - get_info_units(node), get_document_info_units
  - search_info_units (full-text, field=, score)

## Model instances (the core use case)
  - get_node_instances — instances of a StructureNode
  - get_document_instances — instances of a document
  - get_instances_by_class("VariationModel", where={...}) — filter by properties
  - get_instance / get_instance_by_key
  - get_structure_nodes_for_instance — jump back to owner(s) (always a list; why)
  - parents / children / subtree / relationships; shell instances

## Filtering with `where=`
  - dict sugar (bare value == Eq)
  - operator table: Eq Ne Gt Gte Lt Lte In NotIn Contains StartsWith EndsWith Regex IsNull IsNotNull
  - property-name rules; values always parameterised
  - discovering filterable properties: get_model_properties

## Annotation decisions
  - get_model_decision, get_document_model_decisions, coverage, proposed models

## Entities and triples
  - get_instance_entities / get_document_entities / get_labeled_entities
  - reverse lookups; entity & instance relationships; triples

## Schema introspection
  - list_catalog_models, list_model_classes_in_use, list_node_roles, list_themes
  - get_graph_summary

## Power tools
  - neighbors / shortest_path / subgraph (+ NodeSelector)

## Raw queries (escape hatch)
  - execute_raw / execute_raw_one — non-portable, dialect guard, read-only enforced
  - when to reach for it vs. a typed method

## Reading source text
  - navigation/pages.py functions; requires a configured storage backend
  - admonition: StorageError if STORAGE_BACKEND unset

## Error handling
  - NavigationError / GraphConnectionError / UnsupportedOperationError
  - strict= semantics

## Recipes (cookbook)
  - "all tables in a document", "every VariationModel with procedure_type in {IA,IB}",
    "which sections produced instances of model X", "walk a folder tree to depth N",
    "diff two document versions' instance counts"
```

Every snippet uses `pymdownx.superfences` python blocks; long API tables use
`pymdownx.tabbed` where a sync/async or dict/operator contrast helps.

### 6.2 D2 · `docs/api/navigation.md` (new API reference)

Pure mkdocstrings, mirrors `docs/api/storage.md` style:

```markdown
# Navigation API

Read-only, engine-abstracted traversal of the knowledge graph. See the
[Graph Navigation user guide](../user-guides/graph-navigation.md) for tutorials.

## Factory

::: scinr.newton.navigation.factory

## Base Interface

::: scinr.newton.navigation.base

## Return Types

::: scinr.newton.navigation.models

## Filter Operators

::: scinr.newton.navigation.filters

## Neo4j Backend

::: scinr.newton.navigation.neo4j.navigator

## Source-Text Bridge

::: scinr.newton.navigation.pages
```

Requires: complete Google-style docstrings on every public symbol (Args/Returns/
Raises). The ABC methods carry the canonical documentation; the Neo4j backend
methods can be terse ("See :class:`GraphNavigator`.") — mkdocstrings will still
list them.

### 6.3 D3 · `mkdocs.yml` nav edits

Under `User Guides:` — insert after `Neo4j Graph Storage`:
```yaml
      - Graph Navigation: user-guides/graph-navigation.md
```
Under `API Reference:` — insert after `Storage: api/storage.md`:
```yaml
      - Navigation: api/navigation.md
```

### 6.4 D5 · `docs/configuration.md` edits

- **### Neo4j** (env vars): add `GRAPH_BACKEND` — "Graph navigation backend. `neo4j` (default). Selects the `scinr.newton.navigation` implementation."
- **### Neo4j Parameters** (programmatic): add `graph_backend` — "`'neo4j'` (default). Reserved for future engines; validated like `storage_backend`."
- **## Complete Reference: All Settings**: one table row `graph_backend | GRAPH_BACKEND | "neo4j" | Navigation backend selector`.

### 6.5 D6 · `docs/architecture.md` edits

- **§4 Key Configuration Parameters**: add `graph_backend` bullet.
- **§5 Module Structure**: add the `navigation/` subtree to the printed tree.
- New section **"## 12. Graph Navigation Layer"** (or fold under §8): 3–4 paragraphs —
  the ABC + factory + Neo4j-backend shape, why it mirrors storage, the read-only
  guarantee, `execute_raw` as the deliberate non-portable seam, and a one-line
  pointer to D1/D2. Include the small ASCII diagram:
  ```
  get_graph_navigator()  ──reads──▶  cfg.graph_backend
        │
        ▼
  GraphNavigator (ABC, ~50 async read methods, engine-neutral models)
        │
        ▼
  Neo4jGraphNavigator  ──▶ get_async_driver() ──▶ Neo4j (READ tx + retry)
  ```

### 6.6 Smaller edits (D4, D7, D8, D9, D10)

- **D4** `docs/api/index.md`: `- [Navigation](navigation.md): read-only graph traversal — documents, structure nodes, model instances, entities.`
- **D7** `docs/user-guides/neo4j-graph.md`: admonition near **## Query Patterns**: "Prefer Python? The [Graph Navigation](graph-navigation.md) module wraps these patterns in typed, async methods."
- **D8** `docs/index.md`: Key Features bullet ("Read the graph back with a typed, async navigation API — no Cypher required"); Documentation list link.
- **D9** `docs/user-guides/quick-start.md`: closing pointer line.
- **D10** `README.md`: one feature bullet + link to the hosted guide.

### 6.7 Docs acceptance

- `mkdocs build --strict` passes at the end of **every** phase (nav targets must exist — so D1/D2 land as stubs in Phase 1).
- No orphan pages (everything in `nav`).
- `mkdocstrings` renders D2 with no "could not collect" warnings → every referenced module imports cleanly with only stdlib + declared deps.
- Internal links relative and valid.

---

## 7. Release mechanics

### 7.1 Version

`pyproject.toml` `version = "0.3.8"`. (Leave the stale `__init__.py __version__ = "0.2.0"` unless the maintainers want it corrected in the same PR — out of scope, flag it.)

### 7.2 Branch / PR breakdown

| PR | Branch | Contents |
|---|---|---|
| 1 | `feature/0.3.8` (base) | exceptions + config `graph_backend` + `navigation/` scaffold + Groups A/B/C + `where=`/`_safe`/`_translate` + `execute_raw` + Phase-1 docs slice + unit tests. |
| 2 | → same feature branch | Phase 2 (Groups D/E, stats, `describe_node`) + integration test + docs. |
| 3 | → same feature branch | Phase 3 (Groups F/G/H, fulltext) + docs. |
| 4 | → same feature branch | Phase 4 (`pages.py`, polish) + CHANGELOG + version bump + final docs. |

Merge the feature branch to `main` once Phase 4 lands and `mkdocs build --strict` + full test suite are green. (Or ship Phase 1 to `main` behind the frozen ABC if the team prefers incremental releases — the surface won't change.)

### 7.3 `CHANGELOG.md` — `## [0.3.8]` draft

```markdown
## [0.3.8] - <date>

### Added
- **`scinr.newton.navigation` — read-only graph navigation API.** A new
  engine-abstracted, fully `async` module for exploring the knowledge graph
  without writing Cypher by hand: list root documents, walk folder and structure
  trees to a given depth, pull the `StructureNode`s / `InfoUnit`s /
  `ModelInstance`s of a document or node, filter instances by `model_class` and
  properties with classic operators (`Eq`, `Ne`, `Gt`, `Gte`, `Lt`, `Lte`, `In`,
  `NotIn`, `Contains`, `StartsWith`, `EndsWith`, `Regex`, `IsNull`, `IsNotNull`),
  jump from a `ModelInstance` back to its owning `StructureNode`(s) / `Document`(s),
  traverse annotation decisions, entities and triples, introspect the schema, and
  run generic `neighbors` / `shortest_path` / `subgraph` queries. Entry points:
  `get_graph_navigator()` and the `graph_navigator()` async context manager.
- **Pluggable graph backend.** New `graph_backend` config field (env
  `GRAPH_BACKEND`, default `"neo4j"`), mirroring `storage_backend`. The
  navigation layer is an ABC (`GraphNavigator`) plus a concrete
  `Neo4jGraphNavigator`; other engines can be added without changing call sites.
- **`execute_raw()` / `execute_raw_one()`** — optional, non-portable escape hatch
  on the navigator for engine-native read queries, with a `dialect=` guard and a
  write-keyword rejection guard. Read-only enforced.
- New exceptions: `NavigationError`, `GraphConnectionError`,
  `UnsupportedOperationError` (all under `ScinrError`).
- New docs: **Graph Navigation** user guide and **Navigation API** reference.

### Changed
- `configure()` accepts `graph_backend=`. Startup log line now reports it.
```

---

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Variable-length Cypher (`*1..n`, unbounded containment walks) is slow or explodes on large graphs. | `resolve_depth` default cap 10; every unbounded containment query has `LIMIT` + `DISTINCT`; `subgraph` has `max_nodes`; integration test asserts query plans on the seed graph; document the cost in D1. |
| Frozen ABC declared in Phase 1 but bodies land later → `NotImplementedError` leaks to users if a partial release ships. | Ship the whole feature branch at once (§7.2), **or** gate unimplemented methods behind a clear `raise UnsupportedOperationError("navigation: available in 0.3.8 phase 2")` and don't advertise them in D1 until implemented. |
| `configure()` re-resets the async-driver singleton; a cached navigator holds a stale driver. | Navigator re-fetches `get_async_driver()` per session when it doesn't own the driver (§3.1); documented in D1. |
| mkdocstrings fails to import `navigation.*` (circular import via `config`) → empty API page. | Use deferred imports in `factory.py` (mirror `storage/factory.py` pattern); `mkdocs build --strict` in CI catches it. |
| `execute_raw` read-only regex has false negatives (e.g. `CALL apoc.*` procedures that write). | Regex covers the DML keywords + `CALL {} IN TRANSACTIONS`; additionally run in an `execute_read` transaction so the server itself rejects writes; document that `execute_raw` is READ-tx-only. |
| Return-model `raw` dict leaks Neo4j-specific shapes and callers depend on it. | Docstring on `_Base.raw`: "opaque engine-native record — do not depend on its shape across engines"; keep it `repr=False`. |
| Scope creep from ~50 methods. | Strict phase gates; each phase independently mergeable; Groups F–I explicitly deferrable if the release date pressures. |

---

## 9. Task checklist (flat, for tracking)

**Phase 1**
- [ ] `exceptions.py`: 3 classes + docstrings
- [ ] `config.py`: `graph_backend` field + `configure()` param + resolution + validation + log + docstring
- [ ] extend `tests/unit/test_config.py`
- [ ] `navigation/models.py` (all types, §5 design)
- [ ] `navigation/filters.py` + `test_navigation_filters.py`
- [ ] `navigation/neo4j/_safe.py` + `test_navigation_safe.py`
- [ ] `navigation/neo4j/_translate.py` + `test_navigation_translate.py`
- [ ] `navigation/base.py` (full ABC + `execute_raw*` raisers) + `test_navigation_base.py` + `_navigation_fakes.py`
- [ ] `navigation/factory.py` + `test_navigation_factory.py`
- [ ] `navigation/neo4j/queries.py` (Group A/B/C constants)
- [ ] `navigation/neo4j/navigator.py` (Group A/B/C + `execute_raw*` override)
- [ ] `navigation/__init__.py` + `navigation/errors.py`
- [ ] `test_navigation_neo4j_queries.py` (A/B/C + raw guard)
- [ ] `test_navigation_models.py`
- [ ] `scinr/newton/__init__.py` re-exports
- [ ] D1 stub, D2, D3, D4, D5, D7, D8, D9, D10
- [ ] `mkdocs build --strict` green; `ruff`/`mypy` clean

**Phase 2**
- [ ] Group E methods + queries + tests
- [ ] Group D methods + queries + tests
- [ ] `get_document_stats`, `describe_node`
- [ ] `tests/integration/test_navigation_neo4j.py` + seed fixture
- [ ] D1 instance/annotation/stats sections; D6 first draft
- [ ] `mkdocs build --strict` green

**Phase 3**
- [ ] Group F methods + queries + tests
- [ ] Group G methods + queries + tests
- [ ] Group H (`neighbors`/`shortest_path`/`subgraph`) + `NodeSelector`
- [ ] `search_info_units` fulltext + contract
- [ ] D1 entities/introspection/power-tools/search sections
- [ ] `mkdocs build --strict` green

**Phase 4**
- [ ] `navigation/pages.py` + `test_navigation_pages.py`
- [ ] (optional) `SyncGraphNavigator`
- [ ] D1 source-text section; final D1/D2/D6 pass; D10
- [ ] `pyproject.toml` → `0.3.8`; `CHANGELOG.md` `## [0.3.8]`
- [ ] full suite + `mkdocs build --strict` green; tag
