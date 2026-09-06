# Design Plan — Graph Navigation API (`scinr.newton.navigation`)

Status: **finalized — ready to implement (rev. 3)** · Target release: `0.3.7`+ · Owner: TBD

A read-only, **engine-abstracted** API for exploring the knowledge graph that
`scinr.newton` produces — without writing engine-specific queries by hand. It
turns the graph model documented in
[`docs/user-guides/neo4j-graph.md`](../docs/user-guides/neo4j-graph.md) and
[`graph-relationships.md`](../docs/user-guides/graph-relationships.md) into a set of
small, composable, `async` Python methods: list root documents, walk children to
a given depth, pull the `StructureNode`s / `InfoUnit`s / `ModelInstance`s of a
document, filter `ModelInstance`s by `model_class` and properties (with the
classic comparison operators), jump from a `ModelInstance` back to its
`StructureNode`, and so on.

Nothing here mutates the graph. This module is strictly separate from
`ingest/`, `annotation/`, `entity_extraction/`.

### Key decisions (see §8 for the full resolved list)

1. **No CTD/NTA catalog group.** `:CTDSectionSpec` / `SPECIFIED_IN` / `nta_section`
   etc. are just `:LabeledEntity` nodes specific to one domain example — not a
   graph feature. Dropped.
2. **Async-first.** `GraphNavigator` is fully `async`. A sync façade is a
   possible later convenience, not part of v1.
3. **`where=` ships with the classic operators** from v1 (`Eq`, `Ne`, `Gt`,
   `Gte`, `Lt`, `Lte`, `In`, `NotIn`, `Contains`, `StartsWith`, `EndsWith`,
   `Regex`, `IsNull`, `IsNotNull`).
4. **Module name `navigation`.**
5. **Pluggable graph engine.** The graph store is treated exactly like the
   storage layer today: an engine-agnostic ABC + a concrete Neo4j
   implementation, selected by a new `graph_backend` config field
   (**default `"neo4j"`**). Other engines can be added later without touching
   call sites.
6. **Raw-query escape hatch = optional capability.** `execute_raw` /
   `execute_raw_one` live on the ABC (base impl raises
   `UnsupportedOperationError`) with an engine-neutral name and a `dialect=`
   guard, but are **non-portable** by nature — the query string is in the
   current engine's language. Portable code uses the typed methods; `execute_raw`
   is a deliberate, visible coupling.

---

## 1. Graph model this API navigates (ground truth)

Derived from `ingest/nodes.py`, `annotation/neo4j_ops.py`,
`entity_extraction/graph_mapper.py`, `ingest/schema.py`. This is the **logical**
model the abstract API exposes; the Neo4j backend maps it to labels/rels 1:1,
a future backend maps it to its own primitives.

### Node types

| Type | Key attrs | Notes |
|---|---|---|
| **Document** | `path` + `version` (composite id), `name`, `latest`, `is_folder`, `raw_file_id`, `load_date`, `context_instructions` | Folder-parents have `is_folder=true`. `latest=true` on the current version only. |
| **StructureNode** (+ role) | `id` (composite `"{doc_path}::{version}::{ancestor_path/node_id}"`), `node_id`, `title`, `role`, `appearance_order`, `theme`, `source_page_ids: list[str]`, `row_index` (tabular) | Role ∈ `section, subsection, table, appendix, field_group, freeform_block, row` (Neo4j: also a second label). |
| **InfoUnit** | `uid` (SHA-256[:16]), `info_unit_id`, `title`, `description`, `order` | `description` is the citable summary. |
| **ModelDecision** | `uid`, `matched_model_class`, `confidence`, `rationale`, `coverage_gaps`, `propose_new_model`, `proposed_model_description`, `document_name`, `timestamp` | Stage 3. |
| **ExtractionResult** | `uid`, `node_full_id`, `document_name`, `model_class`, `timestamp` | Stage 4. `model_class` = primary model name or `"Triple"`. |
| **ModelInstance** | `uid`, `model_class`, + arbitrary scalar props, + `instance_key` props | Deduplicated globally when it has `instance_key` fields (`utils/uid.py::make_instance_uid`). "Shell" nodes carry only key props. |
| **CatalogModel** | `name` | Registered Pydantic model. |
| **LabeledEntity** | `uid`, `label`, `normalized_value`, `value` | Global singleton keyed `(label, normalized_value)`. Domain identifiers (variation codes, CTD codes, procedure types…) live here. |
| **Entity** | `uid`, `normalized_value`, `value` | Triple-fallback subject/object. |
| **ModelField** | `name` + `model` | |
| **ProposedModel / ProposedField / SupplementaryField / ComplementaryMatch** | annotation sub-nodes | |
| **Theme** | `name`, `path` | |

### Edge types

| Edge | From → To | Meaning |
|---|---|---|
| `IS_COMPOSED_OF` | Document → Document | Folder hierarchy (parent folder → child doc/folder). |
| `HAS_NEWER_VERSION` | Document → Document | Version chain (old → new). |
| `HAS_STRUCTURE` | Document → StructureNode | Doc → top-level nodes. |
| `HAS_CHILD` | StructureNode → StructureNode | Tree edge. |
| `HAS_INFO_UNIT` | StructureNode → InfoUnit | |
| `HAS_MODEL_DECISION` | StructureNode → ModelDecision | |
| `MATCHED_MODEL` | ModelDecision → CatalogModel | Primary model chosen. |
| `HAS_COMPLEMENTARY_MATCH` / `REFERS_TO_MODEL` | ModelDecision → ComplementaryMatch → CatalogModel | |
| `HAS_SUPPLEMENTARY_FIELD` | ModelDecision → SupplementaryField | |
| `HAS_PROPOSED_MODEL` / `HAS_PROPOSED_FIELD` | ModelDecision → ProposedModel → ProposedField | |
| `HAS_EXTRACTION` | StructureNode → ExtractionResult | |
| `USES_PRIMARY_MODEL` / `USES_COMPLEMENTARY_MODEL` | ExtractionResult → CatalogModel | |
| `HAS_<FIELDNAME>` `{index}` | ExtractionResult \| ModelInstance → ModelInstance | Nested-model **containment**. Rel name = `HAS_` + upper-snake of the field. |
| `REFERENCES` `{field_name, list_index}` | ExtractionResult \| ModelInstance → LabeledEntity | Level 1. |
| `<REL_TYPE>` | LabeledEntity → LabeledEntity | Level 2 `field_relationships`. |
| `<REL_TYPE>` | ModelInstance → ModelInstance | Level 3 `instance_relationships` (cross-section, can create shells). |
| `HAS_ENTITY` `{role}` | ExtractionResult → Entity | Triple fallback. |
| `<PREDICATE>` `{predicate_raw}` | Entity → Entity | Triple fallback. |

### Caveats the API must respect (engine-independent)

- **Containment vs. typed edges.** Walking "up" from a `ModelInstance` to its
  `StructureNode` must only follow containment (`HAS_*` / `REFERENCES`), never
  Level-3 `instance_relationships`.
- **Deduplicated / shell instances.** An `instance_key` `ModelInstance` can be
  reachable from several `ExtractionResult`s (several `StructureNode`s, even
  several `Document`s). "Which `StructureNode`?" is genuinely a *list*. Shell
  nodes may have **no** owning `ExtractionResult`.
- **Version.** Almost every question is implicitly "latest version". Default
  `latest_only=True`; every doc-scoped call takes an optional `version: int`.
- **`source_page_ids`** are storage ids. Literal page text requires a configured
  storage backend (`storage/factory.py`), which is itself already abstract.

---

## 2. Architecture — pluggable graph engine

Mirrors the storage layer (`storage/base.py` ABCs → `storage/mongodb/` impl →
`storage/factory.py` → `STORAGE_BACKEND`). Same shape, new axis.

```
src/scinr/newton/navigation/
  __init__.py            # public surface + re-exports
  base.py                # GraphNavigator (ABC) — the engine-agnostic interface
  factory.py             # get_graph_navigator() -> GraphNavigator  (reads cfg.graph_backend)
  models.py              # engine-NEUTRAL return types (*Ref, *Tree, stats, selectors)
  filters.py             # where= operator objects (engine-neutral); backends translate
  errors.py              # NavigationError(ScinrError)
  pages.py               # source-text bridge (uses the existing storage abstraction)
  neo4j/
    __init__.py
    navigator.py         # Neo4jGraphNavigator(GraphNavigator) — the default impl
    queries.py           # Cypher constant strings
    _safe.py             # identifier validation, depth clamp, read-only guard
    _translate.py        # filters.py operator -> Cypher predicate + params
tests/unit/test_navigation_*.py          # backend-agnostic tests via a fake backend + Neo4j tests w/ mocked driver
tests/integration/test_navigation_neo4j.py   # @pytest.mark.integration
docs/user-guides/graph-navigation.md + docs/api/navigation.md + mkdocs nav
```

### 2.1 The interface — `navigation/base.py`

```python
class GraphNavigator(ABC):
    """Engine-agnostic, read-only navigation over the scinr knowledge graph.

    Every method is async. Concrete backends (Neo4j today) translate these
    calls to their native query language. Return types (navigation.models)
    are engine-neutral Pydantic models.
    """

    # lifecycle
    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def close(self) -> None: ...
    async def __aenter__(self) -> "GraphNavigator": ...
    async def __aexit__(self, *exc) -> None: ...
    @abstractmethod
    async def ping(self) -> bool: ...

    # ~50 navigation methods (§4), all `@abstractmethod async def`
    ...
```

- **All read-only.** No method mutates. There is no write path in this module.
- **`execute_raw()` — optional escape hatch, on the ABC but not abstract.**
  It exists on every navigator so the "configure once, call anything" model
  holds, but it is a **non-portable** capability: the `query` string is written
  in the current engine's language. The base implementation raises
  `UnsupportedOperationError`; a backend opts in by overriding it.

  ```python
  class GraphNavigator(ABC):
      dialect: ClassVar[str] = "none"   # "cypher" | "gremlin" | "sql/age" | …

      async def execute_raw(
          self, query: str, params: dict[str, Any] | None = None, *,
          dialect: str | None = None,
      ) -> list[dict[str, Any]]:
          """Run a raw, engine-native READ query. NON-PORTABLE — using this
          couples the caller to `self.dialect`. Returns raw records as dicts
          (never `*Ref` models). Read-only enforced by the backend.

          If `dialect` is given and does not equal `self.dialect`, raises
          `NavigationError` immediately (fail fast on the wrong engine).
          """
          raise UnsupportedOperationError(
              f"{type(self).__name__} (dialect={self.dialect!r}) has no execute_raw"
          )
  ```

  - Not `@abstractmethod` → backends without a raw path inherit the raiser;
    `execute_raw` is always *present*, it may just say "no".
  - `dialect=` guard turns "sent Cypher to a non-Cypher store" into an
    immediate, clear error at the call site.
  - Returns `list[dict[str, Any]]` — raw records, explicitly not typed models.
  - The read-only guard (write-keyword regex, §3.6) lives in each backend's
    override.
  - `execute_raw_one(query, params=None, *, dialect=None) -> dict | None` is the
    single-row convenience, same contract.

  This is the SQLAlchemy pattern: portable typed API + an assumed
  `session.execute(text(...))` escape hatch.

### 2.2 The factory — `navigation/factory.py`

```python
async def get_graph_navigator() -> GraphNavigator:
    """Return the navigator for the configured engine (cfg.graph_backend).

    'neo4j' -> Neo4jGraphNavigator (the only backend today), connection checked eagerly.

    Raises ConfigurationError on an unknown backend,
    GraphConnectionError if the engine is unreachable.
    """
    cfg = get_config()
    backend = cfg.graph_backend
    if backend == "neo4j":
        nav = Neo4jGraphNavigator()          # reads cfg.neo4j_* internally
        await nav.connect()
        return nav
    raise ConfigurationError(f"Unknown graph_backend: {backend!r}. Valid: 'neo4j'.")
```

> A `"custom"` backend (a caller-supplied `GraphNavigator` instance, mirroring
> `custom_storage`) is **not** in scope now. When a second engine is needed, add
> its `"<name>"` branch here plus its `<name>_*` connection fields — no call site
> changes.

Convenience context manager:

```python
@asynccontextmanager
async def graph_navigator() -> AsyncIterator[GraphNavigator]:
    nav = await get_graph_navigator()
    try:
        yield nav
    finally:
        await nav.close()
```

### 2.3 The Neo4j backend — `navigation/neo4j/navigator.py`

- Holds the async driver from `ingest/config.py::get_async_driver()` (or its own,
  configurable). `close()` closes it iff this instance opened it.
- Sessions use `execute_read` / `default_access_mode=READ`.
- Transient errors retried via `utils/neo4j_retry.with_neo4j_retry`.
- All Cypher in `queries.py`; `where=` operators translated by `_translate.py`;
  identifiers/depth validated by `_safe.py`.
- `dialect = "cypher"`. Overrides `execute_raw` / `execute_raw_one` with the
  read-only-guarded implementation (§3.6).

### 2.4 Config changes — `config.py`

| New field | Type / default | Env | Notes |
|---|---|---|---|
| `graph_backend` | `str` = `"neo4j"` (validated against `{"neo4j"}` for now) | `GRAPH_BACKEND` | Selects the navigation backend. Validated like `storage_backend`. |

The existing `neo4j_uri / neo4j_user / neo4j_password / neo4j_database` fields
stay — they are the **Neo4j backend's** connection config (a future engine adds
its own `<engine>_*` fields). `configure()` gains a `graph_backend=` param +
`GRAPH_BACKEND` env resolution, mirroring `storage_backend`. Startup log line
extended with `graph_backend=…`.

### 2.5 What "engine-neutral" buys / constrains

- Return models (`navigation/models.py`) carry **no** Neo4j types. `.raw` is
  `dict[str, Any]` documented as "opaque engine-native record — do not depend on
  its shape across engines".
- Concepts that ARE part of the scinr domain (composite `StructureNode.id`,
  `model_class`, `instance_key`, roles, edge *semantics*) stay in the interface —
  they are not Neo4j-specific.
- Full-text search (`search_info_units`) is declared generically ("relevance
  search over InfoUnit title/description, returns a score"); the Neo4j backend
  uses its fulltext indexes, a backend without FTS falls back to substring match
  and returns `score=1.0`.

---

## 3. Cross-cutting design decisions

### 3.1 Async-first, single entry point

```python
from scinr.newton import configure
from scinr.newton.navigation import get_graph_navigator, graph_navigator

configure(neo4j_user="neo4j", neo4j_password="…")          # graph_backend defaults to "neo4j"

async with graph_navigator() as nav:
    roots  = await nav.list_root_documents()
    tree   = await nav.get_document_tree(roots[0].path, depth=None)
    tables = await nav.get_structure_nodes(roots[0].path, roles=["table"])
    inst   = await nav.get_instances_by_class(
        "VariationModel",
        where={"procedure_type": In(["IA", "IB"]), "confidence": Gte(0.8)},
        limit=50,
    )
```

A synchronous façade (`SyncGraphNavigator` wrapping `asyncio.run` per call) is
**out of scope for v1**; can be added if notebook ergonomics demand it.

### 3.2 Return types — engine-neutral Pydantic, with escape hatch

`navigation/models.py` light `BaseModel`s (see §5). Each carries
`.raw: dict[str, Any]` (opaque) and `.properties` where relevant.
`.model_dump()` for JSON.

- List ops → `list[SomeRef]` (empty list on no match).
- Single-get ops → `SomeRef | None` (never raises on "not found"), unless
  `strict=True` → `NavigationError`.
- **Arity rule:** a method returns a single `SomeRef | None` **only** when its
  arguments are a full unique key (`get_one_document(path, version)`,
  `get_structure_node(node_id)`, `get_instance(uid)`, …). Any selector that can
  legitimately match more than one node — a bare `path`, a `name`, a reverse
  lookup from a deduplicated/shared node — is a plural method returning a list,
  so callers never have to guess arity. This is why the instance→owner lookups
  in Group E are all `*_for_instance → list[...]`.

### 3.3 Depth semantics

`depth: int | None`:
- `1` = direct only (default for `*_children`).
- `n` (explicit positive `int`) = up to `n` hops, **used verbatim** — the caller
  can go as deep as they ask; no cap is imposed on an explicit value.
- `None` = "no explicit limit" → falls back to `DEFAULT_MAX_DEPTH = 10`
  (a guard against runaway traversals, *not* a hard ceiling; pass an explicit
  `depth` to exceed it) with a `logger.debug`.

Validated as a positive `int`; the Neo4j backend interpolates it into the
variable-length pattern only after `_safe.resolve_depth()` (`None → 10`,
explicit → as-is).

### 3.4 `where=` filters — classic operators from v1

`navigation/filters.py`, engine-neutral:

```python
class Op(BaseModel): ...                       # frozen base
class Eq(Op):        value: Any
class Ne(Op):        value: Any
class Gt(Op):        value: Any
class Gte(Op):       value: Any
class Lt(Op):        value: Any
class Lte(Op):       value: Any
class In(Op):        values: Sequence[Any]
class NotIn(Op):     values: Sequence[Any]
class Contains(Op):  value: str
class StartsWith(Op): value: str
class EndsWith(Op):  value: str
class Regex(Op):     pattern: str
class IsNull(Op):    pass
class IsNotNull(Op): pass
```

`where` accepts `dict[str, Any | Op]`; a bare value is sugar for `Eq`.
`{"status": "active", "confidence": Gte(0.8), "code": In(["A","B"])}`.
Property keys validated `^[A-Za-z_][A-Za-z0-9_]*$`; values always parameterised.
`_translate.py` renders each `Op` to a Cypher predicate + params.

### 3.5 Pagination & ordering

Every list op: `limit: int | None = None`, `skip: int = 0`, deterministic
default `ORDER BY` (nodes by `appearance_order`, info units by `order`, docs by
`path`, instances by `uid`, search by score desc). Optional `order_by=` where it
makes sense. Each list op has a `count_*` sibling.

### 3.6 Read-only guarantee

- Backends must only issue read transactions.
- `execute_raw` overrides strip comments and reject any query whose tokens match
  `\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b` or
  `CALL\s*\{[^}]*}\s*IN\s+TRANSACTIONS` → `NavigationError`; then run it in a
  READ tx. Same guard for `execute_raw_one`.

### 3.7 Errors

- `NavigationError(ScinrError)` — bad identifier, write attempt via
  `execute_raw`, wrong `dialect=`, `strict=True` miss, malformed selector.
- `UnsupportedOperationError(NavigationError)` — an optional capability
  (`execute_raw` on a backend that has no raw path) was called.
- `GraphConnectionError(NavigationError)` — engine unreachable (factory /
  `ping`). Mirrors `StorageError`.

### 3.8 Reuse

`utils/uid.py::make_instance_uid`, `utils/document_resolver.py` (fold
`resolve_leaf_document_names_async` into `get_document_leaves`),
`utils/neo4j_retry`, `ingest/config.py::get_async_driver`,
`storage/factory.py::get_storage`, `config.get_config()`.

---

## 4. Method catalogue

All methods on `GraphNavigator` (ABC), all `async`. `nav` = a navigator.
`document` arg accepts a `path` (preferred) or `DocumentRef`; a `name=` variant
exists where names are commonly used. Doc-scoped ops take `version: int | None`
(`None` → latest).

### A. Documents & folder hierarchy

| Method | Returns | Notes |
|---|---|---|
| `list_root_documents(*, latest_only=True, include_folders=True, include_leaves=True, limit=None, skip=0)` | `list[DocumentRef]` | **"documentos padre"** — Documents with **no incoming `IS_COMPOSED_OF`**. |
| `count_root_documents(**filters)` | `int` | |
| `get_one_document(path, version)` | `DocumentRef \| None` | **Singular getter — both args mandatory** (`path` + `version` is the composite key). No `latest` resolution here; `None` if that exact `(path, version)` doesn't exist. |
| `get_documents(*, path=None, name_contains=None, version=None, latest_only=True, is_folder=None, path_prefix=None, theme=None, where=None, limit=None, skip=0)` | `list[DocumentRef]` | **Always a list**, even for an exact `path` — anything looser than the full key can match >1, so callers never have to guess arity. Absorbs the old `list_documents` / `get_documents_by_name`. |
| `document_exists(path, *, version=None)` | `bool` | `version=None` → any version. |
| `get_document_children(path, *, depth=1, version=None, is_folder=None, limit=None)` | `list[DocumentRef]` | **`IS_COMPOSED_OF`, depth param.** Flat. |
| `get_document_tree(path, *, depth=None, version=None)` | `DocumentTree` | Nested `IS_COMPOSED_OF`. |
| `get_document_parent(path, *, version=None)` | `DocumentRef \| None` | |
| `get_document_ancestors(path, *, version=None)` | `list[DocumentRef]` | root → … → parent. |
| `get_document_leaves(path, *, version=None)` | `list[DocumentRef]` | Descendants with no outgoing `IS_COMPOSED_OF`. |
| `iter_document_descendants(path, *, depth=None)` | `AsyncIterator[DocumentRef]` | Streaming. |
| `list_document_versions(path)` | `list[DocumentRef]` | Ascending. |
| `get_latest_version(path)` | `DocumentRef \| None` | |
| `get_version_chain(path)` | `list[DocumentRef]` | Ordered via `HAS_NEWER_VERSION*`. |
| `get_document_stats(path, *, version=None)` | `DocumentStats` | Counts by role / model_class / entity label, etc. |

### B. StructureNodes & the document tree

| Method | Returns | Notes |
|---|---|---|
| `get_structure_nodes(document, *, version=None, roles=None, theme=None, where=None, limit=None, skip=0)` | `list[StructureNodeRef]` | **All** nodes of a doc, flat, `ORDER BY appearance_order`. |
| `count_structure_nodes(document, *, roles=None)` | `int` | |
| `get_root_structure_nodes(document, *, version=None)` | `list[StructureNodeRef]` | `HAS_STRUCTURE` only. |
| `get_structure_node(node_id)` | `StructureNodeRef \| None` | |
| `get_child_nodes(node_id, *, depth=1, roles=None, limit=None)` | `list[StructureNodeRef]` | **`HAS_CHILD`, depth param.** Flat. |
| `get_structure_subtree(node_id, *, depth=None, include_info_units=False)` | `StructureTree` | Nested. |
| `get_parent_node(node_id)` | `StructureNodeRef \| None` | |
| `get_node_ancestors(node_id)` | `list[StructureNodeRef]` | Up to root node. |
| `get_node_path(node_id)` | `NodePath` | `document` + `nodes: [root … self]`. |
| `get_document_of_node(node_id)` | `DocumentRef \| None` | |
| `get_sibling_nodes(node_id, *, include_self=False)` | `list[StructureNodeRef]` | |
| `find_structure_nodes(*, title_contains=None, node_id=None, role=None, theme=None, document=None, where=None, limit=None, skip=0)` | `list[StructureNodeRef]` | Cross-document. |
| `get_nodes_by_theme(theme, *, document=None, limit=None)` | `list[StructureNodeRef]` | |
| `describe_node(node_id, *, include_source_text=False)` | `NodeDescription` | Aggregate: node + info_units + model_decision + extraction summary + child_count + source_page_ids + ancestors (+ literal pages if storage configured). |

### C. InfoUnits

| Method | Returns | Notes |
|---|---|---|
| `get_info_units(node_id, *, order_by="order")` | `list[InfoUnitRef]` | Of one node. |
| `get_document_info_units(document, *, version=None, limit=None, skip=0)` | `list[InfoUnitWithNode]` | **All** of a doc; each carries `node_id` + `node_title`. |
| `count_info_units(document)` | `int` | |
| `search_info_units(text, *, field="both", document=None, limit=25)` | `list[ScoredInfoUnit]` | Relevance search (`field ∈ {"title","description","both"}`), `.score`. |
| `get_info_unit(uid)` | `InfoUnitRef \| None` | |
| `get_node_for_info_unit(uid)` | `StructureNodeRef \| None` | |

### D. Annotation (`ModelDecision`)

| Method | Returns | Notes |
|---|---|---|
| `get_model_decision(node_id)` | `ModelDecisionRef \| None` | + `matched_model`, `complementary_models`, `supplementary_fields`. |
| `get_document_model_decisions(document, *, version=None, matched_only=None)` | `list[ModelDecisionWithNode]` | |
| `get_nodes_by_annotated_model(model_class, *, document=None)` | `list[StructureNodeRef]` | via `MATCHED_MODEL`. |
| `get_unannotated_nodes(document, *, version=None)` | `list[StructureNodeRef]` | No `HAS_MODEL_DECISION`. |
| `get_proposed_models(*, document=None)` | `list[ProposedModelRef]` | + `proposed_fields`, source node. |
| `get_annotation_coverage(document, *, version=None)` | `AnnotationCoverage` | annotated / unannotated / matched / proposed + ratio. |

### E. Extraction & `ModelInstance` — core of the request

| Method | Returns | Notes |
|---|---|---|
| `get_extraction_result(node_id)` | `ExtractionResultRef \| None` | + `primary_model`, `complementary_models`, `is_triple`. |
| `get_document_extraction_results(document, *, version=None, model_class=None, limit=None)` | `list[ExtractionResultWithNode]` | |
| `get_node_instances(node_id, *, model_class=None, where=None, depth=None, direct_only=False)` | `list[ModelInstanceRef]` | **"ModelInstances de un StructureNode"** — all reachable via `HAS_EXTRACTION` → containment. |
| `get_document_instances(document, *, version=None, model_class=None, where=None, limit=None, skip=0)` | `list[ModelInstanceRef]` | **"ModelInstances de un documento"** — union across its nodes, dedupe by `uid`. |
| `count_document_instances(document, *, model_class=None, where=None)` | `int` | |
| `get_instances_by_class(model_class, *, where=None, document=None, order_by=None, limit=None, skip=0)` | `list[ModelInstanceRef]` | **"ModelInstance de una model_class + filtrar por propiedades"**, operator-aware `where`. |
| `count_instances_by_class(model_class, *, where=None, document=None)` | `int` | |
| `get_instance(uid)` | `ModelInstanceRef \| None` | |
| `get_instance_by_key(model_class, key_fields: dict[str,str])` | `ModelInstanceRef \| None` | Rebuilds the deterministic uid via `make_instance_uid` (helper `normalize_key()` exposed). |
| `get_structure_nodes_for_instance(uid)` | `list[StructureNodeRef]` | **"el/los StructureNode a los que pertenece la ModelInstance"** — up via containment only (`HAS_*` / `REFERENCES`). Always a list: a deduplicated `instance_key` instance can belong to several nodes; a shell instance belongs to none (`[]`). No singular variant. |
| `get_documents_for_instance(uid)` | `list[DocumentRef]` | Same reasoning — list. |
| `get_extraction_results_for_instance(uid)` | `list[ExtractionResultRef]` | Same reasoning — list. |
| `get_instance_parents(uid)` | `list[ModelInstanceRef]` | Immediate containing instance(s) via `HAS_*` — list (a shared nested instance can have several parents). |
| `get_instance_children(uid, *, depth=1, rel_type=None)` | `list[ModelInstanceRef]` | Nested `HAS_*`; each carries `.via_rel`, `.index`. |
| `get_instance_subtree(uid, *, depth=None)` | `InstanceTree` | |
| `get_instance_relationships(uid, *, direction="both", rel_type=None)` | `list[InstanceRelation]` | Level-3 edges only (excludes containment). |
| `get_related_instances(uid, rel_type, *, direction="out")` | `list[ModelInstanceRef]` | |
| `find_shell_instances(*, model_class=None, limit=None)` | `list[ModelInstanceRef]` | Few-properties heuristic. |
| `list_instance_relationship_types(*, document=None)` | `list[RelTypeStat]` | distinct `(src_model, rel_type, tgt_model, count)`. |

### F. Entities (`LabeledEntity`, `Entity`, triples)

| Method | Returns | Notes |
|---|---|---|
| `get_instance_entities(uid, *, label=None)` | `list[LabeledEntityRef]` | `REFERENCES` out of an instance; `.field_name`, `.list_index`. |
| `get_node_entities(node_id, *, label=None)` | `list[LabeledEntityRef]` | All under the node's extraction subtree. |
| `get_document_entities(document, *, label=None, version=None, limit=None)` | `list[LabeledEntityRef]` | |
| `list_entity_labels()` | `list[EntityLabelStat]` | distinct `label` + count. |
| `get_labeled_entities(*, label=None, value=None, normalized_value=None, where=None, limit=None, skip=0)` | `list[LabeledEntityRef]` | |
| `get_labeled_entity(uid)` | `LabeledEntityRef \| None` | |
| `get_instances_referencing_entity(uid, *, model_class=None, limit=None)` | `list[ModelInstanceRef]` | Reverse `REFERENCES`. |
| `get_nodes_referencing_entity(uid, *, limit=None)` | `list[StructureNodeRef]` | |
| `get_entity_relationships(uid, *, direction="both", rel_type=None)` | `list[EntityRelation]` | Level-2 `field_relationships`. |
| `get_related_entities(uid, rel_type, *, direction="out")` | `list[LabeledEntityRef]` | |
| `get_triples(node_id)` | `list[Triple]` | For `Triple` extraction results. |
| `get_document_triples(document, *, version=None, limit=None)` | `list[Triple]` | |
| `get_entity_triples(value_or_uid, *, direction="both")` | `list[Triple]` | Triples touching an `Entity`. |

### G. Catalog / schema introspection

| Method | Returns | Notes |
|---|---|---|
| `list_catalog_models()` | `list[CatalogModelRef]` | Registered models. |
| `list_model_classes_in_use(*, document=None)` | `list[ModelClassStat]` | distinct `model_class` + counts. |
| `get_model_properties(model_class, *, document=None)` | `list[str]` | Union of scalar keys across those instances — feeds `where=` discovery. |
| `list_node_roles(*, document=None)` | `list[RoleStat]` | |
| `list_themes()` | `list[ThemeRef]` | |
| `list_relationship_types()` | `list[str]` | Engine-native edge/rel type names. |
| `list_node_labels()` | `list[str]` | Engine-native node type names. |
| `get_graph_summary()` | `GraphSummary` | per-type counts, total docs, latest docs. |

### H. Generic navigation / power tools

| Method | On | Returns | Notes |
|---|---|---|---|
| `neighbors(selector, *, edge_types=None, direction="both", target_types=None, depth=1, limit=None)` | ABC | `list[GraphNode]` | Adjacency from any node via `NodeSelector(type, key, value)`. |
| `shortest_path(from_selector, to_selector, *, max_hops=6, edge_types=None)` | ABC | `PathResult \| None` | |
| `subgraph(selector, *, depth=2, edge_types=None, max_nodes=500)` | ABC | `Subgraph` | `{nodes, edges}` for viz/export. |
| `execute_raw(query, params=None, *, dialect=None)` | ABC (optional; raises `UnsupportedOperationError` if unimplemented) | `list[dict[str, Any]]` | **Non-portable** escape hatch — `query` is in `nav.dialect`'s language. Read-only guard + READ tx in the backend override. `dialect=` mismatch → `NavigationError`. |
| `execute_raw_one(query, params=None, *, dialect=None)` | ABC (optional) | `dict \| None` | Single-row convenience, same contract. |

### I. Source text bridge — `navigation/pages.py` (optional; needs storage)

Async. Raises `StorageError` if `STORAGE_BACKEND` unset; `[]` for a node with no
`source_page_ids`. Uses the **already-abstract** storage layer, so it is
engine-agnostic on the graph side.

| Function | Returns | Notes |
|---|---|---|
| `await get_node_source_page_ids(nav, node_id)` | `list[str]` | Just the ids. |
| `await get_node_source_text(nav, node_id)` | `list[PageText]` | Owning `Document.raw_file_id` + node `source_page_ids` → **verbatim** markdown per page. |
| `await get_document_source_text(nav, document, *, version=None)` | `list[PageText]` | |
| `await get_info_unit_source_text(nav, uid)` | `list[PageText]` | Via owning node. |

---

## 5. Return-type sketches (`navigation/models.py`, engine-neutral)

```python
class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)  # opaque engine record

class DocumentRef(_Base):
    path: str; name: str; version: int; latest: bool; is_folder: bool
    raw_file_id: str | None = None; load_date: str | None = None

class StructureNodeRef(_Base):
    id: str; node_id: str; title: str | None; role: str
    types: list[str]                 # engine-native type tags (Neo4j: labels)
    appearance_order: int; theme: str; source_page_ids: list[str]
    document_path: str | None = None

class InfoUnitRef(_Base):
    uid: str; title: str; description: str; order: int
class InfoUnitWithNode(InfoUnitRef):
    node_id: str; node_title: str | None
class ScoredInfoUnit(InfoUnitWithNode):
    score: float

class ModelDecisionRef(_Base):
    uid: str; matched_model_class: str | None; confidence: float | None
    rationale: str | None; coverage_gaps: str | None
    matched_model: str | None = None; complementary_models: list[str] = []

class ExtractionResultRef(_Base):
    uid: str; model_class: str; document_name: str; timestamp: str | None
    is_triple: bool; primary_model: str | None = None; complementary_models: list[str] = []

class ModelInstanceRef(_Base):
    uid: str; model_class: str; properties: dict[str, Any]
    is_shell: bool | None = None
    via_rel: str | None = None; index: int | None = None      # when returned as a child

class LabeledEntityRef(_Base):
    uid: str; label: str; value: str; normalized_value: str
    field_name: str | None = None; list_index: int | None = None

class InstanceRelation(_Base):
    rel_type: str; direction: Literal["out","in"]; other: ModelInstanceRef
class EntityRelation(_Base):
    rel_type: str; direction: Literal["out","in"]; other: LabeledEntityRef
class Triple(_Base):
    subject: str; predicate: str; object: str; node_id: str | None = None

class DocumentTree(DocumentRef):
    depth: int; children: list["DocumentTree"] = []
class StructureTree(StructureNodeRef):
    depth: int; info_units: list[InfoUnitRef] | None = None; children: list["StructureTree"] = []
class InstanceTree(ModelInstanceRef):
    depth: int; children: list["InstanceTree"] = []

class NodePath(_Base):
    document: DocumentRef; nodes: list[StructureNodeRef]
class NodeDescription(_Base):
    node: StructureNodeRef; ancestors: list[StructureNodeRef]
    info_units: list[InfoUnitRef]; model_decision: ModelDecisionRef | None
    extraction: ExtractionResultRef | None; instance_count: int; child_count: int
    source_page_ids: list[str]; source_text: list["PageText"] | None = None

class DocumentStats(_Base):
    path: str; version: int
    structure_nodes: int; structure_nodes_by_role: dict[str, int]
    info_units: int; model_decisions: int; model_decisions_matched: int; model_decisions_proposed: int
    extraction_results: int; model_instances: int; model_instances_by_class: dict[str, int]
    labeled_entities: int; labeled_entities_by_label: dict[str, int]; triples: int
class GraphSummary(_Base):
    node_counts: dict[str, int]; relationship_counts: dict[str, int]
    documents: int; latest_documents: int

class PageText(_Base):
    page_id: str; index: int | None; markdown: str

class NodeSelector(BaseModel):
    type: str; key: str; value: Any        # e.g. ("ModelInstance", "uid", "…")
class GraphNode(_Base):
    types: list[str]; properties: dict[str, Any]
class PathResult(_Base):
    length: int; nodes: list[GraphNode]; relationships: list[dict[str, Any]]
class Subgraph(_Base):
    nodes: list[GraphNode]; edges: list[dict[str, Any]]

# stat helpers: ModelClassStat · RoleStat · EntityLabelStat · RelTypeStat ·
# ThemeRef · CatalogModelRef · AnnotationCoverage · ProposedModelRef ·
# ModelDecisionWithNode · ExtractionResultWithNode
```

---

## 6. Representative Cypher — Neo4j backend (`navigation/neo4j/queries.py`)

**Root documents**
```cypher
MATCH (d:Document)
WHERE (NOT $latest_only OR d.latest = true)
  AND NOT ( ()-[:IS_COMPOSED_OF]->(d) )
  AND ($include_folders OR d.is_folder = false)
  AND ($include_leaves  OR d.is_folder = true)
RETURN d ORDER BY d.path SKIP $skip LIMIT $limit
```

**Children to depth N** (`get_document_children`; depth int-validated & clamped)
```cypher
MATCH (root:Document {path:$path})
WHERE ($version IS NULL AND root.latest = true) OR root.version = $version
MATCH (root)-[:IS_COMPOSED_OF*1..%(depth)d]->(c:Document)
RETURN DISTINCT c ORDER BY c.path
```

**All ModelInstances of a document** (`get_document_instances`)
```cypher
MATCH (d:Document {path:$path})
WHERE ($version IS NULL AND d.latest = true) OR d.version = $version
MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)
      -[:HAS_EXTRACTION]->(er:ExtractionResult)
MATCH (er)-[rels*0..10]->(mi:ModelInstance)
WHERE all(r IN rels WHERE type(r) STARTS WITH 'HAS_')
  AND ($model_class IS NULL OR mi.model_class = $model_class)
  /* + translated where= predicates on mi.`<key>` */
RETURN DISTINCT mi ORDER BY mi.uid SKIP $skip LIMIT $limit
```

**StructureNode(s) owning a ModelInstance** (`get_structure_nodes_for_instance`)
```cypher
MATCH (mi:ModelInstance {uid:$uid})
MATCH p = (sn:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)-[rels*1..12]->(mi)
WHERE all(r IN rels WHERE type(r) STARTS WITH 'HAS_' OR type(r) = 'REFERENCES')
RETURN DISTINCT sn, length(p) AS hops ORDER BY hops
```

**Instances by class + operator filter** (`get_instances_by_class`,
`where={"procedure_type": In(["IA","IB"]), "confidence": Gte(0.8)}`)
```cypher
MATCH (mi:ModelInstance {model_class:$model_class})
WHERE mi.`procedure_type` IN $w0 AND mi.`confidence` >= $w1
RETURN mi ORDER BY mi.uid SKIP $skip LIMIT $limit
```

**Fulltext InfoUnit search** (`search_info_units`, `field="both"`)
```cypher
CALL db.index.fulltext.queryNodes('infoUnitDescription', $q) YIELD node, score
MATCH (sn:StructureNode)-[:HAS_INFO_UNIT]->(node)
WHERE $doc_prefix IS NULL OR sn.id STARTS WITH $doc_prefix
RETURN node, score, sn.id AS node_id, sn.title AS node_title
ORDER BY score DESC LIMIT $limit
```

**Instance relationships only** (`get_instance_relationships`)
```cypher
MATCH (mi:ModelInstance {uid:$uid})-[r]->(o:ModelInstance)
WHERE NOT type(r) STARTS WITH 'HAS_'
RETURN type(r) AS rel_type, 'out' AS direction, o
UNION
MATCH (mi:ModelInstance {uid:$uid})<-[r]-(o:ModelInstance)
WHERE NOT type(r) STARTS WITH 'HAS_'
RETURN type(r) AS rel_type, 'in' AS direction, o
```

`_safe.py`: `safe_ident(s)` → assert `^[A-Za-z_][A-Za-z0-9_]*$`;
`resolve_depth(d)` → `DEFAULT_MAX_DEPTH` (=10) if `d is None` else `int(d)` verbatim (must be > 0);
`assert_read_only(cypher)` → comment-strip + write-keyword regex.
`_translate.py`: `Op` → `(predicate_fragment, {param_name: value})`.

---

## 7. Phasing

| Phase | Scope | Exit criteria |
|---|---|---|
| **1 — Abstraction + MVP** | `config.graph_backend` + `configure()` wiring. `navigation/{base,factory,models,filters,errors}.py`. `navigation/neo4j/{navigator,queries,_safe,_translate}.py`. Group A (roots, children/depth, tree, parent/ancestors, leaves, versions, `get_one_document`, `get_documents`). Group B (structure nodes of doc, root nodes, `get_structure_node`, child nodes/depth, subtree, parent/ancestors, `get_document_of_node`, `find_structure_nodes`). Group C (`get_info_units`, `get_document_info_units`, `get_info_unit`, `get_node_for_info_unit`). `where=` with all operators. `execute_raw` / `execute_raw_one` — base raiser on the ABC + read-only-guarded override on the Neo4j impl (`dialect="cypher"`). | Backend-agnostic unit tests via a `FakeGraphNavigator`; Neo4j unit tests with mocked async driver; `factory` returns Neo4j by default; `execute_raw` write-guard + `dialect` mismatch tested. |
| **2 — Instances & annotation** | Group E in full (node/doc/by-class + operator `where=`, `get_instance*`, `get_structure_nodes_for_instance`, parent/children/subtree, instance relationships, shells). Group D. `get_document_stats`, `describe_node`. | Integration test against a seeded Neo4j graph (`@pytest.mark.integration`). |
| **3 — Entities & introspection** | Group F, Group G, Group H (`neighbors`, `shortest_path`, `subgraph`). `search_info_units` fulltext + substring fallback contract. | `get_graph_summary` verified; power tools documented. |
| **4 — Polish** | Group I source-text bridge. `docs/user-guides/graph-navigation.md`, `docs/api/navigation.md`, mkdocs nav, README §Neo4j-Schema cross-link, CHANGELOG, version bump. Optional `SyncGraphNavigator` façade. | Docs published; public surface re-exported from `scinr.newton`. |

---

## 8. Resolved decisions

All review questions are settled — the plan above already reflects them.

1. **Config field name** → `graph_backend` (not `graph_motor`), `str`, default
   `"neo4j"`, `GRAPH_BACKEND` env, validated like `storage_backend`.
2. **`custom` backend** → **not supported now.** No `custom_graph_navigator`, no
   `"custom"` branch. A second engine is added as its own named branch +
   `<name>_*` fields when needed, without call-site changes.
3. **Raw escape hatch** → optional `execute_raw` / `execute_raw_one` on the ABC
   (base raises `UnsupportedOperationError`), engine-neutral name, `dialect=`
   guard, returns raw `list[dict]`, read-only enforced per backend. Non-portable
   by nature; using it is a deliberate, visible coupling to `nav.dialect`.
4. **Document getters** → `get_one_document(path, version)` with **both args
   mandatory** (the composite key) returning `DocumentRef | None`;
   `get_documents(*, path=None, name_contains=None, version=None, latest_only=True,
   is_folder=None, path_prefix=None, theme=None, where=None, …)` **always
   returning a list** (covers every looser lookup). General arity rule in §3.2.
5. **Strictness** → single-get miss returns `None`; opt-in `strict=True` →
   `NavigationError`.
6. **Depth** → explicit `depth=n` is honoured verbatim (no cap). `depth=None`
   falls back to `DEFAULT_MAX_DEPTH = 10` as a runaway-traversal guard, not a
   hard ceiling.
7. **`*_for_instance` lookups** → always plural / list
   (`get_structure_nodes_for_instance`, `get_documents_for_instance`,
   `get_extraction_results_for_instance`, `get_instance_parents`). No singular
   "nearest owner" variant — arity can't be guaranteed for deduplicated/shared
   instances.
8. **Group H power tools** → ship `neighbors` + `shortest_path` + `subgraph`
   together.
