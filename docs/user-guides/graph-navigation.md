# Graph Navigation

`scinr.newton.navigation` is a **read-only**, fully `async` API for exploring the
knowledge graph the pipeline produces — without writing Cypher by hand. It turns
the [Neo4j graph model](neo4j-graph.md) into small, composable, typed methods:
list root documents, walk children to a given depth, pull the `StructureNode`s /
`InfoUnit`s / `ModelInstance`s of a document, filter model instances by class and
properties, jump from a model instance back to its structure node, and so on.

Nothing in this module mutates the graph. It is separate from `ingest/`,
`annotation/`, and `entity_extraction/`.

!!! note "Pluggable backend"
    The graph store is abstracted the same way storage is. `GraphNavigator` is
    an engine-agnostic interface; `Neo4jGraphNavigator` is the only backend
    today, chosen by the `graph_backend` config field (default `"neo4j"`).

---

## Configuration

Navigation reuses your existing Neo4j connection settings. The only new field is
`graph_backend`:

| Setting | Env | Default |
|---|---|---|
| `graph_backend` | `GRAPH_BACKEND` | `"neo4j"` |

```python
from scinr.newton import configure

configure(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your_password",
    neo4j_database="neo4j",
    # graph_backend="neo4j" is the default
    # no llm= needed — navigation is read-only and makes no LLM calls
)
```

### No LLM required

Graph navigation is read-only and does **not** require any LLM. Configure only the Neo4j connection:

```python
configure(neo4j_user="neo4j", neo4j_password="...", neo4j_database="neo4j")
```

Then use `graph_navigator()` (async context manager) or `get_graph_navigator()`.

---

## Quick start

```python
import asyncio

from scinr.newton import configure, graph_navigator
from scinr.newton.navigation import In, Gte

async def main():
    configure(neo4j_user="neo4j", neo4j_password="pw", neo4j_database="neo4j")  # no llm= needed for navigation

    async with graph_navigator() as nav:
        roots = await nav.list_root_documents()
        tree  = await nav.get_document_tree(roots[0].path)
        tables = await nav.get_structure_nodes(roots[0].path, roles=["table"])

        rows = await nav.get_model_instances_by_class(
            "VariationCodeModel",
            where={"procedure_type": In(["ia", "ib"])},
            limit=50,
        )
        for mi in rows:
            print(mi.model_class, mi.properties)

asyncio.run(main())
```

Two entry points:

* `get_graph_navigator()` → a connected navigator you must `close()` yourself.
* `graph_navigator()` → an `async with` context manager that closes it for you.

Every method is `async`. Return types are engine-neutral Pydantic models from
`scinr.newton.navigation.models`; each carries an opaque `.raw` dict (do not
depend on its shape).

---

## Scope: tenant, user, job

Every navigation method — except the global-catalogue ones and `execute_raw` —
takes four keyword-only filters that restrict what it sees:

```python
tenant_id: str | None = None                            # None = all tenants
include_public: bool = False                            # add the public documents
created_by_user_id: str | Sequence[str] | None = None   # one value or several
job_id: str | Sequence[str] | None = None               # one value or several
```

| `tenant_id` | `include_public` | What is returned |
|---|---|---|
| `None` | any | **Everything** — every tenant, public and legacy data (no tenant filter) |
| `"__public__"` | any | Only public documents |
| `"acme"` | `False` | Only tenant `acme` |
| `"acme"` | `True` | `acme` **plus** public documents |
| `""` | — | `NavigationError` |

`created_by_user_id` and `job_id` match with **any-of**: `job_id=["j1", "j2"]`
returns what either job produced (a `ModelInstance` shared by both is returned
once). They combine by AND with each other and with the tenant. An empty list
raises `NavigationError`.

```python
async with graph_navigator() as nav:
    docs = await nav.get_documents(tenant_id="acme", include_public=True)
    rows = await nav.get_model_instances_by_class(
        "VariationModel", tenant_id="acme", job_id=["job-1", "job-2"]
    )
```

Stored tenants are exposed as they are: a public document's `DocumentRef.tenant_id`
is `"__public__"` (never `None`, which is reserved for "no filter").

### Always fix the scope once: `nav.scoped(...)`

Because the default is **all tenants**, an API layer that forgets to pass
`tenant_id` exposes everything. `scoped()` returns a navigator that fills the
scope into every call and refuses to widen it:

```python
acme = nav.scoped(tenant_id="acme", include_public=True)
await acme.get_documents()                       # acme + public
await acme.get_documents(job_id="job-1")         # narrowing is fine
await acme.get_documents(tenant_id="globex")     # NavigationError — outside the scope
await acme.execute_raw("MATCH (n) RETURN n")     # NavigationError — cannot be scoped
```

Catalogue methods (`list_catalog_models`, `get_catalog_graph`, `list_themes`,
`list_relationship_types`, `list_node_labels`) are global and take no scope.

### How the filters behave

- **Lists return every match in scope.** With `tenant_id=None`, a path that exists
  in two tenants yields both documents; each `DocumentRef` carries its tenant.
- **Single-document methods** (`get_one_document`, `get_latest_version`,
  `get_document_tree`, `get_document_parent`, `get_document_ancestors`,
  `get_document_stats`, `get_document_model_profile`, `get_annotation_coverage`)
  raise `NavigationError("... exists in several tenants ...")` when the path is
  ambiguous — pass `tenant_id`. With `include_public=True`, a tenant's document
  **shadows** the public one at the same path.
- **A `DocumentRef` selector is authoritative**: it carries its own tenant; an
  explicit, different `tenant_id` raises `NavigationError`.
- **By-id lookups** (`get_structure_node`, `get_structure_nodes_by_ids`,
  `get_model_instance`, …) only resolve
  inside the scope: another tenant's id behaves as if it did not exist.
- **Trees and spines** (`get_document_tree`, `get_structure_subtree`,
  `get_model_instance_subtree`, ancestors): `created_by_user_id` / `job_id` filter
  the **anchor** only, so no intermediate node is dropped and orphaned. Flat lists
  filter every element.
- **Folder documents** are re-`MERGE`d on each ingestion, so their `job_id` /
  `created_by_user_id` are those of the last run that touched them.
- **`get_model_instance_by_key`** cannot span tenants (the instance `uid` embeds
  the tenant): pass a concrete `tenant_id` or `"__public__"`; `None` raises
  `NavigationError`. With `include_public=True` the tenant's instance is tried
  first, then the public one.
- **`get_entity_triples(value)`** matches by value: with `tenant_id=None` the same
  value extracted by several tenants comes back once per tenant. Pass a tenant.
- **Free graph walks** (`neighbors`, `shortest_path`, `subgraph`): with a tenant,
  every intermediate node must belong to it, and catalogue nodes (shared by all
  tenants) can only be the far end of a path — never a bridge into another
  tenant. With `tenant_id=None` the walk is unrestricted.
- **`get_graph_summary`** counts only the scope's nodes (catalogue nodes are
  reported unfiltered).
- **`execute_raw`** is an administrative tool outside this contract.

### Why a traversal cannot leave the tenant

Most methods apply the **tenant** predicate to their anchor (the document, node,
instance or entity they start from) and filter the nodes they reach only by
`created_by_user_id` / `job_id`. They do not re-check the tenant on every hop:
isolation comes from how the graph is written, not from the reads.

**The invariant.** No relationship between data nodes ever connects two
tenants, and "public" (`"__public__"`) counts as one more tenant:

- a document's content (`:Document` → `:StructureNode` → `:InfoUnit` /
  `:ModelDecision` / `:ExtractionResult` → …) is keyed by the document's
  `(tenant_id, path, version)`, so it all has the document's tenant;
- the content-deduplicated nodes (`:ModelInstance`, `:LabeledEntity`, `:Entity`)
  fold the tenant into their `uid` hash. When an extraction links to one
  (`REFERENCES`, `HAS_*` containment, instance-to-instance relationships,
  entity relationships, triples), it computes the target's `uid` with **its own**
  tenant, so it can only `MERGE` onto a node of that same tenant;
- versioning (`HAS_NEWER_VERSION`, `replaces`) and folders (`IS_COMPOSED_OF`) are
  tenant-scoped.

The only nodes shared across tenants are the global catalogue ones
(`:CatalogModel`, `:ModelField`, `:EntityLabel`, `:Theme`). The dedicated
methods never walk *through* them into data of another tenant, and the free
walks (`neighbors`, `shortest_path`, `subgraph`) only accept them as the far end
of a path. So starting from a node inside the scope, every node reachable along
data relationships is inside the scope too, whatever the depth.

**The trade-off: public content is not linked to any tenant.** Because the
public tenant is isolated in the same way, a tenant's extraction never reuses
or links to public nodes. If `acme` extracts an entity that a public document
already contains, `acme` gets its own `:LabeledEntity` / `:ModelInstance`;
nothing connects it to the public one. Consequences:

- With `include_public=True` a tenant sees **its own nodes and the public nodes
  side by side**, possibly two nodes for the same key (e.g.
  `get_model_instances_by_class`, `get_labeled_entities`). Correlate them by
  value or key (`get_model_instance_by_key`, `normalized_value`), not by
  following relationships.
- Relationship-following methods (`get_entity_relationships`,
  `get_model_instances_referencing_entity`, `get_related_model_instances`,
  `get_documents_for_model_instance`, `get_entity_triples`, …) never go from a
  tenant's node to a public one, or back.

This is a deliberate choice: it is what keeps the anchor-based filtering safe.
**Any change that links tenant data to public data** — reusing public nodes
from a tenant extraction, or adding explicit tenant → public relationships —
would let a traversal go `acme → public → globex`. Such a change must, in the
same step, make every navigation method check the tenant on **every** reached
node (including the intermediate nodes of variable-length paths and the
children of `get_model_instance_subtree`), as the free walks already do.

### Known limitations

- Public content is not linked to tenant content (see
  [Why a traversal cannot leave the tenant](#why-a-traversal-cannot-leave-the-tenant)).
- `job_ids` / `created_by_user_ids` on merged nodes (`ModelInstance`,
  `LabeledEntity`, `Entity`) are cumulative provenance: after deleting a job's
  documents, surviving shared nodes keep listing that job.
- A `job_id` filter on merged nodes without a `tenant_id` scans the label (the
  arrays are not indexable); with a tenant, the `tenant_id` index narrows first.
- Navigation does not filter on `frozen`: a frozen document is still listed, but
  its structure, annotations and extractions are not in the graph until
  `restore_document()` (see [Document Freezing](document-freezing.md)). Check the
  `frozen` property of the `:Document` when that matters.

---

## Documents and folders

"Root" (parent) documents are those with **no incoming `IS_COMPOSED_OF`**.

```python
roots      = await nav.list_root_documents()                       # latest only
folders    = await nav.list_root_documents(only_folders=True)
one        = await nav.get_one_document("path/to/doc", version=3)  # both args required
many       = await nav.get_documents(name_contains="annex", is_folder=False)
exists     = await nav.document_exists("path/to/doc")

kids       = await nav.get_child_documents("folder", depth=1)       # child *documents* only
tree       = await nav.get_document_tree("folder", depth=None)      # nested DocumentTree
parent     = await nav.get_document_parent("folder/child")
ancestors  = await nav.get_document_ancestors("a/b/c")             # single-spine DocumentTree, root→parent
leaves     = await nav.get_document_leaves("folder")

versions   = await nav.list_document_versions("path/to/doc")        # ascending
latest     = await nav.get_latest_version("path/to/doc")
chain      = await nav.get_version_chain("path/to/doc")

stats      = await nav.get_document_stats("path/to/doc")            # counts by role / class / label
```

Doc-scoped calls take an optional `version: int` — omit it for the current
(`latest=true`) version. Traversal methods take `depth: int | None`: `1` = direct
only, an explicit `n` verbatim, `None` = "no explicit limit" (a guard of 10 is
applied to prevent runaway traversals; pass an explicit `depth` to exceed it).

---

## Structure nodes

```python
nodes   = await nav.get_structure_nodes("doc", roles=["table"], title_contains="capsule")
roots   = await nav.get_root_structure_nodes("doc")            # HAS_STRUCTURE only
node    = await nav.get_structure_node(node_id)
batch   = await nav.get_structure_nodes_by_ids([id_a, id_b])   # one lookup, request order, found only
kids    = await nav.get_child_nodes(node_id, depth=2)
subtree = await nav.get_structure_subtree(node_id, include_info_units=True)
parent  = await nav.get_parent_node(node_id)
anc     = await nav.get_node_ancestors(node_id)                # root → immediate parent
path    = await nav.get_node_path(node_id)                     # document + node chain
doc     = await nav.get_document_of_node(node_id)              # resolved by traversal, not id-parsing
sibs    = await nav.get_sibling_nodes(node_id)
found   = await nav.find_structure_nodes(title_contains="scope", role="section")
themed  = await nav.get_nodes_by_theme("pharmaceutical_quality")
desc    = await nav.describe_node(node_id, include_source_text=False)
```

The composite `StructureNode.id` is **not** parsed to find the owning document —
every node→document / node→ancestor lookup follows relationships.

---

## InfoUnits

```python
units  = await nav.get_info_units(node_id)
n      = await nav.count_info_units("doc")
hits   = await nav.search_info_units("dutasteride capsule composition", field="both", limit=10)
unit   = await nav.get_info_unit(uid)
owner  = await nav.get_node_for_info_unit(uid)
```

`search_info_units` uses the `infoUnitTitle` / `infoUnitDescription` full-text
indexes and returns a `.score`.

---

## Annotation decisions

```python
decision = await nav.get_model_decision(node_id)
alldec   = await nav.get_document_model_decisions("doc", matched_only=True)
profile  = await nav.get_document_model_profile("doc")   # roll-up: which model classes catalogued this doc
matched  = await nav.get_nodes_by_annotated_model("DrugProductComposition")
gaps     = await nav.get_unannotated_nodes("doc")
proposed = await nav.get_proposed_models()
coverage = await nav.get_annotation_coverage("doc")
```

`ModelDecision.confidence` is a word (`"high"` / `"medium"` / `"low"`), not a
number; `coverage_gaps` is a list of strings.

`get_document_model_profile` answers "how was this document semantically
catalogued?" as a compact `matched` / `complementary` roll-up with per-class node
counts — without walking every individual decision.

---

## Model instances (the core use case)

```python
# Instances of one structure node (via HAS_EXTRACTION → HAS_* containment)
node_mi = await nav.get_node_model_instances(node_id, model_class="ConditionModel")

# Instances anywhere in a document
doc_mi  = await nav.get_document_model_instances("doc", model_class="VariationCodeModel")

# By class + property filter (values matched verbatim — normalise them yourself)
rows = await nav.get_model_instances_by_class(
    "ProcedureTypeModel",
    where={"procedure_type": In(["ia", "ib"])},
    order_by="procedure_type",
)

one    = await nav.get_model_instance(uid)
bykey  = await nav.get_model_instance_by_key("ProcedureTypeModel", {"procedure_type": "IB"})

# Jump back to the owning structure node(s) / document(s) — always a list
owners = await nav.get_structure_nodes_for_model_instance(uid)
docs   = await nav.get_documents_for_model_instance(uid)
ers    = await nav.get_extraction_results_for_model_instance(uid)

# Cross-references between model instances (any relationship type, in/out)
out_mi = await nav.get_outgoing_model_instances(uid, depth=1)
in_mi  = await nav.get_incoming_model_instances(uid)
rels   = await nav.get_model_instance_relationships(uid)
sub    = await nav.get_model_instance_subtree(uid)
shells = await nav.find_shell_model_instances(model_class="VariationCodeModel")
types  = await nav.list_model_instance_relationship_types()
```

Every `ModelInstanceRef` carries `is_shell` — `True` when the node looks like an
unfilled forward reference (only its `instance_key` fields plus `uid` /
`model_class` are set), `None` when the class has no catalog entry to compare
against.

`where=` values are used **verbatim**. Instance-key and entity values are stored
lower-cased and accent-stripped by ingestion — normalise your filters to match
(`scinr.newton.utils.uid.normalize_key` does exactly what ingestion does;
`get_model_instance_by_key` applies it for you).

Filterable properties for a class:

```python
props = await nav.get_model_properties("VariationCodeModel")
# {"declared": [...catalog ModelField names...], "observed": [...seen on instances...]}
```

---

## Filtering with `where=`

`get_documents`, `get_structure_nodes`, `find_structure_nodes`,
`get_labeled_entities`, and every `*_model_instances*` method take a `where=`
mapping of `property_name → value | operator`:

```python
from scinr.newton.navigation import In, Gte

await nav.get_model_instances_by_class(
    "VariationCodeModel",
    where={
        "procedure_type": In(["ia", "ib"]),   # operator object
        "confidence": Gte(0.8),
        "status": "active",                    # bare value == Eq("active")
    },
)
```

Rules:

* A **bare value** is sugar for `Eq` (`{"status": "active"}` ≡ `{"status": Eq("active")}`).
* Property names must match `^[A-Za-z_][A-Za-z0-9_]*$` — no dotted paths or
  expressions. An invalid name raises `NavigationError`.
* Values are **always parameterised** and matched **verbatim**: no normalisation.
  Instance-key and entity values are stored lower-cased and accent-stripped by
  ingestion, so normalise your filter to match — `scinr.newton.utils.uid.normalize_key`
  does exactly what ingestion does.
* `where=` is ANDed with the method's other arguments (and with `latest_only` on
  `get_documents`).
* To match a missing / null property use `IsNull()`; an `Eq` never matches an
  absent property.

| Operator | Meaning |
|---|---|
| `Eq(v)` / bare `v` | equals `v` |
| `Ne(v)` | not equal to `v` |
| `Gt(v)` · `Gte(v)` · `Lt(v)` · `Lte(v)` | ordered comparisons |
| `In([...])` · `NotIn([...])` | membership |
| `Contains(s)` · `StartsWith(s)` · `EndsWith(s)` | substring test on a string property |
| `Regex(pattern)` | full-match regular expression |
| `IsNull()` · `IsNotNull()` | property absent / present |

Operators are importable from `scinr.newton.navigation` (or
`scinr.newton.navigation.filters`). Discover the filterable property names of a
model class with `get_model_properties("ModelClass")`.

---

## Entities and triples

```python
mi_ents   = await nav.get_model_instance_entities(uid, label="ProcedureType")
node_ents = await nav.get_node_entities(node_id)          # via ModelInstance → REFERENCES
doc_ents  = await nav.get_document_entities("doc")
labels    = await nav.list_entity_labels()
ents      = await nav.get_labeled_entities(label="Country", value="Spain")
ent       = await nav.get_labeled_entity(uid)

refs_mi   = await nav.get_model_instances_referencing_entity(uid)
refs_sn   = await nav.get_nodes_referencing_entity(uid)   # ModelInstance → ExtractionResult → StructureNode
rels      = await nav.get_entity_relationships(uid)       # Level-2 field_relationships
related   = await nav.get_related_entities(uid, "SIMILAR_TO")

triples   = await nav.get_triples(node_id)                # Triple-fallback extractions
etr       = await nav.get_entity_triples("metformin")
```

`REFERENCES` only originates from a `:ModelInstance`. `get_triples` pairs each
subject with its object via the predicate edge; a subject with no predicate edge
comes back as a partial `Triple` (`predicate` / `object` are `None`).

---

## Schema introspection

```python
models   = await nav.list_catalog_models(include_fields=True)
catalog  = await nav.get_catalog_graph()                  # models + declared relationships between them
in_use   = await nav.list_model_classes_in_use()
roles    = await nav.list_node_roles()
themes   = await nav.list_themes()
labels   = await nav.list_node_labels()
rels     = await nav.list_relationship_types()            # structural set (~dozens); pass structural_only=False for all
summary  = await nav.get_graph_summary()
```

`list_relationship_types(structural_only=True)` (default) omits the thousands of
one-off normalised `Triple` predicate types, keeping model-instance
cross-references, entity `field_relationships`, and catalog declarations.

---

## Power tools

```python
from scinr.newton.navigation import NodeSelector

sel   = NodeSelector(type="ModelInstance", key="uid", value="abc123")
near  = await nav.neighbors(sel, edge_types=["REFERENCES"], depth=1)
path  = await nav.shortest_path(sel, NodeSelector(type="Document", key="path", value="doc"))
sub   = await nav.subgraph(sel, depth=2, max_nodes=200)
```

---

## Raw queries (escape hatch)

```python
rows = await nav.execute_raw(
    "MATCH (d:Document {latest:true}) RETURN d.path AS p ORDER BY p LIMIT $n",
    {"n": 5},
)
one  = await nav.execute_raw_one("MATCH (d:Document) RETURN count(d) AS c")
```

`execute_raw` is **non-portable** — the query is Cypher, coupling the call to
`nav.dialect == "cypher"`. It is read-only enforced: any write clause (`CREATE`,
`MERGE`, `SET`, `DELETE`, `REMOVE`, `DROP`, `FOREACH`, `LOAD CSV`,
`CALL { … } IN TRANSACTIONS`) is rejected, and the statement runs in a READ
transaction. Pass `dialect="cypher"` to fail fast on the wrong engine. A backend
with no raw path raises `UnsupportedOperationError`.

---

## Reading source text

The navigator resolves the verbatim converted markdown behind structure nodes,
an info unit or a document. These methods combine the graph with the
**storage** abstraction, so they need a persistent storage backend. Like every
other method they take the four scope filters.

```python
result = await nav.get_structure_nodes_source_pages([id_a, id_b])  # StructureNodesSourcePages
pages  = await nav.get_info_unit_source_text(uid)                  # list[PageText]
pages  = await nav.get_document_source_text("Reports/annual", version=2)
```

A `PageText` carries `page_id`, `index`, `markdown`, and the `raw_file_id`,
`filename` and `folder_path` of the upload it comes from. The functions are
also available in `scinr.newton.navigation.pages`, taking the navigator as
first argument (`get_structure_nodes_source_pages(nav, [id_a, id_b], ...)`).
A node's page ids need no storage: read `source_page_ids` from
`get_structure_nodes_by_ids()` (or `get_structure_node()`).

**Structure nodes are read in batches.** `get_structure_nodes_source_pages`
takes the unique `id`s (`StructureNodeRef.id`, not the short local `node_id`)
of one or more `:StructureNode`s — pass a single id for one node. Each page is
returned once, in `pages` (`page_id → PageText`), even when several nodes share
it; each node lists its `page_ids`, ordered by page index. Individual ids never
raise; they are reported in status groups:

| Field | Meaning |
| --- | --- |
| `nodes` | One `StructureNodeSourcePages` (`structure_node_id`, `page_ids`, `not_found_page_ids`) per node with pages, in request order |
| `pages` | `page_id → PageText`, each page once |
| `not_found_structure_nodes` | Ids that do not exist **or** are outside the scope (not told apart, so nothing leaks about another tenant) |
| `structure_nodes_without_pages` | Nodes in scope with empty `source_page_ids` |
| `not_found_page_ids` (per node) | Page ids missing from storage or belonging to another tenant |

```python
acme = nav.scoped(tenant_id="acme", include_public=True)
result = await acme.get_structure_nodes_source_pages([table_id, section_id, "unknown"])
for entry in result.nodes:
    text = "\n\n".join(result.pages[p].markdown for p in entry.page_ids)
    print(entry.structure_node_id, len(text), entry.not_found_page_ids)
print(result.not_found_structure_nodes)       # ["unknown"]
```

Duplicate ids are ignored. `StorageError` is raised only when there are pages
to read and `storage_backend="none"`. There is no size limit on the list (the
lookup uses the unique `id` constraint); an API layer should cap the request
size itself.

**The original uploaded file.** `nav.get_document_original(document, version=None)`
returns an `OriginalFile` — the stored `RawFileRecord` (`filename`,
`content_type`, `size_bytes`, …) plus a single-use stream of the binary — or
`None` when the document is not found or has no stored original:

```python
acme = nav.scoped(tenant_id="acme", include_public=True)
original = await acme.get_document_original("Reports/annual")
if original is not None:
    async for chunk in original.stream:                   # constant memory
        response.write(chunk)
    # or: data = await original.read()
```

**How the pages are read.** The structure nodes are looked up in one graph
query (for an info unit, the node that owns it), and only their
`source_page_ids` are read from storage, by id — one read per stored tenant of
the nodes (normally one; two with `include_public=True`). The owning `:Document` is not resolved and its other pages
are never loaded, so the cost depends on the size of the node, not of the
document. `get_document_source_text` reads every page of the document, and
`get_document_original` reads the file record and its binary with a single
lookup.

**Tenancy.** The node / document is resolved in the graph with the scope (pass
`tenant_id=...`, or use a scoped navigator in a multi-tenant API). The storage
read is then also filtered by the resolved node's / `:Document`'s own stored
tenant (a public node or document only reads public pages): even a page id or
`raw_file_id` pointing at another tenant's upload returns nothing (for a
structure node, it is reported in `not_found_page_ids`). See
[Storage Backends — Multi-tenancy](storage-backends.md#multi-tenancy).

---

## Error handling

| Exception | Raised when |
|---|---|
| `NavigationError` | bad identifier / property name, malformed `where=`, a write via `execute_raw`, a `dialect=` mismatch |
| `GraphConnectionError` | the engine is unreachable (`get_graph_navigator` / `ping`) |
| `UnsupportedOperationError` | an optional capability (`execute_raw`) is not implemented by the backend |

All three inherit from `ScinrError`.

---

## Recipes

**All tables in a document**

```python
tables = await nav.get_structure_nodes("doc", roles=["table"])
```

**Every `VariationCodeModel` with a given procedure type**

```python
rows = await nav.get_model_instances_by_class(
    "VariationCodeModel", where={"procedure_type": "ib"}
)
```

**Which sections produced instances of a model class**

```python
nodes = set()
for mi in await nav.get_model_instances_by_class("ConditionModel"):
    for sn in await nav.get_structure_nodes_for_model_instance(mi.uid):
        nodes.add(sn.id)
```

**Walk a folder tree to depth N**

```python
tree = await nav.get_document_tree("folder", depth=3)
```

**Diff two versions' instance counts**

```python
a = await nav.count_document_model_instances("doc", version=1)
b = await nav.count_document_model_instances("doc", version=2)
print(b - a)
```
