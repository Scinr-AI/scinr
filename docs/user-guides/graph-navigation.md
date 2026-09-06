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
    # graph_backend="neo4j" is the default
)
```

---

## Quick start

```python
import asyncio
from scinr.newton import configure, graph_navigator
from scinr.newton.navigation import In, Gte

async def main():
    configure(neo4j_user="neo4j", neo4j_password="pw")

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

`scinr.newton.navigation.pages` resolves the verbatim converted markdown behind a
node / info unit / document. It uses the **storage** abstraction, so it needs a
persistent storage backend.

```python
from scinr.newton.navigation.pages import (
    get_node_source_page_ids, get_node_source_text,
    get_info_unit_source_text, get_document_source_text,
)

ids   = await get_node_source_page_ids(nav, node_id)      # no storage needed
pages = await get_node_source_text(nav, node_id)          # raises StorageError if storage_backend="none"
```

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
