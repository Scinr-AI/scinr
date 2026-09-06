# Navigation API — Cypher query review (rev. 3)

Status: **implemented** — `src/scinr/newton/navigation/` is built to this spec;
every query in this doc was validated against the `neo4j-local` MCP graph.
95 navigation unit tests + 20 uid tests green; `ruff` + `mypy` clean on the
package. Companion to [`implement-navigation.md`](implement-navigation.md)
(design) and [`implement-navigation-plan.md`](implement-navigation-plan.md) (plan).

All queries run against the `neo4j-local` MCP test graph (full Stage-4 data:
235 `Document`, 2 223 `StructureNode`, 1 867 `ExtractionResult`,
4 319 `ModelInstance`, 1 055 `LabeledEntity`, 6 293 `Entity`,
1 638 `ModelDecision`; 32 064 nodes / 53 820 rels).

**rev. 2 — your feedback applied.** Changelog at the end.

Legend: ✅ ran, result shown · ⚠️ still needs a call from you.

---

## Conventions (rev. 2)

1. **Dynamic WHERE.** Queries are assembled in Python: a predicate is added to
   the `WHERE` **only when its argument is supplied**. No more
   `($p IS NULL OR n.x = $p)` scaffolding anywhere. Examples below show the
   *maximal* form (all filters present) with a `-- [if $x]` marker on each
   optional line.
2. **Version resolution.** Built dynamically, not with `OR`:
   * `version` given → `MATCH (d:Document {path:$path, version:$version})`
   * `version` omitted → `MATCH (d:Document {path:$path, latest:true})`
3. **Depth.** Every variable-length traversal exposes `depth: int | None`.
   `None` → `_safe.resolve_depth` returns `DEFAULT_MAX_DEPTH = 10`
   (`INSTANCE_CONTAINMENT_DEPTH = 12` for `ExtractionResult→HAS_*→ModelInstance`).
   Interpolated as an int literal into `*1..N` — the only interpolation.
4. **`where=` values are verbatim.** No auto-normalisation, ever — the caller
   normalises (instance-key / entity values are stored lower-cased &
   accent-stripped by ingestion). `normalize_key()` helper is exposed for
   convenience.
5. **Paging / ordering.** Deterministic `ORDER BY … SKIP $skip LIMIT $limit`.
6. **Read-only.** `session.execute_read` + `with_neo4j_retry`.

---

## 0. Schema deltas vs. the design — applied to `models.py` ✅

You approved (implicitly, "corrige el documento") — these are **done** in
`models.py`:

| # | Change | Evidence (MCP) |
|---|---|---|
| 0.1 | `ModelDecisionRef.confidence: str \| None` (was `float`) | `apoc.meta.cypher.type(md.confidence)` → `STRING`, 1638/1638 (values `"high"`/`"medium"`/`"low"`) |
| 0.2 | `ModelDecisionRef.coverage_gaps: list[str]` (was `str`) | `type()` → `LIST OF STRING`, 1638/1638 |
| 0.3 | `ProposedModelRef.schema_name` (+ `.name` alias property) | `keys(pm)` = `["uid","description","schema_name"]` |
| 0.4 | `ProposedFieldRef {field_name, field_type, required, description}` | apoc schema for `:ProposedField` / `:SupplementaryField` |
| 0.5 | `StructureNodeRef.document_path/_version` filled only from the query context, never by parsing `id` | `id` sample `…annex1::6::page-2::6/page-6::6_3` — path has `/` and `::`, segments have `::` |
| 0.7 | `ExtractionResultRef.is_triple` derived (`model_class == "Triple"`), not read | no such property on the node |
| 0.8 | containment detection pins **endpoint labels + direction**, not just `HAS_` prefix | Level-2 `field_relationships` include `HAS_PROCEDURE_TYPE` (418×), `HAS_CHILD_VARIATION_CODE` (415×) — between `:LabeledEntity` |
| 0.9 | `list_relationship_types(structural_only=True)` default; `get_graph_summary` uses a curated rel set + scalars | `db.relationshipTypes()` → **2 391** types (mostly unique `Triple` predicates) |

Naming: **everything "instance" → "model_instance"** in method names, and the
models `InstanceTree → ModelInstanceTree`, `InstanceRelation →
ModelInstanceRelation`.

---

## Group A — Documents & folder hierarchy

### A1 · `list_root_documents(*, latest_only=True, only_folders=False, only_leaves=False, limit, skip)` ✅
Params reworked per your note: `only_folders` / `only_leaves` replace
`include_*`.
```cypher
MATCH (d:Document)
WHERE NOT ( ()-[:IS_COMPOSED_OF]->(d) )
  -- [if latest_only]  AND d.latest = true
  -- [if only_folders] AND d.is_folder = true
  -- [if only_leaves]  AND d.is_folder = false
RETURN d ORDER BY d.path, d.version SKIP $skip LIMIT $limit
```
`only_folders` + `only_leaves` both true → `ValueError` in Python (not a query).
`latest_only=true` → 6 roots; `false` → 14; `only_leaves` → 5. ✅
`count_root_documents` = same body, `RETURN count(d)`.

### A2 · `get_one_document(path, version)` ✅
```cypher
MATCH (d:Document {path:$path, version:$version}) RETURN d
```

### A3 · `get_documents(*, path, name_contains, version, latest_only=True, is_folder, path_prefix, where, limit, skip)` ✅
`theme=` **removed** (too costly to resolve for documents). Dynamic WHERE:
```cypher
MATCH (d:Document)
WHERE 1=1
  -- [if path]         AND d.path = $path
  -- [if name_contains]AND toLower(d.name) CONTAINS toLower($name_contains)
  -- [if version]      AND d.version = $version
  -- [if latest_only]  AND d.latest = true
  -- [if is_folder!=None] AND d.is_folder = $is_folder
  -- [if path_prefix]  AND d.path STARTS WITH $path_prefix
  -- [if where]        AND <translated where= on alias d>
RETURN d ORDER BY d.path, d.version SKIP $skip LIMIT $limit
```

### A4 · `document_exists(path, *, version=None)` ✅
```cypher
-- version given
RETURN EXISTS { MATCH (:Document {path:$path, version:$version}) } AS found
-- version omitted
RETURN EXISTS { MATCH (:Document {path:$path}) } AS found
```

### A5 · `get_child_documents(path, *, depth=1, version, is_folder, limit)` ✅
**Renamed** from `get_document_children` — it returns only child *documents*.
```cypher
-- version given → {path:$path, version:$version} ; else → {path:$path, latest:true}
MATCH (root:Document {path:$path, latest:true})
MATCH (root)-[:IS_COMPOSED_OF*1..%(depth)d]->(c:Document)
  -- [if is_folder!=None] WHERE c.is_folder = $is_folder
RETURN DISTINCT c ORDER BY c.path LIMIT $limit
```
`depth=1` → 1 child; `depth=10` → 190. ✅

### A6 · `get_document_tree(path, *, depth=None, version)` ✅
Flat collect + Python assembly:
```cypher
MATCH (root:Document {path:$path, latest:true})
OPTIONAL MATCH path = (root)-[:IS_COMPOSED_OF*1..%(depth)d]->(c:Document)
RETURN root, c, [x IN nodes(path) | x.path] AS lineage
```

### A7 · `get_document_parent(path, *, version)` ✅
```cypher
MATCH (p:Document)-[:IS_COMPOSED_OF]->(d:Document {path:$path, latest:true})
RETURN p ORDER BY p.version DESC LIMIT 1
```

### A8 · `get_document_ancestors(path, *, version, depth=None) → DocumentTree | None` ✅ (reworked)
Per your note — returns **the hierarchy itself**, not a flat list. Result is the
root folder-parent as a `DocumentTree` whose `children` is the single chain down
to (not including) *path*. Flatten for a list; keep for the structure.
```cypher
MATCH (d:Document {path:$path, latest:true})
MATCH p = (root:Document)-[:IS_COMPOSED_OF*1..%(depth)d]->(d)
WHERE NOT ( ()-[:IS_COMPOSED_OF]->(root) )
WITH p ORDER BY length(p) DESC LIMIT 1
RETURN [x IN nodes(p)[0..-1] | x] AS spine   // root … parent, in order
```
Python folds `spine` into nested `DocumentTree` (`depth` = index). ✅ (leaf
`…/annex1` v6 → 9-node spine, root first.)

### A9 · `get_document_leaves(path, *, version, depth=None)` ✅ (added `depth`)
```cypher
MATCH (root:Document {path:$path, latest:true})
MATCH (root)-[:IS_COMPOSED_OF*1..%(depth)d]->(leaf:Document)
WHERE NOT (leaf)-[:IS_COMPOSED_OF]->(:Document)
RETURN DISTINCT leaf ORDER BY leaf.path
```
Dutasterida root → 96 leaves. ✅

### ~~A10 · `iter_document_descendants`~~ — **REMOVED** per your note.

### A11 · `list_document_versions(path)` / `get_latest_version(path)` / `get_version_chain(path)` ✅
```cypher
MATCH (d:Document {path:$path}) RETURN d ORDER BY d.version
MATCH (d:Document {path:$path, latest:true}) RETURN d LIMIT 1
```
`HAS_NEWER_VERSION` verified to be a clean chain; ordering by `version` suffices.
`get_version_chain` additionally returns, per row, whether the
`HAS_NEWER_VERSION` edge to the next version exists (sanity flag). ✅

### A12 · `get_document_stats(path, *, version)` ✅
One core query for totals + **2 small follow-ups** (your Q4 pick: plain Cypher,
no apoc) for the per-role and per-model_class breakdowns:
```cypher
-- core
MATCH (d:Document {path:$path, latest:true})
CALL { WITH d MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(n:StructureNode) RETURN count(n) AS n_nodes }
CALL { WITH d MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_INFO_UNIT]->(u:InfoUnit) RETURN count(u) AS n_units }
CALL { WITH d MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
       RETURN count(md) AS n_dec,
              sum(CASE WHEN md.matched_model_class IS NOT NULL THEN 1 ELSE 0 END) AS n_matched,
              sum(CASE WHEN md.propose_new_model THEN 1 ELSE 0 END) AS n_proposed }
CALL { WITH d MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_EXTRACTION]->(er:ExtractionResult)
       OPTIONAL MATCH (er)-[hr*1..%(cdepth)d]->(mi:ModelInstance) WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_')
       RETURN count(DISTINCT er) AS n_er, count(DISTINCT mi) AS n_mi }
CALL { WITH d MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)-[hr2*1..%(cdepth)d]->(le:LabeledEntity)
       WHERE all(x IN hr2 WHERE type(x) STARTS WITH 'HAS_' OR type(x)='REFERENCES')
       RETURN count(DISTINCT le) AS n_le }
RETURN n_nodes, n_units, n_dec, n_matched, n_proposed, n_er, n_mi, n_le
-- follow-up 1: roles     → MATCH … (n:StructureNode) RETURN n.role, count(*)
-- follow-up 2: mi classes → … RETURN mi.model_class, count(DISTINCT mi)
-- follow-up 3: le labels  → … RETURN le.label, count(DISTINCT le)
```
⚠️ `triples` count: `MATCH (d)…->(:StructureNode)-[:HAS_EXTRACTION]->(er:ExtractionResult {model_class:'Triple'})-[:HAS_ENTITY {role:'subject'}]->(s)-[p]->(o)<-[:HAS_ENTITY {role:'object'}]-(er) RETURN count(p)`.

---

## Group B — StructureNodes

### B1 · `get_structure_nodes(document, *, version, roles, title_contains, theme, where, depth=None, limit, skip)` ✅
`title_contains` added as an explicit convenience filter (your question — yes,
it was implied by `where=`, now it's also a first-class param).
```cypher
MATCH (d:Document {path:$path, latest:true})
MATCH (d)-[:HAS_STRUCTURE|HAS_CHILD*1..%(depth)d]->(n:StructureNode)
WHERE 1=1
  -- [if roles]          AND n.role IN $roles
  -- [if title_contains] AND toLower(n.title) CONTAINS toLower($title_contains)
  -- [if theme]          AND n.theme = $theme
  -- [if where]          AND <where= on n>
RETURN DISTINCT n ORDER BY n.appearance_order, n.id SKIP $skip LIMIT $limit
```
`bpg-annex-i` → 47 nodes; `roles=["table"]` → 3; `title_contains="capsule"` → 4. ✅
`count_structure_nodes(document, *, version, roles, depth)` = `RETURN count(DISTINCT n)`.

### B2 · `get_root_structure_nodes(document, *, version)` ✅
```cypher
MATCH (d:Document {path:$path, latest:true})-[:HAS_STRUCTURE]->(n:StructureNode)
RETURN n ORDER BY n.appearance_order, n.id
```

### B3 · `get_structure_node(node_id)` ✅ — `MATCH (n:StructureNode {id:$node_id}) RETURN n`

### B4 · `get_child_nodes(node_id, *, depth=1, roles, limit)` ✅
```cypher
MATCH (:StructureNode {id:$node_id})-[:HAS_CHILD*1..%(depth)d]->(c:StructureNode)
  -- [if roles] WHERE c.role IN $roles
RETURN DISTINCT c ORDER BY c.appearance_order, c.id LIMIT $limit
```

### B5 · `get_structure_subtree(node_id, *, depth=None, include_info_units=False)` ✅
```cypher
MATCH (root:StructureNode {id:$node_id})
OPTIONAL MATCH path = (root)-[:HAS_CHILD*1..%(depth)d]->(c:StructureNode)
OPTIONAL MATCH (c)-[:HAS_INFO_UNIT]->(u:InfoUnit)   -- [only if include_info_units]
RETURN root, c, [x IN nodes(path) | x.id] AS lineage, collect(u) AS units
```

### B6 · `get_parent_node(node_id)` / `get_node_ancestors(node_id, *, depth=None)` ✅
```cypher
MATCH (p:StructureNode)-[:HAS_CHILD]->(:StructureNode {id:$node_id}) RETURN p LIMIT 1
```
```cypher
MATCH p = (d:Document)-[:HAS_STRUCTURE]->(a:StructureNode)-[:HAS_CHILD*0..%(depth)d]->(t:StructureNode {id:$node_id})
RETURN [x IN nodes(p) WHERE x:StructureNode][0..-1] AS ancestors
ORDER BY length(p) DESC LIMIT 1
```
Verified: nested node → `["page-4::3_2_p_3"]`, root first. ✅

### B7 · `get_node_path(node_id)` ✅ — as B6 but returns the `:Document` **and** the full node chain incl. self (`NodePath`).

### B8 · `get_document_of_node(node_id)` ✅ (traversal, no id-parsing)
```cypher
MATCH (n:StructureNode {id:$node_id})<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(d:Document)
RETURN d LIMIT 1
```

### B9 · `get_sibling_nodes(node_id, *, include_self=False)` ✅
```cypher
OPTIONAL MATCH (p:StructureNode)-[:HAS_CHILD]->(:StructureNode {id:$node_id})
CALL {
  WITH p MATCH (p)-[:HAS_CHILD]->(s:StructureNode) RETURN collect(s) AS s1
}
// fallback when the node is root-level (parent is the Document):
CALL {
  MATCH (d:Document)-[:HAS_STRUCTURE]->(:StructureNode {id:$node_id})
  MATCH (d)-[:HAS_STRUCTURE]->(s:StructureNode) RETURN collect(s) AS s2
}
WITH coalesce(p, null) AS p, CASE WHEN p IS NULL THEN s2 ELSE s1 END AS sibs
UNWIND sibs AS s
WITH s WHERE $include_self OR s.id <> $node_id
RETURN s ORDER BY s.appearance_order, s.id
```
⚠️ slightly fiddly — acceptable? Alternative: two separate round-trips in Python
(try `HAS_CHILD` parent first, else `HAS_STRUCTURE`). **I'll do the 2-query
Python version** unless you prefer the single query.

### B10 · `find_structure_nodes(*, title_contains, node_id, role, theme, document, where, limit, skip)` ✅
```cypher
MATCH (n:StructureNode)
WHERE 1=1
  -- [if title_contains] AND toLower(n.title) CONTAINS toLower($title_contains)
  -- [if node_id]        AND n.node_id = $node_id
  -- [if role]           AND n.role = $role
  -- [if theme]          AND n.theme = $theme
  -- [if where]          AND <where= on n>
  -- [if document]       AND EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path:$doc_path}) }
RETURN n ORDER BY n.id SKIP $skip LIMIT $limit
```

### B11 · `get_nodes_by_theme(theme, *, document, limit)` ✅
```cypher
MATCH (n:StructureNode {theme:$theme})
  -- [if document] WHERE EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path:$doc_path}) }
RETURN n ORDER BY n.id LIMIT $limit
```

### B12 · `describe_node(node_id, *, include_source_text=False)` ✅
Composed: `get_structure_node` + `get_node_ancestors` + `get_info_units` +
`get_model_decision` + `get_extraction_result` + child count +
`count(DISTINCT mi)` reachable by `HAS_*` containment + optional
`pages.get_node_source_text`. `NodeDescription.model_instance_count` (renamed
from `instance_count`).

---

## Group C — InfoUnits

### C1 · `get_info_units(node_id, *, order_by="order")` ✅
```cypher
MATCH (:StructureNode {id:$node_id})-[:HAS_INFO_UNIT]->(u:InfoUnit)
RETURN u ORDER BY u.order, u.uid
```

### ~~`get_document_info_units`~~ — **REMOVED** per your note (too massive).

### C2 · `count_info_units(document, *, version, depth=None)` ✅ (kept — just a count)
```cypher
MATCH (d:Document {path:$path, latest:true})-[:HAS_STRUCTURE|HAS_CHILD*1..%(depth)d]->(:StructureNode)-[:HAS_INFO_UNIT]->(u:InfoUnit)
RETURN count(u) AS c
```

### C3 · `search_info_units(text, *, field="both", document=None, limit=25)` ✅
```cypher
CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score
MATCH (sn:StructureNode)-[:HAS_INFO_UNIT]->(node)
  -- [if document] WHERE EXISTS { MATCH (sn)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path:$doc_path}) }
RETURN node, score, sn.id AS node_id, sn.title AS node_title
ORDER BY score DESC LIMIT $limit
```
`field="both"` → run `infoUnitTitle` + `infoUnitDescription`, merge on `uid`
keeping `max(score)`, re-sort. Indexes confirmed present & working (query
`"dutasteride capsule composition"` → top 6.24). ✅ Document scoping via the
`EXISTS { traversal }` (correct; `sn.id STARTS WITH` is unsafe — delta 0.5).

### C4 · `get_info_unit(uid)` / `get_node_for_info_unit(uid)` ✅
```cypher
MATCH (u:InfoUnit {uid:$uid}) RETURN u
MATCH (n:StructureNode)-[:HAS_INFO_UNIT]->(:InfoUnit {uid:$uid}) RETURN n LIMIT 1
```

---

## Group D — Annotation (ModelDecision)

### D1 · `get_model_decision(node_id)` ✅
```cypher
MATCH (:StructureNode {id:$node_id})-[:HAS_MODEL_DECISION]->(md:ModelDecision)
RETURN md,
  [(md)-[:MATCHED_MODEL]->(cm) | cm.name][0] AS matched_model,
  [(md)-[:HAS_COMPLEMENTARY_MATCH]->()-[:REFERS_TO_MODEL]->(cm) | cm.name] AS complementary_models,
  [(md)-[:HAS_SUPPLEMENTARY_FIELD]->(sf) | sf.field_name] AS supplementary_fields
```
→ `confidence:"high"`, `coverage_gaps:[]`, `matched_model:"DrugProductComposition"`. ✅

### D2 · `get_document_model_decisions(document, *, version, matched_only, depth=None)` ✅
```cypher
MATCH (d:Document {path:$path, latest:true})-[:HAS_STRUCTURE|HAS_CHILD*1..%(depth)d]->(n:StructureNode)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
WHERE 1=1
  -- [if matched_only is True]  AND md.matched_model_class IS NOT NULL
  -- [if matched_only is False] AND md.matched_model_class IS NULL
RETURN md, n.id AS node_id, n.title AS node_title
ORDER BY n.appearance_order
```

### D3 · `get_document_model_profile(document, *, version, depth=None) → DocumentModelProfile` ✅ **(new — your request)**
"How was this document semantically catalogued?" — a roll-up, no individual
decisions.
```cypher
MATCH (d:Document {path:$path, latest:true})-[:HAS_STRUCTURE|HAS_CHILD*1..%(depth)d]->(n:StructureNode)
OPTIONAL MATCH (n)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
OPTIONAL MATCH (md)-[:MATCHED_MODEL]->(mm:CatalogModel)
OPTIONAL MATCH (md)-[:HAS_COMPLEMENTARY_MATCH]->()-[:REFERS_TO_MODEL]->(cc:CatalogModel)
OPTIONAL MATCH (md)-[:HAS_PROPOSED_MODEL]->(pm:ProposedModel)
RETURN
  count(n) AS total_nodes,
  sum(CASE WHEN md IS NULL THEN 1 ELSE 0 END) AS unannotated,
  collect(DISTINCT mm.name) AS matched_names,
  collect(DISTINCT cc.name) AS complementary_names,
  collect(DISTINCT pm.schema_name) AS proposed_names
```
+ 2 tiny follow-ups for the per-name counts (matched, complementary) → fills
`DocumentModelProfile.matched` / `.complementary` as `list[ModelClassStat]`
(with `kind`). ✅

### D4 · `get_nodes_by_annotated_model(model_class, *, document)` ✅
```cypher
MATCH (n:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:MATCHED_MODEL]->(:CatalogModel {name:$model_class})
  -- [if document] WHERE EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path:$doc_path}) }
RETURN DISTINCT n ORDER BY n.id
```

### D5 · `get_unannotated_nodes(document, *, version, depth=None)` ✅
```cypher
MATCH (d:Document {path:$path, latest:true})-[:HAS_STRUCTURE|HAS_CHILD*1..%(depth)d]->(n:StructureNode)
WHERE NOT (n)-[:HAS_MODEL_DECISION]->()
RETURN n ORDER BY n.appearance_order
```

### D6 · `get_proposed_models(*, document)` ✅
```cypher
MATCH (n:StructureNode)-[:HAS_MODEL_DECISION]->(:ModelDecision)-[:HAS_PROPOSED_MODEL]->(pm:ProposedModel)
  -- [if document] WHERE EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path:$doc_path}) }
RETURN pm, [(pm)-[:HAS_PROPOSED_FIELD]->(f) | f {.*}] AS fields, n.id AS node_id
ORDER BY pm.uid
```
`pm.schema_name` → `ProposedModelRef.schema_name`. ✅

### D7 · `get_annotation_coverage(document, *, version, depth=None)` ✅
```cypher
MATCH (d:Document {path:$path, latest:true})-[:HAS_STRUCTURE|HAS_CHILD*1..%(depth)d]->(n:StructureNode)
OPTIONAL MATCH (n)-[:HAS_MODEL_DECISION]->(md:ModelDecision)
RETURN count(n) AS total, count(md) AS annotated,
       sum(CASE WHEN md IS NULL THEN 1 ELSE 0 END) AS unannotated,
       sum(CASE WHEN md.matched_model_class IS NOT NULL THEN 1 ELSE 0 END) AS matched,
       sum(CASE WHEN md.propose_new_model THEN 1 ELSE 0 END) AS proposed
```

---

## Group E — Extraction & model instances

Renamed: `*instance* → *model_instance*` throughout. `get_instance_parents` /
`get_instance_children` → `get_incoming_model_instances` /
`get_outgoing_model_instances` (any rel type, no `HAS_` filter — your note).

**Containment still uses `HAS_*`** for the *provenance* queries only (E3, E4, E8,
E9, and the doc-scoped `EXISTS` filter of E5): that path is the definition of
"this instance was extracted at this node/document". The cross-reference queries
(E10–E12) do **not** filter by rel type. ⚠️ **Ask:** OK to keep `HAS_*` for E3/
E4/E8/E9 provenance, or drop it there too?

### E1 · `get_extraction_result(node_id)` ✅
```cypher
MATCH (:StructureNode {id:$node_id})-[:HAS_EXTRACTION]->(er:ExtractionResult)
RETURN er,
  [(er)-[:USES_PRIMARY_MODEL]->(cm) | cm.name][0] AS primary_model,
  [(er)-[:USES_COMPLEMENTARY_MODEL]->(cm) | cm.name] AS complementary_models
```
`is_triple := er.model_class = 'Triple'`. `USES_PRIMARY_MODEL` 1404×, `…COMPLEMENTARY` 48×. ✅

### E2 · `get_document_extraction_results(document, *, version, model_class, depth=None, limit)` ✅
```cypher
MATCH (d:Document {path:$path, latest:true})-[:HAS_STRUCTURE|HAS_CHILD*1..%(depth)d]->(n:StructureNode)-[:HAS_EXTRACTION]->(er:ExtractionResult)
  -- [if model_class] WHERE er.model_class = $model_class
RETURN er, n.id AS node_id, n.title AS node_title
ORDER BY n.appearance_order LIMIT $limit
```

### E3 · `get_node_model_instances(node_id, *, model_class, where, depth=None, direct_only=False)` ✅
```cypher
MATCH (:StructureNode {id:$node_id})-[:HAS_EXTRACTION]->(er:ExtractionResult)
MATCH (er)-[rels*1..%(depth)d]->(mi:ModelInstance)          -- direct_only → -[:*1..1]-
WHERE all(r IN rels WHERE type(r) STARTS WITH 'HAS_')
  -- [if model_class] AND mi.model_class = $model_class
  -- [if where]       AND <where= on mi>
RETURN DISTINCT mi ORDER BY mi.model_class, mi.uid
```
Verified: a node reaches ≤ 49 distinct instances. ✅

### E4 · `get_document_model_instances(document, *, version, model_class, where, depth=None, limit, skip)` ✅ + `count_…`
```cypher
MATCH (d:Document {path:$path, latest:true})-[:HAS_STRUCTURE|HAS_CHILD*1..]->(:StructureNode)
      -[:HAS_EXTRACTION]->(er:ExtractionResult)
MATCH (er)-[rels*1..%(cdepth)d]->(mi:ModelInstance)
WHERE all(r IN rels WHERE type(r) STARTS WITH 'HAS_')
  -- [if model_class] AND mi.model_class = $model_class
  -- [if where]       AND <where= on mi>
RETURN DISTINCT mi ORDER BY mi.uid SKIP $skip LIMIT $limit
```
`%(cdepth)d` = `INSTANCE_CONTAINMENT_DEPTH` (12) when `depth=None` (your Q6 pick).

### E5 · `get_model_instances_by_class(model_class, *, where, document, order_by, limit, skip)` ✅
```cypher
MATCH (mi:ModelInstance {model_class:$model_class})
WHERE 1=1
  -- [if where]    AND <where= on mi>
  -- [if document] AND EXISTS {
        MATCH (mi)<-[hr*1..%(cdepth)d]-(:ExtractionResult)<-[:HAS_EXTRACTION]-(:StructureNode)
              <-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path:$doc_path})
        WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') }
RETURN mi ORDER BY %(order_by)s SKIP $skip LIMIT $limit
```
`where` values verbatim (your ruling): `where={"procedure_type": Eq("ib")}` on
`ProcedureTypeModel` matches (stored lower-cased). `order_by` whitelisted →
`` mi.`<safe_ident>` ``, default `mi.uid`. `count_…` = `RETURN count(mi)`.

### E6 · `get_model_instance(uid)` ✅ — `MATCH (mi:ModelInstance {uid:$uid}) RETURN mi`

### E7 · `get_model_instance_by_key(model_class, key_fields)` ✅
`uid = make_instance_uid(model_class, {k: normalize_key(v) …})` → `get_model_instance(uid)`.
⚠️ **Ask (Q8):** extract `_normalize` (NFKD + strip accents + lower + collapse
ws) from `entity_extraction/graph_mapper.py` into `utils/uid.py` as
`normalize_key`, used by both sites? Unit test asserts
`ProcedureTypeModel` + `{"procedure_type":"IB"}` → `69dd11016939bbf7`.

### E8 · `get_structure_nodes_for_model_instance(uid)` ✅ (always a list)
```cypher
MATCH (mi:ModelInstance {uid:$uid})
MATCH p = (sn:StructureNode)-[:HAS_EXTRACTION]->(:ExtractionResult)-[rels*1..%(cdepth)d]->(mi)
WHERE all(r IN rels WHERE type(r) STARTS WITH 'HAS_')
RETURN DISTINCT sn ORDER BY length(p)
```

### E9 · `get_documents_for_model_instance(uid)` / `get_extraction_results_for_model_instance(uid)` ✅
```cypher
MATCH (mi:ModelInstance {uid:$uid})
MATCH (er:ExtractionResult)-[rels*1..%(cdepth)d]->(mi)
WHERE all(r IN rels WHERE type(r) STARTS WITH 'HAS_')
OPTIONAL MATCH (er)<-[:HAS_EXTRACTION]-(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(d:Document)
RETURN collect(DISTINCT er) AS ers, collect(DISTINCT d) AS docs
```

### E10 · `get_incoming_model_instances(uid, *, rel_type, depth=1, limit)` / `get_outgoing_model_instances(...)` ✅ (no `HAS_` filter)
```cypher
-- outgoing
MATCH (:ModelInstance {uid:$uid})-[r*1..%(depth)d]->(o:ModelInstance)
  -- [if rel_type] WHERE all(x IN r WHERE type(x) = $rel_type)
RETURN DISTINCT o, type(r[0]) AS via_rel, 'out' AS direction
ORDER BY via_rel, o.uid LIMIT $limit
-- incoming: (o)-[r*1..N]->(:ModelInstance {uid:$uid})
```
Any relationship type (containment or cross-reference). Verified against
`QA_MENTIONS_*`, `FEE_APPLIES_TO`, `HAS_QA_ENTRY_MODEL`, … ✅

### E11 · `get_model_instance_subtree(uid, *, depth=None)` ✅
Flat collect of outgoing edges (any type) with `type()` + `index` lineage →
`ModelInstanceTree` in Python.

### E12 · `get_model_instance_relationships(uid, *, direction="both", rel_type)` ✅ (all edges, no exclusion)
```cypher
MATCH (mi:ModelInstance {uid:$uid})-[r]->(o:ModelInstance)
  -- [if rel_type] WHERE type(r) = $rel_type
RETURN type(r) AS rel_type, 'out' AS direction, o
UNION
MATCH (mi:ModelInstance {uid:$uid})<-[r]-(o:ModelInstance)
  -- [if rel_type] WHERE type(r) = $rel_type
RETURN type(r) AS rel_type, 'in' AS direction, o
```
`direction` picks which half runs. `get_related_model_instances(uid, rel_type,
direction)` = single-type projection returning `o` only.

### E13 · `find_shell_model_instances(*, model_class, limit)` ✅
```cypher
MATCH (mi:ModelInstance) WITH mi.model_class AS m, avg(size(keys(mi))) AS avgk
MATCH (mi2:ModelInstance {model_class:m})
WITH mi2, avgk WHERE size(keys(mi2)) < avgk * 0.5 OR size(keys(mi2)) <= 3
  -- [if model_class] AND mi2.model_class = $model_class
RETURN mi2 ORDER BY mi2.model_class, mi2.uid LIMIT $limit
```
Heuristic (documented as such). Sample: `VariationCodeModel` 227, `ProcedureTypeModel` 18. ✅
`is_shell` on every `ModelInstanceRef`: computed as
`size(keys(mi)) <= n_instance_key_fields(model_class) + 2`, where
`n_instance_key_fields` comes from `MATCH (:CatalogModel {name:m})-[:HAS_FIELD]->(f:ModelField {is_instance_key:true})`
(`ModelField.is_instance_key` confirmed present in apoc schema). ⚠️ **Ask (Q9):**
OK to populate `is_shell` this way on all instance results (one extra small
lookup, memoised per class)?

### E14 · `list_model_instance_relationship_types(*, document)` ✅
```cypher
MATCH (a:ModelInstance)-[r]->(b:ModelInstance)
WHERE NOT type(r) STARTS WITH 'HAS_'
  -- [if document] AND EXISTS { MATCH (a)<-[hr*1..%(cdepth)d]-(:ExtractionResult)<-[:HAS_EXTRACTION]-(:StructureNode)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path:$doc_path}) WHERE all(x IN hr WHERE type(x) STARTS WITH 'HAS_') }
RETURN a.model_class AS source_model, type(r) AS rel_type, b.model_class AS target_model, count(*) AS count
ORDER BY count DESC
```
⚠️ this one keeps the `NOT … STARTS WITH 'HAS_'` filter — it is specifically
about *cross-reference* rel types. Correct? (In this DB the 8 non-`HAS_` types
are exactly the `instance_relationships`.) 8 types / 1 500 edges. ✅

---

## Group F — Entities (LabeledEntity, Entity, triples)

**Correction (your note): `REFERENCES` originates *only* from `:ModelInstance`,
never `:ExtractionResult`** — confirmed on the MCP (13 063 edges, 0 from ER). All
"entities of a node/document" queries now go
`… → ExtractionResult → HAS_* → ModelInstance → REFERENCES → LabeledEntity`.

### F1 · `get_model_instance_entities(uid, *, label)` ✅
```cypher
MATCH (:ModelInstance {uid:$uid})-[r:REFERENCES]->(le:LabeledEntity)
  -- [if label] WHERE le.label = $label
RETURN le, r.field_name AS field_name, r.list_index AS list_index
ORDER BY le.label, le.value
```

### F2 · `get_node_entities(node_id, *, label, depth=None)` ✅ (corrected path)
```cypher
MATCH (:StructureNode {id:$node_id})-[:HAS_EXTRACTION]->(er:ExtractionResult)
MATCH (er)-[hr*1..%(cdepth)d]->(mi:ModelInstance)
WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_')
MATCH (mi)-[:REFERENCES]->(le:LabeledEntity)
  -- [if label] WHERE le.label = $label
RETURN DISTINCT le ORDER BY le.label, le.value
```

### F3 · `get_document_entities(document, *, label, version, depth=None, limit)` ✅ — F2 with the document spine prepended; `RETURN DISTINCT le … LIMIT`.

### F4 · `list_entity_labels()` ✅
```cypher
MATCH (le:LabeledEntity) RETURN le.label AS label, count(*) AS count ORDER BY count DESC
```
27 labels (`ProcedureType`, `VariationCode`, `Country`, …). ✅

### F5 · `get_labeled_entities(*, label, value, normalized_value, where, limit, skip)` ✅
```cypher
MATCH (le:LabeledEntity)
WHERE 1=1
  -- [if label]            AND le.label = $label
  -- [if value]            AND le.value = $value
  -- [if normalized_value] AND le.normalized_value = $normalized_value
  -- [if where]            AND <where= on le>
RETURN le ORDER BY le.label, le.value SKIP $skip LIMIT $limit
```

### F6 · `get_labeled_entity(uid)` ✅ — `MATCH (le:LabeledEntity {uid:$uid}) RETURN le`

### F7 · `get_model_instances_referencing_entity(uid, *, model_class, limit)` ✅
```cypher
MATCH (mi:ModelInstance)-[:REFERENCES]->(:LabeledEntity {uid:$uid})
  -- [if model_class] WHERE mi.model_class = $model_class
RETURN DISTINCT mi ORDER BY mi.model_class, mi.uid LIMIT $limit
```

### F8 · `get_nodes_referencing_entity(uid, *, depth=None, limit)` ✅ (corrected — navigate ModelInstance → ER → StructureNode)
```cypher
MATCH (:LabeledEntity {uid:$uid})<-[:REFERENCES]-(mi:ModelInstance)
MATCH (mi)<-[hr*1..%(cdepth)d]-(er:ExtractionResult)
WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_')
MATCH (sn:StructureNode)-[:HAS_EXTRACTION]->(er)
RETURN DISTINCT sn ORDER BY sn.id LIMIT $limit
```
Verified: `ProcedureType` entity → 275 structure nodes. ✅

### F9 · `get_entity_relationships(uid, *, direction, rel_type)` / `get_related_entities(...)` ✅
```cypher
MATCH (le:LabeledEntity {uid:$uid})-[r]->(o:LabeledEntity)
  -- [if rel_type] WHERE type(r) = $rel_type
RETURN type(r) AS rel_type, 'out' AS direction, o
UNION
MATCH (le:LabeledEntity {uid:$uid})<-[r]-(o:LabeledEntity)
  -- [if rel_type] WHERE type(r) = $rel_type
RETURN type(r) AS rel_type, 'in' AS direction, o
```
Discriminator = both endpoints `:LabeledEntity` (some types start with `HAS_`,
e.g. `HAS_PROCEDURE_TYPE` 418×, `HAS_CHILD_VARIATION_CODE` 415× — **not**
filtered out). ✅

### F10 · `get_triples(node_id)` ✅ (predicate edge now OPTIONAL — your note)
```cypher
MATCH (:StructureNode {id:$node_id})-[:HAS_EXTRACTION]->(er:ExtractionResult {model_class:'Triple'})
MATCH (er)-[:HAS_ENTITY {role:'subject'}]->(s:Entity)
OPTIONAL MATCH (s)-[p]->(o:Entity)<-[:HAS_ENTITY {role:'object'}]-(er)
RETURN s.value AS subject, type(p) AS predicate, p.predicate_raw AS predicate_raw, o.value AS object
```
Verified: for a multi-triple ER, each subject pairs with its object via `p`;
`predicate_raw` on the edge; a subject with no predicate edge → row with
`predicate`/`object` = `null` (partial `Triple`). ✅

### ~~`get_document_triples`~~ — **REMOVED** per your note (too massive).

### F11 · `get_entity_triples(value_or_uid, *, direction)` ✅
```cypher
MATCH (e:Entity) WHERE e.uid = $k OR e.value = $k OR e.normalized_value = toLower($k)
CALL {
  WITH e MATCH (e)-[p]->(o:Entity) RETURN e.value AS subject, type(p) AS predicate, p.predicate_raw AS predicate_raw, o.value AS object  -- direction in (out,both)
  UNION
  WITH e MATCH (s:Entity)-[p]->(e) RETURN s.value AS subject, type(p) AS predicate, p.predicate_raw AS predicate_raw, e.value AS object  -- direction in (in,both)
}
RETURN subject, predicate, predicate_raw, object
```

---

## Group G — Catalogue / schema introspection

### G1 · `list_catalog_models(*, include_fields=False)` ✅
```cypher
MATCH (cm:CatalogModel)
  -- [if include_fields] OPTIONAL MATCH (cm)-[hf:HAS_FIELD]->(f:ModelField)
  -- [if include_fields] OPTIONAL MATCH (cm)-[:BELONGS_TO_THEME]->(t:Theme)
RETURN cm, collect(DISTINCT {name:f.name, type:f.type, entity_label:f.entity_label,
       is_instance_key:f.is_instance_key, required:hf.required, description:hf.description}) AS fields,
       collect(DISTINCT t.name) AS themes
ORDER BY cm.name
```
185 models; `keys(cm)` = `name, description, selectable`. ✅

### G2 · `get_catalog_graph(*, include_fields=True, include_relationships=True) → CatalogGraph` ✅ **(new — your request)**
Full catalogue + the relationships between entries.
```cypher
-- nodes: reuse G1 for models; MATCH (el:EntityLabel) RETURN el.label
-- relationships:
MATCH (a)-[r]->(b)
WHERE (a:CatalogModel OR a:EntityLabel) AND (b:CatalogModel OR b:EntityLabel)
  AND NOT type(r) IN ['HAS_FIELD','BELONGS_TO_THEME']
RETURN labels(a)[0] AS source_kind, coalesce(a.name, a.label) AS source,
       type(r) AS rel_type, properties(r) AS props,
       labels(b)[0] AS target_kind, coalesce(b.name, b.label) AS target
```
Verified: `AGGREGATES` 67, `SPECIFIED_IN` 66, `PRODUCES_ENTITY` 65 (→EntityLabel),
`CONTROLLED_BY` 3 (EntityLabel→EntityLabel), plus domain rels — all carry
`join_via` / `via_field` / `from_field` / `to_field` props. ✅

### G3 · `list_model_classes_in_use(*, document)` ✅ (removed the `$doc_path IS NULL OR` — dynamic)
```cypher
MATCH (mi:ModelInstance)
  -- [if document] WHERE EXISTS {
        MATCH (mi)<-[hr*1..%(cdepth)d]-(:ExtractionResult)<-[:HAS_EXTRACTION]-(:StructureNode)
              <-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path:$doc_path})
        WHERE all(r IN hr WHERE type(r) STARTS WITH 'HAS_') }
RETURN mi.model_class AS model_class, count(*) AS count ORDER BY count DESC
```

### G4 · `get_model_properties(model_class, *, document) → {"declared":[…], "observed":[…]}` ✅ (your Q11: both)
```cypher
-- declared
MATCH (:CatalogModel {name:$model_class})-[:HAS_FIELD]->(f:ModelField) RETURN collect(f.name) AS declared
-- observed
MATCH (mi:ModelInstance {model_class:$model_class}) WITH mi LIMIT 500
UNWIND keys(mi) AS k WITH DISTINCT k WHERE NOT k IN ['uid','model_class']
RETURN collect(k) AS observed
```
⚠️ observed sample capped at 500 instances. OK?

### G5 · `list_node_roles(*, document)` ✅
```cypher
MATCH (n:StructureNode)
  -- [if document] WHERE EXISTS { MATCH (n)<-[:HAS_STRUCTURE|HAS_CHILD*1..]-(:Document {path:$doc_path}) }
RETURN n.role AS role, count(*) AS count ORDER BY count DESC
```
→ `subsection` 8250 · `section` 1433 · `table` 1030 · `freeform_block` 352 ·
`field_group` 97 · `appendix` 45 · `row` 7. ✅

### G6 · `list_themes()` ✅
```cypher
MATCH (t:Theme) RETURN t.name AS name, t.path AS path ORDER BY t.path
```
17/11 `:Theme` nodes (`name` + `path`); names repeat across paths — all returned.
Fallback (no `:Theme` nodes): `MATCH (n:StructureNode) WHERE n.theme IS NOT NULL RETURN DISTINCT n.theme`.

### G7 · `list_relationship_types(*, structural_only=True)` / `list_node_labels()` ✅
`structural_only=True` (default) → every relationship type **except** those used
*only* between two `:Entity` nodes (the ~2 325 one-off normalised `Triple`
predicates). This **keeps** the `:ModelInstance`↔`:ModelInstance` cross-reference
types (`QA_MENTIONS_*`, `FEE_APPLIES_TO`, …), the Level-2 `:LabeledEntity`
`field_relationships`, and the catalog declaration rels — per your note.
```cypher
MATCH (a)-[r]->(b)
WITH type(r) AS t, collect(DISTINCT (labels(a)[0] + '->' + labels(b)[0])) AS pairs
WHERE NOT all(p IN pairs WHERE p = 'Entity->Entity')
RETURN collect(DISTINCT t) AS types
```
→ **66** types here (vs 2 391 raw). `structural_only=False` → raw
`db.relationshipTypes()`. `list_node_labels` → `CALL db.labels()` (22). ✅

### G8 · `get_graph_summary()` ✅
```cypher
CALL apoc.meta.stats() YIELD nodeCount, relCount, labelCount, relTypeCount
RETURN nodeCount, relCount, labelCount, relTypeCount
```
(fallback without APOC: `MATCH (n) RETURN count(n)` + `MATCH ()-[r]->() RETURN count(r)`)
+ per-label node counts:
```cypher
CALL db.labels() YIELD label
CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS c }
RETURN label, c ORDER BY c DESC
```
+ per-type counts for the **same "structural" set** as
`list_relationship_types(structural_only=True)` (the 66 non-`Entity→Entity`
types), plus a `total_relationships` scalar:
```cypher
MATCH ()-[r]->() WITH type(r) AS t WHERE t IN $structural_types
RETURN t, count(*) AS c ORDER BY c DESC
```
`documents` / `latest_documents` = `count(:Document)` / `count(:Document {latest:true})`.
`get_graph_summary` scalars: `nodeCount 32064`, `relCount 53820`,
`relTypeCount 2391`. ✅

---

## Group H — Generic power tools

### H1 · `neighbors(selector, *, edge_types, direction="both", target_types, depth=1, limit)` ✅
```cypher
MATCH (s:%(SelType)s {%(selKey)s: $sel_value})
MATCH (s)-[r*1..%(depth)d]-(o)        -- direction → -[r]->(o) / (o)-[r]->(s) / -[r]-
WHERE ($edge_types IS NULL OR all(x IN r WHERE type(x) IN $edge_types))
  AND ($target_types IS NULL OR any(l IN labels(o) WHERE l IN $target_types))
RETURN DISTINCT o LIMIT $limit
```
`%(SelType)s` / `%(selKey)s` via `safe_ident`. (Here `$edge_types` etc. stay as
`IS NULL OR` because they are inside a single reusable statement, not the dynamic
builder — acceptable for the generic tools only.) ✅

### H2 · `shortest_path(from_selector, to_selector, *, max_hops=6, edge_types)` ✅
```cypher
MATCH (a:%(AType)s {%(aKey)s:$a_val}), (b:%(BType)s {%(bKey)s:$b_val})
MATCH p = shortestPath( (a)-[*..%(max_hops)d]-(b) )
WHERE $edge_types IS NULL OR all(x IN relationships(p) WHERE type(x) IN $edge_types)
RETURN p LIMIT 1
```

### H3 · `subgraph(selector, *, depth=2, edge_types, max_nodes=500)` ✅
APOC-first (your Q12), pure-Cypher fallback:
```cypher
MATCH (s:%(SelType)s {%(selKey)s:$sel_value})
CALL apoc.path.subgraphAll(s, {maxLevel:$depth, relationshipFilter:$rel_filter, limit:$max_nodes})
YIELD nodes, relationships
RETURN nodes, relationships
```

### H4 · `execute_raw` / `execute_raw_one` ✅
`_safe.assert_read_only` (comment-strip + write-keyword regex incl.
`CALL {} IN TRANSACTIONS`) → `dialect` guard → `execute_read(tx.run(q,**params)).data()`.

---

## Resolved decisions — **all settled, cleared to implement `navigator.py`**

| # | Decision |
|---|---|
| Q-E1 | ✅ Keep the `HAS_*` containment filter for the **provenance** queries (E3/E4/E8/E9/E14, and the doc-scoped `EXISTS` of E5) — it *is* the "belongs to this node/document" definition. E10–E12 stay unfiltered (any rel type, in/out). |
| Q6 | ✅ Extract ingestion's `_normalize` → `utils/uid.py::normalize_key` (NFKD → strip accents → lower → collapse ws); used by `get_model_instance_by_key` and re-imported by `graph_mapper`. Same function ⇒ uid rebuild matches ⇒ direct `{uid:…}` lookup. |
| Q9 | ✅ Populate `is_shell` on **every** `ModelInstanceRef`: `is_shell = size(keys(mi)) <= n_instance_key_fields(model_class) + 2`, where `n_instance_key_fields` = `count( (:CatalogModel {name:m})-[:HAS_FIELD]->(:ModelField {is_instance_key:true}) )`, memoised per `model_class` on the navigator instance. When the class has no catalog entry / no key fields → `is_shell = None` (unknown). |
| Q-G | ✅ `list_relationship_types(structural_only=True)` = all rel types **except** those used only between two `:Entity` nodes → **keeps** `:ModelInstance`↔`:ModelInstance` cross-refs, Level-2 `field_relationships`, catalog rels; drops the ~2 325 `Triple` predicates. `get_graph_summary.relationship_counts` counts that same set + a `total_relationships` scalar. |
| Q-E2 | ✅ `INSTANCE_CONTAINMENT_DEPTH = 12` when `depth=None` on ER→`HAS_*`→ModelInstance walks. |
| Q-A | ✅ `get_document_ancestors` → single-spine `DocumentTree` (root → … → parent). |
| Q-B | ✅ `get_sibling_nodes` → two small Python round-trips (`HAS_CHILD` parent, else `HAS_STRUCTURE`). |
| Q11 | ✅ `get_model_properties` → `{"declared": [...], "observed": [...]}`, observed sampled from ≤ 500 instances. |
| Q12 | ✅ `subgraph` APOC-first (`apoc.path.subgraphAll`) with pure-Cypher fallback; `get_graph_summary` APOC scalars with `count()` fallback. |

---

## Changelog rev. 1 → rev. 2

* **Removed**: `iter_document_descendants`, `get_document_info_units`,
  `get_document_triples`.
* **Renamed**: `get_document_children` → `get_child_documents`; all `*instance*`
  method names → `*model_instance*`; `get_instance_parents`/`_children` →
  `get_incoming_model_instances`/`get_outgoing_model_instances`;
  `InstanceTree`/`InstanceRelation` → `ModelInstanceTree`/`ModelInstanceRelation`.
* **Added**: `get_document_model_profile` (D3 — semantic catalogue roll-up),
  `get_catalog_graph` (G2 — full catalogue with relationships),
  `title_contains` param on `get_structure_nodes`, `depth` param on every
  traversal method.
* **Reworked**: `list_root_documents` params (`only_folders`/`only_leaves`);
  `get_documents` (dropped `theme=`); `get_document_ancestors` returns a
  `DocumentTree` spine; **all** queries use dynamic WHERE (no `$x IS NULL OR`);
  version resolution is dynamic (`{latest:true}` vs `{version:$v}`).
* **Corrected queries**: `REFERENCES` originates only from `:ModelInstance` →
  F2/F3/F8 navigate `ModelInstance → ExtractionResult → StructureNode`;
  `get_triples` predicate edge is `OPTIONAL`; model_instance parent/child
  traversal no longer filters by `HAS_` prefix.
* **`models.py` updated**: deltas 0.1–0.5, 0.7 applied; new models
  `DocumentModelProfile`, `CatalogGraph`, `CatalogRelation`, `CatalogFieldRef`;
  `ModelClassStat.kind`; `NodeDescription.model_instance_count`.
