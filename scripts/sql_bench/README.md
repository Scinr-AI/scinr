# sql_bench — graph backend benchmark (WP7.2, decision P7)

Compares the `neo4j`, `postgres` and `sqlite` graph backends on the **same synthetic corpus**, at
1M and 10M nodes. Everything is written and read through the public interfaces (`GraphStore`,
`GraphNavigator`, `delete_document`, `collect_orphans`, `freeze_document`, `restore_document`),
never with raw SQL or Cypher, so it measures the real code path. It is **not part of CI**.

| File | What it does |
|---|---|
| `corpus.py` | Deterministic corpus (fixed seed). `python scripts/sql_bench/corpus.py --nodes 1000000` prints the plan |
| `run.py` | One backend, one size: guard, wipe, ingest, reads, deletes, GC, freeze/restore; writes `results.json` |
| `compare.py` | `accept` (postgres vs neo4j rule) and `scale` (10M/1M ratios) over two `results.json` files |

## The corpus

Per document (about 697 graph nodes): 1 `Document`, 200 `StructureNode` (one tree), 400 `InfoUnit`,
50 `ModelInstance` (class `BenchModel`, keyed by `code`) with 2 relations each (`SUPPLIED_BY` to the
next instance, `RELATED_TO` to the fifth next), 20 `Entity` (10 triples) plus the 26
`ExtractionResult` nodes that tie them to the structure. 10 tenants (`bench-t0` .. `bench-t9`),
one document in ten is public. `--nodes N` becomes `round(N / 697)` documents (1M -> 1,435
documents, 10M -> 14,347). Structure and info units go through `documents.insert_document_graph`;
instances, relations and entities through `extraction.sink()`; the catalog model through
`annotation.catalog_sink()`. Raw files are **not** stored (the cascade delete therefore does not
delete blobs; this is the same for every backend).

## 1. Containers

Benchmark containers of their own (not the test ones: the one on 7698 has a 64 MiB transaction
cap that would distort the ingest). **Same total memory for both** (8 GiB by default). If the
machine cannot give 8 GiB to a container, lower both values in the same proportion and note it
with the results. Run them **one at a time, never together**, on the same machine; stop one
before starting the other.

```bash
docker run -d --name scinr-bench-neo4j --memory=8g -p 7699:7687 \
  -e NEO4J_AUTH=neo4j/benchmark-password \
  -e NEO4J_server_memory_heap_initial__size=3g \
  -e NEO4J_server_memory_heap_max__size=3g \
  -e NEO4J_server_memory_pagecache_size=4g \
  -e NEO4J_db_memory_transaction_max=0 \
  -e NEO4J_db_memory_transaction_total_max=0 \
  -e NEO4J_dbms_memory_transaction_total_max=0 \
  -v /data \
  neo4j:5
```

```bash
docker run -d --name scinr-bench-postgres --memory=8g -p 55433:5432 \
  -e POSTGRES_USER=newton_admin -e POSTGRES_PASSWORD=newton_admin -e POSTGRES_DB=newton_bench \
  -v /var/lib/postgresql/data \
  postgres:16 \
  -c shared_buffers=2GB -c effective_cache_size=6GB \
  -c work_mem=64MB -c maintenance_work_mem=1GB
```

(`-v /data` and `-v /var/lib/postgresql/data` are anonymous volumes. The three `*_memory_transaction_*`
settings set the per-transaction, per-database and global transaction memory limits to 0, i.e.
unlimited; the global one defaults to 70 % of the heap. Check them once the container is up:
`SHOW SETTINGS YIELD name, value WHERE name CONTAINS 'memory.transaction'`.)

Wait until they accept connections (`docker logs -f ...`) before running.

## 2. Running

The script **wipes** its target before every repetition and refuses to start unless
`SCINR_SQL_ALLOW_WIPE=1` and the target is a benchmark target: neo4j on port **7699**, postgres on
port **55433** (schema `newton_bench`), or a SQLite file (`*.db` / `*.sqlite`, default under the
temp dir). Credentials come from `SCINR_BENCH_NEO4J_URI`, `SCINR_BENCH_NEO4J_USER`,
`SCINR_BENCH_NEO4J_PASSWORD`, `SCINR_BENCH_PG_URL` (the defaults match the commands above) or the
`--neo4j-*` / `--sql-url` flags.

```bash
export SCINR_SQL_ALLOW_WIPE=1
mkdir -p results
.venv/bin/python scripts/sql_bench/run.py --backend neo4j    --nodes 1000000  --out results/neo4j-1m.json
.venv/bin/python scripts/sql_bench/run.py --backend postgres --nodes 1000000  --out results/postgres-1m.json
.venv/bin/python scripts/sql_bench/run.py --backend sqlite   --nodes 1000000  --out results/sqlite-1m.json
.venv/bin/python scripts/sql_bench/run.py --backend neo4j    --nodes 10000000 --out results/neo4j-10m.json
.venv/bin/python scripts/sql_bench/run.py --backend postgres --nodes 10000000 --out results/postgres-10m.json
```

Run matrix (decision P7), **one run at a time**:

| Size | neo4j | postgres | sqlite |
|---|---|---|---|
| 1M | yes | yes | yes |
| 10M | yes | yes | no |

Each run does 3 repetitions (`--reps`), each in its own process, from an empty database, and
reports the **median** per metric (the individual repetitions are kept under `runs`). A 10M run
repeats a 10M ingest three times: it takes hours. Use `--reps 1` for a trial. Other flags:
`--workers` (documents ingested at once, default 4), `--large-scale` (size factor of the
freeze/restore document, default 20 = about 14,000 nodes), `--tracemalloc`.

### What is measured

| Operation | Metric | How |
|---|---|---|
| `ingest` | seconds, nodes/s, `peak_mib` | whole corpus through the store, `--workers` documents at once |
| `get_structure_tree` | p50 / p95 ms | `get_structure_subtree(root, include_info_units=True)` on 100 random documents |
| `get_info_units` | p50 / p95 ms | `get_info_units(node_id)` on a random node of 100 documents |
| `search_info_units` | p50 / p95 ms | 50 single-word terms, each in a random tenant, `limit=25` |
| `get_model_instance_by_key` | p50 / p95 ms | 100 random instances |
| `subgraph` | p50 / p95 ms | `subgraph(ModelInstance, depth=2)` on 100 instances |
| `shortest_path` | p50 / p95 ms | between two instances 2 to 14 steps apart, over the two instance relations |
| `delete_document` | mean s | cascade delete of 20 documents, one by one |
| `collect_orphans` | s | one sweep of each of the 10 tenants and the public one, after the deletes |
| `freeze`, `restore` | s, `peak_mib` | `freeze_document` / `restore_document` of one large document |

With fewer than 100 documents (small `--nodes`) the samples are capped at the number of documents.
One warm-up call precedes each series and is not counted.

**Deviations from the plan's table (no method with that exact shape):**

* `get_structure_tree` -> `get_structure_subtree` (the tree of a node; the document root node
  holds all 200 nodes of the document).
* `get_info_units(document_key=...)` -> there is no document-level method: `get_info_units` takes a
  `node_id`. The closest is one node's info units. (`get_document_entities` and friends do
  document-level reads, but not of info units.)
* `shortest_path` is called with `edge_types=["SUPPLIED_BY", "RELATED_TO"]`. Without it the SQL
  navigator may take minutes per call on one document (see "Findings while writing the bench").
* `neo4j` has no SQL storage, so its freeze snapshots go to a throwaway SQLite file
  (`freeze_backend="sqlite"`); `freeze` / `restore` timings for neo4j therefore include that
  snapshot store, not MongoDB. Postgres and SQLite use their own engine for snapshots.

### Memory

`peak_mib` is the peak **resident set size of the benchmark process** above its level when the
step started, sampled every 50 ms. It includes the C buffers of the drivers, which is what the
"memory peak of the process" criterion is about, and does not slow the run. `scripts/memory_bench`
uses `tracemalloc` (peak of live Python allocations, allocator independent); `--tracemalloc`
switches to it, but then **all timings are inflated** (the result records `memory_mode` and
`compare.py` refuses to mix modes). The database server's own memory is not counted: both
containers get the same `--memory`.

## 3. Acceptance

```bash
.venv/bin/python scripts/sql_bench/compare.py accept results/neo4j-1m.json  results/postgres-1m.json
.venv/bin/python scripts/sql_bench/compare.py accept results/neo4j-10m.json results/postgres-10m.json
# growth of each engine from 1M to 10M (to see whether the gap between engines grows)
.venv/bin/python scripts/sql_bench/compare.py scale results/neo4j-1m.json    results/neo4j-10m.json
.venv/bin/python scripts/sql_bench/compare.py scale results/postgres-1m.json results/postgres-10m.json
```

`accept` (baseline neo4j, candidate postgres, **same size**) exits with 1 if

* any navigation operation (p50 or p95) is more than **3x** slower in postgres, or
* the peak memory of the ingest is more than **20 %** above neo4j's.

If the rule is not met at either size, **stop and ask with the numbers**; do not optimize blindly.
Record the machine (printed in `meta`), the container memory used and the results in
`docs/user-guides/sql-backends-performance.md` (not written yet: it needs real numbers).

## Smoke test

`--smoke-allow-port PORT` makes the guard also accept PORT as the benchmark port of the backend. It
exists **only to smoke-test the script** against a disposable test container (for example
`postgres-test` on 55432 or the golden neo4j on 7698) with a small `--nodes`. Never use it for a
real run. A SQLite smoke run needs no flag:

```bash
SCINR_SQL_ALLOW_WIPE=1 .venv/bin/python scripts/sql_bench/run.py \
  --backend sqlite --nodes 20000 --reps 1 --out /tmp/smoke.json
```

## Findings while writing the bench

* `SqlGraphNavigator.shortest_path` without `edge_types`, on SQLite, did not finish in minutes for
  one pair of instances of a ~700-node document (the bounded traversal explodes over all edge
  types). With `edge_types` restricted to the instance relations it takes about 3.5 s per call on
  SQLite at 20k nodes (about 15 ms on Postgres at 8k nodes). Worth a look before the real run.
* `restore_document` of a 14,000-node document took about 34 s and about 175 MiB of RSS on SQLite
  at 20k nodes (freeze: 0.7 s, 14 MiB).
