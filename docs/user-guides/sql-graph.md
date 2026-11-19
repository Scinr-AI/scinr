# SQL Graph Backends (PostgreSQL and SQLite)

Neo4j is the default graph backend, but `scinr.newton` can keep the **whole graph** — the write path of the pipeline, the read-only [navigation API](graph-navigation.md), the raw files and pages, and the document snapshots — in **PostgreSQL** or **SQLite** instead. The public API does not change: the engine is chosen by configuration.

- **PostgreSQL** (no extension required; tested on 14 and 16, and the code enforces no minimum version) suits a team that already runs Postgres and does not want to operate two more databases. It can isolate tenants in the database itself with [Row-Level Security](#row-level-security-postgres-only).
- **SQLite** needs no server: one `.db` file (plus a folder of binaries). It suits local tools, single-machine setups and tests.

Neo4j keeps working exactly as before and remains the default (`graph_backend="neo4j"`).

---

## Installation

```bash
pip install "scinr[postgres]"   # SQLAlchemy + psycopg 3
pip install "scinr[sqlite]"     # SQLAlchemy + aiosqlite
```

Neo4j is an optional extra since this version: install it with `pip install "scinr[neo4j]"` if you use `graph_backend="neo4j"`. Without the extras, `import scinr.newton` and `configure()` still work. What happens when a backend is first used without its extra depends on the component:

- **Graph store** (`get_graph_store()`): an error that names the `pip install` to run (`ConfigurationError` for the SQL backends, `GraphConnectionError` for Neo4j).
- **Navigator, storage and freeze factories** with a SQL backend: no friendly message. The SQL modules import SQLAlchemy directly, so you get the plain `ImportError` (`ModuleNotFoundError`). Install `scinr[postgres]` or `scinr[sqlite]` before using them.

---

## Minimal Configuration

The three axes — `graph_backend`, `storage_backend` and `freeze_backend` — each accept `"postgres"` and `"sqlite"`. All of them share a single connection URL, `sql_url`.

### SQLite

```python
from scinr.newton import configure

configure(
    graph_backend="sqlite",
    storage_backend="sqlite",    # raw files, pages and images (freeze_backend inherits it)
    sql_url="sqlite:////var/lib/scinr/newton.db",
    llm=...,                     # as usual, for the LLM stages
)
```

No Neo4j or MongoDB settings are needed: the Neo4j credentials are required only when `graph_backend="neo4j"`.

newton creates the `.db` file on the first connection, but not its directory: the directory of `sql_url` (and of `sqlite_files_dir`, if you set it) must exist and be writable by the process.

### PostgreSQL

```python
from scinr.newton import configure

configure(
    graph_backend="postgres",
    storage_backend="postgres",
    sql_url="postgresql://newton:secret@db.internal:5432/scinr",
    sql_schema="newton",         # default; the tables live in this Postgres schema
    llm=...,
)
```

Quick local recipe with Docker. The database must exist before newton connects: `POSTGRES_DB` creates it. newton never runs `CREATE DATABASE` and never creates roles; it creates its schema and tables inside the database.

```bash
docker run -d --name scinr-pg -p 5432:5432 \
  -e POSTGRES_USER=newton -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=scinr \
  postgres:16
```

With that container the URL above works as is (the `newton` user it creates is a superuser, which is fine for trying simple mode and is **not** what you want for RLS, see below). On an existing server, create the database yourself and connect with a role that can create the schema in it. Roles are global to the server, so a role created once is valid for every database; if the server revoked `CONNECT` from `PUBLIC`, also run `GRANT CONNECT ON DATABASE scinr TO <role>`.

Or with environment variables:

```bash
GRAPH_BACKEND=postgres
STORAGE_BACKEND=postgres
SQL_URL=postgresql://newton:secret@db.internal:5432/scinr
SQL_SCHEMA=newton
```

### Rules

- `sql_url` is required when any axis is `"postgres"` or `"sqlite"`, and its scheme (`postgresql://`, `postgres://` or `sqlite:///`) must match the axes. A driver suffix (`postgresql+psycopg://`) is accepted and ignored.
- The axes cannot mix engines: the graph in Postgres and the storage in SQLite is a `ConfigurationError`. A SQL engine for one axis and MongoDB for another is fine (`graph_backend="postgres"` with `storage_backend="mongodb"`, or the reverse).
- `sql_url` is treated like the Neo4j and MongoDB URIs: it is redacted in logs and in `repr(config)`.
- `sql_pool_size` (default `10`) sizes the connection pool. `sql_schema` is ignored by SQLite. `sqlite_files_dir` and the binaries are described in [Storage Backends](storage-backends.md#postgresql-and-sqlite-backends).
- `sql_fulltext_config` (default `"simple"`) is the Postgres text-search configuration of the full-text index on `info_units`. It is applied **only when the schema is created**; changing it afterwards logs a warning and has no effect until the schema is recreated.

See [Configuration](../configuration.md#sql-postgresql-and-sqlite) for every setting.

### Simple mode (default)

In simple mode newton creates and migrates the schema by itself, on the first use of the process: if the schema does not exist it is created, and if it is behind it is migrated. The database user must be able to create tables (in Postgres, to create the `sql_schema` schema in its database). No superuser is needed. This is as simple as Neo4j and MongoDB are today, and it is what SQLite always uses.

Isolation between tenants is the tenant filter newton adds to every query — the same barrier Neo4j has.

---

## Row-Level Security (Postgres only)

In **RLS mode** the database itself enforces which tenants a Postgres user can read and write, on top of the tenant filter newton always adds. newton connects with a **restricted** user that owns nothing and cannot create anything; an administrator prepares the schema and declares who sees what. SQLite has no RLS and always uses simple mode.

The mode is detected from the schema, with no setting: the first call of `setup_sql_schema(..., enable_rls=True)` marks the schema as RLS, and from then on newton only verifies it.

### Who does what

| Who | Does | Must not |
|---|---|---|
| **Administrator** (the connection of `admin_url`, normally the owner of the tables) | `setup_sql_schema`, `grant_tenant_access`, `revoke_tenant_access`, `list_tenant_access`, version upgrades | Be used as the connection of the pipeline or the navigator (see below) |
| **Runtime role** (`LOGIN`, not a superuser, no `BYPASSRLS`, not the owner of the tables) | Ingestion and queries on the tenants granted to it. On `newton_tenant_access` it can only `SELECT`, and only its own rows | Create objects or change grants |

Why the administrator should not run the pipeline. The tables are created with `FORCE ROW LEVEL SECURITY`, so the policies apply to their owner as well as to the runtime role: an owner with no grant sees no tenant data. That does not make the administrator connection safe to use at runtime: superusers and roles with `BYPASSRLS` skip the policies entirely, and the owner can alter or disable the policies and edit `newton_tenant_access` (that table has RLS enabled but not forced, and the owner can change it). Keep those credentials out of the processes that handle tenant data.

How many roles to create: one per isolation boundary, that is, one per set of tenants that must not see each other's data. Grant each role only the tenants it serves, one `grant_tenant_access` call per tenant, with `write=True` only for roles that ingest. A role that only reads uses the default `write=False`. `tenant_id=None` grants the public documents (stored as `'*'` in `newton_tenant_access`).

### Why RLS is safe

- One schema holds all tenants. Every data table has a policy that calls the functions `newton_can_read(tenant_id)` / `newton_can_write(tenant_id)`, and those look for a row in `newton_tenant_access` for `current_user` (the connected role) and that tenant.
- A grant is a row in `newton_tenant_access`, not DDL: granting and revoking takes effect in the next transaction and changes no table or policy. The schema changes only when you upgrade `scinr` to a version with a new migration.
- Because identity comes from `current_user`, the application cannot choose its tenants by setting a session variable: what a role sees is what the administrator inserted for it.

Limits to keep in mind:

- The isolation is per database role, not per tenant inside a process (see [Limitation](#limitation)).
- Whoever has the administrator credentials (or a superuser or `BYPASSRLS` role) defeats it.
- The catalog tables (`catalog_models`, `model_fields`, `entity_labels`, `themes`, `catalog_edges`) have no policy: every role sees them.

**Connection poolers.** newton keeps no identity in session variables (it uses `current_user`), but it does issue `SET LOCAL search_path` and `SET TRANSACTION READ ONLY` inside each transaction. Both are transaction-scoped, which should be compatible with a pooler in transaction mode, but this has **not been verified against PgBouncer**.

### Step 1 — Create the runtime role (by hand)

newton never creates Postgres users and never handles their passwords. The operator creates the role the pipeline will connect with:

```sql
-- As a Postgres administrator
CREATE ROLE newton_app LOGIN PASSWORD 'change-me';
```

Do not make it a superuser, do not give it `BYPASSRLS`, and do not make it the owner of the tables: superusers and roles with `BYPASSRLS` skip every policy.

### Step 2 — Create the schema in RLS mode

With an administrator connection (one that can create objects and grants), call `setup_sql_schema`. It creates or migrates the tables, applies the policies, creates `newton_tenant_access` and gives the runtime role the row-level privileges it needs. It is idempotent.

```python
from scinr.newton import setup_sql_schema

status = setup_sql_schema(
    "postgresql://admin:admin-secret@db.internal:5432/scinr",   # admin_url
    enable_rls=True,
    runtime_role="newton_app",          # required with enable_rls; must already exist
    schema="newton",                    # default: Config.sql_schema
)
print(status)   # SchemaStatus(version=..., rls_enabled=True, rls_version=..., fulltext_config='simple')
```

`admin_url` is used by a short-lived connection and forgotten: it is never stored in the configuration, never logged and never part of an error message. These are one-off administration calls, so they are synchronous only. They raise `ValueError` for a URL that is not PostgreSQL, an invalid role name or a role that does not exist yet. To authorize a second runtime role, call it again with `runtime_role` set to that role; only its grants are added.

`setup_sql_schema` also works without `enable_rls` (simple mode, an administrator preparing the schema). A schema already in RLS mode stays in RLS mode: turning it off is not supported.

### Step 3 — Grant tenants to the role

A role sees nothing until it is granted tenants. The grants live in the table `newton_tenant_access` (`role_name`, `tenant_id`, `can_write`):

```python
from scinr.newton import grant_tenant_access, list_tenant_access, revoke_tenant_access

ADMIN_URL = "postgresql://admin:admin-secret@db.internal:5432/scinr"

# Read-only access to tenant "acme"
grant_tenant_access(ADMIN_URL, "newton_app", "acme")

# Read and write
grant_tenant_access(ADMIN_URL, "newton_app", "acme", write=True)   # granting again changes the flag

# Public documents (tenant_id=None) need an explicit grant too
grant_tenant_access(ADMIN_URL, "newton_app", None, write=True)

for grant in list_tenant_access(ADMIN_URL, role="newton_app"):
    print(grant.role, grant.tenant_id, grant.can_write, grant.granted_at)

revoke_tenant_access(ADMIN_URL, "newton_app", "acme")              # effective at once, in the next transaction
```

- `tenant_id=None` means the **public** documents (stored as `"__public__"` in the data, as `'*'` in `newton_tenant_access`). `"*"` itself is reserved: pass `None`.
- There is no "all tenants" grant. A role sees exactly the tenants (and, if granted, the public data) listed for it. A service that attends many tenants from one process needs one grant per tenant.
- `list_tenant_access` returns `TenantAccess(role, tenant_id, can_write, granted_at)` ordered by role and tenant; without `role` it lists every role. `revoke_tenant_access` of a grant that does not exist is not an error.
- All three raise `ValueError` if the schema is not in RLS mode (`RLS is not enabled; run setup_sql_schema(enable_rls=True)`).

### Step 4 — Run newton with the restricted role

```python
configure(
    graph_backend="postgres",
    storage_backend="postgres",
    sql_url="postgresql://newton_app:change-me@db.internal:5432/scinr",
    llm=...,
)
```

The runtime role is not the table owner, so it cannot create anything: newton verifies the schema version and never runs DDL. The one structural change newton still makes while running is the per-property index on `ModelInstance` key fields (the equivalent of Neo4j's on-demand `instance_key` indexes); in RLS mode it goes through a `SECURITY DEFINER` function that can create only that index.

### What a restricted role sees

RLS is not an error condition: it is how the database filters rows. `scinr.newton.sql.schema.rls_enabled()` tells whether the configured target is in RLS mode (it is `False` until a store, navigator or repository has used the schema once).

| Call | Result for a role granted only tenant `A` |
|---|---|
| Read of tenant `A` | The data of `A` |
| Read of tenant `B` | Empty, as if `B` did not exist (no exception) |
| Read with `tenant_id=None` ("all tenants") | **Only `A`**, not everything: the real scope is the role's |
| Read with `include_public=True` | The public data only if the role was granted `None` |
| Write into `B` | `TenantAccessError` |
| `delete_document` of a document the role can see but not write | `TenantAccessError`, before anything is deleted |
| `execute_raw("SELECT ...")` | Only the rows of its tenants |
| Catalog (models, fields, labels, themes) | Visible to every role: the catalog has no policy |

`TenantAccessError` (`scinr.newton.TenantAccessError`) is a `GraphError` and a `StorageError`, so existing `except GraphError` handlers keep catching it.

Public data is one more tenant: a role without the `None` grant does not see public documents even with `include_public=True`, and cannot ingest them.

### Limitation

newton's configuration is global to the process: one process uses one Postgres user. A single process that serves many tenants needs a user with access to all of them (and then its only barrier is newton's tenant filter, as with Neo4j). For database-level isolation per tenant, run processes — or configurations — with different users.

---

## Version Upgrades

The schema carries its version in the table `newton_schema`; migrations only go forward.

| Mode | What to do when you upgrade `scinr` |
|---|---|
| Simple (default, and SQLite) | Nothing. The first use of the new version migrates the schema. |
| RLS | Run `setup_sql_schema(admin_url, enable_rls=True, runtime_role=...)` again with the administrator. Until you do, the runtime role fails with a `SchemaVersionError` (a `GraphConnectionError` and a `ConfigurationError`) that says so. It is not exported from `scinr.newton`: import it from `scinr.newton.sql.schema`. |

A schema that is **newer** than the installed newton raises a `ConfigurationError` asking you to upgrade `scinr`.

---

## Known Differences from Neo4j

The SQL backends are validated against Neo4j with the same scenarios and the same calls, comparing the Pydantic models that come back (`tests/equivalence/`). These differences are known and accepted:

| Area | Neo4j | PostgreSQL | SQLite |
|---|---|---|---|
| **Full-text search** (`search_info_units`) | Lucene query syntax | `websearch_to_tsquery`: words are AND-ed, `"quoted phrases"`, `or`, and `-word` to exclude | FTS5: words and `"quoted phrases"` are AND-ed, a bare `or` becomes OR; `-word` exclusions are **dropped** (FTS5 has no unary NOT) |
| **Full-text score** | Lucene score | `ts_rank` | `bm25` (sign-flipped so higher is better) |
| **`Regex` filter** (`where=`) | Java regex, whole-string match (`=~`) | POSIX (`~`) | Python `re` (`re.search`) |

- **Score and order.** The set of results of a full-text search is what the equivalence suite compares; the score, and therefore the order by score, differs between engines. Do not depend on its absolute value.
- **Regular expressions.** The translator wraps the pattern as `^(?:pattern)$` for every engine. On SQLite the match is a `re.search` of that pattern, and in Python `$` also matches just before a trailing newline, so a value that ends in `\n` can match where Neo4j and PostgreSQL would not. Common syntax (`\d`, classes, alternation, quantifiers) behaves alike; engine-specific constructs (lookarounds, flags, Unicode classes) do not. Write portable patterns if you may switch engines.
- **Restore (D1).** The Neo4j restore creates a relationship unless one with *all* the same properties already exists between the two nodes. In SQL a relationship is unique by its identity (type, ends and the properties that form part of its `MERGE`: `field_name`, `list_index`, `role`), so a snapshot relationship whose identity already exists with other properties is not created. It only differs for a snapshot holding two relationships of equal identity and different properties, which the pipeline never produces.
- **`execute_raw`.** It takes **read-only SQL** with `:name` parameters (`dialect="postgres"` or `"sqlite"`, or omitted), not Cypher. The engine enforces it: a `READ ONLY` transaction in Postgres, `PRAGMA query_only` in SQLite; a statement that writes raises `NavigationError`. SQL is not parsed. On Postgres the configured schema is first in the `search_path`, so table names need no prefix; under RLS the query only sees the rows of the role's tenants. A `dialect=` that does not match the backend fails fast.
- **`get_graph_summary` filtered by `created_by_user_id` or `job_id`.** On Neo4j this call fails with a `CypherSyntaxError` (the alias of the scoped count is also the variable of the array-provenance predicate); the SQL backends answer it. The equivalence suite records it in `KNOWN_DIFFERENCES` (`tests/equivalence/test_read_equivalence.py`).
- **`shortest_path`.** The SQL backends run a bidirectional breadth-first search. When several shortest paths have the same length, the one chosen is the smallest by code-point order of its path string, the same on both engines; Neo4j returns one of them without a defined rule.
- **Variable-length walks and performance.** `neighbors`, `subgraph` and the other variable-length walks search an index per direction and edge table instead of scanning every relationship.
- **Unspecified order.** Where the API does not guarantee an order and the Neo4j query has no total `ORDER BY` (for example `neighbors`, `get_triples`, `list_entity_labels` on ties), Neo4j's order changes from run to run; the SQL backends return a deterministic one.
- **Concurrency on SQLite.** There is one writer: write transactions are serialized (WAL mode, a 30-second busy timeout and `BEGIN IMMEDIATE`), so `parallel_docs > 1` works without `database is locked` errors, but writes do not run in parallel.
- **Tenant isolation on SQLite** is the software filter only (no RLS).
- **External Cypher consumers** (for example a service that runs Cypher against the graph, such as `graphApi`) keep working only with Neo4j.

Snapshots are portable: the format is the same on every backend (`SNAPSHOT_SCHEMA_VERSION = 1`), so a backup made on Neo4j can be restored on Postgres or SQLite and the other way round. This is the migration path between engines; there is no dedicated migration tool.

---

## Table Summary

One table per node type (typed columns for what is filtered or sorted, plus a JSON `props` column for the rest) and one table of relationships. The authoritative definition is the migrations in `scinr.newton.sql` (`migrations/postgres/` and `migrations/sqlite/`).

| Group | Tables |
|---|---|
| Per document (with `tenant_id`, `created_by_user_id`, `job_id`) | `documents`, `structure_nodes`, `info_units`, `model_decisions`, `proposed_models`, `proposed_fields`, `supplementary_fields`, `complementary_matches`, `extraction_results` |
| Deduplicated by content (per tenant) | `model_instances`, `labeled_entities`, `entities` |
| Global catalog (no tenant, no RLS) | `catalog_models`, `model_fields`, `entity_labels`, `themes` |
| Relationships | `edges` (with `tenant_id`) and `catalog_edges` (catalog to catalog, no tenant); the view `all_edges` joins both |
| Storage and snapshots | `raw_files`, `pages`, `page_images`, `frozen_snapshots`, plus `raw_file_chunks`, `page_image_chunks` and `frozen_snapshot_chunks` in Postgres (SQLite keeps the binaries as files) |
| Control | `newton_schema` (version and mode); `newton_tenant_access` in RLS mode |
| SQLite only | `info_units_fts` (FTS5 index) |

Details worth knowing:

- Keys are computed in Python, exactly as for Neo4j (`make_uid`, `make_structure_node_id`), so one document produces the same keys on every engine.
- `edges` is unique on (`src_label`, `src_key`, `type`, `dst_label`, `dst_key` and the identity properties of the relationship). Relationship `type` is validated text (`^[A-Za-z0-9_]+$`), which covers the dynamic types defined by user models.
- There are no foreign keys from `edges` to the node tables; integrity is kept by the store and by the cascading delete. No relationship goes from a catalog node to a data node.
- Columns whose name starts with `_` are internal helpers and never appear in navigation results, snapshots or dumps.
- The tenant of every row is the stored form: `"__public__"` for public data.

---

## See Also

- **[Configuration](../configuration.md)** — every SQL setting and its environment variable.
- **[Storage Backends](storage-backends.md)** — raw files, pages and snapshots on Postgres and SQLite.
- **[Neo4j Graph Storage](neo4j-graph.md)** — the graph model, which the SQL tables reproduce.
- **[Graph Navigation](graph-navigation.md)** — the read-only API, identical on every backend.
- **[Architecture — Graph backends](../architecture.md#8e-graph-backends-neo4j-and-sql)** — how the engines are plugged in, and how to add one.
