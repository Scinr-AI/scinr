# SQL backends: test environment

SQLite needs nothing. Postgres tests run against a disposable container and are skipped
(with the reason) when it is not there.

```bash
docker run -d --name postgres-test -p 55432:5432 \
  -e POSTGRES_USER=newton_admin -e POSTGRES_PASSWORD=newton_admin -e POSTGRES_DB=newton_test \
  postgres:16

export SCINR_SQL_PG_URL=postgresql://newton_admin:newton_admin@127.0.0.1:55432/newton_test
export SCINR_SQL_ALLOW_WIPE=1
.venv/bin/python -m pytest tests/unit/test_sql_dialect.py tests/unit/test_sql_migrations.py
```

| Variable | Value |
|---|---|
| `SCINR_SQL_PG_URL` | URL of the container (the one above is the default) |
| `SCINR_SQL_ALLOW_WIPE` | `1`; without it the Postgres tests are skipped |

The fixtures (`tests/sql/conftest.py`) drop only schemas called `newton_test_<pid>`, and only in
a local database whose name contains `test`. Never point them at another database.

Fixtures other suites can import: `sqlite_url`, `pg_url`, `pg_schema` and `sql_backend`
(parametrised over `sqlite` and `postgres`; it calls `configure(...)` and `reset_engines()`).

The minimum supported Postgres is 14 (`CYCLE` clause): repeat the run with `postgres:14` at
the end of phase 7.
