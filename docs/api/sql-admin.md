# SQL Administration API

One-off administration calls for a PostgreSQL schema: create or verify the schema, and
grant or revoke tenant access when Row-Level Security (RLS) is enabled. They take an
`admin_url` (a connection that can create objects and grants), build their own short-lived
engine and never store the URL. They are synchronous only. For the walkthrough, including
who runs what, see the [SQL graph backends guide](../user-guides/sql-graph.md).

All of them are importable from `scinr.newton`.

::: scinr.newton.sql.admin
    options:
      members:
        - setup_sql_schema
        - grant_tenant_access
        - revoke_tenant_access
        - list_tenant_access
        - SchemaStatus
        - TenantAccess
