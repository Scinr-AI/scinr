-- Version of the Row-Level-Security migrations (migrations/postgres/rls/) applied to this schema.
ALTER TABLE {schema}.newton_schema ADD COLUMN IF NOT EXISTS rls_version INTEGER NOT NULL DEFAULT 0
