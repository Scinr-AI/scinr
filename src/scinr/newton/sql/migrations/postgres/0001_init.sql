CREATE SCHEMA IF NOT EXISTS {schema}
-- statement
-- Per-document node tables
CREATE TABLE {schema}.documents (
    _key               TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL,
    path               TEXT NOT NULL,
    version            BIGINT NOT NULL,
    name               TEXT,
    latest             BOOLEAN,
    is_folder          BOOLEAN,
    raw_file_id        TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    frozen             BOOLEAN,
    frozen_blob_id     TEXT,
    deletion_pending   BOOLEAN,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (tenant_id, path, version)
)
-- statement
CREATE INDEX documents_tenant_latest ON {schema}.documents (tenant_id, latest)
-- statement
CREATE INDEX documents_tenant_name ON {schema}.documents (tenant_id, name)
-- statement
CREATE INDEX documents_user ON {schema}.documents (created_by_user_id)
-- statement
CREATE INDEX documents_job ON {schema}.documents (job_id)
-- statement
CREATE INDEX documents_blob ON {schema}.documents (tenant_id, frozen_blob_id) WHERE frozen_blob_id IS NOT NULL
-- statement
CREATE TABLE {schema}.structure_nodes (
    id                 TEXT PRIMARY KEY,
    tenant_id          TEXT,
    role               TEXT,
    node_id            TEXT,
    title              TEXT,
    theme              TEXT,
    appearance_order   BIGINT,
    row_index          BIGINT,
    source_page_ids    TEXT[],
    created_by_user_id TEXT,
    job_id             TEXT,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    _document_key      TEXT
)
-- statement
CREATE INDEX structure_nodes_doc ON {schema}.structure_nodes (_document_key)
-- statement
CREATE INDEX structure_nodes_tenant_role ON {schema}.structure_nodes (tenant_id, role)
-- statement
CREATE INDEX structure_nodes_tenant_theme ON {schema}.structure_nodes (tenant_id, theme)
-- statement
CREATE INDEX structure_nodes_pages ON {schema}.structure_nodes USING GIN (source_page_ids)
-- statement
CREATE INDEX structure_nodes_user ON {schema}.structure_nodes (created_by_user_id)
-- statement
CREATE INDEX structure_nodes_job ON {schema}.structure_nodes (job_id)
-- statement
CREATE TABLE {schema}.info_units (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    info_unit_id       TEXT,
    title              TEXT,
    description        TEXT,
    "order"            BIGINT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    _document_key      TEXT,
    _structure_node_id TEXT,
    _fts_title         TSVECTOR GENERATED ALWAYS AS
                         (to_tsvector('{fulltext_config}'::regconfig, coalesce(title, ''))) STORED,
    _fts_description   TSVECTOR GENERATED ALWAYS AS
                         (to_tsvector('{fulltext_config}'::regconfig, coalesce(description, ''))) STORED
)
-- statement
CREATE INDEX info_units_doc ON {schema}.info_units (_document_key)
-- statement
CREATE INDEX info_units_node ON {schema}.info_units (_structure_node_id)
-- statement
CREATE INDEX info_units_tenant ON {schema}.info_units (tenant_id)
-- statement
CREATE INDEX info_units_fts_title ON {schema}.info_units USING GIN (_fts_title)
-- statement
CREATE INDEX info_units_fts_description ON {schema}.info_units USING GIN (_fts_description)
-- statement
CREATE TABLE {schema}.model_decisions (
    uid                 TEXT PRIMARY KEY,
    tenant_id           TEXT,
    matched_model_class TEXT,
    source              TEXT,
    created_by_user_id  TEXT,
    job_id              TEXT,
    props               JSONB NOT NULL DEFAULT '{}'::jsonb,
    _document_key       TEXT
)
-- statement
CREATE INDEX model_decisions_doc ON {schema}.model_decisions (_document_key)
-- statement
CREATE INDEX model_decisions_tenant_class ON {schema}.model_decisions (tenant_id, matched_model_class)
-- statement
CREATE TABLE {schema}.proposed_models (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    _document_key      TEXT
)
-- statement
CREATE INDEX proposed_models_doc ON {schema}.proposed_models (_document_key)
-- statement
CREATE TABLE {schema}.proposed_fields (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    _document_key      TEXT
)
-- statement
CREATE INDEX proposed_fields_doc ON {schema}.proposed_fields (_document_key)
-- statement
CREATE TABLE {schema}.supplementary_fields (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    _document_key      TEXT
)
-- statement
CREATE INDEX supplementary_fields_doc ON {schema}.supplementary_fields (_document_key)
-- statement
CREATE TABLE {schema}.complementary_matches (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    model_class        TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    _document_key      TEXT
)
-- statement
CREATE INDEX complementary_matches_doc ON {schema}.complementary_matches (_document_key)
-- statement
CREATE TABLE {schema}.extraction_results (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    node_full_id       TEXT,
    model_class        TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    _document_key      TEXT
)
-- statement
CREATE INDEX extraction_results_doc ON {schema}.extraction_results (_document_key)
-- statement
CREATE INDEX extraction_results_node ON {schema}.extraction_results (node_full_id)
-- statement
CREATE INDEX extraction_results_tenant_class ON {schema}.extraction_results (tenant_id, model_class)
-- statement
-- Deduplicated node tables
CREATE TABLE {schema}.model_instances (
    uid                 TEXT PRIMARY KEY,
    tenant_id           TEXT,
    model_class         TEXT,
    created_by_user_ids TEXT[],
    job_ids             TEXT[],
    props               JSONB NOT NULL DEFAULT '{}'::jsonb
)
-- statement
CREATE INDEX model_instances_tenant_class ON {schema}.model_instances (tenant_id, model_class)
-- statement
CREATE INDEX model_instances_users ON {schema}.model_instances USING GIN (created_by_user_ids)
-- statement
CREATE INDEX model_instances_jobs ON {schema}.model_instances USING GIN (job_ids)
-- statement
CREATE TABLE {schema}.labeled_entities (
    uid                 TEXT PRIMARY KEY,
    tenant_id           TEXT,
    label               TEXT,
    value               TEXT,
    normalized_value    TEXT,
    created_by_user_ids TEXT[],
    job_ids             TEXT[],
    props               JSONB NOT NULL DEFAULT '{}'::jsonb
)
-- statement
CREATE INDEX labeled_entities_tenant_label ON {schema}.labeled_entities (tenant_id, label)
-- statement
CREATE INDEX labeled_entities_tenant_value ON {schema}.labeled_entities (tenant_id, normalized_value)
-- statement
CREATE TABLE {schema}.entities (
    uid                 TEXT PRIMARY KEY,
    tenant_id           TEXT,
    value               TEXT,
    normalized_value    TEXT,
    created_by_user_ids TEXT[],
    job_ids             TEXT[],
    props               JSONB NOT NULL DEFAULT '{}'::jsonb
)
-- statement
CREATE INDEX entities_tenant_value ON {schema}.entities (tenant_id, normalized_value)
-- statement
-- Global catalog
CREATE TABLE {schema}.catalog_models (name TEXT PRIMARY KEY, props JSONB NOT NULL DEFAULT '{}'::jsonb)
-- statement
CREATE TABLE {schema}.model_fields (
    _key  TEXT PRIMARY KEY,
    name  TEXT NOT NULL,
    model TEXT NOT NULL,
    props JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (model, name)
)
-- statement
CREATE TABLE {schema}.entity_labels (label TEXT PRIMARY KEY, props JSONB NOT NULL DEFAULT '{}'::jsonb)
-- statement
CREATE TABLE {schema}.themes (path TEXT PRIMARY KEY, props JSONB NOT NULL DEFAULT '{}'::jsonb)
-- statement
-- Relationships
CREATE TABLE {schema}.edges (
    id        BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    type      TEXT NOT NULL,
    src_label TEXT NOT NULL,
    src_key   TEXT NOT NULL,
    dst_label TEXT NOT NULL,
    dst_key   TEXT NOT NULL,
    _ident    TEXT NOT NULL DEFAULT '',
    props     JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (src_label, src_key, type, dst_label, dst_key, _ident)
)
-- statement
CREATE INDEX edges_dst ON {schema}.edges (dst_label, dst_key, type)
-- statement
CREATE INDEX edges_tenant_type ON {schema}.edges (tenant_id, type)
-- statement
CREATE TABLE {schema}.catalog_edges (
    id        BIGSERIAL PRIMARY KEY,
    type      TEXT NOT NULL,
    src_label TEXT NOT NULL,
    src_key   TEXT NOT NULL,
    dst_label TEXT NOT NULL,
    dst_key   TEXT NOT NULL,
    _ident    TEXT NOT NULL DEFAULT '',
    props     JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (src_label, src_key, type, dst_label, dst_key, _ident)
)
-- statement
CREATE INDEX catalog_edges_dst ON {schema}.catalog_edges (dst_label, dst_key, type)
-- statement
CREATE VIEW {schema}.all_edges AS
    SELECT tenant_id, type, src_label, src_key, dst_label, dst_key, props FROM {schema}.edges
    UNION ALL
    SELECT NULL, type, src_label, src_key, dst_label, dst_key, props FROM {schema}.catalog_edges
-- statement
-- Storage
CREATE TABLE {schema}.raw_files (
    id                 TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL,
    created_by_user_id TEXT,
    job_id             TEXT,
    filename           TEXT NOT NULL,
    folder_path        TEXT,
    content_type       TEXT NOT NULL,
    size_bytes         BIGINT NOT NULL,
    checksum_sha256    TEXT NOT NULL,
    stored_at          TIMESTAMPTZ NOT NULL,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    _chunk_count       INTEGER NOT NULL DEFAULT 0
)
-- statement
CREATE INDEX raw_files_tenant ON {schema}.raw_files (tenant_id, filename, folder_path)
-- statement
CREATE TABLE {schema}.raw_file_chunks (
    file_id   TEXT NOT NULL REFERENCES {schema}.raw_files (id) ON DELETE CASCADE,
    n         INTEGER NOT NULL,
    tenant_id TEXT NOT NULL,
    data      BYTEA NOT NULL,
    PRIMARY KEY (file_id, n)
)
-- statement
CREATE TABLE {schema}.pages (
    id                 TEXT PRIMARY KEY,
    raw_file_id        TEXT NOT NULL,
    tenant_id          TEXT NOT NULL,
    created_by_user_id TEXT,
    job_id             TEXT,
    filename           TEXT NOT NULL,
    folder_path        TEXT,
    page_index         INTEGER NOT NULL,
    markdown           TEXT NOT NULL,
    converted_at       TIMESTAMPTZ NOT NULL,
    start_ms           BIGINT,
    end_ms             BIGINT,
    transcript         JSONB,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb
)
-- statement
CREATE UNIQUE INDEX pages_raw_file ON {schema}.pages (raw_file_id, page_index)
-- statement
CREATE TABLE {schema}.page_images (
    id                 TEXT PRIMARY KEY,
    raw_file_id        TEXT NOT NULL,
    tenant_id          TEXT NOT NULL,
    created_by_user_id TEXT,
    job_id             TEXT,
    page_index         INTEGER NOT NULL,
    image_index        INTEGER NOT NULL,
    kind               TEXT NOT NULL,
    content_type       TEXT NOT NULL,
    size_bytes         BIGINT NOT NULL,
    checksum_sha256    TEXT NOT NULL,
    width              INTEGER,
    height             INTEGER,
    time_ms            BIGINT,
    description        TEXT NOT NULL,
    filename           TEXT NOT NULL,
    folder_path        TEXT,
    stored_at          TIMESTAMPTZ NOT NULL,
    props              JSONB NOT NULL DEFAULT '{}'::jsonb,
    _chunk_count       INTEGER NOT NULL DEFAULT 0
)
-- statement
CREATE INDEX page_images_raw_file ON {schema}.page_images (raw_file_id, page_index, image_index)
-- statement
CREATE UNIQUE INDEX page_images_identity ON {schema}.page_images (tenant_id, raw_file_id, page_index, image_index)
-- statement
CREATE TABLE {schema}.page_image_chunks (
    image_id  TEXT NOT NULL REFERENCES {schema}.page_images (id) ON DELETE CASCADE,
    n         INTEGER NOT NULL,
    tenant_id TEXT NOT NULL,
    data      BYTEA NOT NULL,
    PRIMARY KEY (image_id, n)
)
-- statement
CREATE TABLE {schema}.frozen_snapshots (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    mode         TEXT,
    frozen_at    TEXT,
    content_type TEXT NOT NULL,
    size_bytes   BIGINT NOT NULL,
    stored_at    TIMESTAMPTZ NOT NULL,
    documents    JSONB NOT NULL DEFAULT '[]'::jsonb,
    props        JSONB NOT NULL DEFAULT '{}'::jsonb,
    _chunk_count INTEGER NOT NULL DEFAULT 0
)
-- statement
CREATE INDEX frozen_snapshots_tenant ON {schema}.frozen_snapshots (tenant_id)
-- statement
CREATE TABLE {schema}.frozen_snapshot_chunks (
    snapshot_id TEXT NOT NULL REFERENCES {schema}.frozen_snapshots (id) ON DELETE CASCADE,
    n           INTEGER NOT NULL,
    tenant_id   TEXT NOT NULL,
    data        BYTEA NOT NULL,
    PRIMARY KEY (snapshot_id, n)
)
