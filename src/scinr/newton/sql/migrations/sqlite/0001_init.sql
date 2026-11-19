-- Per-document node tables
CREATE TABLE documents (
    _key               TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL,
    path               TEXT NOT NULL,
    version            INTEGER NOT NULL,
    name               TEXT,
    latest             INTEGER,
    is_folder          INTEGER,
    raw_file_id        TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    frozen             INTEGER,
    frozen_blob_id     TEXT,
    deletion_pending   INTEGER,
    props              TEXT NOT NULL DEFAULT '{}',
    UNIQUE (tenant_id, path, version)
)
-- statement
CREATE INDEX documents_tenant_latest ON documents (tenant_id, latest)
-- statement
CREATE INDEX documents_tenant_name ON documents (tenant_id, name)
-- statement
CREATE INDEX documents_user ON documents (created_by_user_id)
-- statement
CREATE INDEX documents_job ON documents (job_id)
-- statement
CREATE INDEX documents_blob ON documents (tenant_id, frozen_blob_id) WHERE frozen_blob_id IS NOT NULL
-- statement
CREATE TABLE structure_nodes (
    id                 TEXT PRIMARY KEY,
    tenant_id          TEXT,
    role               TEXT,
    node_id            TEXT,
    title              TEXT,
    theme              TEXT,
    appearance_order   INTEGER,
    row_index          INTEGER,
    source_page_ids    TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              TEXT NOT NULL DEFAULT '{}',
    _document_key      TEXT
)
-- statement
CREATE INDEX structure_nodes_doc ON structure_nodes (_document_key)
-- statement
CREATE INDEX structure_nodes_tenant_role ON structure_nodes (tenant_id, role)
-- statement
CREATE INDEX structure_nodes_tenant_theme ON structure_nodes (tenant_id, theme)
-- statement
CREATE INDEX structure_nodes_user ON structure_nodes (created_by_user_id)
-- statement
CREATE INDEX structure_nodes_job ON structure_nodes (job_id)
-- statement
CREATE TABLE info_units (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    info_unit_id       TEXT,
    title              TEXT,
    description        TEXT,
    "order"            INTEGER,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              TEXT NOT NULL DEFAULT '{}',
    _document_key      TEXT,
    _structure_node_id TEXT
)
-- statement
CREATE INDEX info_units_doc ON info_units (_document_key)
-- statement
CREATE INDEX info_units_node ON info_units (_structure_node_id)
-- statement
CREATE INDEX info_units_tenant ON info_units (tenant_id)
-- statement
CREATE TABLE model_decisions (
    uid                 TEXT PRIMARY KEY,
    tenant_id           TEXT,
    matched_model_class TEXT,
    source              TEXT,
    created_by_user_id  TEXT,
    job_id              TEXT,
    props               TEXT NOT NULL DEFAULT '{}',
    _document_key       TEXT
)
-- statement
CREATE INDEX model_decisions_doc ON model_decisions (_document_key)
-- statement
CREATE INDEX model_decisions_tenant_class ON model_decisions (tenant_id, matched_model_class)
-- statement
CREATE TABLE proposed_models (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              TEXT NOT NULL DEFAULT '{}',
    _document_key      TEXT
)
-- statement
CREATE INDEX proposed_models_doc ON proposed_models (_document_key)
-- statement
CREATE TABLE proposed_fields (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              TEXT NOT NULL DEFAULT '{}',
    _document_key      TEXT
)
-- statement
CREATE INDEX proposed_fields_doc ON proposed_fields (_document_key)
-- statement
CREATE TABLE supplementary_fields (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              TEXT NOT NULL DEFAULT '{}',
    _document_key      TEXT
)
-- statement
CREATE INDEX supplementary_fields_doc ON supplementary_fields (_document_key)
-- statement
CREATE TABLE complementary_matches (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    model_class        TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              TEXT NOT NULL DEFAULT '{}',
    _document_key      TEXT
)
-- statement
CREATE INDEX complementary_matches_doc ON complementary_matches (_document_key)
-- statement
CREATE TABLE extraction_results (
    uid                TEXT PRIMARY KEY,
    tenant_id          TEXT,
    node_full_id       TEXT,
    model_class        TEXT,
    created_by_user_id TEXT,
    job_id             TEXT,
    props              TEXT NOT NULL DEFAULT '{}',
    _document_key      TEXT
)
-- statement
CREATE INDEX extraction_results_doc ON extraction_results (_document_key)
-- statement
CREATE INDEX extraction_results_node ON extraction_results (node_full_id)
-- statement
CREATE INDEX extraction_results_tenant_class ON extraction_results (tenant_id, model_class)
-- statement
CREATE VIRTUAL TABLE info_units_fts USING fts5(
    title, description, content='info_units', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
)
-- statement
CREATE TRIGGER info_units_ai AFTER INSERT ON info_units BEGIN
    INSERT INTO info_units_fts(rowid, title, description) VALUES (new.rowid, new.title, new.description);
END
-- statement
CREATE TRIGGER info_units_ad AFTER DELETE ON info_units BEGIN
    INSERT INTO info_units_fts(info_units_fts, rowid, title, description)
    VALUES ('delete', old.rowid, old.title, old.description);
END
-- statement
CREATE TRIGGER info_units_au AFTER UPDATE OF title, description ON info_units BEGIN
    INSERT INTO info_units_fts(info_units_fts, rowid, title, description)
    VALUES ('delete', old.rowid, old.title, old.description);
    INSERT INTO info_units_fts(rowid, title, description) VALUES (new.rowid, new.title, new.description);
END
-- statement
-- Deduplicated node tables
CREATE TABLE model_instances (
    uid                 TEXT PRIMARY KEY,
    tenant_id           TEXT,
    model_class         TEXT,
    created_by_user_ids TEXT,
    job_ids             TEXT,
    props               TEXT NOT NULL DEFAULT '{}'
)
-- statement
CREATE INDEX model_instances_tenant_class ON model_instances (tenant_id, model_class)
-- statement
CREATE TABLE labeled_entities (
    uid                 TEXT PRIMARY KEY,
    tenant_id           TEXT,
    label               TEXT,
    value               TEXT,
    normalized_value    TEXT,
    created_by_user_ids TEXT,
    job_ids             TEXT,
    props               TEXT NOT NULL DEFAULT '{}'
)
-- statement
CREATE INDEX labeled_entities_tenant_label ON labeled_entities (tenant_id, label)
-- statement
CREATE INDEX labeled_entities_tenant_value ON labeled_entities (tenant_id, normalized_value)
-- statement
CREATE TABLE entities (
    uid                 TEXT PRIMARY KEY,
    tenant_id           TEXT,
    value               TEXT,
    normalized_value    TEXT,
    created_by_user_ids TEXT,
    job_ids             TEXT,
    props               TEXT NOT NULL DEFAULT '{}'
)
-- statement
CREATE INDEX entities_tenant_value ON entities (tenant_id, normalized_value)
-- statement
-- Global catalog
CREATE TABLE catalog_models (name TEXT PRIMARY KEY, props TEXT NOT NULL DEFAULT '{}')
-- statement
CREATE TABLE model_fields (
    _key  TEXT PRIMARY KEY,
    name  TEXT NOT NULL,
    model TEXT NOT NULL,
    props TEXT NOT NULL DEFAULT '{}',
    UNIQUE (model, name)
)
-- statement
CREATE TABLE entity_labels (label TEXT PRIMARY KEY, props TEXT NOT NULL DEFAULT '{}')
-- statement
CREATE TABLE themes (path TEXT PRIMARY KEY, props TEXT NOT NULL DEFAULT '{}')
-- statement
-- Relationships
CREATE TABLE edges (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id TEXT NOT NULL,
    type      TEXT NOT NULL,
    src_label TEXT NOT NULL,
    src_key   TEXT NOT NULL,
    dst_label TEXT NOT NULL,
    dst_key   TEXT NOT NULL,
    _ident    TEXT NOT NULL DEFAULT '',
    props     TEXT NOT NULL DEFAULT '{}',
    UNIQUE (src_label, src_key, type, dst_label, dst_key, _ident)
)
-- statement
CREATE INDEX edges_dst ON edges (dst_label, dst_key, type)
-- statement
CREATE INDEX edges_tenant_type ON edges (tenant_id, type)
-- statement
CREATE TABLE catalog_edges (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    type      TEXT NOT NULL,
    src_label TEXT NOT NULL,
    src_key   TEXT NOT NULL,
    dst_label TEXT NOT NULL,
    dst_key   TEXT NOT NULL,
    _ident    TEXT NOT NULL DEFAULT '',
    props     TEXT NOT NULL DEFAULT '{}',
    UNIQUE (src_label, src_key, type, dst_label, dst_key, _ident)
)
-- statement
CREATE INDEX catalog_edges_dst ON catalog_edges (dst_label, dst_key, type)
-- statement
CREATE VIEW all_edges AS
    SELECT tenant_id, type, src_label, src_key, dst_label, dst_key, props FROM edges
    UNION ALL
    SELECT NULL, type, src_label, src_key, dst_label, dst_key, props FROM catalog_edges
-- statement
-- Storage
CREATE TABLE raw_files (
    id                 TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL,
    created_by_user_id TEXT,
    job_id             TEXT,
    filename           TEXT NOT NULL,
    folder_path        TEXT,
    content_type       TEXT NOT NULL,
    size_bytes         INTEGER NOT NULL,
    checksum_sha256    TEXT NOT NULL,
    stored_at          TEXT NOT NULL,
    props              TEXT NOT NULL DEFAULT '{}',
    _file_path         TEXT NOT NULL
)
-- statement
CREATE INDEX raw_files_tenant ON raw_files (tenant_id, filename, folder_path)
-- statement
CREATE TABLE pages (
    id                 TEXT PRIMARY KEY,
    raw_file_id        TEXT NOT NULL,
    tenant_id          TEXT NOT NULL,
    created_by_user_id TEXT,
    job_id             TEXT,
    filename           TEXT NOT NULL,
    folder_path        TEXT,
    page_index         INTEGER NOT NULL,
    markdown           TEXT NOT NULL,
    converted_at       TEXT NOT NULL,
    start_ms           INTEGER,
    end_ms             INTEGER,
    transcript         TEXT,
    props              TEXT NOT NULL DEFAULT '{}'
)
-- statement
CREATE UNIQUE INDEX pages_raw_file ON pages (raw_file_id, page_index)
-- statement
CREATE TABLE page_images (
    id                 TEXT PRIMARY KEY,
    raw_file_id        TEXT NOT NULL,
    tenant_id          TEXT NOT NULL,
    created_by_user_id TEXT,
    job_id             TEXT,
    page_index         INTEGER NOT NULL,
    image_index        INTEGER NOT NULL,
    kind               TEXT NOT NULL,
    content_type       TEXT NOT NULL,
    size_bytes         INTEGER NOT NULL,
    checksum_sha256    TEXT NOT NULL,
    width              INTEGER,
    height             INTEGER,
    time_ms            INTEGER,
    description        TEXT NOT NULL,
    filename           TEXT NOT NULL,
    folder_path        TEXT,
    stored_at          TEXT NOT NULL,
    props              TEXT NOT NULL DEFAULT '{}',
    _file_path         TEXT NOT NULL
)
-- statement
CREATE INDEX page_images_raw_file ON page_images (raw_file_id, page_index, image_index)
-- statement
CREATE UNIQUE INDEX page_images_identity ON page_images (tenant_id, raw_file_id, page_index, image_index)
-- statement
CREATE TABLE frozen_snapshots (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    mode         TEXT,
    frozen_at    TEXT,
    content_type TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    stored_at    TEXT NOT NULL,
    documents    TEXT NOT NULL DEFAULT '[]',
    props        TEXT NOT NULL DEFAULT '{}',
    _file_path   TEXT NOT NULL
)
-- statement
CREATE INDEX frozen_snapshots_tenant ON frozen_snapshots (tenant_id)
