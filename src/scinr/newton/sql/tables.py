"""
sql/tables.py — SQLAlchemy Core definitions of the tables (for building queries only).

The DDL is the one in ``migrations/<dialect>/0001_init.sql``; ``metadata.create_all()`` must
never be called. The tables live in a :class:`~sqlalchemy.MetaData` **without schema**: the
Postgres schema is applied by the engines through ``schema_translate_map``.

* Graph tables, ``edges`` / ``catalog_edges``, the ``all_edges`` view and ``pages`` are the
  same in both engines and are module attributes.
* The tables that hold binaries differ: Postgres keeps ``_chunk_count`` and has ``*_chunks``
  tables, SQLite keeps ``_file_path`` and has no chunk tables. Use :func:`storage_tables`.
* The Postgres generated columns ``info_units._fts_*`` are deliberately not declared (nothing
  writes them or selects them); the dialect refers to them by name.
"""

from __future__ import annotations

from typing import NamedTuple

import sqlalchemy as sa

from scinr.newton.sql.names import (
    CATALOG_TABLES,
    DEDUP_TABLES,
    PER_DOCUMENT_TABLES,
)
from scinr.newton.sql.types import JsonList, JsonObject, UtcDateTime

__all__ = [
    "LIST_COLUMNS",
    "NODE_TABLES",
    "TYPED_COLUMNS",
    "StorageTables",
    "all_edges",
    "catalog_edges",
    "edges",
    "metadata",
    "pages",
    "storage_tables",
    "tables_for",
]

metadata = sa.MetaData()


def _col(name: str, type_: object = sa.Text, **kwargs: object) -> sa.Column:
    return sa.Column(name, type_, **kwargs)  # type: ignore[arg-type]


def _props() -> sa.Column:
    return _col("props", JsonObject, nullable=False, server_default=sa.text("'{}'"))


def _table(name: str, *columns: sa.Column) -> sa.Table:
    return sa.Table(name, metadata, *columns)


def _provenance() -> list[sa.Column]:
    return [_col("created_by_user_id"), _col("job_id")]


def _per_document(name: str, key: str, *typed: sa.Column, extra: list[sa.Column] | None = None):
    return _table(
        name,
        _col(key, primary_key=True),
        _col("tenant_id"),
        *typed,
        *_provenance(),
        _props(),
        _col("_document_key"),
        *(extra or []),
    )


# ── Per-document nodes ───────────────────────────────────────────────────────

documents = _table(
    "documents",
    _col("_key", primary_key=True),
    _col("tenant_id", nullable=False),
    _col("path", nullable=False),
    _col("version", sa.BigInteger, nullable=False),
    _col("name"),
    _col("latest", sa.Boolean),
    _col("is_folder", sa.Boolean),
    _col("raw_file_id"),
    *_provenance(),
    _col("frozen", sa.Boolean),
    _col("frozen_blob_id"),
    _col("deletion_pending", sa.Boolean),
    _props(),
    sa.UniqueConstraint("tenant_id", "path", "version"),
)
structure_nodes = _table(
    "structure_nodes",
    _col("id", primary_key=True),
    _col("tenant_id"),
    _col("role"),
    _col("node_id"),
    _col("title"),
    _col("theme"),
    _col("appearance_order", sa.BigInteger),
    _col("row_index", sa.BigInteger),
    _col("source_page_ids", JsonList),
    _col("source_paragraphs", JsonList),
    _col("paragraph_segmenter_version", sa.BigInteger),
    _col("info_units_mode"),
    *_provenance(),
    _props(),
    _col("_document_key"),
)
info_units = _per_document(
    "info_units",
    "uid",
    _col("info_unit_id"),
    _col("title"),
    _col("description"),
    _col("order", sa.BigInteger),
    extra=[_col("_structure_node_id")],
)
model_decisions = _per_document(
    "model_decisions", "uid", _col("matched_model_class"), _col("source")
)
proposed_models = _per_document("proposed_models", "uid")
proposed_fields = _per_document("proposed_fields", "uid")
supplementary_fields = _per_document("supplementary_fields", "uid")
complementary_matches = _per_document("complementary_matches", "uid", _col("model_class"))
extraction_results = _per_document(
    "extraction_results", "uid", _col("node_full_id"), _col("model_class")
)

# ── Deduplicated nodes ───────────────────────────────────────────────────────


def _dedup(name: str, *typed: sa.Column) -> sa.Table:
    return _table(
        name,
        _col("uid", primary_key=True),
        _col("tenant_id"),
        *typed,
        _col("created_by_user_ids", JsonList),
        _col("job_ids", JsonList),
        _props(),
    )


model_instances = _dedup("model_instances", _col("model_class"))
labeled_entities = _dedup(
    "labeled_entities", _col("label"), _col("value"), _col("normalized_value")
)
entities = _dedup("entities", _col("value"), _col("normalized_value"))

# ── Global catalog ───────────────────────────────────────────────────────────

catalog_models = _table("catalog_models", _col("name", primary_key=True), _props())
model_fields = _table(
    "model_fields",
    _col("_key", primary_key=True),
    _col("name", nullable=False),
    _col("model", nullable=False),
    _props(),
    sa.UniqueConstraint("model", "name"),
)
entity_labels = _table("entity_labels", _col("label", primary_key=True), _props())
themes = _table("themes", _col("path", primary_key=True), _props())

# ── Relationships ────────────────────────────────────────────────────────────


def _edge_columns() -> list[sa.Column]:
    return [
        _col("id", sa.BigInteger().with_variant(sa.Integer, "sqlite"), primary_key=True),
        _col("type", nullable=False),
        _col("src_label", nullable=False),
        _col("src_key", nullable=False),
        _col("dst_label", nullable=False),
        _col("dst_key", nullable=False),
        _col("_ident", nullable=False, server_default=sa.text("''")),
        _props(),
    ]


def _edge_table(name: str, *leading: sa.Column) -> sa.Table:
    columns = _edge_columns()
    # tenant_id sits right after id in the DDL.
    return _table(
        name,
        columns[0],
        *leading,
        *columns[1:],
        sa.UniqueConstraint("src_label", "src_key", "type", "dst_label", "dst_key", "_ident"),
    )


edges = _edge_table("edges", _col("tenant_id", nullable=False))
catalog_edges = _edge_table("catalog_edges")

#: Read-only union of both relationship tables (``tenant_id`` is ``NULL`` for catalog edges).
all_edges = sa.Table(
    "all_edges",
    metadata,
    _col("tenant_id"),
    _col("type"),
    _col("src_label"),
    _col("src_key"),
    _col("dst_label"),
    _col("dst_key"),
    _col("props", JsonObject),
    info={"is_view": True},
)

# ── Pages (same in both engines) ─────────────────────────────────────────────

pages = _table(
    "pages",
    _col("id", primary_key=True),
    _col("raw_file_id", nullable=False),
    _col("tenant_id", nullable=False),
    *_provenance(),
    _col("filename", nullable=False),
    _col("folder_path"),
    _col("page_index", sa.Integer, nullable=False),
    _col("markdown", nullable=False),
    _col("converted_at", UtcDateTime, nullable=False),
    _col("start_ms", sa.BigInteger),
    _col("end_ms", sa.BigInteger),
    _col("transcript", JsonObject),
    _props(),
    sa.UniqueConstraint("raw_file_id", "page_index"),
)

# ── Binary storage: one definition per engine ────────────────────────────────


class StorageTables(NamedTuple):
    """The tables with binary content. The chunk tables are ``None`` where they do not exist."""

    raw_files: sa.Table
    raw_file_chunks: sa.Table | None
    page_images: sa.Table
    page_image_chunks: sa.Table | None
    frozen_snapshots: sa.Table
    frozen_snapshot_chunks: sa.Table | None


def _storage_location(variant: str) -> sa.Column:
    if variant == "postgres":
        return _col("_chunk_count", sa.Integer, nullable=False, server_default=sa.text("0"))
    return _col("_file_path", nullable=False)


def _chunk_table(meta: sa.MetaData, name: str, owner_column: str) -> sa.Table:
    return sa.Table(
        name,
        meta,
        sa.Column(owner_column, sa.Text, primary_key=True),
        sa.Column("n", sa.Integer, primary_key=True),
        sa.Column("tenant_id", sa.Text, nullable=False),
        sa.Column("data", sa.LargeBinary, nullable=False),
    )


def _build_storage(variant: str) -> StorageTables:
    meta = sa.MetaData()

    def table(name: str, *columns: sa.Column) -> sa.Table:
        return sa.Table(name, meta, *columns)

    raw_files = table(
        "raw_files",
        _col("id", primary_key=True),
        _col("tenant_id", nullable=False),
        *_provenance(),
        _col("filename", nullable=False),
        _col("folder_path"),
        _col("content_type", nullable=False),
        _col("size_bytes", sa.BigInteger, nullable=False),
        _col("checksum_sha256", nullable=False),
        _col("stored_at", UtcDateTime, nullable=False),
        _props(),
        _storage_location(variant),
    )
    page_images = table(
        "page_images",
        _col("id", primary_key=True),
        _col("raw_file_id", nullable=False),
        _col("tenant_id", nullable=False),
        *_provenance(),
        _col("page_index", sa.Integer, nullable=False),
        _col("image_index", sa.Integer, nullable=False),
        _col("kind", nullable=False),
        _col("content_type", nullable=False),
        _col("size_bytes", sa.BigInteger, nullable=False),
        _col("checksum_sha256", nullable=False),
        _col("width", sa.Integer),
        _col("height", sa.Integer),
        _col("time_ms", sa.BigInteger),
        _col("description", nullable=False),
        _col("filename", nullable=False),
        _col("folder_path"),
        _col("stored_at", UtcDateTime, nullable=False),
        _props(),
        _storage_location(variant),
        # A duplicate image identity is a bug: it must fail, not overwrite (as in MongoDB).
        sa.UniqueConstraint("tenant_id", "raw_file_id", "page_index", "image_index"),
    )
    frozen_snapshots = table(
        "frozen_snapshots",
        _col("id", primary_key=True),
        _col("tenant_id", nullable=False),
        _col("mode"),
        _col("frozen_at"),
        _col("content_type", nullable=False),
        _col("size_bytes", sa.BigInteger, nullable=False),
        _col("stored_at", UtcDateTime, nullable=False),
        _col("documents", JsonObject, nullable=False, server_default=sa.text("'[]'")),
        _props(),
        _storage_location(variant),
    )
    if variant != "postgres":
        return StorageTables(raw_files, None, page_images, None, frozen_snapshots, None)
    return StorageTables(
        raw_files,
        _chunk_table(meta, "raw_file_chunks", "file_id"),
        page_images,
        _chunk_table(meta, "page_image_chunks", "image_id"),
        frozen_snapshots,
        _chunk_table(meta, "frozen_snapshot_chunks", "snapshot_id"),
    )


_STORAGE: dict[str, StorageTables] = {
    "postgres": _build_storage("postgres"),
    "sqlite": _build_storage("sqlite"),
}


def storage_tables(dialect_name: str) -> StorageTables:
    """The binary-holding tables of an engine (``dialect.name``)."""
    return _STORAGE[dialect_name]


def tables_for(dialect_name: str) -> dict[str, sa.Table]:
    """Every table of an engine by name, the ``all_edges`` view included."""
    found = {t.name: t for t in metadata.tables.values()}
    found.update({t.name: t for t in storage_tables(dialect_name) if t is not None})
    return found


# ── Derived column sets ──────────────────────────────────────────────────────

NODE_TABLES: tuple[str, ...] = (*PER_DOCUMENT_TABLES, *DEDUP_TABLES, *CATALOG_TABLES)

#: For every node table: the columns that are properties (not ``props``, not ``_auxiliary``).
TYPED_COLUMNS: dict[str, frozenset[str]] = {
    name: frozenset(
        c.name for c in metadata.tables[name].columns if c.name != "props" and c.name[0] != "_"
    )
    for name in NODE_TABLES
}

#: For every node table: the typed columns that hold a list of texts.
LIST_COLUMNS: dict[str, frozenset[str]] = {
    "structure_nodes": frozenset({"source_page_ids", "source_paragraphs"}),
    **{name: frozenset({"created_by_user_ids", "job_ids"}) for name in DEDUP_TABLES},
}
