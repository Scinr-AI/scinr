"""
sql/types.py — Column types shared by the tables of both engines.

* :class:`UtcDateTime`: ``timestamptz`` in Postgres, ISO 8601 text in SQLite; always a
  timezone-aware UTC ``datetime`` in Python.
* :data:`JsonObject`: ``jsonb`` / JSON text. :data:`JsonList`: ``text[]`` / JSON text.
  ``None`` is stored as SQL ``NULL`` (a null property does not exist), never as JSON ``null``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

__all__ = ["JsonList", "JsonObject", "UtcDateTime"]


class UtcDateTime(sa.types.TypeDecorator):
    """A timezone-aware UTC ``datetime`` (naive values are taken as UTC)."""

    impl = sa.Text
    cache_ok = True

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(sa.DateTime(timezone=True))
        return dialect.type_descriptor(sa.Text())

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        value = self._utc(value)
        return value if dialect.name == "postgresql" else value.isoformat()

    def process_result_value(self, value: Any, dialect: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        return self._utc(value)


JsonObject = sa.JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")
JsonList = sa.JSON(none_as_null=True).with_variant(ARRAY(sa.Text()), "postgresql")
