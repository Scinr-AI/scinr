"""
Modelos de datos para el módulo de normalización post-extracción.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel


@dataclass(frozen=True)
class NormalizationSpec:
    """Especificación de un campo que requiere normalización."""

    field_name: str
    target_type: type[BaseModel]
    source_fields: list[str] | None  # None = usar todos los campos del modelo


@dataclass
class NormalizationEntry:
    """Entrada individual de normalización."""

    instance_id: int  # id() de la instancia Pydantic
    model_class_name: str
    field_name: str
    target_type: type[BaseModel]
    source_values: dict[str, object]
    unique_key: str  # "{target_type_name}:{md5_hash}"
    row_indices: list[int] = field(default_factory=list)  # row indices from pre-scan (pre-escaneo)
