"""
Módulo de normalización post-extracción para el pipeline tabular.

Detecta campos Pydantic marcados con `normalization_model: True` en su
`json_schema_extra`, y llama a un LLM con structured output para rellenar
los modelos normalizados antes de escribir a Neo4j.
"""

from scinr.newton.tabular.normalization.detector import (
    extract_source_values,
    get_normalization_specs,
    instance_has_normalizable_fields,
)
from scinr.newton.tabular.normalization.engine import NormalizationEngine
from scinr.newton.tabular.normalization.hook import run_normalization_hook
from scinr.newton.tabular.normalization.models import (
    NormalizationEntry,
    NormalizationSpec,
)

__all__ = [
    "NormalizationEngine",
    "NormalizationSpec",
    "NormalizationEntry",
    "get_normalization_specs",
    "instance_has_normalizable_fields",
    "extract_source_values",
    "run_normalization_hook",
]
