"""
Hook de normalización para integrar en el pipeline tabular.

Se invoca entre la instanciación del modelo Pydantic y la escritura a Neo4j.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from scinr.newton.tabular.normalization.engine import NormalizationEngine

logger = logging.getLogger(__name__)


async def run_normalization_hook(
    instances: list[tuple[type[BaseModel], BaseModel]],
    engine: NormalizationEngine | None = None,
) -> list[tuple[type[BaseModel], BaseModel]]:
    """
    Hook a insertar en el pipeline tabular entre instanciación y escritura.

    Si engine es None, retorna las instancias sin modificar (no-op).

    Args:
        instances: Lista de (model_class, instance) recién instanciadas.
        engine: Motor de normalización configurado.

    Returns:
        Lista de instancias con campos normalizados rellenados.
    """
    if engine is None:
        return instances

    try:
        return await engine.normalize_instances(instances)
    except Exception as e:
        logger.error("Normalization hook failed: %s", e, exc_info=True)
        # Best-effort: retornar instancias sin normalizar
        return instances
