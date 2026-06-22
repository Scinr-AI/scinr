"""
Plantilla de modelos de extracción para un nuevo tema.

INSTRUCCIONES:
  1. Copia este archivo a <mi_paquete>/<mi_tema>/<nombre_descriptivo>.py
  2. Renombra las clases (busca/reemplaza):
       SubModelExample   → <TuSubModelo>
       MainModelExample  → <TuModeloPrincipal>
       ExampleEnum       → <TuEnum>
  3. Renombra los valores del enum y los campos según tu dominio
  4. Ajusta TODAS las descriptions — son críticas para la calidad de extracción
  5. Elimina los comentarios de instrucciones cuando el modelo esté listo

CONVENCIONES:
  - Clases en PascalCase: MiModelo
  - Campos en snake_case: mi_campo_aqui
  - Docstring primera línea ≤ 15 palabras, en inglés
  - Todos los campos opcionales: default=None explícito
  - Todos los campos lista: default_factory=list

REGLA DE ORO — IMPORTS RELATIVOS:
  Cuando este archivo comparte paquete con otros módulos propios (base.py, utils.py, etc.),
  usa imports RELATIVOS. El punto inicial significa "mismo directorio que este archivo".

  ✅ CORRECTO — módulo hermano (mismo directorio):
      from .base import MyBase

  ✅ CORRECTO — módulo en el directorio padre:
      from ..shared import SharedHelper

  ❌ INCORRECTO — bare import (frágil, rompe si el directorio no está en sys.path):
      from base import MyBase

  La única excepción es scinr_ingest: es un paquete instalado, por lo que
  `from scinr_ingest.models.base import ExtractionModel` es siempre correcto.
"""
from __future__ import annotations

# Importa Enum solo si necesitas valores controlados
from enum import Enum

# Importa Literal y Annotated solo si usas discriminadores de tipo
# from typing import Literal, Annotated
from pydantic import Field

# ✅ CORRECTO: scinr_ingest es un paquete instalado — import absoluto es correcto aquí.
# Si defines tu propia clase base en un archivo base.py dentro del mismo directorio,
# usa en su lugar: from .base import ExtractionModel
from scinr_ingest.models.base import ExtractionModel

# ─────────────────────────────────────────────────────────────────────────────
# ENUMS — Define valores controlados para campos con opciones fijas
# RENOMBRAR: ExampleEnum → <TuEnum>
# Elimina esta sección si no necesitas enums.
# ─────────────────────────────────────────────────────────────────────────────

class ExampleEnum(str, Enum):
    """
    Descripción del enum: qué tipo de valores controla.
    Usar str como base garantiza serialización JSON directa.

    El LLM leerá el schema JSON generado, por lo que el docstring debe explicar
    cuándo usar cada valor.

    Valores:
      value_a — Cuando el documento indica X o usa el término Y.
      value_b — Cuando el documento indica Z o usa el término W.
      value_c — Para casos donde aplica la condición Q.
    """
    # RENOMBRAR los valores según tu dominio
    VALUE_A = "value_a"
    VALUE_B = "value_b"
    VALUE_C = "value_c"


# ─────────────────────────────────────────────────────────────────────────────
# SUB-MODELOS — De menor a mayor complejidad
# RENOMBRAR: SubModelExample → <TuSubModelo>
#
# Crea sub-modelos para agrupar campos relacionados que representan un concepto
# cohesivo. Son apropiados cuando tienes 3+ campos de la misma "cosa".
# ─────────────────────────────────────────────────────────────────────────────

class SubModelExample(ExtractionModel):
    """
    Primera línea concisa en inglés para el catálogo (≤ 15 palabras).

    Segunda línea opcional con más contexto o referencia normativa.
    """

    # ── Campo de texto opcional — El patrón más común ─────────────────────────
    text_field: str | None = Field(
        default=None,
        description=(
            # Una buena descripción responde:
            # 1. ¿QUÉ es este dato exactamente?
            # 2. ¿En qué FORMATO se espera (e.g. 'C22H30N2O', '310.46 g/mol')?
            # 3. ¿CUÁNDO usar None?
            "Description of this data point. "
            "Expected format with concrete examples (e.g. 'example 1', 'example 2'). "
            "None if not mentioned in the document."
        ),
    )

    # ── Campo lista de strings — Para múltiples valores del mismo tipo ─────────
    list_field: list[str] = Field(
        default_factory=list,   # SIEMPRE usar default_factory para listas, nunca default=[]
        description=(
            "List of items. Each element as a concise string. "
            "Include all items mentioned in the section. "
            "Empty list if no items are present."
        ),
    )

    # ── Campo con enum — Para valores controlados ─────────────────────────────
    enum_field: ExampleEnum | None = Field(
        default=None,
        description=(
            "Type according to ExampleEnum: 'value_a' for X, 'value_b' for Y, 'value_c' for Z. "
            "None if not determinable from the text."
        ),
    )

    # ── Campo numérico como string — Preserva unidades y formato original ─────
    value_with_units: str | None = Field(
        default=None,
        description=(
            "Numeric value with its units exactly as it appears in the document "
            "(e.g. '310.46 g/mol', '25°C/60% RH', '≥ 99.0%'). "
            "Preserve original format including units. "
            "None if not mentioned."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS CON DISCRIMINADOR — Para polimorfismo de tipo (descomenta si lo necesitas)
# Usa Literal["valor"] como campo discriminador cuando tienes dos estructuras
# incompatibles para el mismo concepto.
# ─────────────────────────────────────────────────────────────────────────────

# from typing import Literal, Annotated
#
# class VariantA(ExtractionModel):
#     """Variant A of concept X. Discriminator value: 'variant_a'."""
#
#     variant: Literal["variant_a"] = Field(
#         description=(
#             "Type discriminator. Must be exactly 'variant_a' for this subtype. "
#             "Use 'variant_b' for the other variant."
#         ),
#     )
#     specific_field_a: str | None = Field(default=None, description="...")
#
#
# class VariantB(ExtractionModel):
#     """Variant B of concept X. Discriminator value: 'variant_b'."""
#
#     variant: Literal["variant_b"] = Field(
#         description=(
#             "Type discriminator. Must be exactly 'variant_b' for this subtype. "
#             "Use 'variant_a' for the other variant."
#         ),
#     )
#     specific_field_b: str | None = Field(default=None, description="...")


# ─────────────────────────────────────────────────────────────────────────────
# MODELO PRINCIPAL — Representa una sección completa del documento
# RENOMBRAR: MainModelExample → <TuModeloPrincipal>
# ─────────────────────────────────────────────────────────────────────────────

class MainModelExample(ExtractionModel):
    """
    Primera línea concisa en inglés para el catálogo (≤ 15 palabras).

    Descripción más completa del modelo: qué captura, qué sección del documento
    representa, y cualquier referencia normativa relevante (e.g. CTD 3.2.S.1).
    """

    # ── Entidad nombrada — Campo que genera un nodo Neo4j ────────────────────
    # Usa entity_label cuando el valor es una entidad del mundo real que podría
    # aparecer en múltiples documentos y merece ser un nodo propio en el grafo.
    # Valores comunes: "Substance", "Facility", "CASNumber", "SubstanceINN",
    # "Manufacturer", "Site", "Product"
    primary_entity_name: str | None = Field(
        default=None,
        description=(
            "Name of the primary entity as it appears in the document. "
            "Extract the official or working name without modifications. "
            "None if not explicitly mentioned."
        ),
        json_schema_extra={"entity_label": "EntityLabel"},  # RENOMBRAR: EntityLabel → <TuLabel>
    )

    # ── Entidad con relación — Campo que genera un nodo Y una arista en Neo4j ─
    # Combina entity_label con field_relationships cuando quieres una arista entre
    # dos entidades del mismo modelo. El campo destino TAMBIÉN debe tener entity_label.
    related_entity: str | None = Field(
        default=None,
        description=(
            "Name of the related entity (e.g. manufacturer, testing site). "
            "Extract the full name as it appears in the document. "
            "None if not mentioned."
        ),
        json_schema_extra={
            "entity_label": "OtherLabel",              # RENOMBRAR: OtherLabel → <OtroLabel>
            "field_relationships": [
                {
                    "to_field": "primary_entity_name", # Campo destino (debe tener entity_label)
                    "rel_type": "RELATED_TO",           # RENOMBRAR a tipo específico: UPPER_SNAKE_CASE
                    # Ejemplos: "MANUFACTURED_BY", "LOCATED_IN", "TESTED_BY", "CONTAINS"
                }
            ]
        },
    )

    # ── Campo lista de strings sin entity_label ──────────────────────────────
    methods_list: list[str] = Field(
        default_factory=list,
        description=(
            "List of methods/procedures mentioned. "
            "Each method as a concise string. "
            "Empty list if none are mentioned."
        ),
    )

    # ── Sub-modelo embebido opcional ──────────────────────────────────────────
    detail_section: SubModelExample | None = Field(
        default=None,
        description=(
            "Detailed information from subsection X. "
            "None if this subsection is not present in the node."
        ),
    )

    # ── Lista de sub-modelos — Para secciones con múltiples instancias ────────
    items: list[SubModelExample] = Field(
        default_factory=list,
        description=(
            "List of detailed items. Each item represents one individual instance "
            "of X found in the document. "
            "Empty list if no items are found."
        ),
    )

    # ── Campo de tipo polimórfico (si usas el bloque de discriminador arriba) ─
    # from typing import Annotated
    # structure: Annotated[
    #     VariantA | VariantB,
    #     Field(discriminator="variant")
    # ] = Field(
    #     description=(
    #         "Use VariantA (variant='variant_a') for X. "
    #         "Use VariantB (variant='variant_b') for Y."
    #     ),
    # )


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS ADICIONALES — Añade más clases siguiendo el mismo patrón
# ─────────────────────────────────────────────────────────────────────────────

# class AnotherMainModel(ExtractionModel):
#     """Another top-level model for a different document section type."""
#     ...
