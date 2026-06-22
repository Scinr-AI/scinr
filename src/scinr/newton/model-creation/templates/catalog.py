"""
Plantilla de catalog.py para un nuevo tema de extracción.

INSTRUCCIONES:
  1. Copia este archivo a <mi_paquete>/<mi_tema>/catalog.py
  2. Ajusta las importaciones para tus clases reales
  3. Escribe THEME_DESCRIPTION específica para tu dominio
  4. Añade todos los modelos seleccionables a SELECTABLE_MODELS

REGLAS DE CATALOG.PY:
  - OBLIGATORIO: solo los modelos en SELECTABLE_MODELS son visibles para el agente
    de anotación. Si un sub-modelo no está aquí, el agente no puede seleccionarlo
    directamente como modelo primario.
  - OPCIONAL: los sub-modelos que solo se usan como campos embebidos de otros modelos
    no necesitan estar en SELECTABLE_MODELS.
  - El ThemeRegistry importa este módulo en tiempo de ejecución. Errores de sintaxis
    o importaciones rotas aquí hacen que el tema sea ignorado silenciosamente.

REGLA DE ORO — IMPORTS RELATIVOS:
  Usa SIEMPRE imports relativos para módulos dentro de tu propio paquete.
  El punto inicial indica "busca en el mismo paquete donde estoy yo".

  ✅ CORRECTO:
      from .models import MainModelExample, SubModelExample
      from .second_file import AnotherModel

  ❌ INCORRECTO — import bare (frágil, depende del sys.path):
      from models import MainModelExample

  ❌ INCORRECTO — ruta absoluta del paquete propio:
      from my_package.my_theme.models import MainModelExample

CÓMO ESCRIBIR THEME_DESCRIPTION:
  - Una a tres oraciones, técnicas y específicas
  - Menciona los tipos de documentos que cubre
  - Menciona los estándares regulatorios si aplica
  - Hazla diferenciable de los otros temas existentes:
      ✓ "Regulatory quality documentation for medicinal products (ICH CTD Module 3 / CMC)..."
      ✗ "Documentos técnicos de calidad"  ← Demasiado vago
  - El LLM la leerá para decidir si un nodo pertenece a este tema
"""
from __future__ import annotations

# ─── Importaciones ───────────────────────────────────────────────────────────
# Importa todos los modelos que deben ser SELECCIONABLES por el agente.
# Usa imports RELATIVOS (con punto inicial) para referenciar módulos hermanos.
#
# Ejemplo — un solo archivo de modelos:
# from .models import (
#     MainModelExample,
#     AnotherModel,
# )
#
# Ejemplo — múltiples archivos de modelos:
# from .identity import IdentityModel
# from .manufacturing import ManufacturingModel
# from .stability import StabilityModel

# TODO: reemplazar con tus importaciones reales
# from .models import (
#     MainModelExample,
#     SubModelExample,
# )


# ─── Descripción del tema ─────────────────────────────────────────────────────
# Esta cadena es lo que el LLM lee para clasificar si un nodo pertenece a este tema.
# Es CRÍTICA para la calidad de la anotación.

THEME_DESCRIPTION: str = (
    # TODO: reemplazar con la descripción real de tu tema
    # Debe ser específica, técnica y diferenciable de los otros temas.
    # Ejemplos de cómo NO hacerlo:
    #   ✗ "Documentos técnicos"
    #   ✗ "Informes científicos de calidad"
    # Ejemplos de cómo SÍ hacerlo:
    #   ✓ "Regulatory quality documentation for medicinal products (ICH CTD Module 3 / CMC), "
    #     "covering drug substance and drug product chemistry, manufacturing, controls, "
    #     "characterisation, stability, biological safety, facilities, and regional quality requirements"
    "TODO: replace with a specific, technical, one-to-three sentence description of this theme. "
    "Mention the document types it covers, the regulatory standards involved (if any), "
    "and what distinguishes it from other themes in the system."
)

# ─── Modelos seleccionables ───────────────────────────────────────────────────
# Lista completa de modelos que el agente de anotación puede seleccionar.
#
# ORDEN RECOMENDADO:
#   1. Modelos de alto nivel / más genéricos (los que suelen ganar)
#   2. Modelos de sub-secciones específicas
#   3. Sub-modelos que pueden ser el objetivo primario de extracción
#
# NO incluir aquí sub-modelos que SOLO se usan como campos embebidos en otros
# modelos y nunca serían el modelo "principal" de un nodo.

SELECTABLE_MODELS: list[type] = [
    # TODO: reemplazar con tus modelos reales
    # Ejemplo:
    # MainModelExample,      # Modelo de alto nivel (más frecuentemente seleccionado)
    # SubModelExample,       # Sub-modelo que puede ser objetivo primario directo
]
