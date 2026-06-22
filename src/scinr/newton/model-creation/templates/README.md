# Plantillas de Modelos de Extracción

Esta carpeta contiene plantillas listas para copiar como punto de partida para nuevos temas.

## Archivos disponibles

| Archivo | Uso |
|---------|-----|
| `models.py` | Plantilla para un archivo de modelos de extracción |
| `catalog.py` | Plantilla para el archivo `catalog.py` obligatorio de cada tema |

> **Nota:** No se necesita un archivo `registry.py`. El pipeline descubre todos los modelos
> automáticamente a través de `SELECTABLE_MODELS` en `catalog.py` más una expansión BFS de todos
> los sub-modelos referenciados transitivamente. Crear una nueva carpeta con `catalog.py` es
> suficiente para que el tema y todos sus modelos queden disponibles en el pipeline.

## Cómo usar las plantillas

1. Crea la carpeta del nuevo tema: `models/<mi_tema>/`
2. Copia los dos archivos de plantilla a esa carpeta
3. Renombra `models.py` al nombre descriptivo de los modelos (e.g., `clinical_data.py`)
4. Reemplaza todos los marcadores `<MI_TEMA>`, `<MiModelo>`, etc. por tus nombres reales
5. Sigue la [Guía Paso a Paso](../README.md#4-guía-paso-a-paso) del README principal

## Convenciones de nomenclatura a recordar

```
Carpeta:      models/<mi_tema>/           → snake_case
Archivo:      <nombre_descriptivo>.py     → snake_case
Clase:        class MiModelo              → PascalCase
Campo:        campo_de_modelo             → snake_case
entity_label: "MiEntidad"                → PascalCase
rel_type:     "MI_RELACION"              → UPPER_SNAKE_CASE
```

## Checklist mínimo por tema nuevo

- [ ] `__init__.py` (puede estar vacío)
- [ ] `<modelos>.py` con clases heredando de `ExtractionModel`
- [ ] `catalog.py` con `THEME_DESCRIPTION` y `SELECTABLE_MODELS`

Ver el [README principal](../README.md) para la guía completa y el checklist de integración.
