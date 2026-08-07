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

> Para instrucciones detalladas orientadas a agentes IA, consulta [`AGENTS.md`](../AGENTS.md).

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
instance_key: True (boolean)             → "instance_key": True
```

## Checklist mínimo por tema nuevo

- [ ] `__init__.py` (puede estar vacío)
- [ ] `<modelos>.py` con clases heredando de `ExtractionModel`
- [ ] `catalog.py` con `THEME_DESCRIPTION` y `SELECTABLE_MODELS`
- [ ] Si los modelos se referencian entre sí desde secciones distintas: `instance_key: True` en los campos clave del modelo destino
- [ ] `instance_relationships` declarado en los campos que enlazan con modelos en otros StructureNode
- [ ] Si el modelo se usará (o podría usarse) con el **pipeline tabular** (ingesta directa de CSV/XLSX): todos los campos mapeables son `str` o `list[str]` — nunca `int`, `float`, `date`, `Enum` ni submodelos anidados a ese nivel. **No** usar `@model_validator`/`@field_validator` para coerción de tipo — nunca se ejecutan en este pipeline (`model_construct()` los omite). Ver [`AGENTS.md` §4.7](../AGENTS.md#47-models-used-with-the-tabular-pipeline-fields-must-be-str-or-liststr) para el detalle y la alternativa real (`normalization_model` sobre un submodelo anidado).

Ver el [README principal](../README.md) para la guía completa y el checklist de integración.
