# Work in Progress (wip/)

Esta carpeta es el **sandbox de desarrollo** para modelos de extracción que aún están siendo diseñados, prototipados o validados antes de su integración en el pipeline de producción.

> ⚠️ **El pipeline de producción NO usa esta carpeta.** El `ThemeRegistry` escanea `models/`, no `wip/`. Los modelos aquí son invisibles para el agente de anotación hasta que se muevan a `models/`.

---

## Estructura de una carpeta WIP

Cada tema en desarrollo tiene su propia subcarpeta:

```
wip/
└── <nombre_del_tema>/         ← Nombre descriptivo del tema en desarrollo
    ├── draft_models.py        ← Clases Pydantic en borrador (principal deliverable)
    ├── notes.md               ← Notas de diseño, decisiones, preguntas abiertas
    ├── READY.md               ← Checklist de "listo para integración" (crear cuando esté cerca)
    └── test_samples/          ← (Opcional) Fragmentos de documentos reales para validar
        ├── sample_01.txt
        └── sample_02.txt
```

---

## Flujo de desarrollo recomendado

```
1. PROTOTIPO (wip/<tema>/)
   ├── Crear draft_models.py con las clases iniciales
   ├── Escribir notes.md con decisiones de diseño
   └── Iterar sobre campos y descriptions

        ↓

2. VALIDACIÓN (wip/<tema>/)
   ├── Extraer contra fragmentos de documentos reales (test_samples/)
   ├── Verificar que el LLM llena los campos correctamente
   ├── Ajustar descriptions para mejorar extracción
   └── Revisar entity_labels y field_relationships

        ↓

3. REVISIÓN (wip/<tema>/)
   ├── Crear READY.md y completar el checklist
   ├── Revisión por pares (otro desarrollador o agente)
   └── Confirmar que está listo para integración

        ↓

4. INTEGRACIÓN (models/<tema>/)
   ├── Copiar/mover a models/<tema>/
   ├── Crear catalog.py
   ├── Verificar descubrimiento automático
   ├── Completar el checklist de integración del README principal
   └── Eliminar la carpeta de wip/ una vez integrado
```

---

## ¿Qué significa "listo para integración"?

Un modelo está listo para moverse de `wip/` a `models/` cuando cumple **todos** estos criterios:

### Criterios técnicos
- [ ] Todas las clases heredan de `ExtractionModel`
- [ ] Todos los campos tienen `description` detallada (mínimo 15 palabras, con ejemplos)
- [ ] Todos los campos opcionales tienen `default=None` explícito
- [ ] Todos los campos lista tienen `default_factory=list`
- [ ] Docstring de primera línea en cada clase (≤ 15 palabras, en inglés)
- [ ] Sin errores de importación (`python -c "from wip.<tema>.draft_models import *"`)
- [ ] Sin campos con `default=[]` (usar `default_factory=list` siempre)

### Criterios de calidad de extracción
- [ ] Probado contra al menos 3 fragmentos de documentos reales
- [ ] El LLM llena ≥ 70% de los campos con datos correctos para documentos relevantes
- [ ] La `description` de cada campo produce extracciones consistentes entre ejecuciones
- [ ] Los `entity_label` identifican entidades que realmente aparecen en los documentos

### Criterios de integración
- [ ] `THEME_DESCRIPTION` es específica y diferenciable de los temas existentes
- [ ] Revisada la [tabla de Temas Existentes](../README.md#9-temas-existentes) para confirmar que no duplica nada
- [ ] Decidida la jerarquía: ¿tema nuevo o sub-tema?

---

## Notas para agentes de IA que desarrollan modelos

Si eres un agente de IA desarrollando modelos en esta carpeta:

1. **Empieza siempre en `wip/`**, nunca directamente en `models/`
2. **Crea `notes.md`** con tu razonamiento de diseño — futuras iteraciones necesitarán ese contexto
3. **Incluye ejemplos reales** en las `description` de los campos (cómo aparecería el dato en un documento real)
4. **Nunca uses placeholders** en las descriptions (`"Descripción del campo"`) — escribe la descripción real
5. **Prueba las descriptions** preguntándote: "¿Podría un LLM encontrar este dato en un documento regulatorio real con esta guía?"
6. **Cuando termines**, crea `READY.md` con el checklist completado y notifica al humano para revisión

---

## Ejemplo de `notes.md`

```markdown
# Notas de diseño: <nombre_del_tema>

## Motivación
¿Por qué se necesita este tema? ¿Qué tipos de documentos cubre?

## Decisiones de diseño
- Campo X: decidí usar list[str] en lugar de str porque los documentos suelen listar múltiples valores
- Modelo Y: separé en sub-modelo para agrupar datos relacionados de la sección Z
- entity_label "Facility": los fabricantes aparecen en múltiples documentos del mismo dossier

## Preguntas abiertas
- ¿Los estudios clínicos deben ser un sub-tema separado?
- ¿El campo 'xxx' debería tener entity_label?

## Documentos de referencia usados para diseño
- Tipo de documento 1: [descripción]
- Tipo de documento 2: [descripción]

## Estado
- [ ] Prototipo inicial
- [ ] Validación contra documentos reales
- [ ] Revisión
- [ ] Listo para integración
```
