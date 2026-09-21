# Plan de implementación — Optimización de memoria de scinr

> **Para el agente que ejecute este plan.** Este documento es autosuficiente: recoge el
> diagnóstico verificado, las decisiones ya tomadas con el responsable del proyecto, los cambios
> exactos por fichero, los tests a añadir/actualizar y los criterios de aceptación medibles.
> No hace falta releer la conversación de origen. Referencias `fichero:línea` medidas sobre
> `main` @ `88536b2` (versión `0.3.9`); si han derivado, localiza por nombre de función.

## 0. Objetivo y alcance

**Problema.** Un mismo archivo se materializa varias veces en memoria y, a medida que avanza el
pipeline, la memoria crece cuando solo debería haber picos donde son estrictamente necesarios.
Caso de referencia: un PDF de 5 GB (pico estimado hoy: decenas de GB, sin contar MDM, que aporta
su propia copia) y CSV de gran tamaño.

**Objetivo.** Que el pico de memoria de cada etapa dependa del tamaño de **una unidad de trabajo**
(un chunk de PDF ≤ 45 MB, un lote de 500 filas, un nodo en vuelo por slot de LLM) y no del tamaño
del archivo ni del número de nodos del documento.

**Fuera de alcance (decisiones ya tomadas — no reabrir):**

| Ítem | Decisión | Motivo |
|---|---|---|
| Conversor PowerPoint / imágenes base64 de `PageImage` | **Ignorar** | PPTX no está soportado por ahora. Mistral OCR no devuelve imágenes (no se pide `include_image_base64`). |
| Memoizar `compose_extraction_schema` / `build_annotation_decision_prompt` | **No hacer** | Con el semáforo primero (WP2) solo habrá ≤ `llm_concurrency` schemas vivos. Añade riesgo de caché obsoleta (`prompt_family`, `full_docstring`, recarga de modelos) y claves no hashables (dicts). |
| Pool de N workers para Stages 3/4 | **No hacer** | El semáforo LLM primero da el mismo resultado (+15 MB vs +4 MB) con mucho menos código. |
| `aclose_resources()` / cerrar `AsyncDriver` y cliente Motor en `reset_*` | **No hacer** | MDM lanza **un proceso por job**; el SO recupera todo al terminar. |
| Semáforos globales ligados al primer event loop | **No hacer** | Solo falla si un mismo proceso hace varios `asyncio.run` con contención; no es el caso de MDM. |
| `malloc_trim`, `gc.collect()` por unidad, subprocesos por PDF | **No hacer** | Una prueba limpia mostró que la memoria liberada se reutiliza entre etapas (+929/+937/+945 MB en tres etapas idénticas). Lo que parecía retención en macOS era memoria **viva** o bloques grandes liberados que macOS mantiene en el RSS. |
| Paginar `fetch_extraction_targets` (Stage 4) | **Diferido (WP7, opcional)** | Solo aporta si se procesa página a página; hay que medir antes cuánto pesa el texto de un documento real. |
| Ingesta en una única transacción por documento ([ingest/loader.py:249](../src/scinr/newton/ingest/loader.py)) | **Fuera de alcance** | Presión sobre el servidor Neo4j, no sobre la memoria de la librería. Anotado como riesgo conocido. |

## 1. Diagnóstico verificado (resumen con evidencia)

Todas las cifras: Python 3.14, macOS, dependencias del `uv.lock` actual. Los scripts para
reproducirlas están en [`scripts/memory_bench/`](../scripts/memory_bench/) (ver §9).
**Usa `tracemalloc` (memoria viva) para PDF/CSV, no RSS.**

| # | Hallazgo | Dónde | Evidencia |
|---|---|---|---|
| H1 | `read_bytes()` del fichero completo solo para pasarlo a `store()`; `raw_bytes` vive todo `convert_one`. **Se ejecuta aunque `storage_backend="none"`**: `get_storage()` devuelve repos nulos (no `None`), el `is not None` siempre es verdadero y `NullRawFileRepository.store` descarta los bytes. | `converters/main.py:441`, `:715`; `stages/tabular.py:179-181`; `pipeline.py:541-543`; `storage/null.py:22` | Lectura de código. `GridIn.write` no copia (`BytesIO(data)` comparte buffer) y Motor ejecuta `upload_from_stream` en un executor, así que un file handle funciona. |
| H2 | Segundo `read_bytes()` del mismo PDF + `split_pdf` materializa **todos** los chunks + la caché de objetos del `PdfReader` retiene el contenido de las páginas leídas + la primera ventana serializa `max_pages` (900) páginas aunque luego se descarte. | `converters/pdf.py:204,220`; `pdf_splitter.py:96-199` | Memoria viva: **1458 MB (6,2×)** para un PDF de 235 MB y **4283 MB (4,6×)** para 938 MB. |
| H3 | `read_csv` hace `f.read()` → `io.StringIO` (buffer UCS4, **4×**) → `rows` → `data_rows` → `data_rows` con padding. | `tabular/reader.py:74-97` | CSV de 141 MB → **1780 MB (12,7×)**. Un arreglo de una pasada sin `StringIO` → 741 MB (5,3×); streaming → 1 MB. |
| H4 | Stages 3/4: `asyncio.gather` sobre **todos** los nodos; cada tarea construye contexto/prompt/mensajes **antes** de esperar el semáforo LLM. Neo4j (semáforo 10) es mucho más rápido que el LLM (semáforo 4 por defecto), así que todos los nodos se acumulan esperando el LLM. | `annotation/agent.py:131`, `annotation/nodes.py:157-170,276-282`; `entity_extraction/agent.py:101`, `entity_extraction/nodes.py:125-138,313-324` | Simulación N=10 000: **+701 MB** (`gather` actual) vs **+15 MB** (semáforo primero) vs +4 MB (pool), mismo tiempo (+5%). |
| H5 | `intermediate_doc`, `doc_obj` y `written` viven hasta el `return` de `_process_document_unit`, aunque solo se necesitan hasta Stage 1 / Stage 2. Multiplicado por `parallel_docs` (5 por defecto). | `pipeline.py:853-854,903,910` | Lectura de código. |
| H6 | Con `converter_output_dir=None`, `convert_one` serializa el documento entero (`doc.to_json()`, `indent=2`) a un directorio temporal que se borra inmediatamente. | `converters/main.py:470`; `pipeline.py:883-891` | Pico transitorio de 2–3 copias del documento sin ningún uso. |
| H7 | `all_rows` completo dentro del estado de LangGraph (`ainvoke` devuelve el estado entero) y `_write_tabular_with_normalization` mantiene estructuras O(filas): `dedup_map[...].row_indices`, `written_row_indices`, `remaining_indices`; crea **todas** las tareas de normalización a la vez. | `tabular/state.py`, `tabular/agent.py:118`, `tabular/neo4j_ops.py:266-561` (`:404-407`, `:1136-1235`) | Lectura de código. La escritura ya va por lotes de 500 (`_ROW_BATCH_SIZE`, `neo4j_ops.py:37`). |
| H8 | `wb.close()` fuera de `try/finally` en `read_xlsx`. | `tabular/reader.py:130` | 5 lecturas fallidas → 5 handles abiertos hasta `gc.collect()`. |

## 2. Reglas para el agente

- **No ejecutes nada contra el Neo4j/Mongo/LLM del usuario.** Todo se prueba con mocks/fakes.
- Estilo: `ruff` (`line-length = 100`, reglas `E,F,I,UP,B`), `from __future__ import annotations`.
  **Línea base de lint (medida en `main`): `.venv/bin/ruff check src tests` reporta 46 errores preexistentes** y
  `scripts/` está limpio. No los arregles en masa (ruido en el diff); tu criterio es **no introducir errores nuevos**:
  compara el recuento antes y después, y limpia solo lo que toques (`ruff check <ficheros modificados>`).
- Tests: `pytest` con `asyncio_mode = "auto"`; tests unitarios en `tests/unit/`.
- **Línea base de tests** (medida en `main`): `478 passed, 2 skipped, 3 failed` con
  `.venv/bin/python -m pytest tests/unit -q`. Los 3 fallos son **preexistentes y dependen del orden**
  (`get_config()` sin `configure()`); **no los persigas**:
  - `test_annotation_agent.py::TestRunAnnotationAgentPreconditionConcurrency::test_two_concurrent_calls_overlap_during_leaf_resolution`
  - `test_entity_extraction_agent.py::TestRunEntityExtractionAgentPreconditionConcurrency::test_two_concurrent_calls_overlap_during_leaf_resolution`
  - `test_pipeline_orchestration.py::TestAnnotationEntityExtractionConcurrency::test_multiple_documents_process_annotation_and_entity_extraction_concurrently`

  Tu criterio de no-regresión: el conjunto de `FAILED` debe ser **exactamente ese**.
  `tests/unit/test_delete_document.py` depende de un fixture autouse `_stub_deletion_config`: no lo toques.
- No cambies comportamiento funcional observable (ids, mensajes de error con formato pinneado en tests,
  semántica de `best_effort`/`fail_fast`) salvo donde este plan lo indique expresamente.
- Un commit por WP con mensaje corto en imperativo (estilo del repo: `Fixed delete document behaviour (#21)`).
  Rama sugerida: `feat/0.3.10-memory`. **No** hagas push ni PR sin que te lo pidan.
- Antes de tocar un WP, ejecuta su benchmark "antes" y guarda la cifra; después, la de "después" (§9).

## 3. Orden de ejecución

```
P0  (independientes entre sí; hacer en este orden)
    WP1  Streaming de raw files a GridFS + no leer con repo nulo
    WP2  Semáforo LLM primero en Stages 3/4
    WP3  output_dir=None en convert_one + liberar artefactos por etapa
P1
    WP4  PDF: generador de chunks con reader por ventana
    WP5  read_csv de una pasada + try/finally en read_xlsx
P2
    WP6  Tabular por streaming (pasadas A/B/C)        <- depende de WP5
P3  (opcional, medir antes)
    WP7  Paginación de fetch_extraction_targets
```

Estado (marca al terminar): `[x] WP1  [x] WP2  [x] WP3  [x] WP4  [x] WP5  [x] WP6  [ ] WP7 (opcional, no medido)`

---

## WP1 — Streaming de raw files a GridFS y no leer con repo nulo

**Problema:** H1. **Impacto:** elimina 1× el tamaño del archivo (y toda la lectura si no hay storage).

### Cambios

1. `src/scinr/newton/storage/base.py` — añadir a `RawFileRepository` un método **no abstracto**
   (para no romper a quien use `storage_backend="custom"`):

   ```python
   async def store_file(
       self, path: Path, filename: str, content_type: str, folder_path: str | None
   ) -> str:
       """Persist the file at *path*. Default: read it fully and delegate to store().

       Backends able to stream (MongoDB/GridFS) override this to avoid holding the file in memory.
       """
       content = await asyncio.to_thread(path.read_bytes)
       return await self.store(filename=filename, content=content,
                               content_type=content_type, folder_path=folder_path)
   ```
   Documenta en el docstring de `store()` que los backends propios pueden sobrescribir `store_file`.

2. `src/scinr/newton/storage/null.py` — `NullRawFileRepository.store_file` devuelve `""` **sin abrir el
   fichero** (misma semántica que `store`).

3. `src/scinr/newton/storage/mongodb/raw_files.py` — sobrescribir `store_file`:
   - Envoltorio `_HashingReader` (privado, mismo módulo) con `read(n=-1)` que lee del handle,
     actualiza un `hashlib.sha256()` y acumula `size`, y expone `hexdigest()`.
   - `with path.open("rb") as fh: reader = _HashingReader(fh); gridfs_id = await bucket.upload_from_stream(filename, reader, metadata={...})`.
     Motor ejecuta la subida en un executor, así que las lecturas del handle ocurren en un hilo
     (no bloquean el loop). Verifica en `.venv/lib/python3.14/site-packages/gridfs/synchronous/grid_file.py`
     (`GridIn.write`) que con un objeto con `.read` se lee en bloques de `chunk_size` y nunca `read()` completo.
   - El documento de metadatos usa `size_bytes=reader.size` y `checksum_sha256=reader.hexdigest()`.
   - Refactoriza para que `store` y `store_file` compartan la inserción del documento de metadatos
     (sin duplicar el bloque `doc = {...}` / `insert_one`).

4. Llamadores — sustituir `read_bytes()` + `store(content=...)` por `store_file(path=...)`:
   - `converters/main.py:439-447` (`convert_one`) y `:714-721` (`convert_single_file`).
   - `stages/tabular.py:179-194`: mantener el `try/except` que solo avisa si falla el almacenamiento y el cálculo
     de `_content_type`/`folder_path_str`. Elimina `raw_bytes`.
   - Comportamiento con repo nulo idéntico al actual: `raw_file_id == ""`.

### Tests
- `tests/unit/test_storage_null.py`: `store_file` con una ruta **inexistente** devuelve `""` (prueba que no abre el fichero).
- Nuevo `tests/unit/test_storage_raw_files_streaming.py`:
  - `_HashingReader`: `sha256` y `size` coinciden con `hashlib` sobre el fichero para lecturas de tamaños variables y `read(-1)`.
  - `MongoDBRawFileRepository.store_file` con `get_gridfs_bucket`/`get_db`/`get_config` parcheados: el `source` pasado
    a `upload_from_stream` **no es `bytes`**, y el documento insertado lleva `checksum_sha256`/`size_bytes` correctos.
  - Repo custom que solo implementa `store()`: `store_file` (por defecto) delega y recibe los bytes correctos.
- `tests/unit/test_converters_main.py` (o nuevo): con un `raw_file_repo` fake, `convert_one` llama a
  `store_file` y **no** llama a `Path.read_bytes` (parchéalo para que lance).

### Aceptación
- `grep -rn "read_bytes" src/scinr/newton/converters/main.py src/scinr/newton/stages/tabular.py` → sin resultados.
- Con `storage_backend="none"`, un `convert_one` sobre un fichero nunca abre el fichero para almacenarlo.
- Tests nuevos + los existentes pasan; sin nuevos `FAILED`.

### Riesgos
- Repos custom que ignoren `store_file` siguen leyendo el fichero entero (comportamiento actual, documentado).
- Documentar en `docs/user-guides/storage-backends.md` el nuevo método opcional `store_file`.

---

## WP2 — Semáforo LLM primero en Stages 3 y 4

**Problema:** H4. **Impacto:** de O(nodos × ~70 KB) a O(`llm_concurrency`) — 10 000 nodos: +701 MB → +15 MB.

**Idea:** adquirir el semáforo LLM **al inicio** de la función por nodo, de modo que contexto, prompt, mensajes y
schema solo existan mientras el nodo tiene un slot LLM. `asyncio.gather` sobre todos los nodos se mantiene
(N corrutinas pendientes ≈ 1,5 KB c/u).

> ⚠️ `asyncio.Semaphore` **no es reentrante**: si la función externa lo toma y `_decide_model`/`_extract_entities`
> lo vuelven a tomar → **deadlock**. Hay que **quitar** la adquisición interna.
> Orden de adquisición fijo: **LLM → Neo4j** (nunca al revés) para evitar ciclos.
> Las **escrituras** a Neo4j van **fuera** del slot LLM.

### Cambios

**`src/scinr/newton/annotation/nodes.py`**
- `_decide_model(ctx, node_id, theme, user_context)` (líneas ~131-200): **eliminar el parámetro `semaphore` y el
  `async with semaphore:`**; el cuerpo (`with_llm_retry`, repair loop) queda igual, des-indentado.
- `process_single_annotation_node` (líneas ~242-291) pasa a:

  ```python
  async with bedrock_semaphore:                       # slot LLM primero
      async with neo4j_semaphore:
          ctx = await _fetch_node_context(node_data, driver)
      if ctx is None:
          return {... "fetch_context failed ..."}
      decision, error = await _decide_model(ctx, node_id, theme, user_context)
  if decision is not None:                             # fuera del slot LLM
      async with neo4j_semaphore:
          write_error = await _write_decision(...)
  ```
  Mantener el parámetro `bedrock_semaphore` en la firma pública (lo pasa `annotation/agent.py:133`).

**`src/scinr/newton/entity_extraction/nodes.py`**
- `_extract_entities(composite_schema, info_units, node_full_id, node_id, node_title)` (líneas ~97-168): **quitar
  `semaphore` y su `async with`**.
- `process_single_extraction_target` (líneas ~277-336): tomar `bedrock_semaphore` **antes de `_compose_schema`** y
  soltarlo tras `_extract_entities`; `_write_entities` y `_mark_extracted` quedan fuera, bajo `neo4j_semaphore`.

  ```python
  async with bedrock_semaphore:
      composite_schema, schema_error = _compose_schema(target)
      if composite_schema is None: return {...}
      extraction, extract_error = await _extract_entities(composite_schema, ...)
  if extraction is None: return {...}
  async with neo4j_semaphore: write_error = await _write_entities(...)
  async with neo4j_semaphore: mark_error = await _mark_extracted(...)
  ```

No cambiar `annotation/agent.py` ni `entity_extraction/agent.py` (siguen usando `asyncio.gather`).

### Tests (nuevos, en `test_annotation_agent.py` / `test_entity_extraction_agent.py` o ficheros propios)
- **Cota de concurrencia (annotation):** parchea `scinr.newton.annotation.nodes.get_async_driver` (→ objeto dummy),
  `get_neo4j_semaphore` (→ `Semaphore(10)`), `_fetch_node_context` (fake async que incrementa un contador
  `in_flight` y devuelve un `ctx`), `_decide_model` (fake que `await asyncio.sleep(0.01)` y decrementa)
  y `_write_decision` (→ `None`). Lanza 60 nodos con `asyncio.Semaphore(3)` y `asyncio.gather`.
  Afirma `max(in_flight) <= 3` (antes del cambio sería 60).
- **Cota de concurrencia (entity):** igual, contando entre `_compose_schema` y el fin de `_extract_entities`.
- **Sin deadlock:** el mismo test con `Semaphore(1)` termina (usa `asyncio.wait_for(..., timeout=5)`).
- **Errores:** `ctx is None` devuelve el dict de error y libera el slot (un segundo nodo puede progresar).
- Los tests existentes no referencian estas funciones privadas (verificado); si algo se rompe, adáptalo.

### Aceptación
- `python scripts/memory_bench/bench_fanout.py gather_all 10000` vs `sem_first 10000` reproduce la diferencia
  (referencia del cambio; la aceptación real son los tests de cota de concurrencia).
- Tiempo total del stage no empeora más de ~5% en el bench.

### Riesgos
- El slot LLM se mantiene durante el `fetch` de Neo4j (ms frente a segundos de LLM): coste asumido.
- `with_llm_retry` duerme con el slot tomado, igual que hoy.

---

## WP3 — `output_dir=None` en `convert_one` y liberar artefactos por etapa

**Problema:** H5 y H6.

### Cambios

1. `src/scinr/newton/converters/main.py`
   - `convert_one(entry, output_dir: Path | None, ...)`: **`None` = no escribir a disco.** Si es `None`: no `mkdir`,
     no `write_text(doc.to_json())`, no log "Written:"; devuelve `[(entry, None, doc)]`.
   - Ajustar anotaciones de tipo: `tuple[list[tuple[Path, Path | None, IntermediateDocument]], ...]`
     (también en `convert_folder`, que sigue exigiendo `Path`; `dry_run` sin cambios).
   - Actualizar el docstring (la decisión de escribir la toma **el llamador vía `output_dir`**, no el config).

2. `src/scinr/newton/pipeline.py` `_process_document_unit` (líneas ~857-904)
   - Eliminar la rama `tempfile.TemporaryDirectory()` y `import tempfile` si queda sin uso.
     Si `converter_output_dir` es falsy: `await convert_one(unit.source_path, None, ...)`.
   - Tras `intermediate_doc = written[0][2]`: `del written` (o `written = None`).
   - Tras obtener `doc_obj` en Stage 1: `intermediate_doc = None`.
   - Tras la ingesta (Stage 2, incluida la rama `ingestion_json`): `doc_obj = None`.
   - Actualizar el docstring de `_process_document_unit` (párrafo `converter_output_dir`) y quitar el comentario
     largo sobre el `TemporaryDirectory`.

### Tests
- `tests/unit/test_process_document_unit.py::TestPreprocessTempDirCleanup` (líneas ~653-720) **fija el comportamiento
  antiguo** (le pasa un directorio temporal a `convert_one` y comprueba que se borra). Reescríbela: con
  `converter_output_dir=None`, `convert_one` recibe `output_dir is None` y no aparecen entradas nuevas en
  `tempfile.gettempdir()`. Adapta también su variante "convert_one raises".
- Nuevo: `convert_one(entry, None, ...)` con un converter fake **no crea ficheros** y devuelve `(entry, None, doc)`.
- Nuevo (liberación por etapa): fake `convert_one` que crea `class _Doc: ...` y el test conserva solo un
  `weakref.ref`; fake `ingest_one` (se ejecuta tras Stage 1) afirma `ref() is None`; y un fake
  `run_annotation` afirma que el `doc_obj` ya no está vivo (usa `gc.collect()` si hay ciclos).
  Evita que el propio test guarde referencias fuertes.

### Aceptación
- `grep -n "TemporaryDirectory\|mkdtemp" src/scinr/newton/pipeline.py` → sin resultados.
- `test_process_document_unit.py` y `test_pipeline_orchestration.py` pasan (salvo los 3 preexistentes).

### Riesgos
- Otros llamadores de `convert_one` que lean `written[i][1]` (path): hoy solo `convert_folder`
  (siempre con `Path`) y el pipeline (usa `[2]`). Verifica con `grep -rn "convert_one" src tests`.

---

## WP4 — PDF: generador de chunks con `PdfReader` por ventana

**Problema:** H2. **Impacto (memoria viva):** 235 MB → 226 MB de pico (antes 1458 MB); 938 MB → 328 MB (antes 4283 MB).
El pico deja de crecer con el archivo (depende del tamaño del chunk).

### Diseño (validado con `scripts/memory_bench/bench_pdf.py`, variante `gen_fresh`)
1. Decidir si hay que dividir **sin leer el fichero**: `size = path.stat().st_size`, `total_pages` con
   `PdfReader(fh)` sobre el fichero (solo lee el índice). `must_split = size > max_bytes or total_pages > max_pages`.
2. **Sin split:** leer el fichero una vez (≤ 45 MB) y enviarlo como hoy.
3. **Con split:** generador que produce **un chunk cada vez**; el llamador lo envía y lo suelta antes de pedir el siguiente.
4. **Un `PdfReader` nuevo por ventana**, abierto sobre el fichero y cerrado al terminar la ventana. Motivo (medido):
   el reader de pypdf cachea el contenido de todas las páginas leídas; limpiar `resolved_objects` no basta
   (548 MB vs 328 MB en el PDF de 938 MB). Coste: re-parsear el xref por ventana (10,7 s vs 0,9 s en 938 MB;
   despreciable frente a la latencia del OCR).
5. **Ventana inicial según tamaño medio por página:**
   `window = max(1, min(max_pages, int(0.8 * max_bytes / (size / total_pages))))`. Evita serializar 900 páginas y
   descartarlas (490 MB vs 293 MB). Si aun así una ventana serializada excede `max_bytes`, se bisecciona como hoy.

### Cambios

**`src/scinr/newton/converters/pdf_splitter.py`** (mantener `count_pdf_pages`, `needs_splitting`, `split_pdf` y sus
tests: siguen siendo la API basada en bytes)
- Añadir `probe_pdf(path: Path) -> tuple[int, int]` → `(size_bytes, total_pages)`. Errores de lectura/PDF cifrado →
  `ConversionError` con mensajes análogos a `count_pdf_pages`.
- Añadir `iter_pdf_chunks(path: Path, max_pages: int, max_bytes: int, *, source_name: str = "<document>") -> Iterator[PdfChunk]`
  (generador síncrono; los chunks se serializan con `_serialize_page_range`, reutilizado tal cual).
  - Bisección como generador (versión perezosa de `_bisect_window`); una página sola > `max_bytes` → `PdfSplitError`
    con el **mismo mensaje** que hoy.
  - **Mapeo de errores idéntico a `split_pdf`:** `PdfSplitError` se relanza; cualquier otra excepción (p. ej. PDF cifrado
    que abre bien pero falla al leer páginas — cubierto por `tests/unit/test_pdf_splitter.py` ~l.139-151) →
    `ConversionError(f"Cannot split PDF {source_name}: {exc}")`. Envuelve la iteración **dentro** del generador.

**`src/scinr/newton/converters/pdf.py`** — `PdfConverter.convert` (líneas ~202-286)
- Sustituir `read_bytes()` + `needs_splitting` + `split_pdf` por `probe_pdf` + (ruta sin split | `iter_pdf_chunks`).
- Bucle de chunks: `for chunk in iter_pdf_chunks(...)` (generador síncrono; la serialización de cada ventana bloquea el
  loop igual que hoy lo hace `split_pdf`, y solo ocurre entre `await`s); tras `_convert_chunk(...)` hacer `del chunk`
  antes de la siguiente iteración.
- **Etiquetas/mensajes:** ya no se conoce `N` (nº total de chunks) de antemano. Usa `chunk {i}` (contador corrido) y
  el rango de páginas; `total_pages_original` es `total_pages` de `probe_pdf`. **Conserva el formato
  `[start, end)` y los textos `best_effort`/`MISTRAL_OCR_ERROR_STRATEGY`** (los tests los asertan:
  `"[2, 4)" in message`, `"best_effort"`, `"MISTRAL_OCR_ERROR_STRATEGY"`, y `missing_page_ranges == [(2, 4)]`).
- `fail_fast` / `best_effort` sin cambios de semántica. `IntermediateDocument(pages=all_pages, missing_page_ranges=...)` igual.
- Actualizar docstrings (módulo y clase) y el import de `pdf.py:37`.

### Cambio de comportamiento conocido (documentar en CHANGELOG)
Un `PdfSplitError` (una única página que pesa > `safe_max_bytes`) ahora puede detectarse **después** de haber enviado
chunks anteriores a Mistral (antes fallaba antes de enviar nada). Es un caso raro; el documento sigue abortando
siempre, independientemente de `error_strategy`.

### Tests
- `tests/unit/test_pdf_splitter.py` (ampliar): **paridad** — para varias combinaciones `(max_pages, max_bytes)`,
  `[(c.start_page, c.end_page) for c in iter_pdf_chunks(path,...)]` == mismo cálculo con `split_pdf(bytes,...)`,
  y los `pdf_bytes` reconstruyen el mismo nº de páginas. `probe_pdf` con PDF válido, corrupto y cifrado.
  **Pereza:** con `_serialize_page_range` parcheado con un contador, tomar solo el primer chunk serializa solo la
  primera ventana. Página única demasiado grande → `PdfSplitError` (mismo mensaje).
- `tests/unit/test_pdf_converter_chunking.py`: deben pasar **sin cambios funcionales** (5 páginas, `safe_max_pages=2`
  → chunks `[0,2) [2,4) [4,5)`; el mock de `httpx.AsyncClient.post` se mantiene). Ajusta solo lo estrictamente
  necesario si algún texto de etiqueta estaba pinneado.
- Nuevo: con split, `Path.read_bytes` **no** se llama (parchéalo para que lance) y, en `fail_fast`, tras el fallo del
  chunk 2 no se serializa el chunk 3 (contador sobre `_serialize_page_range`).
- (Opcional, marcado lento) test de escalado con `tracemalloc`: el pico de `iter_pdf_chunks` sobre un PDF 4× mayor
  no supera 1,5× el pico del menor. Genera el PDF con el helper de `bench_pdf.py`.

### Aceptación
- `python scripts/memory_bench/bench_pdf.py make 2400 && python scripts/memory_bench/bench_pdf.py real 2400`
  → pico vivo **< 0,5× el tamaño del archivo** (referencia: 328 MB para 938 MB) y **no** crece linealmente
  al pasar de 600 a 2400 páginas.
- Suite de PDF existente en verde.

### Riesgos / notas
- Sigue existiendo el cuerpo HTTP de ~5,3× el chunk en vuelo (base64 + data-URL + JSON), acotado a un chunk;
  con `parallel_docs=5` ≈ 1,2 GB. **No** se aborda aquí.
- `mistral_ocr_chunk_concurrency` existe en config pero el bucle actual es secuencial; no lo cambies.

---

## WP5 — `read_csv` de una pasada y `read_xlsx` con `try/finally`

**Problema:** H3 y H8. **Impacto:** 12,7× → ~5× el tamaño del CSV, manteniendo el contrato (`all_rows` en lista).

### Cambios — `src/scinr/newton/tabular/reader.py`
- `read_csv(path)`: abrir con `utf-8-sig`, `errors="replace"`, `newline=""`; **sniff** sobre los primeros 4096 caracteres
  (`f.read(4096)`, luego `f.seek(0)`) con `csv.Sniffer().sniff(sample, delimiters=",;\t|")` y fallback a `,`;
  una sola pasada sobre `csv.reader(f, delimiter=...)` sin `io.StringIO`:
  - saltar filas vacías (`not any(c.strip() for c in row)`);
  - la primera fila no vacía → cabeceras (`_deduplicate_headers([_cell_to_str(h) ...], path.name)`);
  - resto: `[_cell_to_str(c) for c in row[:ncols]]` y padding con `""` hasta `ncols`.
  Resultado idéntico al actual (incluido recorte de filas largas y padding de cortas). Referencia funcional:
  `read_csv_one_pass` en `scripts/memory_bench/bench_csv.py`.
- `read_xlsx(path)`: envolver en `try/finally: wb.close()`; construir cada fila una sola vez (sin la lista intermedia
  `rows` completa ni el `[(r + [""]*n)[:n] ...]` posterior): cabeceras = primera fila no vacía, resto con padding en el mismo bucle.

### Tests
- `tests/unit/test_csv_reader.py` debe pasar **sin cambios** (BOM, `;`, tab, duplicadas, vacío, una fila, filas cortas, sheet_name=stem…).
- Añadir: fila más larga que la cabecera se recorta; fichero con solo cabecera; CSV con campos entrecomillados con
  saltos de línea; fichero mayor de 4096 caracteres (sniff con `seek(0)` no pierde datos).
- Añadir: `read_xlsx` cierra el workbook si `iter_rows` lanza (parchea `ReadOnlyWorksheet.iter_rows` con `side_effect`
  y comprueba con un spy/`Mock` sobre `load_workbook` que `close()` se llamó).

### Aceptación
- `python scripts/memory_bench/bench_csv.py`: fila `current read_csv` **≤ 6× el tamaño del archivo**
  (antes 12,7×).

---

## WP6 — Tabular por streaming (pasadas A / B / C)

**Problema:** H7. **Meta:** memoria O(lote de 500 + claves únicas de normalización), independiente del nº de filas.
Es el WP más grande: **antes de codificar, escribe una nota de diseño breve (en la descripción del commit o en
`plans/`) y confirma con el responsable si algo del §"Decisiones abiertas" te bloquea.**

### Diseño

Tres pasadas secuenciales sobre el fichero (E/S secuencial; cada pasada reabre el fichero/workbook):

| Pasada | Qué hace | Memoria |
|---|---|---|
| **A** (en `load_sheets`) | Por hoja: cabeceras, `total_rows`, **preview** (filas en índices `0, n//4, n//2, 3n//4, n-1`, ver `select_preview_rows`). Necesita el total, por eso es una pasada propia. | O(1) |
| **B** (solo si hay campos normalizables) | Recorre por lotes de 500; por fila calcula las **claves de normalización** y rellena `dedup_map: unique_key -> NormalizationEntry` (con `source_values`). **Sin `row_indices`.** Después el LLM normaliza las claves únicas (resultados en `engine.result_cache`). | O(claves únicas) |
| **C** (escritura) | Recorre por lotes de 500 con contador de fila; **por cada fila del lote recalcula su clave y busca el resultado en `engine.result_cache`**; filas sin clave o sin resultado → composite "plain". Escribe con `_write_row_batch`. | O(lote) |

Sustituye así `row_indices`, `written_row_indices` (set) y `remaining_indices` (lista): todos O(filas).

### Cambios

1. **Lector** (`tabular/reader.py`): nueva API por streaming, manteniendo `read_csv`/`read_xlsx`/`read_tabular_file`
   (WP5) como envoltorios que materializan, para no romper tests ni llamadores:
   - `scan_tabular_file(path, delimiter=None) -> list[SheetScan]` (pasada A: `sheet_name`, `headers`, `total_rows`, `preview`).
   - `iter_sheet_batches(path, sheet_name, batch_size=500, delimiter=None) -> Iterator[list[list[str]]]`
     (CSV con `csv.reader` sobre el handle; XLSX con `openpyxl` `read_only=True` + `iter_rows(values_only=True)` y
     `try/finally wb.close()`). Mismas reglas de saneado que WP5 (saltar vacías, `_cell_to_str`, padding/recorte).
   - `.xls` sigue lanzando `ConversionError` como hoy.
2. **Estado LangGraph** (`tabular/state.py`, `tabular/nodes.py`): `TabularFileData` deja de llevar `all_rows`; lleva
   `file_path`, `sheet_name`, `headers`, `total_rows`, `preview`, `preview_markdown`. `load_sheets` usa la pasada A.
   `ainvoke` ya no devuelve las filas.
3. **Clave de fila única y compartida:** extraer de `_build_normalization_dedup_map` (`neo4j_ops.py:1136-1235`) una función
   pura `compute_row_normalization_keys(headers, row_values, mapping, primary_cls, comp_cls_map, comp_class_names, get_specs_fn)
   -> list[tuple[str, NormalizationEntry-like]]` usada **tanto en B como en C**. Si divergen, las filas no encontrarán
   su resultado en `engine.result_cache`. Cubrirlo con un test de igualdad B↔C.
4. **`write_tabular_subgraph` / `_write_tabular_with_normalization`** (`neo4j_ops.py:43-561`): reciben una **fábrica de
   iteradores de lotes** en vez de `all_rows`. Contador de fila corrido para `row_{idx+1}` y `batch_start_index`.
   Se elimina `NormalizationEntry.row_indices` del uso (déjalo en el dataclass si hay más consumidores; verifica con `grep`).
5. **Tareas de normalización acotadas:** `neo4j_ops.py:404-407` crea **todas** las `asyncio.create_task` de golpe. Procesar
   los `key_batches` en rebanadas (p. ej. `llm_concurrency * 4`) o con `asyncio.Semaphore` adquirido **antes** de crear la
   tarea; el semáforo global `get_llm_semaphore()` se mantiene dentro.
6. `stages/tabular.py`: ya sin `read_bytes` (WP1).

### Tests
- `tests/unit/test_csv_reader.py`: sin cambios (contrato conservado). Añadir tests de `scan_tabular_file` (preview en los
  índices esperados, total correcto con filas vacías y multilínea) e `iter_sheet_batches` (lotes de tamaño ≤ 500, orden y
  contenido idénticos a `read_csv`, último lote parcial, hoja vacía).
- **Equivalencia:** para un CSV pequeño y uno con >500 filas, la concatenación de `iter_sheet_batches` == `read_csv(...)[0]["all_rows"]`.
- **Clave B↔C:** `compute_row_normalization_keys` produce la misma clave que la usada para poblar `dedup_map`.
- **Escritura por streaming:** con `_write_row_batch` mockeado, `write_tabular_subgraph` sobre 2 000 filas llama a
  `_write_row_batch` 4 veces con lotes de 500 y `batch_start_index` 0/500/1000/1500, y **nunca** recibe la lista completa.
- **Memoria:** test con `tracemalloc` (marcado lento) sobre un CSV sintético de ~200 000 filas con `_write_row_batch` mockeado:
  pico vivo < 50 MB (frente a cientos de MB con `all_rows`).
- `tests/unit/test_tabular_route_row_values.py` y lo de `test_pipeline_orchestration.py` que toque tabular: verdes.

### Aceptación
- El estado de LangGraph no contiene filas (`assert "all_rows" not in sheet`).
- Pico vivo de la ruta de escritura independiente del nº de filas (test de memoria anterior).

### Decisiones abiertas / caveats
- **Memoria ≠ tiempo/coste.** Con columnas de alta cardinalidad hay millones de claves únicas y millones de llamadas al LLM
  de normalización (`normalization_batch_size=5` claves por llamada); y la escritura hace transacciones por fila
  (`write_extraction_subgraph`). Este WP garantiza acotar la **memoria**, no el tiempo. No lo intentes resolver aquí.
- El fichero **no debe modificarse** entre pasadas (razonable en un pipeline por lotes; documéntalo en el docstring).
- Si hay que priorizar, WP6 puede dividirse: (6a) lector por streaming + estado sin filas, (6b) claves B↔C y escritura por
  streaming, (6c) tareas de normalización acotadas.

---

## WP7 — (opcional, medir primero) Stage 4: no cargar todo el texto del documento

`fetch_extraction_targets` (`entity_extraction/neo4j_ops.py:17-133`) carga el texto completo de todos los `InfoUnit` con
`result.data()` (`:111`, `:118`), y `targets` retiene todo el texto mientras dura el stage. **Solo tiene sentido si se procesa
por páginas** (paginación por clave sobre `appearance_order`, p. ej. 200 targets por página, procesando y descartando cada
página); paginar y luego hacer `gather` sobre la lista completa no gana nada.
**Prerrequisito:** medir cuántos MB de texto pesa un documento real grande tras WP1–WP6. Si es pequeño frente al resto, no hacer.

---

## 8. Cierre: changelog, docs, versión

- `CHANGELOG.md` (formato Keep a Changelog, sección nueva `## [0.3.10] - <fecha>`):
  - **Changed:** PDF grandes se convierten con un chunk en memoria a la vez (cambio de comportamiento de `PdfSplitError` diferido);
    `convert_one(output_dir=None)` no escribe a disco; los raw files se suben a GridFS en streaming (`RawFileRepository.store_file`).
  - **Fixed:** Stages 3/4 ya no acumulan contexto/prompt/schema de todos los nodos antes del semáforo LLM; `read_csv` ~2,5× menos
    memoria; `read_xlsx` cierra el workbook ante errores; los raw files ya no se leen a memoria con `storage_backend="none"`.
  - Subir `version` en `pyproject.toml` a `0.3.10` **solo si el responsable lo confirma**.
- Docs a revisar/actualizar (`grep -rn` por los nombres): `docs/user-guides/storage-backends.md`, `docs/configuration.md`,
  `docs/architecture.md` (líneas ~638 `convert_one`, ~750 flujo en memoria entre etapas). La referencia de API usa mkdocstrings:
  mantén docstrings correctos.
- Rellenar la tabla del §9 con las cifras "antes/después" reales.

## 9. Medición (reproducible)

Scripts en `scripts/memory_bench/` (autocontenidos; no necesitan Neo4j/Mongo/LLM; usa `.venv/bin/python`):

| Script | Qué mide | Cómo |
|---|---|---|
| `bench_fanout.py` | Fan-out Stages 3/4 (simulación): `gather_all` vs `sem_first` vs `bounded` | `python scripts/memory_bench/bench_fanout.py sem_first 10000` (usa RSS: solo comparación relativa) |
| `bench_pdf.py` | Pico **vivo** (tracemalloc) del split de PDF: `current`, `gen_no_hint`, `generator`, `gen_clear`, `gen_fresh`, `real` | `... bench_pdf.py make 2400` y luego `... bench_pdf.py <modo> 2400` |
| `bench_csv.py` | Pico **vivo** de `read_csv` frente a la variante de una pasada y al streaming | `python scripts/memory_bench/bench_csv.py` |

**Cifras base (antes de cualquier cambio) y objetivos:**

| Métrica | Antes | Objetivo tras el WP |
|---|---|---|
| Fan-out 10 000 nodos (RSS relativo) | +701 MB (medido: +717 MB) | WP2: ~+15 MB — **conseguido: +15 MB** (tiempo 26,3 s → 27,9 s, +6 %) |
| PDF 235 MB, memoria viva | 1458 MB (6,2×) | WP4: ≲ 230 MB — **conseguido: 227 MB (0,97×)** |
| PDF 938 MB, memoria viva | 4283 MB (4,6×) | WP4: ≲ 330 MB (no lineal) — **conseguido: 328 MB (0,35×)** |
| `read_csv` 141 MB, memoria viva | 1780 MB (12,7×) | WP5: ≤ 6× (referencia 741 MB) — **conseguido: 741 MB (5,3×)** |
| Ruta de escritura tabular, 1M filas | O(filas) | WP6: O(lote) — **conseguido**: test con 150 000 filas × 8 cols, pico vivo < 50 MB (escritura + scan) |

**Cómo medir memoria correctamente:** en macOS los bloques grandes liberados pueden seguir contando en el RSS y en Linux
se devuelven antes; por eso PDF/CSV se miden con `tracemalloc` (memoria viva). El fan-out se compara con RSS solo entre
modos del mismo script.

## 10. Checklist final

- [x] `.venv/bin/ruff check src tests` ≤ 46 errores (los preexistentes; ninguno nuevo en ficheros tocados) y `ruff check scripts` limpio.
- [x] `.venv/bin/python -m pytest tests/unit -q` → único conjunto de `FAILED` = los 3 preexistentes del §2.
- [x] Cifras "antes/después" anotadas en el §9.
- [x] `CHANGELOG.md` y docs actualizados (§8).
- [x] Ningún `read_bytes` del fichero completo en `converters/main.py`, `converters/pdf.py`, `stages/tabular.py`.
- [x] Sin `TemporaryDirectory`/`mkdtemp` en `pipeline.py`.
