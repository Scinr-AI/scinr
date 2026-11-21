# Configuration

Configure `scinr.newton` the ingestion pipeline using the `configure()` function, environment variables, or a combination of both.

---

## Configuration Resolution

scinr uses a **triple-resolution** system. For every setting, the effective value is determined by the following priority (highest to lowest):

1. **Explicit argument** passed to `configure()`
2. **Environment variable** set in the process environment or loaded from a `.env` file
3. **Hard-coded default** built into the library

This means you can set sensible defaults via environment variables and override individual values at runtime with `configure()`, or vice versa.

```python
# Example: env var sets concurrency to 4, but configure() overrides to 8
# $ export LLM_CONCURRENCY=4
configure(llm_concurrency=8)  # final value: 8
```

---

## Environment Variables

All environment variables are optional unless otherwise noted. They are read when `configure()` runs. A `.env` file is not read by `configure()` itself: several `scinr.newton` modules call `load_dotenv()` when they are imported, so the file is loaded at import time.

### LLM / Model

The usual way to set the LLM is to build a LangChain chat model in code and pass it as `configure(llm=..., repair_llm=...)` (not required at `configure()` time; needed for LLM stages). As a shortcut for AWS Bedrock, when `llm` is not passed and `MODEL_ID` is set, `configure()` builds a `ChatBedrockConverse` itself (temperature `0`) from the variables below; it requires `langchain-aws` and raises `ConfigurationError` if it is missing. `repair_llm` has no environment variable.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_ID` | *(unset)* | Bedrock model id. Used only when `llm=` is not passed. |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region of that Bedrock client. Read only when `MODEL_ID` is used. |
| `MAX_TOKENS` | `65536` | `max_tokens` of that Bedrock client. Read only when `MODEL_ID` is used. |
| `LLM_CONCURRENCY` | `4` | Maximum number of concurrent LLM calls. |

### Neo4j

The Neo4j driver is an optional extra (`pip install "scinr[neo4j]"`). The credentials below are required only when `GRAPH_BACKEND` is `neo4j` (the default).

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt URI for the Neo4j instance. |
| `NEO4J_USER` | *(required)* | Neo4j database username. **Note:** previous versions used `NEO4J_USERNAME` — this was renamed to `NEO4J_USER`. |
| `NEO4J_PASSWORD` | *(required)* | Neo4j user password. |
| `NEO4J_DATABASE` | *(required)* | Neo4j database name |
| `NEO4J_AUTH` | *(unset)* | Fallback in `user/password` format, used for whichever of `NEO4J_USER` / `NEO4J_PASSWORD` is not set otherwise. |
| `GRAPH_CONCURRENCY` | `10` | Maximum async graph-store concurrency (any graph backend). Must be >= 1. Deprecated alias: `NEO4J_CONCURRENCY` (removed in 0.7.0). |
| `GRAPH_SYNC_CONCURRENCY` | `8` | Maximum sync graph-store concurrency (any graph backend). Must be >= 1. Deprecated alias: `NEO4J_SYNC_CONCURRENCY` (removed in 0.7.0). |
| `GRAPH_BACKEND` | `neo4j` | Backend for the graph (the pipeline's write path and the read-only navigation API): `neo4j`, `postgres` or `sqlite`. The Neo4j variables above are required only with `neo4j`; the Neo4j driver is the optional extra `pip install "scinr[neo4j]"`. |

### Storage and Freeze

The backend selectors, and the settings of the MongoDB backend (the `MONGODB_*` variables apply only when the storage or freeze backend is `mongodb`). The PostgreSQL and SQLite settings are in the next section.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `STORAGE_BACKEND` | `none` | Storage backend: `none` (no persistence), `mongodb`, `postgres`, `sqlite`, or `custom`. |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string. |
| `MONGODB_DATABASE` | `scinr` | MongoDB database name. |
| `MONGODB_RAW_FILES_COLLECTION` | `raw_files` | Collection name for raw file metadata. |
| `MONGODB_PAGES_COLLECTION` | `converted_pages` | Collection name for converted document pages. |
| `MONGODB_GRIDFS_BUCKET` | `raw_binaries` | GridFS bucket name for binary file storage. |
| `MONGODB_ENSURE_INDEXES` | `true` | Create the MongoDB indexes automatically on the first `get_storage()` call of the process. Set to `false` when the database user lacks the `createIndex` privilege and operations manage the indexes. |
| `FREEZE_BACKEND` | *(inherits `STORAGE_BACKEND`)* | Backend for document snapshots (`freeze_document()`, `restore_document()`): `none`, `mongodb`, `postgres`, `sqlite` or `custom`. See [Document Freezing](user-guides/document-freezing.md). |
| `MONGODB_FROZEN_COLLECTION` | `frozen_documents` | Collection for snapshot metadata. |
| `MONGODB_FROZEN_GRIDFS_BUCKET` | `frozen_snapshots` | GridFS bucket for snapshot files. |
| `MONGODB_IMAGES_COLLECTION` | `page_images` | Collection for the metadata of stored images and video frames. |
| `MONGODB_IMAGES_GRIDFS_BUCKET` | `page_image_binaries` | GridFS bucket for the bytes of stored images and video frames. |

### SQL (PostgreSQL and SQLite)

Used when `GRAPH_BACKEND`, `STORAGE_BACKEND` or `FREEZE_BACKEND` is `postgres` or `sqlite`. Install `scinr[postgres]` or `scinr[sqlite]`. See [SQL Graph Backends](user-guides/sql-graph.md).

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SQL_URL` | *(required for SQL axes)* | Connection URL shared by every SQL axis: `postgresql://user:pass@host:5432/db` or `sqlite:///path/newton.db` (`sqlite:////abs/path.db` for an absolute path). Its scheme must match the axes, and the axes cannot mix Postgres and SQLite. A driver suffix (`postgresql+psycopg://`) is accepted and ignored. Redacted in logs and in `repr(config)`. |
| `SQL_SCHEMA` | `newton` | Postgres schema that holds the tables. Ignored by SQLite. Must be a plain identifier. |
| `SQL_POOL_SIZE` | `10` | Connection pool size of each SQL engine (minimum `1`). |
| `SQL_MAX_ENGINES` | `64` | Maximum SQL engines kept in cache (separately for the synchronous and the asynchronous ones), one per distinct connection (`sql_url`, `sql_schema`, pool size, text-search config) and, for the asynchronous ones, per event loop (minimum `1`). The least recently used one is disposed when the limit is exceeded. Each engine has its own pool, so a process using several configurations can open up to `SQL_POOL_SIZE` connections per engine. See [Several configurations in one process](user-guides/sql-graph.md#several-configurations-in-one-process). |
| `SQL_FULLTEXT_CONFIG` | `simple` | Postgres text-search configuration of the full-text index on `info_units`. Applied only when the schema is created. Must be a plain identifier. |
| `SQLITE_FILES_DIR` | *(empty: `<db path without extension>.files/`)* | Folder for the binaries (raw files, images, snapshots) kept by the SQLite backend. Required for an in-memory database. |

### PDF / Mistral OCR

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MISTRAL_API_KEY` | `None` | Mistral OCR API key. **Required to ingest any PDF** — PDF conversion is Mistral OCR only, there is no fallback. |
| `MISTRAL_OCR_SAFE_MAX_PAGES` | `900` | Page count above which a PDF is split into chunks before each chunk is sent to the OCR API. |
| `MISTRAL_OCR_SAFE_MAX_BYTES` | `47185920` (45 MiB) | Byte size above which a PDF is split into chunks before each chunk is sent to the OCR API. |
| `MISTRAL_OCR_MAX_RETRIES` | `15` | Number of retry attempts for OCR failures. Retry uses exponential backoff capped at 5 minutes between retries. |
| `MISTRAL_OCR_RETRY_BACKOFF_SECONDS` | `2.0` | Base backoff in seconds between retries. |
| `MISTRAL_OCR_CHUNK_CONCURRENCY` | `1` | Maximum concurrent OCR chunk processing. |
| `MISTRAL_OCR_ERROR_STRATEGY` | `fail_fast` | Error handling: `fail_fast` (abort on first error) or `best_effort` (continue and collect what is possible). |
| `MISTRAL_OCR_CONCURRENCY` | `2` | Maximum PDFs in OCR conversion at the same time per event loop, whatever `parallel_docs` is. Bounds the memory of OCR. |
| `MISTRAL_OCR_IMAGE_MAX_PAGES` | `50` | Pages per OCR chunk when the embedded images are requested (only with `describe_embedded_images` and a `vision_llm`): a response with images is far heavier than text only. The smaller of this and `MISTRAL_OCR_SAFE_MAX_PAGES` applies. |

### Multimodal (images, audio, video)

See [Multimodal Ingestion](user-guides/multimodal-ingestion.md). `vision_llm` and `transcriber` are objects, so they are set with `configure()` only.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DESCRIBE_EMBEDDED_IMAGES` | `true` | Describe the embedded images of PDF, DOCX and PPTX. **Only acts with a `vision_llm` configured**; without one, nothing is requested from Mistral and no image blob is read. |
| `STORE_DESCRIBED_IMAGES` | `true` | Store the original of each embedded image that gets a description. Only acts with a persistent storage backend that stores images (`RawFileRepository.stores_images`). See [Retrieving images and frames](user-guides/multimodal-ingestion.md#retrieving-images-and-frames). |
| `STORE_VIDEO_FRAMES` | `true` | Store each described video frame, with its instant. Long videos take about `video_frames_per_minute × minutes` JPEGs. |
| `MEDIA_ERROR_STRATEGY` | `best_effort` | A failed embedded image, audio/video segment or frame: `best_effort` skips it and records it in `missing_images` / `missing_segments`; `fail_fast` fails the document. |
| `STRIP_LOCATION` | `true` | With `true` the GPS location of images is not read nor stored. |
| `IMAGE_MAX_SIDE_PX` | `1568` | Longer side, in pixels, of the images sent to the `vision_llm` (downscaled before sending). |
| `IMAGE_MIN_SIDE_PX` | `64` | Embedded images whose shorter side is below this are trivial (icons, separators) and are not described. |
| `MAX_IMAGES_PER_PAGE` | `10` | Embedded images described per page; the largest are kept. |
| `IMAGE_DECODE_CONCURRENCY` | `1` | Images read and decoded at once per event loop. The memory peak grows with it (a 12 MP photo is ~36 MB of pixels). Minimum `1`. |
| `MEDIA_SEGMENT_SECONDS` | `300` | Length of each audio / video segment (one page per segment). Minimum `10`. |
| `MEDIA_MIN_SEGMENT_SECONDS` | `10` | Minimum length of the last segment: a shorter tail moves the previous cut earlier. `0` disables it; at most half of `MEDIA_SEGMENT_SECONDS`. |
| `MEDIA_LANGUAGE` | *(unset: auto-detect)* | Audio language, ISO 639-1 (`es`, `en`…). |
| `MAX_MEDIA_DURATION_SECONDS` | `14400` (4 h) | Longest audio / video accepted; longer is a `ConversionError` before anything is spent. `none` = no limit. |
| `VIDEO_FRAMES_PER_MINUTE` | `2.0` | Frames per minute described in videos (uniform sampling). `0` = transcription only. |
| `MEDIA_CONCURRENCY` | `1` | Audio and video files in conversion at the same time per event loop, whatever `parallel_docs` is. |
| `FFMPEG_PATH`, `FFPROBE_PATH` | *(unset: `PATH`)* | Paths to the `ffmpeg` / `ffprobe` system binaries, required for audio and video. |

### Pipeline

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PROMPT_CACHING_ENABLED` | `true` | Enable prompt caching. Currently effective for AWS Bedrock; ignored for other providers. |
| `EXTRACTION_BATCH_SIZE` | `1` | Number of pages per extraction chunk. |
| `AUTO_INFO_UNITS_MAX_INPUT_CHARS` | `10000` | Maximum characters of paragraph text per second-call request of `run_pipeline(info_units_mode="auto")`; longer nodes are split into consecutive groups. Minimum `1000`. The call uses the `extraction` stage of `llm_stage_configuration`. |
| `VERBATIM_INFO_UNIT_MAX_CHARS` | `4000` | Maximum characters of each InfoUnit in `run_pipeline(info_units_mode="verbatim")`; longer paragraphs are split into several InfoUnits. Minimum `100`. |
| `PROMPT_FAMILY` | `generic` | Prompt template family: `generic`, `claude`, or `gpt_reasoning`. |
| `SCINR_EXTRA_MODELS_PATHS` | `""` (empty) | Colon-separated list of extra model package paths. |
| `FULL_DOCSTRING` | `true` | Saves the full docstring from the models. On False, it only saves the first line before a jumpline. |

### Normalization

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NORMALIZATION_ENABLED` | `true` | Enable tabular data normalization via LLM. |
| `NORMALIZATION_BATCH_SIZE` | `5` | Batch size for normalization LLM calls. |

### Consolidation

Used only by the `fast_extraction=True` consolidation step of Stage 1.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `CONSOLIDATION_TOKEN_SAFETY_MARGIN` | `0.75` | Fraction of `max_tokens` reserved for output. |
| `CONSOLIDATION_MAX_OUTPUT_TOKENS` | *(unset)* | Explicit output-token ceiling; derived from the margin when unset. |
| `CONSOLIDATION_MAX_INPUT_TOKENS` | `65536` | Input-token ceiling that sizes the consolidation batches. |

---

## Programmatic Configuration

The `configure()` function is the primary way to set up scinr at runtime. It accepts keyword arguments organized by category. Every parameter is optional at `configure()` time: most fall back to an environment variable or hard-coded default, while `llm=` (which has neither) is only needed once the pipeline runs an LLM stage.

```python
from scinr.newton import configure
```

### LLM Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `llm` | `BaseChatModel \| None` | A ready-built LangChain chat model for all LLM calls (`ChatBedrockConverse`, `ChatOpenAI`, `ChatOllama`, …). Optional at `configure()` time; needed for the extraction / annotation / entity-extraction stages and tabular normalization. |
| `repair_llm` | `BaseChatModel \| None` | Separate model for the JSON repair loop. Falls back to `llm` if not provided. |

### Neo4j Parameters

The same settings as the [Neo4j environment variables](#neo4j), passed as lowercase keyword arguments.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `neo4j_uri` | `str \| None` | Bolt URI. Default `bolt://localhost:7687`. |
| `neo4j_user` | `str \| None` | Username. Required when `graph_backend="neo4j"` (via arg, `NEO4J_USER` or `NEO4J_AUTH`). |
| `neo4j_password` | `str \| None` | Password. Required when `graph_backend="neo4j"` (via arg, `NEO4J_PASSWORD` or `NEO4J_AUTH`). |
| `neo4j_database` | `str \| None` | Database name. Required when `graph_backend="neo4j"` (via arg or `NEO4J_DATABASE`). |
| `graph_concurrency` | `int \| None` | Max concurrent async graph-store sessions (any graph backend). Default `10`. Deprecated alias: `neo4j_concurrency` (removed in 0.7.0). |
| `graph_sync_concurrency` | `int \| None` | Max concurrent sync ingestion dispatches. Default `8`. Deprecated alias: `neo4j_sync_concurrency` (removed in 0.7.0). |
| `graph_backend` | `str \| None` | Graph backend (write path and navigation): `"neo4j"` (default), `"postgres"` or `"sqlite"`. The Neo4j credentials are required only for `"neo4j"`. |

### SQL Parameters

The same settings as the [SQL environment variables](#sql-postgresql-and-sqlite), passed as lowercase keyword arguments.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `sql_url` | `str \| None` | Connection URL shared by the SQL axes. Required when any axis is `"postgres"` or `"sqlite"`. Default `""`. |
| `sql_schema` | `str \| None` | Postgres schema. Default `"newton"`. |
| `sql_pool_size` | `int \| None` | Connection pool size. Default `10`. |
| `sql_max_engines` | `int \| None` | Maximum SQL engines kept in cache per kind, sync and async (least recently used is disposed). Minimum `1`. Default `64`. |
| `sql_fulltext_config` | `str \| None` | Postgres text-search configuration. Default `"simple"`. |
| `sqlite_files_dir` | `str \| None` | Folder for the SQLite binaries. Default `""` (next to the `.db`). |

### Models / Themes Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled_base_themes` | `list[ThemePath \| str] \| None` | List of base themes to enable for extraction. |
| `enabled_user_themes` | `list[str] \| None` | List of user-defined themes to enable. |
| `extra_models_paths` | `list[str \| Path] \| None` | Additional paths to model packages. |

### Storage Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `storage_backend` | `str \| None` | `"none"` (default), `"mongodb"`, `"postgres"`, `"sqlite"` or `"custom"`. |
| `mongodb_uri` | `str \| None` | MongoDB connection string. Default `mongodb://localhost:27017`. |
| `mongodb_database` | `str \| None` | MongoDB database name. Default `scinr`. |
| `mongodb_raw_files_collection` | `str \| None` | Collection for raw file metadata. Default `raw_files`. |
| `mongodb_pages_collection` | `str \| None` | Collection for converted pages. Default `converted_pages`. |
| `mongodb_gridfs_bucket` | `str \| None` | GridFS bucket for binaries. Default `raw_binaries`. |
| `mongodb_ensure_indexes` | `bool \| None` | Create the MongoDB indexes automatically (default `True`). `False` skips it, for users without the `createIndex` privilege. |
| `custom_storage` | `tuple \| None` | `(RawFileRepository, PageRepository)` when `storage_backend="custom"`. |
| `freeze_backend` | `str \| None` | Snapshot backend: `none`, `mongodb`, `postgres`, `sqlite` or `custom`. Unset → `FREEZE_BACKEND`, else the resolved `storage_backend`. |
| `mongodb_frozen_collection` | `str \| None` | Collection for snapshot metadata (default `frozen_documents`). |
| `mongodb_frozen_gridfs_bucket` | `str \| None` | GridFS bucket for snapshot files (default `frozen_snapshots`). |
| `mongodb_images_collection` | `str \| None` | Collection for stored images and frames (default `page_images`). |
| `mongodb_images_gridfs_bucket` | `str \| None` | GridFS bucket for stored images and frames (default `page_image_binaries`). |
| `custom_freeze_storage` | `FreezeRepository \| None` | Snapshot repository instance when `freeze_backend='custom'`. |

### Converter Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `extra_converters` | `dict[str, type] \| None` | Dictionary mapping file extensions to converter classes. |

### PDF / Mistral OCR Parameters

The same settings as the [PDF / Mistral OCR environment variables](#pdf-mistral-ocr), passed as lowercase keyword arguments.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `mistral_api_key` | `str \| None` | Mistral API key for PDF OCR. Default `None`. |
| `mistral_ocr_safe_max_pages` | `int \| None` | Page threshold above which OCR is forced. Default `900`. |
| `mistral_ocr_safe_max_bytes` | `int \| None` | Byte threshold above which OCR is forced. Default `47185920` (45 MiB). |
| `mistral_ocr_max_retries` | `int \| None` | Retry attempts per chunk on retryable errors. Default `15`. |
| `mistral_ocr_retry_backoff_seconds` | `float \| None` | Base backoff between retries (exponential, capped at 5 min). Default `2.0`. |
| `mistral_ocr_chunk_concurrency` | `int \| None` | Concurrent OCR chunk tasks. Default `1`. |
| `mistral_ocr_error_strategy` | `Literal["fail_fast", "best_effort"] \| None` | OCR error handling. Default `"fail_fast"`. |
| `mistral_ocr_concurrency` | `int \| None` | PDFs in OCR conversion at the same time per event loop. Default `2`. |
| `mistral_ocr_image_max_pages` | `int \| None` | Pages per chunk when the embedded images are requested. Default `50`. |

### Multimodal Parameters

The same settings as the [multimodal environment variables](#multimodal-images-audio-video), plus the two model objects. See [Multimodal Ingestion](user-guides/multimodal-ingestion.md).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `vision_llm` | `BaseChatModel \| None` | Multimodal chat model (accepts images) that describes embedded images, image files and video frames. No `with_structured_output` needed. **Does not fall back to `llm`.** Default `None`. |
| `transcriber` | `Transcriber \| None` | Speech-to-text for audio and video (`OpenAICompatibleTranscriber`, `FasterWhisperTranscriber` or your own). Default `None`. |
| `describe_embedded_images` | `bool \| None` | Default `True` (acts only with `vision_llm`). |
| `store_described_images` | `bool \| None` | Default `True` (acts only with a storage backend that stores images). |
| `store_video_frames` | `bool \| None` | Default `True`. |
| `media_error_strategy` | `Literal["fail_fast", "best_effort"] \| None` | Default `"best_effort"`. |
| `strip_location` | `bool \| None` | Default `True`. |
| `image_max_side_px` | `int \| None` | Default `1568` (minimum `64`). |
| `image_min_side_px` | `int \| None` | Default `64`. |
| `max_images_per_page` | `int \| None` | Default `10`. |
| `image_decode_concurrency` | `int \| None` | Default `1` (minimum `1`). |
| `media_segment_seconds` | `int \| None` | Default `300` (minimum `10`). |
| `media_min_segment_seconds` | `int \| None` | Default `10`, capped to half a segment when not given; `0` disables it. |
| `media_language` | `str \| None` | ISO 639-1 code, or `None` to auto-detect. |
| `max_media_duration_seconds` | `int \| None` | Default `14400`; pass `None` explicitly for no limit. |
| `video_frames_per_minute` | `float \| None` | Default `2.0`; `0` = transcription only. |
| `media_concurrency` | `int \| None` | Default `1`. |
| `ffmpeg_path`, `ffprobe_path` | `str \| None` | Default: looked up in the `PATH`. |

`configure()` does not check cross requirements: image, audio and video files are always accepted, and a missing `vision_llm`, `transcriber` or `ffmpeg` fails that document at conversion, with an error that says how to configure it. With `describe_embedded_images=True` and no `vision_llm`, `configure()` logs one `INFO` line saying embedded images will not be described.

### Pipeline Parameters

The same settings as the [Pipeline environment variables](#pipeline), passed as lowercase keyword arguments.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `llm_concurrency` | `int \| None` | Max concurrent LLM calls across all stages. Default `4`. |
| `extraction_batch_size` | `int \| None` | Pages per extraction chunk. Default `1`. |
| `auto_info_units_max_input_chars` | `int \| None` | Maximum characters of paragraph text per second-call request in `info_units_mode="auto"`. Default `10000`, minimum `1000`. The second call uses the LLM, prompt family and caching of the `extraction` stage in `llm_stage_configuration`. |
| `verbatim_info_unit_max_chars` | `int \| None` | Maximum characters of each InfoUnit in `info_units_mode="verbatim"` (longer paragraphs are split). Default `4000`, minimum `100`. |
| `prompt_caching_enabled` | `bool \| None` | Bedrock system-prompt caching. Default `True`. |
| `full_docstring` | `bool \| None` | Store the full model docstring (`True`) or only its first line (`False`). Default `True`. |
| `consolidation_token_safety_margin` | `float \| None` | Fraction of `max_tokens` reserved for output during `fast_extraction=True` consolidation. Default `0.75`. |
| `consolidation_max_output_tokens` | `int \| None` | Explicit output-token ceiling for consolidation; derived from the margin when unset. Default `None`. |
| `consolidation_max_input_tokens` | `int \| None` | Input-token ceiling that sizes the consolidation batches. Default `65536`. |

> **Note:** `info_units_mode`, like `fast_extraction`, is a per-call `run_pipeline()` argument; see [Running the Pipeline](user-guides/running-pipeline.md#info_units_mode--infounits-of-stage-1).

> **Note:** `fast_extraction` itself is **not** a `configure()` setting — it is a per-call `run_pipeline(fast_extraction=...)` argument. See [Running the Pipeline](user-guides/running-pipeline.md#fast_extraction-parallel-stage-1-extraction-opt-in).

### Logging Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `log_level` | `str` | Python log level. Default: `"INFO"`. Accepts `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`. |

### Prompt Family Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `prompt_family` | `PromptFamily \| Literal["generic", "claude", "gpt_reasoning"] \| None` | Prompt template family. See [Prompt Families](#prompt-families) for details. |

### Normalization Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `normalization_enabled` | `bool \| None` | Enable tabular data normalization. |
| `normalization_batch_size` | `int \| None` | Batch size for normalization LLM calls. |
| `normalization_llm` | `Any \| None` | Dedicated LLM client for normalization. Falls back to `llm` if not provided. |

---

## Configuration Examples

### Navigation Only (No LLM)

Graph navigation (`scinr.newton.navigation`) is read-only and does not need any LLM. Configure only the graph connection. On Neo4j:

```python
# Navigation only — no LLM required
configure(
    neo4j_user="neo4j",
    neo4j_password="your_password",
    neo4j_database="neo4j",
)  # storage_backend defaults to "none"; graph_backend to "neo4j"
```

On PostgreSQL or SQLite, set the backend and the URL instead (see [SQL Graph Backends](user-guides/sql-graph.md)):

```python
configure(
    graph_backend="sqlite",
    sql_url="sqlite:////var/lib/scinr/newton.db",
)
```

### Minimal Setup

Put Neo4j / OCR settings in `.env`; build the LLM in code and pass it to `configure()`. The `.env` file is loaded by `python-dotenv` when `scinr.newton` is imported (not by `configure()`), so you never need to import dotenv manually.

```bash
# .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

MISTRAL_API_KEY=your_mistral_key
```

```python
import asyncio
from scinr.newton import configure, run_pipeline
from langchain_aws import ChatBedrockConverse
async def main():
    # The model is passed via llm=; everything else can come from .env.
    llm = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-6", region_name="us-east-1")
    configure(llm=llm)

    result = await run_pipeline(input_raw="./raw_docs")
    print(f"Pipeline: {'success' if result.success else 'failed'}")

asyncio.run(main())
```

> **Note:** `configure()` must be called before `run_pipeline()`. Supply `llm=` whenever the run includes an LLM stage (extraction, annotation, entity extraction, tabular normalization).

### Full AWS Bedrock Setup

Complete programmatic configuration for a production Bedrock deployment.

```python
from scinr.newton import configure
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-6", region_name="us-east-1")
configure(
    # LLM — needed for LLM stages
    llm=llm,
    repair_llm=None,  # falls back to `llm`

    # Neo4j
    neo4j_uri="bolt://neo4j.internal:7687",
    neo4j_user="scinr_ingest",
    neo4j_password="secure_password",
    graph_concurrency=10,
    graph_sync_concurrency=8,

    # PDF / Mistral OCR
    mistral_api_key="your_mistral_key",
    mistral_ocr_safe_max_pages=900,
    mistral_ocr_safe_max_bytes=47185920,
    mistral_ocr_max_retries=15,
    mistral_ocr_error_strategy="best_effort",

    # Pipeline
    prompt_caching_enabled=True,
    extraction_batch_size=1,
    llm_concurrency=4,
    prompt_family="claude", # If using a Claude Model (Best tested performance). Generic for any other model (Kimi K2.5 tested good performance)

    # Logging
    log_level="INFO",
)
```

### With Normalization Enabled

Enable tabular data normalization with a dedicated LLM for the normalization step.

```python
from scinr.newton import configure

configure(
    llm_concurrency=4,
    prompt_family="claude",

    # Normalization
    normalization_enabled=True,
    normalization_batch_size=5,
    # normalization_llm=dedicated_llm_instance,  # optional: separate LLM for normalization
)
```

### With MongoDB Storage

Persist raw files and converted pages to MongoDB.

```python
from scinr.newton import configure

configure(
    storage_backend="mongodb",
    mongodb_uri="mongodb://user:pass@mongo.internal:27017",
    mongodb_database="scinr_production",
    mongodb_raw_files_collection="raw_files",
    mongodb_pages_collection="converted_pages",
    mongodb_gridfs_bucket="raw_binaries",
)
```

### With SQLite (No Neo4j, No MongoDB)

The graph, the navigation API, raw files, pages and snapshots in one SQLite file.

```python
from scinr.newton import configure

configure(
    graph_backend="sqlite",
    storage_backend="sqlite",   # freeze_backend inherits it
    sql_url="sqlite:////var/lib/scinr/newton.db",
)
```

For PostgreSQL (and Row-Level Security) see [SQL Graph Backends](user-guides/sql-graph.md).

---

## Using a `.env` File

scinr reads standard `.env` files. You can copy the provided example from the repository and fill in your values.

```bash
cp .env.example .env
```

The `.env.example` file in the project root contains every available setting with inline comments. Key notes:

- The `.env` file is loaded automatically (via `python-dotenv`) when `scinr.newton` is imported, not by `configure()`. You never import `dotenv` yourself.
- `NEO4J_USER`, `NEO4J_PASSWORD`, and `NEO4J_DATABASE` are required only when `GRAPH_BACKEND=neo4j` (the default); the SQL backends need `SQL_URL` instead. Pass the LLM as `configure(llm=...)` (or use the `MODEL_ID` Bedrock shortcut) when a run includes an LLM stage.
- Environment values are **overridden** by explicit `configure(...)` arguments (see [Configuration Resolution](#configuration-resolution)).
- Leaving an optional variable unset falls back to the hard-coded default in the [Complete Reference](#complete-reference-all-settings) table.

---

## ScinrConfig

`configure()` returns a `ScinrConfig` object that holds the resolved configuration. You can also retrieve the active configuration at any time using `get_config()`.

```python
from scinr.newton import configure, get_config

# Set up configuration
configure(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="secret",
    llm_concurrency=8,
    prompt_family="claude",
)

# Read back the active configuration
config = get_config()
print(f"Neo4j URI: {config.neo4j_uri}")
print(f"LLM Concurrency: {config.llm_concurrency}")
print(f"Prompt Family: {config.prompt_family}")
```

The `ScinrConfig` object is a plain (not frozen) dataclass; treat it as read-only. To change configuration, call `configure()` again with the new values — it will produce a new `ScinrConfig` that replaces the previous one.

---

## Prompt Families

The `prompt_family` parameter selects a set of prompt templates optimized for different LLM providers.

| Family | Description |
| :--- | :--- |
| `generic` | Provider-agnostic prompts. Safe default that works with any LLM. |
| `claude` | Optimized for Anthropic Claude models. Uses Claude-specific formatting and system prompt conventions. |
| `gpt_reasoning` | Optimized for OpenAI reasoning models (o-series). Uses the specific message structure required by reasoning-capable models. |

### Choosing a Prompt Family

- **Claude models on Bedrock** — use `"claude"` for best results.
- **OpenAI o-series models** — use `"gpt_reasoning"`.
- **Other providers or unsure** — use `"generic"` (the default).

```python
configure(prompt_family="claude")  # for Claude models
configure(prompt_family="gpt_reasoning")  # for OpenAI o-series
configure(prompt_family="generic")  # default, works everywhere
```

---

## LLM por etapa (Per-Stage LLM)

By default one `llm`, one `prompt_family` and one `prompt_caching_enabled` serve every LLM
call of the pipeline. `llm_stage_configuration` lets each stage have its own.

```python
from scinr.newton import StageLLMConfig, configure

configure(
    llm=ChatBedrockConverse(model="...sonnet..."),      # default for every stage
    prompt_family="claude",
    llm_stage_configuration={
        "extraction": StageLLMConfig(llm=ChatBedrockConverse(model="...haiku...")),
        "entity_extraction": StageLLMConfig(
            llm=ChatOpenAI(model="..."),
            prompt_family="gpt_reasoning",
        ),
    },
)
```

`StageLLMConfig` has three fields, all optional:

| Field | Type | `None` means |
| :--- | :--- | :--- |
| `llm` | `BaseChatModel \| None` | the global `llm` |
| `prompt_family` | `PromptFamily \| str \| None` | the global `prompt_family` |
| `prompt_caching_enabled` | `bool \| None` | the global `prompt_caching_enabled` |

### Stages

| Key | Calls it covers |
| :--- | :--- |
| `extraction` | Per-chunk structure extraction (stage 1) **and** the consolidation call of `fast_extraction=True` |
| `annotation` | The model decision of annotation (stage 3) |
| `entity_extraction` | Entity extraction (stage 4) |
| `tabular` | `classify_theme`, `decide_model` and `map_columns` of the tabular pipeline |

The consolidation of the fast branch is part of stage 1 and has no configuration of its own:
it uses `extraction`.

### Resolution rules

- **Each field resolves on its own.** A `None` field takes the global value, so you can change
  only the prompt family of a stage and keep the global `llm`.
- **Validation in `configure()`.** An unknown key, a value that is not a `StageLLMConfig`, an
  `llm` that is not a `BaseChatModel`, an invalid family or a non-bool `prompt_caching_enabled`
  raises `ConfigurationError` (the unknown-key message lists the valid keys).
- **No environment variables**: the values are objects, like `llm`.
- **The global `llm` is still required** when a stage that runs has no LLM of its own. The
  error is the usual one, prefixed with the name of the stage that asked for it.
- **The prompt family is never inferred from the model.** A stage with its own `llm` but no
  `prompt_family` uses the global `prompt_family`.
- **Global configuration**: it cannot be changed per `run_pipeline()` call.
- The Bedrock `cachePoint` is added when the **stage's** LLM is `ChatBedrockConverse` and the
  **stage's** caching is enabled.

### What does not change

- `repair_llm` is still a single LLM for all stages (falling back to `llm`).
- `normalization_llm` is unchanged: it falls back to `llm`, not to the `tabular` configuration.
- `vision_llm` and `transcriber` are unchanged.

### Concurrency

There is a single global semaphore (`llm_concurrency`) for all stages. If two stages use
providers with different rate limits, set the global limit to that of the most restrictive one.

### Logging

`configure()` logs (INFO) which stages have their own configuration and the family and caching
each uses, with the model name only, never keys or credentials. Each call logs at DEBUG the
model of its stage.

### Accessors

- `get_stage_config(stage)` returns the resolved `llm`, `prompt_family` and
  `prompt_caching_enabled` of a stage.
- `get_llm(temperature=0.0, *, stage=None)`, `get_prompt_family(*, stage=None)` and
  `make_system_message(text, *, stage=None)`. Without `stage` they use the global values, as
  before.

---

## Complete Reference: All Settings

For quick lookup, here is every configurable setting with its resolution chain:

| Setting | `configure()` param | Environment Variable | Default |
| :--- | :--- | :--- | :--- |
| **LLM Client** | `llm` | `MODEL_ID` (Bedrock shortcut, with `AWS_DEFAULT_REGION` and `MAX_TOKENS`) | `None` |
| **Repair LLM Client** | `repair_llm` | *(none)* | `None` (falls back to `llm`) |
| **Neo4j URI** | `neo4j_uri` | `NEO4J_URI` | `bolt://localhost:7687` |
| **Neo4j User** | `neo4j_user` | `NEO4J_USER` | *(required)* |
| **Neo4j Password** | `neo4j_password` | `NEO4J_PASSWORD` | *(required)* |
| **Neo4j Database** | `neo4j_database` | `NEO4J_DATABASE` | *(required)* |
| **Neo4j Auth** | *(none)* | `NEO4J_AUTH` | Unset (fallback in `user/password` format) |
| **Graph Async Concurrency** | `graph_concurrency` | `GRAPH_CONCURRENCY` | `10` |
| **Graph Sync Concurrency** | `graph_sync_concurrency` | `GRAPH_SYNC_CONCURRENCY` | `8` |
| **Graph Backend** | `graph_backend` | `GRAPH_BACKEND` | `neo4j` (`neo4j`, `postgres`, `sqlite`) |
| **LLM Concurrency** | `llm_concurrency` | `LLM_CONCURRENCY` | `4` |
| **Storage Backend** | `storage_backend` | `STORAGE_BACKEND` | `none` (`none`, `mongodb`, `postgres`, `sqlite`, `custom`) |
| **MongoDB URI** | `mongodb_uri` | `MONGODB_URI` | `mongodb://localhost:27017` |
| **MongoDB Database** | `mongodb_database` | `MONGODB_DATABASE` | `scinr` |
| **MongoDB Raw Files** | `mongodb_raw_files_collection` | `MONGODB_RAW_FILES_COLLECTION` | `raw_files` |
| **MongoDB Pages** | `mongodb_pages_collection` | `MONGODB_PAGES_COLLECTION` | `converted_pages` |
| **MongoDB GridFS** | `mongodb_gridfs_bucket` | `MONGODB_GRIDFS_BUCKET` | `raw_binaries` |
| **MongoDB Ensure Indexes** | `mongodb_ensure_indexes` | `MONGODB_ENSURE_INDEXES` | `True` |
| **Custom Storage** | `custom_storage` | *(none)* | `None` |
| **Freeze Backend** | `freeze_backend` | `FREEZE_BACKEND` | resolved `storage_backend` |
| **MongoDB Frozen Collection** | `mongodb_frozen_collection` | `MONGODB_FROZEN_COLLECTION` | `frozen_documents` |
| **MongoDB Frozen GridFS** | `mongodb_frozen_gridfs_bucket` | `MONGODB_FROZEN_GRIDFS_BUCKET` | `frozen_snapshots` |
| **MongoDB Images Collection** | `mongodb_images_collection` | `MONGODB_IMAGES_COLLECTION` | `page_images` |
| **MongoDB Images GridFS** | `mongodb_images_gridfs_bucket` | `MONGODB_IMAGES_GRIDFS_BUCKET` | `page_image_binaries` |
| **Custom Freeze Storage** | `custom_freeze_storage` | *(none)* | `None` |
| **SQL URL** | `sql_url` | `SQL_URL` | `""` (required for SQL axes) |
| **SQL Schema** | `sql_schema` | `SQL_SCHEMA` | `newton` |
| **SQL Pool Size** | `sql_pool_size` | `SQL_POOL_SIZE` | `10` |
| **SQL Max Engines** | `sql_max_engines` | `SQL_MAX_ENGINES` | `64` |
| **SQL Full-text Config** | `sql_fulltext_config` | `SQL_FULLTEXT_CONFIG` | `simple` |
| **SQLite Files Dir** | `sqlite_files_dir` | `SQLITE_FILES_DIR` | `""` (`<db path>.files/`) |
| **Mistral API Key** | `mistral_api_key` | `MISTRAL_API_KEY` | `None` |
| **OCR Max Pages** | `mistral_ocr_safe_max_pages` | `MISTRAL_OCR_SAFE_MAX_PAGES` | `900` |
| **OCR Max Bytes** | `mistral_ocr_safe_max_bytes` | `MISTRAL_OCR_SAFE_MAX_BYTES` | `47185920` |
| **OCR Max Retries** | `mistral_ocr_max_retries` | `MISTRAL_OCR_MAX_RETRIES` | `15` |
| **OCR Backoff** | `mistral_ocr_retry_backoff_seconds` | `MISTRAL_OCR_RETRY_BACKOFF_SECONDS` | `2.0` |
| **OCR Concurrency** | `mistral_ocr_chunk_concurrency` | `MISTRAL_OCR_CHUNK_CONCURRENCY` | `1` |
| **OCR Error Strategy** | `mistral_ocr_error_strategy` | `MISTRAL_OCR_ERROR_STRATEGY` | `fail_fast` |
| **OCR PDFs in Flight** | `mistral_ocr_concurrency` | `MISTRAL_OCR_CONCURRENCY` | `2` |
| **OCR Pages per Chunk with Images** | `mistral_ocr_image_max_pages` | `MISTRAL_OCR_IMAGE_MAX_PAGES` | `50` |
| **Vision LLM** | `vision_llm` | *(none)* | `None` (no fallback to `llm`) |
| **Transcriber** | `transcriber` | *(none)* | `None` |
| **Describe Embedded Images** | `describe_embedded_images` | `DESCRIBE_EMBEDDED_IMAGES` | `true` (only with `vision_llm`) |
| **Store Described Images** | `store_described_images` | `STORE_DESCRIBED_IMAGES` | `true` (only with image storage) |
| **Store Video Frames** | `store_video_frames` | `STORE_VIDEO_FRAMES` | `true` |
| **Media Error Strategy** | `media_error_strategy` | `MEDIA_ERROR_STRATEGY` | `best_effort` |
| **Strip Location** | `strip_location` | `STRIP_LOCATION` | `true` |
| **Image Max Side** | `image_max_side_px` | `IMAGE_MAX_SIDE_PX` | `1568` |
| **Image Min Side** | `image_min_side_px` | `IMAGE_MIN_SIDE_PX` | `64` |
| **Max Images per Page** | `max_images_per_page` | `MAX_IMAGES_PER_PAGE` | `10` |
| **Image Decode Concurrency** | `image_decode_concurrency` | `IMAGE_DECODE_CONCURRENCY` | `1` |
| **Media Segment** | `media_segment_seconds` | `MEDIA_SEGMENT_SECONDS` | `300` |
| **Media Min Segment** | `media_min_segment_seconds` | `MEDIA_MIN_SEGMENT_SECONDS` | `10` |
| **Media Language** | `media_language` | `MEDIA_LANGUAGE` | `None` (auto-detect) |
| **Max Media Duration** | `max_media_duration_seconds` | `MAX_MEDIA_DURATION_SECONDS` | `14400` |
| **Video Frames per Minute** | `video_frames_per_minute` | `VIDEO_FRAMES_PER_MINUTE` | `2.0` |
| **Media Concurrency** | `media_concurrency` | `MEDIA_CONCURRENCY` | `1` |
| **ffmpeg / ffprobe** | `ffmpeg_path`, `ffprobe_path` | `FFMPEG_PATH`, `FFPROBE_PATH` | `PATH` lookup |
| **Prompt Caching** | `prompt_caching_enabled` | `PROMPT_CACHING_ENABLED` | `true` |
| **Full Docstring** | `full_docstring` | `FULL_DOCSTRING` | `true` |
| **Extraction Batch** | `extraction_batch_size` | `EXTRACTION_BATCH_SIZE` | `1` |
| **Auto-mode second-call input size** | `auto_info_units_max_input_chars` | `AUTO_INFO_UNITS_MAX_INPUT_CHARS` | `10000` |
| **Verbatim InfoUnit size** | `verbatim_info_unit_max_chars` | `VERBATIM_INFO_UNIT_MAX_CHARS` | `4000` |
| **Prompt Family** | `prompt_family` | `PROMPT_FAMILY` | `generic` |
| **Extra Models Paths** | `extra_models_paths` | `SCINR_EXTRA_MODELS_PATHS` | `""` |
| **Enabled Base Themes** | `enabled_base_themes` | *(none)* | `None` |
| **Enabled User Themes** | `enabled_user_themes` | *(none)* | `None` |
| **Extra Converters** | `extra_converters` | *(none)* | `None` |
| **Normalization** | `normalization_enabled` | `NORMALIZATION_ENABLED` | `true` |
| **Normalization Batch** | `normalization_batch_size` | `NORMALIZATION_BATCH_SIZE` | `5` |
| **Normalization LLM** | `normalization_llm` | *(none)* | `None` (falls back to `llm`) |
| **Consolidation output margin** | `consolidation_token_safety_margin` | `CONSOLIDATION_TOKEN_SAFETY_MARGIN` | `0.75` (used only when `fast_extraction=True`) |
| **Consolidation max output tokens** | `consolidation_max_output_tokens` | `CONSOLIDATION_MAX_OUTPUT_TOKENS` | `None` (derived from the margin) |
| **Consolidation max input tokens** | `consolidation_max_input_tokens` | `CONSOLIDATION_MAX_INPUT_TOKENS` | `65536` |
| **Log Level** | `log_level` | *(none)* | `"INFO"` |
