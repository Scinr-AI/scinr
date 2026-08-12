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

All environment variables are optional unless otherwise noted. They are read at configuration time (when `configure()` is first called or when the config is first accessed).

### LLM / Model

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_ID` | *(required if no `llm` arg)* | Model ID for the primary LLM. For AWS Bedrock, use ARNs such as `us.anthropic.claude-sonnet-4-6`. |
| `REPAIR_MODEL_ID` | *(falls back to `MODEL_ID`)* | Model ID used for repair and retry LLM calls. Can be a cheaper/faster model (e.g. `us.anthropic.claude-haiku-3`). |
| `AWS_DEFAULT_REGION` | `us-east-1` | AWS region for Bedrock calls. |
| `MAX_TOKENS` | `65536` | Maximum tokens for Bedrock LLM calls. |
| `LLM_CONCURRENCY` | `4` | Maximum number of concurrent LLM calls. |

### Neo4j

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NEO4J_URI` | `bolt://localhost:7687` | Bolt URI for the Neo4j instance. |
| `NEO4J_USER` | *(required)* | Neo4j database username. **Note:** previous versions used `NEO4J_USERNAME` — this was renamed to `NEO4J_USER`. |
| `NEO4J_PASSWORD` | *(required)* | Neo4j user password. |
| `NEO4J_AUTH` | *(fallback "user/password")* | Alternative authentication format as a single `user/password` string. Used if `NEO4J_USER` and `NEO4J_PASSWORD` are not both set. |
| `NEO4J_CONCURRENCY` | `10` | Maximum async Neo4j concurrency. |
| `NEO4J_SYNC_CONCURRENCY` | `8` | Maximum sync Neo4j concurrency. |

### Storage (MongoDB)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `STORAGE_BACKEND` | `none` | Storage backend: `none` (no persistence), `mongodb`, or `custom`. |
| `MONGODB_URI` | `mongodb://localhost:27017` | MongoDB connection string. |
| `MONGODB_DATABASE` | `scinr` | MongoDB database name. |
| `MONGODB_RAW_FILES_COLLECTION` | `raw_files` | Collection name for raw file metadata. |
| `MONGODB_PAGES_COLLECTION` | `converted_pages` | Collection name for converted document pages. |
| `MONGODB_GRIDFS_BUCKET` | `raw_binaries` | GridFS bucket name for binary file storage. |

### PDF / Mistral OCR

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MISTRAL_API_KEY` | `None` | Mistral API key for PDF OCR extraction. Required to process PDF files. |
| `MISTRAL_OCR_SAFE_MAX_PAGES` | `900` | Maximum number of pages before OCR becomes mandatory. |
| `MISTRAL_OCR_SAFE_MAX_BYTES` | `47185920` (45 MiB) | Maximum file size in bytes before OCR is required. |
| `MISTRAL_OCR_MAX_RETRIES` | `3` | Number of retry attempts for OCR failures. |
| `MISTRAL_OCR_RETRY_BACKOFF_SECONDS` | `2.0` | Base backoff in seconds between retries. |
| `MISTRAL_OCR_CHUNK_CONCURRENCY` | `1` | Maximum concurrent OCR chunk processing. |
| `MISTRAL_OCR_ERROR_STRATEGY` | `fail_fast` | Error handling: `fail_fast` (abort on first error) or `best_effort` (continue and collect what is possible). |

### Pipeline

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PROMPT_CACHING_ENABLED` | `true` | Enable prompt caching. Currently effective for AWS Bedrock; ignored for other providers. |
| `EXTRACTION_BATCH_SIZE` | `1` | Number of pages per extraction chunk. |
| `PROMPT_FAMILY` | `generic` | Prompt template family: `generic`, `claude`, or `gpt_reasoning`. |
| `SCINR_EXTRA_MODELS_PATHS` | `""` (empty) | Colon-separated list of extra model package paths. |

### Normalization

| Variable | Default | Description |
| :--- | :--- | :--- |
| `NORMALIZATION_ENABLED` | `false` | Enable tabular data normalization via LLM. |
| `NORMALIZATION_BATCH_SIZE` | `5` | Batch size for normalization LLM calls. |

---

## Programmatic Configuration

The `configure()` function is the primary way to set up scinr at runtime. It accepts keyword arguments organized by category. All parameters are optional — omitting a parameter falls back to the environment variable or hard-coded default.

```python
from scinr.newton import configure
```

### LLM Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `llm` | `Any \| None` | Pre-constructed LLM client instance. When provided, bypasses `MODEL_ID` and AWS Bedrock auto-configuration. |
| `repair_llm` | `Any \| None` | Separate LLM client for repair/retry operations. Falls back to `llm` if not provided. |

### Neo4j Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `neo4j_uri` | `str \| None` | Bolt URI for the Neo4j instance (e.g. `bolt://localhost:7687`). |
| `neo4j_user` | `str \| None` | Neo4j username. |
| `neo4j_password` | `str \| None` | Neo4j password. |

### Models / Themes Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `enabled_base_themes` | `list[ThemePath \| str] \| None` | List of base themes to enable for extraction. |
| `enabled_user_themes` | `list[str] \| None` | List of user-defined themes to enable. |
| `extra_models_paths` | `list[str \| Path] \| None` | Additional paths to model packages. |

### Storage Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `storage_backend` | `Literal["none", "mongodb", "custom"] \| None` | Storage backend type. `none` = no persistence, `mongodb` = MongoDB, `custom` = user-provided storage. |
| `mongodb_uri` | `str \| None` | MongoDB connection string. |
| `mongodb_database` | `str \| None` | MongoDB database name. |
| `mongodb_raw_files_collection` | `str \| None` | Collection for raw file metadata. |
| `mongodb_pages_collection` | `str \| None` | Collection for converted pages. |
| `mongodb_gridfs_bucket` | `str \| None` | GridFS bucket for binary storage. |
| `custom_storage` | `tuple \| None` | Custom storage backend tuple (driver, connection). |

### Converter Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `extra_converters` | `dict[str, type] \| None` | Dictionary mapping file extensions to converter classes. |

### PDF / Mistral OCR Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `mistral_api_key` | `str \| None` | Mistral API key for PDF OCR. |
| `mistral_ocr_safe_max_pages` | `int \| None` | Max pages before OCR is required. |
| `mistral_ocr_safe_max_bytes` | `int \| None` | Max file size (bytes) before OCR is required. |
| `mistral_ocr_max_retries` | `int \| None` | OCR retry count. |
| `mistral_ocr_retry_backoff_seconds` | `float \| None` | Retry backoff in seconds. |
| `mistral_ocr_chunk_concurrency` | `int \| None` | Concurrent OCR chunk processing. |
| `mistral_ocr_error_strategy` | `Literal["fail_fast", "best_effort"] \| None` | OCR error handling strategy. |

### Pipeline Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `prompt_caching_enabled` | `bool \| None` | Enable prompt caching (Bedrock). |
| `full_docstring` | `bool \| None` | Use the full class docstring (vs. only its first line) when building the model catalog description for LLM prompts (annotation stage) and Neo4j `CatalogModel.description`. |
| `extraction_batch_size` | `int \| None` | Pages per extraction chunk. |
| `llm_concurrency` | `int \| None` | Maximum concurrent LLM calls. |
| `neo4j_concurrency` | `int \| None` | Maximum async Neo4j concurrency. |
| `neo4j_sync_concurrency` | `int \| None` | Maximum sync Neo4j concurrency. |

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

### Minimal Setup (Environment Variables Only)

The simplest approach: set environment variables and call `configure()` to let scinr pick them up automatically. `configure()` always reads `.env` via `python-dotenv`, so you never need to import dotenv manually.

```bash
# .env file
MODEL_ID=us.anthropic.claude-sonnet-4-6
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
MISTRAL_API_KEY=your_mistral_key
```

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def main():
    # configure() reads .env automatically — no arguments needed
    configure()

    result = await run_pipeline(input_raw="./raw_docs")
    print(f"Pipeline: {'success' if result.success else 'failed'}")

asyncio.run(main())
```

> **Note:** `configure()` is always required before calling `run_pipeline()`. Even when all values come from environment variables, you must call `configure()` to resolve and validate the configuration.

### Full AWS Bedrock Setup

Complete programmatic configuration for a production Bedrock deployment.

```python
from scinr.newton import configure

configure(
    # LLM — AWS Bedrock
    llm=None,  # let scinr auto-create from MODEL_ID env var
    repair_llm=None,  # use same model for repairs

    # Neo4j
    neo4j_uri="bolt://neo4j.internal:7687",
    neo4j_user="scinr_ingest",
    neo4j_password="secure_password",
    neo4j_concurrency=10,
    neo4j_sync_concurrency=8,

    # PDF / Mistral OCR
    mistral_api_key="your_mistral_key",
    mistral_ocr_safe_max_pages=900,
    mistral_ocr_safe_max_bytes=47185920,
    mistral_ocr_max_retries=3,
    mistral_ocr_error_strategy="best_effort",

    # Pipeline
    prompt_caching_enabled=True,
    extraction_batch_size=1,
    llm_concurrency=4,
    prompt_family="claude",

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

### Custom LLM Client

Bring your own LLM client instance (e.g., a custom wrapper or non-Bedrock provider).

```python
from scinr.newton import configure

# Construct your own LLM client
my_llm = build_my_custom_llm()

configure(
    llm=my_llm,
    repair_llm=my_llm,  # reuse same client for repairs
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password",
    prompt_family="generic",
)
```

---

## Using a `.env` File

scinr reads standard `.env` files. Copy the provided example and fill in your values:

```bash
# Copy the template
cp .env.example .env

# Edit with your values
# $EDITOR .env
```

The `.env.example` file is provided in the project root and contains all available settings with helpful comments. Key notes:

- **`NEO4J_USER`** — previous versions used `NEO4J_USERNAME`. The variable was renamed. If you have an old `.env`, update it.
- **`LLM_CONCURRENCY`** — previously named `BEDROCK_CONCURRENCY`. The variable was renamed for provider-agnostic naming.
- Values in `.env` are overridden by any explicit arguments to `configure()`.

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

The `ScinrConfig` object is immutable after creation. To change configuration, call `configure()` again with the new values — it will produce a new `ScinrConfig` that replaces the previous one.

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

## Complete Reference: All Settings

For quick lookup, here is every configurable setting with its resolution chain:

| Setting | `configure()` param | Environment Variable | Default |
| :--- | :--- | :--- | :--- |
| **LLM Client** | `llm` | *(none)* | `None` |
| **Repair LLM Client** | `repair_llm` | *(none)* | `None` (falls back to `llm`) |
| **Model ID** | *(via `llm`)* | `MODEL_ID` | *(required if no `llm`)* |
| **Repair Model ID** | *(via `repair_llm`)* | `REPAIR_MODEL_ID` | Falls back to `MODEL_ID` |
| **AWS Region** | *(via `llm`)* | `AWS_DEFAULT_REGION` | `us-east-1` |
| **Max Tokens** | *(via `llm`)* | `MAX_TOKENS` | `65536` |
| **Neo4j URI** | `neo4j_uri` | `NEO4J_URI` | `bolt://localhost:7687` |
| **Neo4j User** | `neo4j_user` | `NEO4J_USER` | *(required)* |
| **Neo4j Password** | `neo4j_password` | `NEO4J_PASSWORD` | *(required)* |
| **Neo4j Auth** | *(derived)* | `NEO4J_AUTH` | Fallback format |
| **Neo4j Async Concurrency** | `neo4j_concurrency` | `NEO4J_CONCURRENCY` | `10` |
| **Neo4j Sync Concurrency** | `neo4j_sync_concurrency` | `NEO4J_SYNC_CONCURRENCY` | `8` |
| **LLM Concurrency** | `llm_concurrency` | `LLM_CONCURRENCY` | `4` |
| **Storage Backend** | `storage_backend` | `STORAGE_BACKEND` | `none` |
| **MongoDB URI** | `mongodb_uri` | `MONGODB_URI` | `mongodb://localhost:27017` |
| **MongoDB Database** | `mongodb_database` | `MONGODB_DATABASE` | `scinr` |
| **MongoDB Raw Files** | `mongodb_raw_files_collection` | `MONGODB_RAW_FILES_COLLECTION` | `raw_files` |
| **MongoDB Pages** | `mongodb_pages_collection` | `MONGODB_PAGES_COLLECTION` | `converted_pages` |
| **MongoDB GridFS** | `mongodb_gridfs_bucket` | `MONGODB_GRIDFS_BUCKET` | `raw_binaries` |
| **Custom Storage** | `custom_storage` | *(none)* | `None` |
| **Mistral API Key** | `mistral_api_key` | `MISTRAL_API_KEY` | `None` |
| **OCR Max Pages** | `mistral_ocr_safe_max_pages` | `MISTRAL_OCR_SAFE_MAX_PAGES` | `900` |
| **OCR Max Bytes** | `mistral_ocr_safe_max_bytes` | `MISTRAL_OCR_SAFE_MAX_BYTES` | `47185920` |
| **OCR Max Retries** | `mistral_ocr_max_retries` | `MISTRAL_OCR_MAX_RETRIES` | `3` |
| **OCR Backoff** | `mistral_ocr_retry_backoff_seconds` | `MISTRAL_OCR_RETRY_BACKOFF_SECONDS` | `2.0` |
| **OCR Concurrency** | `mistral_ocr_chunk_concurrency` | `MISTRAL_OCR_CHUNK_CONCURRENCY` | `1` |
| **OCR Error Strategy** | `mistral_ocr_error_strategy` | `MISTRAL_OCR_ERROR_STRATEGY` | `fail_fast` |
| **Prompt Caching** | `prompt_caching_enabled` | `PROMPT_CACHING_ENABLED` | `true` |
| **Full Docstring** | `full_docstring` | `FULL_DOCSTRING` | `true` |
| **Extraction Batch** | `extraction_batch_size` | `EXTRACTION_BATCH_SIZE` | `1` |
| **Prompt Family** | `prompt_family` | `PROMPT_FAMILY` | `generic` |
| **Extra Models Paths** | `extra_models_paths` | `SCINR_EXTRA_MODELS_PATHS` | `""` |
| **Enabled Base Themes** | `enabled_base_themes` | *(none)* | `None` |
| **Enabled User Themes** | `enabled_user_themes` | *(none)* | `None` |
| **Extra Converters** | `extra_converters` | *(none)* | `None` |
| **Normalization** | `normalization_enabled` | `NORMALIZATION_ENABLED` | `false` |
| **Normalization Batch** | `normalization_batch_size` | `NORMALIZATION_BATCH_SIZE` | `5` |
| **Normalization LLM** | `normalization_llm` | *(none)* | `None` (falls back to `llm`) |
| **Log Level** | `log_level` | *(none)* | `"INFO"` |
