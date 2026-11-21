"""
config.py — Global scinr-ingest configuration.

Call configure() once at startup to set LLM, graph backend, storage and other
pipeline parameters. All pipeline modules read settings via get_config().

Parameter resolution order for all parameters:
  1. Explicit argument passed to configure()
  2. Environment variable (via os.getenv)
  3. Hard-coded default value

Usage (library mode):
    from langchain_openai import ChatOpenAI
    from scinr.newton.config import configure
    configure(llm=ChatOpenAI(model="gpt-4o"), neo4j_user="neo4j", neo4j_password="...", neo4j_database="neo4j")

Usage (navigation only — no LLM required):
    from scinr.newton.config import configure
    from scinr.newton.navigation import graph_navigator
    configure(neo4j_user="neo4j", neo4j_password="...", neo4j_database="neo4j")

Usage (environment variables / .env file):
    configure()  # Reads everything from environment variables
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from scinr.newton.exceptions import ConfigurationError

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public type aliases (for IDE autocomplete — no runtime cost)
# ---------------------------------------------------------------------------


class PromptFamily(str, Enum):
    """Selects which prompt variant family to use for LLM calls.

    GENERIC (default): Simplified, model-agnostic prompts. Work correctly
        across all LLM families (OpenAI, Kimi, GLM, Claude, Ollama, etc.).
    CLAUDE: Prompts optimized for Claude/Sonnet. Use XML-structured instructions,
        multi-step protocols, and internal checklists that leverage Claude's
        extended reasoning capabilities.
    GPT_REASONING: Prompts for OpenAI reasoning models (GPT-5.5, o3, o4-mini).
        Use Markdown section headers and goal-based language. No step-by-step
        protocols, no auto-checklists, no XML instruction wrappers. Use GENERIC
        for non-reasoning GPT models (GPT-4o, GPT-4.1, GPT-4.5).

    To add support for a new model family in the future, add a new member here
    and create the corresponding prompt files (_newmodel.py) in each stage directory.
    """

    GENERIC = "generic"  # default — model-agnostic, all LLM families
    CLAUDE = "claude"  # XML-structured, extended reasoning protocols
    GPT_REASONING = "gpt_reasoning"  # OpenAI reasoning models (GPT-5.5, o3, o4-mini) — goal-based Markdown, no CoT elicitation; use GENERIC for non-reasoning GPT models


ThemePath = Literal[
    "default",
    "equipment_qualification",
    "pharmaceutical_quality",
    "structural_specs",
    "pharma_operations",
    "pharma_operations/product_master",
    "pharma_operations/commercial_sales",
    "pharma_operations/regulatory_portfolio",
    "pharma_operations/batch_manufacturing",
    "pharma_operations/serialization",
    "pharma_regulatory/qa",
    "pharma_regulatory/bpg",
    "pharma_regulatory/variation_guidelines",
]
"""Theme path identifiers for the built-in scinr-ingest theme library.

Use these values in ``enabled_base_themes`` to activate specific built-in themes.
Extend with plain ``str`` values for user-defined themes added via
``extra_models_paths``.

Examples::

    configure(
        llm=...,
        enabled_base_themes=["pharmaceutical_quality", "pharma_operations/batch_manufacturing"],
    )
"""

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------



#: Stages that accept their own LLM configuration (llm_stage_configuration keys).
LLM_STAGES: tuple[str, ...] = ("extraction", "annotation", "entity_extraction", "tabular")


@dataclass(frozen=True)
class StageLLMConfig:
    """LLM settings of one pipeline stage. ``None`` in a field means "use the global value"."""

    llm: Any = None  # BaseChatModel
    prompt_family: PromptFamily | str | None = None
    prompt_caching_enabled: bool | None = None


@dataclass(frozen=True)
class ResolvedStageConfig:
    """Effective settings of one stage: every field already resolved against the globals."""

    stage: str
    llm: Any  # None only when neither the stage nor the global config has an LLM
    prompt_family: PromptFamily
    prompt_caching_enabled: bool


@dataclass
class ScinrConfig:
    # LLM
    llm: Any = None  # BaseChatModel
    repair_llm: Any = None  # BaseChatModel — falls back to llm if None
    max_tokens: int = 65536
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = ""
    neo4j_password: str = ""
    neo4j_database: str = ""
    # Graph navigation
    graph_backend: str = "neo4j"  # "neo4j" | "postgres" | "sqlite"
    # Models
    enabled_base_themes: list[ThemePath | str] | None = None
    enabled_user_themes: list[str] | None = None
    extra_models_paths: list[Path] = field(default_factory=list)
    # Storage
    storage_backend: str = "none"  # "none" | "mongodb" | "postgres" | "sqlite" | "custom"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "scinr"
    mongodb_raw_files_collection: str = "raw_files"
    mongodb_pages_collection: str = "converted_pages"
    mongodb_gridfs_bucket: str = "raw_binaries"
    mongodb_ensure_indexes: bool = True
    custom_storage: tuple | None = None  # (RawFileRepository, PageRepository)
    # Freeze (document snapshots) — resolved: never None after configure()
    freeze_backend: str = "none"  # "none" | "mongodb" | "postgres" | "sqlite" | "custom"
    mongodb_frozen_collection: str = "frozen_documents"
    mongodb_frozen_gridfs_bucket: str = "frozen_snapshots"
    custom_freeze_storage: Any = None  # FreezeRepository
    # Page images (described embedded images and video frames)
    mongodb_images_collection: str = "page_images"
    mongodb_images_gridfs_bucket: str = "page_image_binaries"
    # SQL (Postgres / SQLite)
    sql_url: str = ""
    sql_schema: str = "newton"
    sql_pool_size: int = 10
    sql_max_engines: int = 64
    sql_fulltext_config: str = "simple"
    sqlite_files_dir: str = ""
    # Converters
    extra_converters: dict[str, type] = field(default_factory=dict)
    # PDF
    mistral_api_key: str | None = None
    mistral_ocr_safe_max_pages: int = 900
    mistral_ocr_safe_max_bytes: int = 45 * 1024 * 1024  # 45 MiB
    mistral_ocr_max_retries: int = 15
    mistral_ocr_retry_backoff_seconds: float = 2.0
    mistral_ocr_chunk_concurrency: int = 1  # reservado para paralelismo futuro, no usado aún
    mistral_ocr_error_strategy: Literal["fail_fast", "best_effort"] = "fail_fast"
    mistral_ocr_concurrency: int = 2  # PDFs en conversión OCR a la vez, por event loop
    mistral_ocr_image_max_pages: int = 50  # páginas por chunk cuando se piden imágenes
    # Multimodal (images, audio, video)
    vision_llm: Any = None  # BaseChatModel multimodal — NO recurre a llm
    transcriber: Any = None  # scinr.newton.media.Transcriber
    describe_embedded_images: bool = True  # solo actúa con vision_llm
    store_described_images: bool = True  # guarda el original descrito; solo con stores_images=True
    store_video_frames: bool = True  # guarda cada fotograma descrito, con su instante
    media_error_strategy: Literal["fail_fast", "best_effort"] = "best_effort"
    strip_location: bool = True
    image_max_side_px: int = 1568
    image_min_side_px: int = 64
    max_images_per_page: int = 10
    image_decode_concurrency: int = 1  # imágenes leídas y decodificadas a la vez, por event loop
    media_segment_seconds: int = 300
    media_min_segment_seconds: int = 10  # duración mínima del último tramo
    media_language: str | None = None
    max_media_duration_seconds: int | None = 14400
    video_frames_per_minute: float = 2.0
    media_concurrency: int = 1  # audios y vídeos en conversión a la vez, por event loop
    ffmpeg_path: str | None = None
    ffprobe_path: str | None = None
    # Pipeline behaviour
    prompt_caching_enabled: bool = True
    full_docstring: bool = True
    extraction_batch_size: int = 1
    verbatim_info_unit_max_chars: int = 4000
    auto_info_units_max_input_chars: int = 10000
    llm_concurrency: int = 4
    graph_concurrency: int = 10
    graph_sync_concurrency: int = 8
    # Consolidation (fast_extraction Stage 1 post-extraction merge step)
    consolidation_token_safety_margin: float = 0.75
    consolidation_max_output_tokens: int | None = None
    consolidation_max_input_tokens: int | None = 65536
    # Logging
    log_level: str = "INFO"
    # Prompt family
    prompt_family: PromptFamily = PromptFamily.GENERIC
    # Normalization
    normalization_enabled: bool = True
    normalization_batch_size: int = 5
    normalization_llm: Any = None  # BaseChatModel — falls back to llm if None
    # Per-stage LLM configuration (validated; prompt_family already a PromptFamily)
    llm_stage_configuration: dict[str, StageLLMConfig] = field(default_factory=dict)

    def __repr__(self) -> str:
        # The generated dataclass repr would print the secrets verbatim — and it
        # ends up in tracebacks that capture locals, debuggers and error reports.
        from scinr.newton.utils.redaction import MASK, redact_uri

        parts = []
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _SECRET_FIELDS and value:
                value = MASK
            elif f.name in _URI_FIELDS:
                value = redact_uri(value)
            parts.append(f"{f.name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    @property
    def neo4j_concurrency(self) -> int:
        """Deprecated alias of ``graph_concurrency`` (removed in 0.7.0)."""
        warnings.warn(
            "neo4j_concurrency is deprecated, use graph_concurrency",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.graph_concurrency

    @property
    def neo4j_sync_concurrency(self) -> int:
        """Deprecated alias of ``graph_sync_concurrency`` (removed in 0.7.0)."""
        warnings.warn(
            "neo4j_sync_concurrency is deprecated, use graph_sync_concurrency",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.graph_sync_concurrency


class _Unset:
    """Sentinel type: distinguishes "argument not passed" from an explicit ``None``."""

    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


_SECRET_FIELDS = frozenset({"neo4j_password", "mistral_api_key"})
_URI_FIELDS = frozenset({"neo4j_uri", "mongodb_uri", "sql_url"})

#: Values accepted by the three backend axes.
_GRAPH_BACKENDS = ("neo4j", "postgres", "sqlite")
_STORAGE_BACKENDS = ("none", "mongodb", "postgres", "sqlite", "custom")
_SQL_BACKENDS = ("postgres", "sqlite")
_SQL_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: URL scheme (before any ``+driver``) -> SQL backend name.
_SQL_URL_SCHEMES = {"postgresql": "postgres", "postgres": "postgres", "sqlite": "sqlite"}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_config: ScinrConfig | None = None

#: Configuration that replaces the global one in the current context (see use_config()).
_config_override: ContextVar[ScinrConfig | None] = ContextVar(
    "scinr_config_override", default=None
)


@contextmanager
def use_config(cfg: ScinrConfig) -> Iterator[ScinrConfig]:
    """Use *cfg* instead of the global configuration in this context (task / thread copy).

    Overrides only what :func:`get_config` returns: it triggers none of the side effects of
    :func:`configure`. In particular, ``extra_converters`` of *cfg* are not applied (only
    ``configure()`` does), and the process-wide caches are not reset: clients and engines are
    cached per connection parameters, so a different configuration gets its own.
    """
    token = _config_override.set(cfg)
    try:
        yield cfg
    finally:
        _config_override.reset(token)


def get_config() -> ScinrConfig:
    """
    Return the active ScinrConfig: the one set by use_config() in this context, else the
    global one set by configure().

    Raises:
        ConfigurationError: If neither use_config() nor configure() has set one.
    """
    override = _config_override.get()
    if override is not None:
        return override
    global _config
    if _config is None:
        raise ConfigurationError(
            "scinr-ingest has not been configured yet.\n"
            "Call configure() before using any pipeline function:\n"
            "\n"
            "  from scinr.newton import configure\n"
            "  configure(llm=your_llm, neo4j_uri=..., neo4j_user=..., neo4j_password=..., neo4j_database=...)"
        )
    return _config  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------


def build_config(
    # LLM
    llm: Any | None = None,
    repair_llm: Any | None = None,
    # Neo4j
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
    # Graph navigation
    graph_backend: Literal["neo4j", "postgres", "sqlite"] | None = None,
    # Models
    enabled_base_themes: list[ThemePath | str] | None = None,
    enabled_user_themes: list[str] | None = None,
    extra_models_paths: list[str | Path] | None = None,
    # Storage
    storage_backend: Literal["none", "mongodb", "postgres", "sqlite", "custom"] | None = None,
    mongodb_uri: str | None = None,
    mongodb_database: str | None = None,
    mongodb_raw_files_collection: str | None = None,
    mongodb_pages_collection: str | None = None,
    mongodb_gridfs_bucket: str | None = None,
    mongodb_ensure_indexes: bool | None = None,
    custom_storage: tuple | None = None,
    # Freeze (document snapshots)
    freeze_backend: Literal["none", "mongodb", "postgres", "sqlite", "custom"] | None = None,
    mongodb_frozen_collection: str | None = None,
    mongodb_frozen_gridfs_bucket: str | None = None,
    custom_freeze_storage: Any | None = None,
    # Page images (described embedded images and video frames)
    mongodb_images_collection: str | None = None,
    mongodb_images_gridfs_bucket: str | None = None,
    # SQL (Postgres / SQLite)
    sql_url: str | None = None,
    sql_schema: str | None = None,
    sql_pool_size: int | None = None,
    sql_max_engines: int | None = None,
    sql_fulltext_config: str | None = None,
    sqlite_files_dir: str | None = None,
    # Converters
    extra_converters: dict[str, type] | None = None,
    # PDF
    mistral_api_key: str | None = None,
    mistral_ocr_safe_max_pages: int | None = None,
    mistral_ocr_safe_max_bytes: int | None = None,
    mistral_ocr_max_retries: int | None = None,
    mistral_ocr_retry_backoff_seconds: float | None = None,
    mistral_ocr_chunk_concurrency: int | None = None,
    mistral_ocr_error_strategy: Literal["fail_fast", "best_effort"] | None = None,
    mistral_ocr_concurrency: int | None = None,
    mistral_ocr_image_max_pages: int | None = None,
    # Multimodal (images, audio, video)
    vision_llm: Any | None = None,
    transcriber: Any | None = None,
    describe_embedded_images: bool | None = None,
    store_described_images: bool | None = None,
    store_video_frames: bool | None = None,
    media_error_strategy: Literal["fail_fast", "best_effort"] | None = None,
    strip_location: bool | None = None,
    image_max_side_px: int | None = None,
    image_min_side_px: int | None = None,
    max_images_per_page: int | None = None,
    image_decode_concurrency: int | None = None,
    media_segment_seconds: int | None = None,
    media_min_segment_seconds: int | None = None,
    media_language: str | None = None,
    max_media_duration_seconds: int | None | _Unset = _UNSET,
    video_frames_per_minute: float | None = None,
    media_concurrency: int | None = None,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    # Pipeline behaviour
    prompt_caching_enabled: bool | None = None,
    full_docstring: bool | None = None,
    extraction_batch_size: int | None = None,
    llm_concurrency: int | None = None,
    graph_concurrency: int | None = None,
    graph_sync_concurrency: int | None = None,
    neo4j_concurrency: int | None = None,
    neo4j_sync_concurrency: int | None = None,
    # Consolidation (fast_extraction Stage 1 post-extraction merge step)
    consolidation_token_safety_margin: float | None = None,
    consolidation_max_output_tokens: int | None = None,
    consolidation_max_input_tokens: int | None = None,
    # Logging
    log_level: str = "INFO",
    # Prompt family
    prompt_family: PromptFamily | Literal["generic", "claude", "gpt_reasoning"] | None = None,
    # Normalization
    normalization_enabled: bool | None = None,
    normalization_batch_size: int | None = None,
    normalization_llm: Any | None = None,
    llm_stage_configuration: dict[str, StageLLMConfig] | None = None,
    verbatim_info_unit_max_chars: int | None = None,
    auto_info_units_max_input_chars: int | None = None,
) -> ScinrConfig:
    """
    Build and validate a ScinrConfig without installing it.

    Takes exactly the parameters of :func:`configure` and resolves them in the same order
    (explicit argument > environment variable > default), but assigns no global state and runs
    no side effects: it does not touch the process-wide configuration, the logging setup, the
    converter registry or any client cache. Use it with :func:`use_config` to run a block with
    another configuration. Settings that only :func:`configure` applies (``extra_converters``,
    log redaction, cache resets) are NOT applied to a configuration used through ``use_config``.

    Args:
        llm: LangChain BaseChatModel instance for LLM-dependent stages. Optional — not required
            for graph navigation. When omitted and MODEL_ID is unset, scinr runs without an LLM.
        repair_llm: LangChain BaseChatModel for the JSON repair loop. Falls back to `llm` if None;
            stays None when no LLM is configured (navigation-only mode).
        neo4j_uri: Neo4j connection URI. Env: `NEO4J_URI`. Default: `bolt://localhost:7687`.
        neo4j_user: Neo4j username. Env: `NEO4J_USER`. Required when `graph_backend='neo4j'` (default).
        neo4j_password: Neo4j password. Env: `NEO4J_PASSWORD`. Required when `graph_backend='neo4j'` (default).
        neo4j_database: Neo4j target database name. Env: `NEO4J_DATABASE`. Required when
            `graph_backend='neo4j'` (default).
        graph_backend: Backend for the graph (write path and the read-only navigation API
            `scinr.newton.navigation`): `'neo4j'` (default), `'postgres'` or `'sqlite'`.
            Env: `GRAPH_BACKEND`. The Neo4j credentials are required only for `'neo4j'`.
        enabled_base_themes: Whitelist of built-in theme paths to activate (`ThemePath` values).
        enabled_user_themes: Whitelist of user theme paths to activate.
        extra_models_paths: Filesystem paths to scan for additional user-defined theme models.
        storage_backend: Storage type: `'none'` (default), `'mongodb'`, `'postgres'`,
            `'sqlite'` or `'custom'`. Env: `STORAGE_BACKEND`.
        mongodb_uri: MongoDB connection URI. Env: `MONGODB_URI`.
        mongodb_database: MongoDB database name. Env: `MONGODB_DATABASE`.
        mongodb_raw_files_collection: Collection for raw file metadata.
        mongodb_pages_collection: Collection for converted pages.
        mongodb_gridfs_bucket: GridFS bucket name for binary files.
        mongodb_ensure_indexes: Create the MongoDB indexes automatically the first time
            `get_storage()` is used in the process. Set to `False` when the database user
            lacks the `createIndex` privilege and indexes are managed by operations.
            Env: `MONGODB_ENSURE_INDEXES`. Default: `True`.
        custom_storage: Tuple `(RawFileRepository, PageRepository)` when `storage_backend='custom'`.
        freeze_backend: Backend for document snapshots (`freeze_document()`,
            `restore_document()`, `export_document_snapshot(destination='storage')`):
            `'none'`, `'mongodb'`, `'postgres'`, `'sqlite'` or `'custom'`. Resolution: this argument, else env
            `FREEZE_BACKEND`, else **inherits the resolved `storage_backend`**. Leaving it
            `None` (not specified) inherits; passing `'none'` explicitly disables freezing
            even when `storage_backend='mongodb'`. Reuses `mongodb_uri` / `mongodb_database`.
        mongodb_frozen_collection: Collection for snapshot metadata.
            Env: `MONGODB_FROZEN_COLLECTION`. Default: `'frozen_documents'`.
        mongodb_frozen_gridfs_bucket: GridFS bucket for snapshot files.
            Env: `MONGODB_FROZEN_GRIDFS_BUCKET`. Default: `'frozen_snapshots'`.
        custom_freeze_storage: A `FreezeRepository` instance when `freeze_backend='custom'`.
        mongodb_images_collection: Collection for the metadata of the described
            page images and video frames. Env: `MONGODB_IMAGES_COLLECTION`.
            Default: `'page_images'`.
        mongodb_images_gridfs_bucket: GridFS bucket for the original bytes of the
            described page images and video frames. Env:
            `MONGODB_IMAGES_GRIDFS_BUCKET`. Default: `'page_image_binaries'`.
        sql_url: SQLAlchemy URL of the SQL backends, without driver:
            `postgresql://user:pass@host:5432/db` or `sqlite:///path/newton.db`. A URL that
            names a driver (`postgresql+psycopg://`) is accepted and the driver is ignored.
            Required when any axis is `'postgres'` or `'sqlite'`; all the SQL axes share it
            and must match its scheme. Env: `SQL_URL`.
        sql_schema: Postgres schema that holds the tables. Ignored by SQLite.
            Env: `SQL_SCHEMA`. Default: `'newton'`.
        sql_pool_size: Connection pool size of each SQL engine (minimum 1).
            Env: `SQL_POOL_SIZE`. Default: `10`.
        sql_max_engines: Maximum number of SQL engines kept in cache (one per distinct
            `sql_url`/`sql_schema`/pool/text-search config, and per event loop for the asynchronous
            ones). The least recently used one is disposed when the limit is exceeded (minimum 1).
            Env: `SQL_MAX_ENGINES`. Default: `64`.
        sql_fulltext_config: Postgres text-search configuration used by `to_tsvector`.
            Applied only when the schema is created. Env: `SQL_FULLTEXT_CONFIG`.
            Default: `'simple'`.
        sqlite_files_dir: Folder for the binaries kept by the SQLite backend. Empty means
            `<db path without extension>.files/` next to the `.db`. Env: `SQLITE_FILES_DIR`.
        extra_converters: Dict mapping file extensions to custom `BaseConverter` subclasses.
        mistral_api_key: Mistral API key for PDF OCR conversion.
        mistral_ocr_safe_max_pages: Maximum number of pages per PDF chunk sent to the
            Mistral OCR API before splitting it. Env: `MISTRAL_OCR_SAFE_MAX_PAGES`.
            Default: `900`.
        mistral_ocr_safe_max_bytes: Maximum number of bytes per PDF chunk (already
            serialized size) sent to the Mistral OCR API before splitting it.
            Env: `MISTRAL_OCR_SAFE_MAX_BYTES`. Default: `45 * 1024 * 1024` (45 MiB).
        mistral_ocr_max_retries: Maximum number of attempts per chunk on network or
            retryable HTTP errors. Env: `MISTRAL_OCR_MAX_RETRIES`. Default: `15`.
        mistral_ocr_retry_backoff_seconds: Base (in seconds) of the exponential backoff
            between retries. Env: `MISTRAL_OCR_RETRY_BACKOFF_SECONDS`. Default: `2.0`.
        mistral_ocr_chunk_concurrency: Reserved for future parallelism across PDF
            chunks; currently unused (processing is always sequential).
            Env: `MISTRAL_OCR_CHUNK_CONCURRENCY`. Default: `1`.
        mistral_ocr_error_strategy: Error-handling strategy when converting PDFs split
            into chunks: `'fail_fast'` (default, aborts the whole document if any
            chunk fails) or `'best_effort'` (skips the chunks that fail and continues
            with the rest). Env: `MISTRAL_OCR_ERROR_STRATEGY`.
            Default: `'fail_fast'`.
            Note: this strategy only applies to per-chunk network/API failures
            (retries exhausted, non-retryable HTTP errors) once the PDF has already
            been split into chunks. It does NOT cover the case where the initial
            split itself fails structurally (`PdfSplitError`, a single page exceeds
            `mistral_ocr_safe_max_bytes` even in isolation) — in that case the
            document always aborts, regardless of the value of
            `mistral_ocr_error_strategy`.
        mistral_ocr_concurrency: Maximum number of PDFs undergoing OCR conversion at
            the same time (from reading the PDF until the response for its last chunk
            arrives), per event loop. Bounds OCR memory independently of
            `parallel_docs`: each conversion holds at most one chunk and its
            response. Env: `MISTRAL_OCR_CONCURRENCY`. Default: `2`.
        mistral_ocr_image_max_pages: Pages per PDF chunk when embedded images are
            requested from Mistral OCR (only with `describe_embedded_images` and a
            `vision_llm`): the response with images is much heavier than a text-only
            one. The smaller of this value and `mistral_ocr_safe_max_pages` applies.
            Env: `MISTRAL_OCR_IMAGE_MAX_PAGES`. Default: `50`.
        vision_llm: Multimodal LangChain `BaseChatModel` (accepts images) used to
            describe embedded images, standalone images and video frames. It does not
            need `with_structured_output`. **It does not fall back to `llm`**: without
            it no embedded image is described, and image files fail with an error
            explaining how to configure it. Default: `None`.
        transcriber: Instance of `scinr.newton.media.Transcriber` for audio and video
            (for example `OpenAICompatibleTranscriber` or `FasterWhisperTranscriber`).
            Without it, audio files (and videos with an audio track) fail with an
            error explaining how to configure it. Default: `None`.
        describe_embedded_images: Describe the embedded images of PDF, DOCX and PPTX
            files and insert the description into the markdown of their page. **Only
            takes effect when a `vision_llm` is set**; without it no images are
            requested from Mistral and no blobs are extracted.
            Env: `DESCRIBE_EMBEDDED_IMAGES`. Default: `True`.
        store_described_images: Keep the original of each embedded image that gets
            described. Only takes effect with a persistent storage backend whose
            `stores_images` is `True`. Env: `STORE_DESCRIBED_IMAGES`. Default: `True`.
        store_video_frames: Keep each described frame, with its timestamp. For long
            videos this amounts to about `video_frames_per_minute × minutes` JPEGs.
            Env: `STORE_VIDEO_FRAMES`. Default: `True`.
        media_error_strategy: What to do when an embedded image, an audio or video
            segment or a frame fails: `'best_effort'` (default: it is skipped,
            recorded in the document's `missing_images` / `missing_segments`, and
            processing continues) or `'fail_fast'` (the document fails).
            Env: `MEDIA_ERROR_STRATEGY`.
        strip_location: With `True` (default) the GPS location of images is neither
            read nor stored. Env: `STRIP_LOCATION`.
        image_max_side_px: Longest side, in pixels, of the images sent to the
            `vision_llm` (they are downscaled before sending). Env: `IMAGE_MAX_SIDE_PX`.
            Default: `1568`.
        image_min_side_px: Embedded images whose shorter side is below this value are
            considered trivial (icons, separators) and are not described.
            Env: `IMAGE_MIN_SIDE_PX`. Default: `64`.
        max_images_per_page: Maximum number of embedded images described per page; the
            ones with the largest area are kept. Env: `MAX_IMAGES_PER_PAGE`. Default: `10`.
        image_decode_concurrency: Number of images (embedded, standalone or frames)
            read, decoded and downscaled at the same time, per event loop. Decoding is
            what costs memory (a 12 MP photo is about 36 MB of pixels): the peak grows
            with this value. Model calls remain limited by `llm_concurrency`, so
            raising it only speeds things up when the model answers faster than images
            are decoded. Minimum `1`. Env: `IMAGE_DECODE_CONCURRENCY`. Default: `1`.
        media_segment_seconds: Duration, in seconds, of each audio or video segment
            (one page per segment). Minimum `10`. Env: `MEDIA_SEGMENT_SECONDS`.
            Default: `300`.
        media_min_segment_seconds: Minimum duration of the last segment. If the final
            remainder is shorter, the previous cut is moved earlier (to the whole
            second) so that the last segment lasts at least this long: 300.05 s with
            300 s segments gives 290 s + 10.05 s. Avoids millisecond-long segments,
            which some providers reject. `0` disables it. It cannot exceed half of
            `media_segment_seconds` (if not given, it is capped to that).
            Env: `MEDIA_MIN_SEGMENT_SECONDS`. Default: `10`.
        media_language: Audio language (ISO 639-1 code, e.g. `'es'`) or `None` to let
            the transcriber detect it. Env: `MEDIA_LANGUAGE`. Default: `None`.
        max_media_duration_seconds: Maximum accepted duration for an audio or video
            file; exceeding it is a `ConversionError` raised before spending anything.
            `None` = no limit. Env: `MAX_MEDIA_DURATION_SECONDS` (`none` = no limit).
            Default: `14400` (4 h).
        video_frames_per_minute: Frames per minute described in videos (uniform
            sampling). `0` = transcription only. Env: `VIDEO_FRAMES_PER_MINUTE`.
            Default: `2.0`.
        media_concurrency: Maximum number of audio and video files being converted at
            the same time, per event loop, independently of `parallel_docs`.
            Env: `MEDIA_CONCURRENCY`. Default: `1`.
        ffmpeg_path: Path to the `ffmpeg` binary (a system requirement for audio and
            video; it is not installed with the package). `None` = look it up on the
            `PATH`. Env: `FFMPEG_PATH`.
        ffprobe_path: Path to the `ffprobe` binary (same requirements as `ffmpeg_path`).
            `None` = look it up on the `PATH`. Env: `FFPROBE_PATH`.
        prompt_caching_enabled: Enable prompt caching for supported LLM providers.
        full_docstring: Use the full class docstring (True, default) or only its first
            non-empty line (False) when building the model catalog description shown
            to the LLM during annotation and stored as `CatalogModel.description` in
            the graph database. Env: `FULL_DOCSTRING`. Default: `True`.
        extraction_batch_size: Pages per extraction chunk (default: `1`).
        verbatim_info_unit_max_chars: Maximum length, in characters, of each InfoUnit created
            in `info_units_mode="verbatim"`; longer paragraphs are split into several
            InfoUnits. Must be >= 100, otherwise `ConfigurationError` is raised.
            Env: `VERBATIM_INFO_UNIT_MAX_CHARS`. Default: `4000`.
        auto_info_units_max_input_chars: Maximum characters of paragraph text sent in one
            second-call request of `info_units_mode="auto"` (the call that writes the InfoUnits
            of a node decided as `summary`). A node with more text is split into consecutive
            groups of paragraphs, one call each; a single longer paragraph goes alone. Must be
            >= 1000, otherwise `ConfigurationError` is raised.
            Env: `AUTO_INFO_UNITS_MAX_INPUT_CHARS`. Default: `10000`.
        llm_concurrency: Max concurrent LLM calls (semaphore size, default: `4`).
        graph_concurrency: Max concurrent graph-store sessions during annotation and entity
            extraction, for any graph backend (Neo4j or SQL). Must be >= 1.
            Env: `GRAPH_CONCURRENCY`. Default: `10`.
        graph_sync_concurrency: Max concurrent Stage 2 (sync ingestion) dispatches
            to asyncio.to_thread(), for any graph backend. Must be >= 1.
            Env: `GRAPH_SYNC_CONCURRENCY`. Default: `8`.
        neo4j_concurrency: Deprecated alias of `graph_concurrency` (env `NEO4J_CONCURRENCY`);
            removed in 0.7.0.
        neo4j_sync_concurrency: Deprecated alias of `graph_sync_concurrency`
            (env `NEO4J_SYNC_CONCURRENCY`); removed in 0.7.0.
        consolidation_token_safety_margin: Fraction of `max_tokens` used as the
            output-token ceiling for the Stage 1 `fast_extraction` consolidation
            LLM call, when `consolidation_max_output_tokens` is unset. Env:
            `CONSOLIDATION_TOKEN_SAFETY_MARGIN`. Default: `0.75`.
        consolidation_max_output_tokens: Explicit output-token ceiling for the
            Stage 1 `fast_extraction` consolidation LLM call's decisions array.
            When `None` (default), derived as `max_tokens * consolidation_token_safety_margin`.
            Env: `CONSOLIDATION_MAX_OUTPUT_TOKENS`.
        consolidation_max_input_tokens: Input-token ceiling that governs the batch
            size of the Stage 1 `fast_extraction` consolidation's sliding-window
            algorithm (see `structure_consolidation.consolidate_structure()`) —
            each batch of consecutive chunks (plus its backward buffer from the
            immediately preceding batch) is kept under this ceiling. Default:
            `65536` (64k). Passing `None` explicitly to `ScinrConfig` directly
            (bypassing `configure()`) skips the ceiling check entirely as a
            defensive fallback — not the normal path. Env: `CONSOLIDATION_MAX_INPUT_TOKENS`.
        log_level: Logging level string (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`).
        prompt_family: Prompt family to use (`"generic"`, `"claude"`, or `"gpt_reasoning"`).
        normalization_enabled: Enable post-extraction normalization for tabular data.
        normalization_batch_size: Max normalization entries per LLM batch (default: `5`).
        normalization_llm: Dedicated LLM model instance for tabular normalization.
        llm_stage_configuration: Per-stage LLM settings, as a dict of stage name ->
            `StageLLMConfig`. Valid stages: `"extraction"`, `"annotation"`,
            `"entity_extraction"`, `"tabular"`. Each `StageLLMConfig` field left as
            `None` takes the global value (`llm`, `prompt_family`,
            `prompt_caching_enabled`). The structure consolidation of
            `fast_extraction` uses the `"extraction"` stage. `repair_llm`,
            `normalization_llm`, `vision_llm` and `transcriber` are not affected.
            Not read from environment variables. Example::

                configure(
                    llm=ChatBedrockConverse(model="us.anthropic.claude-sonnet-4-6"),
                    llm_stage_configuration={
                        "extraction": StageLLMConfig(
                            llm=ChatOpenAI(model="gpt-5.5"),
                            prompt_family="gpt_reasoning",
                        ),
                        "annotation": StageLLMConfig(prompt_family="claude"),
                    },
                )

    Returns:
        The validated ScinrConfig (not installed as the global configuration).

    Raises:
        ConfigurationError: If conflicting settings or invalid URIs are supplied.
    """
    # Resolved early (moved up from its original position below) because the
    # Bedrock client construction just below needs it to size its connection
    # pool (max_pool_connections) relative to the configured LLM concurrency.
    _concurrency_env = os.getenv("LLM_CONCURRENCY", "4")
    resolved_concurrency = llm_concurrency if llm_concurrency is not None else int(_concurrency_env)

    # ── LLM ──────────────────────────────────────────────────────────────────
    resolved_llm = llm
    if resolved_llm is None:
        model_id = os.getenv("MODEL_ID")
        if model_id:
            try:
                from botocore.config import Config as BotocoreConfig
                from langchain_aws import ChatBedrockConverse
            except ImportError as exc:
                raise ConfigurationError(
                    "MODEL_ID is set but langchain-aws is not installed.\n"
                    "Install the Bedrock extra: pip install 'scinr-ingest[bedrock]'"
                ) from exc
            resolved_llm = ChatBedrockConverse(
                model=model_id,
                region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                max_tokens=int(os.getenv("MAX_TOKENS", "65536")),
                temperature=0,
                config=BotocoreConfig(max_pool_connections=max(resolved_concurrency * 2, 20)),
            )
        else:
            log.info(
                "No LLM configured. scinr will run without an LLM "
                "(navigation-only mode). LLM-dependent stages (extraction, "
                "annotation, entity extraction, tabular mapping/normalization) "
                "will raise a ConfigurationError until an LLM is supplied via "
                "configure(llm=...) or the MODEL_ID env var."
            )

    if resolved_llm is not None:
        _validate_llm(resolved_llm)

    resolved_repair_llm = repair_llm
    if resolved_repair_llm is None:
        if resolved_llm is not None:
            log.warning(
                "Configure -- Specific Repair LLM has not been defined, fallback to main LLM. It is recommended to use a smaller or cheaper model for reparation steps."
            )
        resolved_repair_llm = resolved_llm  # may be None (navigation-only)

    # ── Graph backend ─────────────────────────────────────────────────────────
    # Resolved first: the Neo4j credentials are only required for graph_backend="neo4j".
    resolved_graph_backend = graph_backend or os.getenv("GRAPH_BACKEND", "neo4j")
    if resolved_graph_backend not in _GRAPH_BACKENDS:
        raise ConfigurationError(
            f"Unknown graph_backend: {resolved_graph_backend!r}. "
            f"Valid values: 'neo4j', 'postgres', 'sqlite'."
        )

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    resolved_neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    resolved_neo4j_user = neo4j_user or os.getenv("NEO4J_USER")
    resolved_neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
    resolved_neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE")
    # Parse NEO4J_AUTH fallback ("user/password")
    if not resolved_neo4j_user or not resolved_neo4j_password:
        auth_raw = os.getenv("NEO4J_AUTH")
        if auth_raw and "/" in auth_raw:
            parts = auth_raw.split("/", maxsplit=1)
            resolved_neo4j_user = resolved_neo4j_user or parts[0]
            resolved_neo4j_password = resolved_neo4j_password or parts[1]

    if resolved_graph_backend == "neo4j":
        if not resolved_neo4j_user:
            raise ConfigurationError(
                "Neo4j username is not configured. Options:\n"
                "  - Set NEO4J_USER=neo4j in your .env file\n"
                "  - Pass neo4j_user='neo4j' to configure()\n"
                "  - Set NEO4J_AUTH=neo4j/password in your .env file"
            )
        if not resolved_neo4j_password:
            raise ConfigurationError(
                "Neo4j password is not configured. Options:\n"
                "  - Set NEO4J_PASSWORD=your_password in your .env file\n"
                "  - Pass neo4j_password='...' to configure()\n"
                "  - Set NEO4J_AUTH=neo4j/password in your .env file"
            )
        if not resolved_neo4j_database:
            raise ConfigurationError(
                "Neo4j database is not configured. Options:\n"
                "  - Set NEO4J_DATABASE=neo4j in your .env file\n"
                "  - Pass neo4j_database='neo4j' to configure()"
            )
    else:
        resolved_neo4j_user = resolved_neo4j_user or ""
        resolved_neo4j_password = resolved_neo4j_password or ""
        resolved_neo4j_database = resolved_neo4j_database or ""

    # ── Storage ───────────────────────────────────────────────────────────────
    resolved_storage_backend = storage_backend or os.getenv("STORAGE_BACKEND", "none")
    if resolved_storage_backend not in _STORAGE_BACKENDS:
        raise ConfigurationError(
            f"Unknown storage_backend: {resolved_storage_backend!r}. "
            f"Valid values: 'none', 'mongodb', 'postgres', 'sqlite', 'custom'."
        )

    # ── Freeze ────────────────────────────────────────────────────────────────
    # No default of its own: when neither the argument nor FREEZE_BACKEND is
    # set, freezing inherits the resolved storage backend. An explicit 'none'
    # is kept (disables freezing without disabling raw-file storage).
    resolved_freeze_backend = (
        freeze_backend or os.getenv("FREEZE_BACKEND") or resolved_storage_backend
    )
    if resolved_freeze_backend not in _STORAGE_BACKENDS:
        raise ConfigurationError(
            f"Unknown freeze_backend: {resolved_freeze_backend!r}. "
            f"Valid values: 'none', 'mongodb', 'postgres', 'sqlite', 'custom'."
        )

    # ── SQL (Postgres / SQLite) ───────────────────────────────────────────────
    resolved_sql_url = sql_url or os.getenv("SQL_URL", "")
    resolved_sql_schema = sql_schema or os.getenv("SQL_SCHEMA", "newton")
    resolved_sql_pool_size = _resolve_int(
        sql_pool_size, "SQL_POOL_SIZE", 10, "sql_pool_size", minimum=1
    )
    resolved_sql_max_engines = _resolve_int(
        sql_max_engines, "SQL_MAX_ENGINES", 64, "sql_max_engines", minimum=1
    )
    resolved_sql_fulltext_config = sql_fulltext_config or os.getenv("SQL_FULLTEXT_CONFIG", "simple")
    resolved_sqlite_files_dir = sqlite_files_dir or os.getenv("SQLITE_FILES_DIR", "")
    _validate_sql_config(
        sql_url=resolved_sql_url,
        sql_schema=resolved_sql_schema,
        sql_fulltext_config=resolved_sql_fulltext_config,
        axes={
            "graph_backend": resolved_graph_backend,
            "storage_backend": resolved_storage_backend,
            "freeze_backend": resolved_freeze_backend,
        },
    )

    # ── PDF / Mistral OCR chunking ───────────────────────────────────────────
    resolved_mistral_ocr_safe_max_pages = (
        mistral_ocr_safe_max_pages
        if mistral_ocr_safe_max_pages is not None
        else int(os.getenv("MISTRAL_OCR_SAFE_MAX_PAGES", "900"))
    )
    if resolved_mistral_ocr_safe_max_pages < 1:
        raise ConfigurationError(
            f"mistral_ocr_safe_max_pages must be >= 1. "
            f"Received: {resolved_mistral_ocr_safe_max_pages!r}."
        )
    resolved_mistral_ocr_safe_max_bytes = (
        mistral_ocr_safe_max_bytes
        if mistral_ocr_safe_max_bytes is not None
        else int(os.getenv("MISTRAL_OCR_SAFE_MAX_BYTES", str(45 * 1024 * 1024)))
    )
    if resolved_mistral_ocr_safe_max_bytes < 1:
        raise ConfigurationError(
            f"mistral_ocr_safe_max_bytes must be >= 1. "
            f"Received: {resolved_mistral_ocr_safe_max_bytes!r}."
        )
    resolved_mistral_ocr_max_retries = (
        mistral_ocr_max_retries
        if mistral_ocr_max_retries is not None
        else int(os.getenv("MISTRAL_OCR_MAX_RETRIES", "15"))
    )
    if resolved_mistral_ocr_max_retries < 1:
        raise ConfigurationError(
            f"mistral_ocr_max_retries must be >= 1. Received: {resolved_mistral_ocr_max_retries!r}."
        )
    resolved_mistral_ocr_retry_backoff_seconds = (
        mistral_ocr_retry_backoff_seconds
        if mistral_ocr_retry_backoff_seconds is not None
        else float(os.getenv("MISTRAL_OCR_RETRY_BACKOFF_SECONDS", "2.0"))
    )
    resolved_mistral_ocr_chunk_concurrency = (
        mistral_ocr_chunk_concurrency
        if mistral_ocr_chunk_concurrency is not None
        else int(os.getenv("MISTRAL_OCR_CHUNK_CONCURRENCY", "1"))
    )
    resolved_mistral_ocr_error_strategy = mistral_ocr_error_strategy or os.getenv(
        "MISTRAL_OCR_ERROR_STRATEGY", "fail_fast"
    )
    if resolved_mistral_ocr_error_strategy not in ("fail_fast", "best_effort"):
        raise ConfigurationError(
            f"Unknown mistral_ocr_error_strategy: {resolved_mistral_ocr_error_strategy!r}. "
            f"Valid values: 'fail_fast', 'best_effort'."
        )
    resolved_mistral_ocr_concurrency = (
        mistral_ocr_concurrency
        if mistral_ocr_concurrency is not None
        else int(os.getenv("MISTRAL_OCR_CONCURRENCY", "2"))
    )
    if resolved_mistral_ocr_concurrency < 1:
        raise ConfigurationError(
            f"mistral_ocr_concurrency must be >= 1. Received: {resolved_mistral_ocr_concurrency!r}."
        )

    resolved_mistral_ocr_image_max_pages = _resolve_int(
        mistral_ocr_image_max_pages,
        "MISTRAL_OCR_IMAGE_MAX_PAGES",
        50,
        "mistral_ocr_image_max_pages",
        minimum=1,
    )

    # ── Multimodal (images, audio, video) ─────────────────────────────────────
    if vision_llm is not None:
        _validate_vision_llm(vision_llm)
    if transcriber is not None:
        from scinr.newton.media.transcriber import Transcriber

        if not isinstance(transcriber, Transcriber):
            raise ConfigurationError(
                f"'transcriber' must be a scinr.newton.media.Transcriber instance. "
                f"Received: {type(transcriber).__name__}.\n"
                f"Example:\n"
                f"  from scinr.newton.media.transcribers import OpenAICompatibleTranscriber\n"
                f"  configure(transcriber=OpenAICompatibleTranscriber("
                f"model='whisper-1', api_key=...))"
            )
    resolved_describe_embedded_images = _resolve_bool(
        describe_embedded_images, "DESCRIBE_EMBEDDED_IMAGES", True
    )
    resolved_store_described_images = _resolve_bool(
        store_described_images, "STORE_DESCRIBED_IMAGES", True
    )
    resolved_store_video_frames = _resolve_bool(store_video_frames, "STORE_VIDEO_FRAMES", True)
    resolved_media_error_strategy = media_error_strategy or os.getenv(
        "MEDIA_ERROR_STRATEGY", "best_effort"
    )
    if resolved_media_error_strategy not in ("fail_fast", "best_effort"):
        raise ConfigurationError(
            f"Unknown media_error_strategy: {resolved_media_error_strategy!r}. "
            f"Valid values: 'fail_fast', 'best_effort'."
        )
    resolved_strip_location = _resolve_bool(strip_location, "STRIP_LOCATION", True)
    resolved_image_max_side_px = _resolve_int(
        image_max_side_px, "IMAGE_MAX_SIDE_PX", 1568, "image_max_side_px", minimum=64
    )
    resolved_image_min_side_px = _resolve_int(
        image_min_side_px, "IMAGE_MIN_SIDE_PX", 64, "image_min_side_px", minimum=0
    )
    resolved_max_images_per_page = _resolve_int(
        max_images_per_page, "MAX_IMAGES_PER_PAGE", 10, "max_images_per_page", minimum=1
    )
    resolved_image_decode_concurrency = _resolve_int(
        image_decode_concurrency,
        "IMAGE_DECODE_CONCURRENCY",
        1,
        "image_decode_concurrency",
        minimum=1,
    )
    resolved_media_segment_seconds = _resolve_int(
        media_segment_seconds,
        "MEDIA_SEGMENT_SECONDS",
        300,
        "media_segment_seconds",
        minimum=10,
    )
    _min_segment_explicit = (
        media_min_segment_seconds is not None or os.getenv("MEDIA_MIN_SEGMENT_SECONDS") is not None
    )
    resolved_media_min_segment_seconds = _resolve_int(
        media_min_segment_seconds,
        "MEDIA_MIN_SEGMENT_SECONDS",
        10,
        "media_min_segment_seconds",
        minimum=0,
    )
    _half_segment = resolved_media_segment_seconds // 2
    if resolved_media_min_segment_seconds > _half_segment:
        if _min_segment_explicit:
            raise ConfigurationError(
                f"media_min_segment_seconds must be at most half of media_segment_seconds "
                f"({resolved_media_segment_seconds} // 2 = {_half_segment}). "
                f"Received: {resolved_media_min_segment_seconds!r}."
            )
        resolved_media_min_segment_seconds = _half_segment
    resolved_media_language = media_language or os.getenv("MEDIA_LANGUAGE") or None
    if resolved_media_language is not None:
        resolved_media_language = resolved_media_language.strip().lower()
        if not (len(resolved_media_language) == 2 and resolved_media_language.isalpha()):
            raise ConfigurationError(
                f"media_language must be an ISO 639-1 code (two letters, e.g. 'es') or "
                f"None for automatic detection. Received: {media_language!r}."
            )
    if isinstance(max_media_duration_seconds, _Unset):
        _env_max_duration = os.getenv("MAX_MEDIA_DURATION_SECONDS")
        if _env_max_duration is None:
            resolved_max_media_duration_seconds: int | None = 14400
        elif _env_max_duration.strip().lower() in ("", "none"):
            resolved_max_media_duration_seconds = None
        else:
            resolved_max_media_duration_seconds = _parse_int(
                _env_max_duration, "MAX_MEDIA_DURATION_SECONDS"
            )
    else:
        resolved_max_media_duration_seconds = max_media_duration_seconds
    if resolved_max_media_duration_seconds is not None and resolved_max_media_duration_seconds < 1:
        raise ConfigurationError(
            f"max_media_duration_seconds must be >= 1 or None (no limit). "
            f"Received: {resolved_max_media_duration_seconds!r}."
        )
    if video_frames_per_minute is not None:
        resolved_video_frames_per_minute = float(video_frames_per_minute)
    else:
        _env_fpm = os.getenv("VIDEO_FRAMES_PER_MINUTE", "2.0")
        try:
            resolved_video_frames_per_minute = float(_env_fpm)
        except ValueError:
            raise ConfigurationError(
                f"VIDEO_FRAMES_PER_MINUTE must be a number. Received: {_env_fpm!r}."
            ) from None
    if resolved_video_frames_per_minute < 0:
        raise ConfigurationError(
            f"video_frames_per_minute must be >= 0 (0 = transcription only). "
            f"Received: {resolved_video_frames_per_minute!r}."
        )
    resolved_media_concurrency = _resolve_int(
        media_concurrency, "MEDIA_CONCURRENCY", 1, "media_concurrency", minimum=1
    )
    resolved_ffmpeg_path = ffmpeg_path or os.getenv("FFMPEG_PATH") or None
    resolved_ffprobe_path = ffprobe_path or os.getenv("FFPROBE_PATH") or None
    if resolved_describe_embedded_images and vision_llm is None:
        log.info(
            "describe_embedded_images is enabled but no vision_llm is configured: "
            "embedded images in PDF, DOCX and PPTX will not be described. "
            "Pass configure(vision_llm=...) with a multimodal chat model to enable it."
        )

    # ── Pipeline behaviour ────────────────────────────────────────────────────
    resolved_caching = (
        prompt_caching_enabled
        if prompt_caching_enabled is not None
        else os.getenv("PROMPT_CACHING_ENABLED", "true").lower() == "true"
    )
    resolved_full_docstring = (
        full_docstring
        if full_docstring is not None
        else os.getenv("FULL_DOCSTRING", "true").lower() == "true"
    )
    resolved_batch_size = (
        extraction_batch_size
        if extraction_batch_size is not None
        else int(os.getenv("EXTRACTION_BATCH_SIZE", "1"))
    )
    resolved_verbatim_max_chars = (
        verbatim_info_unit_max_chars
        if verbatim_info_unit_max_chars is not None
        else int(os.getenv("VERBATIM_INFO_UNIT_MAX_CHARS", "4000"))
    )
    if resolved_verbatim_max_chars < 100:
        raise ConfigurationError(
            f"verbatim_info_unit_max_chars must be >= 100, got {resolved_verbatim_max_chars}."
        )
    resolved_auto_max_chars = (
        auto_info_units_max_input_chars
        if auto_info_units_max_input_chars is not None
        else int(os.getenv("AUTO_INFO_UNITS_MAX_INPUT_CHARS", "10000"))
    )
    if resolved_auto_max_chars < 1000:
        raise ConfigurationError(
            f"auto_info_units_max_input_chars must be >= 1000, got {resolved_auto_max_chars}."
        )
    # resolved_concurrency is computed earlier (see "── LLM" section above),
    # before the Bedrock client is constructed, so it's available for the
    # botocore connection pool sizing there too.

    resolved_graph_concurrency = _resolve_graph_concurrency(
        graph_concurrency,
        neo4j_concurrency,
        env="GRAPH_CONCURRENCY",
        old_env="NEO4J_CONCURRENCY",
        default=10,
        name="graph_concurrency",
        old_name="neo4j_concurrency",
    )
    resolved_graph_sync_concurrency = _resolve_graph_concurrency(
        graph_sync_concurrency,
        neo4j_sync_concurrency,
        env="GRAPH_SYNC_CONCURRENCY",
        old_env="NEO4J_SYNC_CONCURRENCY",
        default=8,
        name="graph_sync_concurrency",
        old_name="neo4j_sync_concurrency",
    )

    # ── Consolidation (fast_extraction Stage 1 post-extraction merge step) ────
    resolved_consolidation_token_safety_margin = (
        consolidation_token_safety_margin
        if consolidation_token_safety_margin is not None
        else float(os.getenv("CONSOLIDATION_TOKEN_SAFETY_MARGIN", "0.75"))
    )
    _consolidation_max_output_tokens_env = os.getenv("CONSOLIDATION_MAX_OUTPUT_TOKENS")
    resolved_consolidation_max_output_tokens = (
        consolidation_max_output_tokens
        if consolidation_max_output_tokens is not None
        else (
            int(_consolidation_max_output_tokens_env)
            if _consolidation_max_output_tokens_env
            else None
        )
    )
    _consolidation_max_input_tokens_env = os.getenv("CONSOLIDATION_MAX_INPUT_TOKENS")
    resolved_consolidation_max_input_tokens = (
        consolidation_max_input_tokens
        if consolidation_max_input_tokens is not None
        else (
            int(_consolidation_max_input_tokens_env)
            if _consolidation_max_input_tokens_env
            else 65536
        )
    )

    # ── Prompt family ─────────────────────────────────────────────────────────
    _env_prompt_family = os.getenv("PROMPT_FAMILY", "generic").lower()
    if prompt_family is not None:
        try:
            resolved_prompt_family = PromptFamily(prompt_family)
        except ValueError:
            valid = [m.value for m in PromptFamily]
            raise ConfigurationError(
                f"Invalid prompt_family: {prompt_family!r}. Valid values: {valid}"
            ) from None
    else:
        try:
            resolved_prompt_family = PromptFamily(_env_prompt_family)
        except ValueError:
            resolved_prompt_family = PromptFamily.GENERIC
            log.warning(
                "Unknown PROMPT_FAMILY env value %r, defaulting to 'generic'.",
                _env_prompt_family,
            )

    resolved_llm_stage_configuration = _resolve_llm_stage_configuration(llm_stage_configuration)

    # ── Normalization ─────────────────────────────────────────────────────────
    resolved_normalization_enabled = (
        normalization_enabled
        if normalization_enabled is not None
        else os.getenv("NORMALIZATION_ENABLED", "true").lower() == "true"
    )
    resolved_normalization_batch_size = (
        normalization_batch_size
        if normalization_batch_size is not None
        else int(os.getenv("NORMALIZATION_BATCH_SIZE", "5"))
    )
    resolved_normalization_llm = normalization_llm  # Can be None — engine uses main llm as fallback

    # ── Resolve extra_models_paths ────────────────────────────────────────────
    resolved_extra_models_paths: list[Path] = []
    if extra_models_paths:
        resolved_extra_models_paths = [Path(p) for p in extra_models_paths]
    else:
        env_paths = os.getenv("SCINR_EXTRA_MODELS_PATHS", "")
        if env_paths.strip():
            resolved_extra_models_paths = [
                Path(p.strip()) for p in env_paths.split(":") if p.strip()
            ]

    # ── Build config ──────────────────────────────────────────────────────────
    return ScinrConfig(
        llm=resolved_llm,
        repair_llm=resolved_repair_llm,
        neo4j_uri=resolved_neo4j_uri,
        neo4j_user=resolved_neo4j_user,
        neo4j_password=resolved_neo4j_password,
        neo4j_database=resolved_neo4j_database,
        graph_backend=resolved_graph_backend,
        enabled_base_themes=enabled_base_themes,
        enabled_user_themes=enabled_user_themes,
        extra_models_paths=resolved_extra_models_paths,
        storage_backend=resolved_storage_backend,
        mongodb_uri=mongodb_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
        mongodb_database=mongodb_database or os.getenv("MONGODB_DATABASE", "scinr"),
        mongodb_raw_files_collection=(
            mongodb_raw_files_collection or os.getenv("MONGODB_RAW_FILES_COLLECTION", "raw_files")
        ),
        mongodb_pages_collection=(
            mongodb_pages_collection or os.getenv("MONGODB_PAGES_COLLECTION", "converted_pages")
        ),
        mongodb_gridfs_bucket=(
            mongodb_gridfs_bucket or os.getenv("MONGODB_GRIDFS_BUCKET", "raw_binaries")
        ),
        mongodb_ensure_indexes=(
            mongodb_ensure_indexes
            if mongodb_ensure_indexes is not None
            else os.getenv("MONGODB_ENSURE_INDEXES", "true").lower() == "true"
        ),
        custom_storage=custom_storage,
        freeze_backend=resolved_freeze_backend,
        mongodb_frozen_collection=(
            mongodb_frozen_collection or os.getenv("MONGODB_FROZEN_COLLECTION", "frozen_documents")
        ),
        mongodb_frozen_gridfs_bucket=(
            mongodb_frozen_gridfs_bucket
            or os.getenv("MONGODB_FROZEN_GRIDFS_BUCKET", "frozen_snapshots")
        ),
        custom_freeze_storage=custom_freeze_storage,
        mongodb_images_collection=(
            mongodb_images_collection or os.getenv("MONGODB_IMAGES_COLLECTION", "page_images")
        ),
        mongodb_images_gridfs_bucket=(
            mongodb_images_gridfs_bucket
            or os.getenv("MONGODB_IMAGES_GRIDFS_BUCKET", "page_image_binaries")
        ),
        sql_url=resolved_sql_url,
        sql_schema=resolved_sql_schema,
        sql_pool_size=resolved_sql_pool_size,
        sql_max_engines=resolved_sql_max_engines,
        sql_fulltext_config=resolved_sql_fulltext_config,
        sqlite_files_dir=resolved_sqlite_files_dir,
        extra_converters=extra_converters or {},
        mistral_api_key=mistral_api_key or os.getenv("MISTRAL_API_KEY"),
        mistral_ocr_safe_max_pages=resolved_mistral_ocr_safe_max_pages,
        mistral_ocr_safe_max_bytes=resolved_mistral_ocr_safe_max_bytes,
        mistral_ocr_max_retries=resolved_mistral_ocr_max_retries,
        mistral_ocr_retry_backoff_seconds=resolved_mistral_ocr_retry_backoff_seconds,
        mistral_ocr_chunk_concurrency=resolved_mistral_ocr_chunk_concurrency,
        mistral_ocr_error_strategy=resolved_mistral_ocr_error_strategy,
        mistral_ocr_concurrency=resolved_mistral_ocr_concurrency,
        mistral_ocr_image_max_pages=resolved_mistral_ocr_image_max_pages,
        vision_llm=vision_llm,
        transcriber=transcriber,
        describe_embedded_images=resolved_describe_embedded_images,
        store_described_images=resolved_store_described_images,
        store_video_frames=resolved_store_video_frames,
        media_error_strategy=resolved_media_error_strategy,
        strip_location=resolved_strip_location,
        image_max_side_px=resolved_image_max_side_px,
        image_min_side_px=resolved_image_min_side_px,
        max_images_per_page=resolved_max_images_per_page,
        image_decode_concurrency=resolved_image_decode_concurrency,
        media_segment_seconds=resolved_media_segment_seconds,
        media_min_segment_seconds=resolved_media_min_segment_seconds,
        media_language=resolved_media_language,
        max_media_duration_seconds=resolved_max_media_duration_seconds,
        video_frames_per_minute=resolved_video_frames_per_minute,
        media_concurrency=resolved_media_concurrency,
        ffmpeg_path=resolved_ffmpeg_path,
        ffprobe_path=resolved_ffprobe_path,
        prompt_caching_enabled=resolved_caching,
        full_docstring=resolved_full_docstring,
        extraction_batch_size=resolved_batch_size,
        verbatim_info_unit_max_chars=resolved_verbatim_max_chars,
        auto_info_units_max_input_chars=resolved_auto_max_chars,
        llm_concurrency=resolved_concurrency,
        graph_concurrency=resolved_graph_concurrency,
        graph_sync_concurrency=resolved_graph_sync_concurrency,
        consolidation_token_safety_margin=resolved_consolidation_token_safety_margin,
        consolidation_max_output_tokens=resolved_consolidation_max_output_tokens,
        consolidation_max_input_tokens=resolved_consolidation_max_input_tokens,
        log_level=log_level,
        prompt_family=resolved_prompt_family,
        normalization_enabled=resolved_normalization_enabled,
        normalization_batch_size=resolved_normalization_batch_size,
        normalization_llm=resolved_normalization_llm,
        llm_stage_configuration=resolved_llm_stage_configuration,
    )


def configure(
    # LLM
    llm: Any | None = None,
    repair_llm: Any | None = None,
    # Neo4j
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    neo4j_database: str | None = None,
    # Graph navigation
    graph_backend: Literal["neo4j", "postgres", "sqlite"] | None = None,
    # Models
    enabled_base_themes: list[ThemePath | str] | None = None,
    enabled_user_themes: list[str] | None = None,
    extra_models_paths: list[str | Path] | None = None,
    # Storage
    storage_backend: Literal["none", "mongodb", "postgres", "sqlite", "custom"] | None = None,
    mongodb_uri: str | None = None,
    mongodb_database: str | None = None,
    mongodb_raw_files_collection: str | None = None,
    mongodb_pages_collection: str | None = None,
    mongodb_gridfs_bucket: str | None = None,
    mongodb_ensure_indexes: bool | None = None,
    custom_storage: tuple | None = None,
    # Freeze (document snapshots)
    freeze_backend: Literal["none", "mongodb", "postgres", "sqlite", "custom"] | None = None,
    mongodb_frozen_collection: str | None = None,
    mongodb_frozen_gridfs_bucket: str | None = None,
    custom_freeze_storage: Any | None = None,
    # Page images (described embedded images and video frames)
    mongodb_images_collection: str | None = None,
    mongodb_images_gridfs_bucket: str | None = None,
    # SQL (Postgres / SQLite)
    sql_url: str | None = None,
    sql_schema: str | None = None,
    sql_pool_size: int | None = None,
    sql_max_engines: int | None = None,
    sql_fulltext_config: str | None = None,
    sqlite_files_dir: str | None = None,
    # Converters
    extra_converters: dict[str, type] | None = None,
    # PDF
    mistral_api_key: str | None = None,
    mistral_ocr_safe_max_pages: int | None = None,
    mistral_ocr_safe_max_bytes: int | None = None,
    mistral_ocr_max_retries: int | None = None,
    mistral_ocr_retry_backoff_seconds: float | None = None,
    mistral_ocr_chunk_concurrency: int | None = None,
    mistral_ocr_error_strategy: Literal["fail_fast", "best_effort"] | None = None,
    mistral_ocr_concurrency: int | None = None,
    mistral_ocr_image_max_pages: int | None = None,
    # Multimodal (images, audio, video)
    vision_llm: Any | None = None,
    transcriber: Any | None = None,
    describe_embedded_images: bool | None = None,
    store_described_images: bool | None = None,
    store_video_frames: bool | None = None,
    media_error_strategy: Literal["fail_fast", "best_effort"] | None = None,
    strip_location: bool | None = None,
    image_max_side_px: int | None = None,
    image_min_side_px: int | None = None,
    max_images_per_page: int | None = None,
    image_decode_concurrency: int | None = None,
    media_segment_seconds: int | None = None,
    media_min_segment_seconds: int | None = None,
    media_language: str | None = None,
    max_media_duration_seconds: int | None | _Unset = _UNSET,
    video_frames_per_minute: float | None = None,
    media_concurrency: int | None = None,
    ffmpeg_path: str | None = None,
    ffprobe_path: str | None = None,
    # Pipeline behaviour
    prompt_caching_enabled: bool | None = None,
    full_docstring: bool | None = None,
    extraction_batch_size: int | None = None,
    llm_concurrency: int | None = None,
    graph_concurrency: int | None = None,
    graph_sync_concurrency: int | None = None,
    neo4j_concurrency: int | None = None,
    neo4j_sync_concurrency: int | None = None,
    # Consolidation (fast_extraction Stage 1 post-extraction merge step)
    consolidation_token_safety_margin: float | None = None,
    consolidation_max_output_tokens: int | None = None,
    consolidation_max_input_tokens: int | None = None,
    # Logging
    log_level: str = "INFO",
    # Prompt family
    prompt_family: PromptFamily | Literal["generic", "claude", "gpt_reasoning"] | None = None,
    # Normalization
    normalization_enabled: bool | None = None,
    normalization_batch_size: int | None = None,
    normalization_llm: Any | None = None,
    llm_stage_configuration: dict[str, StageLLMConfig] | None = None,
    verbatim_info_unit_max_chars: int | None = None,
    auto_info_units_max_input_chars: int | None = None,
) -> ScinrConfig:
    """
    Configure the scinr-ingest library.

    Builds the configuration with :func:`build_config` (same parameters, same resolution order:
    explicit argument > environment variable > default; see there for each parameter), installs
    it as the process-wide configuration and applies its side effects: log redaction, converter
    overrides and the reset of every cache that depends on it (theme registry, Neo4j driver,
    graph store, MongoDB client, SQL engines, catalog memoization).

    Returns:
        ScinrConfig singleton containing active library settings.

    Raises:
        ConfigurationError: If conflicting settings or invalid URIs are supplied.
    """
    global _config
    # Snapshot of the parameters; must stay the first statement (locals() holds only them).
    params = dict(locals())

    # Setup logging first
    root_had_handlers = bool(logging.getLogger().handlers)
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))
    if not root_had_handlers:
        # The handler basicConfig just created is ours: make it mask credentials.
        # An application's own handlers are left alone (see redact_handlers()).
        from scinr.newton.utils.logging_config import redact_handlers

        redact_handlers()

    cfg = build_config(**params)
    _config = cfg

    # ── Post-init: apply converter overrides ──────────────────────────────────
    if cfg.extra_converters:
        from scinr.newton.converters.registry import apply_converter_overrides

        apply_converter_overrides(cfg.extra_converters)

    # ── Reset lazy singletons that depend on config ───────────────────────────
    try:
        from scinr.newton.utils.theme_registry import reset_theme_registry

        reset_theme_registry()
    except Exception:
        pass  # theme_registry may not be initialized yet

    try:
        from scinr.newton.graph.neo4j.driver import _reset_async_driver_singleton

        _reset_async_driver_singleton()
    except Exception:
        pass  # ingest module may not be initialized yet

    try:
        from scinr.newton.graph import reset_graph_store

        reset_graph_store()
    except Exception:
        pass  # graph package may not be initialized yet

    try:
        from scinr.newton.storage.mongodb.client import reset_client

        reset_client()
    except Exception:
        pass  # storage mongodb module may not be initialized yet

    try:
        from scinr.newton.sql.engine import reset_engines

        reset_engines()
    except ImportError:
        pass  # SQL extras not installed

    try:
        from scinr.newton.annotation.catalog import reset_catalog_memoization

        reset_catalog_memoization()
    except Exception:
        pass  # annotation module may not be initialized yet

    if cfg.llm_stage_configuration:
        stage_summaries = []
        for stage_name, stage_sc in cfg.llm_stage_configuration.items():
            stage_llm = stage_sc.llm if stage_sc.llm is not None else cfg.llm
            resolved_stage = ResolvedStageConfig(
                stage=stage_name,
                llm=stage_llm,
                prompt_family=(
                    stage_sc.prompt_family
                    if stage_sc.prompt_family is not None
                    else cfg.prompt_family
                ),
                prompt_caching_enabled=(
                    stage_sc.prompt_caching_enabled
                    if stage_sc.prompt_caching_enabled is not None
                    else cfg.prompt_caching_enabled
                ),
            )
            llm_label = _llm_label(stage_llm) if stage_sc.llm is not None else "global"
            stage_summaries.append(
                f"{stage_name}: llm={llm_label}, "
                f"prompt_family={resolved_stage.prompt_family.value}, "
                f"prompt_caching={resolved_stage.prompt_caching_enabled}"
            )
        log.info("LLM stage configuration: %s", "; ".join(stage_summaries))

    log.debug(
        "scinr-ingest configured: storage=%s, graph_backend=%s, llm_concurrency=%d, "
        "graph_concurrency=%d",
        cfg.storage_backend,
        cfg.graph_backend,
        cfg.llm_concurrency,
        cfg.graph_concurrency,
    )
    return cfg


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_int(raw: str, name: str) -> int:
    try:
        return int(raw)
    except ValueError:
        raise ConfigurationError(f"{name} must be an integer. Received: {raw!r}.") from None


def _resolve_int(value: int | None, env: str, default: int, name: str, *, minimum: int) -> int:
    """Explicit argument > env variable > default, validated against *minimum*."""
    if value is not None:
        resolved = value
    else:
        raw = os.getenv(env)
        resolved = default if raw is None else _parse_int(raw, env)
    if resolved < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}. Received: {resolved!r}.")
    return resolved


def _resolve_graph_concurrency(
    value: int | None,
    old_value: int | None,
    *,
    env: str,
    old_env: str,
    default: int,
    name: str,
    old_name: str,
) -> int:
    """Explicit argument > deprecated argument > env > deprecated env > default (>= 1)."""
    if old_value is not None:
        warnings.warn(f"{old_name} is deprecated, use {name}", DeprecationWarning, stacklevel=3)
        if value is not None and value != old_value:
            raise ConfigurationError(
                f"{name} and {old_name} differ. Received: {value!r} and {old_value!r}."
            )
        value = old_value
    if value is None and os.getenv(env) is None and os.getenv(old_env) is not None:
        warnings.warn(f"{old_env} is deprecated, use {env}", DeprecationWarning, stacklevel=3)
        env = old_env
    return _resolve_int(value, env, default, name, minimum=1)


def _validate_sql_config(
    *, sql_url: str, sql_schema: str, sql_fulltext_config: str, axes: dict[str, str]
) -> None:
    """Validate the SQL settings against the three backend axes.

    Raises:
        ConfigurationError: If an axis needs ``sql_url`` and it is missing, if the URL
            scheme does not match an axis, if the axes mix Postgres and SQLite, or if
            ``sql_schema`` / ``sql_fulltext_config`` are not plain identifiers.
    """
    from scinr.newton.utils.redaction import redact_uri

    sql_axes = {name: value for name, value in axes.items() if value in _SQL_BACKENDS}
    if not sql_axes:
        return

    first_name, first_value = next(iter(sql_axes.items()))
    if not sql_url:
        raise ConfigurationError(f"{first_name} {first_value!r} requires sql_url (or SQL_URL).")

    if len(set(sql_axes.values())) > 1:
        detail = ", ".join(f"{name}={value!r}" for name, value in sql_axes.items())
        raise ConfigurationError(
            f"The SQL axes cannot mix Postgres and SQLite ({detail}): they share one sql_url."
        )

    scheme = sql_url.partition("://")[0].partition("+")[0].lower()
    url_backend = _SQL_URL_SCHEMES.get(scheme)
    if url_backend != first_value:
        raise ConfigurationError(
            f"{first_name} {first_value!r} does not match sql_url {redact_uri(sql_url)!r}."
        )

    if not _SQL_SCHEMA_RE.match(sql_schema):
        raise ConfigurationError(
            f"sql_schema must match {_SQL_SCHEMA_RE.pattern}. Received: {sql_schema!r}."
        )
    if not _SQL_IDENTIFIER_RE.match(sql_fulltext_config):
        raise ConfigurationError(
            f"sql_fulltext_config must match {_SQL_IDENTIFIER_RE.pattern}. "
            f"Received: {sql_fulltext_config!r}."
        )


def _resolve_bool(value: bool | None, env: str, default: bool) -> bool:
    """Explicit argument > env variable (``true`` / ``false``) > default."""
    if value is not None:
        return bool(value)
    raw = os.getenv(env)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_VISION_LLM_EXAMPLE = (
    "Example:\n"
    "  from langchain_openai import ChatOpenAI\n"
    "  configure(vision_llm=ChatOpenAI(model='gpt-4o-mini'))\n"
    "  # or any multimodal chat model: ChatBedrockConverse, ChatOllama,\n"
    "  # ChatOpenAI(base_url=...) against an OpenAI-compatible server…"
)


def _validate_vision_llm(vision_llm: Any) -> None:
    """``vision_llm`` must be a LangChain chat model (no structured output needed)."""
    try:
        from langchain_core.language_models import BaseChatModel
    except ImportError:  # pragma: no cover - langchain_core is a base dependency
        return
    if not isinstance(vision_llm, BaseChatModel):
        raise ConfigurationError(
            f"'vision_llm' must be a LangChain BaseChatModel instance that accepts "
            f"images. Received: {type(vision_llm).__name__}.\n{_VISION_LLM_EXAMPLE}"
        )


def _validate_llm(llm: Any) -> None:
    """Validate that llm is a LangChain BaseChatModel with structured output support."""
    try:
        from langchain_core.language_models import BaseChatModel

        if not isinstance(llm, BaseChatModel):
            raise ConfigurationError(
                f"'llm' must be a LangChain BaseChatModel instance. "
                f"Received: {type(llm).__name__}.\n"
                f"Compatible models:\n"
                f"  from langchain_openai import ChatOpenAI\n"
                f"  from langchain_aws import ChatBedrockConverse\n"
                f"  from langchain_ollama import ChatOllama\n"
                f"  from langchain_anthropic import ChatAnthropic\n"
                f"See: https://python.langchain.com/docs/integrations/chat/"
            )
    except ImportError:
        pass  # langchain_core not available yet — skip validation

    if not hasattr(llm, "with_structured_output"):
        raise ConfigurationError(
            f"{type(llm).__name__} does not implement with_structured_output, "
            f"which is required for structured extraction.\n"
            f"See: https://python.langchain.com/docs/concepts/structured_outputs"
        )


def _llm_label(llm: Any) -> str:
    """Model name of *llm* for logs: never credentials."""
    for attr in ("model_id", "model", "model_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(llm).__name__


def _resolve_llm_stage_configuration(value: Any) -> dict[str, StageLLMConfig]:
    """Validate ``llm_stage_configuration`` and normalise each ``prompt_family``."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(
            "llm_stage_configuration must be a dict of stage name -> StageLLMConfig, "
            f"got {type(value).__name__}."
        )
    resolved: dict[str, StageLLMConfig] = {}
    for stage, sc in value.items():
        if stage not in LLM_STAGES:
            raise ConfigurationError(
                f"Unknown stage {stage!r} in llm_stage_configuration. "
                f"Valid stages: {list(LLM_STAGES)}"
            )
        if not isinstance(sc, StageLLMConfig):
            raise ConfigurationError(
                f"llm_stage_configuration[{stage!r}] must be a StageLLMConfig, "
                f"got {type(sc).__name__}."
            )
        if sc.llm is not None:
            _validate_llm(sc.llm)
        family: PromptFamily | None = None
        if sc.prompt_family is not None:
            try:
                family = PromptFamily(sc.prompt_family)
            except ValueError:
                raise ConfigurationError(
                    f"Invalid prompt_family {sc.prompt_family!r} for stage {stage!r}. "
                    f"Valid values: {[m.value for m in PromptFamily]}"
                ) from None
        if sc.prompt_caching_enabled is not None and not isinstance(
            sc.prompt_caching_enabled, bool
        ):
            raise ConfigurationError(
                f"prompt_caching_enabled for stage {stage!r} must be a bool or None."
            )
        resolved[stage] = dataclasses.replace(sc, prompt_family=family)
    return resolved


def _require_llm(stage: str | None = None) -> Any:
    """Return the configured main LLM or raise a descriptive ConfigurationError.

    Raises:
        ConfigurationError: If no LLM is configured (``cfg.llm is None``).
    """
    cfg = get_config()
    if cfg.llm is None:
        prefix = (
            f"Stage {stage!r} has no LLM in llm_stage_configuration and no global llm "
            "is configured.\n\n"
            if stage is not None
            else ""
        )
        raise ConfigurationError(
            prefix + "An LLM is required for this operation but none is configured.\n"
            "\n"
            "Configure one before running LLM-dependent stages (extraction, "
            "annotation, entity extraction, or tabular mapping/normalization):\n"
            "\n"
            "  Option 1 — Pass a LangChain model:\n"
            "    from langchain_openai import ChatOpenAI\n"
            "    configure(llm=ChatOpenAI(model='gpt-4o'))\n"
            "\n"
            "  Option 2 — Use AWS Bedrock (define in .env or environment):\n"
            "    MODEL_ID=us.anthropic.claude-sonnet-4-6\n"
            "    AWS_DEFAULT_REGION=us-east-1\n"
            "\n"
            "Note: graph navigation (scinr.newton.navigation) does NOT require an LLM."
        )
    return cfg.llm


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_available_themes() -> dict[str, list[str]]:
    """
    Return all currently registered theme paths, grouped by origin.

    Does **not** require :func:`configure` to have been called first —
    it initialises the registry with default settings if needed.

    Returns:
        dict with two keys:

        ``"builtin"``
            Theme paths that ship with the scinr-ingest package.
        ``"user"``
            Theme paths loaded from ``extra_models_paths``.

    Examples::

        from scinr.newton import get_available_themes

        themes = get_available_themes()
        print(themes["builtin"])   # ['default', 'equipment_qualification', ...]
        print(themes["user"])      # ['my_custom_theme', ...]
    """
    from scinr.newton.utils.theme_registry import get_theme_registry

    registry = get_theme_registry()
    all_paths = registry.get_all_theme_paths()
    user_paths = registry._user_theme_paths
    return {
        "builtin": sorted(p for p in all_paths if p not in user_paths),
        "user": sorted(p for p in all_paths if p in user_paths),
    }


def get_stage_config(stage: str) -> ResolvedStageConfig:
    """Effective LLM settings of *stage* (each field: the stage's value, else the global one)."""
    if stage not in LLM_STAGES:
        raise ValueError(f"Unknown stage {stage!r}. Valid stages: {list(LLM_STAGES)}")
    cfg = get_config()
    sc = cfg.llm_stage_configuration.get(stage) or StageLLMConfig()
    return ResolvedStageConfig(
        stage=stage,
        llm=sc.llm if sc.llm is not None else cfg.llm,
        prompt_family=(
            PromptFamily(sc.prompt_family) if sc.prompt_family is not None else cfg.prompt_family
        ),
        prompt_caching_enabled=(
            sc.prompt_caching_enabled
            if sc.prompt_caching_enabled is not None
            else cfg.prompt_caching_enabled
        ),
    )


def get_llm(temperature: float = 0.0, *, stage: str | None = None):
    """Return the configured LLM (of *stage* when given). Optionally binds a temperature."""
    if stage is None:
        llm = _require_llm()
    else:
        llm = get_stage_config(stage).llm
        if llm is None:
            llm = _require_llm(stage=stage)
        log.debug("LLM for stage %s: %s", stage, _llm_label(llm))
    if temperature == 0.0:
        return llm
    try:
        return llm.bind(temperature=temperature)
    except Exception:
        return llm


def get_repair_llm(temperature: float = 0.0):
    """Return the configured repair LLM. Optionally binds a temperature."""
    cfg = get_config()
    llm = cfg.repair_llm if cfg.repair_llm is not None else _require_llm()
    if temperature == 0.0:
        return llm
    try:
        return llm.bind(temperature=temperature)
    except Exception:
        return llm


def get_vision_llm():
    """Return the configured ``vision_llm``.

    It never falls back to ``llm``: vision depends on a multimodal model
    configured on purpose.

    Raises:
        ConfigurationError: If no ``vision_llm`` is configured.
    """
    cfg = get_config()
    if cfg.vision_llm is None:
        raise ConfigurationError(
            "A vision model is required to describe images but none is configured "
            "(vision_llm does not fall back to llm).\n" + _VISION_LLM_EXAMPLE
        )
    return cfg.vision_llm


def get_transcriber():
    """Return the configured ``transcriber``.

    Raises:
        ConfigurationError: If no ``transcriber`` is configured.
    """
    cfg = get_config()
    if cfg.transcriber is None:
        raise ConfigurationError(
            "A transcriber is required to convert audio but none is configured.\n"
            "Example:\n"
            "  from scinr.newton.media.transcribers import OpenAICompatibleTranscriber\n"
            "  configure(transcriber=OpenAICompatibleTranscriber(model='whisper-1', "
            "api_key=...))\n"
            "  # or OpenAICompatibleTranscriber(model=..., base_url='http://my-server/v1')"
            " for a self-hosted OpenAI-compatible server,\n"
            "  # or FasterWhisperTranscriber() for local inference "
            "(pip install 'scinr[media]')."
        )
    return cfg.transcriber


def embedded_images_enabled() -> bool:
    """``True`` when embedded images must be described.

    That is ``describe_embedded_images`` **and** a configured ``vision_llm``.
    The only condition the PDF, DOCX and PPTX converters and the enrichment
    step check. ``False`` when :func:`configure` has not been called.
    """
    cfg = _config_override.get() or _config
    if cfg is None:
        return False
    return bool(cfg.describe_embedded_images and cfg.vision_llm is not None)


def get_prompt_family(*, stage: str | None = None) -> PromptFamily:
    """Return the configured prompt family (GENERIC, CLAUDE, or GPT_REASONING).

    Controls which prompt variant is used for all LLM calls in the pipeline.
    Set via configure(prompt_family=PromptFamily.CLAUDE) or PROMPT_FAMILY=claude env var.
    With *stage*, returns the family of that stage (its own, else the global one).
    """
    if stage is not None:
        return get_stage_config(stage).prompt_family
    return get_config().prompt_family


def make_system_message(text: str, *, stage: str | None = None):
    """
    Create a SystemMessage. If the LLM is ChatBedrockConverse and prompt caching
    is enabled, appends a Bedrock-specific cachePoint block (~90% token cost reduction
    on repeated calls with the same prompt). For all other providers, returns a
    standard SystemMessage. With *stage*, the LLM and the caching flag are those of
    that stage.
    """
    from langchain_core.messages import SystemMessage

    try:
        from langchain_aws import ChatBedrockConverse

        if stage is None:
            cfg = get_config()
            llm, caching_enabled = cfg.llm, cfg.prompt_caching_enabled
        else:
            stage_cfg = get_stage_config(stage)
            llm, caching_enabled = stage_cfg.llm, stage_cfg.prompt_caching_enabled
        if isinstance(llm, ChatBedrockConverse) and caching_enabled:
            return SystemMessage(
                content=[
                    {"type": "text", "text": text},
                    {"cachePoint": {"type": "default"}},
                ]
            )
    except ImportError:
        pass  # langchain-aws not installed — not a Bedrock user
    return SystemMessage(content=text)


def get_llm_semaphore():
    """Return (creating if needed) the global asyncio.Semaphore for LLM concurrency."""
    import asyncio

    global _llm_semaphore
    if _llm_semaphore is None:
        cfg = get_config()
        _llm_semaphore = asyncio.Semaphore(cfg.llm_concurrency)
    return _llm_semaphore


def reset_llm_semaphore() -> None:
    """Reset the LLM semaphore (used after configure() changes llm_concurrency)."""
    global _llm_semaphore
    _llm_semaphore = None


_llm_semaphore = None

_graph_semaphore = None


def get_graph_semaphore():
    """Return (creating if needed) the global asyncio.Semaphore for graph-store session concurrency.

    The semaphore bounds the number of concurrent graph-store sessions during
    annotation (Stage 3) and entity extraction (Stage 4), for any graph backend
    (Neo4j or SQL). It is created lazily on the first call using the
    graph_concurrency value from ScinrConfig.

    To change the concurrency at runtime, call configure(graph_concurrency=N)
    followed by reset_graph_semaphore() before running the relevant stages.
    """
    import asyncio

    global _graph_semaphore
    if _graph_semaphore is None:
        cfg = get_config()
        _graph_semaphore = asyncio.Semaphore(cfg.graph_concurrency)
    return _graph_semaphore


def reset_graph_semaphore() -> None:
    """Reset the graph-store semaphore (used after configure() changes graph_concurrency).

    Call this after configure(graph_concurrency=N) to ensure the new value
    takes effect on the next call to get_graph_semaphore().
    """
    global _graph_semaphore
    _graph_semaphore = None


_graph_sync_semaphore = None


def get_graph_sync_semaphore():
    """Global semaphore bounding how many concurrent dispatches to
    asyncio.to_thread() for Stage 2 (synchronous ingestion) are in flight.
    It must ALWAYS be acquired/released on the event loop (by the caller of
    asyncio.to_thread), never inside the worker thread — asyncio.Semaphore
    is not usable outside a coroutine with a running loop.
    """
    import asyncio

    global _graph_sync_semaphore
    if _graph_sync_semaphore is None:
        cfg = get_config()
        _graph_sync_semaphore = asyncio.Semaphore(cfg.graph_sync_concurrency)
    return _graph_sync_semaphore


def reset_graph_sync_semaphore() -> None:
    """Manual reset after configure(graph_sync_concurrency=N) — NOT
    auto-invoked from configure(); same manual contract as
    reset_graph_semaphore()/reset_llm_semaphore()."""
    global _graph_sync_semaphore
    _graph_sync_semaphore = None


def get_neo4j_semaphore():
    """Deprecated alias of :func:`get_graph_semaphore` (removed in 0.7.0)."""
    warnings.warn(
        "get_neo4j_semaphore is deprecated, use get_graph_semaphore",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_graph_semaphore()


def reset_neo4j_semaphore() -> None:
    """Deprecated alias of :func:`reset_graph_semaphore` (removed in 0.7.0)."""
    warnings.warn(
        "reset_neo4j_semaphore is deprecated, use reset_graph_semaphore",
        DeprecationWarning,
        stacklevel=2,
    )
    reset_graph_semaphore()


def get_neo4j_sync_semaphore():
    """Deprecated alias of :func:`get_graph_sync_semaphore` (removed in 0.7.0)."""
    warnings.warn(
        "get_neo4j_sync_semaphore is deprecated, use get_graph_sync_semaphore",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_graph_sync_semaphore()


def reset_neo4j_sync_semaphore() -> None:
    """Deprecated alias of :func:`reset_graph_sync_semaphore` (removed in 0.7.0)."""
    warnings.warn(
        "reset_neo4j_sync_semaphore is deprecated, use reset_graph_sync_semaphore",
        DeprecationWarning,
        stacklevel=2,
    )
    reset_graph_sync_semaphore()
