"""
config.py — Global scinr-ingest configuration.

Call configure() once at startup to set LLM, Neo4j, storage and other
pipeline parameters. All pipeline modules read settings via get_config().

Parameter resolution order for all parameters:
  1. Explicit argument passed to configure()
  2. Environment variable (via os.getenv)
  3. Hard-coded default value

Usage (library mode):
    from langchain_openai import ChatOpenAI
    from scinr.newton.config import configure
    configure(llm=ChatOpenAI(model="gpt-4o"), neo4j_user="neo4j", neo4j_password="...")

Usage (CLI mode / .env file):
    configure()  # Reads everything from environment variables
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
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
    GENERIC       = "generic"        # default — model-agnostic, all LLM families
    CLAUDE        = "claude"         # XML-structured, extended reasoning protocols
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

Example::

    configure(
        llm=...,
        enabled_base_themes=["pharmaceutical_quality", "pharma_operations/batch_manufacturing"],
    )
"""

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


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
    # Models
    enabled_base_themes: list[ThemePath | str] | None = None
    enabled_user_themes: list[str] | None = None
    extra_models_paths: list[Path] = field(default_factory=list)
    # Storage
    storage_backend: str = "none"  # "none" | "mongodb" | "custom"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "scinr"
    mongodb_raw_files_collection: str = "raw_files"
    mongodb_pages_collection: str = "converted_pages"
    mongodb_gridfs_bucket: str = "raw_binaries"
    custom_storage: tuple | None = None  # (RawFileRepository, PageRepository)
    # Converters
    extra_converters: dict[str, type] = field(default_factory=dict)
    # PDF
    mistral_api_key: str | None = None
    mistral_ocr_safe_max_pages: int = 900
    mistral_ocr_safe_max_bytes: int = 45 * 1024 * 1024  # 45 MiB
    mistral_ocr_max_retries: int = 3
    mistral_ocr_retry_backoff_seconds: float = 2.0
    mistral_ocr_chunk_concurrency: int = 1  # reservado para paralelismo futuro, no usado aún
    mistral_ocr_error_strategy: Literal["fail_fast", "best_effort"] = "fail_fast"
    # Pipeline behaviour
    prompt_caching_enabled: bool = True
    extraction_batch_size: int = 3
    llm_concurrency: int = 4
    neo4j_concurrency: int = 10
    neo4j_sync_concurrency: int = 8
    # Logging
    log_level: str = "INFO"
    # Prompt family
    prompt_family: PromptFamily = PromptFamily.GENERIC
    # Normalization
    normalization_enabled: bool = True
    normalization_batch_size: int = 3
    normalization_llm: Any = None  # BaseChatModel — falls back to llm if None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_config: ScinrConfig | None = None


def get_config() -> ScinrConfig:
    """
    Return the current ScinrConfig singleton.

    Raises
    ------
    ConfigurationError
        If configure() has not been called yet.
    """
    global _config
    if _config is None:
        raise ConfigurationError(
            "scinr-ingest has not been configured yet.\n"
            "Call configure() before using any pipeline function:\n"
            "\n"
            "  from scinr.newton import configure\n"
            "  configure(llm=your_llm, neo4j_uri=..., neo4j_user=..., neo4j_password=...)\n"
            "\n"
            "For CLI usage with environment variables, scinr-ingest handles this automatically."
        )
    return _config  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# configure()
# ---------------------------------------------------------------------------


def configure(
    # LLM
    llm: Any | None = None,
    repair_llm: Any | None = None,
    # Neo4j
    neo4j_uri: str | None = None,
    neo4j_user: str | None = None,
    neo4j_password: str | None = None,
    # Models
    enabled_base_themes: list[ThemePath | str] | None = None,
    enabled_user_themes: list[str] | None = None,
    extra_models_paths: list[str | Path] | None = None,
    # Storage
    storage_backend: Literal["none", "mongodb", "custom"] | None = None,
    mongodb_uri: str | None = None,
    mongodb_database: str | None = None,
    mongodb_raw_files_collection: str | None = None,
    mongodb_pages_collection: str | None = None,
    mongodb_gridfs_bucket: str | None = None,
    custom_storage: tuple | None = None,
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
    # Pipeline behaviour
    prompt_caching_enabled: bool | None = None,
    extraction_batch_size: int | None = None,
    llm_concurrency: int | None = None,
    neo4j_concurrency: int | None = None,
    neo4j_sync_concurrency: int | None = None,
    # Logging
    log_level: str = "INFO",
    # Prompt family
    prompt_family: PromptFamily | Literal["generic", "claude", "gpt_reasoning"] | None = None,
    # Normalization
    normalization_enabled: bool | None = None,
    normalization_batch_size: int | None = None,
    normalization_llm: Any | None = None,
) -> ScinrConfig:
    """
    Configure the scinr-ingest library.

    Parameter resolution order: explicit argument > environment variable > default.

    Args:
        llm: LangChain BaseChatModel instance to use for all LLM calls.
        repair_llm: LangChain BaseChatModel for the JSON repair loop. Falls back to `llm` if None.
        neo4j_uri: Neo4j connection URI. Env: `NEO4J_URI`. Default: `bolt://localhost:7687`.
        neo4j_user: Neo4j username. Env: `NEO4J_USER`.
        neo4j_password: Neo4j password. Env: `NEO4J_PASSWORD`.
        enabled_base_themes: Whitelist of built-in theme paths to activate (`ThemePath` values).
        enabled_user_themes: Whitelist of user theme paths to activate.
        extra_models_paths: Filesystem paths to scan for additional user-defined theme models.
        storage_backend: Storage type: `'none'` (default), `'mongodb'`, or `'custom'`.
        mongodb_uri: MongoDB connection URI. Env: `MONGODB_URI`.
        mongodb_database: MongoDB database name. Env: `MONGODB_DATABASE`.
        mongodb_raw_files_collection: Collection for raw file metadata.
        mongodb_pages_collection: Collection for converted pages.
        mongodb_gridfs_bucket: GridFS bucket name for binary files.
        custom_storage: Tuple `(RawFileRepository, PageRepository)` when `storage_backend='custom'`.
        extra_converters: Dict mapping file extensions to custom `BaseConverter` subclasses.
        mistral_api_key: Mistral API key for PDF OCR conversion.
        mistral_ocr_safe_max_pages: Máximo de páginas por chunk de PDF enviado a la
            API de Mistral OCR antes de dividirlo. Env: `MISTRAL_OCR_SAFE_MAX_PAGES`.
            Default: `900`.
        mistral_ocr_safe_max_bytes: Máximo de bytes por chunk de PDF (tamaño ya
            serializado) enviado a la API de Mistral OCR antes de dividirlo.
            Env: `MISTRAL_OCR_SAFE_MAX_BYTES`. Default: `45 * 1024 * 1024` (45 MiB).
        mistral_ocr_max_retries: Número máximo de intentos por chunk ante errores
            de red o HTTP reintentables. Env: `MISTRAL_OCR_MAX_RETRIES`. Default: `3`.
        mistral_ocr_retry_backoff_seconds: Base (en segundos) del backoff exponencial
            entre reintentos. Env: `MISTRAL_OCR_RETRY_BACKOFF_SECONDS`. Default: `2.0`.
        mistral_ocr_chunk_concurrency: Reservado para paralelismo futuro entre chunks
            de PDF; actualmente no se usa (procesamiento siempre secuencial).
            Env: `MISTRAL_OCR_CHUNK_CONCURRENCY`. Default: `1`.
        mistral_ocr_error_strategy: Estrategia de manejo de errores al convertir
            PDFs divididos en chunks: `'fail_fast'` (default, aborta el documento
            completo si algún chunk falla) o `'best_effort'` (omite los chunks que
            fallen y continúa con el resto). Env: `MISTRAL_OCR_ERROR_STRATEGY`.
            Default: `'fail_fast'`.
            Nota: esta estrategia solo aplica a fallos de red/API por chunk
            (reintentos agotados, errores HTTP no reintentables) una vez que
            el PDF ya fue dividido en chunks. NO cubre el caso en que la
            propia partición inicial falla estructuralmente (`PdfSplitError`,
            una página individual excede `mistral_ocr_safe_max_bytes` incluso
            aislada) — en ese caso el documento aborta siempre,
            independientemente del valor de `mistral_ocr_error_strategy`.
        prompt_caching_enabled: Enable prompt caching for supported LLM providers.
        extraction_batch_size: Pages per extraction chunk (default: `1`).
        llm_concurrency: Max concurrent LLM calls (semaphore size, default: `4`).
        neo4j_concurrency: Max concurrent Neo4j write sessions (default: `10`).
        neo4j_sync_concurrency: Max concurrent Stage 2 (sync ingestion) dispatches
            to asyncio.to_thread() (default: `8`).
        log_level: Logging level string (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`).
        prompt_family: Prompt family to use (`"generic"`, `"claude"`, or `"gpt_reasoning"`).
        normalization_enabled: Enable post-extraction normalization for tabular data.
        normalization_batch_size: Max normalization entries per LLM batch (default: `5`).
        normalization_llm: Dedicated LLM model instance for tabular normalization.

    Returns:
        ScinrConfig singleton containing active library settings.

    Raises:
        ConfigurationError: If conflicting settings or invalid URIs are supplied.
    """
    global _config

    # Setup logging first
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

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
                config=BotocoreConfig(
                    max_pool_connections=max(resolved_concurrency * 2, 20)
                ),
            )
        else:
            raise ConfigurationError(
                "No LLM configured. Options:\n"
                "\n"
                "  Option 1 — Use any LangChain model:\n"
                "    from langchain_openai import ChatOpenAI\n"
                "    from scinr.newton.config import configure\n"
                "    configure(llm=ChatOpenAI(model='gpt-4o'))\n"
                "\n"
                "  Option 2 — Use AWS Bedrock (define in .env or environment):\n"
                "    MODEL_ID=us.anthropic.claude-sonnet-4-6\n"
                "    AWS_DEFAULT_REGION=us-east-1\n"
            )

    _validate_llm(resolved_llm)

    resolved_repair_llm = repair_llm
    if resolved_repair_llm is None:
        log.warning(
            "Configure -- Specific Repair LLM has not been defined, fallback to main LLM. It is recommended to use a smaller or cheaper model for reparation steps."
        )
        resolved_repair_llm = resolved_llm  # fall back to main LLM

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    resolved_neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    resolved_neo4j_user = neo4j_user or os.getenv("NEO4J_USER")
    resolved_neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")

    # Parse NEO4J_AUTH fallback ("user/password")
    if not resolved_neo4j_user or not resolved_neo4j_password:
        auth_raw = os.getenv("NEO4J_AUTH")
        if auth_raw and "/" in auth_raw:
            parts = auth_raw.split("/", maxsplit=1)
            resolved_neo4j_user = resolved_neo4j_user or parts[0]
            resolved_neo4j_password = resolved_neo4j_password or parts[1]

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

    # ── Storage ───────────────────────────────────────────────────────────────
    resolved_storage_backend = storage_backend or os.getenv("STORAGE_BACKEND", "none")
    if resolved_storage_backend not in ("none", "mongodb", "custom"):
        raise ConfigurationError(
            f"Unknown storage_backend: {resolved_storage_backend!r}. "
            f"Valid values: 'none', 'mongodb', 'custom'."
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
        else int(os.getenv("MISTRAL_OCR_MAX_RETRIES", "3"))
    )
    if resolved_mistral_ocr_max_retries < 1:
        raise ConfigurationError(
            f"mistral_ocr_max_retries must be >= 1. "
            f"Received: {resolved_mistral_ocr_max_retries!r}."
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

    # ── Pipeline behaviour ────────────────────────────────────────────────────
    resolved_caching = (
        prompt_caching_enabled
        if prompt_caching_enabled is not None
        else os.getenv("PROMPT_CACHING_ENABLED", "true").lower() == "true"
    )
    resolved_batch_size = (
        extraction_batch_size
        if extraction_batch_size is not None
        else int(os.getenv("EXTRACTION_BATCH_SIZE", "1"))
    )
    # resolved_concurrency is computed earlier (see "── LLM" section above),
    # before the Bedrock client is constructed, so it's available for the
    # botocore connection pool sizing there too.

    _neo4j_concurrency_env = os.getenv("NEO4J_CONCURRENCY", "10")
    resolved_neo4j_concurrency = neo4j_concurrency if neo4j_concurrency is not None else int(_neo4j_concurrency_env)

    _neo4j_sync_concurrency_env = os.getenv("NEO4J_SYNC_CONCURRENCY", "8")
    resolved_neo4j_sync_concurrency = (
        neo4j_sync_concurrency
        if neo4j_sync_concurrency is not None
        else int(_neo4j_sync_concurrency_env)
    )

    # ── Prompt family ─────────────────────────────────────────────────────────
    _env_prompt_family = os.getenv("PROMPT_FAMILY", "generic").lower()
    if prompt_family is not None:
        try:
            resolved_prompt_family = PromptFamily(prompt_family)
        except ValueError:
            valid = [m.value for m in PromptFamily]
            raise ConfigurationError(
                f"Invalid prompt_family: {prompt_family!r}. "
                f"Valid values: {valid}"
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

    # ── Normalization ─────────────────────────────────────────────────────────
    resolved_normalization_enabled = (
        normalization_enabled
        if normalization_enabled is not None
        else os.getenv("NORMALIZATION_ENABLED", "false").lower() == "true"
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
    _config = ScinrConfig(
        llm=resolved_llm,
        repair_llm=resolved_repair_llm,
        neo4j_uri=resolved_neo4j_uri,
        neo4j_user=resolved_neo4j_user,
        neo4j_password=resolved_neo4j_password,
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
        custom_storage=custom_storage,
        extra_converters=extra_converters or {},
        mistral_api_key=mistral_api_key or os.getenv("MISTRAL_API_KEY"),
        mistral_ocr_safe_max_pages=resolved_mistral_ocr_safe_max_pages,
        mistral_ocr_safe_max_bytes=resolved_mistral_ocr_safe_max_bytes,
        mistral_ocr_max_retries=resolved_mistral_ocr_max_retries,
        mistral_ocr_retry_backoff_seconds=resolved_mistral_ocr_retry_backoff_seconds,
        mistral_ocr_chunk_concurrency=resolved_mistral_ocr_chunk_concurrency,
        mistral_ocr_error_strategy=resolved_mistral_ocr_error_strategy,
        prompt_caching_enabled=resolved_caching,
        extraction_batch_size=resolved_batch_size,
        llm_concurrency=resolved_concurrency,
        neo4j_concurrency=resolved_neo4j_concurrency,
        neo4j_sync_concurrency=resolved_neo4j_sync_concurrency,
        log_level=log_level,
        prompt_family=resolved_prompt_family,
        normalization_enabled=resolved_normalization_enabled,
        normalization_batch_size=resolved_normalization_batch_size,
        normalization_llm=resolved_normalization_llm,
    )

    # ── Post-init: apply converter overrides ──────────────────────────────────
    if _config.extra_converters:
        from scinr.newton.converters.registry import apply_converter_overrides

        apply_converter_overrides(_config.extra_converters)

    # ── Reset lazy singletons that depend on config ───────────────────────────
    try:
        from scinr.newton.utils.theme_registry import reset_theme_registry

        reset_theme_registry()
    except Exception:
        pass  # theme_registry may not be initialized yet

    try:
        from scinr.newton.ingest.config import _reset_async_driver_singleton
        _reset_async_driver_singleton()
    except Exception:
        pass  # ingest module may not be initialized yet

    try:
        from scinr.newton.storage.mongodb.client import reset_client
        reset_client()
    except Exception:
        pass  # storage mongodb module may not be initialized yet

    try:
        from scinr.newton.annotation.neo4j_ops import reset_catalog_memoization
        reset_catalog_memoization()
    except Exception:
        pass  # annotation module may not be initialized yet

    log.debug(
        "scinr-ingest configured: storage=%s, llm_concurrency=%d, neo4j_concurrency=%d",
        resolved_storage_backend,
        resolved_concurrency,
        resolved_neo4j_concurrency,
    )
    return _config


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_available_themes() -> dict[str, list[str]]:
    """
    Return all currently registered theme paths, grouped by origin.

    Does **not** require :func:`configure` to have been called first —
    it initialises the registry with default settings if needed.

    Returns
    -------
    dict with two keys:

    ``"builtin"``
        Theme paths that ship with the scinr-ingest package.
    ``"user"``
        Theme paths loaded from ``extra_models_paths``.

    Example::

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


def get_llm(temperature: float = 0.0):
    """Return the configured LLM. Optionally binds a temperature."""
    cfg = get_config()
    if temperature == 0.0:
        return cfg.llm
    try:
        return cfg.llm.bind(temperature=temperature)
    except Exception:
        return cfg.llm


def get_repair_llm(temperature: float = 0.0):
    """Return the configured repair LLM. Optionally binds a temperature."""
    cfg = get_config()
    if temperature == 0.0:
        return cfg.repair_llm
    try:
        return cfg.repair_llm.bind(temperature=temperature)
    except Exception:
        return cfg.repair_llm


def get_prompt_family() -> PromptFamily:
    """Return the configured prompt family (GENERIC, CLAUDE, or GPT_REASONING).

    Controls which prompt variant is used for all LLM calls in the pipeline.
    Set via configure(prompt_family=PromptFamily.CLAUDE) or PROMPT_FAMILY=claude env var.
    """
    return get_config().prompt_family


def make_system_message(text: str):
    """
    Create a SystemMessage. If the LLM is ChatBedrockConverse and prompt caching
    is enabled, appends a Bedrock-specific cachePoint block (~90% token cost reduction
    on repeated calls with the same prompt). For all other providers, returns a
    standard SystemMessage.
    """
    from langchain_core.messages import SystemMessage

    try:
        from langchain_aws import ChatBedrockConverse

        cfg = get_config()
        if isinstance(cfg.llm, ChatBedrockConverse) and cfg.prompt_caching_enabled:
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

_neo4j_semaphore = None


def get_neo4j_semaphore():
    """Return (creating if needed) the global asyncio.Semaphore for Neo4j session concurrency.

    The semaphore bounds the number of concurrent Neo4j sessions during
    annotation (Stage 3) and entity extraction (Stage 4). It is created
    lazily on the first call using the neo4j_concurrency value from ScinrConfig.

    To change the concurrency at runtime, call configure(neo4j_concurrency=N)
    followed by reset_neo4j_semaphore() before running the relevant stages.
    """
    import asyncio

    global _neo4j_semaphore
    if _neo4j_semaphore is None:
        cfg = get_config()
        _neo4j_semaphore = asyncio.Semaphore(cfg.neo4j_concurrency)
    return _neo4j_semaphore


def reset_neo4j_semaphore() -> None:
    """Reset the Neo4j semaphore (used after configure() changes neo4j_concurrency).

    Call this after configure(neo4j_concurrency=N) to ensure the new value
    takes effect on the next call to get_neo4j_semaphore().
    """
    global _neo4j_semaphore
    _neo4j_semaphore = None


_neo4j_sync_semaphore = None


def get_neo4j_sync_semaphore():
    """Semáforo global que acota cuántos despachos concurrentes a
    asyncio.to_thread() para Stage 2 (ingestión síncrona) están en vuelo.
    Debe adquirirse/liberarse SIEMPRE en el event loop (por el llamador de
    asyncio.to_thread), nunca dentro del worker thread — asyncio.Semaphore
    no es utilizable fuera de una corrutina con loop activo.
    """
    import asyncio

    global _neo4j_sync_semaphore
    if _neo4j_sync_semaphore is None:
        cfg = get_config()
        _neo4j_sync_semaphore = asyncio.Semaphore(cfg.neo4j_sync_concurrency)
    return _neo4j_sync_semaphore


def reset_neo4j_sync_semaphore() -> None:
    """Reset manual tras configure(neo4j_sync_concurrency=N) — NO se
    auto-invoca desde configure(), mismo contrato manual que
    reset_neo4j_semaphore()/reset_llm_semaphore()."""
    global _neo4j_sync_semaphore
    _neo4j_sync_semaphore = None
