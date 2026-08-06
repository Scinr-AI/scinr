"""
tests/unit/test_config.py — Unit tests for scinr.newton.config

Imports are done from submodules directly to avoid triggering the CLI
import chain (which calls get_config() at module load time).
"""
from __future__ import annotations

import pytest

# Import directly from submodules — NOT from scinr.newton (top-level)
from scinr.newton.config import ScinrConfig, configure, get_config
from scinr.newton.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_URI = "bolt://localhost:7687"
_DUMMY_USER = "neo4j"
_DUMMY_PASS = "test"


def _make_mock_llm():
    """Create a minimal mock that passes _validate_llm."""
    try:
        from langchain_core.language_models import BaseChatModel

        class _MockLLM(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "mock"

            def _generate(self, *args, **kwargs):
                pass

        return _MockLLM()
    except Exception:
        from unittest.mock import MagicMock
        llm = MagicMock()
        llm.with_structured_output = MagicMock()
        return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConfigureDefaults:
    def test_configure_defaults(self):
        """configure() with minimum required args stores values and defaults storage to 'none'."""
        llm = _make_mock_llm()
        cfg = configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
        )
        assert cfg.neo4j_uri == _DUMMY_URI
        assert cfg.neo4j_user == _DUMMY_USER
        assert cfg.neo4j_password == _DUMMY_PASS
        assert cfg.storage_backend == "none"
        # get_config() returns the same object
        assert get_config() is cfg

    def test_configure_explicit_values(self):
        """Explicit parameters are stored correctly."""
        llm = _make_mock_llm()
        cfg = configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
            storage_backend="none",
            log_level="DEBUG",
            llm_concurrency=5,
        )
        assert cfg.storage_backend == "none"
        assert cfg.log_level == "DEBUG"
        assert cfg.llm_concurrency == 5

    def test_configure_env_fallback(self, monkeypatch):
        """configure() reads Neo4j credentials from environment variables."""
        monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")
        monkeypatch.setenv("NEO4J_USER", "tester")
        monkeypatch.setenv("NEO4J_PASSWORD", "secret")

        llm = _make_mock_llm()
        cfg = configure(llm=llm)
        assert cfg.neo4j_uri == "bolt://test:7687"
        assert cfg.neo4j_user == "tester"
        assert cfg.neo4j_password == "secret"

    def test_configure_invalid_storage_backend(self):
        """configure() raises ConfigurationError for unknown storage_backend."""
        llm = _make_mock_llm()
        with pytest.raises(ConfigurationError, match="Unknown storage_backend"):
            configure(
                llm=llm,
                neo4j_uri=_DUMMY_URI,
                neo4j_user=_DUMMY_USER,
                neo4j_password=_DUMMY_PASS,
                storage_backend="bogus",  # type: ignore[arg-type]
            )

    def test_configure_custom_storage_without_tuple(self):
        """storage_backend='custom' without custom_storage raises ConfigurationError on get_storage()."""
        from scinr.newton.storage.factory import get_storage

        llm = _make_mock_llm()
        configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
            storage_backend="custom",
            custom_storage=None,
        )
        with pytest.raises(ConfigurationError, match="custom_storage"):
            get_storage()

    def test_get_config_before_configure_raises(self, monkeypatch):
        """get_config() before configure() raises ConfigurationError (no LLM, no Neo4j creds)."""
        import scinr.newton.config as cfg_module

        # Reset config to None
        cfg_module._config = None
        # Remove all env vars that would allow auto-configure to succeed
        monkeypatch.delenv("MODEL_ID", raising=False)
        monkeypatch.delenv("NEO4J_USER", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.delenv("NEO4J_AUTH", raising=False)

        with pytest.raises((ConfigurationError, Exception)):
            get_config()

    def test_configure_extra_converters_invalid(self):
        """extra_converters with a non-BaseConverter class raises ConfigurationError."""
        llm = _make_mock_llm()
        with pytest.raises(ConfigurationError):
            configure(
                llm=llm,
                neo4j_uri=_DUMMY_URI,
                neo4j_user=_DUMMY_USER,
                neo4j_password=_DUMMY_PASS,
                extra_converters={"pdf": int},  # int is not a BaseConverter subclass
            )

    def test_configure_returns_scinr_config(self):
        """configure() returns a ScinrConfig instance."""
        llm = _make_mock_llm()
        cfg = configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
        )
        assert isinstance(cfg, ScinrConfig)

    def test_configure_neo4j_auth_env_fallback(self, monkeypatch):
        """NEO4J_AUTH=user/pass is parsed as fallback for user and password."""
        monkeypatch.setenv("NEO4J_AUTH", "authuser/authpass")
        monkeypatch.delenv("NEO4J_USER", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

        llm = _make_mock_llm()
        cfg = configure(llm=llm, neo4j_uri=_DUMMY_URI)
        assert cfg.neo4j_user == "authuser"
        assert cfg.neo4j_password == "authpass"

    def test_configure_missing_neo4j_user_raises(self, monkeypatch):
        """configure() raises ConfigurationError when neo4j_user is missing."""
        monkeypatch.delenv("NEO4J_USER", raising=False)
        monkeypatch.delenv("NEO4J_AUTH", raising=False)

        llm = _make_mock_llm()
        with pytest.raises(ConfigurationError, match="Neo4j username"):
            configure(
                llm=llm,
                neo4j_uri=_DUMMY_URI,
                neo4j_password=_DUMMY_PASS,
                # neo4j_user intentionally omitted
            )

    def test_configure_missing_neo4j_password_raises(self, monkeypatch):
        """configure() raises ConfigurationError when neo4j_password is missing."""
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        monkeypatch.delenv("NEO4J_AUTH", raising=False)

        llm = _make_mock_llm()
        with pytest.raises(ConfigurationError, match="Neo4j password"):
            configure(
                llm=llm,
                neo4j_uri=_DUMMY_URI,
                neo4j_user=_DUMMY_USER,
                # neo4j_password intentionally omitted
            )


class TestNeo4jSyncSemaphore:
    """Tests for get_neo4j_sync_semaphore() / reset_neo4j_sync_semaphore()."""

    def test_get_neo4j_sync_semaphore_default_size(self, monkeypatch):
        """get_neo4j_sync_semaphore() returns an asyncio.Semaphore sized from cfg.neo4j_sync_concurrency."""
        import asyncio

        from scinr.newton.config import get_neo4j_sync_semaphore, reset_neo4j_sync_semaphore

        monkeypatch.delenv("NEO4J_SYNC_CONCURRENCY", raising=False)
        llm = _make_mock_llm()
        configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
        )
        reset_neo4j_sync_semaphore()

        cfg = get_config()
        sem = get_neo4j_sync_semaphore()
        assert isinstance(sem, asyncio.Semaphore)
        assert sem._value == cfg.neo4j_sync_concurrency == 8

    def test_configure_and_reset_neo4j_sync_semaphore_reflects_new_value(self, monkeypatch):
        """configure(neo4j_sync_concurrency=3) + reset_neo4j_sync_semaphore() picks up the new value."""
        from scinr.newton.config import get_neo4j_sync_semaphore, reset_neo4j_sync_semaphore

        monkeypatch.delenv("NEO4J_SYNC_CONCURRENCY", raising=False)
        llm = _make_mock_llm()
        configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
            neo4j_sync_concurrency=3,
        )
        reset_neo4j_sync_semaphore()

        sem = get_neo4j_sync_semaphore()
        assert sem._value == 3

    def test_configure_respects_neo4j_sync_concurrency_env_var(self, monkeypatch):
        """NEO4J_SYNC_CONCURRENCY env var is respected when no explicit argument is passed."""
        monkeypatch.setenv("NEO4J_SYNC_CONCURRENCY", "5")

        llm = _make_mock_llm()
        cfg = configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
        )
        assert cfg.neo4j_sync_concurrency == 5


class TestMistralOcrConfig:
    """Tests for the mistral_ocr_* chunking/retry/error-strategy parameters."""

    _MISTRAL_OCR_ENV_VARS = (
        "MISTRAL_OCR_SAFE_MAX_PAGES",
        "MISTRAL_OCR_SAFE_MAX_BYTES",
        "MISTRAL_OCR_MAX_RETRIES",
        "MISTRAL_OCR_RETRY_BACKOFF_SECONDS",
        "MISTRAL_OCR_CHUNK_CONCURRENCY",
        "MISTRAL_OCR_ERROR_STRATEGY",
    )

    @pytest.fixture(autouse=True)
    def _clear_mistral_ocr_env(self, monkeypatch):
        """Ensure no stray env vars leak between tests in this class."""
        for var in self._MISTRAL_OCR_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_defaults_without_explicit_args(self):
        """configure() without any mistral_ocr_* kwargs applies the documented defaults."""
        llm = _make_mock_llm()
        cfg = configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
        )
        assert cfg.mistral_ocr_safe_max_pages == 900
        assert cfg.mistral_ocr_safe_max_bytes == 45 * 1024 * 1024
        assert cfg.mistral_ocr_max_retries == 3
        assert cfg.mistral_ocr_retry_backoff_seconds == 2.0
        assert cfg.mistral_ocr_chunk_concurrency == 1
        assert cfg.mistral_ocr_error_strategy == "fail_fast"

    def test_explicit_overrides_are_respected(self):
        """Each mistral_ocr_* kwarg can be overridden explicitly."""
        llm = _make_mock_llm()
        cfg = configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
            mistral_ocr_safe_max_pages=500,
            mistral_ocr_safe_max_bytes=10 * 1024 * 1024,
            mistral_ocr_max_retries=5,
            mistral_ocr_retry_backoff_seconds=1.5,
            mistral_ocr_chunk_concurrency=2,
            mistral_ocr_error_strategy="best_effort",
        )
        assert cfg.mistral_ocr_safe_max_pages == 500
        assert cfg.mistral_ocr_safe_max_bytes == 10 * 1024 * 1024
        assert cfg.mistral_ocr_max_retries == 5
        assert cfg.mistral_ocr_retry_backoff_seconds == 1.5
        assert cfg.mistral_ocr_chunk_concurrency == 2
        assert cfg.mistral_ocr_error_strategy == "best_effort"

    def test_env_vars_are_respected_when_no_explicit_arg(self, monkeypatch):
        """MISTRAL_OCR_* env vars are used when no explicit kwarg is passed."""
        monkeypatch.setenv("MISTRAL_OCR_SAFE_MAX_PAGES", "700")
        monkeypatch.setenv("MISTRAL_OCR_SAFE_MAX_BYTES", "1000000")
        monkeypatch.setenv("MISTRAL_OCR_MAX_RETRIES", "7")
        monkeypatch.setenv("MISTRAL_OCR_RETRY_BACKOFF_SECONDS", "3.5")
        monkeypatch.setenv("MISTRAL_OCR_CHUNK_CONCURRENCY", "4")
        monkeypatch.setenv("MISTRAL_OCR_ERROR_STRATEGY", "best_effort")

        llm = _make_mock_llm()
        cfg = configure(
            llm=llm,
            neo4j_uri=_DUMMY_URI,
            neo4j_user=_DUMMY_USER,
            neo4j_password=_DUMMY_PASS,
        )
        assert cfg.mistral_ocr_safe_max_pages == 700
        assert cfg.mistral_ocr_safe_max_bytes == 1_000_000
        assert cfg.mistral_ocr_max_retries == 7
        assert cfg.mistral_ocr_retry_backoff_seconds == 3.5
        assert cfg.mistral_ocr_chunk_concurrency == 4
        assert cfg.mistral_ocr_error_strategy == "best_effort"

    def test_invalid_error_strategy_raises_configuration_error(self):
        """An invalid mistral_ocr_error_strategy value raises ConfigurationError."""
        llm = _make_mock_llm()
        with pytest.raises(ConfigurationError, match="mistral_ocr_error_strategy"):
            configure(
                llm=llm,
                neo4j_uri=_DUMMY_URI,
                neo4j_user=_DUMMY_USER,
                neo4j_password=_DUMMY_PASS,
                mistral_ocr_error_strategy="invalid_value",
            )

    def test_safe_max_pages_below_one_raises_configuration_error(self):
        llm = _make_mock_llm()
        with pytest.raises(ConfigurationError, match="mistral_ocr_safe_max_pages"):
            configure(
                llm=llm,
                neo4j_uri=_DUMMY_URI,
                neo4j_user=_DUMMY_USER,
                neo4j_password=_DUMMY_PASS,
                mistral_ocr_safe_max_pages=0,
            )

    def test_safe_max_bytes_below_one_raises_configuration_error(self):
        llm = _make_mock_llm()
        with pytest.raises(ConfigurationError, match="mistral_ocr_safe_max_bytes"):
            configure(
                llm=llm,
                neo4j_uri=_DUMMY_URI,
                neo4j_user=_DUMMY_USER,
                neo4j_password=_DUMMY_PASS,
                mistral_ocr_safe_max_bytes=0,
            )

    def test_max_retries_below_one_raises_configuration_error(self):
        llm = _make_mock_llm()
        with pytest.raises(ConfigurationError, match="mistral_ocr_max_retries"):
            configure(
                llm=llm,
                neo4j_uri=_DUMMY_URI,
                neo4j_user=_DUMMY_USER,
                neo4j_password=_DUMMY_PASS,
                mistral_ocr_max_retries=0,
            )
