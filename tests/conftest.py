"""
tests/conftest.py — Shared fixtures for the scinr-ingest test suite.

The import chain scinr.newton.__init__ → cli → (many heavy modules) triggers
get_config() at module load time, which requires a configured LLM. We break
this chain by pre-stubbing scinr.newton.cli in sys.modules before any test
module is collected.

All module-level statements here run before pytest collects any test file.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Step 1: Patch dotenv.load_dotenv to a no-op BEFORE any scinr.newton import
# ---------------------------------------------------------------------------

try:
    import dotenv as _dotenv_module
    _dotenv_module.load_dotenv = lambda *a, **kw: None
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Step 2: Clear environment variables that trigger Bedrock LLM construction
# ---------------------------------------------------------------------------

os.environ.pop("MODEL_ID", None)
os.environ.pop("REPAIR_MODEL_ID", None)
os.environ.pop("STORAGE_BACKEND", None)

# ---------------------------------------------------------------------------
# Step 3: Pre-stub scinr.newton.cli in sys.modules so that __init__.py's
# "from scinr.newton.cli import ..." does not trigger the heavy import chain.
# ---------------------------------------------------------------------------


def _make_stub_module(name: str, **attrs) -> types.ModuleType:
    """Create a minimal stub module with the given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# Stub for scinr.newton.cli — provides all symbols that __init__.py imports
_cli_stub = _make_stub_module(
    "scinr.newton.cli",
    run_preprocess=AsyncMock(),
    run_extraction=AsyncMock(),
    run_ingestion=AsyncMock(),
    run_annotation=AsyncMock(),
    run_entity_extraction=AsyncMock(),
    run_tabular_pipeline=AsyncMock(),
    main=MagicMock(),
)
sys.modules["scinr.newton.cli"] = _cli_stub


# ---------------------------------------------------------------------------
# Config isolation — reset global _config after every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_config():
    """Reset the global scinr.newton config singleton after each test."""
    import scinr.newton.config as cfg_module

    original_config = cfg_module._config
    original_semaphore = cfg_module._llm_semaphore
    yield
    cfg_module._config = original_config
    cfg_module._llm_semaphore = original_semaphore

    # Also reset the theme registry so it doesn't bleed between tests
    try:
        import scinr.newton.utils.theme_registry as tr_module
        tr_module._registry_instance = None
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CSV / XLSX file factories
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_csv_file(tmp_path):
    """Factory fixture: returns a callable make_csv(content, name) -> Path."""

    def make_csv(content: str, name: str = "test.csv") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    return make_csv


@pytest.fixture
def tmp_xlsx_file(tmp_path):
    """Factory fixture: returns a callable make_xlsx(rows, name) -> Path.

    rows: list of lists (first row = headers).
    Skips if openpyxl is not installed.
    """
    openpyxl = pytest.importorskip("openpyxl")

    def make_xlsx(rows: list[list], name: str = "test.xlsx") -> Path:
        wb = openpyxl.Workbook()
        ws = wb.active
        for row in rows:
            ws.append(row)
        p = tmp_path / name
        wb.save(str(p))
        return p

    return make_xlsx


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """Return a MagicMock that satisfies BaseChatModel duck-typing."""
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
        llm = MagicMock()
        llm.with_structured_output = MagicMock()
        return llm
