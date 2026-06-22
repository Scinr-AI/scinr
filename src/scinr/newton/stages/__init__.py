"""
stages/__init__.py — Re-exports all public stage functions and helpers.

This package replaces the former flat ``stages.py`` module. Each stage lives in
its own sub-module; this init re-exports everything so existing imports of the
form ``from scinr.newton.stages import run_preprocess`` continue to work without
any changes in calling code.
"""

from scinr.newton.stages.annotation import run_annotation
from scinr.newton.stages.entity_extraction import run_entity_extraction
from scinr.newton.stages.extraction import run_extraction
from scinr.newton.stages.ingestion import apply_replacement, preflight_check_replaces, run_ingestion
from scinr.newton.stages.preprocess import run_preprocess
from scinr.newton.stages.tabular import run_tabular_pipeline

__all__ = [
    "run_preprocess",
    "run_extraction",
    "run_ingestion",
    "run_annotation",
    "run_entity_extraction",
    "run_tabular_pipeline",
    "preflight_check_replaces",
    "apply_replacement",
]
