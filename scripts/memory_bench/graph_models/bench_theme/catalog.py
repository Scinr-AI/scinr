"""bench_graph.py theme."""
from __future__ import annotations

from bench_models import BenchItem, BenchWide

THEME_DESCRIPTION: str = "Synthetic models of the graph-write memory bench."

SELECTABLE_MODELS: list[type] = [BenchItem, BenchWide]
