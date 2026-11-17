"""Golden harness theme."""
from __future__ import annotations

from golden_models import GoldenNormRow, GoldenProduct, GoldenRow, GoldenSupplier

THEME_DESCRIPTION: str = (
    "Synthetic theme of the graph-write golden harness: products, suppliers and "
    "tabular rows. Not used by any real document."
)

SELECTABLE_MODELS: list[type] = [GoldenProduct, GoldenSupplier, GoldenRow, GoldenNormRow]
