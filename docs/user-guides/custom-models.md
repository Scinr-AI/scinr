# Custom Extraction Models

Define domain-specific Pydantic extraction models to extract structured entities from unstructured documents.

---

## Defining a Model

Inherit from `pydantic.BaseModel`. Field docstrings and descriptions serve as guidance for the LLM extraction engine.

> **Note**: Do not alter model field docstrings once registered, as they directly format the prompt schema.

```python
from pydantic import BaseModel, Field

class CompoundAssayResult(BaseModel):
    """Structured measurement of compound potency in a biological assay."""

    compound_name: str = Field(..., description="Canonical chemical or code name of tested compound.")
    target_protein: str = Field(..., description="Biological target protein or receptor.")
    ic50_nm: float | None = Field(None, description="Half maximal inhibitory concentration in nanomolar (nM).")
    assay_type: str = Field(..., description="Assay format, e.g., Binding, Enzymatic, Cell-based.")
```
