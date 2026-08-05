# Custom Extraction Models

Create extraction models when your documents contain domain-specific data that should be represented consistently in the graph.

## Define a model

Models are Pydantic classes derived from `ExtractionModel`. Use clear field names, types, and descriptions:

```python
from pydantic import Field
from scinr.newton.models.base import ExtractionModel

class CompoundAssayResult(ExtractionModel):
    compound_name: str = Field(description="Canonical name of the tested compound.")
    target_protein: str = Field(description="Biological target protein or receptor.")
    ic50_nm: float | None = Field(default=None, description="Half maximal inhibitory concentration in nM.")
```

## Make models available

Store related models together and configure their location with `extra_models_paths`:

```python
from scinr.newton import configure

configure(
    llm=my_llm,
    neo4j_user="neo4j",
    neo4j_password="...",
    extra_models_paths=["/path/to/custom/models"],
)
```

Use stable model and field names once data has been ingested so downstream graph queries remain consistent.
