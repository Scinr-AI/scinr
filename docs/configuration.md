# Configuration

Configure `scinr` with `configure()` in library mode or environment variables in CLI mode.

## Library mode

Call `configure()` once before using pipeline functions:

```python
from scinr.newton import configure

configure(
    llm=my_llm,
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="your-password",
)
```

Explicit arguments take precedence over environment variables, which take precedence over defaults.

### Common options

| Parameter | Purpose |
|---|---|
| `llm` | Processing model client used by the library. |
| `neo4j_uri` | Neo4j connection URI. |
| `neo4j_user` | Neo4j username. |
| `neo4j_password` | Neo4j password. |
| `storage_backend` | Optional raw-document storage backend. |
| `extra_converters` | Custom converters for additional file extensions. |
| `extra_models_paths` | Locations containing custom extraction models. |
| `llm_concurrency` | Maximum concurrent document-processing requests. |
| `neo4j_concurrency` | Maximum concurrent Neo4j write sessions. |

Use `configure()` to set any additional library options required by your deployment.

## CLI mode

The `newton` command reads configuration from environment variables:

| Variable | Purpose |
|---|---|
| `MODEL_ID` | Required runtime model identifier. |
| `NEO4J_URI` | Neo4j connection URI; defaults to `bolt://localhost:7687`. |
| `NEO4J_USER` | Neo4j username. |
| `NEO4J_PASSWORD` | Neo4j password. |
| `NEO4J_AUTH` | Alternative combined username/password value. |
| `STORAGE_BACKEND` | Optional raw-document storage backend. |
| `MONGODB_URI` | Connection URI when using the MongoDB storage backend. |
| `MONGODB_DATABASE` | Database name when using the MongoDB storage backend. |
| `LLM_CONCURRENCY` | Maximum concurrent document-processing requests. |
| `NEO4J_CONCURRENCY` | Maximum concurrent Neo4j write sessions. |
| `SCINR_EXTRA_MODELS_PATHS` | Colon-separated locations containing custom extraction models. |
