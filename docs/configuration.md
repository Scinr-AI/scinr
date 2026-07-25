# Configuration

Configure `scinr` programmatic execution using the `configure()` function or environment variables.

---

## Environment Variables

| Variable | Description | Default |
| :--- | :--- | :--- |
| `NEO4J_URI` | Bolt URI for Neo4j instance | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j database user | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j user password | — |
| `OPENAI_API_KEY` | OpenAI API key for LLM calls | — |
| `MONGODB_URI` | Connection string for MongoDB storage | `mongodb://localhost:27017` |

---

## Programmatic Configuration

```python
from scinr.newton import configure, get_config

configure(
    neo4j_uri="bolt://localhost:7687",
    neo4j_auth=("neo4j", "secret_pass"),
    llm_provider="openai",
    llm_model="gpt-4o",
    parallel_docs=4,
)

config = get_config()
print(f"Active Provider: {config.llm_provider}")
```
