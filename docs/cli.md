# CLI Reference

The `newton` command line tool orchestrates `scinr` ingestion pipeline tasks from your terminal.

---

## Commands

### `newton run`

Orchestrates full or partial ingestion pipeline runs.

```bash
newton run [OPTIONS]
```

#### Key Options

* `--input-raw PATH`: Directory containing input document files.
* `--stages LIST`: Comma-separated list of stages to run (e.g. `preprocess,extraction,ingestion`).
* `--model-class NAME`: Registered Pydantic target extraction model class name.
* `--parallel-docs INT`: Number of documents to process in parallel (default: `1`).
* `--neo4j-uri URI`: Neo4j Bolt connection URI.
* `--neo4j-user USER`: Neo4j database username.
* `--neo4j-pass PASS`: Neo4j database password.
* `--llm-provider NAME`: Provider name (`openai`, `bedrock`, `ollama`).
* `--llm-model NAME`: Model identifier (e.g., `gpt-4o`, `claude-3-5-sonnet`).

---

## Usage Examples

Run Stages 0 to 2 only:

```bash
newton run --input-raw ./data --stages preprocess,extraction,ingestion
```
