# Getting Started

## Installation

Install `scinr` using `pip` or `uv`:

```bash
pip install scinr
```

Or with `uv`:

```bash
uv add scinr
```

### Optional Provider Dependencies

`scinr` provides optional extras for different LLM providers and storage backends:

```bash
# OpenAI support
pip install "scinr[openai]"

# AWS Bedrock support
pip install "scinr[bedrock]"

# Ollama support
pip install "scinr[ollama]"

# MongoDB storage support
pip install "scinr[mongodb]"

# Install all optional dependencies
pip install "scinr[openai,bedrock,ollama,mongodb]"
```

---

## Prerequisites

Before running the full extraction pipeline, ensure you have:

1. **Python 3.11+** installed.
2. **Neo4j 5.0+** instance running (Bolt protocol).
3. **MongoDB 4.6+** (optional, for persistent document storage).
4. API keys for your preferred LLM provider (e.g., `OPENAI_API_KEY`, `AWS_ACCESS_KEY_ID`).

---

## First Ingestion Run

Create a raw data directory with document files:

```bash
mkdir raw_docs
# Place sample PDF, DOCX, or XLSX files in raw_docs/
```

Run via the `newton` CLI:

```bash
newton run \
  --input-raw ./raw_docs \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-pass password \
  --llm-provider openai \
  --llm-model gpt-4o
```
