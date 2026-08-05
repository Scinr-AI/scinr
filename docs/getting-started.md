# Getting Started

## Installation

```bash
pip install scinr
```

## Prerequisites

Before running an ingestion:

1. Install Python 3.11 or later.
2. Start a Neo4j 5.0 or later instance.
3. Configure a supported model provider.

## Choose a model provider

`scinr` accepts LangChain chat models that support structured output. Create one of the following clients, then pass it to `configure(llm=...)`.

### OpenAI

```bash
pip install "scinr[openai]"
```

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
```

### AWS Bedrock

```bash
pip install "scinr[bedrock]"
```

```python
from langchain_aws import ChatBedrockConverse

llm = ChatBedrockConverse(model="your-model-id", region_name="your-region")
```

### Ollama

```bash
pip install "scinr[ollama]"
```

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="your-local-model")
```

### Anthropic

```bash
pip install langchain-anthropic
```

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="your-model-id")
```

Other compatible LangChain chat models can be used in the same way.

## First ingestion

```python
import asyncio
from langchain_openai import ChatOpenAI
from scinr.newton import configure, run_pipeline

async def main():
    configure(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="your-password",
    )
    result = await run_pipeline(input_raw="./raw_docs")
    print(result.success)

asyncio.run(main())
```

Place supported documents in `raw_docs/` and pass that directory to `run_pipeline()`. CSV, XLSX, and XLS files are automatically routed as tabular input; other supported formats follow the document-ingestion path.

## Command line

The CLI reads its settings from the environment. Provide the required runtime and Neo4j settings, then run:

```bash
newton --stage all --input-raw ./raw_docs
```

See [Configuration](configuration.md) for environment variables and [CLI Reference](cli.md) for commands and options.
