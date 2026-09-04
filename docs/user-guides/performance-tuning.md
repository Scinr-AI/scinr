# Performance Tuning

This guide covers every lever available to tune `scinr.newton` pipeline performance for production deployments. It walks through concurrency, batch sizes, prompt optimization, OCR tuning, scenario-based recommendations, and diagnostic techniques.

---

## Introduction

Performance tuning is critical for production deployments of `scinr.newton`. The pipeline processes documents through six stages — preprocess, extraction, ingestion, annotation, entity extraction, and tabular normalization — each with different resource profiles and bottlenecks.

### Three Levers

There are three primary levers for performance tuning:

1. **Concurrency** — How many operations run in parallel (LLM calls, Neo4j sessions, document processing).
2. **Batch Sizes** — How many items are grouped per LLM call (extraction pages, normalization entries).
3. **Prompt Optimization** — Reducing token waste via prompt caching and model-specific prompt families.

### The Trade-off Triangle

Every tuning decision involves a trade-off between:

| Dimension | What it means | How to optimize |
| :--- | :--- | :--- |
| **Speed** | Wall-clock time to process a batch | Increase concurrency, parallel docs |
| **Cost** | Token consumption and API charges | Increase batch sizes, use prompt caching |
| **Quality** | Accuracy of extraction and annotation | Lower batch sizes, appropriate prompt families |

The default configuration (`llm_concurrency=4`, `parallel_docs=5`, `extraction_batch_size=1`) is a conservative starting point that prioritizes quality and cost. Adjust from there based on your workload.

---

## Concurrency Tuning

Concurrency is the most impactful performance lever. The pipeline uses three independent semaphores to control parallelism at different layers.

### LLM Concurrency (`llm_concurrency`)

**Default:** `4`  **Env var:** `LLM_CONCURRENCY`

Controls the maximum number of simultaneous LLM calls across **all** pipeline stages. Every extraction chunk, annotation decision, entity extraction, and tabular normalization call acquires this semaphore before invoking the LLM.

```python
from scinr.newton import configure

configure(
    llm_concurrency=4,  # Default
)
```

#### How it works

The global `get_llm_semaphore()` is sized to `llm_concurrency`. All LLM calls across all stages — extraction, annotation, entity extraction, and tabular normalization — share this single semaphore. This prevents exceeding provider rate limits and connection pool limits.

#### Recommendations by workload

| Workload | Docs | `llm_concurrency` | Notes |
| :--- | :--- | :--- | :--- |
| Small | < 10 | 2-4 | Conservative; avoids rate limiting |
| Medium | 10-50 | 4-8 | Balanced speed and reliability |
| Large | 50+ | 8-16 | Requires checking provider limits |
| Very large | 100+ | 16-32 | Monitor rate limits closely |

#### Provider rate limits

| Provider | Default limit | Notes |
| :--- | :--- | :--- |
| AWS Bedrock | Varies by model | Check your account's TPMS/RPM limits |
| OpenAI | Varies by tier | Free tier: very restrictive; paid: higher |
| Ollama | Hardware-bound | Limited by your GPU/CPU capacity |

> **Warning:** Setting `llm_concurrency` too high will cause rate-limit errors. Start with the default (4) and increase gradually while monitoring for `429 Too Many Requests` responses.

### Neo4j Concurrency (`neo4j_concurrency`, `neo4j_sync_concurrency`)

**Default:** `10` / `8`  **Env vars:** `NEO4J_CONCURRENCY` / `NEO4J_SYNC_CONCURRENCY`

Two separate semaphores control Neo4j access:

```python
configure(
    neo4j_concurrency=10,       # Async operations (Stages 3, 4)
    neo4j_sync_concurrency=8,   # Sync operations (Stage 2)
)
```

#### `neo4j_concurrency` — Async Operations

Bounds concurrent Neo4j async sessions during annotation (Stage 3) and entity extraction (Stage 4). These stages use the async Neo4j driver for read-heavy operations.

#### `neo4j_sync_concurrency` — Sync Operations

Bounds concurrent dispatches to `asyncio.to_thread()` for Stage 2 (synchronous ingestion). The sync driver is used for document ingestion because the Neo4j Python driver's synchronous API handles bulk writes efficiently. The semaphore must be acquired and released on the event loop, never inside the worker thread.

#### Recommendations by graph size

| Graph size | `neo4j_concurrency` | `neo4j_sync_concurrency` | Notes |
| :--- | :--- | :--- | :--- |
| Small (< 1k nodes) | 5-10 | 5-8 | Default settings are fine |
| Medium (1k-10k) | 10-20 | 8-15 | Monitor Neo4j CPU |
| Large (10k-100k) | 20-30 | 15-25 | Watch memory and transaction logs |
| Very large (100k+) | 30-50 | 25-50 | Consider Neo4j cluster |

> **Tip:** If Neo4j CPU is consistently above 80% during ingestion, reduce `neo4j_sync_concurrency`. If the database has ample capacity and ingestion is slow, increase it.

### Parallel Documents (`parallel_docs`)

**Default:** `5`  **Set via:** `run_pipeline()` argument

Controls the number of documents processed concurrently across all stages. Each document goes through all stages independently, bounded by this semaphore for its entire duration.

```python
from scinr.newton import run_pipeline

# Process 10 documents concurrently
result = await run_pipeline(
    input_raw="./raw_docs",
    parallel_docs=10,
)

# Process one document at a time (sequential)
result = await run_pipeline(
    input_raw="./raw_docs",
    parallel_docs=1,
)
```

#### How it works

The pipeline uses `asyncio.Semaphore(parallel_docs)` at the document level. Each document is dispatched as an independent task via `asyncio.gather()` and runs through its applicable stages sequentially. Within each stage, additional concurrency control is provided by `llm_concurrency` and `neo4j_concurrency`.

#### Recommendations by document profile

| Profile | `parallel_docs` | Notes |
| :--- | :--- | :--- |
| Few large documents | 1-3 | Large docs consume more memory per unit |
| Many small documents | 5-10 | Parallelism helps significantly |
| Mixed sizes | 3-5 | Balanced approach |
| Memory-constrained | 1-2 | Reduce to avoid OOM |

> **Important:** `parallel_docs` is set per `run_pipeline()` call, not in `configure()`. This allows you to adjust parallelism per batch without changing global configuration.

---

## Batch Size Tuning

Batch sizes control how many items are grouped into a single LLM call. Larger batches reduce the number of API calls (lowering cost) but increase context size per call (potentially affecting quality).

### Extraction Batch Size (`extraction_batch_size`)

**Default:** `1`  **Env var:** `EXTRACTION_BATCH_SIZE`

The number of pages grouped per extraction LLM call. Each call sends the grouped pages to the LLM for structure parsing.

```python
configure(extraction_batch_size=1)  # Default: 1 page per chunk
```

#### Trade-offs

| Value | Pros | Cons |
| :--- | :--- | :--- |
| `1` (default) | Smallest context, highest quality | More LLM calls, higher cost |
| `2-3` | Fewer calls, moderate context | Slightly larger prompts |
| `4-5` | Fewest calls, lowest cost | Large context may reduce quality |
| `6+` | Maximum cost savings | Risk of context overflow, quality drop |

#### Recommendations by document type

| Document type | `extraction_batch_size` | Reasoning |
| :--- | :--- | :--- |
| Dense technical documents | 1 | Each page has significant content; keeping context small preserves quality |
| Sparse documents (mostly headers) | 2-3 | Pages are lighter; grouping is efficient |
| Very long documents (100+ pages) | 2-3 | Reduces call count significantly |
| Documents with complex tables | 1 | Tables need focused context; grouping may lose detail |

### Fast Extraction (`fast_extraction`)

**Default:** `False`  **Passed to:** `run_pipeline(fast_extraction=...)` — a per-call
`run_pipeline()` argument, not a `configure()` setting or environment variable
(see [Running the Pipeline](running-pipeline.md#fast_extraction) for why).

By default, Stage 1 extraction processes chunks **sequentially**: each chunk's LLM
call sees the deterministic `active_hierarchy` built from every chunk processed so
far, and the growing document tree is merged one chunk at a time
(`compact_extraction()`). When `fast_extraction=True`, all chunks are extracted **in
parallel** instead — each chunk is extracted independently, with cross-chunk
hierarchy resolution deferred to a single post-extraction consolidation LLM call
that decides every "orphan" node's true parent from the full cross-chunk node pool.

```python
result = await run_pipeline(
    input_raw="files/",
    stages=["preprocess", "extraction", "ingestion"],
    fast_extraction=True,  # opt-in: parallel Stage 1 chunks + consolidation
)
```

Two new `configure()` settings tune the consolidation call's token-ceiling
estimation (used only when `fast_extraction=True`):

```python
configure(
    consolidation_token_safety_margin=0.75,     # fraction of max_tokens reserved for output
    consolidation_max_output_tokens=None,       # explicit override; derived from the margin above if unset
    consolidation_max_input_tokens=None,        # explicit input-size ceiling; skipped (no check) if unset
)
```

#### Trade-offs

| Aspect | `fast_extraction=False` (default) | `fast_extraction=True` |
| :--- | :--- | :--- |
| Stage 1 wall-clock time | Sequential — each chunk waits for the previous one | Parallel — can reduce wall-clock time substantially for multi-chunk documents |
| Hierarchy resolution | Incremental, deterministic, per-chunk (`compact_extraction()`) | Deferred to one consolidation LLM call across the whole document |
| Failure blast radius | A bad chunk only affects that chunk's own nodes | A degenerate or partially-failed consolidation response affects every orphan's placement at once |
| Risk profile | Safe, well-tested legacy behavior | Concentrates hierarchy-correctness risk into one LLM call — recommended only after validating output quality against representative documents |

**Risk note:** When `True`, Stage 1 extraction runs chunks in parallel and defers
cross-chunk hierarchy resolution to a single post-extraction consolidation LLM call
instead of incremental deterministic prefix-matching. This can reduce Stage 1
wall-clock time substantially for multi-chunk documents, but concentrates
hierarchy-correctness risk into one LLM call — a degenerate or partially-failed
consolidation response has a larger blast radius than the default per-chunk
behavior. Recommended only after validating output quality against representative
documents. Default: `False` (safe, unchanged legacy behavior).

**Checkpoint artifact note:** While the Map phase runs, a `map-checkpoint-{doc_name}.json`
sibling file is written next to the eventual `extract-{doc_name}.json` output. It is
deleted automatically once consolidation succeeds, but if the consolidation call fails
(or the process crashes) after the Map phase completes, that checkpoint file may be
left behind in the output directory. This is currently a manual-recovery artifact
only — there is no automatic resume logic in this version, so a leftover checkpoint
file can simply be deleted once you've re-run extraction for that document.

### Normalization Batch Size (`normalization_batch_size`)

**Default:** `5`  **Env var:** `NORMALIZATION_BATCH_SIZE`

The number of normalization entries grouped per LLM call in the tabular pipeline. Each call sends multiple table entries to the LLM for structural normalization.

```python
configure(normalization_batch_size=5)  # Default: 5 entries per call
```

#### Trade-offs

| Value | Pros | Cons |
| :--- | :--- | :--- |
| `3-5` (default) | Balanced quality and cost | Moderate number of calls |
| `10-15` | Fewer calls, lower cost | Larger context per call |
| `15-20` | Maximum cost savings | Risk of context overflow |
| `20+` | Fewest calls | May exceed model context window |

#### Recommendations by complexity

| Normalization complexity | `normalization_batch_size` | Reasoning |
| :--- | :--- | :--- |
| Simple (few columns, clean data) | 10-20 | Entries are small; batching is efficient |
| Complex (many columns, messy data) | 3-5 | Each entry needs more context |
| Cost-sensitive | 10-15 | Good balance of cost and quality |
| Quality-critical | 3-5 | Smaller batches = more focused normalization |

---

## Prompt Optimization

Prompt optimization reduces token waste and improves LLM response quality through caching and model-specific formatting.

### Prompt Caching (Bedrock)

**Default:** `True`  **Env var:** `PROMPT_CACHING_ENABLED`

Caches system prompts across LLM calls. When enabled, the system prompt (which is identical across calls within a stage) is cached on the Bedrock side, reducing both latency and token costs.

```python
configure(prompt_caching_enabled=True)  # Default: True
```

#### When it helps

| Scenario | Impact |
| :--- | :--- |
| Many small extraction calls | High — system prompt is a large fraction of total tokens |
| Few large extraction calls | Low — system prompt is a small fraction |
| Annotation stage | Moderate — repeated system prompt across many nodes |
| Entity extraction | Moderate — repeated system prompt across many nodes |

#### When to disable

- **Non-Bedrock providers:** Prompt caching is a Bedrock-specific feature. It is ignored for OpenAI, Ollama, and other providers.
- **Very few LLM calls:** If you process only 1-2 documents with few pages, caching overhead may outweigh benefits.
- **Frequently changing prompts:** If you use `context_instructions` that change between runs, caching may be less effective.

```python
# Disable prompt caching (rarely needed)
configure(prompt_caching_enabled=False)
```

### Prompt Families

**Default:** `generic`  **Env var:** `PROMPT_FAMILY`

The `prompt_family` parameter selects a set of prompt templates optimized for different LLM providers.

```python
configure(prompt_family="generic")  # Default
```

| Family | Best for | Characteristics |
| :--- | :--- | :--- |
| `generic` | Any model | Standard prompt format; safe default that works with any LLM |
| `claude` | Anthropic Claude models | Optimized for Claude reasoning; uses Claude-specific formatting and system prompt conventions |
| `gpt_reasoning` | OpenAI o-series | Optimized for reasoning models; uses the specific message structure required by reasoning-capable models |

#### Choosing a prompt family

- **Claude models on Bedrock** — use `"claude"` for best results.
- **OpenAI o-series models** — use `"gpt_reasoning"`.
- **Other providers or unsure** — use `"generic"` (the default).

```python
# Claude on Bedrock
configure(
    prompt_family="claude",
    prompt_caching_enabled=True,
)

# OpenAI o-series
configure(
    prompt_family="gpt_reasoning",
)

# Ollama or other local model
configure(
    prompt_family="generic",
)
```

> **Tip:** Using the wrong prompt family with a model can degrade extraction quality significantly. Always match the prompt family to your model.

---

## Mistral OCR Tuning

PDF processing uses a two-path strategy: small PDFs are processed with `pdfplumber` (fast, no API cost), while large or complex PDFs use Mistral OCR (slower, API cost). The tuning parameters control this boundary and the OCR behavior itself.

```python
configure(
    mistral_ocr_safe_max_pages=900,        # Pages threshold
    mistral_ocr_safe_max_bytes=47185920,   # Size threshold (45 MiB)
    mistral_ocr_max_retries=3,             # Retry count
    mistral_ocr_retry_backoff_seconds=2.0, # Retry backoff
    mistral_ocr_chunk_concurrency=1,       # OCR chunk concurrency
    mistral_ocr_error_strategy="best_effort",  # Error handling
)
```

### `mistral_ocr_safe_max_pages`

**Default:** `900`  **Env var:** `MISTRAL_OCR_SAFE_MAX_PAGES`

PDFs with fewer pages than this threshold use `pdfplumber` (fast, no API cost) if they contain extractable text. PDFs at or above this threshold use Mistral OCR regardless.

| Value | Effect |
| :--- | :--- |
| Lower (e.g., 100) | More PDFs use OCR; better quality for scanned docs |
| Higher (e.g., 2000) | More PDFs use pdfplumber; faster, cheaper |
| Default (900) | Balanced; most PDFs use pdfplumber |

### `mistral_ocr_safe_max_bytes`

**Default:** `47185920` (45 MiB)  **Env var:** `MISTRAL_OCR_SAFE_MAX_BYTES`

PDFs larger than this threshold always use Mistral OCR, regardless of page count. Large files are more likely to be scanned images or have complex layouts that benefit from OCR.

| Value | Effect |
| :--- | :--- |
| Lower (e.g., 10 MiB) | More files use OCR |
| Higher (e.g., 100 MiB) | Fewer files use OCR |
| Default (45 MiB) | Balanced |

### `mistral_ocr_max_retries`

**Default:** `3`  **Env var:** `MISTRAL_OCR_MAX_RETRIES`

Number of retry attempts for OCR failures. Each retry uses exponential backoff based on `mistral_ocr_retry_backoff_seconds`.

### `mistral_ocr_retry_backoff_seconds`

**Default:** `2.0`  **Env var:** `MISTRAL_OCR_RETRY_BACKOFF_SECONDS`

Base backoff in seconds between retries. Actual backoff is exponential: first retry waits 2s, second waits 4s, third waits 8s.

### `mistral_ocr_chunk_concurrency`

**Default:** `1`  **Env var:** `MISTRAL_OCR_CHUNK_CONCURRENCY`

Number of concurrent OCR chunk processing tasks. A large PDF is split into chunks for parallel OCR processing.

| Value | Effect |
| :--- | :--- |
| `1` (default) | Sequential processing; safest, most predictable |
| `2-4` | Parallel chunks; faster for very large PDFs |
| `4+` | Maximum parallelism; may hit Mistral rate limits |

> **Warning:** Increasing `mistral_ocr_chunk_concurrency` increases API calls proportionally. Monitor Mistral rate limits.

### `mistral_ocr_error_strategy`

**Default:** `"fail_fast"`  **Env var:** `MISTRAL_OCR_ERROR_STRATEGY`

Controls error handling during OCR processing.

| Value | Behavior |
| :--- | :--- |
| `"fail_fast"` | Stops processing the document on first OCR error |
| `"best_effort"` | Continues processing and collects whatever OCR results are available |

For production pipelines processing many documents, `"best_effort"` is recommended to avoid a single OCR failure blocking the entire batch.

---

## Scenario-Based Recommendations

### Small Scale (1-10 documents)

Lightweight configuration for development, testing, or small document sets.

```python
from scinr.newton import configure, run_pipeline

configure(
    llm_concurrency=2,           # Conservative LLM calls
    neo4j_concurrency=5,         # Light Neo4j load
    neo4j_sync_concurrency=5,    # Light sync load
    extraction_batch_size=1,     # Maximum quality
    normalization_batch_size=5,  # Default
    prompt_caching_enabled=True, # Still useful for small batches
)

result = await run_pipeline(
    input_raw="./docs",
    parallel_docs=2,             # Low document parallelism
)
```

**Expected profile:**
- Speed: Moderate (quality-focused)
- Cost: Higher per document (small batches)
- Quality: Maximum (small context per call)

### Medium Scale (10-100 documents)

Balanced configuration for typical production workloads.

```python
configure(
    llm_concurrency=8,           # More parallel LLM calls
    neo4j_concurrency=15,        # Moderate Neo4j load
    neo4j_sync_concurrency=12,   # Moderate sync load
    extraction_batch_size=2,     # Slightly larger batches
    normalization_batch_size=10, # Larger normalization batches
    prompt_caching_enabled=True, # Significant savings at this scale
)

result = await run_pipeline(
    input_raw="./docs",
    parallel_docs=5,             # Default parallelism
)
```

**Expected profile:**
- Speed: Good balance
- Cost: Moderate (larger batches reduce calls)
- Quality: High (still reasonable context sizes)

### Large Scale (100+ documents)

Maximum throughput configuration for large document sets.

```python
configure(
    llm_concurrency=16,          # High parallel LLM calls
    neo4j_concurrency=25,        # High Neo4j throughput
    neo4j_sync_concurrency=20,   # High sync throughput
    extraction_batch_size=2,     # Larger extraction batches
    normalization_batch_size=15, # Large normalization batches
    prompt_caching_enabled=True, # Critical for cost at this scale
)

result = await run_pipeline(
    input_raw="./docs",
    parallel_docs=10,            # High document parallelism
)
```

**Expected profile:**
- Speed: Maximum (high concurrency everywhere)
- Cost: Lower per document (large batches, prompt caching)
- Quality: Good (slightly larger contexts)

> **Warning:** At large scale, monitor Neo4j CPU, memory, and transaction logs. You may need to tune Neo4j database settings independently.

### Cost-Sensitive

Minimize API costs while maintaining acceptable quality.

```python
configure(
    llm_concurrency=2,            # Fewer parallel calls (no rush)
    extraction_batch_size=3,      # More pages per call
    normalization_batch_size=15,  # Larger normalization batches
    prompt_caching_enabled=True,  # Cache prompts (Bedrock)
    mistral_ocr_safe_max_pages=2000,  # Prefer pdfplumber over OCR
    mistral_ocr_safe_max_bytes=104857600,  # 100 MiB threshold
)

result = await run_pipeline(
    input_raw="./docs",
    parallel_docs=3,              # Moderate parallelism
)
```

**Cost-saving strategies:**
- Larger extraction batches = fewer LLM calls per document
- Larger normalization batches = fewer LLM calls per table
- Higher OCR thresholds = more pdfplumber usage (free)
- Prompt caching = reduced token costs on Bedrock
- Lower concurrency = no wasted retries from rate limiting

### Speed-Sensitive

Minimize wall-clock time while maintaining acceptable cost.

```python
configure(
    llm_concurrency=16,          # Max parallel LLM calls
    neo4j_concurrency=30,        # Max Neo4j throughput
    neo4j_sync_concurrency=25,   # Max sync throughput
    extraction_batch_size=1,     # Smaller batches = faster per call
    normalization_batch_size=5,  # Default batch size
    mistral_ocr_chunk_concurrency=4,  # Parallel OCR
    mistral_ocr_error_strategy="best_effort",  # Don't wait on failures
)

result = await run_pipeline(
    input_raw="./docs",
    parallel_docs=10,            # Max document parallelism
)
```

**Speed-optimization strategies:**
- Maximum concurrency at every layer
- Smaller extraction batches = faster individual LLM calls
- Parallel OCR chunks = faster PDF processing
- `best_effort` error strategy = no blocking on failures
- Higher `parallel_docs` = more documents in flight

---

## Monitoring and Diagnostics

### Pipeline Result Inspection

`run_pipeline()` returns a `PipelineResult` with structured per-stage metrics. Use these to identify bottlenecks.

```python
result = await run_pipeline(input_raw="./docs")

# Overall timing
print(f"Total: {result.total_duration_seconds:.1f}s")

# Per-stage timing
for stage_name in result.stages_executed:
    stage = getattr(result, stage_name)
    if stage:
        print(f"{stage_name}: {stage.duration_seconds:.1f}s, "
              f"{stage.total_processed} processed, {stage.total_failed} failed")
```

### Detailed Per-Stage Breakdown

```python
result = await run_pipeline(input_raw="./docs")

# Calculate stage percentages
total = result.total_duration_seconds
for stage_name in result.stages_executed:
    stage = getattr(result, stage_name)
    if stage and total > 0:
        pct = (stage.duration_seconds / total) * 100
        print(f"{stage_name:>20s}: {stage.duration_seconds:6.1f}s ({pct:5.1f}%)")
```

### Bottleneck Identification

| Symptom | Likely Bottleneck | Tuning Action |
| :--- | :--- | :--- |
| Stage 0 (preprocess) is the slowest stage | PDF conversion or OCR bottleneck | Increase `mistral_ocr_chunk_concurrency`, check Mistral API |
| Stage 1 (extraction) is the slowest stage | LLM concurrency too low | Increase `llm_concurrency` |
| Stage 2 (ingestion) is the slowest stage | Neo4j sync concurrency too low | Increase `neo4j_sync_concurrency` |
| Stage 3 (annotation) is the slowest stage | LLM concurrency too low | Increase `llm_concurrency` |
| Stage 4 (entity extraction) is the slowest stage | LLM concurrency too low | Increase `llm_concurrency` |
| High LLM costs per document | Batch sizes too small | Increase `extraction_batch_size` |
| OCR timeout errors | Large PDFs, low concurrency | Increase `mistral_ocr_chunk_concurrency` |
| Memory errors (OOM) | Too many parallel documents | Decrease `parallel_docs` |
| Neo4j connection pool exhausted | Neo4j concurrency too high | Decrease `neo4j_concurrency` |
| Rate limit errors (429) | LLM concurrency too high | Decrease `llm_concurrency` |
| Inconsistent extraction quality | Batch size too large | Decrease `extraction_batch_size` |

### Finding the Dominant Stage

The stage that consumes the most time is your primary bottleneck. Focus tuning efforts there:

```python
result = await run_pipeline(input_raw="./docs")

# Find the slowest stage
slowest = max(
    (getattr(result, name) for name in result.stages_executed),
    key=lambda s: s.duration_seconds if s else 0
)

print(f"Bottleneck: {slowest.stage} ({slowest.duration_seconds:.1f}s)")
```

### Monitoring Concurrency Saturation

If a stage is slow but you suspect concurrency is the issue, check if you're hitting semaphore limits:

```python
from scinr.newton import get_config

config = get_config()
print(f"LLM semaphore size: {config.llm_concurrency}")
print(f"Neo4j async semaphore: {config.neo4j_concurrency}")
print(f"Neo4j sync semaphore: {config.neo4j_sync_concurrency}")
```

If the LLM stage is slow and `llm_concurrency` is at its maximum, try increasing it. If Neo4j stages are slow and the semaphore is maxed, try increasing the relevant Neo4j concurrency.

---

## Provider-Specific Tuning

### AWS Bedrock

Bedrock is the primary supported provider with the most tuning options.

```python
configure(
    # Prompt caching — critical for cost at scale
    prompt_caching_enabled=True,

    # Use a cheaper model for repair/retry operations
    # repair_llm=ChatBedrockConverse(model="us.anthropic.claude-haiku-3"),

    # Set appropriate MAX_TOKENS for your model
    # (via environment variable: MAX_TOKENS=65536)

    # Prompt family for Claude models
    prompt_family="claude",

    # Concurrency tuned for Bedrock rate limits
    llm_concurrency=8,
)
```

**Bedrock-specific tips:**
- **Prompt caching** is the single biggest cost reducer. Always keep it enabled.
- **REPAIR_MODEL_ID** can be set to a cheaper/faster model (e.g., Claude Haiku) for JSON repair operations, saving cost on retries.
- **MAX_TOKENS** should be set appropriately for your model. Too high wastes tokens; too low truncates responses.
- Check your account's **TPMS** (Tokens Per Minute) and **RPM** (Requests Per Minute) limits in the AWS Console.

### OpenAI

OpenAI has different rate limit characteristics and prompt requirements.

```python
from langchain_openai import ChatOpenAI
from scinr.newton import configure

configure(
    llm=ChatOpenAI(model="gpt-4o"),

    # Use gpt_reasoning family for o-series models
    prompt_family="gpt_reasoning",  # for o-series
    # prompt_family="generic",      # for gpt-4o, etc.

    # OpenAI rate limits vary by tier
    llm_concurrency=4,  # Start conservative
)
```

**OpenAI-specific tips:**
- **Rate limits** vary significantly by tier. Free tier is very restrictive; paid tiers are more permissive. Check your dashboard.
- Use `gpt_reasoning` prompt family for o-series models (o1, o3, etc.). These models require a specific message structure.
- Use `generic` prompt family for standard models (gpt-4o, gpt-4o-mini, etc.).
- Prompt caching is not available; consider larger batch sizes to compensate.

### Ollama (Local Models)

Local models have no rate limits but are hardware-bound.

```python
from langchain_ollama import ChatOllama
from scinr.newton import configure

configure(
    llm=ChatOllama(model="llama3"),

    # No rate limits, but hardware-bound
    llm_concurrency=2,  # Start low; increase if GPU has headroom

    # Generic prompt family for local models
    prompt_family="generic",

    # Prompt caching not applicable
    prompt_caching_enabled=False,
)
```

**Ollama-specific tips:**
- **No rate limits** — you can set `llm_concurrency` as high as your hardware allows.
- **GPU-bound** — each concurrent LLM call consumes GPU memory. Start with `llm_concurrency=2` and increase if you have headroom.
- **CPU-bound** — if running on CPU, keep `llm_concurrency=1` to avoid context-switching overhead.
- **Model size matters** — larger models (70B+) may only support 1 concurrent call on consumer hardware.
- Prompt caching is not applicable for local models.

---

## Advanced Tuning Patterns

### Two-Tier Configuration

Use different concurrency settings for different pipeline phases by reconfiguring between runs:

```python
from scinr.newton import configure, run_pipeline

# Phase 1: Preprocess + Extraction (LLM-heavy)
configure(
    llm_concurrency=16,           # Max LLM calls for extraction
    neo4j_concurrency=5,          # Minimal Neo4j (not used yet)
)
result1 = await run_pipeline(
    input_raw="./docs",
    extraction_output_dir="./data/extracted/",
    stages=["preprocess", "extraction"],
    parallel_docs=10,
)

# Phase 2: Ingestion (Neo4j-heavy)
configure(
    llm_concurrency=2,            # Minimal LLM (not used)
    neo4j_concurrency=25,         # Max Neo4j for ingestion
    neo4j_sync_concurrency=20,
)
result2 = await run_pipeline(
    ingestion_input_dir="./data/extracted/",
    stages=["ingestion"],
    parallel_docs=10,
)

# Phase 3: Annotation + Extraction (LLM-heavy again)
configure(
    llm_concurrency=16,           # Max LLM calls
    neo4j_concurrency=15,         # Moderate Neo4j
)
result3 = await run_pipeline(
    stages=["annotation", "entity_extraction"],
    document_names_dir="./data/extracted/",
    parallel_docs=10,
)
```

### Adaptive Concurrency

Dynamically adjust concurrency based on pipeline feedback:

```python
import asyncio
from scinr.newton import configure, run_pipeline

async def adaptive_run(input_dir: str, max_docs: int = 100) -> None:
    # Start conservative
    configure(
        llm_concurrency=4,
        neo4j_concurrency=10,
    )

    # Process a small batch first
    result = await run_pipeline(
        input_raw=input_dir,
        parallel_docs=3,
    )

    # Analyze results
    for stage_name in result.stages_executed:
        stage = getattr(result, stage_name)
        if stage and stage.total_failed == 0:
            # No failures — safe to increase concurrency
            pass
        elif stage and stage.total_failed > 0:
            # Failures detected — keep conservative settings
            print(f"Warning: {stage_name} had failures. "
                  f"Keeping conservative concurrency.")
            return

    # If first batch was clean, increase concurrency for the main run
    configure(
        llm_concurrency=12,
        neo4j_concurrency=20,
    )

    result = await run_pipeline(
        input_raw=input_dir,
        parallel_docs=8,
    )

    print(f"Adaptive run complete: {result.success}")
```

### Batch Processing with Progress Tracking

For very large document sets, process in batches with progress reporting:

```python
import asyncio
from pathlib import Path
from scinr.newton import configure, run_pipeline

async def batch_run(input_dir: str, batch_size: int = 20) -> None:
    configure(
        llm_concurrency=8,
        neo4j_concurrency=15,
        parallel_docs=5,
    )

    files = list(Path(input_dir).rglob("*"))
    files = [f for f in files if f.is_file()]

    total = len(files)
    processed = 0

    for i in range(0, total, batch_size):
        batch = files[i:i + batch_size]
        batch_dir = f"./temp_batch_{i // batch_size}"
        Path(batch_dir).mkdir(exist_ok=True)

        # Copy batch files to temp directory
        for f in batch:
            import shutil
            shutil.copy(f, batch_dir)

        result = await run_pipeline(
            input_raw=batch_dir,
            parallel_docs=5,
        )

        processed += len(batch)
        print(f"Batch {i // batch_size + 1}: "
              f"{processed}/{total} files "
              f"({result.total_duration_seconds:.1f}s)")

        # Clean up temp directory
        import shutil
        shutil.rmtree(batch_dir)
```

---

## See Also

- **[Configuration](../configuration.md)** — Complete reference for `configure()`, environment variables, and all settings.
- **[Running the Pipeline](running-pipeline.md)** — Full `run_pipeline()` reference including `parallel_docs` and other parameters.
- **[Architecture](../architecture.md)** — Detailed walkthrough of concurrency layers, semaphores, and async design.
- **[Tabular Pipeline](tabular-pipeline.md)** — Tabular normalization performance and `normalization_batch_size` tuning.
- **[Custom Models](custom-models.md)** — Defining extraction models that affect annotation and extraction stage performance.
- **[Pipeline API](../api/pipeline.md)** — Auto-generated docstring for `run_pipeline()`.
- **[Results API](../api/results.md)** — Auto-generated documentation for `PipelineResult`, `StageResult`, and `DocumentResult`.
