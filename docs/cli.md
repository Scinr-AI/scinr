# CLI Reference

The `newton` command runs document ingestion from a terminal. See [Configuration](configuration.md#cli-mode) for required environment settings.

## Usage

```bash
newton [OPTIONS]
```

## Options

| Flag | Description |
|---|---|
| `--stage` | Select `preprocess`, `extract`, `ingest`, `annotate`, `entity_extract`, `tabular`, or `all` (default). |
| `--input-raw` | Directory containing source files. |
| `--input` | Directory containing converted input files; defaults to `data/json/`. |
| `--output` | Directory for extracted output; defaults to `data/output/`. |
| `--document` | Document name for targeted annotation or extraction. |
| `--update` | Update the latest ingested version in place. |
| `--replaces` | Record a newly ingested document as replacing an existing document. |
| `--parallel-docs` | Number of documents to process concurrently. |
| `--only-unannotated` | Skip content that has already been prepared for extraction. |
| `--only-unextracted` | Skip content that has already been extracted. |
| `--manual` | Apply a specified extraction model. Requires `--model`. |
| `--model` | Extraction model class name used with `--manual`. |
| `--context` | Additional context about the input documents. |

## Examples

Run a complete ingestion:

```bash
newton --stage all --input-raw files/
```

Convert source files only:

```bash
newton --stage preprocess --input-raw files/
```

Run a targeted operation for an existing document:

```bash
newton --stage annotate --document "MyDocument"
```

Process tabular files:

```bash
newton --stage tabular --input-raw files/
```

Replace a document:

```bash
newton --stage all --input-raw files/new/ --replaces "OldDocumentName"
```
