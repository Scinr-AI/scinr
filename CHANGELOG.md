# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-01

### Added
- `configure()` API for provider-agnostic LLM configuration (T-01)
- Exception hierarchy: `ScinrError`, `ConfigurationError`, `PreconditionError`, `ExtractionError`, `IngestionError`, `ModelError`, `StorageError`, `ConversionError` (T-02)
- `ThemeRegistry` with lazy loading, `enabled_base_themes` filtering, and external entry-point discovery (T-03)
- LLM decoupling: `make_llm()` abstraction replaces direct Bedrock coupling (T-04)
- `llm_retry` generalized for Bedrock, OpenAI, and Anthropic (T-05)
- Storage Null Object pattern: `NullRawFileRepository`, `NullPageRepository` (T-06)
- Stage preconditions with actionable error messages (T-08)
- CSV auto-detect separator, UTF-8-BOM support, duplicate header deduplication (T-14)
- MongoDB connection health check at startup (T-13)
- Custom storage backend registration via `configure(custom_storage=...)` (T-13)
- Custom converter registration via `configure(extra_converters=...)` (T-12)
- `ModelField` MERGE key fixed to composite `{name, model}` (T-18)
- `src/` package layout with `scinr.newton` namespace (T-11)

### Fixed
- Silent errors in storage initialization (T-09)
- PDF converter now shows actionable error when `MISTRAL_API_KEY` is missing (T-10)
- `.env.example` corrected with all required variables (T-07)
