# API Reference Overview

Welcome to the `scinr.newton` API reference. All API documentation is auto-generated from Python docstrings using mkdocstrings.

## Core Modules

- [Pipeline](pipeline.md): ``run_pipeline()`` orchestrator.
- [Configuration](config.md): ``configure()``, ``get_config()``, ``ScinrConfig``.
- [Stages](stages.md): Individual stage runner functions (Stages 0-5).
- [Normalization](normalization.md): ``NormalizationEngine`` and normalization utilities.
- [Results](results.md): ``PipelineResult``, ``StageResult``, ``DocumentResult``, ``DeletionResult``.
- [Exceptions](exceptions.md): ``ScinrError`` hierarchy.
- [Deletion](deletion.md): ``delete_document()`` — permanent document removal with cascade and garbage collection.
- [Converters](converters.md): Document format converters.
- [Storage](storage.md): Storage backends.
- [Navigation](navigation.md): Read-only, engine-abstracted graph traversal — documents, structure nodes, model instances, entities.
- [Utilities](utilities.md): Theme registry, LLM factory, and utilities.

## User Guides

For tutorials and how-to guides, see the [User Guides](../user-guides/quick-start.md) section.
