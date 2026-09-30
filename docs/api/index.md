# API Reference Overview

Welcome to the `scinr.newton` API reference. All API documentation is auto-generated from Python docstrings using mkdocstrings.

## Core Modules

- [Pipeline](pipeline.md): ``run_pipeline()`` orchestrator.
- [Configuration](config.md): ``configure()``, ``get_config()``, ``ScinrConfig``.
- [Stages](stages.md): Individual stage runner functions (Stages 0-5).
- [Normalization](normalization.md): ``NormalizationEngine`` and normalization utilities.
- [Results](results.md): ``PipelineResult``, ``StageResult``, ``DocumentResult``, ``DeletionResult``, ``OrphanCollectionResult``, ``FreezeResult``, ``RestoreResult``.
- [Exceptions](exceptions.md): ``ScinrError`` hierarchy.
- [Deletion](deletion.md): ``delete_document()`` — permanent document removal with cascade and garbage collection, always scoped to one tenant (``tenant_id`` is mandatory; ``None`` / ``"__public__"`` = public documents); ``collect_orphans()`` — maintenance sweep of every orphaned extraction node of a tenant.
- [Freezing](freezing.md): ``freeze_document()``, ``restore_document()``, ``export_document_snapshot()`` — archive a document's subtree to a snapshot, reduce it to a stub and rebuild it later; same tenant-scoped selector as ``delete_document()``.
- [Converters](converters.md): Document format converters.
- [Storage](storage.md): Storage backends.
- [Navigation](navigation.md): Read-only, engine-abstracted graph traversal — documents, structure nodes, model instances, entities.
- [Utilities](utilities.md): Theme registry, LLM factory, and utilities.

## User Guides

For tutorials and how-to guides, see the [User Guides](../user-guides/quick-start.md) section.
