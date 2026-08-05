"""
Motor de normalización post-extracción.

Agrupa instancias por tipo de normalización, batchea llamadas al LLM
con un solo ainvoke() por batch, y aplica los resultados estructurados
a los modelos Pydantic mediante mapeo explícito por clave única.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, TypeAdapter

from scinr.newton.tabular.normalization.detector import (
    extract_source_values,
    get_normalization_specs,
)
from scinr.newton.tabular.normalization.models import NormalizationEntry
from scinr.newton.utils.llm_retry import with_llm_retry

logger = logging.getLogger(__name__)

_NORMALIZATION_SYSTEM_PROMPT = (
    "You are a data normalization assistant. You receive raw extracted data "
    "and must normalize it into a structured format. Fill in all fields you can "
    "confidently identify from the source data. Leave uncertain fields as null."
)

_DEFAULT_CONCURRENCY = 5


class NormalizationEngine:
    """
    Motor de normalización post-extracción.

    Agrupa instancias por tipo de modelo normalizado, batchea llamadas
    al LLM con structured output (una llamada por batch), y aplica los
    resultados mediante mapeo explícito por clave única.
    """

    def __init__(
        self,
        llm: BaseLanguageModel,
        batch_size: int = 5,
        concurrency: int = _DEFAULT_CONCURRENCY,
    ) -> None:
        """
        Args:
            llm: Modelo de LangChain para las llamadas de normalización.
            batch_size: Número máximo de entradas por batch LLM.
            concurrency: Máximo de llamadas LLM en paralelo.
        """
        self.llm = llm
        self.batch_size = batch_size
        # NOTE: kept for API compatibility, but no longer used to create a
        # local semaphore in normalize_instances(). Real LLM concurrency is
        # now governed globally by config.get_llm_semaphore(), shared across
        # all pipeline stages (extraction, entity_extraction, annotation,
        # normalization) to avoid exceeding the Bedrock botocore connection
        # pool.
        self.concurrency = concurrency
        self.result_cache: dict[str, BaseModel] = {}

    async def normalize_instances(
        self,
        instances: list[tuple[type[BaseModel], BaseModel]],
    ) -> list[tuple[type[BaseModel], BaseModel]]:
        """
        Normaliza los campos marcados en una lista de instancias.

        Args:
            instances: Lista de (model_class, instance).

        Returns:
            Lista de instancias con campos normalizados rellenados.
        """
        if not instances:
            return instances

        # Phase 1: Collect entries with unique keys
        entries: list[NormalizationEntry] = []
        for model_cls, instance in instances:
            specs = get_normalization_specs(model_cls)
            for spec in specs:
                source_values = extract_source_values(spec, instance)
                if not source_values:
                    continue
                unique_key = (
                    f"{spec.target_type.__name__}:"
                    f"{self._hash_source_values(source_values)}"
                )
                entry = NormalizationEntry(
                    instance_id=id(instance),
                    model_class_name=model_cls.__name__,
                    field_name=spec.field_name,
                    target_type=spec.target_type,
                    source_values=source_values,
                    unique_key=unique_key,
                )
                entries.append(entry)

        if not entries:
            return instances
        
        # Phase 2: Build key → targets map
        key_to_targets: dict[str, list[tuple[int, str]]] = {}
        for entry in entries:
            key = entry.unique_key
            if key not in key_to_targets:
                key_to_targets[key] = []
            key_to_targets[key].append((entry.instance_id, entry.field_name))

        # Phase 3: Group unique entries by target_type
        unique_entries: dict[str, list[NormalizationEntry]] = {}
        seen_keys: set[str] = set()
        for entry in entries:
            if entry.unique_key in seen_keys:
                continue
            seen_keys.add(entry.unique_key)
            type_key = entry.target_type.__name__
            if type_key not in unique_entries:
                unique_entries[type_key] = []
            unique_entries[type_key].append(entry)

        # Phase 4: Process each target_type in batches.
        # NOTE: concurrency towards the LLM is now governed globally by
        # get_llm_semaphore() (config.py), not by self.concurrency, so that
        # all Bedrock calls across the whole pipeline share one bounded pool.
        # Lazy import to avoid a circular import with config.py.
        from scinr.newton.config import get_llm_semaphore

        async def _process_type_batch(
            batch_entries: list[NormalizationEntry],
        ) -> None:
            async with get_llm_semaphore():
                await self._call_llm_batch(
                    batch_entries, instances, key_to_targets,
                )

        tasks: list[asyncio.Task[None]] = []
        for type_entries in unique_entries.values():
            for i in range(0, len(type_entries), self.batch_size):
                chunk = type_entries[i : i + self.batch_size]
                tasks.append(asyncio.create_task(_process_type_batch(chunk)))

        await asyncio.gather(*tasks)
        return instances

    async def process_key_batch(
        self,
        entries: list[NormalizationEntry],
        retry_count: int = 0,
    ) -> dict[str, BaseModel]:
        """Process a batch of unique normalization keys via LLM.

        Returns {unique_key: normalized_result} for successfully processed keys.
        Results are cached in self.result_cache for reuse across batches.
        """
        if not entries:
            return {}

        target_type = entries[0].target_type

        # Validate all entries share the same target_type
        if len(entries) > 1:
            assert all(
                e.target_type is target_type for e in entries
            ), (
                f"process_key_batch received heterogeneous target_types: "
                f"{ {e.target_type.__name__ for e in entries} }"
            )

        batch_results: dict[str, BaseModel] = {}

        # Build dynamic output schemas (same as _call_llm_batch)
        BatchOutput = type(
            f"BatchOutput_{target_type.__name__}",
            (BaseModel,),
            {
                "__annotations__": {"key": str, "result": target_type},
                "model_config": ConfigDict(extra="forbid"),
            },
        )
        BatchOutput.__doc__ = (
            f"Normalization result. 'key' must match the input key exactly. "
            f"'result' is the normalized {target_type.__name__}."
        )

        BatchResponse = type(
            f"BatchResponse_{target_type.__name__}",
            (BaseModel,),
            {
                "__annotations__": {"results": list[BatchOutput]},
                "model_config": ConfigDict(extra="forbid"),
            },
        )
        BatchResponse.__doc__ = (
            f"Container for normalization batch results. "
            f"'results' is a list of {target_type.__name__} normalizations."
        )

        structured_llm = self.llm.with_structured_output(BatchResponse)
        messages = self._build_batch_messages(entries)

        try:
            result = await with_llm_retry(lambda: structured_llm.ainvoke(messages))

            if not isinstance(result, BatchResponse):
                try:
                    adapter = TypeAdapter(BatchResponse)
                    result = adapter.validate_python(result)
                except Exception:
                    logger.warning(
                        "Normalization result coercion failed for %s: expected %s, got %s",
                        target_type.__name__, BatchResponse.__name__, type(result).__name__,
                    )
                    return batch_results

            processed_keys: set[str] = set()
            for item in result.results:
                key = item.key
                processed_keys.add(key)
                batch_results[key] = item.result
                self.result_cache[key] = item.result

            # Retry missing keys once
            all_keys = {e.unique_key for e in entries}
            missing_keys = all_keys - processed_keys
            if missing_keys and retry_count < 1:
                retry_entries = [e for e in entries if e.unique_key in missing_keys]
                logger.warning(
                    "Normalization: %d/%d results missing, retrying: %s",
                    len(missing_keys), len(entries), missing_keys,
                )
                retry_results = await self.process_key_batch(
                    retry_entries, retry_count=retry_count + 1
                )
                batch_results.update(retry_results)

        except Exception as e:
            logger.warning(
                "Normalization batch failed for %s (%d entries): %s",
                target_type.__name__, len(entries), e,
            )

        return batch_results

    def apply_cached_to_instance(
        self,
        instance: BaseModel,
        field_name: str,
        unique_key: str,
    ) -> bool:
        """Apply cached normalization result to a specific instance field.

        Returns True if the key was found in cache and applied, False otherwise.
        """
        if unique_key not in self.result_cache:
            return False

        normalized = self.result_cache[unique_key]
        try:
            setattr(instance, field_name, normalized)
        except Exception:
            object.__setattr__(instance, field_name, normalized)
        return True

    async def _call_llm_batch(
        self,
        entries: list[NormalizationEntry],
        all_instances: list[tuple[type[BaseModel], BaseModel]],
        key_to_targets: dict[str, list[tuple[int, str]]],
        retry: bool = False,
    ) -> None:
        """
        Llama al LLM con un batch de entradas y aplica resultados.
        Una sola llamada LLM por batch.
        """
        target_type = entries[0].target_type

        # Build dynamic output schemas
        BatchOutput = type(
            f"BatchOutput_{target_type.__name__}",
            (BaseModel,),
            {
                "__annotations__": {
                    "key": str,
                    "result": target_type,
                },
                "model_config": ConfigDict(extra="forbid"),
            },
        )
        BatchOutput.__doc__ = (
            f"Normalization result. 'key' must match the input key exactly. "
            f"'result' is the normalized {target_type.__name__}."
        )

        # Wrap in a container model — with_structured_output needs a class, not list[X]
        BatchResponse = type(
            f"BatchResponse_{target_type.__name__}",
            (BaseModel,),
            {
                "__annotations__": {
                    "results": list[BatchOutput],  # type: ignore[misc]
                },
                "model_config": ConfigDict(extra="forbid"),
            },
        )
        BatchResponse.__doc__ = (
            f"Container for normalization batch results. "
            f"'results' is a list of {target_type.__name__} normalizations."
        )

        structured_llm = self.llm.with_structured_output(BatchResponse)

        # Build prompt
        messages = self._build_batch_messages(entries)

        try:
            result = await with_llm_retry(lambda: structured_llm.ainvoke(messages))

            # Coerce result to the expected container type.
            # with_structured_output() may return dict with some providers.
            if not isinstance(result, BatchResponse):
                try:
                    adapter = TypeAdapter(BatchResponse)
                    result = adapter.validate_python(result)
                except Exception:
                    logger.warning(
                        "Normalization result coercion failed for %s: expected %s, got %s",
                        target_type.__name__, BatchResponse.__name__, type(result).__name__,
                    )
                    return

            # Apply results
            processed_keys: set[str] = set()
            for item in result.results:  # unwrap from container
                key = item.key
                processed_keys.add(key)
                targets = key_to_targets.get(key, [])
                for instance_id, field_name in targets:
                    self._apply_to_instance(
                        instance_id, field_name, item.result, all_instances,
                    )

            # Check for missing results — retry ONCE if needed
            all_keys = {e.unique_key for e in entries}
            missing_keys = all_keys - processed_keys
            if missing_keys and not retry:
                retry_entries = [
                    e for e in entries if e.unique_key in missing_keys
                ]
                logger.warning(
                    "Normalization: %d/%d results missing, retrying: %s",
                    len(missing_keys), len(entries), missing_keys,
                )
                await self._call_llm_batch(
                    retry_entries, all_instances, key_to_targets, retry=True,
                )

        except Exception as e:
            logger.warning(
                "Normalization batch failed for %s (%d entries): %s",
                target_type.__name__, len(entries), e,
            )

    def _build_batch_messages(
        self,
        entries: list[NormalizationEntry],
    ) -> list[SystemMessage | HumanMessage]:
        """Construye el prompt para un batch de entradas."""
        entries_text = ""
        for entry in entries:
            source_text = "\n".join(
                f"  {key}: {value}" for key, value in entry.source_values.items()
            )
            entries_text += (
                f"--- Entry {entry.unique_key} ---\n"
                f"Source data:\n{source_text}\n\n"
            )

        return [
            SystemMessage(content=_NORMALIZATION_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    "Normalize the following extracted data entries into "
                    "structured format.\n\n"
                    "For each entry, return a result with:\n"
                    "- key: the exact unique key from the entry (must match "
                    "exactly)\n"
                    "- result: the normalized structured output\n\n"
                    f"Entries:\n{entries_text}\n"
                    "Return a list of results, one per entry."
                )
            ),
        ]

    def _apply_to_instance(
        self,
        instance_id: int,
        field_name: str,
        normalized: BaseModel,
        all_instances: list[tuple[type[BaseModel], BaseModel]],
    ) -> None:
        """Aplica el resultado normalizado a la instancia correspondiente.

        Usa setattr con validate_assignment como primer intento.
        Si falla (validación estricta, mismatch de clases), recurre a
        object.__setattr__ para bypass de validación.
        """
        for _, instance in all_instances:
            if id(instance) == instance_id:
                try:
                    setattr(instance, field_name, normalized)
                except Exception:
                    # Fallback: bypass Pydantic validation.
                    # This handles validate_assignment=True rejecting the value
                    # or class identity mismatches. The field IS declared in the
                    # model schema, so we just need to skip re-validation.
                    object.__setattr__(instance, field_name, normalized)
                return

        # Instance not found — this shouldn't happen, but log it
        logger.warning(
            "Normalization: instance id %d not found for %s.%s, skipping",
            instance_id, field_name, normalized.__class__.__name__,
        )

    @staticmethod
    def _hash_source_values(source_values: dict[str, Any]) -> str:
        """Genera hash determinista de los valores fuente."""
        normalized = str(sorted(source_values.items())).lower()
        return hashlib.md5(normalized.encode()).hexdigest()
