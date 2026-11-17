"""
tests/golden/_llm.py — Scripted LLM and deterministic ``uuid4`` of the golden harness.

``ScriptedLLM`` answers every ``with_structured_output(schema)`` call from a ``Script``
set by the scenario, so that no response depends on a model. It is a plain
``BaseChatModel`` (``configure(llm=...)`` accepts it) whose only supported operation is
``with_structured_output``; the answer is chosen from the schema and from a marker
found in the messages:

- ``AnnotationDecision`` of the annotation agent: the ``<id>`` of the node.
- ``AnnotationDecision`` of the tabular agent (messages carry ``File: <name>``): the name.
- ``ThemeClassification`` (tabular): always ``golden_theme``.
- ``ColumnMapping`` (tabular): every header maps to the model field with its name, except
  headers starting with ``x_`` (``__extra__``).
- ``BatchResponse_*`` (tabular normalization): one deterministic result per entry key.
- Any other schema (composite extraction schema, ``Triple``): the payload registered for
  the ``<id>`` of the node, validated against the schema.
"""

from __future__ import annotations

import hashlib
import re
import sys
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult

NO_MODEL: dict[str, Any] = {
    "matched_model_class": None,
    "confidence": "low",
    "rationale": "No catalog model fits this node.",
}


def decision(model: str, *, complementary: tuple[str, ...] = (), supplementary: bool = False):
    """Keyword arguments of an ``AnnotationDecision`` that matches *model*."""
    return {
        "matched_model_class": model,
        "confidence": "high",
        "rationale": f"The node matches {model}.",
        "complementary_models": [
            {"model_class": c, "coverage_note": f"{c} covers the supplier data."}
            for c in complementary
        ],
        "coverage_gaps": ["Warranty period"] if supplementary else [],
        "supplementary_fields": (
            [
                {
                    "field_name": "warranty_period",
                    "field_type": "str | None",
                    "description": "Warranty period.",
                    "required": False,
                }
            ]
            if supplementary
            else []
        ),
    }


@dataclass
class Script:
    """What the scripted LLM answers. ``annotations`` and ``extractions`` are keyed by the
    short ``node_id``; ``tabular`` by a substring of the file name."""

    annotations: dict[str, dict[str, Any]] = field(default_factory=dict)
    tabular: dict[str, dict[str, Any]] = field(default_factory=dict)
    extractions: dict[str, dict[str, Any]] = field(default_factory=dict)


_SCRIPT = Script()


def set_script(script: Script) -> None:
    global _SCRIPT
    _SCRIPT = script


def _text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    return "\n".join(str(getattr(m, "content", m)) for m in messages)


def _first(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1) if m else None


def _respond(schema: type, messages: Any) -> Any:
    text = _text(messages)
    name = schema.__name__
    if name == "ThemeClassification":
        return schema(justification="Synthetic data.", theme="golden_theme")
    if name == "AnnotationDecision":
        file_marker = _first(r"File: ([^\n]+)", text)
        if file_marker is not None:
            for marker, kwargs in _SCRIPT.tabular.items():
                if marker in file_marker:
                    return schema(**kwargs)
            raise AssertionError(f"no tabular answer scripted for file {file_marker!r}")
        node_id = _first(r"<id>(.*?)</id>", text)
        if node_id not in _SCRIPT.annotations:
            raise AssertionError(f"no annotation scripted for node {node_id!r}")
        return schema(**_SCRIPT.annotations[node_id])
    if name == "ColumnMapping":
        headers = re.findall(r"'([^']*)'", _first(r"Headers: ([^\n]*)", text) or "")
        return schema(
            mappings=[
                {
                    "column_name": h,
                    "model_field_name": "__extra__" if h.startswith("x_") else h,
                    "confidence": "high",
                }
                for h in headers
            ],
            unmapped_columns=[h for h in headers if h.startswith("x_")],
        )
    if name.startswith("BatchResponse_"):
        output_cls = schema.model_fields["results"].annotation.__args__[0]
        target = output_cls.model_fields["result"].annotation
        results = []
        for key in re.findall(r"--- Entry (\S+) ---", text):
            values = {
                fname: f"norm-{key.rsplit(':', 1)[-1][:6]}"
                for fname, info in target.model_fields.items()
                if info.annotation in (str, str | None)
            }
            results.append({"key": key, "result": values})
        return schema(results=results)
    node_id = _first(r"<id>(.*?)</id>", text)
    if node_id not in _SCRIPT.extractions:
        raise AssertionError(f"no extraction scripted for node {node_id!r} ({name})")
    return schema.model_validate(_SCRIPT.extractions[node_id])


class _Structured:
    def __init__(self, schema: type, include_raw: bool) -> None:
        self._schema = schema
        self._include_raw = include_raw

    async def ainvoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        return self.invoke(messages)

    def invoke(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        parsed = _respond(self._schema, messages)
        if self._include_raw:
            return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}
        return parsed


class ScriptedLLM(BaseChatModel):
    """Chat model whose structured outputs come from the current ``Script``."""

    @property
    def _llm_type(self) -> str:
        return "golden-scripted"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("ScriptedLLM only supports with_structured_output")

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kwargs: Any):
        return _Structured(schema, include_raw)


# ---------------------------------------------------------------------------
# Deterministic uuid4
# ---------------------------------------------------------------------------


class _Uuid4:
    """``uuid.uuid4`` replacement with a value that does not depend on scheduling.

    The only two callers (``entity_extraction/graph_mapper._write_model_fields``) give an unkeyed
    ``ModelInstance`` a random uid. Tabular rows are written 10 at a time
    (``_PARALLEL_ROW_WRITES``, not configurable), so a global sequence would hand the
    values to rows in a different order on every run. The value is therefore derived from
    the position of the instance in the tree, read from the caller's locals
    (``parent_uid``, ``field_name``, ``i``); a call from anywhere else falls back to a
    global sequence. Hashing is stable (it does not use ``hash()``).
    """

    def __init__(self) -> None:
        self._seen: dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self) -> uuid.UUID:
        caller = sys._getframe(1).f_locals
        if "parent_uid" in caller and "field_name" in caller:
            key = f"{caller['parent_uid']}|{caller['field_name']}|{caller.get('i')}"
        else:
            key = "global"
        with self._lock:
            n = self._seen.get(key, 0)
            self._seen[key] = n + 1
        return uuid.UUID(hashlib.md5(f"golden-uuid4:{key}:{n}".encode()).hexdigest())


@contextmanager
def deterministic_uuid4() -> Iterator[None]:
    """Patch ``uuid.uuid4`` globally (the code calls it through the module)."""
    with patch("uuid.uuid4", _Uuid4()):
        yield
