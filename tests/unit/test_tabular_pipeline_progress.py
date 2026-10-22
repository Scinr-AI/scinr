"""
tests/unit/test_tabular_pipeline_progress.py — run_tabular_pipeline() per-file
progress events and outcome classification, with the agent, Neo4j and storage
mocked out.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from scinr.newton.progress import ProgressEvent, ProgressTracker
from scinr.newton.stages import tabular as tabular_stage

# A real AppleDouble resource fork (``ditto -c -k --sequesterRsrc`` of a CSV with
# extended attributes): binary, but its xattr payload contains a newline, so read
# as text it would look like a one-row CSV.
_APPLEDOUBLE_CSV = bytes.fromhex(
    "00051607000200004d6163204f532058202020202020202000020000000900000032000000f4"
    "00000002000001260000000000000000000000000000000000000000000000000000000000000000"
    "000000000000415454520000000000000126000000e40000004200000000000000000000000000"
    "000003000000e400000022000025636f6d2e6170706c652e6d657461646174613a6b4d44497465"
    "6d576865726546726f6d7300000001060000000b000015636f6d2e6170706c652e70726f76656e"
    "616e6365000000011100000015000011636f6d2e6578616d706c652e7465737400687474703a2f"
    "2f6578616d706c652e636f6d2f736f6d652f6c6f6e672f76616c75650102001cf678c36965a5c0"
    "68656c6c6f2c776f726c640a6c696e65322c666f6f"
)


class _Emitter:
    """Collects emitted events and folds them into a ProgressTracker."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.tracker = ProgressTracker()

    def emit(self, kind, **fields):
        from datetime import UTC, datetime

        event = ProgressEvent(kind=kind, timestamp=datetime.now(UTC), **fields)
        self.events.append(event)
        self.tracker._apply(event)

    def for_doc(self, index: int) -> list[tuple[str, str | None, str | None]]:
        return [
            (e.kind, e.stage, e.status) for e in self.events if e.document_index == index
        ]


def _fake_driver():
    record = {"max_version": None}
    session = MagicMock()
    session.run.return_value.single.return_value = record
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver


@pytest.fixture
def env(monkeypatch):
    """Mocks every external dependency of run_tabular_pipeline()."""

    async def _noop(*a, **k):
        return None

    cfg = MagicMock()
    cfg.neo4j_database = "neo4j"
    monkeypatch.setattr(tabular_stage, "get_driver", _fake_driver)
    monkeypatch.setattr(tabular_stage, "get_async_driver", lambda: object())
    monkeypatch.setattr(tabular_stage, "setup_schema", lambda d: None)
    monkeypatch.setattr(tabular_stage, "get_config", lambda: cfg)
    monkeypatch.setattr("scinr.newton.annotation.neo4j_ops.ensure_catalog_models_once", _noop)
    monkeypatch.setattr("scinr.newton.annotation.neo4j_ops.ensure_theme_structure_once", _noop)
    monkeypatch.setattr("scinr.newton.utils.theme_registry.get_theme_registry", lambda: object())

    raw_repo = MagicMock()
    raw_repo.store_file = AsyncMock(return_value="raw-1")
    monkeypatch.setattr("scinr.newton.storage.factory.get_storage", lambda: (raw_repo, None))

    agent = AsyncMock(
        return_value={"errors": [], "ingested_table_node_ids": ["t-1"]}
    )
    monkeypatch.setattr("scinr.newton.tabular.agent.run_tabular_agent", agent)
    return {"agent": agent, "raw_repo": raw_repo}


def _csv(path: Path, text: str = "a,b\n1,2\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


async def test_success_emits_the_per_file_sequence(tmp_path, env):
    f = _csv(tmp_path / "sub" / "data.csv")
    emitter = _Emitter()

    sr = await tabular_stage.run_tabular_pipeline(
        str(tmp_path), files=[(f, 3)], emitter=emitter, tenant_id="acme"
    )

    assert emitter.for_doc(3) == [
        ("document_started", None, None),
        ("stage_started", "tabular", None),
        ("stage_completed", "tabular", None),
        ("document_finished", None, "success"),
    ]
    started = emitter.events[0]
    assert started.planned_stages == ("tabular",)
    assert started.doc_path == "sub/data"
    assert started.document_name == "data"
    assert started.source_path == str(f)
    assert started.tenant_id == "acme"
    completed = emitter.events[2]
    assert dict(completed.metrics) == {"nodes_processed": 1, "nodes_failed": 0}
    assert sr.success is True
    assert (sr.documents[0].nodes_processed, sr.documents[0].nodes_failed) == (1, 0)

    snap = emitter.tracker.snapshot()
    assert snap.documents[3].status == "success"
    assert snap.completed_by_stage == {"tabular": 1}


async def test_errors_without_tables_is_failed(tmp_path, env):
    f = _csv(tmp_path / "data.csv")
    env["agent"].return_value = {
        "errors": ["decide_model failed for sheet 'data'"],
        "ingested_table_node_ids": [],
    }
    emitter = _Emitter()

    sr = await tabular_stage.run_tabular_pipeline(str(tmp_path), emitter=emitter)

    assert emitter.for_doc(0)[-2:] == [
        ("stage_failed", "tabular", None),
        ("document_finished", None, "failed"),
    ]
    finished = emitter.events[-1]
    assert finished.stopped_at == "tabular"
    assert "decide_model failed" in finished.error
    assert sr.success is False
    assert (sr.documents[0].nodes_processed, sr.documents[0].nodes_failed) == (0, 1)
    assert sr.documents[0].errors == ["decide_model failed for sheet 'data'"]
    assert emitter.tracker.snapshot().failed_by_stage == {"tabular": 1}
    assert f.exists()


async def test_errors_with_tables_is_partial(tmp_path, env):
    _csv(tmp_path / "data.csv")
    env["agent"].return_value = {
        "errors": ["write_tabular failed for sheet 1: boom"],
        "ingested_table_node_ids": ["t-0"],
    }
    emitter = _Emitter()

    sr = await tabular_stage.run_tabular_pipeline(str(tmp_path), emitter=emitter)

    assert emitter.for_doc(0)[-2:] == [
        ("stage_completed", "tabular", None),
        ("document_finished", None, "partial"),
    ]
    assert dict(emitter.events[-2].metrics) == {"nodes_processed": 1, "nodes_failed": 1}
    assert "boom" in emitter.events[-1].error
    assert sr.success is False
    assert (sr.documents[0].nodes_processed, sr.documents[0].nodes_failed) == (1, 1)


async def test_agent_exception_is_failed(tmp_path, env):
    _csv(tmp_path / "data.csv")
    env["agent"].side_effect = RuntimeError("neo4j down")
    emitter = _Emitter()

    sr = await tabular_stage.run_tabular_pipeline(str(tmp_path), emitter=emitter)

    assert emitter.for_doc(0)[-2:] == [
        ("stage_failed", "tabular", None),
        ("document_finished", None, "failed"),
    ]
    assert emitter.events[-1].error == "neo4j down"
    assert sr.success is False
    assert sr.documents[0].errors == ["neo4j down"]


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        ("__MACOSX/._data.csv", _APPLEDOUBLE_CSV, "binary content"),
        ("empty.csv", b"", "no data rows"),
        ("header_only.csv", b"a,b\n", "no data rows"),
        ("junk.xlsx", b"not a zip file", "unreadable tabular file"),
    ],
)
async def test_unreadable_or_empty_file_is_skipped_before_anything_is_stored(
    tmp_path, env, name, content, reason
):
    good = _csv(tmp_path / "good.csv")
    bad = tmp_path / name
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(content)
    emitter = _Emitter()

    sr = await tabular_stage.run_tabular_pipeline(
        str(tmp_path), files=[(bad, 0), (good, 1)], emitter=emitter
    )

    assert emitter.for_doc(0) == [
        ("document_started", None, None),
        ("stage_started", "tabular", None),
        ("document_finished", None, "skipped"),
    ]
    assert reason in emitter.events[2].error
    # Neither the raw file nor the :Document (created by the agent) for the skipped file.
    assert [c.kwargs["path"] for c in env["raw_repo"].store_file.await_args_list] == [good]
    assert [c.kwargs["file_path"] for c in env["agent"].await_args_list] == [good]
    assert sr.success is True
    skipped = next(d for d in sr.documents if d.nodes_processed == 0)
    assert (skipped.nodes_processed, skipped.nodes_failed) == (0, 0)

    snap = emitter.tracker.snapshot()
    assert snap.documents_skipped == 1
    assert snap.documents_succeeded == 1
    assert snap.failed_by_stage == {}


async def test_delimiter_and_scans_reach_the_agent(tmp_path, env):
    f = _csv(tmp_path / "semi.csv", "a;b\n1;2\n")

    sr = await tabular_stage.run_tabular_pipeline(str(tmp_path), tabular_delimiter=";")

    assert sr.success is True
    kwargs = env["agent"].await_args.kwargs
    assert kwargs["delimiter"] == ";"
    assert kwargs["file_path"] == f
    (scan,) = kwargs["scans"]
    assert scan["headers"] == ["a", "b"]
    assert scan["total_rows"] == 1


async def test_default_discovery_numbers_files_in_sorted_order(tmp_path, env):
    _csv(tmp_path / "b.csv")
    _csv(tmp_path / "a.csv")
    emitter = _Emitter()

    await tabular_stage.run_tabular_pipeline(str(tmp_path), emitter=emitter)

    started = [e for e in emitter.events if e.kind == "document_started"]
    assert sorted((e.document_index, e.document_name) for e in started) == [(0, "a"), (1, "b")]
