"""
tests/unit/test_converters_main.py — Unit tests for scinr.newton.converters.main.

Currently focused on the `parallel_docs` guard in `convert_folder()`: passing
`parallel_docs=0` (or negative) must raise `ValueError` immediately, rather
than creating an unacquirable `asyncio.Semaphore(0)` and hanging forever.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from scinr.newton.converters.main import convert_folder

# Short guard timeout: if the `parallel_docs < 1` validation is ever removed
# or bypassed, convert_folder() would hang indefinitely on an unacquirable
# asyncio.Semaphore(0)/negative-sized semaphore. Wrapping the call in
# asyncio.wait_for() ensures this test fails fast with a clear TimeoutError
# instead of hanging the whole test suite.
_HANG_GUARD_TIMEOUT_SECONDS = 1.0


class TestConvertFolderParallelDocsGuard:
    @pytest.mark.parametrize("bad_value", [0, -1, -5])
    async def test_parallel_docs_below_one_raises_value_error(self, tmp_path: Path, bad_value: int):
        """`parallel_docs < 1` raises ValueError immediately (no hang).

        A source file is created so `convert_folder` actually schedules an
        entry and awaits the semaphore — with the guard removed,
        `parallel_docs=0` would hang forever on `asyncio.Semaphore(0)`
        (an empty *tmp_path* would never exercise the semaphore at all,
        masking the bug), and negative values raise asyncio's own generic
        ``ValueError`` instead of the clear, immediate one this guard adds.
        """
        (tmp_path / "doc.txt").write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError, match="parallel_docs must be >= 1"):
            await asyncio.wait_for(
                convert_folder(tmp_path, tmp_path, parallel_docs=bad_value),
                timeout=_HANG_GUARD_TIMEOUT_SECONDS,
            )

    async def test_parallel_docs_one_does_not_raise(self, tmp_path: Path):
        """Sanity check: the default/valid `parallel_docs=1` still works."""
        result = await asyncio.wait_for(
            convert_folder(tmp_path, tmp_path, parallel_docs=1),
            timeout=_HANG_GUARD_TIMEOUT_SECONDS,
        )
        written, failures = result
        assert written == []
        assert failures == []


class TestConvertFolderUnsupportedFormat:
    async def test_unsupported_extension_is_skipped_silently(self, tmp_path: Path):
        """An unsupported-extension file must not appear in `failures`.

        `convert_one` catches `UnsupportedFormatError` internally and returns
        `([], [])` for that entry (a deliberate "skip silently" design — see
        its docstring), rather than surfacing it as a failure. Mixed with a
        valid file in the same directory, the valid file must still be
        converted (present in `written`) while the unsupported one is simply
        absent from both `written` and `failures`.
        """
        (tmp_path / "doc.txt").write_text("hello world", encoding="utf-8")
        (tmp_path / "unsupported.xyz123").write_text("ignored content", encoding="utf-8")

        written, failures = await asyncio.wait_for(
            convert_folder(tmp_path, tmp_path, parallel_docs=1),
            timeout=_HANG_GUARD_TIMEOUT_SECONDS,
        )

        assert len(written) == 1
        raw_source, _json_written, _doc = written[0]
        assert raw_source.name == "doc.txt"
        assert failures == []
