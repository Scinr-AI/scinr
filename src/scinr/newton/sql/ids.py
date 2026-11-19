"""
sql/ids.py — Identifiers of the SQL storage rows, laid out like MongoDB ObjectIds.

``raw_file_id`` and ``frozen_blob_id`` end up in the graph, and the normalisation of the
golden dumps numbers ObjectIds by creation order. An id with the same shape (24 hex chars,
creation-ordered) keeps the dumps of Neo4j + MongoDB and of the SQL backends comparable.
"""

from __future__ import annotations

import os
import re
import secrets
import threading
import time

__all__ = ["ID_RE", "is_valid_id", "new_id"]

ID_RE = re.compile(r"^[0-9a-f]{24}$")
"""Shape of every id :func:`new_id` returns (also the only shape used to build file paths)."""

_COUNTER_MASK = 0xFFFFFF

_lock = threading.Lock()
_pid = os.getpid()
_process_random = os.urandom(5)
_counter = secrets.randbits(24)


def is_valid_id(value: object) -> bool:
    """Whether *value* has the shape of an id produced by :func:`new_id`."""
    return isinstance(value, str) and ID_RE.fullmatch(value) is not None


def new_id() -> str:
    """A new id: 4-byte time in seconds, 5 random bytes per process, 3-byte counter.

    Unique across threads and increasing within a process (until the counter wraps after
    16 million ids in the same second).
    """
    global _pid, _process_random, _counter
    with _lock:
        pid = os.getpid()
        if pid != _pid:  # forked: do not share the random part with the parent
            _pid = pid
            _process_random = os.urandom(5)
        _counter = (_counter + 1) & _COUNTER_MASK
        counter = _counter
        process_random = _process_random
    return (int(time.time()).to_bytes(4, "big") + process_random + counter.to_bytes(3, "big")).hex()
