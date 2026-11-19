"""tests/unit/test_sql_ids.py -- ``sql/ids.py``: ObjectId-shaped, creation-ordered ids."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from scinr.newton.sql.ids import ID_RE, is_valid_id, new_id


def test_shape_is_24_lowercase_hex_chars() -> None:
    value = new_id()
    assert re.fullmatch(r"[0-9a-f]{24}", value)
    assert ID_RE.fullmatch(value)
    assert is_valid_id(value)


def test_ids_of_one_thread_are_unique_and_increasing() -> None:
    ids = [new_id() for _ in range(10_000)]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_ids_of_four_threads_are_unique() -> None:
    def make(_: int) -> list[str]:
        return [new_id() for _ in range(5_000)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        chunks = list(pool.map(make, range(4)))
    everything = [i for chunk in chunks for i in chunk]
    assert len(set(everything)) == len(everything)


def test_is_valid_id_rejects_other_shapes() -> None:
    for bad in ("", "abc", "G" * 24, "A" * 24, "a" * 23, "a" * 25, "../etc/passwd", None, 5):
        assert not is_valid_id(bad)
