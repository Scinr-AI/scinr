"""
Regression guard: no ``MATCH`` / ``MERGE`` node pattern without a label.

A node pattern such as ``(parent {uid: $uid})`` has no label, so Neo4j cannot
use any index or constraint and scans every node of every tenant. This test
reads the library sources and fails on any ``(<var> {<prop>: ...})`` pattern.
A legitimate exception must be added to ``_ALLOWED`` with a reason.
"""

from __future__ import annotations

import re
from pathlib import Path

import scinr.newton

_SRC = Path(scinr.newton.__file__).parent

# Unlabelled node pattern with a property map; ``{{`` covers f-strings.
_UNLABELLED = re.compile(r"\((\w+) \{\{?\w+:")

# Words that precede a map literal rather than name a node variable.
_NOT_A_NODE = {"DISTINCT"}

# (relative path, stripped line) pairs that are known to be fine.
_ALLOWED: set[tuple[str, str]] = set()


def test_no_unlabelled_node_pattern_with_property_map() -> None:
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for m in _UNLABELLED.finditer(line):
                if m.group(1) in _NOT_A_NODE:
                    continue
                if (rel, line.strip()) in _ALLOWED:
                    continue
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "Node patterns without a label cannot use any index (full graph scan). "
        "Add the label, e.g. (n:ModelInstance {uid: $uid}):\n" + "\n".join(offenders)
    )
