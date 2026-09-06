"""Unit tests for the GraphNavigator ABC contract."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from _navigation_fakes import FakeGraphNavigator  # noqa: E402

from scinr.newton.exceptions import UnsupportedOperationError  # noqa: E402
from scinr.newton.navigation.base import GraphNavigator  # noqa: E402


def test_abc_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        GraphNavigator()  # type: ignore[abstract]


def test_fake_satisfies_full_interface() -> None:
    nav = FakeGraphNavigator()
    assert isinstance(nav, GraphNavigator)


async def test_context_manager_calls_connect_close() -> None:
    nav = FakeGraphNavigator()
    async with nav as n:
        assert n is nav
        assert nav.connected is True
    assert nav.closed is True


async def test_execute_raw_base_raises_unsupported() -> None:
    nav = FakeGraphNavigator()
    with pytest.raises(UnsupportedOperationError):
        await nav.execute_raw("RETURN 1")
    with pytest.raises(UnsupportedOperationError):
        await nav.execute_raw_one("RETURN 1")


def test_default_dialect_on_abc() -> None:
    assert GraphNavigator.dialect == "none"


async def test_list_methods_return_lists() -> None:
    nav = FakeGraphNavigator()
    assert await nav.list_root_documents() == []
    assert await nav.get_structure_nodes("a/b") == []
    assert await nav.count_root_documents() == 0


def test_renamed_methods_present_and_old_gone() -> None:
    assert hasattr(GraphNavigator, "get_child_documents")
    assert hasattr(GraphNavigator, "get_node_model_instances")
    assert hasattr(GraphNavigator, "get_incoming_model_instances")
    assert hasattr(GraphNavigator, "get_outgoing_model_instances")
    assert hasattr(GraphNavigator, "get_document_model_profile")
    assert hasattr(GraphNavigator, "get_catalog_graph")
    # removed
    assert not hasattr(GraphNavigator, "get_document_children")
    assert not hasattr(GraphNavigator, "iter_document_descendants")
    assert not hasattr(GraphNavigator, "get_document_info_units")
    assert not hasattr(GraphNavigator, "get_document_triples")
    assert not hasattr(GraphNavigator, "get_node_instances")
