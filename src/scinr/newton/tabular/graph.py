"""tabular/graph.py — LangGraph StateGraph for the tabular ingestion pipeline."""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from scinr.newton.tabular.nodes import (
    check_done_router,
    classify_theme,
    decide_model,
    load_sheets,
    map_columns,
    prepare_sheet,
    write_tabular,
)
from scinr.newton.tabular.state import TabularState


def build_tabular_graph():
    """Build and compile the tabular LangGraph StateGraph.

    Flow:
        load_sheets → [check_done_router] ──end──► END
                              │
                       prepare_sheet
                              │
                       classify_theme     ← LLM Call 0: ThemeClassification
                              │
                       decide_model       ← LLM Call 1: AnnotationDecision
                              │
                       map_columns        ← LLM Call 2: ColumnMapping
                              │
                       write_tabular      ← Neo4j write (Table + Rows)
                              │
                       [check_done_router] → loop or END
    """
    graph = StateGraph(TabularState)

    graph.add_node("load_sheets", load_sheets)
    graph.add_node("prepare_sheet", prepare_sheet)
    graph.add_node("classify_theme", classify_theme)
    graph.add_node("decide_model", decide_model)
    graph.add_node("map_columns", map_columns)
    graph.add_node("write_tabular", write_tabular)

    graph.set_entry_point("load_sheets")

    graph.add_conditional_edges(
        "load_sheets",
        check_done_router,
        {"prepare_sheet": "prepare_sheet", "end": END},
    )
    graph.add_edge("prepare_sheet", "classify_theme")
    graph.add_edge("classify_theme", "decide_model")
    graph.add_edge("decide_model", "map_columns")
    graph.add_edge("map_columns", "write_tabular")
    graph.add_conditional_edges(
        "write_tabular",
        check_done_router,
        {"prepare_sheet": "prepare_sheet", "end": END},
    )

    return graph.compile()


tabular_graph = build_tabular_graph()
