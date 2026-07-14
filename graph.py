"""
graph.py — LangGraph StateGraph definition for the Personal Finance Assistant.

Two distinct flows share the same FinanceState:

  1. ETL pipeline (CSV upload):
     ingest → categorize → store → anomaly → report → END

  2. Q&A path (standalone):
     Build the ReAct agent once via build_qa_agent() and call qa_node()
     directly — it is intentionally outside the ETL graph so the
     stateful agent isn't reconstructed on every question.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from nodes.anomaly import anomaly_node
from nodes.categorize import categorize_node
from nodes.ingest import ingest_node
from nodes.report import report_node
from nodes.store import store_node


# ── Shared state ───────────────────────────────────────────────────────────────
class FinanceState(TypedDict, total=False):
    # ETL pipeline fields
    csv_path: str
    raw_transactions: list
    categorized_transactions: list
    ingestion_summary: dict
    anomalies: list
    report: dict

    # Q&A branch fields (used outside the ETL graph)
    user_question: str
    answer: str

    # Metadata
    month: str
    user_id: str


# ── Graph builder ──────────────────────────────────────────────────────────────
def build_graph(conn: sqlite3.Connection, user_id: str = "default"):
    """
    Compile and return the LangGraph StateGraph for the ETL pipeline.
    The Q&A agent is built separately via nodes/qa_agent.py::build_qa_agent().
    """

    def _ingest(state: FinanceState) -> dict[str, Any]:
        return ingest_node(state)

    def _categorize(state: FinanceState) -> dict[str, Any]:
        return categorize_node(state, conn)

    def _store(state: FinanceState) -> dict[str, Any]:
        return store_node(state, conn, user_id=user_id)

    def _anomaly(state: FinanceState) -> dict[str, Any]:
        return anomaly_node(state, conn, user_id=user_id)

    def _report(state: FinanceState) -> dict[str, Any]:
        return report_node(state, conn, user_id=user_id)

    graph = StateGraph(FinanceState)

    graph.add_node("ingest",     _ingest)
    graph.add_node("categorize", _categorize)
    graph.add_node("store",      _store)
    graph.add_node("anomaly",    _anomaly)
    graph.add_node("report",     _report)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest",     "categorize")
    graph.add_edge("categorize", "store")
    graph.add_edge("store",      "anomaly")
    graph.add_edge("anomaly",    "report")
    graph.add_edge("report",     END)

    return graph.compile()
