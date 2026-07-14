"""
nodes/report.py — Generate the monthly MonthlyReport using the LLM.

The LLM receives pre-computed totals (from pandas/SQL) and only needs
to write human-readable suggestions and rephrase anomalies — it never
does arithmetic itself.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from models import MonthlyReport

load_dotenv()

logger = logging.getLogger("finance-assistant.report")

_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
_structured_llm = _llm.with_structured_output(MonthlyReport)

REPORT_PROMPT = """\
You are a personal finance advisor generating a monthly summary report.

Month: {month}
Spending by category (expenses only, USD):
{totals}

Total income this month: ${income:.2f}
Total spent (expenses, excluding extreme outliers): ${total_spent:.2f}

Anomalies detected (pre-computed stats):
{anomalies}

Your task:
- Set total_spent, total_income, savings_rate (= (income - spent) / income * 100). NOTE: Allow negative values for savings_rate if total_spent > total_income!
- List the top 3 spending categories (by amount).
- Ensure you copy the anomaly messages exactly as provided, do not rephrase or alter the numbers.
- Write 2–3 actionable suggestions tied to the ACTUAL numbers above — no generic advice.
"""


def _get_totals(conn, user_id: str, month: str, anomalies: list[dict] = None) -> dict[str, float]:
    """Sum spending per category for the given YYYY-MM month, excluding extreme outliers."""
    df = pd.read_sql(
        """
        SELECT category, SUM(ABS(amount)) as total
        FROM transactions
        WHERE user_id = ?
          AND strftime('%Y-%m', date) = ?
          AND category != 'Income'
          AND is_outlier = 0
        GROUP BY category
        ORDER BY total DESC
        """,
        conn,
        params=(user_id, month),
    )
    return dict(zip(df["category"], df["total"].round(2)))


def _get_income(conn, user_id: str, month: str) -> float:
    """Sum income-category rows for the given YYYY-MM month."""
    row = conn.execute(
        """
        SELECT COALESCE(SUM(ABS(amount)), 0)
        FROM transactions
        WHERE user_id = ?
          AND strftime('%Y-%m', date) = ?
          AND category = 'Income'
        """,
        (user_id, month),
    ).fetchone()
    return float(row[0])


def _current_month(conn, user_id: str) -> str:
    """Return the most recent YYYY-MM present in the DB for this user."""
    row = conn.execute(
        "SELECT strftime('%Y-%m', MAX(date)) FROM transactions WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return row[0] or "unknown"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _call_llm(prompt: str) -> MonthlyReport:
    return _structured_llm.invoke(prompt)


def report_node(
    state: dict[str, Any], conn, user_id: str = "default"
) -> dict[str, Any]:
    """
    LangGraph node: produce a structured MonthlyReport for the most recent month.
    Returns {'report': {...}}.
    """
    month = _current_month(conn, user_id)
    anomalies = state.get("anomalies", [])
    totals = _get_totals(conn, user_id, month, anomalies)
    income = _get_income(conn, user_id, month)
    total_spent = sum(totals.values())

    totals_str = "\n".join(f"  {k}: ${v:.2f}" for k, v in totals.items()) or "  (no data)"
    anomalies_str = (
        "\n".join(f"  - {a['message']}" for a in anomalies) if anomalies else "  None"
    )

    prompt = REPORT_PROMPT.format(
        month=month,
        totals=totals_str,
        income=income,
        total_spent=total_spent,
        anomalies=anomalies_str,
    )

    try:
        report: MonthlyReport = _call_llm(prompt)
    except Exception as exc:
        logger.error("report_node: LLM call failed (%s)", exc)
        # Graceful fallback — build the report from raw numbers
        report = MonthlyReport(
            month=month,
            total_spent=round(total_spent, 2),
            total_income=round(income, 2),
            savings_rate=round((income - total_spent) / income * 100, 1) if income else 0.0,
            top_categories=list(totals.keys())[:3],
            anomalies=[],
            suggestions=["Review spending by category for potential savings."],
        )

    logger.info("report_node: report generated for month=%s user=%s", month, user_id)
    return {"report": report.model_dump()}
