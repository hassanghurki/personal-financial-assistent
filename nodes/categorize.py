"""
nodes/categorize.py — LLM-based transaction categorisation.

Design decisions:
- Batches 25 rows per LLM call (cheap, fast, reliable structured output).
- Merchant-description cache: descriptions already seen are looked up in
  SQLite before calling the LLM, so recurring merchants are free.
- Uses with_structured_output(CategorizedBatch) — no JSON parsing needed.
- Falls back to 'Other' if the LLM call fails for a batch.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from models import CategorizedBatch

load_dotenv()

logger = logging.getLogger("finance-assistant.categorize")

BATCH_SIZE = 25

_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_structured_llm = _llm.with_structured_output(CategorizedBatch)

CATEGORIZE_PROMPT = """\
You are a financial transaction categoriser. Categorise each transaction \
into EXACTLY one of these categories:
  Food, Rent, Transport, Shopping, Utilities, Entertainment, Health, Income, Other

Rules:
- Negative amounts are typically expenses; positive amounts may be income/refunds.
- Salary / direct deposit → Income.
- If unsure, use Other.
- Return ALL transactions — do not skip any.

Transactions (JSON list):
{transactions}
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _call_llm(batch: list[dict]) -> CategorizedBatch:
    prompt = CATEGORIZE_PROMPT.format(transactions=json.dumps(batch, indent=2))
    return _structured_llm.invoke(prompt)


def _load_cache(conn) -> dict[str, str]:
    """Load description→category mapping from the SQLite cache table."""
    rows = conn.execute("SELECT description, category FROM category_cache").fetchall()
    return {row["description"]: row["category"] for row in rows}


def _save_cache(conn, new_entries: dict[str, str]) -> None:
    """Persist newly learned description→category pairs."""
    conn.executemany(
        "INSERT OR REPLACE INTO category_cache (description, category) VALUES (?, ?)",
        new_entries.items(),
    )
    conn.commit()


def categorize_node(state: dict[str, Any], conn) -> dict[str, Any]:
    """
    LangGraph node: categorise raw_transactions using the LLM (with caching).
    Returns {'categorized_transactions': [...]}.
    """
    raw: list[dict] = state["raw_transactions"]
    cache = _load_cache(conn)

    cached_results: list[dict] = []
    needs_llm: list[dict] = []

    for tx in raw:
        desc = tx["description"]
        desc_upper = desc.upper()
        amt = float(tx["amount"])

        # Deterministic rule pre-check for clear income deposits
        if amt > 0 and any(kw in desc_upper for kw in ["PAYROLL", "SALARY", "DIRECT DEPOSIT", "STIPEND", "PAYCHECK"]):
            cached_results.append({**tx, "category": "Income", "confidence": 1.0})
        elif desc in cache:
            cached_results.append({**tx, "category": cache[desc], "confidence": 1.0})
        else:
            needs_llm.append(tx)

    logger.info(
        "categorize_node: %d cached, %d need LLM", len(cached_results), len(needs_llm)
    )

    new_cache: dict[str, str] = {}
    llm_results: list[dict] = []

    # Process in batches
    for i in range(0, len(needs_llm), BATCH_SIZE):
        batch = needs_llm[i : i + BATCH_SIZE]
        try:
            result: CategorizedBatch = _call_llm(batch)
            for t in result.transactions:
                d = t.model_dump()
                llm_results.append(d)
                new_cache[t.description] = t.category
        except Exception as exc:
            logger.error("categorize_node: LLM batch failed (%s), using 'Other'", exc)
            for tx in batch:
                llm_results.append({**tx, "category": "Other", "confidence": 0.0})

    if new_cache:
        _save_cache(conn, new_cache)

    all_results = cached_results + llm_results
    logger.info("categorize_node: %d transactions categorised", len(all_results))
    return {"categorized_transactions": all_results}
