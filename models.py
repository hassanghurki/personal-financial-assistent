"""
models.py — Pydantic schemas for structured LLM output.

Using structured output means the LLM returns a validated Python object,
not raw JSON that might be malformed.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

# ── Category taxonomy ──────────────────────────────────────────────────────────
Category = Literal[
    "Food",
    "Rent",
    "Transport",
    "Shopping",
    "Utilities",
    "Entertainment",
    "Health",
    "Income",
    "Other",
]


# ── Categorization ─────────────────────────────────────────────────────────────
class CategorizedTransaction(BaseModel):
    date: str
    description: str
    amount: float
    category: Category
    confidence: float = Field(ge=0.0, le=1.0, description="LLM confidence 0–1")


class CategorizedBatch(BaseModel):
    transactions: List[CategorizedTransaction]


# ── Anomaly ────────────────────────────────────────────────────────────────────
class AnomalyFlag(BaseModel):
    category: str
    current_amount: float
    average_amount: float
    pct_increase: float
    message: str
    severity: Literal["low", "medium", "high"]


# ── Monthly Report ─────────────────────────────────────────────────────────────
class MonthlyReport(BaseModel):
    month: str
    total_spent: float
    total_income: float
    savings_rate: float = Field(description="Savings as a % of income")
    top_categories: List[str] = Field(description="Top 3 spending categories by amount")
    anomalies: List[AnomalyFlag] = Field(default_factory=list)
    suggestions: List[str] = Field(description="2–3 specific, actionable tips")
