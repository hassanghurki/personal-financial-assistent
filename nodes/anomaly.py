"""
nodes/anomaly.py — Statistical anomaly detection with single-transaction outlier flagging & robust baseline statistics.

Strategy:
1. Individual Outlier Detection: Flags single transactions that are unusually large
   (e.g. >5-10x trailing transaction baseline) as standalone alerts ("⚠️ Unusually large transaction...").
2. Outlier-Resilient Baseline Calculation: Trims extreme values before computing category
   baseline averages and standard deviations, preventing typos/wire transfers from corrupting baseline math.
3. Category-Level Monthly Anomaly Detection: Z-score evaluation against trimmed category baselines.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger = logging.getLogger("finance-assistant.anomaly")

Z_THRESHOLD = 2.5  # flag if category spend is >2.5 std-devs above robust category baseline

_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)


def trim_outliers(series: pd.Series, max_multiplier: float = 3.5) -> pd.Series:
    """
    Trim extreme statistical outliers from baseline history using robust IQR/median bounds.
    Prevents one-off large wire transfers, fraud, or typos from distorting baseline averages.
    """
    if series.empty or len(series) < 3:
        return series

    median = series.median()
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    iqr = q75 - q25

    upper_bound = max(median * max_multiplier, q75 + 3.0 * iqr) if iqr > 0 else median * max_multiplier
    trimmed = series[series <= upper_bound]

    return trimmed if len(trimmed) >= 2 else series


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _phrase_anomaly(category: str, current: float, average: float, pct: float) -> str:
    prompt = (
        f"You are a friendly finance assistant. Write ONE short sentence (max 20 words) "
        f"explaining that the user spent ${current:,.0f} on {category} this month, "
        f"which is {pct:.0f}% above their usual ${average:,.0f}. Be specific."
    )
    return _llm.invoke(prompt).content.strip()


def anomaly_node(state: dict[str, Any], conn, user_id: str = "default") -> dict[str, Any]:
    """
    LangGraph node: detect both individual single-transaction outliers and category-level monthly spending anomalies.
    Returns {'anomalies': [...]}.
    """
    import datetime

    df = pd.read_sql(
        "SELECT id, date, description, category, amount FROM transactions WHERE user_id = ? AND category != 'Income'",
        conn,
        params=(user_id,),
    )

    if df.empty:
        logger.warning("anomaly_node: no transaction data found")
        return {"anomalies": []}

    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")

    # Always work with absolute expense amounts
    expense_df = df[df["amount"] < 0].copy()
    expense_df["amount"] = expense_df["amount"].abs()

    if expense_df.empty:
        expense_df = df.copy()
        expense_df["amount"] = expense_df["amount"].abs()

    # Determine real calendar current month
    today_period = pd.Period(datetime.date.today(), "M")
    db_max_month = expense_df["month"].max()
    current_month = today_period if today_period <= db_max_month else db_max_month

    anomalies: list[dict] = []

    # Build signature set for transactions in the current upload batch (if present)
    uploaded_txs = state.get("categorized_transactions", [])
    uploaded_sigs = set()
    uploaded_categories = set()
    if uploaded_txs:
        for t in uploaded_txs:
            try:
                d_str = pd.to_datetime(t.get("date")).strftime("%Y-%m-%d")
            except Exception:
                d_str = str(t.get("date", ""))
            desc = str(t.get("description", "")).strip().lower()
            amt = round(abs(float(t.get("amount", 0))), 2)
            uploaded_sigs.add((d_str, desc, amt))
            if t.get("category"):
                uploaded_categories.add(t.get("category"))

    # ── PASS 1: Single-Transaction Individual Outliers ─────────────────────────
    # We scan ALL expenses so massive outliers in past months (like a $450k yacht charter)
    # get correctly marked as is_outlier=1 in the database and excluded from charts.
    history_txs = expense_df[expense_df["month"] < current_month]["amount"]
    current_txs = expense_df[expense_df["month"] == current_month]

    # Derive outlier threshold
    if not history_txs.empty:
        trimmed_tx_history = trim_outliers(history_txs, max_multiplier=4.0)
        max_normal_tx = trimmed_tx_history.max() if not trimmed_tx_history.empty else history_txs.max()
        median_normal_tx = trimmed_tx_history.median() if not trimmed_tx_history.empty else history_txs.median()
        tx_outlier_threshold = max(max_normal_tx * 5.0, median_normal_tx * 8.0, 6000.0)
    else:
        batch_amounts = current_txs["amount"]
        if not batch_amounts.empty:
            batch_median = batch_amounts.median()
            batch_p75 = batch_amounts.quantile(0.75)
            max_normal_tx = batch_p75
            tx_outlier_threshold = max(batch_median * 10.0, batch_p75 * 5.0, 5000.0)
        else:
            max_normal_tx = 0.0
            tx_outlier_threshold = float("inf")

    already_flagged_ids = set(a.get("tx_id") for a in anomalies if "tx_id" in a)

    # Scan ENTIRE expense_df, not just current_txs
    for _, row in expense_df.iterrows():
        amt = row["amount"]

        if row["category"] in ("Rent", "Mortgage"):
            continue

        if amt >= tx_outlier_threshold:
            desc = row["description"]
            dt_raw = row["date"]
            tx_id = row["id"]

            if tx_id in already_flagged_ids:
                continue

            dt_str = str(dt_raw)
            try:
                dt_str = pd.to_datetime(dt_raw).strftime("%m/%d")
            except Exception:
                pass

            # Mark the transaction as an outlier in the database
            conn.execute("UPDATE transactions SET is_outlier = 1 WHERE id = ?", (tx_id,))
            conn.commit()

            # Filter: only report individual anomaly if transaction belongs to current uploaded batch
            # We check if the key exists in state to distinguish between CSV upload vs full report generation
            if "categorized_transactions" in state:
                try:
                    row_d_str = pd.to_datetime(dt_raw).strftime("%Y-%m-%d")
                except Exception:
                    row_d_str = str(dt_raw)
                row_desc = str(desc).strip().lower()
                row_amt = round(float(amt), 2)
                if (row_d_str, row_desc, row_amt) not in uploaded_sigs:
                    continue

            message = f"⚠️ Unusually large transaction: ${amt:,.2f} for '{desc}' on {dt_str} — please verify this is correct."
            anomalies.append(
                {
                    "tx_id": tx_id,
                    "category": row["category"],
                    "current_amount": round(amt, 2),
                    "average_amount": round(max_normal_tx, 2),
                    "pct_increase": round((amt / max_normal_tx * 100) if max_normal_tx else 0, 1),
                    "message": message,
                    "severity": "high",
                    "type": "individual_outlier",
                }
            )
            already_flagged_ids.add(tx_id)

    # ── PASS 2: Category Monthly Summary Anomalies (Robust Trimmed Baseline) ──
    # Exclude extreme individual outliers from category monthly totals baseline calculation
    has_history = 'history_txs' in locals() and not history_txs.empty
    tx_cutoff = (max_normal_tx * 5.0) if (has_history and 'max_normal_tx' in locals()) else float("inf")
    clean_expense_df = expense_df[expense_df["amount"] < tx_cutoff]

    monthly = (
        clean_expense_df
        .groupby(["month", "category"])["amount"]
        .sum()
        .reset_index()
    )

    for category in monthly["category"].unique():
        if "categorized_transactions" in state and category not in uploaded_categories:
            continue
        cat_data = monthly[monthly["category"] == category]

        history_rows = cat_data[cat_data["month"] < current_month]
        history = history_rows[history_rows["amount"] > 0]["amount"]
        current_row = cat_data[cat_data["month"] == current_month]["amount"]

        if len(history) < 2 or current_row.empty:
            continue

        # Trim extreme baseline outliers from monthly history
        clean_history = trim_outliers(history, max_multiplier=3.5)

        avg = clean_history.mean()
        std = clean_history.std()
        cur_val = current_row.values[0]

        if std > 0 and (cur_val - avg) / std > Z_THRESHOLD:
            pct = (cur_val - avg) / avg * 100
            diff = cur_val - avg
            
            # Reduce noise by ensuring it's both a high percentage and a meaningful absolute amount
            if pct > 500 and diff > 2500:
                severity = "high" if pct > 75 else ("medium" if pct > 35 else "low")
                try:
                    message = _phrase_anomaly(category, cur_val, avg, pct)
                except Exception as exc:
                    logger.error("anomaly_node: LLM phrasing failed (%s)", exc)
                    message = (
                        f"You spent ${cur_val:,.0f} on {category} this month — "
                        f"{pct:.0f}% above your average of ${avg:,.0f}."
                    )

                anomalies.append(
                    {
                        "category": category,
                        "current_amount": round(cur_val, 2),
                        "average_amount": round(avg, 2),
                        "pct_increase": round(pct, 1),
                        "message": message,
                        "severity": severity,
                        "type": "category_monthly",
                    }
                )

    logger.info("anomaly_node: %d total anomaly flags generated for user=%s", len(anomalies), user_id)
    return {"anomalies": anomalies}
