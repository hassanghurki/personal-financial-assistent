"""
nodes/qa_agent.py — Multi-step Q&A using LangGraph's built-in ReAct agent.

The agent has access to 4 read-only, purpose-built tools:
  1. spending_by_category  — per-category totals for a month
  2. spending_by_week      — weekly totals within a month
  3. historical_average    — trailing N-month average (optional category filter)
  4. compare_periods       — diff two months side-by-side

Why tools instead of raw SQL access:
- Bounded, auditable action space — the agent decides WHICH tool to call,
  the tool handles the SQL correctly (no off-by-one, correct boundaries).
- Every tool call is logged → every answer is traceable.
- Easy to unit-test each tool independently.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv()

logger = logging.getLogger("finance-assistant.qa")

import calendar
import datetime


def _format_date_range(start: datetime.date, end: datetime.date) -> str:
    """Format a date range like 'July 8-14' or 'June 28-July 4'."""
    if start.month == end.month:
        return f"{start.strftime('%B')} {start.day}-{end.day}"
    return f"{start.strftime('%B %d')}-{end.strftime('%B %d')}"


def _month_week_bounds(year: int, month: int, week_idx: int) -> tuple[datetime.date, datetime.date]:
    """Return the calendar start/end dates for a 0-based week index within a month."""
    last_day = calendar.monthrange(year, month)[1]
    start_day = week_idx * 7 + 1
    end_day = min((week_idx + 1) * 7, last_day)
    return datetime.date(year, month, start_day), datetime.date(year, month, end_day)


def _build_system_prompt(conn) -> str:
    """Build a date-stamped system prompt so the agent always knows the real current date."""
    today = datetime.date.today()
    today_str = today.strftime("%Y-%m-%d")
    current_month_str = today.strftime("%Y-%m")

    # Try to find the latest month actually present in the database
    try:
        row = conn.execute(
            "SELECT strftime('%Y-%m', MAX(date)) as latest FROM transactions WHERE user_id='default'"
        ).fetchone()
        db_latest_month = row[0] if row and row[0] else current_month_str
    except Exception:
        db_latest_month = current_month_str

    return f"""\
You are a knowledgeable, warm, and concise personal finance advisor embedded in a finance dashboard.

DATE CONTEXT (CRITICAL — ALWAYS USE THESE VALUES):
- Today's date is: {today_str}
- Current month (YYYY-MM): {current_month_str}
- Latest month with data in the database: {db_latest_month}
- When the user says "this month", use {current_month_str}.
- When the user says "last month", compute the month before {current_month_str}.
- NEVER rely on your training-data knowledge of dates. ALWAYS use the values above.

══════════════════════════════════════════
TOOL USAGE RULES
══════════════════════════════════════════

1. LOGGING EXPENSES / INCOME
   - Trigger phrases: "record", "log", "add", "spent", "I paid", "track"
     e.g. "Record $120 for Electric Bill" or "I spent $45 on groceries"
   - YOU MUST call `log_expense` immediately — do NOT ask clarifying questions first.
   - Default date to {today_str} if the user doesn't specify one.
   - After success, confirm with a short friendly message:
     e.g. "Logged Electric Bill — $120.00 under Utilities on {today_str}."

   DUPLICATE DETECTION:
   - If `log_expense` returns status="duplicate", show the duplicate and ask:
     "This looks like it already exists: [details]. Add it again anyway?"
   - If user confirms → call `log_expense` again with force="yes".
   - If user declines → confirm nothing was added.

2. QUERYING & ANALYTICS
   - Always call the appropriate tool before answering ANY spending question.
   - Never make up numbers. Every figure must come from a tool call.
   - When no data is found for a period, respond warmly:
     "I don't see any transactions for [period] yet. Try uploading a bank statement on the Upload tab!"
   - Prefer using `spending_by_week` for "this month" breakdowns unless the user asks for categories specifically.

3. DELETING TRANSACTIONS (CONFIRMATION REQUIRED)
   - Trigger phrases: "delete", "remove", "clear", "erase"
   - Call `delete_transaction` with the search params first (confirm="").
   - Show matching transactions and ask: "Are you sure you want to permanently delete these?"
   - Only call `delete_transaction` with confirm="yes" after explicit user confirmation.
   - If declined, confirm nothing was deleted.

══════════════════════════════════════════
RESPONSE FORMAT (STRICT — FOLLOW EXACTLY)
══════════════════════════════════════════

Write plain, valid GitHub-flavored Markdown. No exceptions to the rules below:

- Start with a short header, e.g. `### Spending Breakdown — July 2026`
- Each bullet is its own line, starting with `- `. Never put two bullets on one line.
- Use a real Markdown table when comparing categories or months side by side —
  do not write comparisons as inline bullet lists of numbers.
- Bold only whole words or whole dollar amounts (e.g. `**$1,234.56**`). Never bold
  a lone punctuation mark, and never leave a stray `*` in the text.
- Use plain hyphens `-` for bullets and minus signs. Do not use `—`, `−`, or `|` as
  bullet separators.
- Numbers: always `$1,234.56` format (dollar sign, comma thousands separator, 2 decimals).
- End with one short, clear insight or recommendation in a normal sentence.

Follow this exact shape for a category comparison between two months:

### Spending Breakdown — July 2026 vs June 2026

| Category      | July 2026  | June 2026  | Change     |
|----------------|-----------:|-----------:|-----------:|
| Food           | $1,427.50  | $396.86    | +$1,030.64 |
| Shopping       | $2,668.98  | $715.66    | +$1,953.32 |
| Rent           | $0.00      | $2,900.00  | -$2,900.00 |

**Total spending:** $8,959.46 in July vs $8,018.97 in June (+$940.49).

The biggest driver of the increase was Shopping and Food. Consider reviewing
recent purchases in those categories if this wasn't planned.

Match this structure and tone for every analytical answer. Do not deviate from
the table format for comparisons, and do not compress bullets or table rows
onto a single line.
"""

MAX_ITERATIONS = 12  # cap recursion to control cost


def make_tools(conn, user_id: str = "default") -> list:
    """Create the read-only and write finance tools bound to this user's data."""

    @tool
    def log_expense(amount: str, description: str, category: str = "", date: str = "", force: str = "") -> dict:
        """Record or log a manual expense or income transaction in the database.

        Args:
            amount: The numerical or dollar amount of the transaction (e.g. 120, "$120", 45.50).
            description: Short text description of the merchant/item (e.g. 'Electric Bill', 'Groceries', 'Coffee', 'Salary').
            category: Optional category string (Food, Rent, Transport, Shopping, Utilities, Entertainment, Health, Income, Other).
            date: Optional YYYY-MM-DD date. Defaults to today's date if omitted.
            force: Set to "yes" to insert even if a duplicate exists (user confirmed they want to add it again).
        """
        try:
            cleaned_amt = str(amount).replace("$", "").replace(",", "").strip()
            num_amount = float(cleaned_amt)
        except Exception:
            num_amount = 0.0

        if not date:
            date = datetime.date.today().strftime("%Y-%m-%d")

        cat_upper = category.strip().title() if category else ""
        valid_cats = ["Food", "Rent", "Transport", "Shopping", "Utilities", "Entertainment", "Health", "Income", "Other"]

        if cat_upper not in valid_cats:
            # Auto-infer category from description keywords if not explicitly valid
            desc_lower = description.lower()
            if any(w in desc_lower for w in ["food", "grocery", "groceries", "restaurant", "burger", "pizza", "coffee", "starbucks", "lunch", "dinner"]):
                cat_upper = "Food"
            elif any(w in desc_lower for w in ["rent", "lease", "housing"]):
                cat_upper = "Rent"
            elif any(w in desc_lower for w in ["uber", "lyft", "gas", "fuel", "train", "bus", "transport", "parking"]):
                cat_upper = "Transport"
            elif any(w in desc_lower for w in ["utility", "utilities", "electric", "water", "internet", "wifi", "bill", "phone"]):
                cat_upper = "Utilities"
            elif any(w in desc_lower for w in ["movie", "netflix", "spotify", "game", "cinema", "entertainment"]):
                cat_upper = "Entertainment"
            elif any(w in desc_lower for w in ["doctor", "pharmacy", "medicine", "health", "gym", "dental"]):
                cat_upper = "Health"
            elif any(w in desc_lower for w in ["salary", "payroll", "stipend", "income", "freelance", "deposit"]):
                cat_upper = "Income"
            elif any(w in desc_lower for w in ["store", "amazon", "clothes", "shopping", "shoes"]):
                cat_upper = "Shopping"
            else:
                cat_upper = "Other"

        # Expenses stored as negative numbers, Income stored as positive
        final_amount = abs(num_amount) if cat_upper == "Income" else -abs(num_amount)

        # ── Duplicate detection (skip if user forced re-add) ───────────────────
        if force.strip().lower() not in ("yes", "y", "sure", "yeah", "true"):
            existing = conn.execute(
                "SELECT date, description, amount, category FROM transactions "
                "WHERE user_id = ? AND date = ? AND description = ? AND amount = ?",
                (user_id, date, description, final_amount),
            ).fetchone()
            if existing:
                dup_date, dup_desc, dup_amt, dup_cat = existing
                return {
                    "status": "duplicate",
                    "message": (
                        f"A matching transaction already exists: "
                        f"{dup_desc} — ${abs(dup_amt):,.2f} ({dup_cat}) on {dup_date}. "
                        f"Would you like to add it again anyway?"
                    ),
                    "existing": {
                        "date": dup_date,
                        "description": dup_desc,
                        "amount": abs(dup_amt),
                        "category": dup_cat,
                    },
                }

        conn.execute(
            "INSERT INTO transactions (user_id, date, description, amount, category, source) "
            "VALUES (?, ?, ?, ?, ?, 'manual')",
            (user_id, date, description, final_amount, cat_upper),
        )
        conn.commit()

        logger.info("log_expense: added %s ($%.2f) on %s [%s]", description, final_amount, date, cat_upper)

        return {
            "status": "success",
            "message": f"Successfully logged ${abs(final_amount):,.2f} for \"{description}\" under category {cat_upper} on {date}.",
            "date": date,
            "description": description,
            "amount": final_amount,
            "category": cat_upper,
        }

    @tool
    def spending_by_category(month: str) -> dict:
        """Total spending per category for a given month (format YYYY-MM).
        Returns a dict {category: total_amount}."""
        df = pd.read_sql(
            "SELECT * FROM transactions WHERE user_id = ? AND is_outlier = 0 AND category != 'Income'", conn, params=(user_id,)
        )
        if df.empty:
            return {}
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
        subset = df[df["month"] == month]
        result = subset.groupby("category")["amount"].apply(lambda x: x.abs().sum())
        return result.round(2).to_dict()

    @tool
    def spending_by_week(month: str) -> list[dict]:
        """Total spending per week for a given month (format YYYY-MM).
        Returns a list of dicts with week_label, date_range, total, and top_categories.
        The week_label is a human-friendly 'Week 1', 'Week 2', etc. relative to the month.
        top_categories shows the top 3 spending categories for that week."""
        df = pd.read_sql(
            "SELECT * FROM transactions WHERE user_id = ? AND is_outlier = 0 AND category != 'Income'", conn, params=(user_id,)
        )
        if df.empty:
            return []
        df["date_parsed"] = pd.to_datetime(df["date"])
        df["month"] = df["date_parsed"].dt.to_period("M").astype(str)
        df = df[df["month"] == month]
        if df.empty:
            return []

        year, month_num = map(int, month.split("-"))
        # Calendar weeks within the month: days 1-7, 8-14, 15-21, etc.
        df["week_idx"] = (df["date_parsed"].dt.day - 1) // 7
        today = datetime.date.today()

        result = []
        for week_idx in sorted(df["week_idx"].unique()):
            week_df = df[df["week_idx"] == week_idx]
            total = round(float(week_df["amount"].abs().sum()), 2)

            start_date, end_date = _month_week_bounds(year, month_num, week_idx)
            date_range = _format_date_range(start_date, end_date)

            # Top 3 categories by spend for that week
            cat_totals = week_df.groupby("category")["amount"].apply(lambda x: x.abs().sum()).sort_values(ascending=False)
            top_cats = {cat: round(float(amt), 2) for cat, amt in cat_totals.head(3).items()}

            week_label = f"Week {week_idx + 1}"
            if start_date <= today <= end_date and month == today.strftime("%Y-%m"):
                week_label += " (partial)"

            result.append({
                "week": week_label,
                "week_label": week_label,
                "date_range": date_range,
                "total": total,
                "top_categories": top_cats,
            })

        return result

    @tool
    def historical_average(category: str = "", months_back: int = 6) -> dict:
        """Average monthly spend over the trailing N months (default 6),
        optionally filtered to one category. Excludes the current in-progress month.
        Returns {average: float, months_counted: int, latest_month: str}."""
        df = pd.read_sql(
            "SELECT * FROM transactions WHERE user_id = ? AND is_outlier = 0 AND category != 'Income'", conn, params=(user_id,)
        )
        if df.empty:
            return {"average": 0.0, "months_counted": 0, "latest_month": "N/A"}
        df["date_parsed"] = pd.to_datetime(df["date"])
        df["month"] = df["date_parsed"].dt.to_period("M")
        current = df["month"].max()
        latest_month = str(current)

        window = df[df["month"] < current].copy()
        window = window[window["month"] >= current - months_back]
        if category:
            window = window[window["category"] == category]

        monthly = window.groupby("month")["amount"].apply(lambda x: x.abs().sum())
        from nodes.anomaly import trim_outliers
        clean_monthly = trim_outliers(monthly)
        avg = round(float(clean_monthly.mean()), 2) if not clean_monthly.empty else 0.0
        return {
            "average": avg,
            "months_counted": len(clean_monthly),
            "latest_month": latest_month,
        }

    @tool
    def compare_periods(month_a: str, month_b: str) -> dict:
        """Compare total and per-category spending between two months (format YYYY-MM).
        Returns {month_a_total, month_b_total, diff_by_category}
        where positive diff means month_a > month_b."""
        df = pd.read_sql(
            "SELECT * FROM transactions WHERE user_id = ? AND is_outlier = 0 AND category != 'Income'", conn, params=(user_id,)
        )
        if df.empty:
            return {}
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
        a = df[df["month"] == month_a].groupby("category")["amount"].apply(lambda x: x.abs().sum())
        b = df[df["month"] == month_b].groupby("category")["amount"].apply(lambda x: x.abs().sum())
        diff = (a - b).fillna(a).fillna(-b).round(2)
        return {
            "month_a_total": round(float(a.sum()), 2),
            "month_b_total": round(float(b.sum()), 2),
            "diff_by_category": diff.to_dict(),
        }

    @tool
    def delete_transaction(
        description: str = "",
        amount: str = "",
        date: str = "",
        transaction_id: int = 0,
        confirm: str = "",
    ) -> dict:
        """Delete specific transactions or matching transactions from the database after explicit user confirmation.

        Args:
            description: Keyword/name to match in description (e.g. 'Yacht', 'Starbucks', 'SUSPICIOUS').
            amount: Optional transaction amount to match (e.g. 95000, 450000, '$95,000').
            date: Optional YYYY-MM-DD date string.
            transaction_id: Optional exact integer ID of the transaction if known.
            confirm: Set to 'yes' ONLY IF the user has explicitly confirmed they want to delete these transactions. Leave empty for initial search/preview.
        """
        params = [user_id]
        sql_parts = ["WHERE user_id = ?"]

        if transaction_id > 0:
            sql_parts.append("AND id = ?")
            params.append(transaction_id)
        else:
            if description.strip():
                sql_parts.append("AND description LIKE ?")
                params.append(f"%{description.strip()}%")
            if date.strip():
                sql_parts.append("AND date = ?")
                params.append(date.strip())
            if amount:
                try:
                    num = abs(float(str(amount).replace("$", "").replace(",", "").strip()))
                    sql_parts.append("AND (ABS(amount) = ? OR amount = ?)")
                    params.extend([num, -num])
                except Exception:
                    pass

        query_sql = "SELECT id, date, description, amount, category FROM transactions " + " ".join(sql_parts)
        matches = conn.execute(query_sql, params).fetchall()

        if not matches:
            return {
                "status": "not_found",
                "message": "No matching transactions found in the database to delete.",
                "matches": [],
            }

        match_list = [
            {
                "id": m[0],
                "date": m[1],
                "description": m[2],
                "amount": abs(m[3]),
                "category": m[4],
            }
            for m in matches
        ]

        # STEP 1: If user hasn't explicitly confirmed yet, return matches for user preview
        if confirm.strip().lower() not in ("yes", "y", "sure", "yeah", "true", "force", "confirm"):
            match_details = ", ".join(
                f"ID #{m['id']}: {m['description']} (${m['amount']:,.2f} on {m['date']})"
                for m in match_list[:5]
            )
            return {
                "status": "needs_confirmation",
                "message": (
                    f"Found {len(match_list)} matching transaction(s): {match_details}. "
                    f"Are you sure you want to permanently delete these transaction(s)?"
                ),
                "matches": match_list,
            }

        # STEP 2: Confirmed! Execute deletion (strictly scoped to user_id)
        match_ids = [m["id"] for m in match_list]
        placeholders = ",".join(["?"] * len(match_ids))
        conn.execute(f"DELETE FROM transactions WHERE user_id = ? AND id IN ({placeholders})", [user_id] + match_ids)
        conn.commit()

        logger.info("delete_transaction: deleted %d transaction(s) for user=%s", len(match_ids), user_id)

        return {
            "status": "success",
            "message": f"Successfully deleted {len(match_ids)} transaction(s) from the database.",
            "deleted_count": len(match_ids),
            "matches": match_list,
        }

    @tool
    def get_flagged_transactions(month: str) -> list[dict]:
        """Fetch transactions flagged as unusually large individual outliers for a given month (format YYYY-MM).
        Returns a list of transaction details."""
        df = pd.read_sql(
            "SELECT date, description, amount, category FROM transactions WHERE user_id = ? AND is_outlier = 1", conn, params=(user_id,)
        )
        if df.empty:
            return []
        df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
        subset = df[df["month"] == month].copy()

        return subset[["date", "description", "amount", "category"]].to_dict(orient="records")

    return [log_expense, delete_transaction, spending_by_category, spending_by_week, historical_average, compare_periods, get_flagged_transactions]


def build_qa_agent(conn, user_id: str = "default"):
    """
    Build the ReAct agent once at startup; reuse it across all Q&A requests.
    The system prompt is generated fresh each time so it contains the real current date.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = make_tools(conn, user_id)
    system_prompt = _build_system_prompt(conn)
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    logger.info("qa_agent: agent built for user=%s with today=%s", user_id, datetime.date.today())
    return agent


import re


def _clean_markdown(text: str) -> str:
    """Light, non-destructive cleanup of LLM markdown output.

    Deliberately minimal: earlier versions of this function used aggressive
    regexes to rewrite bullets/bold spacing, which frequently corrupted
    well-formed markdown (stripped spaces inside sentences, mangled bold
    markers, etc). The system prompt now does the heavy lifting by asking
    the model for a strict, example-driven format, so this function only
    fixes clearly broken artifacts rather than rewriting structure.
    """
    if not text:
        return ""

    # Normalize the Unicode math asterisk (U+2217) to a plain ASCII '*'.
    text = text.replace("\u2217", "*")

    # If a bullet got glued onto the end of the previous line
    # (e.g. "...text. - **Food**: ..."), push it onto its own line.
    text = re.sub(r'(?<!\n)[ \t]-[ \t]\*\*', '\n- **', text)

    # Make sure headers start on their own line.
    text = re.sub(r'(?<!\n)\n?(#{1,6}[ \t])', r'\n\n\1', text)

    # Collapse 3+ blank lines down to a single blank line.
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def qa_node(state: dict[str, Any], agent) -> dict[str, Any]:
    """
    LangGraph node (or standalone callable): invoke the ReAct agent with full
    conversation message history or user_question and return {'answer': '...'}.
    """
    history = state.get("messages", [])
    question: str = state.get("user_question", "")

    input_messages: list[dict[str, str]] = []
    for msg in history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            input_messages.append({"role": msg["role"], "content": msg["content"]})
        elif hasattr(msg, "content"):
            role = getattr(msg, "type", "user")
            if role == "human": role = "user"
            elif role == "ai": role = "assistant"
            input_messages.append({"role": role, "content": msg.content})

    if question and (not input_messages or input_messages[-1]["content"] != question):
        input_messages.append({"role": "user", "content": question})

    logger.info("qa_node: invoking agent with %d message turns", len(input_messages))

    try:
        result = agent.invoke(
            {"messages": input_messages},
            config={"recursion_limit": MAX_ITERATIONS},
        )
        final = result["messages"][-1]
        raw_answer = final.content if hasattr(final, "content") else str(final)
        answer = _clean_markdown(raw_answer)
    except Exception as exc:
        import traceback
        logger.error("qa_node: agent failed — %s\n%s", exc, traceback.format_exc())
        answer = (
            f"Something went wrong while processing your request. "
            f"Error: `{type(exc).__name__}: {exc}`. "
            "Please try rephrasing, or check the logs for details."
        )

    return {"answer": answer}