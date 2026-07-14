"""
nodes/store.py — Write categorised transactions to SQLite.

Skips duplicates using INSERT OR IGNORE on a composite unique key
(user_id, date, description, amount) so re-uploading a CSV is safe.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("finance-assistant.store")

# Add a unique constraint at first use; SQLite DDL alter is limited so
# we recreate the index approach with INSERT OR IGNORE.
_UPSERT_SQL = """
    INSERT OR IGNORE INTO transactions
        (user_id, date, description, amount, category, source)
    VALUES (?, ?, ?, ?, ?, ?)
"""


from collections import Counter

def store_node(state: dict[str, Any], conn, user_id: str = "default") -> dict[str, Any]:
    """
    LangGraph node: bulk-insert categorised_transactions into SQLite.
    Frequency-aware deduplication preserves multiple legitimate identical transactions in a single CSV
    while preventing duplicate full-file re-uploads.
    """
    rows = state.get("categorized_transactions", [])
    if not rows:
        logger.warning("store_node: nothing to store")
        return {}

    cur = conn.cursor()
    cur.execute("DROP INDEX IF EXISTS idx_uniq_tx")

    # Count frequencies of existing transaction signatures in DB
    db_rows = cur.execute(
        "SELECT date, description, amount FROM transactions WHERE user_id = ?",
        (user_id,),
    ).fetchall()
    db_counts = Counter(
        (str(r[0]).strip(), str(r[1]).strip(), round(float(r[2]), 2))
        for r in db_rows
    )

    # Track frequencies seen in current batch
    batch_seen = Counter()
    new_rows = []

    for t in rows:
        sig = (
            str(t["date"]).strip(),
            str(t["description"]).strip(),
            round(float(t["amount"]), 2),
        )
        batch_seen[sig] += 1

        # Insert if current count in batch exceeds existing count in DB
        if batch_seen[sig] > db_counts[sig]:
            new_rows.append(
                (
                    user_id,
                    t["date"],
                    t["description"],
                    t["amount"],
                    t.get("category", "Other"),
                    "upload",
                )
            )

    if new_rows:
        cur.executemany(
            "INSERT INTO transactions (user_id, date, description, amount, category, source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            new_rows,
        )
        conn.commit()
        logger.info("store_node: %d new row(s) written for user=%s", len(new_rows), user_id)
    else:
        logger.info("store_node: all %d uploaded row(s) already exist in DB; skipped", len(rows))

    return {}
