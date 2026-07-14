"""
nodes/ingest.py — Parse and normalise an uploaded bank CSV with resilient amount/date parsing & row-level diagnostic reporting.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd
from dateutil import parser as date_parser

logger = logging.getLogger("finance-assistant.ingest")

# Column alias mapping for canonical names (kept distinct for debit/credit)
COLUMN_ALIASES: dict[str, str] = {
    "transaction date":   "date",
    "trans date":         "date",
    "posted date":        "date",
    "value date":         "date",
    "date posted":        "date",
    "narrative":          "description",
    "memo":               "description",
    "details":            "description",
    "particulars":        "description",
    "merchant":           "description",
    "payee":              "description",
    "description":        "description",
    "transaction amount": "amount",
    "total amount":       "amount",
    "withdrawal amount":  "debit",
    "deposit amount":     "credit",
}


def parse_amount(val: Any) -> float | None:
    """
    Resiliently parse currency amounts from various string formats:
    - "$2,600.00" -> 2600.0
    - "(1,450.00)" or "(1450)" -> -1450.0
    - "-$40.81" or "$-40.81" -> -40.81
    - "123.45 DR" -> -123.45 (Debit)
    - "$500.00 CR" -> 500.0 (Credit)
    - "-1.299,50" -> -1299.50 (European decimal format)
    - "-8.82E1" -> -88.20 (Scientific notation)
    Returns float or None if unparseable.
    """
    if val is None:
        return None

    if isinstance(val, pd.Series):
        val = val.dropna()
        if val.empty:
            return None
        val = val.iloc[0]

    if pd.isna(val):
        return None

    if isinstance(val, (int, float)):
        return float(val)

    val_str = str(val).strip()
    if not val_str:
        return None

    val_upper = val_str.upper()
    is_negative = False

    # Check DR / CR suffix
    if "DR" in val_upper:
        is_negative = True
    elif "CR" in val_upper:
        is_negative = False
    elif (val_str.startswith("(") and val_str.endswith(")")) or ("-" in val_str):
        is_negative = True

    # European format conversion: -1.299,50 -> -1299.50
    if re.search(r"\d+\.\d{3},\d{2}$", val_str):
        val_str = val_str.replace(".", "").replace(",", ".")

    # Scientific notation conversion: -8.82E1 -> -88.20
    if re.search(r"[eE][-+]?\d+", val_str):
        clean_sci = re.sub(r"[()$,€£\s]", "", val_str)
        try:
            return float(clean_sci)
        except ValueError:
            pass

    # Strip currency symbols, parentheses, commas, whitespace, minus signs, DR/CR
    clean_str = re.sub(r"[()$,€£\s\-]|DR|CR", "", val_str, flags=re.IGNORECASE)

    if not clean_str:
        return None

    try:
        num = float(clean_str)
        return -abs(num) if is_negative else num
    except ValueError:
        return None


def parse_date(val: Any) -> str | None:
    """
    Resiliently parse dates in ISO, US, European, or natural language formats:
    - "2026-07-02" -> "2026-07-02"
    - "Jul 3, 2026" / "July 3, 2026" -> "2026-07-03"
    - "07/03/2026" -> "2026-07-03"
    Returns YYYY-MM-DD string or None if unparseable.
    """
    if val is None:
        return None

    if isinstance(val, pd.Series):
        val = val.dropna()
        if val.empty:
            return None
        val = val.iloc[0]

    if pd.isna(val):
        return None

    val_str = str(val).strip()
    if not val_str:
        return None

    # Standard ISO format YYYY-MM-DD
    if re.match(r"^\d{4}-\d{2}-\d{2}$", val_str):
        return val_str

    try:
        dt = pd.to_datetime(val_str, format="mixed", errors="coerce")
        if not pd.isna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass

    try:
        dt = date_parser.parse(val_str, fuzzy=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def ingest_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: read a CSV from state['csv_path'], normalise it with row-level diagnostics,
    and return {'raw_transactions': [...], 'ingestion_summary': {...}}.
    """
    csv_path: str = state["csv_path"]
    logger.info("ingest_node: reading %s", csv_path)

    # Count raw file data lines (excluding header) for accounting audit
    raw_file_rows = 0
    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]
            raw_file_rows = max(0, len(lines) - 1)
    except Exception:
        pass

    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")
        except Exception as exc:
            logger.error("ingest_node: failed to read CSV (%s)", exc)
            return {
                "raw_transactions": [],
                "ingestion_summary": {
                    "total_rows": raw_file_rows,
                    "processed_count": 0,
                    "skipped_count": raw_file_rows,
                    "skipped_details": [{"row": 0, "reason": f"File read error: {exc}", "raw": {}}],
                },
            }

    # Track delimiter tokenization error skips at C-reader layer
    c_skipped_count = max(0, raw_file_rows - len(df))
    skipped: list[dict] = []

    if c_skipped_count > 0:
        skipped.append({
            "row": "Tokenization",
            "reason": f"{c_skipped_count} row(s) had malformed delimiter/field counts and were skipped by the CSV parser.",
            "raw": {"note": "Row skipped during CSV tokenization"}
        })

    # Normalize column headers
    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        if c in COLUMN_ALIASES:
            rename_map[c] = COLUMN_ALIASES[c]
    df = df.rename(columns=rename_map)

    has_amount = "amount" in df.columns
    has_debit = "debit" in df.columns
    has_credit = "credit" in df.columns

    processed: list[dict] = []

    for idx, row in df.iterrows():
        line_num = idx + 2  # 1-indexed CSV line number (line 1 is header)

        raw_dict = {str(k): (str(v) if not pd.isna(v) else "") for k, v in row.to_dict().items()}

        # Check for blank row
        non_empty = [v for v in raw_dict.values() if v.strip() != ""]
        if not non_empty:
            skipped.append({"row": line_num, "reason": "Blank row", "raw": raw_dict})
            continue

        # Check for repeated mid-file header row
        date_str_raw = str(row.get("date", "")).strip().lower()
        desc_str_raw = str(row.get("description", "")).strip().lower()
        if date_str_raw in ("date", "transaction date", "trans date") or desc_str_raw in ("description", "narrative", "details"):
            skipped.append({"row": line_num, "reason": "Repeated embedded CSV header row", "raw": raw_dict})
            continue

        # Parse date
        raw_date = row.get("date", None)
        parsed_date = parse_date(raw_date)
        if not parsed_date:
            skipped.append({
                "row": line_num,
                "reason": f"Unparseable date '{raw_date if not pd.isna(raw_date) else 'BLANK'}'",
                "raw": raw_dict
            })
            continue

        # Parse description
        raw_desc = row.get("description", None)
        desc_str = str(raw_desc).strip() if not pd.isna(raw_desc) else ""
        if not desc_str:
            skipped.append({
                "row": line_num,
                "reason": "Missing transaction description",
                "raw": raw_dict
            })
            continue

        # Parse amount
        parsed_amt = None
        raw_amt_disp = "BLANK"

        if has_amount:
            raw_amt = row.get("amount", None)
            raw_amt_disp = str(raw_amt) if not pd.isna(raw_amt) else "BLANK"
            parsed_amt = parse_amount(raw_amt)
        elif has_debit or has_credit:
            raw_debit = row.get("debit", None)
            raw_credit = row.get("credit", None)

            parsed_debit = parse_amount(raw_debit)
            parsed_credit = parse_amount(raw_credit)

            if parsed_debit is not None and parsed_debit != 0:
                parsed_amt = -abs(parsed_debit)
                raw_amt_disp = str(raw_debit)
            elif parsed_credit is not None and parsed_credit != 0:
                parsed_amt = abs(parsed_credit)
                raw_amt_disp = str(raw_credit)

        # Sanity Check: If parsed_amt is None OR parsed_amt == 0.0 when raw input was NOT explicit zero
        is_explicit_zero = raw_amt_disp.strip() in ("0", "0.0", "0.00", "$0", "$0.00")
        if parsed_amt is None or (parsed_amt == 0.0 and not is_explicit_zero):
            skipped.append({
                "row": line_num,
                "reason": f"Unparseable or zero-coerced amount '{raw_amt_disp}'",
                "raw": raw_dict
            })
            continue

        processed.append({
            "date": parsed_date,
            "description": desc_str,
            "amount": parsed_amt,
        })

    # Pass 2: Month cluster alignment for ambiguous dates (e.g. 01-07-2026 -> 2026-07-01 vs 2026-01-07)
    valid_yms = [t["date"][:7] for t in processed if t.get("date")]
    if valid_yms:
        from collections import Counter
        dominant_ym = Counter(valid_yms).most_common(1)[0][0]

        for t in processed:
            curr_ym = t["date"][:7]
            if curr_ym != dominant_ym:
                parts = t["date"].split("-")
                if len(parts) == 3:
                    swapped = f"{parts[0]}-{parts[2]}-{parts[1]}"
                    try:
                        dt_swap = pd.to_datetime(swapped, errors="coerce")
                        if not pd.isna(dt_swap) and dt_swap.strftime("%Y-%m") == dominant_ym:
                            logger.info("ingest_node: aligned date %s -> %s to match cluster %s", t["date"], dt_swap.strftime("%Y-%m-%d"), dominant_ym)
                            t["date"] = dt_swap.strftime("%Y-%m-%d")
                    except Exception:
                        pass

    actual_skipped_count = len(skipped)
    if c_skipped_count > 0:
        actual_skipped_count += (c_skipped_count - 1)  # -1 because one dict is added for all tokenization errors

    actual_total_rows = len(processed) + actual_skipped_count
    
    # If the file had completely blank lines that pandas dropped but we didn't count,
    # adjust the total to match what we actually observed.
    actual_total_rows = max(raw_file_rows, actual_total_rows)

    summary = {
        "total_rows": actual_total_rows,
        "processed_count": len(processed),
        "skipped_count": actual_total_rows - len(processed),
        "skipped_details": skipped,
    }

    logger.info(
        "ingest_node: processed %d/%d rows (%d skipped)",
        len(processed),
        actual_total_rows,
        summary["skipped_count"],
    )

    return {
        "raw_transactions": processed,
        "ingestion_summary": summary,
    }
