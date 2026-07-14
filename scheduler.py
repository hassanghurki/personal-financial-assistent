"""
scheduler.py — Weekly background job using APScheduler.

Runs every Sunday at 18:00, generates the monthly report, and logs it.
Email delivery is intentionally disabled for now — uncomment the
send_email block and add SMTP_USER/SMTP_PASS to .env when ready.
"""

from __future__ import annotations

import json
import logging
import os

from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from db import init_db
from graph import build_graph

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("finance-assistant.scheduler")


def weekly_job() -> None:
    """Generate a weekly report and print it to stdout (or email when enabled)."""
    conn = init_db()
    graph = build_graph(conn)

    # The graph requires a csv_path to start — for the scheduled run we skip
    # the ingest/categorize nodes by invoking report_node directly.
    from nodes.anomaly import anomaly_node
    from nodes.report import report_node

    anomaly_state = anomaly_node({}, conn)
    report_state = report_node(anomaly_state, conn)
    report = report_state.get("report", {})

    summary = (
        f"\n{'='*48}\n"
        f"  📊  Weekly Finance Summary\n"
        f"  Month        : {report.get('month', 'N/A')}\n"
        f"  Total Spent  : ${report.get('total_spent', 0):.2f}\n"
        f"  Total Income : ${report.get('total_income', 0):.2f}\n"
        f"  Savings Rate : {report.get('savings_rate', 0):.1f}%\n"
        f"  Top Categories: {', '.join(report.get('top_categories', []))}\n"
        f"\n  Suggestions:\n"
        + "".join(f"    • {s}\n" for s in report.get("suggestions", []))
        + f"{'='*48}\n"
    )
    logger.info(summary)

    # ── Uncomment below to enable email delivery ───────────────────────────────
    # import smtplib
    # from email.mime.text import MIMEText
    # msg = MIMEText(summary)
    # msg["Subject"] = f"Weekly Finance Summary — {report.get('month', '')}"
    # msg["From"] = os.environ["SMTP_USER"]
    # msg["To"]   = os.environ["SMTP_USER"]
    # with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    #     server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
    #     server.send_message(msg)
    # logger.info("scheduler: weekly email sent")


if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(weekly_job, "cron", day_of_week="sun", hour=18)
    logger.info("scheduler: starting — will run every Sunday at 18:00")
    scheduler.start()
