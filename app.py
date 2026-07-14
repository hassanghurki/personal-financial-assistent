"""
app.py — Streamlit UI for the Personal Finance Assistant.

Tabs:
  1. Upload Statement  — CSV → ingest → categorize → store → anomaly → report
  2. Ask a Question    — ReAct Q&A agent with 4 read-only tools
  3. Monthly Report    — view / refresh the latest stored report
  4. Transactions      — browse all stored rows with filters

Run:
  streamlit run app.py
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import datetime as _dt
# pyrefly: ignore [missing-import]
import plotly.express as px
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from db import init_db
from graph import build_graph
from nodes.qa_agent import build_qa_agent, qa_node
from auth import render_auth_page, is_logged_in, get_current_user_id, get_current_username, logout

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💰 Personal Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load styles ────────────────────────────────────────────────────────────────
def _load_css(path: str) -> None:
    with open(path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css(os.path.join(os.path.dirname(__file__), "style.css"))

# ── Initialise DB (always needed, even for login page) ────────────────────────
@st.cache_resource
def _get_db():
    return init_db()

conn = _get_db()

# ── Auth gate — show login if not authenticated ────────────────────────────────
if not is_logged_in():
    render_auth_page(conn)
    st.stop()

# ── Authenticated — get current user context ───────────────────────────────────
_user_id   = get_current_user_id()
_username  = get_current_username()

# ── Per-user graph + agent (cache-busted by user_id + qa_agent.py mtime + date)
@st.cache_resource
def get_resources(user_id: str, mtime: float = 0.0, today_date: str = ""):
    graph = build_graph(conn, user_id=user_id)
    agent = build_qa_agent(conn, user_id=user_id)
    return graph, agent

_qa_file = os.path.join(os.path.dirname(__file__), "nodes", "qa_agent.py")
_mtime   = os.path.getmtime(_qa_file) if os.path.exists(_qa_file) else 0.0
_today   = _dt.date.today().isoformat()
graph, qa_agent = get_resources(user_id=_user_id, mtime=_mtime, today_date=_today)

# ── Hero header + user bar ─────────────────────────────────────────────────────
_avatar_letter = _username[0].upper() if _username else "U"

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-user-container">
          <div class="user-avatar">{_avatar_letter}</div>
          <span class="user-name">@{_username}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("🚪 Logout", use_container_width=True, key="logout_btn"):
        logout()

st.markdown(
    """
    <div class="hero-container">
      <div class="hero-badge">✨ AI-POWERED WEALTH ANALYTICS</div>
      <h1 class="hero-title">Personal Finance Assistant</h1>
      <p class="hero-subtitle">Upload bank statements · Ask intelligent questions · Gain instant financial clarity</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_upload, tab_chat, tab_report, tab_transactions = st.tabs(
    ["📤 Upload Statement", "💬 Ask a Question", "📊 Monthly Report", "🗂 Transactions"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload Statement
# ═══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload your bank statement CSV")
    st.markdown(
        "<p class='sub-text'>Supported columns: "
        "<code>date</code>, <code>description</code>, <code>amount</code> "
        "(and common bank aliases). Amounts can be negative (expenses) or positive (income).</p>",
        unsafe_allow_html=True,
    )

    col_up, col_sample = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Drop your CSV here", type=["csv"], label_visibility="collapsed"
        )
    with col_sample:
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_transactions.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                st.download_button(
                    "⬇ Sample CSV",
                    f,
                    file_name="sample_transactions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    if not uploaded_file:
        st.markdown(
            """
            <div class="features-grid">
              <div class="feature-card">
                <div class="feature-icon-wrapper">📁</div>
                <div class="feature-title">1. CSV Ingestion</div>
                <p class="feature-desc">Drag & drop your raw bank export. Supports standard column headers and bank aliases automatically.</p>
              </div>
              <div class="feature-card">
                <div class="feature-icon-wrapper">🤖</div>
                <div class="feature-title">2. AI Categorisation</div>
                <p class="feature-desc">LangGraph pipeline classifies line items, detects spending anomalies, and logs data in SQLite.</p>
              </div>
              <div class="feature-card">
                <div class="feature-icon-wrapper">💬</div>
                <div class="feature-title">3. Intelligent Q&A</div>
                <p class="feature-desc">Ask natural-language questions about your spending trends with full data-backed tool responses.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file:
        # Preview with resilient parsing fallback
        try:
            preview_df = pd.read_csv(uploaded_file, on_bad_lines="skip")
            uploaded_file.seek(0)
            with st.expander("👀 Preview (first 5 rows)", expanded=False):
                st.dataframe(preview_df.head(), use_container_width=True)
        except Exception as exc:
            st.warning(f"⚠️ Note: CSV contains non-standard row formatting ({exc}). Processing will proceed using resilient row-by-row parsing.")
            uploaded_file.seek(0)

        if st.button("⚡ Process Statement", use_container_width=False):
            with st.spinner("Running pipeline: ingest → categorise → store → anomaly → report …"):
                tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                tmp.write(uploaded_file.read())
                tmp.close()
                try:
                    result = graph.invoke({"csv_path": tmp.name})
                finally:
                    os.unlink(tmp.name)

            summary = result.get("ingestion_summary", {})
            proc = summary.get("processed_count", len(result.get("categorized_transactions", [])))
            total = summary.get("total_rows", proc)
            skipped_count = summary.get("skipped_count", 0)

            if skipped_count == 0:
                st.success(f"✅ Pipeline complete! Successfully processed all {proc} row(s).")
            else:
                st.warning(f"⚠️ Processed {proc} of {total} row(s). {skipped_count} row(s) were skipped.")
                skipped_details = summary.get("skipped_details", [])
                if skipped_details:
                    with st.expander(f"📋 View Skipped Row Details ({skipped_count})", expanded=False):
                        skip_df = pd.DataFrame([
                            {"CSV Line #": str(s["row"]), "Skipped Reason": s["reason"], "Raw Row Content": str(s.get("raw", {}))}
                            for s in skipped_details
                        ])
                        st.dataframe(skip_df, use_container_width=True)

            report = result.get("report", {})
            if report:
                st.session_state["last_report"] = report

            st.markdown("#### 📈 Quick Summary")
            c1, c2, c3, c4 = st.columns(4)
            upload_txns = result.get("categorized_transactions", [])
            txns = len(upload_txns)
            anomalies = result.get("anomalies", []) or report.get("anomalies", [])
            individual_outliers = [a for a in anomalies if a.get("type") == "individual_outlier"]
            outlier_amounts = set(round(a["current_amount"], 2) for a in individual_outliers)

            def _is_outlier(t: dict) -> bool:
                return round(abs(float(t.get("amount", 0))), 2) in outlier_amounts

            upload_spent = sum(abs(t["amount"]) for t in upload_txns if t.get("category", "").upper() != "INCOME" and not _is_outlier(t))
            upload_income = sum(abs(t["amount"]) for t in upload_txns if t.get("category", "").upper() == "INCOME")
            upload_rate = round((upload_income - upload_spent) / upload_income * 100, 1) if upload_income else 0.0
            rate_class = "green" if upload_rate >= 20 else ("amber" if upload_rate >= 5 else "red")

            c1.markdown(f'<div class="kpi-card"><div class="kpi-header-row"><span class="kpi-title">Total Spent</span><span class="kpi-icon">💸</span></div><div class="kpi-val red">${upload_spent:,.2f}</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="kpi-card"><div class="kpi-header-row"><span class="kpi-title">Total Income</span><span class="kpi-icon">💵</span></div><div class="kpi-val green">${upload_income:,.2f}</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="kpi-card"><div class="kpi-header-row"><span class="kpi-title">Savings Rate</span><span class="kpi-icon">🎯</span></div><div class="kpi-val {rate_class}">{upload_rate:.1f}%</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="kpi-card"><div class="kpi-header-row"><span class="kpi-title">Transactions</span><span class="kpi-icon">🧾</span></div><div class="kpi-val">{txns}</div></div>', unsafe_allow_html=True)

            if anomalies:
                st.markdown("#### ⚠️ Anomalies Detected")
                for a in anomalies:
                    st.markdown(f'<div class="anomaly-{a.get("severity", "low")}">{a.get("message", "").replace("⚠️ ", "")}</div>', unsafe_allow_html=True)

            outlier_suggestions = []
            flagged_txns = [t for t in upload_txns if _is_outlier(t)]
            from collections import defaultdict
            grouped_outliers = defaultdict(lambda: {"count": 0, "total": 0.0})
            for t in flagged_txns:
                entity = t.get("description", "this transaction")
                amt = abs(t.get("amount", 0))
                grouped_outliers[entity]["count"] += 1
                grouped_outliers[entity]["total"] += amt
                
            for entity, data in grouped_outliers.items():
                count = data["count"]
                total = data["total"]
                if count > 1:
                    msg = f"Please review <strong>'{entity}'</strong> ({count} transactions totaling <strong>${total:,.2f}</strong>) carefully — these transactions were flagged as unusually large and excluded from your Total Spent."
                else:
                    msg = f"Please review <strong>'{entity}'</strong> (<strong>${total:,.2f}</strong>) carefully — this transaction was flagged as unusually large and excluded from your Total Spent."
                outlier_suggestions.append(msg)

            suggestions = report.get("suggestions", [])
            all_suggestions = outlier_suggestions + suggestions
            if all_suggestions:
                st.markdown("#### 💡 AI Suggestions")
                for s in all_suggestions:
                    import re
                    clean_s = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', s)
                    clean_s = clean_s.replace("💡 ", "").replace("⚠️ ", "").strip()
                    st.markdown(f'<div class="suggestion-card">💡 {clean_s}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Ask a Question
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    col_t, col_c = st.columns([4, 1])
    with col_t:
        st.markdown("### Ask anything about your spending")
        st.markdown(
            "<p class='sub-text'>The AI analyst remembers your conversation context — feel free to ask follow-up questions or clarify details.</p>",
            unsafe_allow_html=True,
        )
    with col_c:
        if st.button("🧹 Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["q_input"] = ""
            st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggested questions chips
    example_qs = [
        "Log $45 for Groceries today",
        "How much did I spend on Food last month?",
        "Why did I spend more this month? Break it down by category.",
        "Compare this month's spending to last month.",
        "Which week did I spend the most this month?",
    ]

    st.markdown("<p class='section-badge'>💡 Try these questions:</p>", unsafe_allow_html=True)
    q_col1, q_col2 = st.columns(2)

    chip_selected = None
    for i, eq in enumerate(example_qs):
        col = q_col1 if i % 2 == 0 else q_col2
        if col.button(eq, key=f"eq_{i}", use_container_width=True):
            chip_selected = eq

    # Render previous conversation thread
    if st.session_state["chat_history"]:
        st.markdown("---")
        for msg in st.session_state["chat_history"]:
            role = msg["role"]
            avatar = "👤" if role == "user" else "🤖"
            with st.chat_message(role, avatar=avatar):
                st.markdown(msg["content"])
        st.markdown("---")

    question = st.text_input(
        "Your question",
        placeholder="Type a message or answer (e.g. '2026-07 and 2026-06' or 'Yes')...",
        key="q_input",
        label_visibility="collapsed",
    )

    analyse_clicked = st.button("🔍 Send / Analyse", key="btn_analyse")

    user_query = chip_selected or (question if analyse_clicked else None)

    if user_query:
        # Append user message
        st.session_state["chat_history"].append({"role": "user", "content": user_query})

        with st.spinner("Thinking & analyzing conversation …"):
            res = qa_node({"messages": st.session_state["chat_history"]}, qa_agent)
            answer = res.get("answer", "No answer generated.")

        # Append assistant message
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Monthly Report
# ═══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("### Monthly Report")

    # Use last processed report or generate fresh
    report_data = st.session_state.get("last_report", {})

    col_gen, _ = st.columns([2, 5])
    if col_gen.button("🔄 Generate Fresh Report"):
        with st.spinner("Generating report from stored data …"):
            from nodes.anomaly import anomaly_node
            from nodes.report import report_node

            a_state = anomaly_node({}, conn, user_id=_user_id)
            r_state = report_node(a_state, conn, user_id=_user_id)
            report_data = r_state.get("report", {})
            st.session_state["last_report"] = report_data

    if report_data:
        _month_label = report_data.get("month", "")
        if not _month_label or _month_label == "unknown":
            # Resolve directly from DB for this user
            _row = conn.execute(
                "SELECT strftime('%Y-%m', MAX(date)) FROM transactions WHERE user_id=?",
                (_user_id,)
            ).fetchone()
            _month_label = _row[0] if _row and _row[0] else "No data yet"
        st.markdown(f"#### 📅 {_month_label}")


        # KPIs row
        c1, c2, c3 = st.columns(3)
        spent  = report_data.get("total_spent", 0)
        income = report_data.get("total_income", 0)
        rate   = report_data.get("savings_rate", 0)
        rate_class = "green" if rate >= 20 else ("amber" if rate >= 5 else "red")

        c1.markdown(f'<div class="kpi-card"><div class="kpi-header-row"><span class="kpi-title">Total Spent</span><span class="kpi-icon">💸</span></div><div class="kpi-val red">${spent:,.2f}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="kpi-card"><div class="kpi-header-row"><span class="kpi-title">Total Income</span><span class="kpi-icon">💵</span></div><div class="kpi-val green">${income:,.2f}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="kpi-card"><div class="kpi-header-row"><span class="kpi-title">Savings Rate</span><span class="kpi-icon">🎯</span></div><div class="kpi-val {rate_class}">{rate:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        col_l, col_r = st.columns(2)

        # Spending breakdown chart (from DB)
        with col_l:
            st.markdown("**Spending by Category**")
            try:
                df_chart = pd.read_sql(
                    "SELECT category, SUM(ABS(amount)) as total FROM transactions "
                    "WHERE user_id=? AND category!='Income' AND is_outlier=0 GROUP BY category ORDER BY total DESC",
                    conn, params=(_user_id,),
                )
                if not df_chart.empty:
                    fig = px.pie(
                        df_chart, names="category", values="total",
                        color_discrete_sequence=px.colors.sequential.Plasma_r,
                        hole=0.4,
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ffffff", size=13, family="Plus Jakarta Sans, sans-serif"),
                        showlegend=True,
                        legend=dict(
                            font=dict(color="#ffffff", size=13),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        margin=dict(t=10, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Upload a statement first to see the chart.")

        # Monthly trend chart
        with col_r:
            st.markdown("**Monthly Spending vs Income Trend**")
            try:
                df_trend = pd.read_sql(
                    "SELECT strftime('%Y-%m', date) as month, "
                    "SUM(CASE WHEN category != 'Income' THEN ABS(amount) ELSE 0 END) as spent, "
                    "SUM(CASE WHEN category = 'Income' THEN ABS(amount) ELSE 0 END) as income "
                    "FROM transactions WHERE user_id=? AND is_outlier=0 "
                    "GROUP BY month ORDER BY month",
                    conn, params=(_user_id,),
                )
                if not df_trend.empty:
                    fig2 = px.bar(
                        df_trend, x="month", y=["spent", "income"],
                        barmode="group",
                        color_discrete_map={"spent": "#8b5cf6", "income": "#10b981"},
                        labels={"month": "Month", "value": "Amount ($)", "variable": "Category"},
                    )
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ffffff", size=13, family="Plus Jakarta Sans, sans-serif"),
                        xaxis=dict(gridcolor="rgba(255,255,255,.12)", tickfont=dict(color="#ffffff")),
                        yaxis=dict(gridcolor="rgba(255,255,255,.12)", tickfont=dict(color="#ffffff")),
                        legend=dict(
                            title="", 
                            font=dict(color="#ffffff", size=12),
                            bgcolor="rgba(0,0,0,0)",
                        ),
                        margin=dict(t=10, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.info("Upload a statement first to see the chart.")

        # Anomalies
        anomalies = report_data.get("anomalies", [])
        if anomalies:
            st.markdown("**⚠️ Anomalies**")
            for a in anomalies:
                sev = a.get("severity", "low")
                st.markdown(
                    f'<div class="anomaly-{sev}">🚨 {a.get("message", "")}</div>',
                    unsafe_allow_html=True,
                )

        # Suggestions
        suggestions = report_data.get("suggestions", [])
        if suggestions:
            st.markdown("**💡 Suggestions**")
            for s in suggestions:
                st.markdown(f'<div class="suggestion-card">💡 {s}</div>', unsafe_allow_html=True)

        # Top categories
        top = report_data.get("top_categories", [])
        if top:
            st.markdown(f"**🏆 Top spending categories:** {', '.join(top)}")

    else:
        st.info("📂 Upload a statement or click 'Generate Fresh Report' to get started.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Browse Transactions
# ═══════════════════════════════════════════════════════════════════════════════
with tab_transactions:
    st.markdown("### All Transactions")

    try:
        df_all = pd.read_sql(
            "SELECT date, description, amount, category, source, ingested_at "
            "FROM transactions WHERE user_id=? ORDER BY date DESC",
            conn, params=(_user_id,),
        )
    except Exception:
        df_all = pd.DataFrame()

    if df_all.empty:
        st.info("📂 No transactions yet — upload a CSV statement to get started.")
    else:
        # Filters
        fc1, fc2, fc3 = st.columns(3)
        all_cats = ["All"] + sorted(df_all["category"].dropna().unique().tolist())
        sel_cat = fc1.selectbox("Category", all_cats, key="filter_cat")

        all_months = ["All"] + sorted(
            pd.to_datetime(df_all["date"]).dt.to_period("M").astype(str).unique().tolist(),
            reverse=True,
        )
        sel_month = fc2.selectbox("Month", all_months, key="filter_month")

        search = fc3.text_input("Search description", key="filter_search")

        df_filtered = df_all.copy()
        if sel_cat != "All":
            df_filtered = df_filtered[df_filtered["category"] == sel_cat]
        if sel_month != "All":
            df_filtered = df_filtered[
                pd.to_datetime(df_filtered["date"]).dt.to_period("M").astype(str) == sel_month
            ]
        if search:
            df_filtered = df_filtered[
                df_filtered["description"].str.contains(search, case=False, na=False)
            ]

        st.markdown(
            f"<p class='table-meta-text'>{len(df_filtered):,} transactions</p>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            df_filtered,
            use_container_width=True,
            height=400,
        )

        # Summary
        c_tot = st.columns(1)[0]
        with c_tot:
            total_abs = df_filtered["amount"].abs().sum() if not df_filtered.empty else 0
            st.markdown(
                f"<p class='table-meta-text'>Total (abs): <strong>${total_abs:,.2f}</strong></p>",
                unsafe_allow_html=True,
            )
