"""
ui/dashboard.py — Financial Dashboard (Streamlit)
==================================================
Interactive dashboard for visualising Gmail-sourced financial expenditure.

Run:
  streamlit run ui/dashboard.py

Tabs:
  Overview     — KPIs, daily trend, category donut, MoM, bank bar
  Merchants    — Configurable top-N bar + full merchant summary table
  Categories   — Category summary table + per-category merchant drilldown
  Transactions — Full filterable transaction table (search, category, type, amount)
"""

import asyncio
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.metrics import (
    PERIOD_LABELS,
    category_summary,
    compute_kpis,
    daily_spend,
    get_period_dates,
    merchant_summary,
    monthly_comparison,
    spend_by_bank,
    spend_by_category,
    top_merchants,
    top_merchants_by_category,
)
from data.storage import (
    get_all_banks,
    get_all_categories,
    get_last_sync_time,
    get_transactions,
)

# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title = "Financial Dashboard",
    page_icon  = "💰",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# =============================================================================
# Custom CSS  (dark-theme metric cards)
# =============================================================================

st.markdown(
    """
    <style>
    .kpi-card {
        background    : #1e1e2e;
        border        : 1px solid #313244;
        border-radius : 14px;
        padding       : 1.2rem 1.4rem;
        text-align    : center;
    }
    .kpi-value  { font-size: 1.9rem; font-weight: 700; color: #cba6f7; }
    .kpi-label  { font-size: 0.82rem; color: #a6adc8; margin-top: 0.35rem; }
    .kpi-sub    { font-size: 0.72rem; color: #6c7086; margin-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# Chart theme helper
# =============================================================================

_PLOT_LAYOUT = dict(
    plot_bgcolor  = "rgba(0,0,0,0)",
    paper_bgcolor = "rgba(0,0,0,0)",
    font_color    = "#cdd6f4",
    margin        = dict(l=0, r=0, t=10, b=0),
)

_GRID_COLOR = "#313244"


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("## 💰 Financial Dashboard")
    st.divider()

    period = st.selectbox(
        "📅 Time Period",
        options      = list(PERIOD_LABELS.keys()),
        format_func  = lambda k: PERIOD_LABELS[k],
        index        = 0,
        key          = "period",
    )

    custom_start = custom_end = None
    if period == "custom":
        col_a, col_b = st.columns(2)
        with col_a:
            custom_start = st.date_input("From", value=date.today().replace(day=1))
        with col_b:
            custom_end = st.date_input("To", value=date.today())

    start_date, end_date = get_period_dates(period, custom_start, custom_end)
    st.caption(f"📆 {start_date.strftime('%d %b %Y')}  →  {end_date.strftime('%d %b %Y')}")

    st.divider()

    all_cats  = get_all_categories()
    all_banks = get_all_banks()

    selected_categories = st.multiselect(
        "🏷️ Categories",
        options = all_cats,
        default = all_cats,
        key     = "cats",
    )

    selected_banks = st.multiselect(
        "🏦 Banks / Apps",
        options = all_banks,
        default = all_banks,
        key     = "banks",
    )

    st.divider()

    last_sync = get_last_sync_time()
    if last_sync:
        st.caption(f"🕒 Last synced: {last_sync.strftime('%d %b %Y, %H:%M')} UTC")
    else:
        st.caption("🕒 Never synced")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        do_sync    = st.button("🔄 Sync",         use_container_width=True, type="primary")
    with btn_col2:
        do_refresh = st.button("🗑️ Full Refresh", use_container_width=True, type="secondary")


# =============================================================================
# Sync / Refresh Handler
# =============================================================================

def _run_sync(full_refresh: bool, progress_slot) -> dict:
    """Run the async sync in a fresh event loop (safe from Streamlit's thread)."""
    from agents.email_sync import sync_emails

    messages: list[str] = []

    def _cb(msg: str) -> None:
        messages.append(msg)
        progress_slot.info("\n\n".join(messages[-4:]))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(
            sync_emails(full_refresh=full_refresh, progress_callback=_cb)
        )
    finally:
        loop.close()


if do_sync or do_refresh:
    progress_slot = st.empty()
    with st.spinner("Connecting to Gmail MCP server…"):
        try:
            stats = _run_sync(full_refresh=do_refresh, progress_slot=progress_slot)
            progress_slot.empty()
            st.success(
                f"✅ Sync complete — "
                f"Fetched: **{stats['fetched']}** | "
                f"Saved: **{stats['saved']}** | "
                f"Skipped: **{stats['skipped']}** | "
                f"Errors: **{stats['errors']}**"
            )
            st.rerun()
        except Exception as exc:
            progress_slot.empty()
            st.error(f"❌ Sync failed: {exc}")


# =============================================================================
# Load & Filter Data
# =============================================================================

raw_transactions = get_transactions(
    start_date = str(start_date),
    end_date   = str(end_date),
    categories = selected_categories if selected_categories else None,
    banks      = selected_banks      if selected_banks      else None,
)

df = pd.DataFrame(raw_transactions) if raw_transactions else pd.DataFrame()


# =============================================================================
# Header
# =============================================================================

st.markdown("# 💰 Financial Dashboard")
st.caption(
    f"Period: **{PERIOD_LABELS[period]}**  ·  "
    f"{start_date.strftime('%d %b %Y')} → {end_date.strftime('%d %b %Y')}"
)

if df.empty:
    st.info(
        "No transactions found for this period. "
        "Use **🔄 Sync** in the sidebar to fetch data from Gmail, "
        "or adjust the period / filter settings."
    )
    st.stop()


# =============================================================================
# KPI Cards  (always visible above tabs)
# =============================================================================

kpis = compute_kpis(df)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">₹{kpis["total"]:,.0f}</div>'
        f'<div class="kpi-label">Total Spent</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{kpis["count"]}</div>'
        f'<div class="kpi-label">Transactions</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">₹{kpis["avg"]:,.0f}</div>'
        f'<div class="kpi-label">Avg per Transaction</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-value">₹{kpis["max"]:,.0f}</div>'
        f'<div class="kpi-label">Largest Transaction</div>'
        f'<div class="kpi-sub">{kpis["max_merchant"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# =============================================================================
# Tabs
# =============================================================================

tab_overview, tab_merchants, tab_categories, tab_txns = st.tabs(
    ["📈 Overview", "🏪 Merchants", "🏷️ Categories", "📋 Transactions"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

with tab_overview:

    row1_left, row1_right = st.columns([3, 2], gap="large")

    with row1_left:
        st.subheader("📈 Daily Spend Trend")
        daily_df = daily_spend(df)
        if not daily_df.empty:
            fig = px.area(
                daily_df, x="date", y="amount",
                labels                  = {"amount": "Amount (₹)", "date": ""},
                color_discrete_sequence = ["#cba6f7"],
            )
            fig.update_traces(fillcolor="rgba(203,166,247,0.15)", line_width=2)
            fig.update_layout(
                **_PLOT_LAYOUT,
                xaxis = dict(showgrid=False, tickfont_size=11),
                yaxis = dict(gridcolor=_GRID_COLOR, tickfont_size=11, tickprefix="₹"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with row1_right:
        st.subheader("🏷️ By Category")
        cat_df = spend_by_category(df)
        if not cat_df.empty:
            fig = px.pie(
                cat_df, values="amount", names="category",
                hole                    = 0.48,
                color_discrete_sequence = px.colors.qualitative.Pastel,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(**_PLOT_LAYOUT, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    row2_left, row2_right = st.columns([2, 3], gap="large")

    with row2_left:
        st.subheader("🏪 Top 10 Merchants")
        merch_df = top_merchants(df, n=10)
        if not merch_df.empty:
            fig = px.bar(
                merch_df, x="amount", y="merchant", orientation="h",
                labels                  = {"amount": "Amount (₹)", "merchant": ""},
                color_discrete_sequence = ["#89b4fa"],
                text_auto               = ".2s",
            )
            fig.update_layout(
                **_PLOT_LAYOUT,
                yaxis = dict(autorange="reversed", showgrid=False, tickfont_size=11),
                xaxis = dict(gridcolor=_GRID_COLOR, tickprefix="₹"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with row2_right:
        st.subheader("📊 Month-over-Month")
        monthly_df = monthly_comparison(df)
        if not monthly_df.empty:
            fig = px.bar(
                monthly_df, x="month", y="amount",
                labels                  = {"amount": "Amount (₹)", "month": ""},
                color_discrete_sequence = ["#a6e3a1"],
                text_auto               = ".2s",
            )
            fig.update_layout(
                **_PLOT_LAYOUT,
                xaxis = dict(showgrid=False, tickfont_size=11),
                yaxis = dict(gridcolor=_GRID_COLOR, tickprefix="₹"),
            )
            st.plotly_chart(fig, use_container_width=True)

    bank_df = spend_by_bank(df)
    if len(bank_df) > 1:
        st.subheader("🏦 Spend by Bank / App")
        fig = px.bar(
            bank_df, x="bank_or_source", y="amount",
            labels                  = {"amount": "Amount (₹)", "bank_or_source": ""},
            color_discrete_sequence = ["#fab387"],
            text_auto               = ".2s",
        )
        fig.update_layout(
            **_PLOT_LAYOUT,
            xaxis = dict(showgrid=False, tickfont_size=11),
            yaxis = dict(gridcolor=_GRID_COLOR, tickprefix="₹"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Merchant Analysis
# ---------------------------------------------------------------------------

with tab_merchants:

    top_n = st.slider("Top N merchants to show in chart", min_value=5, max_value=50, value=15, step=5)

    merch_sum = merchant_summary(df)

    st.subheader(f"🏪 Top {top_n} Merchants by Spend")
    chart_df = merch_sum.head(top_n).rename(columns={"total": "amount"})
    if not chart_df.empty:
        fig = px.bar(
            chart_df, x="amount", y="merchant", orientation="h",
            labels                  = {"amount": "Total Spend (₹)", "merchant": ""},
            color                   = "amount",
            color_continuous_scale  = "Blues",
            text_auto               = ".2s",
        )
        fig.update_layout(
            **_PLOT_LAYOUT,
            yaxis              = dict(autorange="reversed", showgrid=False, tickfont_size=11),
            xaxis              = dict(gridcolor=_GRID_COLOR, tickprefix="₹"),
            coloraxis_showscale = False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Full Merchant Summary")
    disp = merch_sum.copy()
    disp.index = range(1, len(disp) + 1)
    disp.index.name = "#"
    disp.columns = ["Merchant", "Total (₹)", "Transactions", "Avg (₹)", "% of Spend"]
    disp["Total (₹)"] = disp["Total (₹)"].apply(lambda x: f"₹{x:,.2f}")
    disp["Avg (₹)"]   = disp["Avg (₹)"].apply(lambda x: f"₹{x:,.2f}")
    disp["% of Spend"] = disp["% of Spend"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(disp, use_container_width=True)
    st.caption(f"{len(merch_sum)} unique merchants")


# ---------------------------------------------------------------------------
# Tab 3 — Category Analysis
# ---------------------------------------------------------------------------

with tab_categories:

    cat_sum = category_summary(df)

    left_col, right_col = st.columns([2, 3], gap="large")

    with left_col:
        st.subheader("🏷️ Category Summary")
        disp = cat_sum.copy()
        disp.index = range(1, len(disp) + 1)
        disp.index.name = "#"
        disp.columns = ["Category", "Total (₹)", "Transactions", "Avg (₹)", "% of Spend"]
        disp["Total (₹)"] = disp["Total (₹)"].apply(lambda x: f"₹{x:,.2f}")
        disp["Avg (₹)"]   = disp["Avg (₹)"].apply(lambda x: f"₹{x:,.2f}")
        disp["% of Spend"] = disp["% of Spend"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(disp, use_container_width=True, hide_index=False)

    with right_col:
        st.subheader("�️ Spend by Category")
        if not cat_sum.empty:
            fig = px.bar(
                cat_sum, x="total", y="category", orientation="h",
                labels                  = {"total": "Total Spend (₹)", "category": ""},
                color                   = "total",
                color_continuous_scale  = "Purples",
                text_auto               = ".2s",
            )
            fig.update_layout(
                **_PLOT_LAYOUT,
                yaxis              = dict(autorange="reversed", showgrid=False, tickfont_size=11),
                xaxis              = dict(gridcolor=_GRID_COLOR, tickprefix="₹"),
                coloraxis_showscale = False,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Top Merchants per Category")
    cats_available = sorted(df["category"].dropna().unique().tolist())
    selected_cat   = st.selectbox("Select category to drill down", ["— All —"] + cats_available)

    if selected_cat == "— All —":
        drill_df = top_merchants_by_category(df, n=5)
        for cat, grp in drill_df.groupby("category"):
            with st.expander(f"**{cat}**  ·  {len(grp)} merchants shown"):
                g = grp[["merchant", "total", "count"]].copy()
                g.columns = ["Merchant", "Total (₹)", "Transactions"]
                g["Total (₹)"] = g["Total (₹)"].apply(lambda x: f"₹{x:,.2f}")
                st.dataframe(g, use_container_width=True, hide_index=True)
    else:
        filtered = df[df["category"] == selected_cat]
        cat_merch = merchant_summary(filtered)
        if not cat_merch.empty:
            disp2 = cat_merch[["merchant", "total", "count", "avg", "pct"]].copy()
            disp2.columns = ["Merchant", "Total (₹)", "Transactions", "Avg (₹)", "% in Category"]
            disp2["Total (₹)"] = disp2["Total (₹)"].apply(lambda x: f"₹{x:,.2f}")
            disp2["Avg (₹)"]   = disp2["Avg (₹)"].apply(lambda x: f"₹{x:,.2f}")
            disp2["% in Category"] = disp2["% in Category"].apply(lambda x: f"{x:.1f}%")
            disp2.index = range(1, len(disp2) + 1)
            st.dataframe(disp2, use_container_width=True)

            fig = px.bar(
                cat_merch.head(15), x="total", y="merchant", orientation="h",
                labels                  = {"total": "Total Spend (₹)", "merchant": ""},
                color_discrete_sequence = ["#89dceb"],
                text_auto               = ".2s",
            )
            fig.update_layout(
                **_PLOT_LAYOUT,
                yaxis = dict(autorange="reversed", showgrid=False, tickfont_size=11),
                xaxis = dict(gridcolor=_GRID_COLOR, tickprefix="₹"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4 — Transactions  (with inline filters)
# ---------------------------------------------------------------------------

with tab_txns:

    st.subheader("📋 All Transactions")

    # ---- Inline filter row ----
    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])

    with f1:
        search_text = st.text_input("🔍 Search merchant", placeholder="e.g. Swiggy, Amazon…")
    with f2:
        all_cats_txn   = sorted(df["category"].dropna().unique().tolist())
        filter_cats    = st.multiselect("Category", all_cats_txn, placeholder="All")
    with f3:
        all_types      = sorted(df["transaction_type"].dropna().unique().tolist())
        filter_types   = st.multiselect("Type", all_types, placeholder="All")
    with f4:
        min_amt = float(df["amount"].min())
        max_amt = float(df["amount"].max())
        if max_amt > min_amt:
            amt_range = st.slider(
                "Amount (₹)", min_value=min_amt, max_value=max_amt,
                value=(min_amt, max_amt), step=1.0,
            )
        else:
            amt_range = (min_amt, max_amt)

    # ---- Apply filters ----
    filtered_df = df.copy()
    if search_text:
        filtered_df = filtered_df[
            filtered_df["merchant"].str.contains(search_text, case=False, na=False)
        ]
    if filter_cats:
        filtered_df = filtered_df[filtered_df["category"].isin(filter_cats)]
    if filter_types:
        filtered_df = filtered_df[filtered_df["transaction_type"].isin(filter_types)]
    filtered_df = filtered_df[
        (filtered_df["amount"] >= amt_range[0]) & (filtered_df["amount"] <= amt_range[1])
    ]

    # ---- Display ----
    display_cols = ["date", "merchant", "category", "amount", "bank_or_source", "transaction_type"]
    display_df   = filtered_df[display_cols].copy()
    display_df.columns = ["Date", "Merchant", "Category", "Amount (₹)", "Bank / App", "Type"]
    display_df["Amount (₹)"] = display_df["Amount (₹)"].apply(lambda x: f"₹{x:,.2f}")
    display_df = display_df.sort_values("Date", ascending=False)

    st.dataframe(
        display_df,
        use_container_width = True,
        hide_index          = True,
        column_config = {
            "Date":       st.column_config.TextColumn("Date",       width="small"),
            "Merchant":   st.column_config.TextColumn("Merchant"),
            "Category":   st.column_config.TextColumn("Category",   width="small"),
            "Amount (₹)": st.column_config.TextColumn("Amount (₹)", width="small"),
            "Bank / App": st.column_config.TextColumn("Bank / App"),
            "Type":       st.column_config.TextColumn("Type",       width="small"),
        },
    )
    st.caption(
        f"Showing **{len(display_df)}** of {len(df)} transactions · amounts in INR"
    )
