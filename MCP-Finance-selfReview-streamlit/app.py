"""
app.py  —  Streamlit UI for MCP Finance Research System
=========================================================
Run:
    Terminal 1: python news-server.py
    Terminal 2: streamlit run app.py
"""

import sys
import asyncio
import concurrent.futures
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import run_orchestrator
from agents.common import CONFIG

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="MCP Finance Research",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================

if "history" not in st.session_state:
    st.session_state.history = []          # list of {run_id, query, result, timestamp}
if "displayed_result" not in st.session_state:
    st.session_state.displayed_result = None
if "displayed_query" not in st.session_state:
    st.session_state.displayed_query = None
if "query_textarea" not in st.session_state:
    st.session_state.query_textarea = ""

# =============================================================================
# ASYNC → SYNC BRIDGE
# =============================================================================

def run_sync(query: str) -> str:
    """Execute the async orchestrator in a worker thread with its own event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, run_orchestrator(query))
        return future.result()

# =============================================================================
# EXAMPLE QUERIES (one per query type)
# =============================================================================

EXAMPLE_QUERIES = [
    (
        "TYPE A",
        "Market Headlines",
        "Give me Indian market headlines and IT sector news for today.",
    ),
    (
        "TYPE B",
        "Buy / Sell / Hold",
        "Should I buy, sell, or hold TCS.NS for a 6-month horizon? "
        "Fetch LIVE data from Yahoo Finance tools.",
    ),
    (
        "TYPE C",
        "Sector Selection",
        "Which sector should I invest in right now — IT, Banking, or Energy? "
        "Fetch LIVE data from Yahoo Finance tools.",
    ),
    (
        "TYPE D",
        "Deep Analysis",
        "Give me a full analysis report on Reliance Industries (RELIANCE.NS). "
        "Fetch LIVE data from Yahoo Finance tools.",
    ),
    (
        "TYPE E",
        "Macro Impact",
        "How do rising US interest rates affect Indian IT stocks? "
        "Fetch LIVE data and relevant news.",
    ),
]

# =============================================================================
# CALLBACKS
# =============================================================================

def set_query(text: str) -> None:
    """Pre-fill the query text area from a quick-select button."""
    st.session_state.query_textarea = text


def load_history_entry(entry: dict) -> None:
    """Restore a past query and its result from session history."""
    st.session_state.query_textarea   = entry["query"]
    st.session_state.displayed_result = entry["result"]
    st.session_state.displayed_query  = entry["query"]

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("📈 MCP Finance")
    st.caption("Multi-Agent Financial Research")

    st.divider()

    # System info pulled from agent_config.yaml
    st.subheader("⚙️ System Info")
    models   = CONFIG.get("models", {})
    provider = models.get("provider", "mistral").capitalize()
    st.markdown(f"**Provider:** `{provider}`")
    st.markdown(f"**Large model:** `{models.get('large', 'N/A')}`")
    st.markdown(f"**Small model:** `{models.get('small', 'N/A')}`")
    st.info(
        "The news server must be running before you submit a query:\n"
        "```\npython news-server.py\n```",
        icon="🔌",
    )

    st.divider()

    # Session history
    st.subheader("🕒 Session History")
    if not st.session_state.history:
        st.caption("No queries yet in this session.")
    else:
        for entry in reversed(st.session_state.history):
            label = (
                entry["query"][:42] + "…"
                if len(entry["query"]) > 42
                else entry["query"]
            )
            st.button(
                f"[{entry['timestamp']}] {label}",
                key=f"hist_{entry['run_id']}",
                on_click=load_history_entry,
                args=(entry,),
                use_container_width=True,
            )

# =============================================================================
# MAIN AREA
# =============================================================================

st.title("📊 MCP Finance Research")
st.markdown(
    f"Multi-agent financial analysis powered by **{provider} · LangGraph · MCP**"
)

st.divider()

# ── Quick-select TYPE buttons ─────────────────────────────────────────────────
st.subheader("Quick Select Query Type")
cols = st.columns(5)
for col, (type_id, desc, query_text) in zip(cols, EXAMPLE_QUERIES):
    with col:
        st.button(
            f"{type_id} — {desc}",
            key=f"btn_{type_id}",
            on_click=set_query,
            args=(query_text,),
            use_container_width=True,
            help=query_text,
        )

st.divider()

# ── Query input ───────────────────────────────────────────────────────────────
st.subheader("Your Query")
st.text_area(
    label="Enter your financial research query:",
    height=120,
    placeholder="e.g. Should I buy Infosys (INFY.NS) for a 1-year horizon?",
    key="query_textarea",
)

run_col, _, clear_col = st.columns([2, 7, 1])
with run_col:
    run_clicked = st.button(
        "🚀 Run Analysis", type="primary", use_container_width=True
    )
with clear_col:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.query_textarea   = ""
        st.session_state.displayed_result = None
        st.session_state.displayed_query  = None
        st.rerun()

# ── Execute analysis ──────────────────────────────────────────────────────────
if run_clicked:
    query = st.session_state.query_textarea
    if not query.strip():
        st.warning("Please enter a query before running.")
    else:
        with st.spinner("🔍 Running multi-agent analysis… this may take 30–90 seconds."):
            try:
                result    = run_sync(query)
                timestamp = datetime.now().strftime("%H:%M:%S")
                run_id    = datetime.now().strftime("%Y%m%d%H%M%S%f")
                entry = {
                    "run_id":    run_id,
                    "query":     query,
                    "result":    result,
                    "timestamp": timestamp,
                }
                st.session_state.history.append(entry)
                st.session_state.displayed_result = result
                st.session_state.displayed_query  = query
            except Exception as exc:
                print(f" Error running analysis: {exc}")
                st.error(f"❌ Error running analysis: Please retry after sometime")

# ── Display result ────────────────────────────────────────────────────────────
def clean_result(text: str) -> str:
    """Strip template markers and escape $ to prevent Streamlit LaTeX rendering."""
    lines = text.splitlines()
    cleaned = [
        line for line in lines
        if line.strip() not in ("--- TEMPLATE START ---", "--- TEMPLATE END ---")
    ]
    result = "\n".join(cleaned).strip()
    result = result.replace("$", r"\$")
    return result


if st.session_state.displayed_result:
    st.divider()

    if st.session_state.displayed_query:
        st.markdown(f"**Query:** _{st.session_state.displayed_query}_")

    result_text = clean_result(st.session_state.displayed_result)

    if "DATA QUALITY WARNING" in result_text:
        st.error(
            "⚠️ Data Quality Warning — automated review detected issues. "
            "Verify key figures independently before making investment decisions."
        )

    st.markdown(result_text)
