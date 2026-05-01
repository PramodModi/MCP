# MCP Finance — Multi-Agent Financial Research System

A financial research system with a **Streamlit UI**, built on the **Model Context Protocol (MCP)**, multi-agent orchestration via **LangGraph**, and a built-in **self-review quality gate** that validates every report before it reaches the user.

Supports multiple LLM providers: **Mistral**, **Gemini**, **DeepSeek**, and **Groq** — switchable from a single config file.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [LLM Provider Configuration](#llm-provider-configuration)
- [Configuration Files](#configuration-files)
- [Project Structure](#project-structure)
- [Self-Review System](#self-review-system)
- [Query Types](#query-types)
- [Tool Tracing](#tool-tracing)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

**MCP Finance** provides comprehensive financial research for Indian (NSE/BSE) and global equity markets through a Streamlit web UI backed by a multi-agent pipeline.

| Capability | Detail |
|-----------|--------|
| Live market data | Yahoo Finance via MCP |
| Real-time news | Economic Times, Moneycontrol, Reuters, TechCrunch, VentureBeat, arXiv, Hacker News |
| Self-review quality gate | Every report reviewed and retried before the user sees it |
| Multi-LLM | Mistral, Gemini, DeepSeek, Groq — one line to switch |
| Streamlit UI | Query input, quick-select buttons, session history |

---

## 🏗️ Architecture

```
Streamlit UI (app.py)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                  ORCHESTRATOR                          │
│             (large LLM, LangGraph)                    │
│  • Classifies query → TYPE A / B / C / D / E          │
│  • Delegates to specialist agents                      │
│  • Synthesises final answer                            │
└──────────────┬──────────────────────────┬─────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│       NEWS AGENT         │  │       ANALYSIS AGENT          │
│       (small LLM)        │  │         (large LLM)           │
│                          │  │                               │
│  Tools (8):              │  │  Tools (20+):                 │
│  get_india_business_news │  │  get_current_stock_price      │
│  get_global_business_news│  │  get_historical_stock_prices  │
│  get_tech_news           │  │  get_income_statement         │
│  get_ai_news             │  │  get_cashflow                 │
│  get_ai_research_papers  │  │  get_balance_sheet            │
│  get_sector_news         │  │  get_recommendations          │
│  get_hacker_news_top     │  │  search_news / fetch          │
│  search_news             │  │                               │
│         │                │  │         │                     │
│         ▼                │  │         ▼                     │
│   NEWS REVIEW AGENT      │  │   ANALYSIS REVIEW AGENT       │
│  Checks: freshness,      │  │  Checks: live citation,       │
│  sources, format,        │  │  tool coverage, date,         │
│  min articles            │  │  recommendation, consistency  │
│  → retry if failed       │  │  → retry if failed            │
└──────────────────────────┘  └──────────────────────────────┘
```

---

## ✅ Prerequisites

- **Python 3.11+**
- An API key for one of: Mistral, Gemini, DeepSeek, or Groq

---

## 🚀 Installation

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd MCP-Finance-selfReview-streamlit
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Copy the example file and fill in your key:

```bash
cp agents/.env.example agents/.env
```

Edit `agents/.env`:

```env
# Choose whichever provider you set in agent_config.yaml
MISTRAL_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here
# DEEPSEEK_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here
```

---

## ⚡ Quick Start

**Terminal 1 — start the news server:**
```bash
python news-server.py
```
Expected: `Endpoint : http://127.0.0.1:8001/mcp`

**Terminal 2 — launch the Streamlit app:**
```bash
streamlit run app.py
```
Open the URL shown (usually `http://localhost:8501`).

---

## 🤖 LLM Provider Configuration

All provider and model settings live in `agent_config.yaml`. Only one block is active at a time.

```yaml
# Mistral (default)
models:
  provider: "mistral"
  large: "mistral-large-latest"
  small: "mistral-small-latest"

# Groq — free tier, very fast
# models:
#   provider: "groq"
#   large: "llama-3.3-70b-versatile"
#   small: "llama-3.1-8b-instant"

# Gemini — free tier available
# models:
#   provider: "gemini"
#   large: "gemini-2.0-flash"
#   small: "gemini-2.0-flash"

# DeepSeek
# models:
#   provider: "deepseek"
#   large: "deepseek-chat"
#   small: "deepseek-chat"
```

Set the matching API key in `agents/.env` and restart the app.

---

## ⚙️ Configuration Files

### `agent_config.yaml` — main runtime config

| Section | Purpose |
|---------|---------|
| `models` | Provider, large/small model names, temperature |
| `mcp_servers` | Transport type, URL/command for each MCP server |
| `agent_servers` | Which MCP servers each agent connects to |
| `agents` | Recursion limits and tool call caps per agent |

### `news_feeds.yaml` — RSS feed registry

Controls which RSS sources `news-server.py` pulls from. Edit feed entries to add, remove, or disable sources — no Python changes needed.

```yaml
feeds:
  economic_times:
    url:      "https://economictimes.indiatimes.com/rssfeedsdefault.cms"
    category: business
    region:   India
    label:    "Economic Times"
    # enabled: false   ← add this line to temporarily disable
```

Add a new feed by copying any existing entry and changing the key and URL.

---

## 📁 Project Structure

```
MCP-Finance-selfReview-streamlit/
├── app.py                  # Streamlit UI
├── orchestrator.py         # Multi-agent orchestrator (LangGraph)
├── news-server.py          # News MCP server (HTTP, port 8001)
├── news_feeds.yaml         # RSS feed registry for news-server.py
├── agent_config.yaml       # Single config for all agents and models
├── requirements.txt        # Python dependencies
│
└── agents/
    ├── __init__.py         # Package exports
    ├── .env                # Your API keys (never commit)
    ├── .env.example        # API key template
    ├── common.py           # LLM factory, MCP config builder, ReAct builder
    ├── tracer.py           # Tool call tracer (run_id / trace_id hierarchy)
    ├── news_agent.py       # News specialist agent + review loop
    ├── analysis_agent.py   # Financial analysis agent + review loop
    └── review_agent.py     # Review agent (analysis checks + news checks)
```

---

## 🛡️ Self-Review System

Both specialist agents run an internal review-and-retry loop before returning results to the orchestrator.

### Analysis Review (7 checks)

| Check | What it verifies |
|-------|-----------------|
| `price_has_live_citation` | Current price attributed to Yahoo Finance with a recent date |
| `minimum_tool_calls_met` | At least 4 distinct tool types used (price, income, cashflow, news) |
| `date_is_current` | "Data as of" date matches the live price fetch date |
| `no_trailing_question` | Report ends with disclaimer, not "Would you like…" |
| `metrics_have_citations` | Key metrics (P/E, revenue, FCF, ROE) have source attribution |
| `recommendation_present` | Clear BUY / SELL / HOLD verdict exists |
| `price_consistent` | Same price used consistently across all sections |

### News Review (5 checks)

| Check | What it verifies |
|-------|-----------------|
| `news_is_recent` | Articles ≤7 days old for "today" queries; older allowed for historical queries |
| `sources_cited` | Every headline has `[Source, Date]` attribution |
| `format_complete` | All four sections present: MARKET HEADLINES, SECTOR PULSE, TECH & AI CONTEXT, ANALYST BRIEF |
| `no_trailing_question` | Report ends with disclaimer |
| `minimum_articles` | At least 3 distinct articles under MARKET HEADLINES |

If a review fails, the agent retries once with the specific failure reasons appended to the task. The Streamlit UI shows a `⚠️ DATA QUALITY WARNING` banner only if both attempts fail.

---

## 🗂️ Query Types

| Type | Use case | Example |
|------|---------|---------|
| **A** | Headlines, market briefs, sector news | "Give me Indian market headlines and IT sector news for today" |
| **B** | Buy / Sell / Hold for a specific stock | "Should I buy TCS.NS for a 6-month horizon?" |
| **C** | Sector selection — which sector to invest in | "Which sector should I invest in — IT, Banking, or Energy?" |
| **D** | Full deep-dive on one company | "Give me a full analysis of Reliance Industries (RELIANCE.NS)" |
| **E** | Macro impact on stocks | "How do rising US interest rates affect Indian IT stocks?" |

---

## 📊 Tool Tracing

Every tool call is logged with a two-level ID hierarchy:

- **run_id** — shared across all agents in one orchestrator run
- **trace_id** — unique per agent invocation

Example console output:
```
[7538fa56][20fe4964][analysis_agent][#1]  get_current_stock_price({'symbol': 'TCS.NS'})
[7538fa56][20fe4964][analysis_agent][#2]  get_historical_stock_prices({'symbol': 'TCS.NS', 'period': '2y'})
[7538fa56][20fe4964][analysis_agent][#3]  get_income_statement({'symbol': 'TCS.NS', 'freq': 'yearly'})
```

A summary table with timing is printed at the end of each agent run.

---

## 🔧 Troubleshooting

### News server connection refused

```bash
# Make sure it is running in a separate terminal
python news-server.py
# Expected: Endpoint : http://127.0.0.1:8001/mcp
```

### Port 8001 already in use

```bash
lsof -ti:8001 | xargs kill -9
```

### `MISTRAL_API_KEY not found` (or other provider)

Check that `agents/.env` exists and contains the correct variable name for your chosen provider.

### `Error 429: Rate limit exceeded`

- Mistral free tier: wait ~1 minute and retry
- Switch to Groq (free, high rate limits) in `agent_config.yaml`

### `mcp-yahoo-finance` or `mcp-server-fetch` not found

```bash
pip install mcp-yahoo-finance mcp-server-fetch
```

### Agent shows outdated stock prices

Include the phrase **"Fetch LIVE data from Yahoo Finance"** in your query to force tool usage instead of training-data recall.

### `FileNotFoundError: agent_config.yaml`

Always run commands from the project root directory:
```bash
cd MCP-Finance-selfReview-streamlit
streamlit run app.py
```

---

## 📚 Resources

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP](https://github.com/jlowin/fastmcp)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [LangChain MCP Adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Mistral AI Docs](https://docs.mistral.ai/)
- [Groq Console](https://console.groq.com/) — free tier, fast inference

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. Financial data and AI-generated analysis are not financial advice. Market data may be delayed or inaccurate. Consult a SEBI-registered investment advisor before making investment decisions.
