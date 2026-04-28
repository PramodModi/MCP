# MCP-Finance: Multi-Agent Financial Research System

A production-grade financial research system using the **Model Context Protocol (MCP)** with a multi-agent architecture powered by Mistral AI and LangGraph.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Running the Orchestrator](#running-the-orchestrator)
  - [Running Standalone Agents](#running-standalone-agents)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Tool Tracing](#tool-tracing)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

---

## 🎯 Overview

**MCP-Finance** is a multi-agent system that provides comprehensive financial research and analysis for Indian (NSE/BSE) and global equity markets. It demonstrates:

- **Multi-agent orchestration** using the supervisor pattern
- **Live market data** via Yahoo Finance MCP server
- **Real-time news** from multiple RSS feeds (Economic Times, Moneycontrol, Reuters, TechCrunch, etc.)
- **Tool call tracing** with hierarchical run IDs for debugging and observability
- **Configurable workflows** via YAML configuration
- **Modular architecture** with reusable components

### Key Features

✅ **Three Specialist Agents**:
- **Orchestrator**: Routes queries and synthesizes final answers
- **News Agent**: Fetches and summarizes market/tech/AI news
- **Analysis Agent**: Deep financial analysis with live price data

✅ **Multiple MCP Servers**:
- Enhanced News Server (HTTP, port 8001) - 8 news tools
- Yahoo Finance (stdio) - Live stock data
- Fetch Server (stdio) - Web content retrieval

✅ **Production-Ready Features**:
- Comprehensive error handling
- Tool call tracing with run_id/trace_id hierarchy
- Configurable recursion limits and model selection
- Real-time tool call logging

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR                                 │
│                    (mistral-large-latest)                            │
│                                                                       │
│  • Query classification (TYPE A/B/C/D/E)                             │
│  • Task delegation via tool calls                                    │
│  • Final synthesis and formatting                                    │
└────────────┬────────────────────────────────┬────────────────────────┘
             │                                │
             ▼                                ▼
┌────────────────────────┐      ┌────────────────────────────────────┐
│     NEWS AGENT         │      │      ANALYSIS AGENT                │
│  (mistral-small)       │      │   (mistral-small/large)            │
│                        │      │                                    │
│  Connected to:         │      │  Connected to:                     │
│  • enhanced-news-server│      │  • yahoo_finance (stdio)           │
│    (HTTP, port 8001)   │      │  • fetch_server (stdio)            │
│                        │      │  • enhanced-news-server (HTTP)     │
│                        │      │                                    │
│  Tools (8):            │      │  Tools (20+):                      │
│  • get_india_business  │      │  • get_current_stock_price         │
│  • get_global_business │      │  • get_historical_prices           │
│  • get_tech_news       │      │  • get_income_statement            │
│  • get_ai_news         │      │  • get_balance_sheet               │
│  • get_ai_research     │      │  • get_cash_flow                   │
│  • get_sector_news     │      │  • get_key_statistics              │
│  • get_hacker_news_top │      │  • search_news                     │
│  • search_news         │      │  • fetch (web content)             │
└────────────────────────┘      └────────────────────────────────────┘
```

### Agent Responsibilities

| Agent | Purpose | Model | MCP Servers |
|-------|---------|-------|-------------|
| **Orchestrator** | Query routing, delegation, synthesis | mistral-large | None (delegates to specialists) |
| **News Agent** | Market briefs, headlines, sector news | mistral-small | enhanced-news-server |
| **Analysis Agent** | Buy/Sell/Hold decisions, fundamentals | mistral-small/large | yahoo_finance, fetch_server, enhanced-news-server |

---

## ✅ Prerequisites

- **Python 3.11+** (required for type hints like `str | None`)
- **pip** package manager
- **Mistral API Key** - Get it from [Mistral AI Console](https://console.mistral.ai/)
- **Git** (to clone the repository)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd MCP-Finance
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies installed:**
- `fastmcp` - MCP server framework
- `python-dotenv` - Environment variable management
- `langchain-mcp-adapters` - MCP client for LangChain
- `langgraph` - Agent workflow orchestration
- `langchain-mistralai` - Mistral AI integration
- `mcp-server-fetch` - Web content fetching
- `pyyaml` - YAML configuration parsing
- `mcp-yahoo-finance` - Live stock data

### 4. Set Up Environment Variables

Create a `.env` file in the `agents/` directory:

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
```

**Important**: 
- Never commit your `.env` file to version control
- The `.env` file should be in `/MCP-Finance/agents/.env`
- Get your API key from https://console.mistral.ai/

---

## ⚡ Quick Start

### Terminal 1: Start the News Server

```bash
python news-server.py
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

### Terminal 2: Run the Orchestrator

```bash
python orchestrator.py
```

The orchestrator will:
1. Classify your query
2. Delegate to appropriate specialist agents
3. Synthesize and display the final analysis

---

## 📖 Usage Guide

### Running the Orchestrator

The orchestrator is the main entry point for complex queries that require multiple agents.

**Edit the query in `orchestrator.py`:**

```python
async def main():
    task = "I need a detailed analysis on Maruti Suzuki (MARUTI), also let me know if I should hold or sell"
    # ... rest of the code
```

**Run:**
```bash
python orchestrator.py
```

**Query Types Supported:**

| Type | Description | Example |
|------|-------------|---------|
| **TYPE A** | Market brief, headlines, sector snapshot | "Give me today's market headlines and IT sector news" |
| **TYPE B** | Buy/Sell/Hold decision for specific stock | "Should I buy TCS.NS? Give me a verdict" |
| **TYPE C** | Sector selection recommendation | "Which sector should I invest in right now?" |
| **TYPE D** | Deep company analysis | "Full analysis of Reliance Industries" |
| **TYPE E** | Macro impact analysis | "How will the Fed rate cut affect Indian IT stocks?" |

### Running Standalone Agents

You can run each specialist agent independently for testing or specific use cases.

#### News Agent (Standalone)

**Edit the query in `agents/news_agent.py`:**

```python
async def main():
    task = "What are the latest innovations in AI?"
    await run_news_agent(task, verbose=True)
```

**Run:**
```bash
python agents/news_agent.py
```

**Prerequisites:** News server must be running on port 8001.

#### Analysis Agent (Standalone)

**Edit the query in `agents/analysis_agent.py`:**

```python
async def main():
    task = """Fetch LIVE data from Yahoo Finance for TCS.NS.
    Get current price, 52-week range, 2-year history, P/E, revenue, FCF.
    Search news for recent earnings and AI strategy.
    Give a HOLD/SELL verdict for 6-month horizon."""
    
    await run_analysis_agent(task, verbose=True)
```

**Run:**
```bash
python agents/analysis_agent.py
```

**Prerequisites:** Yahoo Finance and Fetch MCP servers (automatically started via stdio).

---

## ⚙️ Configuration

All configuration is centralized in `agent_config.yaml`.

### Key Configuration Sections

#### 1. LLM Models

```yaml
models:
  large: "mistral-large-latest"   # orchestrator + analysis agent
  small: "mistral-small-latest"   # news agent
  temperature: 0                   # deterministic tool-calling
```

#### 2. MCP Servers

```yaml
mcp_servers:
  enhanced-news-server:
    transport: http
    url: "http://127.0.0.1:8001/mcp"
    enabled: true
  
  yahoo_finance:
    transport: stdio
    command: "mcp-yahoo-finance"
    args: []
    enabled: true
  
  fetch_server:
    transport: stdio
    command: "mcp-server-fetch"
    args: []
    enabled: true
```

#### 3. Agent → Server Mapping

```yaml
agent_servers:
  news_agent:
    - enhanced-news-server
  
  analysis_agent:
    - yahoo_finance
    - fetch_server
    - enhanced-news-server
```

#### 4. Agent Behavior

```yaml
agents:
  news_agent:
    recursion_limit: 20
    max_tool_calls: 10
  
  analysis_agent:
    recursion_limit: 40
    max_tool_calls: 15
  
  orchestrator:
    recursion_limit: 60
```

### News Feeds Configuration

Edit `news_feeds.yaml` to customize RSS feeds:

```yaml
india_business:
  - name: "Economic Times - Markets"
    url: "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"
  - name: "Moneycontrol - Markets"
    url: "https://www.moneycontrol.com/rss/marketreports.xml"
```

---

## 📁 Project Structure

```
MCP-Finance/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── agent_config.yaml              # Main configuration file
├── news_feeds.yaml                # RSS feed URLs
├── .gitignore                     # Git ignore rules
│
├── news-server.py                 # Enhanced news MCP server (HTTP, port 8001)
├── orchestrator.py                # Main orchestrator agent
│
└── agents/                        # Agent package
    ├── __init__.py                # Package exports
    ├── .env                       # Environment variables (create this)
    ├── common.py                  # Shared utilities (LLM, MCP config, ReAct builder)
    ├── tracer.py                  # Tool call tracing system
    ├── news_agent.py              # News specialist agent
    └── analysis_agent.py          # Financial analysis specialist agent
```

---

## 🔍 How It Works

### 1. Query Flow (Orchestrator Mode)

```
User Query
    ↓
Orchestrator classifies query type (A/B/C/D/E)
    ↓
Orchestrator calls delegate_to_news_agent() or delegate_to_analysis_agent()
    ↓
Specialist agent runs full ReAct loop:
  • Connects to MCP servers
  • Discovers tools
  • Calls tools iteratively
  • Returns final answer
    ↓
Orchestrator synthesizes final response
    ↓
Display to user
```

### 2. ReAct Agent Loop (Each Specialist)

```
START
  ↓
Agent Node:
  • Receives user task + conversation history
  • Calls Mistral LLM with bound tools
  • LLM decides: call tool(s) OR give final answer
  ↓
Tools Condition:
  • If tool_calls present → execute tools
  • If no tool_calls → END (final answer ready)
  ↓
TracedToolNode:
  • Executes each tool call
  • Records: tool name, args, result, timing
  • Returns ToolMessages
  ↓
Loop back to Agent Node
```

### 3. MCP Server Connection

**HTTP Transport (News Server):**
```python
mcp_config = {
    "enhanced-news-server": {
        "transport": "http",
        "url": "http://127.0.0.1:8001/mcp"
    }
}
```

**Stdio Transport (Yahoo Finance, Fetch):**
```python
mcp_config = {
    "yahoo_finance": {
        "transport": "stdio",
        "command": "mcp-yahoo-finance",  # Resolved via shutil.which()
        "args": []
    }
}
```

---

## 📊 Tool Tracing

The system includes a comprehensive tool tracing system for debugging and observability.

### Two-Level ID Hierarchy

- **run_id**: Parent ID shared across all agents in one orchestrator run
- **trace_id**: Child ID unique per agent run

### Example Trace Output

```
============================================================
  TOOL TRACE  |  run_id: f8a2d301  |  agent: analysis_agent  |  trace_id: d91e4f02  |  9 calls
============================================================
  #    Tool                                ms  Arguments
  ──────────────────────────────────────────────────────────
  1    get_current_stock_price            142  ticker='TCS.NS'
       ↳ {"price": 3456.75, "currency": "INR", "timestamp": "2025-01-15T15:30:00"}
  2    get_historical_stock_prices        891  ticker='TCS.NS', period='2y'
       ↳ {"data": [...], "52w_high": 4150.0, "52w_low": 3100.0}
  3    get_income_statement               456  ticker='TCS.NS'
       ↳ {"revenue": 2.3e11, "net_income": 4.2e10, "eps": 115.6}
  ...
  ──────────────────────────────────────────────────────────
  run_id: f8a2d301  |  trace_id: d91e4f02  |  total tool time: 4821ms
============================================================
```

### Real-Time Tool Call Logging

During execution, you'll see:
```
[f8a2d301][d91e4f02][analysis_agent][#1] get_current_stock_price({'ticker': 'TCS.NS'})
[f8a2d301][d91e4f02][analysis_agent][#2] get_historical_stock_prices({'ticker': 'TCS.NS'...})
```

Format: `[run_id][trace_id][agent_name][#call_number] tool_name(args)`

---

## 🔧 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'agents'`

**Cause**: Running the script from the wrong directory or Python path issues.

**Solution**: 
```bash
# Run from MCP-Finance root directory
cd /path/to/MCP-Finance
python agents/news_agent.py

# OR run as a module
python -m agents.news_agent
```

### Issue: `FileNotFoundError: agent_config.yaml`

**Cause**: Config file path issue.

**Solution**: Ensure `agent_config.yaml` is in the MCP-Finance root directory (not in agents/).

### Issue: `MISTRAL_API_KEY not found`

**Solution**:
1. Create `.env` file in `agents/` directory
2. Add: `MISTRAL_API_KEY=your_key_here`
3. Verify `python-dotenv` is installed

### Issue: `Error response 429: Rate limit exceeded`

**Cause**: Too many API calls to Mistral.

**Solution**:
- Wait a few minutes before retrying
- Consider using `mistral-small-latest` instead of `mistral-large-latest`
- Reduce `recursion_limit` in `agent_config.yaml`

### Issue: News server connection failed

**Cause**: News server not running.

**Solution**:
```bash
# Terminal 1
python news-server.py

# Wait for: "Uvicorn running on http://127.0.0.1:8001"
# Then run your agent in Terminal 2
```

### Issue: `mcp-yahoo-finance not found in PATH`

**Solution**:
```bash
pip install mcp-yahoo-finance
```

### Issue: Port 8001 already in use

**Solution**:
```bash
# macOS/Linux
lsof -ti:8001 | xargs kill -9

# Or change the port in news-server.py and agent_config.yaml
```

### Issue: Agent gives outdated stock prices

**Cause**: Agent using memorized training data instead of live tools.

**Solution**: Ensure your task string includes:
```
"Fetch LIVE data from Yahoo Finance tools for [TICKER]..."
```

The phrase "fetch LIVE data from Yahoo Finance tools" triggers the agent to use tools instead of memorized data.

---

## 📚 Additional Resources

- **MCP Documentation**: https://modelcontextprotocol.io/
- **FastMCP**: https://github.com/jlowin/fastmcp
- **Mistral AI**: https://docs.mistral.ai/
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **LangChain MCP Adapters**: https://github.com/langchain-ai/langchain-mcp-adapters
- **Yahoo Finance MCP**: https://github.com/modelcontextprotocol/servers/tree/main/src/yahoo-finance

---

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Guidelines

1. Follow the existing code structure
2. Add type hints to all functions
3. Update `agent_config.yaml` for new configuration options
4. Test both standalone and orchestrator modes
5. Update this README with new features

---

## ⚠️ Disclaimer

**Not financial advice.** The financial data and analysis provided by this system are for demonstration and educational purposes only. 

- Market data may be delayed or inaccurate
- Analysis is generated by AI and may contain errors
- Past performance does not guarantee future results
- Always do your own research
- Consult with SEBI-registered financial advisors before making investment decisions

---

## 🎓 Learning Objectives

This project demonstrates:

✅ **Multi-agent orchestration** using the supervisor pattern  
✅ **Model Context Protocol (MCP)** for dynamic tool discovery  
✅ **LangGraph** for structured agent workflows  
✅ **Tool call tracing** for debugging and observability  
✅ **Configuration-driven architecture** for maintainability  
✅ **Production-ready error handling** and logging  
✅ **Hybrid transport** (HTTP + stdio) MCP server connections  

---

**Built with ❤️ using MCP, Mistral AI, and LangGraph**
