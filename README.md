# MCP Server-Agent Examples

This repository contains multiple examples demonstrating the **Model Context Protocol (MCP)** with different architectures and LLM frameworks.
Go to folder [MCP-Server-Agent-Both_Local]
## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Example 1: Basic MCP Server-Client (Financial Markets)](#example-1-basic-mcp-server-client-financial-markets)
- [Example 2: Multi-Server Agent with Mistral](#example-2-multi-server-agent-with-mistral)
- [Example 3: Multi-Server Agent with LangGraph](#example-3-multi-server-agent-with-langgraph)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

**Project Name : MCP Server-Agent Examples**
**Folder : MCP-Server-Agent-Both_Local**

This project demonstrates three different MCP implementations:

1. **Basic MCP Server-Client**: A financial markets knowledge server with a Mistral-powered agent
2. **Multi-Server Agent (Mistral)**: Connects to multiple MCP servers for news research
3. **Multi-Server Agent (LangGraph)**: Uses LangChain and LangGraph for orchestration

All examples use the **Model Context Protocol (MCP)** to expose tools and resources that AI agents can discover and use dynamically.

---

## ✅ Prerequisites

- **Python 3.11+** (recommended)
- **pip** package manager
- **Mistral API Key** (get it from [Mistral AI](https://console.mistral.ai/))
- **Git** (to clone the repository)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd MCP-Server-Agent-Both_Local
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

Create a `requirements.txt` file with the following content:

```txt
fastmcp
mistralai
python-dotenv
langchain-mcp-adapters
langgraph
langchain-mistralai
mcp-server-fetch
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
MISTRAL_API_KEY=your_mistral_api_key_here
```

**Important**: Never commit your `.env` file to version control!

---

## 📊 Example 1: Basic MCP Server-Client (Financial Markets)

This example demonstrates a standalone MCP server exposing financial market tools, with a Mistral-powered agent client.

### Architecture

```
┌─────────────────────┐         HTTP          ┌──────────────────────┐
│  mcpserver_local.py │ ◄──────────────────► │  mcpclient_local.py  │
│                     │  (port 8000)          │                      │
│  - Financial Tools  │                       │  - Mistral Agent     │
│  - Market Data      │                       │  - Tool Calling      │
│  - Resources        │                       │  - Multi-turn Loop   │
└─────────────────────┘                       └──────────────────────┘
```

### Features

**Server (`mcpserver_local.py`):**
- Exposes NSE stock data and portfolio analytics
- Tools: `get_stock_fundamentals`, `compare_stocks`
- Resources: `market://nifty50/snapshot`
- Runs on `http://127.0.0.1:8000/mcp`

**Client (`mcpclient_local.py`):**
- Connects to the server via HTTP
- Uses Mistral AI for intelligent tool calling
- Multi-turn agent loop with circuit breaker
- Demonstrates MCP capability discovery

### How to Run

#### Terminal 1: Start the MCP Server

```bash
python mcpserver_local.py
```

You should see:
```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

#### Terminal 2: Run the Agent Client

```bash
python mcpclient_local.py
```

The agent will:
1. Discover available tools from the server
2. Read market snapshot resource
3. Execute financial research queries
4. Display analysis results

### Expected Output

```
==============================================================
MCP SERVER CAPABILITIES DISCOVERED
==============================================================

TOOLS (2):
  • get_stock_fundamentals
  • compare_stocks

RESOURCES (1):
  • market://nifty50/snapshot

──────────────────────────────────────────────────────────────
USER QUERY: Give me a full analysis of TCS — is it a good buy...
──────────────────────────────────────────────────────────────

[Iteration 1] Calling Mistral (mistral-small-latest)...
  → Calling MCP tool: get_stock_fundamentals({"ticker": "TCS"})
  ← Result (450 chars): {"name": "Tata Consultancy Services"...}

==============================================================
AGENT FINAL ANSWER
==============================================================
[Analysis output here...]
```

### Stopping the Servers

Press `Ctrl+C` in each terminal to stop the processes.

---

## 🌐 Example 2: Multi-Server Agent with Mistral

This example connects to **two independent MCP servers** and orchestrates them with Mistral AI.

### Architecture

```
┌──────────────────────┐
│  multi_mcp_agent.py  │
│   (Mistral Agent)    │
└──────────┬───────────┘
           │
           ├──► Server A: mcp-news-server.py (HTTP, port 8001)
           │    Tools: get_top_business_news, search_news,
           │           get_top_tech_stories, get_sector_news
           │
           └──► Server B: mcp-server-fetch (stdio subprocess)
                Tool: fetch (retrieve full article content)
```

### Features

- **Server A (News Server)**: Uses free public APIs (Google News RSS + Hacker News)
- **Server B (Fetch Server)**: Official Anthropic MCP server for fetching web content
- **Agent**: Orchestrates both servers to produce comprehensive market briefs

### How to Run

#### Terminal 1: Start the News Server

```bash
python mcp-news-server.py
```

You should see:
```
Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
```

#### Terminal 2: Run the Multi-Server Agent

```bash
python multi_mcp_agent.py
```

The agent will:
1. Connect to both MCP servers (HTTP + stdio)
2. Discover all available tools
3. Execute a comprehensive market research task
4. Display formatted market brief

### Expected Output

```
==============================================================
MULTI-SERVER MCP AGENT — FINANCIAL NEWS RESEARCH
==============================================================

Connecting to 2 MCP servers...
  ✓ news-server (HTTP): 4 tools discovered
  ✓ fetch-server (stdio): 1 tool discovered

──────────────────────────────────────────────────────────────
RESEARCH TASK: Give me a comprehensive market brief...
──────────────────────────────────────────────────────────────

[Iteration 1] Calling Mistral...
  → get_top_business_news(region="India", limit=10)
  → get_sector_news(sector="Technology", limit=5)
  ...

MARKET HEADLINES
• [Headline 1]
• [Headline 2]
...

ANALYST BRIEF
[Market analysis...]

Not financial advice. Data from public news feeds.
```

---

## 🔗 Example 3: Multi-Server Agent with LangGraph

This example uses **LangChain** and **LangGraph** for a more structured agentic workflow.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                      │
│                                                             │
│   START → [agent node] ──tool_calls──► [tools node]        │
│                 ▲                           │               │
│                 └──────── results ──────────┘               │
│                 │                                           │
│              stop → END                                     │
└─────────────────────────────────────────────────────────────┘
         │                          │
    ChatMistralAI              MultiServerMCPClient
    (mistral-large)            ┌────────────────────┐
                               │ news-server (HTTP)  │ → 4 tools
                               │   port 8001         │
                               ├────────────────────┤
                               │ mcp-server-fetch    │ → 1 tool
                               │   (stdio)           │
                               └────────────────────┘
```

### Features

- **LangGraph StateGraph**: Structured agent workflow with nodes and edges
- **MultiServerMCPClient**: Automatic tool discovery from multiple servers
- **ChatMistralAI**: LangChain integration with Mistral
- **ToolNode**: Prebuilt node for executing tool calls

### How to Run

#### Terminal 1: Start the News Server

```bash
python mcp-news-server.py
```

#### Terminal 2: Run the LangGraph Agent

```bash
python multi_mcp_agent_langchain.py
```

The agent will:
1. Build a LangGraph state machine
2. Connect to both MCP servers via `MultiServerMCPClient`
3. Execute the research task through the graph
4. Display the final market brief

### Expected Output

```
[Tool objects discovered...]

MARKET HEADLINES
• [Headlines from Google News RSS]

SECTOR PULSE
• Technology: [Analysis]
• Banking: [Analysis]

TECH CONTEXT
• [Top Hacker News stories]

ANALYST BRIEF
[Comprehensive market analysis...]

STOCKS TO WATCH
• TCS - [Rationale]
• HDFCBANK - [Rationale]

Not financial advice. Data from public news feeds.
```

---

## 📁 Project Structure

```
MCP-Server-Agent-Both_Local/
├── README.md                          # This file
├── .env                               # API keys (create this, not in git)
├── .gitignore                         # Git ignore file
├── requirements.txt                   # Python dependencies
│
├── mcpserver_local.py                 # Example 1: Financial markets MCP server
├── mcpclient_local.py                 # Example 1: Mistral agent client
│
├── mcp-news-server.py                 # Example 2 & 3: News MCP server
├── multi_mcp_agent.py                 # Example 2: Multi-server Mistral agent
└── multi_mcp_agent_langchain.py       # Example 3: LangGraph agent
```

---

## 🔧 Troubleshooting

### Issue: `ImportError: cannot import name 'Mistral'`

**Solution**: Update the import statement:
```python
from mistralai.client import Mistral  # Correct
# NOT: from mistralai import Mistral
```

### Issue: `RuntimeError: Client failed to connect`

**Cause**: The MCP server isn't running.

**Solution**: Make sure to start the server first in Terminal 1 before running the client in Terminal 2.

### Issue: `'CallToolResult' object is not iterable`

**Cause**: Incorrect result handling in older code.

**Solution**: Access the `content` attribute:
```python
result = await client.call_tool(tool_name, tool_args)
for block in result.content:  # Correct
    if hasattr(block, "text"):
        result_text += block.text
```

### Issue: `MISTRAL_API_KEY not found`

**Solution**: 
1. Create a `.env` file in the project root
2. Add: `MISTRAL_API_KEY=your_key_here`
3. Make sure `python-dotenv` is installed

### Issue: `mcp-server-fetch not found in PATH`

**Solution**: Install the package:
```bash
pip install mcp-server-fetch
```

### Issue: Port 8000 or 8001 already in use

**Solution**: Kill the existing process:
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9
lsof -ti:8001 | xargs kill -9

# Or use different ports in the code
```

---

## 📚 Additional Resources

- **MCP Documentation**: [Model Context Protocol](https://modelcontextprotocol.io/)
- **FastMCP**: [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- **Mistral AI**: [Mistral Documentation](https://docs.mistral.ai/)
- **LangGraph**: [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

## 📝 License

This project is for educational purposes. Feel free to use and modify as needed.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## ⚠️ Disclaimer

**Not financial advice.** The financial data and analysis provided by these examples are for demonstration purposes only. Always do your own research and consult with financial professionals before making investment decisions.
