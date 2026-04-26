"""
=============================================================================
program3_langgraph_agent.py  —  LangGraph + LangChain MCP Agent
=============================================================================
Architecture:
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
 
Key LangGraph concepts shown:
  • StateGraph with MessagesState — the graph state is just a list of messages
  • Two nodes: 'agent' (LLM call) and 'tools' (MCP tool execution)
  • tools_condition — prebuilt edge that routes to 'tools' or END
  • ToolNode — prebuilt node that executes whichever tools the LLM requested
  • MultiServerMCPClient — discovers tools from all servers, wraps as LangChain tools
 
Key MCP concepts shown:
  • HTTP transport for standalone server (news-server on port 8001)
  • stdio transport for subprocess server (mcp-server-fetch)
  • Tool schemas flow automatically: MCP → LangChain → Mistral
 
Install:
    pip install langchain-mcp-adapters langgraph langchain-mistralai mcp-server-fetch
 
Start the news server first:
    Terminal 1: python program3_news_server.py
 
Then run this agent:
    Terminal 2: MISTRAL_API_KEY=your_key python program3_langgraph_agent.py
=============================================================================
"""

import asyncio
from dotenv import load_dotenv, find_dotenv
import os
import shutil

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mistralai import ChatMistralAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, ToolMessage

load_dotenv(find_dotenv())
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

MISTRAL_MODEL   = "mistral-large-latest"
NEWS_SERVER_URL = "http://127.0.0.1:8001/mcp"

llm = ChatMistralAI(api_key=MISTRAL_API_KEY, model=MISTRAL_MODEL, temperature=0)

# =============================================================================
# PART B — MCP Server Configuration
#
# MultiServerMCPClient accepts a dict of server configs — same format as
# Claude Desktop's claude_desktop_config.json (just the mcpServers contents).
#
# Two transport types used here:
#
#   "http"  — connects to a running HTTP server by URL.
#             program3_news_server.py must already be running on port 8001.
#
#   "stdio" — spawns a subprocess and communicates via stdin/stdout.
#             mcp-server-fetch is an installed pip package; shutil.which()
#             finds its entry-point script in the active venv's bin/.
#
# MultiServerMCPClient.get_tools() discovers all tools from all servers
# and wraps each one as a standard LangChain BaseTool. From this point on,
# LangGraph and Mistral treat them identically — no MCP-specific code needed.
# =============================================================================

def get_mcp_server_config() -> dict:
    """
    Build the MultiServerMCPClient config dict.
    Resolves mcp-server-fetch path at runtime so it works regardless
    of where pip installed it.
    """
    fetch_cmd = shutil.which("mcp-server-fetch")
    if fetch_cmd is None:
        raise RuntimeError(
            "mcp-server-fetch not found in PATH.\n"
            "Fix: pip install mcp-server-fetch"
        )
 
    return {
        "news-server": {
            "transport": "http",
            "url": NEWS_SERVER_URL,
        },
 
        # ── Server 2: mcp-server-fetch ────────────────────────────────────────
        # stdio transport — MultiServerMCPClient spawns it as a subprocess.
        "fetch-server": {
            "transport": "stdio",
            "command":    fetch_cmd,
            "args":       [],
        },
    }

# =============================================================================
# PART C — LangGraph graph construction
#
# The ReAct loop here is:
#   START
#     │
#     ▼
#  [agent]  ← calls LLM with current messages + bound tools
#     │
#     ├── finish_reason=tool_calls → [tools] → back to [agent]
#     │
#     └── finish_reason=stop ──────────────────────────────► END
#
# tools_condition is a prebuilt conditional edge that reads finish_reason.
# ToolNode is a prebuilt node that executes all tool_calls in the last message.
# =============================================================================

def build_graph(tools : list):
    llm_with_tools = llm.bind_tools(tools)
    
    
    def _normalize_tool_messages(messages: list) -> list:
        """
        Mistral API requires tool message content to be a plain string.
        langchain-mcp-adapters returns content as a list of dicts like:
          [{"type": "text", "text": "...", "id": "lc_..."}]
        This function flattens that to a plain string before sending to Mistral.
        Root cause: MCP tools use response_format='content_and_artifact' which
        produces structured content blocks that Mistral's API rejects (422 error).
        """
        normalized = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
                # Extract text from each content block and join as plain string
                text = "\n".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in msg.content
                )
                # Rebuild ToolMessage with string content
                msg = ToolMessage(
                    content        = text,
                    tool_call_id   = msg.tool_call_id,
                    name           = msg.name,
                )
            normalized.append(msg)
        return normalized
    
    # ── Agent node ────────────────────────────────────────────────────────────
    # Receives state, prepends system message, calls LLM, returns AI message.
    # The system message is injected here (not in the initial user message)
    # so it applies on every iteration of the loop.
    def agent_node(state: MessagesState) -> dict:
        system = SystemMessage(content="You are a financial news analyst covering Indian and global markets.\n\n"
            "Research workflow — use available tools in this order:\n"
            "1. Fetch broad market headlines first (India region).\n"
            "2. Drill into 1-2 relevant sectors.\n"
            "3. Check tech industry stories for IT sector context.\n"
            "4. Search for any specific company or event needing more depth.\n"
            "5. Optionally fetch 1 key article URL for full detail.\n\n"
            "Output format:\n"
            "  MARKET HEADLINES    — 5-6 key headlines with source names\n"
            "  SECTOR PULSE        — 2-3 bullets per sector analysed\n"
            "  TECH CONTEXT        — top tech stories relevant to markets\n"
            "  ANALYST BRIEF       — 4-5 sentences on overall sentiment\n"
            "  STOCKS TO WATCH     — 3-4 tickers with one-line rationale\n\n"
            "Quote actual headlines. Cite sources. Be precise with numbers.\n"
            "End with: 'Not financial advice. Data from public news feeds.'")

        messages = [system] + _normalize_tool_messages(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()

async def main():
    server_config = get_mcp_server_config()
    mcp_client = MultiServerMCPClient(server_config)
    tools = await mcp_client.get_tools()
    print(tools)

    graph = build_graph(tools)
        
    # Define the research task
    task = (
        "Give me a comprehensive market brief for today. "
        "Cover Indian stock market headlines, IT and Banking sector news, "
        "top tech stories from the global community, and give me "
        "3-4 specific stocks or sectors to watch with clear rationale."
    )


    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config = {"recursion_limit": 25}
    )
    final_message = final_state["messages"][-1].content
    print(final_message)
if __name__ == "__main__":
    asyncio.run(main())