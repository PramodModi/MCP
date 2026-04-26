"""
=============================================================================
program3_public_mcp_agent.py  —  Multi-Server News Research Agent
=============================================================================
Connects to TWO independent MCP servers and orchestrates them with Mistral:
 
  Server A — mcp_news.py  (http://127.0.0.1:8001/mcp)
             Our own news server. Tools: get_top_business_news, search_news,
             get_top_tech_stories, get_sector_news.
             Data: Google News RSS + Hacker News. No API key needed.
 
  Server B — mcp-server-fetch  (stdio subprocess)
             Official Anthropic pip package.
             Tool: fetch — retrieves full content of any URL as clean text.
             Used to read full article body when a headline needs more detail.
 
The agent uses both servers to produce a live market brief printed to console.
No files written. All output on stdout.
 
LLM      : Mistral AI (mistral-large-latest)
 
Start order:
  Terminal 1:  python program3_news_server.py        ← must be running first
  Terminal 2:  python program1_mcp_server.py         ← optional, not used here
  Terminal 3:  MISTRAL_API_KEY=xxx python program3_public_mcp_agent.py
 
Install  : pip install fastmcp mistralai mcp-server-fetch
=============================================================================
"""

import sys
import asyncio
import json
import os
from dotenv import load_dotenv, find_dotenv
from dataclasses import dataclass, field
import shutil
 
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from mistralai.client import Mistral

load_dotenv(find_dotenv())
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-small-latest"
NEWS_SERVER_URL  = "http://127.0.0.1:8001/mcp"

mistral = Mistral(api_key=MISTRAL_API_KEY)

# =============================================================================
# PART A — Server registry
#
# Each server gets its own Client instance.
# Tool names are prefixed so the agent knows which server to route to:
#   news__get_top_business_news
#   news__search_news
#   news__get_top_tech_stories
#   news__get_sector_news
#   fetch__fetch
# =============================================================================

class ServerConn:
    """Holds one MCP server connection and its discovered tools."""
    def __init__(self, name: str, description: str, client, tools=None, prefix=""):
        self.name        = name
        self.description = description
        self.client      = client
        self.tools       = tools if tools is not None else []
        self.prefix      = prefix


async def connect_all() -> dict[str, ServerConn]:
    """
    Connect to both MCP servers independently.
    A failure in one does not prevent the other from connecting.
    """
    registry: dict[str, ServerConn] = {}
 
    # ── Server A: our news server over HTTP ───────────────────────────────────
    # Client("http://...") — connects to an already-running HTTP MCP server.
    # mcp_news.py must be started first in a separate terminal.
    print(f"\nConnecting to news server at {NEWS_SERVER_URL} ...")
    try:
        news_client = Client(NEWS_SERVER_URL)
        await news_client.__aenter__()
        news_tools = await news_client.list_tools()
 
        registry["news"] = ServerConn(
            name        = "news",
            description = "Live business/financial news — Google News RSS + Hacker News",
            client      = news_client,
            tools       = news_tools,
            prefix      = "news",
        )
        print(f"  ✓ news server: {len(news_tools)} tools")
        for t in news_tools:
            print(f"      • {t.name}")
 
    except Exception as e:
        print(f"  ✗ news server failed: {e}")
        print(f"    Make sure program3_news_server.py is running first.")
 
    print(f"\nConnecting to mcp-server-fetch (stdio) ...")
    try:
        # Use StdioTransport with explicit command + args.
        # shutil.which() finds the mcp-server-fetch script installed by pip
        # in the active venv — works regardless of which Python is sys.executable.
        mcp_fetch_cmd = shutil.which("mcp-server-fetch")
        if mcp_fetch_cmd is None:
            raise RuntimeError(
                "mcp-server-fetch not found in PATH. "
                "Run: pip install mcp-server-fetch"
            )
        fetch_client = Client(
            StdioTransport(
                command = mcp_fetch_cmd,
                args    = [],
            )
        )
        await fetch_client.__aenter__()
        fetch_tools = await fetch_client.list_tools()
 
        registry["fetch"] = ServerConn(
            name        = "fetch",
            description = "Fetch full content of any URL as clean text",
            client      = fetch_client,
            tools       = fetch_tools,
            prefix      = "fetch",
        )
        print(f"  ✓ fetch server: {len(fetch_tools)} tools")
        for t in fetch_tools:
            print(f"      • {t.name}")
 
    except Exception as e:
        print(f"  ✗ fetch server failed: {e}")
        print(f"    Run: pip install mcp-server-fetch")
 
    return registry

async def disconnect_all(registry: dict[str, ServerConn]):
    for name, conn in registry.items():
        try:
            await conn.client.__aexit__(None, None, None)
            print(f"  ✓ Disconnected: {name}")
        except Exception:
            pass

# =============================================================================
# PART B — Unified tool list + routing
# =============================================================================
 
def build_mistral_tools(registry: dict[str, ServerConn]) -> list[dict]:
    """
    Flatten all server tools into one Mistral-format list.
    Each tool name is prefixed with its server prefix to enable routing.
 
    MCP inputSchema is a JSON Schema object — it passes directly to
    Mistral's 'parameters' field with zero transformation.
    """
    tools = []
    for conn in registry.values():
        for t in conn.tools:
            tools.append({
                "type": "function",
                "function": {
                    "name":        f"{conn.prefix}__{t.name}",
                    "description": f"[{conn.name}] {t.description or ''}",
                    "parameters":  t.inputSchema,
                }
            })
    return tools

def route(prefixed_name: str, registry: dict[str, ServerConn]) -> tuple:
    """
    'news__search_news'  →  (news_conn, 'search_news')
    'fetch__fetch'       →  (fetch_conn, 'fetch')
    Unknown              →  (None, '')
    """
    for conn in registry.values():
        prefix = conn.prefix + "__"
        if prefixed_name.startswith(prefix):
            return conn, prefixed_name[len(prefix):]
    return None, ""


# =============================================================================
# PART C — Agent loop with Mistral native tool calling
# =============================================================================
 
async def run_agent(registry: dict[str, ServerConn], task: str):
    """
    Multi-turn Mistral agent loop.
    Calls tools across both servers, prints final answer to console.
    No files written.
    """
    mistral_tools = build_mistral_tools(registry)
 
    if not mistral_tools:
        print("No tools available — check server connections above.")
        return
 
    available = [t["function"]["name"] for t in mistral_tools]
    print(f"\n{'=' * 65}")
    print(f"TASK  : {task}")
    print(f"TOOLS : {available}")
    print(f"{'=' * 65}")
 
    system = {
        "role": "system",
        "content": (
            "You are a financial news analyst covering Indian and global markets.\n\n"
            "You have access to these tools:\n"
            "  news__get_top_business_news  — top business headlines by region\n"
            "  news__search_news            — search any topic on Google News\n"
            "  news__get_top_tech_stories   — top Hacker News stories by score\n"
            "  news__get_sector_news        — sector-specific NSE headlines\n"
            "  fetch__fetch                 — fetch full text of any article URL\n\n"
            "Workflow:\n"
            "1. Call news__get_top_business_news(region='India') for macro headlines.\n"
            "2. Call news__get_sector_news for 1-2 relevant sectors.\n"
            "3. Call news__get_top_tech_stories for IT sector context.\n"
            "4. Call news__search_news for any specific topic needing more depth.\n"
            "5. Optionally call fetch__fetch on 1 key article URL for full detail.\n\n"
            "Print a structured market brief with these sections:\n\n"
            "── MARKET HEADLINES ─────────────────────────────────────\n"
            "  5-6 key financial headlines with source names.\n\n"
            "── SECTOR PULSE ─────────────────────────────────────────\n"
            "  2-3 bullet points per sector you looked at.\n\n"
            "── TECH & GLOBAL CONTEXT ────────────────────────────────\n"
            "  3-4 top HN stories with scores relevant to markets.\n\n"
            "── ANALYST BRIEF ────────────────────────────────────────\n"
            "  4-5 sentences synthesising overall market sentiment.\n\n"
            "── STOCKS TO WATCH ──────────────────────────────────────\n"
            "  3-4 specific tickers or sectors with one-line rationale.\n\n"
            "Be precise. Quote actual headlines. Cite sources.\n"
            "End with: 'Not financial advice. Data from public news feeds.'"
        )
    }
 
    messages = [system, {"role": "user", "content": task}]
    max_iters = 12
 
    for iteration in range(1, max_iters + 1):
        print(f"\n[Iter {iteration}] Calling {MISTRAL_MODEL} ...")
 
        response = mistral.chat.complete(
            model       = MISTRAL_MODEL,
            messages    = messages,
            tools       = mistral_tools,
            tool_choice = "auto",
        )
 
        msg    = response.choices[0].message
        reason = response.choices[0].finish_reason
 
        # ── Model wants to call tools ─────────────────────────────────────────
        if reason == "tool_calls" and msg.tool_calls:
 
            # Append the full assistant message (Mistral requires this for threading)
            messages.append({
                "role":       "assistant",
                "content":    msg.content or "",
                "tool_calls": [
                    {
                        "id":       tc.id,
                        "type":     "function",
                        "function": {
                            "name":      tc.function.name,
                            "arguments": (
                                tc.function.arguments
                                if isinstance(tc.function.arguments, str)
                                else json.dumps(tc.function.arguments)
                            ),
                        }
                    }
                    for tc in msg.tool_calls
                ]
            })
 
            # Execute each tool call via the correct MCP server
            for tc in msg.tool_calls:
                prefixed  = tc.function.name
                raw_args  = tc.function.arguments
                args      = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
 
                print(f"  → {prefixed}({json.dumps(args)[:60]})")
 
                conn, original = route(prefixed, registry)
                print(f"  → conn: {conn}")
                print(f"  → original: {original}")
 
                if conn is None:
                    result_text = json.dumps({"error": f"No server for tool: {prefixed}"})
                else:
                    try:
                        result      = await conn.client.call_tool(original, args)
                        result_text = "".join(
                            b.text for b in result.content if hasattr(b, "text")
                        )
                        # Guard: truncate very large fetched pages
                        # Full HTML can be 50k+ tokens; model needs the gist only
                        if len(result_text) > 6000:
                            result_text = result_text[:6000] + "\n...[truncated]"
                    except Exception as e:
                        result_text = json.dumps({"error": str(e)})
 
                print(f"  ← {len(result_text)} chars from [{conn.name}]")
                #print(f" news result text: ← {(result_text)} ")
 
                # Inject result as tool-role message (Mistral threading requirement)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         prefixed,
                    "content":      result_text,
                })
 
        # ── Model is done — print the brief ───────────────────────────────────
        elif reason == "stop":
            final = msg.content or ""
            print(f"\n{'=' * 65}")
            print("MARKET BRIEF")
            print('=' * 65)
            print(final)
            print('=' * 65)
            return
 
        else:
            print(f"  Unexpected finish_reason: {reason}. Stopping.")
            return
 
    print("Agent reached max iterations.")


async def main():
    print("=" * 65)
    print("Program 3: Multi-Server News Research Agent")
    print(f"Model         : {MISTRAL_MODEL}")
    print(f"News server   : {NEWS_SERVER_URL}  (program3_news_server.py)")
    print(f"Fetch server  : mcp-server-fetch (stdio subprocess)")
    print("=" * 65)
    print("\nPrerequisites:")
    # Connect to both servers
    registry = await connect_all()
 
    if not registry:
        print("\nNo servers connected. Exiting.")
        return
 
    # Run the research task
    task = (
        # "Give me a comprehensive market brief for today. "
        # "Cover Indian stock market headlines, IT and Banking sector news, "
        # "top tech stories from the global community, and give me "
        # "3-4 specific stocks or sectors to watch with clear rationale."
        "why ServiceNow share is declining although there is good quarterly results and guidance"
    )

    await run_agent(registry, task)
 
    # Clean disconnect
    print("\nDisconnecting ...")
    await disconnect_all(registry)
 
 
if __name__ == "__main__":
    asyncio.run(main())

 
 
