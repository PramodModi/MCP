"""
=============================================================================
PROGRAM 2: MCP Client / Agent — Financial Research Agent (Mistral LLM)
=============================================================================
Purpose  : Connects to the independently running Program 1 server over HTTP,
           discovers its tools via the MCP initialize handshake, then runs a
           multi-turn agent loop powered by Mistral.
 
           Program 1 and Program 2 are completely independent processes.
           Start Program 1 first, then run this file in a separate terminal.
 
LLM      : Mistral AI (mistral-large-latest by default).
           Set env var: MISTRAL_API_KEY=your_key
 
Transport: streamable-http — connects to http://127.0.0.1:8000/mcp
           Override with: MCP_SERVER_URL=http://host:port/mcp
 
Key concepts demonstrated:
  • HTTP-based MCP connection (no subprocess spawning)
  • MCP capability discovery (list_tools / list_resources / list_prompts)
  • MCP JSON Schema → Mistral function-calling format (direct passthrough)
  • Mistral native tool-calling loop (tool_calls objects, not JSON prose)
  • tool-role message injection after each MCP result
  • Circuit breaker (max_iterations) preventing infinite agent loops
 
Install  : pip install fastmcp mistralai
Run      : # Terminal 1 — start the server first
           python program1_mcp_server.py
 
           # Terminal 2 — run the agent
           MISTRAL_API_KEY=your_key python program2_local_agent.py
=============================================================================
"""

import asyncio
import json
import os
from dotenv import load_dotenv, find_dotenv
from typing import Any, Dict, List, Optional

from fastmcp import Client
from mistralai.client import Mistral

# Load environment variables from .env file
load_dotenv(find_dotenv())
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-small-latest"
mistral = Mistral(api_key=MISTRAL_API_KEY)

# =============================================================================
# PART A — Server inspection
# Every MCP host must discover capabilities on connect before using any tools.
# =============================================================================
async def inspect_server(client: Client) -> dict:
    """
    Run the MCP capability discovery sequence:
      initialize handshake  →  list_tools  →  list_resources  →  list_prompts
 
    Returns raw capability objects the agent loop will use to build
    Mistral-format tool definitions.
    """
    tools     = await client.list_tools()
    resources = await client.list_resources()
    prompts   = await client.list_prompts()
 
    print("\n" + "=" * 62)
    print("MCP SERVER CAPABILITIES DISCOVERED")
    print("=" * 62)
 
    print(f"\nTOOLS ({len(tools)}):")
    for t in tools:
        desc_preview = (t.description or "").split("\n")[0][:65]
        params       = list(t.inputSchema.get("properties", {}).keys())
        required     = t.inputSchema.get("required", [])
        print(f"  • {t.name}")
        print(f"    {desc_preview}")
        print(f"    params: {params}  required: {required}")
 
    print(f"\nRESOURCES ({len(resources)}):")
    for r in resources:
        print(f"  • {r.uri}")
 
    print(f"\nPROMPTS ({len(prompts)}):")
    for p in prompts:
        print(f"  • {p.name}")
 
    return {"tools": tools, "resources": resources, "prompts": prompts}


    # PART B — Convert MCP tool definitions → Mistral tool format
#
# Mistral's tool-calling API expects this structure:
#   {
#     "type": "function",
#     "function": {
#       "name":        <str>,
#       "description": <str>,
#       "parameters":  <JSON Schema object>   ← exactly what MCP gives us
#     }
#   }
#
# MCP's inputSchema IS a JSON Schema object, so the conversion is 1-to-1.
# This is not a coincidence — MCP was designed to compose with LLM tool APIs.
# =============================================================================
 
def mcp_tools_to_mistral_format(mcp_tools: list) -> list:
    """
    Convert FastMCP tool objects → Mistral function-calling format.
    The JSON Schema passthrough is the key architectural win of MCP:
    the same schema the server declared is sent verbatim to the model.
    """
    mistral_tools = []
    for tool in mcp_tools:
        mistral_tools.append({
            "type": "function",
            "function": {
                "name":        tool.name,
                "description": tool.description or "",
                "parameters":  tool.inputSchema,   # direct passthrough — no transformation needed
            }
        })
    return mistral_tools

# =============================================================================
# PART C — The Agent Loop (native tool-calling variant)
#
# Message flow:
#   user_message
#     → Mistral (with tool definitions in tools=[])
#     → model returns: finish_reason="tool_calls" or finish_reason="stop"
#     → if tool_calls: execute each call via MCP, append tool messages, loop
#     → if stop: return content as final answer
# =============================================================================

async def run_financial_agent(client: Client, capabilities: dict, user_query: str) -> str:
    """
    Multi-turn financial research agent using Mistral native tool calling.
 
    The loop never manually parses JSON from the model — Mistral returns
    structured tool_calls objects. This is more robust than the prose-JSON
    pattern and is how production agentic systems are built.
    """
    mistral_tools = mcp_tools_to_mistral_format(capabilities["tools"])
 
    # System prompt: give the model its persona and workflow guidance
    system_msg = {
        "role": "system",
        "content": (
            "You are a SEBI-registered equity research analyst assistant specialising in "
            "NSE-listed Indian stocks. You have access to real-time market tools.\n\n"
            "Research workflow:\n"
            "1. Always call get_market_news first to establish macro context.\n"
            "2. Call get_sector_overview for the relevant sector.\n"
            "3. Call get_stock_fundamentals for specific stocks.\n"
            "4. Call compare_stocks when ranking 2+ stocks.\n"
            "5. Call analyse_portfolio if the user provides holdings.\n\n"
            "Be analytical. Cite specific numbers. Always add a disclaimer at the end."
        )
    }
 
    messages = [
        system_msg,
        {"role": "user", "content": user_query}
    ]
 
    print(f"\n{'─' * 62}")
    print(f"USER QUERY: {user_query}")
    print(f"{'─' * 62}")
    max_iterations = 10   # circuit breaker: models can loop on bad tool results
    iteration      = 0
 
    while iteration < max_iterations:
        iteration += 1
        print(f"\n[Iteration {iteration}] Calling Mistral ({MISTRAL_MODEL})...")
        # ── Mistral inference call ────────────────────────────────────────────
        response = mistral.chat.complete(
            model    = "mistral-small-latest",
            messages = messages,
            tools    = mistral_tools,
            tool_choice = "auto",   # let model decide whether to call a tool
        )
 
        msg           = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        # ── Branch: model wants to call tools ────────────────────────────────
        if finish_reason == "tool_calls" and msg.tool_calls:
            # Append the assistant message (contains tool_calls metadata)
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments
                            if isinstance(tc.function.arguments, str)
                            else json.dumps(tc.function.arguments)
                    }
                }
                for tc in msg.tool_calls
            ]})

        # Execute each tool call via MCP — model may request multiple at once
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                # arguments may arrive as str or dict depending on Mistral version
                raw_args  = tc.function.arguments
                tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
 
                print(f"  → Calling MCP tool: {tool_name}({json.dumps(tool_args)[:70]}...)")

                # ── The MCP call ──────────────────────────────────────────────
                try:
                    result       = await client.call_tool(tool_name, tool_args)
                    result_text  = ""
                    # CallToolResult has a content attribute containing the list of blocks
                    for block in result.content:
                        if hasattr(block, "text"):
                            result_text += block.text
                except Exception as e:
                    result_text = json.dumps({"error": str(e)})
 
                print(f"  ← Result ({len(result_text)} chars): {result_text[:100]}...")
 
                # Inject result as a tool-role message (Mistral convention)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         tool_name,
                    "content":      result_text,
                })
    # ── Branch: model is done ─────────────────────────────────────────────
        elif finish_reason == "stop":
            final_answer = msg.content or ""
            print(f"\n{'=' * 62}")
            print("AGENT FINAL ANSWER")
            print('=' * 62)
            print(final_answer)
            return final_answer
 
        else:
            print(f"  Unexpected finish_reason: {finish_reason}. Stopping.")
            break
 
    return "Agent reached maximum iterations."

# =============================================================================
# PART D — Resource reading (separate from tool calls)
# =============================================================================
 
async def read_market_snapshot(client: Client):
    """
    Demonstrate reading an MCP resource — pure GET, no tool invocation.
    Resources are for data the host wants to pre-load, not query dynamically.
    """
    print("\n" + "─" * 62)
    print("RESOURCE: market://nifty50/snapshot")
    result = await client.read_resource("market://nifty50/snapshot")
    for block in result:
        if hasattr(block, "text"):
            data = json.loads(block.text)
            print(f"  NIFTY 50: {data['level']}  ({data['1d_chg_pct']:+.2f}%)")
            print(f"  Top movers: {[m['ticker'] for m in data['top_movers']]}")

async def main():
    # ---------------------------------------------------------------------------
    # Server URL — defaults to local Program 1, overridable via env var.
    # If Program 1 is running on a different host or port:
    #   MCP_SERVER_URL=http://192.168.1.10:9000/mcp python program2_local_agent.py
    # ---------------------------------------------------------------------------
    server_url = "http://127.0.0.1:8000/mcp"
 
    print("=" * 62)
    print("Program 2: Financial Research Agent (Mistral)")
    print(f"MCP server: {server_url}")
    print("=" * 62)
    print("\nMake sure mcpserver_local is running before this step.")
    print("  Terminal 1: python mcpserver_local.py")
    print("  Terminal 2: python mcpclient_local.py  (this)\n")
 

    # The server must already be up and accepting connections at this URL.
    # FastMCP handles the MCP initialize handshake on __aenter__.
    async with Client(server_url) as client:
 
        # Step 1: capability discovery (initialize → list_tools/resources/prompts)
        capabilities = await inspect_server(client)
 
        # Step 2: read a resource — pure HTTP GET, no tool invocation
        await read_market_snapshot(client)
 
        # Step 3: run the agent on realistic financial queries
        queries = [
            "Give me a full analysis of TCS — is it a good buy right now for a moderate investor?",
            "Compare HDFCBANK and SBIN. Which is better value in the current rate environment?",
        ]
 
        for query in queries:
            await run_financial_agent(client, capabilities, query)
            print("\n" + "─" * 62 + "\n")
 
 
if __name__ == "__main__":
    asyncio.run(main())
 
