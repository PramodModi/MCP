"""
agents/news_agent.py  —  News Agent
=====================================
Specialist agent for all news-related queries.
 
Responsibility:
  Fetches, filters, and summarises live business, technology, and AI news.
  Handles TYPE A queries (market briefs, headlines, sector snapshots).
  Provides news context to the Analysis Agent when requested by orchestrator.
 
MCP Servers connected:
  news-server (http://127.0.0.1:8001) — 8 tools:
    get_india_business_news    get_global_business_news
    get_tech_news              get_ai_news
    get_ai_research_papers     get_sector_news
    get_hacker_news_top        search_news
 
LLM: mistral-small-latest
  News summarisation needs speed and breadth, not deep financial reasoning.
  Using small model here saves ~70% cost vs large for these queries.
 
Can be run standalone:
  python agents/news_agent.py
Or imported and called by orchestrator:
  from agents.news_agent import run_news_agent
"""

import asyncio
import sys
from pathlib import Path

# Allow running as a standalone script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_mcp_adapters.client import MultiServerMCPClient
from agents.common import (
    CONFIG,
    get_llm,
    build_mcp_config,
    build_react_agent,
    ToolTracer,
)

# =============================================================================
# SYSTEM PROMPT
# =============================================================================
# Behavioural guidance only — no tool names listed here.
# Tool schemas are sent to Mistral via bind_tools() automatically.
# Adding a new MCP tool to the news server requires zero changes here.
# =============================================================================
 
NEWS_AGENT_SYSTEM_PROMPT = """You are a financial news analyst specialising in \
Indian and global markets. Your job is to fetch, filter, and summarise the most \
relevant recent news for the user's query.
 
Research workflow:
1. For Indian market queries → start with get_india_business_news
2. For global/US context   → use get_global_business_news
3. For sector-specific news → use get_sector_news with the relevant sector
4. For tech industry context → use get_tech_news (relevant for IT stocks)
5. For AI industry news     → use get_ai_news (relevant for tech stocks)
6. For community sentiment  → use get_hacker_news_top
7. For targeted queries     → use search_news with specific keywords
 
Output format:
  MARKET HEADLINES    — 5-6 key headlines, each with [Source, Date]
  SECTOR PULSE        — 2-3 bullets per sector covered
  TECH & AI CONTEXT   — top stories relevant to markets with HN scores if available
  ANALYST BRIEF       — 4-5 sentences synthesising overall news sentiment
 
Rules:
- Every headline must include the source name and date from the tool result
- Quote actual headlines verbatim — do not paraphrase titles
- If a topic has no news in the last 48 hours, say so explicitly
- Flag if the same story appears across multiple sources (indicates high impact)
 
End with: "Data from public RSS feeds and APIs. Not financial advice.\""""


# =============================================================================
# AGENT RUNNER
# =============================================================================
 
async def run_news_agent(task: str, verbose: bool = True, run_id: str | None = None) -> str:
    """
    Run the News Agent on a given task and return the final answer as a string.
 
    Args:
        task:    The news research task (natural language)
        verbose: Print tool calls and iteration info to stdout
 
    Returns:
        Final answer string from the agent
    """
    agent_cfg = CONFIG["agents"]["news_agent"]
 
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"[NEWS AGENT] Starting task")
        print(f"Task    : {task[:100]}{'...' if len(task) > 100 else ''}")
        print(f"Model   : {CONFIG['models']['small']}")
        print(f"{'─' * 60}")
 
    # Build MCP config from yaml — only news_server for this agent
    mcp_config = build_mcp_config("news_agent")
 
    # Connect to MCP servers and load tools
    mcp_client = MultiServerMCPClient(mcp_config)
    tools      = await mcp_client.get_tools()
 
    if verbose:
        print(f"Tools loaded: {[t.name for t in tools]}\n")
 
    # Build the ReAct graph
    llm   = get_llm("small")
    graph, tracer  = build_react_agent(tools, NEWS_AGENT_SYSTEM_PROMPT, llm, agent_name="news_agent", run_id=run_id)
 
    # Run the agent
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": agent_cfg["recursion_limit"]},
    )

    # Print tool trace summary
    tracer.print_summary()
 
    final = result["messages"][-1].content
 
    if verbose:
        print(f"\n[NEWS AGENT] Complete")
        print(f"{'─' * 60}")
        print(final)
        print(f"{'─' * 60}\n")
 
    return final


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================
 
async def main():
    """Run the News Agent standalone using the active_query from config.yaml."""

    task = "What are the latest innovations in AI?"
    print("=" * 60)
    print("News Agent — Standalone Mode")
    print("=" * 60)
 
    await run_news_agent(task, verbose=True)
 
 
if __name__ == "__main__":
    asyncio.run(main())