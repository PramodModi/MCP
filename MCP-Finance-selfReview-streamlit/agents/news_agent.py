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
- If a topic has no news in the last 7 days, say so explicitly
- Flag if the same story appears across multiple sources (indicates high impact)
 
End with: "Data from public RSS feeds and APIs. Not financial advice.\""""


# =============================================================================
# AGENT RUNNER
# =============================================================================

async def _run_news_graph(
    task:      str,
    tools:     list,
    agent_cfg: dict,
    run_id:    str | None,
    verbose:   bool,
) -> str:
    """
    Run one pass of the news ReAct graph.
    Extracted so both the first attempt and any retry reuse identical logic.
    Returns the final report string.
    """
    llm           = get_llm("small")
    graph, tracer = build_react_agent(
        tools, NEWS_AGENT_SYSTEM_PROMPT, llm,
        agent_name="news_agent", run_id=run_id,
    )
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": agent_cfg["recursion_limit"]},
    )
    tracer.print_summary()
    return result["messages"][-1].content


async def run_news_agent(task: str, verbose: bool = True, run_id: str | None = None) -> str:
    """
    Run the News Agent with an embedded review + one retry.

    Flow:
      1. Connect to MCP servers, load tools (once — reused on retry)
      2. Run news ReAct graph → produce report
      3. Call run_news_review_agent(report, task) — checks freshness, format, citations
      4. Review passed  → return report immediately
      5. Review failed  → append failure context to task, re-run graph once
      6. Return final report regardless of retry outcome

    Args:
        task:    The news research task (natural language)
        verbose: Print tool calls, review results, and iteration info to stdout
        run_id:  Optional orchestrator run ID for log correlation

    Returns:
        Final report string from the agent (reviewed, possibly retried)
    """
    from agents.review_agent import run_news_review_agent

    agent_cfg = CONFIG["agents"]["news_agent"]

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"[NEWS AGENT] Starting task")
        print(f"Task    : {task[:100]}{'...' if len(task) > 100 else ''}")
        print(f"Model   : {CONFIG['models']['small']}")
        print(f"{'─' * 60}")

    # Build MCP config from yaml — only news_server for this agent
    mcp_config = build_mcp_config("news_agent")

    # Connect to MCP servers and load tools (done once, reused on retry)
    mcp_client = MultiServerMCPClient(mcp_config)
    tools      = await mcp_client.get_tools()

    if verbose:
        print(f"Tools loaded: {[t.name for t in tools]}\n")

    # ── Attempt 1 ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"[NEWS AGENT] Attempt 1")

    report = await _run_news_graph(task, tools, agent_cfg, run_id, verbose)

    if verbose:
        print(f"\n[NEWS AGENT] Attempt 1 complete — sending to News Review Agent...")

    review = await run_news_review_agent(report, task, verbose=verbose, run_id=run_id)

    if review.passed:
        if verbose:
            print(f"[NEWS AGENT] Review PASSED on first attempt.")
    else:
        # ── Retry once with failure context ───────────────────────────────────
        if verbose:
            print(f"\n[NEWS AGENT] Review FAILED. Starting retry.")
            for r in review.failure_reasons:
                print(f"  • {r}")

        retry_task = (
            f"{task}\n\n"
            f"PREVIOUS ATTEMPT FAILED QUALITY REVIEW. Fix these specific issues:\n"
            + "\n".join(f"  - {r}" for r in review.failure_reasons)
            + "\n\nIMPORTANT: Only include news articles from the last 7 days. "
            "If a source has no articles from the past week, omit it or explicitly note "
            "'No recent news from [source]' instead of showing older articles."
        )

        report = await _run_news_graph(retry_task, tools, agent_cfg, run_id, verbose)

        if verbose:
            print(f"\n[NEWS AGENT] Retry complete.")

    if verbose:
        print(f"\n[NEWS AGENT] Complete")
        print(f"{'─' * 60}")
        print(report)
        print(f"{'─' * 60}\n")

    return report


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