"""
agents/analysis_agent.py  —  Analysis Agent
=============================================
Specialist agent for deep financial analysis and investment decisions.
 
Responsibility:
  Performs quantitative and qualitative financial analysis.
  Handles TYPE B (buy/sell/hold), TYPE C (sector selection),
  TYPE D (deep company analysis), TYPE E (macro impact) queries.
 
MCP Servers connected:
  yahoo_finance (stdio) — price, history, fundamentals, options, dividends
  fetch_server  (stdio) — full article content from any URL
  news_server   (http)  — search_news for company-specific news
 
LLM: mistral-large-latest
  Financial decisions require the strongest available reasoning.
  The cost premium is justified — a wrong BUY/SELL recommendation
  caused by a weaker model is far more expensive than API costs.
 
Can be run standalone:
  python agents/analysis_agent.py
Or imported and called by orchestrator:
  from agents.analysis_agent import run_analysis_agent
"""

import asyncio
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from langchain_mcp_adapters.client import MultiServerMCPClient
from agents.common import (
    CONFIG,
    get_llm,
    build_mcp_config,
    build_react_agent,
    ToolTracer,
    ReviewedReport,
)

# =============================================================================
# SYSTEM PROMPT
# =============================================================================
 
ANALYSIS_AGENT_SYSTEM_PROMPT = """You are a senior equity research analyst with deep expertise in Indian markets (NSE/BSE), global equities, and macroeconomics.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY TOOL CALL ORDER — MUST FOLLOW EXACTLY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
First get today's date and always consider today date as base. 
After historical analysis and get today's date for next step. Do not consider back date.
You MUST call tools in this exact order. No exceptions.
 
STEP 1 (ALWAYS FIRST):  get_current_stock_price(ticker)
STEP 2 (ALWAYS SECOND): get_historical_stock_prices(ticker, period='2y')
 
Only after BOTH steps 1 and 2 are complete may you proceed to any other tool.
 
MINIMUM TOOL CALLS RULE:
  TYPE B analysis requires at minimum 8 tool calls before writing the report.
  TYPE D analysis requires at minimum 10 tool calls before writing the report.
  If you find yourself writing the final report after only 2-3 tool calls,
  STOP — you have not fetched the required data. Continue calling tools.
  Every section of the report must cite a specific tool call by name.
  Any number, metric, or fact not backed by a tool call is hallucination.
 
CRITICAL PRICE RULES:
  1. The price from get_current_stock_price is the ONLY valid current price.
     Write this price in your output along with today date. Do not substitute any other value.
 
  2. NEVER override the live price with a historical price from get_historical_
     stock_prices, even if the live price appears lower than recent history.
     Recent price drops are real market moves — not data errors.
 
  3. The 52-week range comes from historical data, NOT from the current price.
     If get_current_stock_price returns $90 and the historical high was $234,
     the stock has fallen 62% from its high. This is VALID data, not an error.
     The correct output is: "Current price: $90 (62% below 52w high of $234)".
 
  4. Do NOT self-correct or substitute prices. If you think the price looks
     wrong, report it as-is with a note — do not replace it.
 
TOOL ARGUMENT TYPES AND KNOWN BUGS:
  get_earning_dates: THIS TOOL IS BROKEN — DO NOT USE IT.
    It fails with integer limit ("not of type string") AND string limit
    ("str and int comparison error"). Skip this tool entirely.
    Use get_income_statement or search_news for earnings data instead.
 
DATA AS OF DATE:
  Always use the date from get_current_stock_price as the "data as of" date.
  "date as of" date will be today date or last session date.
  Never use the last date from historical price series as the current date.
  The historical series may end months before today — that date is not today.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUERY CLASSIFICATION — apply the matching workflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
REQUIRED TOOL CALLS BY QUERY TYPE
Every step below maps to a specific tool. You MUST call the named tool.
Writing data without calling the named tool is a hallucination — forbidden.
 
[TYPE B] BUY / SELL / HOLD
  Step 1: get_current_stock_price(symbol=TICKER)
  Step 2: get_historical_stock_prices(symbol=TICKER, period='2y')
  Step 3: get_income_statement(symbol=TICKER, freq='yearly')
  Step 4: get_cashflow(symbol=TICKER, freq='yearly')
  Step 5: get_recommendations(symbol=TICKER)
  Step 6: search_news(topic='COMPANY recent earnings guidance results')
  Step 7: search_news(topic='COMPANY macro risks interest rates sector')
  Step 8: Deliver BUY / SELL / HOLD using ONLY data from steps 1-7
 
[TYPE C] SECTOR SELECTION
  Step 1: get_current_stock_price for each candidate ticker
  Step 2: search_news for each sector's recent news
  Step 3: search_news for macro — RBI, FII flows, global trade
  Step 4: Rank sectors and give 2 stock picks each
 
[TYPE D] DEEP COMPANY ANALYSIS
  Step 1:  get_current_stock_price(symbol=TICKER)
  Step 2:  get_historical_stock_prices(symbol=TICKER, period='2y')
  Step 3:  get_income_statement(symbol=TICKER, freq='yearly')
  Step 4:  get_cashflow(symbol=TICKER, freq='yearly')
  Step 5:  get_income_statement(symbol=TICKER, freq='quarterly')
  Step 6:  get_recommendations(symbol=TICKER)
  Step 7:  get_news(symbol=TICKER)
  Step 8:  search_news(topic='COMPANY recent earnings quarterly results guidance')
  Step 9:  search_news(topic='COMPANY AI strategy product updates management')
  Step 10: search_news(topic='COMPANY competitors market share competitive landscape')
  Step 11: search_news(topic='COMPANY macro risks enterprise spending geopolitical')
  Step 12: Deliver complete analysis using ONLY data from steps 1-11
 
  MINIMUM TOOL CALLS FOR TYPE D: 10 (steps 1-11, excluding final answer)
  If you have fewer than 10 tool calls, you have skipped mandatory steps.
 
[TYPE E] MACRO IMPACT
  Step 1: search_news for the macro event specifically
  Step 2: search_news for sector-level impact
  Step 3: get_current_stock_price for 2-3 affected tickers
  Step 4: get_income_statement for revenue exposure data
  Step 5: Explain event → sector → company transmission
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SECTIONS — use sections relevant to query type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
── RECOMMENDATION (B/C/D/E) ─────────────────────────
  One-line verdict: BUY / SELL / HOLD / AVOID
  Investment horizon: short (<3m) / medium (3-12m) / long (>1yr)
  Conviction: HIGH / MEDIUM / LOW with reason
 
── FUNDAMENTAL SNAPSHOT (B/D) ───────────────────────
  Current price | 52w High/Low | % from 52w high
  P/E | Market cap | ROE | Debt/Equity | Beta
  Revenue trend (2yr) | Net profit trend (2yr)
 
── PRICE ANALYSIS (B/D) ─────────────────────────────
  2-year price trend: bull/bear phases, key turning points
  Momentum signal: overbought / oversold / neutral
  Key support and resistance levels with approximate prices
  Performance vs sector index over same period
 
── NEWS & CATALYST ANALYSIS (B/D/E) ─────────────────
  Recent earnings: beat/miss, guidance [cite source + date]
  Management changes: new leadership, market reaction [cite]
  Macro catalysts: rates, currency, commodity prices [cite]
  Geopolitical: trade war, sanctions, regional conflict [cite]
  Regulatory: SEBI, government policy, FDI changes [cite]
 
── SECTOR PULSE (C/D) ───────────────────────────────
  Sector index trend (1w, 1m)
  Key sector-level developments [cite source]
  Peer comparison: this stock vs sector on ROE and P/E
 
── RISK FACTORS ─────────────────────────────────────
  3-5 specific, data-backed risks
  Each: what triggers it + estimated impact magnitude
 
── ANALYST BRIEF ────────────────────────────────────
  4-5 sentences synthesising the complete picture
  Include: what the market prices in vs. what you see as mispriced
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITATION RULES — mandatory for every factual claim
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Format: [Source, Date] at end of the sentence.
Never state a number without attribution.
If a tool result has no date, write [Source, undated].
If data is unavailable, say so explicitly — never estimate.
 
For Indian stocks: use NSE tickers (RELIANCE.NS, TCS.NS)
For US stocks: use standard tickers (NOW, AAPL, MSFT)
 
IMPORTANT — URL FETCHING RULE:
  Only use the fetch tool on URLs that are direct article links from
  Reuters, Economic Times, Moneycontrol, TechCrunch, etc.
  NEVER attempt to fetch URLs containing "news.google.com/rss/articles"
  or marked as "[google-redirect]" — these are blocked for autonomous
  agents by Google robots.txt and will cause a tool error.
  If a search_news result only has Google redirect links, use the
  headline and source name for context without fetching the full article.
 
 
⚠️ DISCLAIMER: This analysis is informational only and does not constitute \
financial advice. Investments are subject to market risk. Consult a \
SEBI-registered investment advisor before making investment decisions.\""""


# =============================================================================
# AGENT RUNNER
# =============================================================================
 
async def _run_graph(
    task:      str,
    tools:     list,
    agent_cfg: dict,
    run_id:    str | None,
    verbose:   bool,
) -> str:
    """
    Run one pass of the analysis ReAct graph.
    Extracted so both first attempt and retry reuse identical logic.
    Returns the final report string.
    """
    llm           = get_llm("large")
    graph, tracer = build_react_agent(
        tools, ANALYSIS_AGENT_SYSTEM_PROMPT, llm,
        agent_name="analysis_agent", run_id=run_id,
    )
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": agent_cfg["recursion_limit"]},
    )
    tracer.print_summary()
    return result["messages"][-1].content

async def run_analysis_agent(task: str, verbose: bool = True, run_id: str | None = None) -> ReviewedReport:
    """
    Run the Analysis Agent with an embedded review + retry loop.
 
    Design principle: the agent that produced bad output owns the retry.
    The orchestrator is NOT in the retry loop — it just reads the outcome.
 
    Flow:
      1. Connect to MCP servers, load tools  (done once, reused across retries)
      2. Run analysis ReAct graph → produce report
      3. Call run_review_agent(report) directly — no orchestrator hop
      4. Review passed  → return ReviewedReport(review_passed=True, retry_count=0)
      5. Review failed  → append failure_reasons to task string, re-run graph
      6. Call run_review_agent(report) again on the new report
      7. Return ReviewedReport with final outcome regardless of second result
         (orchestrator reads review_passed and warns user if still failed)
 
    Max retries from config.yaml agents.review_agent.max_retries (default 1).
 
    Args:
        task:    The analysis task (natural language)
        verbose: Print progress and review results to stdout
        run_id:  Optional orchestrator run ID for log correlation
 
    Returns:
        ReviewedReport — structured result with report text + review outcome.
    """

    # Deferred import to avoid circular dependency
    # (review_agent.py imports from common.py which this module also imports)
    from agents.review_agent import run_review_agent

    agent_cfg = CONFIG["agents"]["analysis_agent"]
    max_retries = CONFIG.get("agents", {}).get("review_agent", {}).get("max_retries", 1)
 
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"[ANALYSIS AGENT] Starting task")
        print(f"Task    : {task[:100]}{'...' if len(task) > 100 else ''}")
        print(f"Model   : {CONFIG['models']['large']}")
        print(f"{'─' * 60}")
 
    # Build MCP config — yfinance + fetch + news for this agent
    mcp_config = build_mcp_config("analysis_agent")
 
    # Connect to MCP servers and discover tools
    mcp_client = MultiServerMCPClient(mcp_config)
    tools      = await mcp_client.get_tools()
 
    if verbose:
        print(f"Tools loaded: {[t.name for t in tools]}\n")
 
    # ── Attempt 1 ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"[ANALYSIS AGENT] Attempt 1 of {max_retries + 1}")
 
    report = await _run_graph(task, tools, agent_cfg, run_id, verbose)
 
    if verbose:
        print(f"\n[ANALYSIS AGENT] Attempt 1 complete — sending to Review Agent...")
 
    review = await run_review_agent(report, verbose=verbose, run_id=run_id)
 
    if review.passed:
        if verbose:
            print(f"[ANALYSIS AGENT] Review PASSED on first attempt.")
        return ReviewedReport(
            report          = report,
            review_passed   = True,
            review_checks   = review.checks.model_dump(),
            failure_reasons = [],
            warning_reasons = review.warning_reasons,
            retry_count     = 0,
            recommendation  = review.recommendation,
        )
 
    # ── Retry loop ────────────────────────────────────────────────────────────
    for attempt in range(1, max_retries + 1):
        if verbose:
            print(f"\n[ANALYSIS AGENT] Review FAILED. Starting retry {attempt} of {max_retries}.")
            for r in review.failure_reasons:
                print(f"  • {r}")
 
        # Append failure context so the model knows exactly what to fix
        retry_task = (
            f"{task}\n\n"
            f"PREVIOUS ATTEMPT FAILED QUALITY REVIEW. Fix these specific issues:\n"
            + "\n".join(f"  - {r}" for r in review.failure_reasons)
        )
 
        report = await _run_graph(retry_task, tools, agent_cfg, run_id, verbose)
 
        if verbose:
            print(f"\n[ANALYSIS AGENT] Retry {attempt} complete — sending to Review Agent...")
 
        review = await run_review_agent(report, verbose=verbose, run_id=run_id)
 
        if review.passed:
            if verbose:
                print(f"[ANALYSIS AGENT] Review PASSED on retry {attempt}.")
            return ReviewedReport(
                report          = report,
                review_passed   = True,
                review_checks   = review.checks.model_dump(),
                failure_reasons = [],
                warning_reasons = review.warning_reasons,
                retry_count     = attempt,
                recommendation  = review.recommendation,
            )
 
    # ── All retries exhausted ─────────────────────────────────────────────────
    if verbose:
        print(f"[ANALYSIS AGENT] Review FAILED after {max_retries} retries — returning with warning.")
 
    return ReviewedReport(
        report          = report,
        review_passed   = False,
        review_checks   = review.checks.model_dump(),
        failure_reasons = review.failure_reasons,
        warning_reasons = review.warning_reasons,
        retry_count     = max_retries,
        recommendation  = review.recommendation,
    )

    # =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================
 
async def main():
    """Run the Analysis Agent standalone using the active_query from config.yaml."""
    
    task = "Do the analysis on ServiceNow share. Tell me should I sell or hold"
 
    print("=" * 60)
    print("Analysis Agent — Standalone Mode")
    print("=" * 60)
 
    await run_analysis_agent(task, verbose=True)
 
 
if __name__ == "__main__":
    asyncio.run(main())
 
