"""
orchestrator.py  —  Orchestrator Agent (Supervisor Pattern)
============================================================
The top-level ReAct agent. Classifies incoming queries, plans the research
approach, delegates to specialist agents via tool calls, and synthesises
a final coherent answer.
 
Architecture (tool-calling supervisor — LangChain recommended 2025):
  The orchestrator treats each specialist agent as a tool.
  It calls delegate_to_news_agent() or delegate_to_analysis_agent()
  the same way any ReAct agent calls a tool.
 
  This is cleaner than the subgraph approach because:
  • Context flow is explicit and controllable
  • Each delegation is a single tool call with a clear task string
  • Easy to debug — you can see exactly what task was delegated
  • Easy to extend — add a new agent = add a new @tool function
 
Flow:
  User query
      ↓
  Orchestrator classifies: TYPE A → news_agent (no review)
                           TYPE B/C/D/E → analysis_agent → review_agent
                           Complex → both agents → review_agent
      ↓
  Orchestrator calls specialist delegation tools
      ↓
  For TYPE B/D/E: Review Agent validates the analysis report
      ↓
  If review passes  → orchestrator synthesises → Final Answer
  If review fails   → orchestrator retries or warns user with failures

 
Run:
  Terminal 1: python news_server.py
  Terminal 2: python orchestrator.py
"""

import sys
import asyncio
import json
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
 
from agents.common import CONFIG, get_llm, normalize_tool_messages, ReviewedReport
from agents.news_agent     import run_news_agent
from agents.analysis_agent import run_analysis_agent

# =============================================================================
# DELEGATION TOOLS
#
# Each specialist agent is wrapped as a LangChain @tool.
# The orchestrator calls these tools to delegate subtasks.
#
# run_id design: @tool functions cannot take extra parameters beyond what
# the LLM passes (only the declared task string). We use a module-level
# variable _current_run_id that the orchestrator sets once before the graph
# runs. Both delegation tools read it from this shared scope.
# This is safe for async use because each orchestrator run is sequential
# (it awaits each delegation before calling the next).
# =============================================================================
 
# Module-level run_id — set by main() before graph.ainvoke()
_current_run_id: str | None = None

@tool
async def delegate_to_news_agent(task: str) -> str:
    """
    Delegate a news research task to the News Agent.
 
    Use this for:
    - General market briefs and today's headlines
    - Sector news snapshots (IT, Banking, Energy, Pharma, Auto, FMCG)
    - Tech industry news (TechCrunch, Hacker News, The Verge)
    - AI industry news (VentureBeat, MIT Review, arXiv)
    - Any query that needs broad news coverage without financial analysis
 
    The News Agent connects to the news MCP server and fetches live RSS
    feeds from Economic Times, Moneycontrol, Reuters, TechCrunch, and more.
 
    Args:
        task: Clear description of what news to fetch and summarise.
              Be specific — include sector, company names, or time period.
              Example: "Fetch Indian market headlines and IT sector news for today"
    """
    
    print(f"\n  [ORCHESTRATOR → NEWS AGENT] Full task string:")
    print(f"  {chr(45)*55}")
    print(task)
    print(f"  {chr(45)*55}")
    result = await run_news_agent(task, verbose=True)
    return result
 
 
@tool
async def delegate_to_analysis_agent(task: str) -> str:
    """
    Delegate a financial analysis task to the Analysis Agent.
 
    The Analysis Agent runs tools, produces a report, sends it to the
    Review Agent internally, retries if needed, and returns a ReviewedReport.
    The orchestrator does NOT manage the review/retry loop — the agent does.
 
    Use this for:
    - Buy / Sell / Hold decisions for a specific stock
    - Sector selection recommendations
    - Deep company analysis (fundamentals, price history, valuation)
    - Macro impact analysis (how an event affects specific stocks)
 
    Returns:
        JSON string with review_passed, report, failure_reasons, recommendation.
        The orchestrator reads review_passed to decide how to frame the answer.
    """
    
    print(f"\n  [ORCHESTRATOR → ANALYSIS AGENT] Full task string:")
    print(f"  {chr(45)*55}")
    print(task)
    print(f"  {chr(45)*55}")
    # run_analysis_agent now returns a ReviewedReport (not a plain string)
    # It owns the review + retry loop internally
    reviewed = await run_analysis_agent(task, verbose=True, run_id=_current_run_id)
    reviewed.print_summary()
 
    # Return as JSON so the orchestrator LLM can read review_passed
    return json.dumps({
        "review_passed":   reviewed.review_passed,
        "report":          reviewed.report,
        "failure_reasons": reviewed.failure_reasons,
        "warning_reasons": reviewed.warning_reasons,
        "recommendation":  reviewed.recommendation,
        "retry_count":     reviewed.retry_count,
    })
 
 
ORCHESTRATOR_SYSTEM_PROMPT = """You are a financial research orchestrator. \
Your job is to understand the user's query, plan the best research approach, \
delegate to the right specialist agents, and synthesise their findings into \
a final coherent answer.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUERY CLASSIFICATION AND ROUTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
[TYPE A] Market briefs, headlines, sector snapshots, general news
  → delegate_to_news_agent only
  → No review needed — news briefs have lower data accuracy requirements
 
[TYPE A] Market briefs, headlines, sector snapshots, general news
  → delegate_to_news_agent only
  → No review — news briefs have lower data accuracy requirements
 
[TYPE B] Buy / Sell / Hold decision for a specific stock
  Step 1: delegate_to_analysis_agent
          The agent runs tools, reviews internally, retries if needed.
          Returns JSON with: review_passed, report, failure_reasons, recommendation.
  Step 2: If review_passed=true  → synthesise final answer from the report
          If review_passed=false → present report with DATA QUALITY WARNING header
 
[TYPE C] Sector selection — which sector or stocks to invest in
  Step 1: delegate_to_news_agent      (macro news, sector sentiment)
  Step 2: delegate_to_analysis_agent  (price trends, valuation — includes review)
  Step 3: Synthesise as per review_passed
 
[TYPE D] Deep company analysis — full research report on one company
  Step 1: delegate_to_analysis_agent  (comprehensive — includes internal review)
  Step 2: Optionally delegate_to_news_agent for broader market context
  Step 3: Synthesise as per review_passed
 
[TYPE E] Macro impact analysis — how an event affects stocks
  Step 1: delegate_to_news_agent      (news about the macro event)
  Step 2: delegate_to_analysis_agent  (price/fundamental impact — includes review)
  Step 3: Synthesise as per review_passed
 
HOW TO READ THE ANALYSIS AGENT RESULT:
  The analysis agent returns JSON. Read these fields:
    review_passed   : true → report is quality-validated
                      false → report had issues, present with warning
    report          : the full analysis report text to present
    failure_reasons : issues detected (empty if passed)
    recommendation  : BUY / SELL / HOLD extracted from report
    retry_count     : how many retries were needed (0 = first attempt passed)
 
  If review_passed=false, prepend this warning to the report:
    "⚠️ DATA QUALITY WARNING: Automated review detected issues: {failure_reasons}
     Please verify key figures independently before making investment decisions."
 
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DELEGATION GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE RULE: Your job is ROUTING, not research.
Do NOT inject facts, dates, names, or numbers into the task string.
The analysis agent will discover all facts via live tools.
If you inject stale facts (wrong split dates, old CEO names, old quarter
numbers), you pollute the agent's context and cause wrong output.
 
When delegating to analysis_agent, the task string must contain:
  1. Exact ticker symbol — derive from the user query
     (US stocks: NOW, AAPL, MSFT — NSE stocks: RELIANCE.NS, TCS.NS)
  2. The phrase "fetch LIVE data from Yahoo Finance tools"
  3. What categories of data to fetch (price, history, fundamentals, news)
  4. The investment question and time horizon from the user's query
 
WHAT TO INCLUDE in the task string:
  ✓ Ticker symbol
  ✓ "fetch LIVE data from Yahoo Finance tools"
  ✓ Data categories: current price, 52w range, 2yr history, P/E, ROE, FCF
  ✓ News categories: recent earnings, management, AI strategy, competitors
  ✓ The investment question: HOLD/SELL/BUY + horizon from user query
 
WHAT TO NEVER INCLUDE in the task string:
  ✗ Specific dates ("Q4 2024", "December 2025") — agent finds these via tools
  ✗ People's names ("Bill McDermott", "Gina Mastantuono") — may be outdated
  ✗ Split dates or split ratios — agent verifies via news search
  ✗ Historical prices or price targets — agent fetches these live
  ✗ Any fact you "know" about the company — your training data may be stale
 
BAD task string (injects stale facts):
  "Analyse NOW. Split was 5-for-1 in Dec 2024. CEO is Bill McDermott.
   Q4 2024 earnings showed 22% growth. Gina Mastantuono is CFO."
 
GOOD task string (routing only, no injected facts):
  "Fetch LIVE data from Yahoo Finance tools for ServiceNow (ticker: NOW).
   Get: current price, 52-week range, 2-year price history, P/E ratio,
   revenue growth, FCF margin, ROE, debt-to-equity.
   Search news for: recent quarterly earnings, AI strategy and adoption,
   competitor moves, management commentary, macro risks.
   Give a HOLD/SELL/BUY verdict for a 6-month horizon with conviction level."
 
For complex queries call both agents then merge their findings.
Do NOT redo work the specialist already did — trust their output.
 
SYNTHESIS — final answer rules
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After receiving specialist results, produce your final answer using
EXACTLY this template. Fill in each section. Do not deviate.
 
--- TEMPLATE START ---
 
**ORCHESTRATOR SYNTHESIS**
[One paragraph: verdict (BUY/SELL/HOLD), conviction level, 2-3 key reasons,
single most important risk. 4-6 sentences maximum. No questions.]
 
---
 
[Paste the COMPLETE specialist report here — every section, every table,
every citation, exactly as the specialist produced it. Do not shorten,
do not paraphrase, do not omit any section.]
 
---
 
⚠️ DISCLAIMER: For informational purposes only. Not financial advice.
Consult a SEBI-registered investment advisor before investing.
 
--- TEMPLATE END ---
 
ABSOLUTE PROHIBITIONS — the final answer must NOT contain:
  ✗ Any question mark at the end ("Would you like...?")
  ✗ Offers to provide more information ("I can expand on...")
  ✗ Invitations to continue the conversation ("Let me know if...")
  ✗ Any text after the DISCLAIMER line
  ✗ Prices that differ from what the specialist's tools returned
The answer is complete when the DISCLAIMER line is printed. Stop there.
"""

# =============================================================================
# ORCHESTRATOR GRAPH
# =============================================================================
 
def build_orchestrator() -> object:
    """
    Build the orchestrator as a LangGraph StateGraph.
 
    The orchestrator is itself a ReAct agent whose "tools" are the two
    specialist agent delegation functions. When it calls a delegation tool,
    LangGraph executes the full specialist agent loop and returns the result
    as a ToolMessage — exactly like any other tool call.
 
    Graph structure:
      START
        ↓
      [orchestrator node]   ← LLM call: classify query, plan, decide what to delegate
        ↓
      tools_condition()     ← routing: tool_calls? → "tools" : END
        ↓              ↓
      [tools node]     END  ← tools node executes the specialist agent
        ↓
      [orchestrator node]   ← LLM call: receive specialist results, synthesise
        ↓
      tools_condition()     ← no more tool_calls → END
        ↓
       END
    """
    llm   = get_llm("large")
    tools = [delegate_to_news_agent, delegate_to_analysis_agent]
 
    llm_with_tools = llm.bind_tools(tools)
 
    def orchestrator_node(state: MessagesState) -> dict:
        system   = SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)
        messages = [system] + normalize_tool_messages(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
 
    builder = StateGraph(MessagesState)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("tools", ToolNode(tools))
 
    builder.add_edge(START, "orchestrator")
 
    builder.add_conditional_edges("orchestrator", tools_condition)
 
    builder.add_edge("tools", "orchestrator")  # after delegation, synthesise
 
    return builder.compile()

# =============================================================================
# MAIN
# =============================================================================
 
async def main():
    task = "Please do the analysis and suggest if I should sell or Hold Maruti Suzuki share?"
    
    orch_cfg = CONFIG["agents"]["orchestrator"]
 
    print("=" * 65)
    print("=" * 65)
    print(f"\nQUERY: {task}\n")
    print("─" * 65)
    print("Prerequisites:")
    print("  Terminal 1 → python news_server.py  (must be running)")
    print("─" * 65 + "\n")
 
    # Generate a unique run_id for this orchestrator run.
    # This ID is shared across all agent delegations so every tool call
    # in every specialist agent can be correlated back to this single run.
    global _current_run_id
    _current_run_id = uuid.uuid4().hex[:8]
    print(f"  run_id: {_current_run_id}  (shared across all agent delegations)\n")

    graph = build_orchestrator()
 
    final_state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": orch_cfg["recursion_limit"]},
    )
 
    final = final_state["messages"][-1].content
 
    print("\n" + "=" * 65)
    print("FINAL ANSWER")
    print("=" * 65)
    print(final)
    print("=" * 65)
 
 
if __name__ == "__main__":
    asyncio.run(main())