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
  Orchestrator classifies: TYPE A → news_agent
                           TYPE B/C/D/E → analysis_agent
                           Complex → both agents in sequence
      ↓
  Orchestrator calls delegate_to_news_agent(task="...")
  and/or delegate_to_analysis_agent(task="...")
      ↓
  Each specialist agent runs its own full ReAct loop internally
      ↓
  Orchestrator receives results as tool outputs
  Orchestrator synthesises → Final Answer
 
Run:
  Terminal 1: python news_server.py
  Terminal 2: python orchestrator.py
"""

import asyncio
import sys
from pathlib import Path

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
 
from agents.common import CONFIG, get_llm, normalize_tool_messages
from agents.news_agent     import run_news_agent
from agents.analysis_agent import run_analysis_agent

# =============================================================================
# DELEGATION TOOLS
#
# Each specialist agent is wrapped as a LangChain @tool.
# The orchestrator calls these tools to delegate subtasks.
#
# IMPORTANT: These tools are async and call the full agent ReAct loop.
# The tool description is what the orchestrator reads to decide which
# agent to invoke — write it clearly.
# =============================================================================
 
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
 
    Use this for:
    - Buy / Sell / Hold decisions for a specific stock
    - Sector selection recommendations
    - Deep company analysis (fundamentals, price history, valuation)
    - Macro impact analysis (how an event affects specific stocks)
    - Portfolio assessment
    - Any query requiring price data, financial ratios, or investment recommendations
 
    The Analysis Agent connects to Yahoo Finance (live price/fundamental data),
    a web fetch tool (full article content), and the news server (company search).
 
    Args:
        task: Detailed analysis task. Include: stock ticker(s), time horizon,
              investor profile if relevant, and specific questions to answer.
              Example: "Analyse TCS.NS — 2yr price history, fundamentals,
                        Q4 results, and give a 12-month investment verdict"
    """
    
    print(f"\n  [ORCHESTRATOR → ANALYSIS AGENT] Full task string:")
    print(f"  {chr(45)*55}")
    print(task)
    print(f"  {chr(45)*55}")
    result = await run_analysis_agent(task, verbose=True)
    return result
 
 
ORCHESTRATOR_SYSTEM_PROMPT = """You are a financial research orchestrator. \
Your job is to understand the user's query, plan the best research approach, \
delegate to the right specialist agents, and synthesise their findings into \
a final coherent answer.
 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUERY CLASSIFICATION AND ROUTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
[TYPE A] Market briefs, headlines, sector snapshots, general news
  → delegate_to_news_agent only
  → One delegation call is sufficient
 
[TYPE B] Buy / Sell / Hold decision for a specific stock
  → delegate_to_analysis_agent (primary — price data + fundamentals + news)
  → Optionally delegate_to_news_agent if broader market context is needed
 
[TYPE C] Sector selection — which sector or stocks to invest in
  → delegate_to_news_agent first (macro news, sector sentiment)
  → delegate_to_analysis_agent second (price trends, valuation comparison)
 
[TYPE D] Deep company analysis — full research report on one company
  → delegate_to_analysis_agent (comprehensive — it fetches everything needed)
  → May optionally delegate_to_news_agent for broader market context
 
[TYPE E] Macro impact analysis — how an event affects stocks
  → delegate_to_news_agent first (news about the event itself)
  → delegate_to_analysis_agent (price/fundamental impact on specific stocks)
 
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
After receiving specialist results:
  - Lead with the most important finding or recommendation
  - Integrate news context with financial analysis coherently
  - Add your own synthesis — do not just concatenate agent outputs
  - Maintain all source citations from specialist outputs
 
SYNTHESIS — final answer rules
After receiving specialist results, produce your final answer using
EXACTLY this template. Fill in each section. Do not deviate.

--- TEMPLATE START ---

**ORCHESTRATOR SYNTHESIS**
[One paragraph: verdict, conviction, 2-3 key reasons, single most important risk.]

---
[Paste the COMPLETE specialist report here — every section, every table,
every citation, exactly as the specialist produced it.]
---

⚠️ DISCLAIMER: ...

--- TEMPLATE END ---

ABSOLUTE PROHIBITIONS — the final answer must NOT contain:
  ✗ Any question mark at the end ("Would you like...?")
  ✗ Offers to provide more information ("I can expand on...")
  ✗ Invitations to continue ("Let me know if...")
  ✗ Any text after the DISCLAIMER line
 
End every response with:
⚠️ DISCLAIMER: For informational purposes only. Not financial advice. Consult a SEBI-registered investment advisor before investing.\""""


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
    task = "I need a detail analysis on Maruti Suzuki (MARUTI), also let me know if I should hold or sell"
    
    orch_cfg = CONFIG["agents"]["orchestrator"]
 
    print("=" * 65)
    print("=" * 65)
    print(f"\nQUERY: {task}\n")
    print("─" * 65)
    print("Prerequisites:")
    print("  Terminal 1 → python news_server.py  (must be running)")
    print("─" * 65 + "\n")
 
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