
"""
agents/common.py  —  Shared utilities for all agents
=====================================================
Loaded by news_agent.py, analysis_agent.py, and orchestrator.py.
Contains:
  • Config loader (reads config.yaml)
  • MCP server config builder
  • ToolMessage normaliser (Mistral compatibility fix)
  • ReAct agent builder (standard pattern reused by all agents)
"""
 
import json
from dotenv import load_dotenv, find_dotenv
import os
import shutil
from pathlib import Path
from pydantic import BaseModel, Field
 
import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from agents.tracer import ToolTracer, TracedToolNode
 
# Load environment variables
load_dotenv(find_dotenv())

 
# =============================================================================
# CONFIG LOADER
# =============================================================================
 
def load_config(config_path: str) -> dict:
    """
    Load config.yaml from the project root.
    Override path via CONFIG_PATH env var or explicit argument.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
 
 
# Loaded once at import time — shared across all agents in the process
CONFIG = load_config("agent_config.yaml")

# =============================================================================
# LLM FACTORY
# =============================================================================
 
def get_llm(size: str = "small"):
    """
    Return an LLM instance for the given size ('large' or 'small').
    Provider, model names, and temperature come from agent_config.yaml.

    Supported providers (set via models.provider in agent_config.yaml):
      'mistral'  — uses ChatMistralAI, requires MISTRAL_API_KEY env var
      'gemini'   — uses ChatGoogleGenerativeAI, requires GOOGLE_API_KEY env var
      'deepseek' — uses ChatOpenAI (OpenAI-compat), requires DEEPSEEK_API_KEY env var
      'groq'     — uses ChatGroq (free tier), requires GROQ_API_KEY env var
    """
    models_cfg = CONFIG["models"]
    provider   = models_cfg.get("provider", "mistral").lower()
    model      = models_cfg.get(size, models_cfg["small"])
    temperature = models_cfg.get("temperature", 0)

    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "MISTRAL_API_KEY not set.\n"
                "export MISTRAL_API_KEY=your_key_here"
            )
        return ChatMistralAI(
            model       = model,
            api_key     = api_key,
            temperature = temperature,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not set.\n"
                "export GOOGLE_API_KEY=your_key_here"
            )
        return ChatGoogleGenerativeAI(
            model          = model,
            google_api_key = api_key,
            temperature    = temperature,
        )

    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "DEEPSEEK_API_KEY not set.\n"
                "export DEEPSEEK_API_KEY=your_key_here"
            )
        return ChatOpenAI(
            model       = model,
            api_key     = api_key,
            base_url    = "https://api.deepseek.com",
            temperature = temperature,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set.\n"
                "export GROQ_API_KEY=your_key_here"
            )
        return ChatGroq(
            model       = model,
            api_key     = api_key,
            temperature = temperature,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}' in agent_config.yaml.\n"
            "Supported values: 'mistral', 'gemini', 'deepseek', 'groq'"
        )

# =============================================================================
# MCP SERVER CONFIG BUILDER
# =============================================================================
 
def build_mcp_config(agent_name: str) -> dict:
    """
    Build a MultiServerMCPClient config dict for a given agent.
    Reads which servers the agent should connect to from config.yaml
    (agent_servers section), then builds the transport config for each.
 
    Args:
        agent_name: Key in config.yaml agent_servers (e.g. "news_agent")
 
    Returns:
        Dict ready to pass to MultiServerMCPClient(config)
    """
    all_servers    = CONFIG["mcp_servers"]
    agent_server_keys = CONFIG["agent_servers"].get(agent_name, [])
    mcp_config     = {}
 
    for key in agent_server_keys:
        server = all_servers.get(key)
        if not server:
            raise ValueError(f"Server '{key}' not found in config.yaml mcp_servers")
        if not server.get("enabled", True):
            print(f"  [config] Server '{key}' is disabled — skipping")
            continue
 
        transport = server["transport"]
 
        if transport == "http":
            mcp_config[key] = {
                "transport": "http",
                "url":        server["url"],
            }
 
        elif transport == "stdio":
            cmd = shutil.which(server["command"])
            if not cmd:
                raise RuntimeError(
                    f"Command '{server['command']}' not found in PATH.\n"
                    f"Install with: pip install {server['command']}"
                )
            mcp_config[key] = {
                "transport": "stdio",
                "command":    cmd,
                "args":       server.get("args", []),
            }
 
        else:
            raise ValueError(f"Unknown transport '{transport}' for server '{key}'")
 
    return mcp_config

# =============================================================================
# TOOL MESSAGE NORMALISER
# =============================================================================
 
def normalize_tool_messages(messages: list) -> list:
    """
    Mistral API requires tool message content to be a plain string.
    langchain-mcp-adapters returns content as a list of content blocks:
      [{"type": "text", "text": "...", "id": "lc_..."}]
 
    This flattens that structure to a plain string before every LLM call.
    Applied inside agent_node() of every agent.
 
    Root cause: MCP tools use response_format='content_and_artifact' which
    produces structured content that Mistral's /v1/chat/completions rejects
    with a 422 error.
    """
    normalized = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and isinstance(msg.content, list):
            text = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in msg.content
            )
            msg = ToolMessage(
                content      = text,
                tool_call_id = msg.tool_call_id,
                name         = msg.name,
            )
        normalized.append(msg)
    return normalized

# =============================================================================
# REACT AGENT BUILDER
# =============================================================================
 
def build_react_agent(
    tools: list,
    system_prompt: str,
    llm: BaseChatModel,
    agent_name: str = "agent",
    run_id: str | None = None,
) -> tuple:
    """
    Build a standard ReAct agent as a compiled LangGraph StateGraph.
    Returns (compiled_graph, tracer) — call tracer.print_summary() after
    ainvoke() to see the full tool call trace.
 
    Args:
        run_id: Optional parent run ID from the orchestrator. When provided,
                all tool call records include this ID so you can correlate
                all agents in one orchestrator run. None for standalone runs.
 
    Pattern:
      START → agent_node ──tools_condition──► TracedToolNode → agent_node
                         └──────────────────► END
 
    The agent_node:
      1. Prepends system_prompt as SystemMessage
      2. Normalises ToolMessages for Mistral compatibility
      3. Calls the LLM with bound tools
      4. Returns the AI response to be appended to state
 
    The TracedToolNode (replaces plain ToolNode):
      Executes tool calls exactly as ToolNode does, but additionally:
      - Records tool name, arguments, result preview, and timing
      - Stores records in ToolTracer for post-run summary printing
      - handle_tool_errors=True — errors become ToolMessages, not crashes
 
    Args:
        tools:         List of LangChain BaseTool objects (from MCP or @tool)
        system_prompt: The agent's persona and behavioural instructions
        llm:           ChatMistralAI instance
        agent_name:    Label used in trace output (e.g. "analysis_agent")
 
    Returns:
        (compiled_graph, tracer) tuple
    """
    tracer         = ToolTracer(agent_name=agent_name, run_id=run_id)
    llm_with_tools = llm.bind_tools(tools)
 
    # Print IDs at agent start so real-time [TOOL CALL] lines are correlatable
    run_label = f"run_id: {run_id}  |  " if run_id else ""
    print(f"  [{agent_name}] {run_label}trace_id: {tracer.trace_id}")
 
    def agent_node(state: MessagesState) -> dict:
        system   = SystemMessage(content=system_prompt)
        messages = [system] + normalize_tool_messages(state["messages"])
        response = llm_with_tools.invoke(messages)
 
        # Log each tool call in real-time with trace_id for log correlation
        # Format: [trace_id][agent_name][#N] tool_name(args)
        if hasattr(response, "tool_calls") and response.tool_calls:
            call_num = len(tracer.calls) + 1
            for tc in response.tool_calls:
                args_preview = str(tc.get("args", {}))[:80]
                run_prefix = f"[{run_id}]" if run_id else ""
                print(
                    f"  {run_prefix}[{tracer.trace_id}][{agent_name}]"
                    f"[#{call_num}] {tc['name']}({args_preview})"
                )
                call_num += 1
 
        return {"messages": [response]}
 
    traced_tool_node = TracedToolNode(
        tools,
        tracer          = tracer,
        handle_tool_errors = True,
    )
 
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", traced_tool_node)
 
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
 
    return builder.compile(), tracer


# =============================================================================
# REVIEWED REPORT MODEL
# =============================================================================
 
class ReviewedReport(BaseModel):
    """
    Structured return value from run_analysis_agent().
 
    The Analysis Agent owns the full produce → review → retry loop internally.
    When it returns to the orchestrator, it returns this model — not a plain string.
    The orchestrator reads review_passed to decide how to frame the final answer,
    without being involved in the retry loop at all.
 
    This is the key design principle:
      The agent that produced bad output owns the retry — not the dispatcher.
 
    Fields:
        report         : The final analysis report text (after review/retry).
        review_passed  : True if review passed on final attempt.
        review_checks  : Dict of check_name → bool from the last ReviewResult.
        failure_reasons: Reasons from the last review (empty if passed).
        warning_reasons: Non-blocking warnings from the last review.
        retry_count    : How many retry attempts were made (0 = first attempt passed).
        recommendation : BUY / SELL / HOLD extracted by review agent.
    """
    report:          str            = Field(description="Final analysis report text")
    review_passed:   bool           = Field(description="True if review passed on final attempt")
    review_checks:   dict[str, bool]= Field(default_factory=dict, description="Check results from last review")
    failure_reasons: list[str]      = Field(default_factory=list, description="Failure reasons from last review")
    warning_reasons: list[str]      = Field(default_factory=list, description="Non-blocking warnings")
    retry_count:     int            = Field(default=0, ge=0, description="Number of retry attempts made")
    recommendation:  str            = Field(default="NOT FOUND", description="BUY/SELL/HOLD from review agent")
 
    def print_summary(self) -> None:
        """Print a one-line summary of the review outcome."""
        status = "✓ PASSED" if self.review_passed else "✗ FAILED (presenting with warning)"
        print(f"  [REVIEWED REPORT] {status}  |  retries: {self.retry_count}  |  verdict: {self.recommendation}")
        if self.failure_reasons:
            for r in self.failure_reasons:
                print(f"    ⚠ {r}")