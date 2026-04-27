
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
 
import yaml
from langchain_mistralai import ChatMistralAI
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
 
def get_llm(size: str = "small") -> ChatMistralAI:
    """
    Return a ChatMistralAI instance for the given size ('large' or 'small').
    Model names and temperature come from config.yaml.
    """
    api_key    = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "MISTRAL_API_KEY not set.\n"
            "export MISTRAL_API_KEY=your_key_here"
        )
 
    models = CONFIG["models"]
    model  = models.get(size, models["small"])
 
    return ChatMistralAI(
        model       = model,
        api_key     = api_key,
        temperature = models.get("temperature", 0),
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
    llm: ChatMistralAI,
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