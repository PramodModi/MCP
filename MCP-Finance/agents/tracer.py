"""
agents/tracer.py  —  Tool Call Tracing
=======================================
Standalone module. Zero dependency on config, LLM, or MCP.
Can be imported independently for testing or reuse in other projects.
 
Exports:
  ToolTracer      — records tool calls with run_id / trace_id hierarchy
  TracedToolNode  — LangGraph ToolNode subclass that feeds into ToolTracer
 
Two-level ID hierarchy:
  run_id    — parent ID, shared across all agents in one orchestrator run.
              Generated once in orchestrator.py and passed down to every
              agent. Allows you to group all tool calls from one request.
              None when an agent runs standalone (no orchestrator).
 
  trace_id  — child ID, unique per agent run.
              Generated fresh by ToolTracer.__init__ regardless of run_id.
              Identifies which specific agent made which tool calls.
 
Example output (orchestrator run with two agents):
 
  run_id: f8a2d301  (shared across all agent delegations)
 
  [news_agent]     run_id: f8a2d301  |  trace_id: a3f7c21b
  [f8a2d301][a3f7c21b][news_agent][#1] get_india_business_news({})
 
  [analysis_agent] run_id: f8a2d301  |  trace_id: d91e4f02
  [f8a2d301][d91e4f02][analysis_agent][#1] get_current_stock_price({'ticker': 'NOW'})
 
  ══════════════════════════════════════════════════════════════════════
  TOOL TRACE  |  run_id: f8a2d301  |  agent: analysis_agent  |  trace_id: d91e4f02  |  9 calls
  ══════════════════════════════════════════════════════════════════════
  #    Tool                                ms  Arguments
  ──────────────────────────────────────────────────────────────────────
  1    get_current_stock_price            142  ticker='NOW'
       ↳ {"price": 90.21, "currency": "USD"}
  ...
  ──────────────────────────────────────────────────────────────────────
  run_id: f8a2d301  |  trace_id: d91e4f02  |  total tool time: 4821ms
  ══════════════════════════════════════════════════════════════════════
"""
 
import time
import uuid
from langgraph.prebuilt import ToolNode

# =============================================================================
# TOOL TRACER
# =============================================================================
 
class ToolTracer:
    """
    Records every tool call made by an agent during a single run.
 
    Attributes:
        run_id    : Parent orchestrator run ID (None for standalone runs).
        trace_id  : Unique ID for this agent run — always generated fresh.
        agent_name: Label used in all log output.
        calls     : List of call records — each is a dict with run_id,
                    trace_id, order, tool, args, result_preview, elapsed_ms.
    """
 
    def __init__(self, agent_name: str = "agent", run_id: str | None = None):
        self.agent_name = agent_name
        self.run_id     = run_id
        # Always generated fresh — unique per agent run, regardless of run_id
        self.trace_id   = uuid.uuid4().hex[:8]
        self.calls: list[dict] = []
        self._call_counter = 0
 
    # ── Data capture ──────────────────────────────────────────────────────────
 
    def record(
        self,
        tool_name:  str,
        args:       dict,
        result:     str,
        elapsed_ms: float,
    ) -> None:
        """Append one tool call record. Called by TracedToolNode after each call."""
        self._call_counter += 1
        self.calls.append({
            "run_id":         self.run_id,
            "trace_id":       self.trace_id,
            "order":          self._call_counter,
            "tool":           tool_name,
            "args":           args,
            "result_preview": result[:120].replace("\n", " ") if result else "",
            "elapsed_ms":     elapsed_ms,
        })
 
    # ── Output ────────────────────────────────────────────────────────────────
 
    def print_summary(self) -> None:
        """
        Print a formatted summary table of all tool calls after the run.
        Includes run_id (if set), trace_id, call order, timing, and arguments.
        """
        if not self.calls:
            print(f"  [{self.agent_name}][{self.trace_id}] No tool calls recorded.")
            return
 
        run_label = f"run_id: {self.run_id}  |  " if self.run_id else ""
 
        # Header
        print(f"\n{'=' * 70}")
        print(
            f"  TOOL TRACE  |  {run_label}"
            f"agent: {self.agent_name}  |  "
            f"trace_id: {self.trace_id}  |  "
            f"{len(self.calls)} calls"
        )
        print(f"{'=' * 70}")
        print(f"  {'#':<3}  {'Tool':<35}  {'ms':>6}  Arguments")
        print(f"  {'-' * 66}")
 
        # One row per call
        for c in self.calls:
            arg_str = (
                ", ".join(f"{k}={repr(v)[:30]}" for k, v in c["args"].items())
                if c["args"] else "(no args)"
            )
            print(f"  {c['order']:<3}  {c['tool']:<35}  {c['elapsed_ms']:>5.0f}  {arg_str}")
            print(f"       ↳ {c['result_preview']}")
 
        # Footer
        total_ms   = sum(c["elapsed_ms"] for c in self.calls)
        run_footer = f"run_id: {self.run_id}  |  " if self.run_id else ""
        print(f"  {'-' * 66}")
        print(f"  {run_footer}trace_id: {self.trace_id}  |  total tool time: {total_ms:.0f}ms")
        print(f"{'=' * 70}\n")
 
    # ── Utilities ─────────────────────────────────────────────────────────────
 
    def reset(self) -> None:
        """Clear all recorded calls. Reuse the same tracer for a new run."""
        self.calls = []
        self._call_counter = 0
 
    def to_dict(self) -> dict:
        """
        Serialise the full trace to a dict — useful for writing to a log store,
        database, or passing to an observability backend (Langfuse, OpenTelemetry).
        """
        return {
            "run_id":     self.run_id,
            "trace_id":   self.trace_id,
            "agent_name": self.agent_name,
            "total_calls": len(self.calls),
            "total_ms":   sum(c["elapsed_ms"] for c in self.calls),
            "calls":      self.calls,
        }
 
# =============================================================================
# TRACED TOOL NODE
# =============================================================================
 
class TracedToolNode(ToolNode):
    """
    LangGraph ToolNode subclass that records every tool call into a ToolTracer.
 
    Inherits all ToolNode behaviour:
      - Executes tool_calls from the last AIMessage
      - handle_tool_errors=True — exceptions become ToolMessages, not crashes
      - Supports parallel tool execution via asyncio.gather
 
    Adds:
      - Timing (monotonic clock, millisecond precision)
      - Argument capture (from tool_call["args"])
      - Result preview (first 120 chars of the ToolMessage content)
      - All recorded into the injected ToolTracer instance
 
    Usage:
        tracer = ToolTracer(agent_name="my_agent", run_id="abc123")
        node   = TracedToolNode(tools, tracer=tracer, handle_tool_errors=True)
        builder.add_node("tools", node)
    """
 
    def __init__(self, tools: list, tracer: ToolTracer, **kwargs):
        super().__init__(tools, **kwargs)
        self.tracer   = tracer
 
    async def _arun_one(self, tool_call, input_type, config):
        """
        Override async single-tool execution to add tracing.
 
        Execution order:
          1. Capture tool name and arguments from tool_call dict
          2. Start monotonic timer
          3. Delegate to super()._arun_one() — actual tool execution
          4. Stop timer, extract result text from ToolMessage content
          5. Record into self.tracer
          6. Return result unchanged (transparent to caller)
        """
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
 
        start  = time.monotonic()
        result = await super()._arun_one(tool_call, input_type, config)
        elapsed_ms = (time.monotonic() - start) * 1000
 
        # Extract text from ToolMessage content for preview
        # Content can be a plain string or a list of content blocks
        result_text = ""
        if hasattr(result, "content"):
            if isinstance(result.content, list):
                result_text = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in result.content
                )
            else:
                result_text = str(result.content)
 
        self.tracer.record(tool_name, tool_args, result_text, elapsed_ms)
        return result 
