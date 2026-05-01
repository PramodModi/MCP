"""
agents/review_agent.py  —  Review Agent
=========================================
Validates analysis reports produced by the Analysis Agent before they
reach the user. Acts as a quality gate — catches hallucinations, missing
tool citations, date errors, and formatting violations.
 
Responsibility:
  Run structured pass/fail checks on a report.
  Return a ReviewResult with: passed flag, check details, failure reasons.
  If failed: orchestrator either retries the analysis agent or warns the user.
 
LLM: mistral-small-latest
  Checking is a simpler task than reasoning — small model is sufficient
  and saves cost on every analysis run.
 
Structured output:
  Uses llm.with_structured_output(ReviewResult) — LangChain calls the
  Mistral API with a JSON Schema derived from the Pydantic model.
  The LLM is constrained to return valid JSON that matches the schema.
  This eliminates manual JSON parsing and markdown-fence stripping entirely.
 
Why Pydantic over dataclass:
  1. with_structured_output() integration — LangChain uses model_json_schema()
     to build the function-calling schema sent to Mistral
  2. Field-level validation — recommendation must be BUY/SELL/HOLD/NOT FOUND
  3. Type coercion — "4" string coerced to int 4 for tool_call_count
  4. model_validator — enforces passed=False when any check is False
  5. Field descriptions appear in the JSON Schema the LLM sees — guides output
  6. .model_dump() for clean serialisation
 
Query types reviewed: TYPE B, D, E (investment decisions, deep analysis).
TYPE A (news briefs) is skipped — lower stakes, latency more visible.
 
Standalone test:
  python agents/review_agent.py
 
Imported by orchestrator.py:
  from agents.review_agent import run_review_agent, ReviewResult
"""

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Literal
 
sys.path.insert(0, str(Path(__file__).parent.parent))
 
from pydantic import BaseModel, Field, field_validator, model_validator
 
from agents.common import CONFIG, get_llm

# =============================================================================
# REVIEW CHECKS MODEL
# =============================================================================
 
class ReviewChecks(BaseModel):
    """
    Seven binary pass/fail checks on the analysis report.
    Each field is True (passed) or False (failed).
    Field descriptions appear in the JSON Schema sent to Mistral —
    they guide the LLM on exactly what each check means.
    """
 
    price_has_live_citation: bool = Field(
        description=(
            "True if the current stock price is attributed to a live source "
            "(Yahoo Finance, 'live', 'as of [recent date]'). "
            "False if price has no citation or is from training data."
        )
    )
 
    minimum_tool_calls_met: bool = Field(
        description=(
            "True if the report contains data from at least 4 distinct tool types: "
            "price, income statement, cashflow, and news/recommendations. "
            "False if the report uses only price data with all other metrics "
            "appearing without any data source citation."
        )
    )
 
    date_is_current: bool = Field(
        description=(
            "True if the 'data as of' date matches the current price fetch date. "
            "False if the report shows a stale date (e.g. 2025-11-01) when "
            "the current price is from a more recent date like April 2026."
        )
    )
 
    no_trailing_question: bool = Field(
        description=(
            "True if the report ends with a disclaimer or recommendation. "
            "False if the last substantive content contains a question: "
            "'Would you like', 'Shall I', 'Let me know if', 'I can provide'."
        )
    )
 
    metrics_have_citations: bool = Field(
        description=(
            "True if most key metrics (P/E, revenue, FCF, ROE) have source "
            "attribution. False only if MORE THAN HALF of key metrics have no "
            "source. Partial missing citations are warnings, not failures."
        )
    )
 
    recommendation_present: bool = Field(
        description=(
            "True if the report contains a clear BUY, SELL, or HOLD verdict. "
            "False if no investment recommendation is found."
        )
    )
 
    price_consistent: bool = Field(
        description=(
            "True if the same current price is used consistently across all sections. "
            "False if different prices appear in different sections without explanation."
        )
    )
 
 
# =============================================================================
# REVIEW RESULT MODEL
# =============================================================================
 
class ReviewResult(BaseModel):
    """
    Structured output from the Review Agent.
 
    This is the with_structured_output() target — LangChain derives the
    JSON Schema from this model and sends it to Mistral as a function
    definition. Mistral returns JSON constrained to this schema. Pydantic
    validates and constructs the instance automatically — no manual parsing.
 
    Fields:
        passed         : True only when ALL checks pass (auto-enforced by validator).
        checks         : Nested ReviewChecks with 7 binary results.
        failure_reasons: Specific description of each failed check.
        warning_reasons: Non-blocking issues to flag to the user.
        tool_call_count: Estimated distinct tool types used in the report.
        recommendation : BUY / SELL / HOLD extracted from the report.
    """
 
    passed: bool = Field(
        description=(
            "True if ALL seven checks passed. False if any single check is False. "
            "Warnings in warning_reasons do not affect this field."
        )
    )
 
    checks: ReviewChecks = Field(
        description="Seven binary pass/fail checks on the analysis report."
    )
 
    failure_reasons: list[str] = Field(
        default_factory=list,
        description=(
            "Specific description of each failed check. Empty list if all passed. "
            "Each entry should name the check and describe what was wrong."
        )
    )
 
    warning_reasons: list[str] = Field(
        default_factory=list,
        description=(
            "Non-blocking issues to note. E.g. some metrics lack citations "
            "but not more than half."
        )
    )
 
    tool_call_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Estimated number of distinct tool types used in the report. "
            "Count: price, income statement, cashflow, news, recommendations. "
            "Maximum meaningful value is 7."
        )
    )
 
    recommendation: Literal["BUY", "SELL", "HOLD", "NOT FOUND"] = Field(
        description=(
            "The investment recommendation extracted from the report. "
            "Must be exactly: BUY, SELL, HOLD, or NOT FOUND."
        )
    )
 
    # ── Validators ────────────────────────────────────────────────────────────
 
    @model_validator(mode="after")
    def sync_passed_with_checks(self) -> "ReviewResult":
        """
        Enforce: if ANY check is False, passed must be False.
        This prevents the LLM from returning passed=True with failed checks.
        Runs after all fields are validated — has access to self.checks.
        """
        all_passed = all([
            self.checks.price_has_live_citation,
            self.checks.minimum_tool_calls_met,
            self.checks.date_is_current,
            self.checks.no_trailing_question,
            self.checks.metrics_have_citations,
            self.checks.recommendation_present,
            self.checks.price_consistent,
        ])
        if not all_passed:
            # Use object.__setattr__ because Pydantic models are immutable by default
            object.__setattr__(self, "passed", False)
        return self
 
    @field_validator("failure_reasons", "warning_reasons", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        """Accept None or a single string — coerce to list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        return v
 
    @field_validator("tool_call_count", mode="before")
    @classmethod
    def coerce_to_int(cls, v):
        """Accept string integers like '4' — coerce to int."""
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0
 
    # ── Display ───────────────────────────────────────────────────────────────
 
    def print_summary(
        self,
        run_id:   str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Print a formatted review result table to stdout."""
        run_label   = f"run_id: {run_id}  |  " if run_id else ""
        trace_label = f"trace_id: {trace_id}  |  " if trace_id else ""
 
        print(f"\n{'=' * 65}")
        print(
            f"  REVIEW RESULT  |  {run_label}{trace_label}"
            f"{'✓ PASSED' if self.passed else '✗ FAILED'}"
        )
        print(f"{'=' * 65}")
 
        # model_dump() on ReviewChecks gives {field_name: bool} — iterate cleanly
        for check_name, result in self.checks.model_dump().items():
            icon = "✓" if result else "✗"
            print(f"  {icon}  {check_name}")
 
        if self.failure_reasons:
            print(f"\n  FAILURES ({len(self.failure_reasons)}):")
            for reason in self.failure_reasons:
                print(f"    • {reason}")
 
        if self.warning_reasons:
            print(f"\n  WARNINGS ({len(self.warning_reasons)}):")
            for w in self.warning_reasons:
                print(f"    ⚠ {w}")
 
        print(f"\n  Recommendation : {self.recommendation}")
        print(f"  Tool call count: ~{self.tool_call_count} distinct types")
        print(f"{'=' * 65}\n")
 
    def to_dict(self) -> dict:
        """Serialise to plain dict for JSON serialisation or passing to orchestrator."""
        return self.model_dump()
 
 
# =============================================================================
# SYSTEM PROMPT
# =============================================================================
 
REVIEW_SYSTEM_PROMPT = """You are a financial report quality reviewer.
Evaluate the analysis report against this checklist. Return a structured result.
 
CHECKLIST:
 
1. price_has_live_citation
   Current stock price must cite a live source (Yahoo Finance, "live", recent date).
   FAIL: price has no source or appears without any citation.
 
2. minimum_tool_calls_met
   Report must contain data from at least 4 distinct tool types:
   price, income statement, cashflow, and news/recommendations.
   FAIL: report uses only price data; all other metrics have no source citation.
 
3. date_is_current
   "Data as of" date must match the current price fetch date.
   FAIL: report shows 2025-11-01 but current price is from April 2026.
 
4. no_trailing_question
   Report must end with a disclaimer or recommendation — not a question.
   FAIL: contains "Would you like", "Shall I", "Let me know if", "I can provide".
 
5. metrics_have_citations
   Key metrics (P/E, revenue, FCF, ROE) must have source attribution.
   FAIL only if MORE THAN HALF have no source. Partial = warning only.
 
6. recommendation_present
   Report must contain BUY, SELL, or HOLD.
   FAIL: no recommendation found.
 
7. price_consistent
   Same current price used across all sections.
   FAIL: different prices in different sections without explanation.
 
RULE: passed = True only when ALL seven checks are True.
"""
 
 
# =============================================================================
# RUNNER
# =============================================================================
 
async def run_review_agent(
    report:  str,
    verbose: bool = True,
    run_id:  str | None = None,
) -> ReviewResult:
    """
    Run the Review Agent and return a validated ReviewResult.
 
    Uses llm.with_structured_output(ReviewResult):
      LangChain generates a JSON Schema from ReviewResult.model_json_schema()
      Sends it to Mistral as a function definition
      Mistral returns constrained JSON → LangChain constructs ReviewResult
      Pydantic validators run (sync_passed_with_checks, coerce_to_int, etc.)
      Returns a fully validated ReviewResult — no parsing needed by caller
 
    Fail-open: on any LLM or validation error, returns passed=True with a
    warning. This prevents the review gate from blocking all output on error.
    """
    trace_id  = uuid.uuid4().hex[:8]
    run_label = f"run_id: {run_id}  |  " if run_id else ""
 
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"[REVIEW AGENT] Starting review")
        print(f"  {run_label}trace_id: {trace_id}")
        print(f"  Model   : {CONFIG['models']['small']}")
        print(f"  Report  : {len(report)} chars")
        print(f"{'─' * 60}")
 
    llm            = get_llm("small")
    structured_llm = llm.with_structured_output(ReviewResult)
 
    # Keep start and end of long reports — middle can be truncated
    # Start: has price, date, data sources  |  End: has disclaimer/trailing question
    max_chars = 8000
    if len(report) > max_chars:
        report_for_review = (
            report[:4000]
            + "\n\n...[middle section truncated for review]...\n\n"
            + report[-4000:]
        )
    else:
        report_for_review = report
 
    prompt = (
        f"Review this financial analysis report:\n\n"
        f"{'─' * 50}\n"
        f"{report_for_review}\n"
        f"{'─' * 50}"
    )
 
    try:
        result: ReviewResult = structured_llm.invoke([
            ("system", REVIEW_SYSTEM_PROMPT),
            ("human",  prompt),
        ])
 
    except Exception as e:
        if verbose:
            print(f"  [REVIEW AGENT] Error — failing open: {e}")
 
        # Fail open: review error must not block the user from seeing the report
        result = ReviewResult(
            passed          = True,
            checks          = ReviewChecks(
                price_has_live_citation = True,
                minimum_tool_calls_met  = True,
                date_is_current         = True,
                no_trailing_question    = True,
                metrics_have_citations  = True,
                recommendation_present  = True,
                price_consistent        = True,
            ),
            failure_reasons = [],
            warning_reasons = [f"Review agent error (fail open): {e}"],
            tool_call_count = 0,
            recommendation  = "NOT FOUND",
        )
 
    if verbose:
        result.print_summary(run_id=run_id, trace_id=trace_id)
 
    return result


# =============================================================================
# NEWS REVIEW — CHECKS MODEL
# =============================================================================

class NewsReviewChecks(BaseModel):
    """Five binary pass/fail checks specific to news reports."""

    news_is_recent: bool = Field(
        description=(
            "True if article dates match the time intent of the task. "
            "If the task asks for 'today', 'latest', or 'current' news, "
            "ALL cited articles must be from within the last 7 days. "
            "Saying 'No recent news found' for a topic is acceptable — mark True. "
            "If the task asks for a historical period ('last 2 years', '2023'), "
            "older articles are fine — mark True. "
            "FAIL: task asks for today's news but articles from weeks or months "
            "ago are presented as current without acknowledging their age."
        )
    )

    sources_cited: bool = Field(
        description=(
            "True if every headline listed includes [Source, Date] attribution "
            "with both a source name and a publication date. "
            "False if any headline is missing either the source or the date."
        )
    )

    format_complete: bool = Field(
        description=(
            "True if the report contains ALL four required sections: "
            "MARKET HEADLINES, SECTOR PULSE, TECH & AI CONTEXT, ANALYST BRIEF. "
            "False if any section is entirely absent."
        )
    )

    no_trailing_question: bool = Field(
        description=(
            "True if the report ends with the disclaimer line "
            "'Data from public RSS feeds and APIs. Not financial advice.' "
            "False if the last substantive content is a question or offer to help: "
            "'Would you like', 'Shall I', 'Let me know if', 'Can I help', 'I can provide'."
        )
    )

    minimum_articles: bool = Field(
        description=(
            "True if at least 3 distinct news articles are listed under "
            "MARKET HEADLINES. False if fewer than 3 articles appear."
        )
    )


# =============================================================================
# NEWS REVIEW — RESULT MODEL
# =============================================================================

class NewsReviewResult(BaseModel):
    """Structured output from the News Review Agent."""

    passed: bool = Field(
        description=(
            "True only if ALL five checks are True. "
            "False if any single check is False."
        )
    )

    checks: NewsReviewChecks = Field(
        description="Five binary pass/fail checks on the news report."
    )

    failure_reasons: list[str] = Field(
        default_factory=list,
        description=(
            "Specific description of each failed check. Empty if all passed. "
            "Each entry should name the check and explain what was wrong."
        )
    )

    warning_reasons: list[str] = Field(
        default_factory=list,
        description="Non-blocking issues to note, e.g. some articles slightly older than expected."
    )

    @model_validator(mode="after")
    def sync_passed_with_checks(self) -> "NewsReviewResult":
        all_passed = all([
            self.checks.news_is_recent,
            self.checks.sources_cited,
            self.checks.format_complete,
            self.checks.no_trailing_question,
            self.checks.minimum_articles,
        ])
        if not all_passed:
            object.__setattr__(self, "passed", False)
        return self

    @field_validator("failure_reasons", "warning_reasons", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        return v

    def print_summary(
        self,
        run_id:   str | None = None,
        trace_id: str | None = None,
    ) -> None:
        run_label   = f"run_id: {run_id}  |  " if run_id else ""
        trace_label = f"trace_id: {trace_id}  |  " if trace_id else ""

        print(f"\n{'=' * 65}")
        print(
            f"  NEWS REVIEW  |  {run_label}{trace_label}"
            f"{'✓ PASSED' if self.passed else '✗ FAILED'}"
        )
        print(f"{'=' * 65}")
        for check_name, result in self.checks.model_dump().items():
            icon = "✓" if result else "✗"
            print(f"  {icon}  {check_name}")
        if self.failure_reasons:
            print(f"\n  FAILURES ({len(self.failure_reasons)}):")
            for reason in self.failure_reasons:
                print(f"    • {reason}")
        if self.warning_reasons:
            print(f"\n  WARNINGS ({len(self.warning_reasons)}):")
            for w in self.warning_reasons:
                print(f"    ⚠ {w}")
        print(f"{'=' * 65}\n")


# =============================================================================
# NEWS REVIEW — SYSTEM PROMPT
# =============================================================================

NEWS_REVIEW_SYSTEM_PROMPT = """You are a news report quality reviewer.
Evaluate the news report against the checklist below.
You receive the ORIGINAL TASK (to understand the intended time period) and the NEWS REPORT.

CHECKLIST:

1. news_is_recent
   Article dates must match the time intent of the task.
   - Task says "today", "latest", "current", "now" → ALL articles ≤7 days old.
   - "No recent news found" for a topic is acceptable → True.
   - Task specifies a historical window ("last 2 years", "2023") → older is fine → True.
   FAIL: task asks for today's news but articles from weeks/months ago are listed
         as current news without any acknowledgment that they are old.
   Example of FAIL: today is May 2026 but articles are from before April 2026 (older than 7 days).

2. sources_cited
   Every headline must have [Source, Date] with both source name and date.
   FAIL: any headline missing source name or publication date.

3. format_complete
   Output must contain ALL four sections:
     MARKET HEADLINES | SECTOR PULSE | TECH & AI CONTEXT | ANALYST BRIEF
   FAIL: any section is entirely absent.

4. no_trailing_question
   Report must end with the disclaimer:
     "Data from public RSS feeds and APIs. Not financial advice."
   FAIL: ends with "Would you like", "Shall I", "Let me know if", "Can I help".

5. minimum_articles
   At least 3 distinct articles under MARKET HEADLINES.
   FAIL: fewer than 3 articles listed.

RULE: passed = True only when ALL five checks are True.
"""


# =============================================================================
# NEWS REVIEW — RUNNER
# =============================================================================

async def run_news_review_agent(
    report:  str,
    task:    str,
    verbose: bool = True,
    run_id:  str | None = None,
) -> NewsReviewResult:
    """
    Review a news report for freshness, completeness, and formatting.

    Fail-open: on any LLM or validation error, returns passed=True with a
    warning so the review gate never blocks the user from seeing the report.

    Args:
        report:  The news report to review.
        task:    The original task string — used to judge expected time range.
        verbose: Print review result to stdout.
        run_id:  Optional orchestrator run ID for log correlation.

    Returns:
        NewsReviewResult with pass/fail outcome and failure reasons.
    """
    trace_id  = uuid.uuid4().hex[:8]
    run_label = f"run_id: {run_id}  |  " if run_id else ""

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"[NEWS REVIEW] Starting review")
        print(f"  {run_label}trace_id: {trace_id}")
        print(f"  Model   : {CONFIG['models']['small']}")
        print(f"  Report  : {len(report)} chars")
        print(f"{'─' * 60}")

    llm            = get_llm("small")
    structured_llm = llm.with_structured_output(NewsReviewResult)

    max_chars = 6000
    if len(report) > max_chars:
        report_for_review = (
            report[:3000]
            + "\n\n...[middle section truncated for review]...\n\n"
            + report[-3000:]
        )
    else:
        report_for_review = report

    prompt = (
        f"Original task: {task}\n\n"
        f"Review this news report:\n\n"
        f"{'─' * 50}\n"
        f"{report_for_review}\n"
        f"{'─' * 50}"
    )

    try:
        result: NewsReviewResult = structured_llm.invoke([
            ("system", NEWS_REVIEW_SYSTEM_PROMPT),
            ("human",  prompt),
        ])

    except Exception as e:
        if verbose:
            print(f"  [NEWS REVIEW] Error — failing open: {e}")

        result = NewsReviewResult(
            passed          = True,
            checks          = NewsReviewChecks(
                news_is_recent       = True,
                sources_cited        = True,
                format_complete      = True,
                no_trailing_question = True,
                minimum_articles     = True,
            ),
            failure_reasons = [],
            warning_reasons = [f"News review error (fail open): {e}"],
        )

    if verbose:
        result.print_summary(run_id=run_id, trace_id=trace_id)

    return result