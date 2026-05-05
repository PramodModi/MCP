"""
agents/email_sync.py  —  Email Sync (direct MCP + batched LLM extraction)
==========================================================================
Flow (all deterministic — no LLM for tool calls):
  1. query_gmail_emails           → email IDs          (direct MCP tool call)
  2. gmail_get_message_details    → email bodies        (parallel, no LLM)
  3. parse_transaction()          → Pydantic output     (batched, concurrent LLM)

MCP server and tool names configured in agent_config.yaml → mcp_servers / agent_servers.

CLI:
  python agents/email_sync.py                 # incremental sync
  python agents/email_sync.py --full-refresh  # wipe DB and re-sync
  python agents/email_sync.py --list-tools    # list MCP server tools
"""

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(find_dotenv())

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.common import CONFIG, build_mcp_config
from agents.transaction_parser import parse_transaction
from data.storage import (
    delete_all_transactions,
    email_already_parsed,
    get_last_sync_time,
    save_transaction,
    update_last_sync_time,
)

_gmail_cfg   = CONFIG["gmail"]
_exclusions  = CONFIG.get("transaction_exclusions", {})
_gmail_srv   = CONFIG.get("mcp_servers", {}).get("gmail", {})
_TOOL_SEARCH = _gmail_srv.get("tools", {}).get("search",    "query_gmail_emails")
_TOOL_GET    = _gmail_srv.get("tools", {}).get("get_email", "gmail_get_message_details")

_BATCH_SIZE       = 10   # concurrent LLM parse calls per batch
_FETCH_CONCURRENCY = 10  # max parallel gmail_get_message_details calls


# =============================================================================
# Logging
# =============================================================================

def _log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# Helpers
# =============================================================================

def _build_query(since: datetime | None) -> str:
    base = _gmail_cfg["search_query"].strip()
    if since:
        cutoff = since
    else:
        years  = int(_gmail_cfg.get("lookback_years", 2))
        cutoff = datetime.now(timezone.utc).replace(
            year=datetime.now(timezone.utc).year - years
        )
    return f"{base} after:{cutoff.strftime('%Y/%m/%d')}"


def _to_str(raw: Any) -> str:
    """Normalise any tool response type to a plain string."""
    if isinstance(raw, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in raw
        )
    return str(raw)


def _parse_ids(tool_response: Any) -> list[str]:
    """Extract email IDs from query_gmail_emails tool response."""
    text = _to_str(tool_response)
    try:
        data = json.loads(text)
        if isinstance(data, list):
            ids = []
            for item in data:
                if isinstance(item, dict):
                    eid = (item.get("id") or item.get("messageId")
                           or item.get("message_id") or item.get("email_id"))
                    if eid:
                        ids.append(str(eid))
                elif isinstance(item, str):
                    ids.append(item)
            return ids
        if isinstance(data, dict):
            for key in ("messages", "emails", "results", "items"):
                if key in data and isinstance(data[key], list):
                    return _parse_ids(json.dumps(data[key]))
    except Exception:
        pass
    return []


def _clean_body(raw: str) -> str:
    """
    Strip HTML (if present) then normalise whitespace:
      - Each line is stripped of leading/trailing spaces and \r
      - Consecutive blank lines are collapsed
    This handles both HTML bodies AND plain-text bodies with heavy indentation
    (e.g. Axis Bank emails rendered from HTML tables).
    """
    if not raw:
        return ""
    if "<" in raw and ">" in raw:
        try:
            from bs4 import BeautifulSoup
            raw = BeautifulSoup(raw, "lxml").get_text(separator="\n", strip=True)
        except Exception:
            raw = re.sub(r"<[^>]+>", " ", raw)
    lines = [line.strip() for line in raw.splitlines()]
    return "\n".join(line for line in lines if line)


def _parse_email_fields(tool_response: Any, email_id: str) -> dict:
    """Convert gmail_get_message_details response to email_data dict."""
    text = _to_str(tool_response)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}
    raw_body = (data.get("body", data.get("snippet", data.get("text", text))) or "")
    body     = _clean_body(raw_body)[:4000]
    return {
        "id":      email_id,
        "subject": data.get("subject", ""),
        "from":    data.get("from", data.get("sender", data.get("from_email", ""))),
        "date":    data.get("date", data.get("received_date", data.get("internalDate", ""))),
        "body":    body,
    }


def _is_excluded(transaction: dict) -> bool:
    merchant = (transaction.get("merchant") or "").lower()
    excluded = [m.lower() for m in _exclusions.get("merchants", [])]
    return any(ex in merchant for ex in excluded)


def _chunks(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


# =============================================================================
# Public API
# =============================================================================

async def sync_emails(
    full_refresh:      bool = False,
    progress_callback: Any  = None,
) -> dict:
    """
    Sync financial emails from Gmail.

    Args:
        full_refresh:      Clear DB and re-fetch all emails within lookback window.
        progress_callback: Optional callable(str) for live progress messages (Streamlit).

    Returns:
        Stats dict: {fetched, saved, skipped, errors}
    """
    def _cb(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    if full_refresh:
        delete_all_transactions()
        _log("Full refresh: database cleared.")
        _cb("Full refresh: database cleared.")

    since = None if full_refresh else get_last_sync_time()
    if since:
        _log(f"Incremental sync from {since.strftime('%d %b %Y %H:%M')} UTC.")
    else:
        _log("No prior sync found — applying lookback date filter.")

    query       = _build_query(since)
    max_results = int(_gmail_cfg.get("max_results_per_sync", 200))
    stats       = {"fetched": 0, "saved": 0, "skipped": 0, "errors": 0}

    # --- Connect to MCP, get tool handles ---
    mcp_conf   = build_mcp_config("email_sync")
    mcp_client = MultiServerMCPClient(mcp_conf)
    all_tools  = await mcp_client.get_tools()
    tool_map   = {t.name: t for t in all_tools}

    search_tool = tool_map.get(_TOOL_SEARCH)
    get_tool    = tool_map.get(_TOOL_GET)
    if not search_tool:
        raise RuntimeError(f"Tool '{_TOOL_SEARCH}' not found. Run --list-tools.")
    if not get_tool:
        raise RuntimeError(f"Tool '{_TOOL_GET}' not found. Run --list-tools.")

    # --------------------------------------------------------------------------
    # Step 1 — Search for email IDs  (direct tool call, no LLM)
    # --------------------------------------------------------------------------
    _log(f"\n[1/3] Searching Gmail (max {max_results})...")
    _log(f"      query: {query[:120]}...")
    search_result = await search_tool.ainvoke({"query": query, "max_results": max_results})
    all_ids       = _parse_ids(search_result)
    new_ids       = [eid for eid in all_ids if not email_already_parsed(eid)]

    stats["fetched"]  = len(all_ids)
    stats["skipped"] += len(all_ids) - len(new_ids)
    _log(f"      Found {len(all_ids)} emails — {len(new_ids)} new, {stats['skipped']} already synced.")
    _cb(f"Found {len(all_ids)} emails — {len(new_ids)} new.")

    if not new_ids:
        update_last_sync_time(datetime.now(timezone.utc))
        _log("      Nothing to process.")
        return stats

    # --------------------------------------------------------------------------
    # Step 2 — Fetch email bodies in parallel  (direct tool calls, no LLM)
    # --------------------------------------------------------------------------
    _log(f"\n[2/3] Fetching {len(new_ids)} email bodies (parallel, max {_FETCH_CONCURRENCY})...")
    semaphore = asyncio.Semaphore(_FETCH_CONCURRENCY)

    async def _fetch(eid: str) -> tuple[str, Any]:
        async with semaphore:
            result = await get_tool.ainvoke({"email_id": eid})
            return eid, result

    fetch_results = await asyncio.gather(
        *[_fetch(eid) for eid in new_ids], return_exceptions=True
    )

    email_data_list: list[dict] = []
    for idx, res in enumerate(fetch_results):
        eid = new_ids[idx]
        if isinstance(res, Exception):
            _log(f"      ✗ Fetch failed {eid}: {res}")
            stats["errors"] += 1
        else:
            _, raw = res
            email_data_list.append(_parse_email_fields(raw, eid))

    _log(f"      Fetched {len(email_data_list)} emails successfully.")
    # _log(f"      Fetched body  {email_data_list}")
    # return

    # --------------------------------------------------------------------------
    # Step 3 — Batched LLM extraction  (with_structured_output → Pydantic)
    # --------------------------------------------------------------------------
    _log(f"\n[3/3] Extracting transactions in batches of {_BATCH_SIZE}...")

    for batch_num, batch in enumerate(_chunks(email_data_list, _BATCH_SIZE), 1):
        start = (batch_num - 1) * _BATCH_SIZE + 1
        end   = min(start + _BATCH_SIZE - 1, len(email_data_list))
        _cb(f"Parsing emails {start}–{end} of {len(email_data_list)}...")

        tasks   = [parse_transaction(ed) for ed in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for email_data, txn in zip(batch, results):
            eid = email_data["id"]
            if isinstance(txn, Exception):
                stats["errors"] += 1
                _log(f"      ✗ Parse error {eid}: {txn}")
            elif txn is None or _is_excluded(txn):
                stats["skipped"] += 1
            else:
                txn["email_id"] = eid
                save_transaction(txn)
                stats["saved"] += 1
                _log(
                    f"      ✓ {txn.get('merchant','?')} "
                    f"₹{txn.get('amount',0):.2f} [{txn.get('category','?')}]"
                )

    update_last_sync_time(datetime.now(timezone.utc))
    summary = (f"Done — saved: {stats['saved']}, "
               f"skipped: {stats['skipped']}, errors: {stats['errors']}")
    _log(f"\n{'='*55}\n{summary}\n{'='*55}")
    _cb(summary)
    return stats


# =============================================================================
# Tool Discovery (--list-tools)
# =============================================================================

async def list_tools() -> None:
    mcp_conf = build_mcp_config("email_sync")
    print(f"Connecting to MCP server(s): {list(mcp_conf.keys())}")
    print("-" * 55)
    mcp_client = MultiServerMCPClient(mcp_conf)
    tools      = await mcp_client.get_tools()
    print(f"Found {len(tools)} tools:\n")
    for t in tools:
        desc = getattr(t, "description", "") or ""
        print(f"  {t.name}")
        if desc:
            print(f"    {desc[:100]}")
    print()


# =============================================================================
# Standalone Entry Point
# =============================================================================

async def _main(list_only: bool = False, full_refresh: bool = False) -> None:
    if list_only:
        await list_tools()
        return
    _log("=" * 55)
    _log("Email Sync — Standalone Mode  (direct MCP + batched LLM)")
    _log("=" * 55)
    if full_refresh:
        _log("Mode: FULL REFRESH")
    await sync_emails(full_refresh=full_refresh)


if __name__ == "__main__":
    asyncio.run(_main(
        list_only    = "--list-tools"   in sys.argv,
        full_refresh = "--full-refresh" in sys.argv,
    ))
