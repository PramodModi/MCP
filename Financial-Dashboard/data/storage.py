"""
data/storage.py — SQLite Storage Layer
=======================================
All database operations for transactions and sync metadata.

Schema
------
transactions
  id               INTEGER PRIMARY KEY AUTOINCREMENT
  email_id         TEXT UNIQUE  — Gmail message ID (deduplication key)
  date             TEXT         — YYYY-MM-DD
  amount           REAL
  merchant         TEXT
  category         TEXT
  transaction_type TEXT         — 'debit' | 'credit'
  bank_or_source   TEXT
  confidence       REAL
  created_at       TEXT         — ISO timestamp

sync_meta
  key   TEXT PRIMARY KEY
  value TEXT

Database is created (tables + indexes) automatically on first import.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent

with open(PROJECT_ROOT / "agent_config.yaml") as f:
    CONFIG = yaml.safe_load(f)

DB_PATH = PROJECT_ROOT / CONFIG["database"]["path"]


# =============================================================================
# Connection Helper
# =============================================================================

@contextmanager
def _conn():
    """Yield a committed, auto-closing SQLite connection."""
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


# =============================================================================
# Initialisation
# =============================================================================

def init_db() -> None:
    """Create tables and indexes if they do not exist yet."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                email_id         TEXT    UNIQUE NOT NULL,
                date             TEXT    NOT NULL,
                amount           REAL    NOT NULL,
                merchant         TEXT,
                category         TEXT,
                transaction_type TEXT,
                bank_or_source   TEXT,
                confidence       REAL,
                created_at       TEXT    DEFAULT (datetime('now'))
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS sync_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        con.execute("CREATE INDEX IF NOT EXISTS idx_tx_date     ON transactions(date)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_tx_category ON transactions(category)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_tx_type     ON transactions(transaction_type)")


# =============================================================================
# Write Operations
# =============================================================================

def save_transaction(tx: dict) -> None:
    """
    Insert a transaction. Uses INSERT OR REPLACE so re-parsing the same
    email_id updates the record rather than raising a duplicate-key error.
    """
    with _conn() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO transactions
                (email_id, date, amount, merchant, category,
                 transaction_type, bank_or_source, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tx["email_id"],
                tx["date"],
                tx["amount"],
                tx.get("merchant", "Unknown"),
                tx.get("category", "other"),
                tx.get("transaction_type", "debit"),
                tx.get("bank_or_source", "Unknown"),
                tx.get("confidence", 0.0),
            ),
        )


def delete_all_transactions() -> None:
    """Wipe all transactions and reset the last-sync timestamp (full refresh)."""
    with _conn() as con:
        con.execute("DELETE FROM transactions")
        con.execute("DELETE FROM sync_meta WHERE key = 'last_sync'")


# =============================================================================
# Read Operations
# =============================================================================

def email_already_parsed(email_id: str) -> bool:
    """Return True if this Gmail message ID is already in the database."""
    with _conn() as con:
        row = con.execute(
            "SELECT id FROM transactions WHERE email_id = ?", (email_id,)
        ).fetchone()
    return row is not None


def get_transactions(
    start_date: str,
    end_date: str,
    categories: list[str] | None = None,
    banks: list[str] | None = None,
    tx_type: str = "debit",
) -> list[dict]:
    """
    Return transactions filtered by date range, optional category list,
    optional bank list, and transaction type.

    Args:
        start_date: Inclusive start date as YYYY-MM-DD string.
        end_date:   Inclusive end date as YYYY-MM-DD string.
        categories: If provided, only return transactions in these categories.
        banks:      If provided, only return transactions from these banks/apps.
        tx_type:    'debit' (default) or 'credit'.
    """
    query  = "SELECT * FROM transactions WHERE date BETWEEN ? AND ? AND transaction_type = ?"
    params: list = [start_date, end_date, tx_type]

    if categories:
        placeholders = ",".join("?" * len(categories))
        query += f" AND category IN ({placeholders})"
        params.extend(categories)

    if banks:
        placeholders = ",".join("?" * len(banks))
        query += f" AND bank_or_source IN ({placeholders})"
        params.extend(banks)

    query += " ORDER BY date DESC"

    with _conn() as con:
        rows = con.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_all_categories() -> list[str]:
    """Return distinct category values present in the debit transactions."""
    with _conn() as con:
        rows = con.execute(
            "SELECT DISTINCT category FROM transactions "
            "WHERE transaction_type = 'debit' AND category IS NOT NULL "
            "ORDER BY category"
        ).fetchall()
    return [r["category"] for r in rows]


def get_all_banks() -> list[str]:
    """Return distinct bank/app names present in the debit transactions."""
    with _conn() as con:
        rows = con.execute(
            "SELECT DISTINCT bank_or_source FROM transactions "
            "WHERE transaction_type = 'debit' AND bank_or_source IS NOT NULL "
            "ORDER BY bank_or_source"
        ).fetchall()
    return [r["bank_or_source"] for r in rows]


# =============================================================================
# Sync Metadata
# =============================================================================

def get_last_sync_time() -> datetime | None:
    """Return the last successful sync timestamp (UTC), or None if never synced."""
    with _conn() as con:
        row = con.execute(
            "SELECT value FROM sync_meta WHERE key = 'last_sync'"
        ).fetchone()
    return datetime.fromisoformat(row["value"]) if row else None


def update_last_sync_time(ts: datetime) -> None:
    """Record the latest sync timestamp."""
    with _conn() as con:
        con.execute(
            "INSERT OR REPLACE INTO sync_meta (key, value) VALUES ('last_sync', ?)",
            (ts.isoformat(),),
        )


# =============================================================================
# Auto-init on import
# =============================================================================

init_db()
