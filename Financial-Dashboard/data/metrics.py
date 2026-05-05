"""
data/metrics.py — Metrics Engine
==================================
Pure-function aggregations over a transactions DataFrame.
All functions accept a pandas DataFrame and return a DataFrame or scalar dict.
No database calls happen here — callers load data via data/storage.py first.
"""

from datetime import date, timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta


# =============================================================================
# Period Helpers
# =============================================================================

PERIOD_LABELS = {
    "this_month":    "This Month",
    "last_month":    "Last Month",
    "last_2_months": "Last 2 Months",
    "last_quarter":  "Last Quarter",
    "last_6_months": "Last 6 Months",
    "ytd":           "Year to Date",
    "custom":        "Custom Range",
}


def get_period_dates(
    period: str,
    custom_start: date | None = None,
    custom_end:   date | None = None,
) -> tuple[date, date]:
    """
    Return (start_date, end_date) for the given period key.

    Periods:
      this_month    — 1st of current month → today
      last_month    — full previous calendar month
      last_2_months — 2 full previous months + current month to today
      last_quarter  — last 3 full calendar months
      last_6_months — last 6 months (from 1st, 5 months ago → today)
      ytd           — Jan 1 of current year → today
      custom        — uses custom_start / custom_end arguments
    """
    today = date.today()

    if period == "this_month":
        return date(today.year, today.month, 1), today

    elif period == "last_month":
        first_of_current = date(today.year, today.month, 1)
        end   = first_of_current - timedelta(days=1)
        start = date(end.year, end.month, 1)
        return start, end

    elif period == "last_2_months":
        start = date(today.year, today.month, 1) - relativedelta(months=2)
        return start, today

    elif period == "last_quarter":
        first_of_current = date(today.year, today.month, 1)
        end   = first_of_current - timedelta(days=1)
        start = date(end.year, end.month, 1) - relativedelta(months=2)
        return start, end

    elif period == "last_6_months":
        start = date(today.year, today.month, 1) - relativedelta(months=5)
        return start, today

    elif period == "ytd":
        return date(today.year, 1, 1), today

    elif period == "custom" and custom_start and custom_end:
        return custom_start, custom_end

    # Fallback — current month
    return date(today.year, today.month, 1), today


# =============================================================================
# KPI Metrics
# =============================================================================

def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute top-level KPI scalars from a filtered transactions DataFrame.

    Returns dict with keys:
      total, count, avg, max, max_merchant
    """
    if df.empty:
        return {
            "total":        0.0,
            "count":        0,
            "avg":          0.0,
            "max":          0.0,
            "max_merchant": "—",
        }

    idx_max = df["amount"].idxmax()
    return {
        "total":        round(float(df["amount"].sum()), 2),
        "count":        len(df),
        "avg":          round(float(df["amount"].mean()), 2),
        "max":          round(float(df["amount"].max()), 2),
        "max_merchant": str(df.loc[idx_max, "merchant"]),
    }


# =============================================================================
# Time-Series Aggregations
# =============================================================================

def daily_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily total spend, sorted ascending by date.
    Returns columns: date (datetime), amount (float).
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "amount"])
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    return (
        d.groupby("date")["amount"]
        .sum()
        .reset_index()
        .sort_values("date")
    )


def monthly_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly total spend for month-over-month bar chart.
    Returns columns: month (YYYY-MM string), amount (float).
    """
    if df.empty:
        return pd.DataFrame(columns=["month", "amount"])
    d = df.copy()
    d["date"]  = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.to_period("M").astype(str)
    return (
        d.groupby("month")["amount"]
        .sum()
        .reset_index()
        .sort_values("month")
    )


# =============================================================================
# Categorical Aggregations
# =============================================================================

def spend_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total spend per category, sorted descending by amount.
    Returns columns: category, amount.
    """
    if df.empty:
        return pd.DataFrame(columns=["category", "amount"])
    return (
        df.groupby("category")["amount"]
        .sum()
        .reset_index()
        .sort_values("amount", ascending=False)
    )


def top_merchants(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Top N merchants by total spend, sorted descending.
    Returns columns: merchant, amount.
    """
    if df.empty:
        return pd.DataFrame(columns=["merchant", "amount"])
    return (
        df.groupby("merchant")["amount"]
        .sum()
        .reset_index()
        .sort_values("amount", ascending=False)
        .head(n)
    )


def spend_by_bank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total spend per bank/app, sorted descending.
    Returns columns: bank_or_source, amount.
    """
    if df.empty:
        return pd.DataFrame(columns=["bank_or_source", "amount"])
    return (
        df.groupby("bank_or_source")["amount"]
        .sum()
        .reset_index()
        .sort_values("amount", ascending=False)
    )


def weekly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weekly total spend (ISO week), sorted ascending.
    Returns columns: week (YYYY-Www string), amount.
    """
    if df.empty:
        return pd.DataFrame(columns=["week", "amount"])
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = d["date"].dt.strftime("%Y-W%V")
    return (
        d.groupby("week")["amount"]
        .sum()
        .reset_index()
        .sort_values("week")
    )
