"""
=============================================================================
PROGRAM 1: Local MCP Server — Financial Markets Knowledge Server (FastMCP)
=============================================================================
Purpose  : Exposes NSE stock data, portfolio analytics, and market research
           tools over MCP. Runs as a standalone HTTP server process.
 
Transport: streamable-http — server binds to 127.0.0.1:8000 and stays up.
           Program 2 (the agent) connects to it over HTTP independently.
           The two processes have NO parent-child relationship.
 
Run      : python program1_mcp_server.py
           Server starts and stays running until you Ctrl+C.
           Start this BEFORE running Program 2.
 
MCP endpoint : http://127.0.0.1:8000/mcp
Health check : http://127.0.0.1:8000/  (returns 200 if up)
 
Install  : pip install fastmcp
=============================================================================
"""

import json
from typing import Literal
from fastmcp import FastMCP

mcp = FastMCP("financial-markets-server")

# ---------------------------------------------------------------------------
# In-memory dataset — simulates a real market data provider (NSE feed,
# Bloomberg, yfinance). In production: replace each dict lookup with an
# async call to your actual data source.
# ---------------------------------------------------------------------------
 
STOCKS = {
    "RELIANCE": {
        "name": "Reliance Industries Ltd",
        "sector": "Energy",
        "exchange": "NSE",
        "market_cap_cr": 1_892_000,
        "pe_ratio": 28.4,
        "52w_high": 3217.0,
        "52w_low": 2220.3,
        "current_price": 2987.5,
        "beta": 1.12,
        "div_yield_pct": 0.38,
        "revenue_cr": 871_000,
        "net_profit_cr": 67_000,
        "debt_equity": 0.41,
        "roe_pct": 9.1,
        "price_history": [2810, 2855, 2901, 2933, 2987, 2950, 2987],
    },
    "TCS": {
        "name": "Tata Consultancy Services",
        "sector": "IT",
        "exchange": "NSE",
        "market_cap_cr": 1_411_000,
        "pe_ratio": 33.1,
        "52w_high": 4592.0,
        "52w_low": 3311.0,
        "current_price": 3887.0,
        "beta": 0.74,
        "div_yield_pct": 1.62,
        "revenue_cr": 240_893,
        "net_profit_cr": 46_099,
        "debt_equity": 0.04,
        "roe_pct": 52.4,
        "price_history": [3901, 3875, 3840, 3858, 3887, 3910, 3887],
    },
    "HDFCBANK": {
        "name": "HDFC Bank Ltd",
        "sector": "Banking",
        "exchange": "NSE",
        "market_cap_cr": 1_180_000,
        "pe_ratio": 19.3,
        "52w_high": 1880.0,
        "52w_low": 1363.6,
        "current_price": 1715.0,
        "beta": 0.89,
        "div_yield_pct": 1.21,
        "revenue_cr": 166_172,
        "net_profit_cr": 65_000,
        "debt_equity": 7.2,
        "roe_pct": 16.8,
        "price_history": [1690, 1705, 1720, 1698, 1715, 1725, 1715],
    },
    "INFY": {
        "name": "Infosys Ltd",
        "sector": "IT",
        "exchange": "NSE",
        "market_cap_cr": 640_000,
        "pe_ratio": 27.6,
        "52w_high": 2006.0,
        "52w_low": 1358.0,
        "current_price": 1501.0,
        "beta": 0.81,
        "div_yield_pct": 2.18,
        "revenue_cr": 160_264,
        "net_profit_cr": 26_000,
        "debt_equity": 0.09,
        "roe_pct": 31.7,
        "price_history": [1482, 1495, 1501, 1510, 1498, 1501, 1501],
    },
    "WIPRO": {
        "name": "Wipro Ltd",
        "sector": "IT",
        "exchange": "NSE",
        "market_cap_cr": 265_000,
        "pe_ratio": 22.1,
        "52w_high": 575.0,
        "52w_low": 400.1,
        "current_price": 481.3,
        "beta": 0.78,
        "div_yield_pct": 0.21,
        "revenue_cr": 90_487,
        "net_profit_cr": 11_299,
        "debt_equity": 0.19,
        "roe_pct": 15.9,
        "price_history": [470, 475, 480, 478, 481, 483, 481],
    },
    "TATASTEEL": {
        "name": "Tata Steel Ltd",
        "sector": "Metals",
        "exchange": "NSE",
        "market_cap_cr": 189_000,
        "pe_ratio": 14.8,
        "52w_high": 184.9,
        "52w_low": 120.3,
        "current_price": 155.2,
        "beta": 1.38,
        "div_yield_pct": 0.64,
        "revenue_cr": 229_170,
        "net_profit_cr": 8_900,
        "debt_equity": 1.02,
        "roe_pct": 7.8,
        "price_history": [148, 151, 155, 158, 155, 153, 155],
    },
    "SBIN": {
        "name": "State Bank of India",
        "sector": "Banking",
        "exchange": "NSE",
        "market_cap_cr": 650_000,
        "pe_ratio": 10.1,
        "52w_high": 912.0,
        "52w_low": 601.8,
        "current_price": 777.0,
        "beta": 1.05,
        "div_yield_pct": 1.93,
        "revenue_cr": 445_000,
        "net_profit_cr": 61_000,
        "debt_equity": 13.5,
        "roe_pct": 18.4,
        "price_history": [762, 770, 778, 771, 777, 780, 777],
    },
    "BAJFINANCE": {
        "name": "Bajaj Finance Ltd",
        "sector": "NBFC",
        "exchange": "NSE",
        "market_cap_cr": 435_000,
        "pe_ratio": 30.5,
        "52w_high": 7830.0,
        "52w_low": 6187.5,
        "current_price": 7015.0,
        "beta": 1.21,
        "div_yield_pct": 0.29,
        "revenue_cr": 55_900,
        "net_profit_cr": 14_451,
        "debt_equity": 3.7,
        "roe_pct": 22.1,
        "price_history": [6950, 6980, 7010, 7050, 7015, 7020, 7015],
    },
}
 
SECTOR_INDICES = {
    "IT":      {"index": "NIFTY IT",      "1d_chg_pct": -0.42, "1w_chg_pct":  1.81},
    "Banking": {"index": "BANK NIFTY",    "1d_chg_pct":  0.61, "1w_chg_pct":  2.34},
    "Energy":  {"index": "NIFTY ENERGY",  "1d_chg_pct":  0.15, "1w_chg_pct": -0.72},
    "Metals":  {"index": "NIFTY METAL",   "1d_chg_pct": -1.20, "1w_chg_pct": -2.11},
    "NBFC":    {"index": "NIFTY FIN SVC", "1d_chg_pct":  0.33, "1w_chg_pct":  1.55},
}
 


# =============================================================================
# TOOLS
# FastMCP auto-generates the JSON Schema from type hints + docstrings.
# The docstring is the description the LLM reads to decide WHEN to call it.
# Write docstrings for two audiences: the model AND the human developer.
# =============================================================================

@mcp.tool
def get_stock_fundamentals( ticker: str, include_technicals: bool = True) -> str:
    
    """
    Retrieve fundamental and technical data for a single NSE-listed stock.
    Use this when a user asks about a specific company's financials,
    valuation, or price momentum. Always call get_market_news first for context.
 
    Returns: P/E ratio, market cap, ROE, debt/equity, 52-week range,
             dividend yield, revenue, net profit, and momentum signal.
 
    Supported tickers: RELIANCE, TCS, HDFCBANK, INFY, WIPRO,
                       TATASTEEL, SBIN, BAJFINANCE
 
    Args:
        ticker:              NSE ticker symbol, uppercase (e.g. "TCS").
        include_technicals:  If True, adds 7-session price history + momentum.
    """
    ticker = ticker.upper().strip()
    stock  = STOCKS.get(ticker)
    if not stock:
        return json.dumps({"error": f"Unknown ticker: {ticker} ",
         "Supported" : list(STOCKS.keys())})
    result = {
        "ticker" : ticker,
        "name":   stock["name"],
        "sector": stock["sector"],
        "current_price_inr": stock["current_price"],
        "market_cap_cr":     stock["market_cap_cr"],
        "valuation": {
            "pe_ratio":      stock["pe_ratio"],
            "div_yield_pct": stock["div_yield_pct"],
        },
        "profitability": {
            "roe_pct":       stock["roe_pct"],
            "net_profit_cr": stock["net_profit_cr"],
            "revenue_cr":    stock["revenue_cr"],
        },
        "risk": {
            "beta":         stock["beta"],
            "debt_equity":  stock["debt_equity"],
        },
        "price_range_52w": {
            "high":          stock["52w_high"],
            "low":           stock["52w_low"],
            "pct_from_high": round((stock["current_price"] - stock["52w_high"]) / stock["52w_high"] * 100, 2),
            "pct_from_low":  round((stock["current_price"] - stock["52w_low"])  / stock["52w_low"]  * 100, 2),
        },

    }
    return json.dumps(result)

@mcp.tool
def compare_stocks(tickers: str, metric: Literal["valuation", "profitability", "risk", "all"] = "all",) -> str:
    """
    Side-by-side comparison of multiple NSE stocks on key financial metrics.
    Use this when a user wants to pick between 2 or more stocks in the same sector, or is building a shortlist. Results are ranked by ROE.

    Args:
        tickers: Comma-separated NSE ticker symbols. E.g. "TCS,INFY,WIPRO"
        metric:  'valuation' → P/E, div yield
                'profitability' → ROE, net margin
                'risk' → beta, debt/equity, momentum
                'all' → all of the above
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    missing = [t for t in ticker_list if t not in STOCKS]
    if missing:
        return json.dumps({"error": f"Unknown tickers: {missing}",
            "Supported" : list(STOCKS.keys())})

    rows = []
    for ticker in ticker_list:
        s   = STOCKS[ticker]
        row = {
            "ticker":  ticker,
            "name":    s["name"],
            "sector":  s["sector"],
            "price":   s["current_price"],
        }
        if metric in ("valuation", "all"):
            row["pe_ratio"]       = s["pe_ratio"]
            row["div_yield_pct"]  = s["div_yield_pct"]
            row["market_cap_cr"]  = s["market_cap_cr"]
        if metric in ("profitability", "all"):
            row["roe_pct"]        = s["roe_pct"]
            row["net_margin_pct"] = round(s["net_profit_cr"] / s["revenue_cr"] * 100, 1)
        if metric in ("risk", "all"):
            row["beta"]           = s["beta"]
            row["debt_equity"]    = s["debt_equity"]
            row["momentum"]       = _momentum_signal(s["price_history"])
        rows.append(row)

    rows.sort(key=lambda x: x.get("roe_pct", 0), reverse=True)

    return json.dumps({
        "comparison":  rows,
        "ranked_by":   "ROE descending",
        "note":        "Higher ROE = better capital efficiency, all else equal.",
    }, indent=2)

@mcp.resource("market://nifty50/snapshot")
def nifty50_snapshot() -> str:
    """Live Nifty50 snapshot with index level and top movers."""

    movers = sorted(STOCKS.items(), key=lambda x: x[1]["beta"], reverse=True)[:3]
    return json.dumps({
        "index":       "NIFTY 50",
        "level":       24_850.3,
        "1d_chg_pct":  0.38,
        "top_movers":  [{"ticker": k, "price": v["current_price"]} for k, v in movers],
        "as_of":       "2025-04-25 15:30 IST",
    }, indent=2)


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # HTTP transport — server binds and stays up as a standalone process.
    #
    # streamable-http is the modern MCP transport (replaces the older SSE mode).
    # It supports both request/response and server-sent streaming in one endpoint.
    #
    # The MCP endpoint is served at: http://{host}:{port}/mcp
    # Any MCP client (Program 2, Claude Desktop, curl) connects to that URL.
    #
    # To change port: set env var MCP_PORT=9000 before running.
    # To allow remote connections: change host to "0.0.0.0" (adds security risk).
    # ---------------------------------------------------------------------------
   
 
    host = "127.0.0.1"
    port =  "8000"
 
    print(f"Financial Markets MCP Server")
    print(f"Transport : streamable-http")
    print(f"Endpoint  : http://{host}:{port}/mcp")
    print(f"Press Ctrl+C to stop.\n")
 
    mcp.run(
        transport = "streamable-http",
        host      = host,
        port      = port,
    )





