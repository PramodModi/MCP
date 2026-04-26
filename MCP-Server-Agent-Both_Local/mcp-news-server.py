"""
=============================================================================
program3_news_server.py  —  Business News MCP Server
=============================================================================
Standalone MCP server exposing live business and financial news tools.
Uses two completely FREE public APIs — no API key, no signup required:
  • Google News RSS         https://news.google.com/rss
  • Hacker News JSON API    https://hacker-news.firebaseio.com
 
Transport : streamable-http  →  http://127.0.0.1:8001/mcp
Start     : python program3_news_server.py
            Keep this running in Terminal 1.
            program3_public_mcp_agent.py connects to it from Terminal 3.
 
Install   : pip install fastmcp
            (no other deps — uses Python stdlib urllib + xml only)
=============================================================================
"""

import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Literal
import fastmcp
from fastmcp import FastMCP
mcp = FastMCP("business-news-server")

def _get(url: str, timeout: int = 10) -> str:
    """Simple HTTP GET. Google News RSS blocks the default urllib user-agent."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; NewsServerMCP/1.0)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")
 
 
def _parse_rss(xml_text: str, limit: int) -> list[dict]:
    """Parse Google News RSS XML into a list of clean article dicts."""
    root    = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []
    articles = []
    for item in channel.findall("item")[:limit]:
        source_el = item.find("source")
        articles.append({
            "title":   item.findtext("title", "").strip(),
            "source":  source_el.text.strip() if source_el is not None else "",
            "pubDate": item.findtext("pubDate", "").strip(),
            "link":    item.findtext("link", "").strip(),
        })
    return articles
 
 
def _google_news(query: str, limit: int) -> str:
    """Fetch + parse Google News RSS for a given query string."""
    encoded = urllib.parse.quote(query)
    url     = (
        f"https://news.google.com/rss/search"
        f"?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    xml_text = _get(url)
    return _parse_rss(xml_text, limit)
 
@mcp.tool
def get_top_business_news(
    region: Literal["India", "Global", "US", "Asia"] = "India",
    max_results: int = 10,
) -> str:
    """
    Fetch the latest top business and financial news headlines from Google News.
    Always call this first in any research workflow to establish macro context.
    Returns article titles, sources, and publication dates.
 
    Args:
        region:      Geographic focus.
                     'India'  → NSE/BSE, RBI, Indian corporates
                     'Global' → International markets, world economy
                     'US'     → Fed policy, S&P500, US earnings
                     'Asia'   → Asia-Pacific, China, Japan markets
        max_results: How many articles to return (capped at 20).
    """
    max_results = min(max_results, 20)
 
    query_map = {
        "India":  "India business finance stock market NSE BSE economy",
        "Global": "global financial markets economy business news",
        "US":     "US stock market economy Federal Reserve earnings",
        "Asia":   "Asia Pacific markets China Japan economy finance",
    }
    query = query_map[region]
    try:
        articles = _google_news(query, max_results)
        return json.dumps({
            "source":   "Google News RSS",
            "region":   region,
            "count":    len(articles),
            "articles": articles,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool
def search_news(
    topic: str,
    max_results: int = 8,
) -> str:
    """
    Search Google News RSS for recent articles on any specific topic.
    Use this for targeted queries about a company, sector, event, or person.
 
    Args:
        topic:       Any search phrase. Good examples:
                     "RBI repo rate decision"
                     "Reliance Industries Q4 results"
                     "Nifty50 all time high"
                     "TCS Infosys IT sector outlook"
                     "Indian rupee dollar exchange rate"
        max_results: Number of articles to return (capped at 15).
    """
    max_results = min(max_results, 15)
    try:
        articles = _google_news(topic, max_results)
        return json.dumps({
            "source":   "Google News RSS",
            "topic":    topic,
            "count":    len(articles),
            "articles": articles,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool
def get_top_tech_stories(limit: int = 10) -> str:
    """
    Fetch current top stories from Hacker News ranked by community score.
    Useful for understanding tech industry sentiment — relevant when analysing
    IT sector stocks (TCS, Infosys, Wipro) or AI-related market themes.
    Returns titles, URLs, scores, and comment counts.
 
    Args:
        limit: Number of stories to return (capped at 20).
    """
    limit = min(limit, 20)
    try:
        # Step 1: get ranked story ID list
        ids   = json.loads(_get("https://hacker-news.firebaseio.com/v0/topstories.json"))
        top   = ids[:limit]
 
        # Step 2: fetch each story item
        stories = []
        for sid in top:
            try:
                item = json.loads(
                    _get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json")
                )
                stories.append({
                    "rank":     len(stories) + 1,
                    "title":    item.get("title", ""),
                    "url":      item.get("url", f"https://news.ycombinator.com/item?id={sid}"),
                    "score":    item.get("score", 0),
                    "comments": item.get("descendants", 0),
                    "by":       item.get("by", ""),
                })
            except Exception:
                continue
 
        return json.dumps({
            "source":  "Hacker News",
            "count":   len(stories),
            "stories": stories,
        }, indent=2)
 
    except Exception as e:
        return json.dumps({"error": str(e)})

@mcp.tool
def get_sector_news(
    sector: Literal["IT", "Banking", "Energy", "Pharma", "Auto", "FMCG"],
    max_results: int = 6,
) -> str:
    """
    Fetch news specifically about an Indian stock market sector.
    Returns the latest headlines affecting that sector's stocks.
 
    Args:
        sector:      NSE sector to fetch news for.
        max_results: Number of articles to return.
    """
    sector_queries = {
        "IT":      "India IT sector technology companies TCS Infosys Wipro HCL",
        "Banking": "India banking sector HDFC SBI ICICI Kotak RBI NPA",
        "Energy":  "India energy sector Reliance Oil gas ONGC power",
        "Pharma":  "India pharma sector Sun Cipla Dr Reddy pharmaceutical",
        "Auto":    "India automobile sector Tata Motors Maruti Bajaj Auto",
        "FMCG":    "India FMCG consumer goods HUL ITC Nestle Dabur",
    }
    query = sector_queries[sector]
    try:
        articles = _google_news(query, min(max_results, 15))
        return json.dumps({
            "source":   "Google News RSS",
            "sector":   sector,
            "count":    len(articles),
            "articles": articles,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# RESOURCE — server metadata the client can inspect
# ---------------------------------------------------------------------------
 
@mcp.resource("news://server/info")
def server_info() -> str:
    """Describes this server's data sources and capabilities."""
    return json.dumps({
        "server":       "business-news-server",
        "data_sources": ["Google News RSS", "Hacker News Firebase API"],
        "api_key":      "not required",
        "tools": [
            "get_top_business_news",
            "search_news",
            "get_top_tech_stories",
            "get_sector_news",
        ],
    }, indent=2)


if __name__ == "__main__":
    print("Business News MCP Server started")
    mcp.run(transport="streamable-http", host="127.0.0.1", port="8001")

