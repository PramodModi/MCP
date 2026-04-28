"""
=============================================================================
news_server.py  —  Enhanced Business + Tech + AI News MCP Server
=============================================================================
Standalone MCP server exposing live news from curated FREE RSS feeds across
three categories: Business/Financial, Technology, and Artificial Intelligence.
 
No API key required. All sources are public RSS feeds.
 
RSS Feed Registry (check in news_feed.yaml file )
 
Transport : streamable-http → http://127.0.0.1:8002/mcp
Start     : python news_server.py
Install   : pip install fastmcp pyyaml
=============================================================================
"""
import json
import yaml
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Literal
from fastmcp import FastMCP


# =============================================================================
# Load Config
# =============================================================================

def _load_config(config_path : str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

config = _load_config("news_feeds.yaml")
FEEDS = config["feeds"]
SECTOR_QUERIES = config.get("sector_queries", {})
_srv = config.get("server", {})
mcp = FastMCP(_srv.get("name", "enhanced-news-server"))

# =============================================================================
# INTERNAL HELPERS
# =============================================================================
 
def _get(url: str, timeout: int = 10) -> str:
    """HTTP GET with browser-like user-agent (required by some feeds)."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent":      "Mozilla/5.0 (compatible; NewsServerMCP/2.0)",
            "Accept":          "application/rss+xml, application/xml, text/xml, */*",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")

def _parse_rss(xml_text: str, limit: int, source_label: str) -> list[dict]:
    """
    Parse RSS/Atom XML into clean article dicts.
    Handles both standard RSS <item> and Atom <entry> formats.
    Strips CDATA wrappers and HTML tags from titles/descriptions.
    """
    import re
 
    def clean(text: str) -> str:
        """Remove HTML tags, CDATA markers, and excessive whitespace."""
        if not text:
            return ""
        text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", "", text)
        return " ".join(text.split()).strip()
 
    try:
        root    = ET.fromstring(xml_text)
    except ET.ParseError:
        # Some feeds have encoding issues — try stripping the XML declaration
        xml_text = re.sub(r"<\?xml[^>]+\?>", "", xml_text)
        root    = ET.fromstring(xml_text)
 
    articles = []
    ns       = {"atom": "http://www.w3.org/2005/Atom"}
 
    # Standard RSS 2.0 format
    channel = root.find("channel")
    if channel is not None:
        for item in channel.findall("item")[:limit]:
            source_el = item.find("source")
            articles.append({
                "title":   clean(item.findtext("title", "")),
                "source":  source_el.text.strip() if source_el is not None else source_label,
                "pubDate": item.findtext("pubDate", "").strip(),
                "link":    item.findtext("link", "").strip(),
                "summary": clean(item.findtext("description", ""))[:200],
            })
        return articles
 
    # Atom 1.0 format (used by arXiv, some others)
    entries = root.findall("atom:entry", ns) or root.findall("entry")
    for entry in entries[:limit]:
        title   = entry.find("atom:title", ns) or entry.find("title")
        link_el = entry.find("atom:link", ns)  or entry.find("link")
        date_el = entry.find("atom:published", ns) or entry.find("published") or \
                  entry.find("atom:updated", ns)   or entry.find("updated")
        summary_el = entry.find("atom:summary", ns) or entry.find("summary")
 
        link = ""
        if link_el is not None:
            link = link_el.get("href", link_el.text or "")
 
        articles.append({
            "title":   clean(title.text if title is not None else ""),
            "source":  source_label,
            "pubDate": date_el.text.strip() if date_el is not None else "",
            "link":    link.strip(),
            "summary": clean(summary_el.text if summary_el is not None else "")[:200],
        })
 
    return articles

def _fetch_feed(feed_key: str, limit: int) -> list[dict]:
    """Fetch and parse a single feed by its registry key."""
    feed = FEEDS[feed_key]  
    try:
        xml  = _get(feed["url"])
        return _parse_rss(xml, limit, feed["label"])
    except Exception as e:
        return [{"error": f"{feed['label']}: {str(e)}", "title": "", "source": feed["label"]}]

def _fetch_multiple(feed_keys: list, limit_each: int) -> list[dict]:
    """
    Fetch from multiple feeds and merge results.
    Deduplicates by title to avoid identical headlines from overlapping feeds.
    """
    seen_titles = set()
    merged      = []
    for key in feed_keys:
        items = _fetch_feed(key, limit_each)
        for item in items:
            title = item.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                merged.append(item)
    return merged

def _clean_google_url(url: str) -> str:
    """
    Google News RSS returns redirect URLs in this format:
      https://news.google.com/rss/articles/CBMid0FV...?oc=5
 
    These URLs are blocked by Google's robots.txt for autonomous agents
    (user-agent: ModelContextProtocol/1.0 is explicitly disallowed).
 
    The actual article URL is embedded inside the <link> tag's redirect
    but cannot be extracted without following the redirect.
 
    Strategy: return the URL as-is for display purposes (headline + source
    is usually sufficient context). Flag it as a Google redirect so the
    agent knows not to attempt fetching it via mcp-server-fetch.
    The agent should use search_news results for headlines only, and
    fetch only direct URLs from non-Google sources (Reuters, ET, etc.)
    """
    if "news.google.com/rss/articles" in url:
        # Mark as non-fetchable redirect — agent will not attempt to fetch
        return f"[google-redirect — do not fetch] {url}"
    return url
 
 
def _google_news(query: str, limit: int) -> list[dict]:
    """
    Google News RSS search.
    Returns articles with titles, sources, and dates.
    Links from Google News are redirect URLs blocked for autonomous agents —
    they are marked as non-fetchable in the link field.
    Use these results for headlines and context only, not for URL fetching.
    """
    encoded = urllib.parse.quote(query)
    url     = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        xml      = _get(url)
        articles = _parse_rss(xml, limit, "Google News")
        # Mark all Google redirect links as non-fetchable
        for art in articles:
            if "link" in art:
                art["link"]      = _clean_google_url(art["link"])
                art["fetchable"] = False
        return articles
    except Exception as e:
        return [{"error": str(e), "title": "", "source": "Google News"}]
 

def _hn_top_stories(limit: int) -> list[dict]:
    """Hacker News Firebase JSON API — top stories ranked by score."""
    try:
        ids_raw  = _get("https://hacker-news.firebaseio.com/v0/topstories.json")
        story_ids = json.loads(ids_raw)[:limit]
        stories  = []
        for sid in story_ids:
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
                    "source":   "Hacker News",
                })
            except Exception:
                continue
        return stories
    except Exception as e:
        return [{"error": str(e)}]


# =============================================================================
# TOOLS
# =============================================================================
 
@mcp.tool
def get_india_business_news(max_results: int = 12) -> str:
    """
    Fetch latest Indian business and financial news from multiple sources:
    Economic Times, Moneycontrol, Financial Express, and Business Line.
    Deduplicates headlines across sources. Always call this first for
    Indian market context — NSE/BSE, RBI policy, corporate earnings, FII flows.
 
    Args:
        max_results: Total articles to return across all sources (max 30).
    """
    max_results  = min(max_results, 30)
    limit_each   = max(4, max_results // 4)
    articles     = _fetch_multiple(
        ["economic_times", "moneycontrol", "financial_express", "business_line"],
        limit_each
    )[:max_results]
 
    return json.dumps({
        "category": "Indian Business News",
        "sources":  ["Economic Times", "Moneycontrol", "Financial Express", "Business Line"],
        "count":    len(articles),
        "articles": articles,
    }, indent=2)
 
 
@mcp.tool
def get_global_business_news(
    region: Literal["Global", "US", "Asia"] = "Global",
    max_results: int = 10,
) -> str:
    """
    Fetch global business and financial news from Reuters and MarketWatch.
    Use this for international market context, US Fed decisions, global trade,
    commodity prices, and cross-market macro analysis.
 
    Args:
        region:      'Global' (Reuters + MarketWatch), 'US' (MarketWatch only),
                     'Asia' (Reuters + Google News Asia filter)
        max_results: Total articles to return (max 20).
    """
    max_results = min(max_results, 20)
 
    if region == "US":
        articles = _fetch_multiple(["marketwatch"], max_results)
    elif region == "Asia":
        articles = _fetch_multiple(["reuters_business"], max_results // 2)
        articles += _google_news("Asia Pacific markets economy finance", max_results // 2)
    else:
        articles = _fetch_multiple(["reuters_business", "marketwatch"], max_results // 2)
 
    return json.dumps({
        "category": f"Global Business News ({region})",
        "count":    len(articles[:max_results]),
        "articles": articles[:max_results],
    }, indent=2)
 
 
@mcp.tool
def get_tech_news(
    source: Literal["all", "techcrunch", "verge", "wired", "techmeme", "hackernews"] = "all",
    max_results: int = 10,
) -> str:
    """
    Fetch latest technology news from TechCrunch, The Verge, Wired, Techmeme,
    and Hacker News. Covers startup funding, product launches, big tech moves,
    platform changes, and developer ecosystem news.
 
    Use this for IT sector context (TCS, Infosys, Wipro, HCL) and to
    understand global tech industry sentiment affecting Indian IT stocks.
 
    Args:
        source:      Specific source or 'all' to merge all tech feeds.
        max_results: Number of articles/stories to return (max 25).
    """
    max_results = min(max_results, 25)
 
    if source == "hackernews":
        stories = _hn_top_stories(max_results)
        return json.dumps({
            "category": "Hacker News Top Stories",
            "count":    len(stories),
            "stories":  stories,
        }, indent=2)
 
    feed_map = {
        "techcrunch": ["techcrunch"],
        "verge":      ["the_verge"],
        "wired":      ["wired"],
        "techmeme":   ["techmeme"],
        "all":        ["techcrunch", "the_verge", "wired", "techmeme"],
    }
 
    feed_keys = feed_map.get(source, feed_map["all"])
    limit_each = max(3, max_results // len(feed_keys))
    articles   = _fetch_multiple(feed_keys, limit_each)[:max_results]
 
    return json.dumps({
        "category": f"Tech News ({source})",
        "sources":  [FEEDS[k]["label"] for k in feed_keys],
        "count":    len(articles),
        "articles": articles,
    }, indent=2)
 
 
@mcp.tool
def get_ai_news(
    source: Literal[
        "all", "venturebeat", "techcrunch", "mit_review", "decoder", "arxiv"
    ] = "all",
    max_results: int = 10,
) -> str:
    """
    Fetch the latest Artificial Intelligence news from VentureBeat AI,
    TechCrunch AI, MIT Technology Review, The Decoder, and arXiv cs.AI.
 
    Use this for:
    - AI product launches (new models, tools, APIs)
    - Enterprise AI adoption trends affecting IT services companies
    - Research breakthroughs that may shift the competitive landscape
    - AI regulation and policy developments
    - Funding rounds in AI startups
 
    Note: arXiv source returns research paper titles/abstracts — high signal
    but dense. Use 'all' for a balanced mix of editorial + research.
 
    Args:
        source:      Specific source or 'all' to merge all AI feeds.
        max_results: Number of articles to return (max 25).
    """
    max_results = min(max_results, 25)
 
    feed_map = {
        "venturebeat": ["venturebeat_ai"],
        "techcrunch":  ["techcrunch_ai"],
        "mit_review":  ["mit_tech_review"],
        "decoder":     ["the_decoder"],
        "arxiv":       ["arxiv_ai"],
        "all":         ["venturebeat_ai", "techcrunch_ai", "mit_tech_review", "the_decoder"],
    }
 
    feed_keys  = feed_map.get(source, feed_map["all"])
    limit_each = max(3, max_results // len(feed_keys))
    articles   = _fetch_multiple(feed_keys, limit_each)[:max_results]
 
    return json.dumps({
        "category": f"AI News ({source})",
        "sources":  [FEEDS[k]["label"] for k in feed_keys],
        "count":    len(articles),
        "articles": articles,
    }, indent=2)
 
 
@mcp.tool
def get_ai_research_papers(
    topic: Literal["cs.AI", "cs.LG", "cs.CL", "stat.ML"] = "cs.AI",
    limit: int = 8,
) -> str:
    """
    Fetch the latest research papers from arXiv in AI/ML sub-categories.
    Use this for cutting-edge research context — what is being published
    today in AI that may affect tech companies and AI product development.
 
    Categories:
      cs.AI  — Artificial Intelligence (broad)
      cs.LG  — Machine Learning
      cs.CL  — Computation and Language (LLMs, NLP)
      stat.ML — Statistical Machine Learning
 
    Args:
        topic: arXiv category to fetch.
        limit: Number of papers to return (max 15).
    """
    limit = min(limit, 15)
    url   = f"https://arxiv.org/rss/{topic}"
 
    try:
        xml      = _get(url)
        articles = _parse_rss(xml, limit, f"arXiv {topic}")
        return json.dumps({
            "category": f"arXiv Research Papers ({topic})",
            "source":   f"https://arxiv.org/rss/{topic}",
            "count":    len(articles),
            "papers":   articles,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
 
 
@mcp.tool
def search_news(topic: str, max_results: int = 10) -> str:
    """
    Search Google News RSS for recent articles on any specific topic.
    Use this for targeted queries that the dedicated feeds may not cover —
    a specific company name, ticker, event, person, or narrow topic.
 
    Examples:
      "Reliance Industries JioFinance Q4 results"
      "RBI repo rate April 2025"
      "ServiceNow AI agents enterprise"
      "US tariff India IT sector impact"
      "Nvidia Blackwell GPU shortage"
 
    Args:
        topic:       Search query — be specific for better results.
        max_results: Number of articles to return (max 20).
    """
    max_results = min(max_results, 20)
    articles    = _google_news(topic, max_results)
 
    return json.dumps({
        "query":    topic,
        "source":   "Google News RSS",
        "count":    len(articles),
        "articles": articles,
    }, indent=2)
 
 
@mcp.tool
def get_sector_news(
    sector: Literal["IT", "Banking", "Energy", "Pharma", "Auto", "FMCG", "Infra", "Telecom"],
    max_results: int = 8,
) -> str:
    """
    Fetch news specifically about an Indian NSE/BSE sector.
    Combines Google News RSS search with sector-specific keywords
    to return the most relevant recent headlines.
 
    Args:
        sector:      NSE market sector.
        max_results: Number of articles to return (max 15).
    """
    max_results = min(max_results, 15)
 
    # sector_queries loaded from news_feeds.yaml — edit there to tune keywords
    query    = SECTOR_QUERIES.get(sector, f"India {sector} sector NSE BSE")
    articles = _google_news(query, max_results)
 
    return json.dumps({
        "sector":   sector,
        "source":   "Google News RSS",
        "count":    len(articles),
        "articles": articles,
    }, indent=2)
 
 
@mcp.tool
def get_hacker_news_top(limit: int = 15) -> str:
    """
    Fetch current top stories from Hacker News ranked by community score.
    High-signal for tech and AI themes — the stories the global developer
    and tech investor community finds most important right now.
 
    Returns titles, URLs, upvote scores, and comment counts.
    Useful for understanding: AI product launches, tech industry moves,
    developer tool trends, and startup ecosystem news.
 
    Args:
        limit: Number of top stories to return (max 30).
    """
    limit   = min(limit, 30)
    stories = _hn_top_stories(limit)
 
    return json.dumps({
        "source":  "Hacker News",
        "url":     "https://news.ycombinator.com",
        "count":   len(stories),
        "stories": stories,
    }, indent=2)
 
 
# ---------------------------------------------------------------------------
# RESOURCE — server metadata
# ---------------------------------------------------------------------------
 
@mcp.resource("news://server/feeds")
def list_feeds() -> str:
    """Lists all configured RSS feeds with their categories and sources."""
    summary = {}
    for key, feed in FEEDS.items():
        cat = feed["category"]
        if cat not in summary:
            summary[cat] = []
        summary[cat].append({"key": key, "label": feed["label"], "region": feed["region"]})
 
    return json.dumps({
        "server":          "enhanced-news-server",
        "total_feeds":     len(FEEDS),
        "categories":      list(summary.keys()),
        "feeds_by_category": summary,
    }, indent=2)
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    # Host/port: env vars override yaml config, yaml overrides hardcoded defaults
    host = os.environ.get("NEWS_MCP_HOST", _srv.get("host", "127.0.0.1"))
    port = int(os.environ.get("NEWS_MCP_PORT", _srv.get("port", 8001)))
 
    print("Enhanced Business + Tech + AI News MCP Server")
    print(f"Config    : news_feeds.yaml")
    print(f"Endpoint  : http://{host}:{port}/mcp")
    print(f"Feeds     : {len(FEEDS)} active RSS sources")
    print()
    print("Tools:")
    print("  get_india_business_news   get_global_business_news")
    print("  get_tech_news             get_ai_news")
    print("  get_ai_research_papers    get_sector_news")
    print("  get_hacker_news_top       search_news")
    print()
    print("Press Ctrl+C to stop.\n")
 
    mcp.run(transport="streamable-http", host=host, port=port)






