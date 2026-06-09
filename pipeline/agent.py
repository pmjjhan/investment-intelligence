"""
agent.py - Thesis synthesis pipeline for the Investment Intelligence dashboard.

Fetches RSS feeds from VC/institutional sources, sends content to Gemini Flash,
extracts investment theses scored against the portfolio, and writes data/theses.json.

Pipeline order:
  1. python3 pipeline/agent.py   - fetch feeds, synthesize theses, write data/theses.json
  2. python3 pipeline/render.py  - fetch market data, render HTML dashboard to dist/
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import feedparser
import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
THESES_FILE = DATA_DIR / "theses.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

MAX_ENTRIES_PER_FEED = 5
MAX_CONTENT_CHARS    = 1500
MAX_ARTICLE_AGE_DAYS = 30

# ---------------------------------------------------------------------------
# RSS Sources
# ---------------------------------------------------------------------------
RSS_SOURCES: list[dict[str, str]] = [
    {"name": "a16z",            "url": "https://a16z.com/feed"},
    {"name": "Sequoia",         "url": "https://medium.com/feed/sequoia-capital"},
    {"name": "Not Boring",      "url": "https://www.notboring.co/feed"},
    {"name": "The Generalist",  "url": "https://thegeneralist.substack.com/feed"},
    {"name": "Tomasz Tunguz",   "url": "https://www.tomasztunguz.com/index.xml"},
    {"name": "Elad Gil",        "url": "https://eladgil.substack.com/feed"},
]

# ---------------------------------------------------------------------------
# Portfolio context
# ---------------------------------------------------------------------------
PORTFOLIO_HOLDINGS = [
    "MSFT", "GOOGL", "AVGO", "WMT", "TSM", "META",
    "CRM", "AMZN", "AXP", "VGT", "QQQM",
]

PORTFOLIO_WATCHLIST = ["NOW", "ADBE", "CRWD", "PLTR", "ARM", "APP", "DDOG"]

CURRENT_THESIS = "AI distribution moat - companies owning workflow + data layer"

KNOWN_GAPS = [
    "Healthcare",
    "Energy",
    "International developed markets",
    "AI agent infrastructure",
    "Pure-play workflow automation",
    "Security (CRWD)",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a Principal Investment Intelligence Agent running an elite private tech portfolio monitor.
Your objective is to identify leading-edge structural investment frameworks and capital reallocation theses BEFORE they reach mainstream distribution.

CRITERIA FOR A HIGH-SIGNAL THESIS:
1. Value Capture Shifts: Where profit margins are migrating
2. Defensibility & Moats: New frameworks on distribution advantages
3. Margin Compression/Expansion: How compute costs alter software margins
4. TAM Re-engineering: Software shifting from IT budget to replacing enterprise labor
5. Platform Consolidation Signals: Which workflow incumbents are becoming the default AI deployment layer

IGNORE: basic product announcements, funding rounds, generic AI hype, earnings summaries.\
"""

# ---------------------------------------------------------------------------
# Feed fetching
# ---------------------------------------------------------------------------

def _parse_entry_date(entry: Any) -> datetime | None:
    """Return a timezone-aware datetime from a feedparser entry, or None."""
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def fetch_feed(source: dict[str, str]) -> list[dict[str, str]]:
    """Fetch and parse a single RSS feed. Returns a list of article dicts."""
    name = source["name"]
    url  = source["url"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=MAX_ARTICLE_AGE_DAYS)

    try:
        feed = feedparser.parse(url)
        if feed.bozo and not feed.entries:
            log.warning("%-18s  feed parse error: %s", name, feed.bozo_exception)
            return []
    except Exception as exc:
        log.warning("%-18s  fetch failed: %s", name, exc)
        return []

    articles: list[dict[str, str]] = []
    for entry in feed.entries:
        pub_dt = _parse_entry_date(entry)
        if pub_dt and pub_dt < cutoff:
            continue  # too old

        # Extract best available content
        content = ""
        if hasattr(entry, "content") and entry.content:
            content = entry.content[0].get("value", "")
        if not content and hasattr(entry, "summary"):
            content = entry.summary or ""

        # Strip HTML tags simply
        content = re.sub(r"<[^>]+>", " ", content)
        content = re.sub(r"\s+", " ", content).strip()
        content = content[:MAX_CONTENT_CHARS]

        articles.append({
            "source":    name,
            "title":     getattr(entry, "title", ""),
            "link":      getattr(entry, "link", ""),
            "published": pub_dt.strftime("%Y-%m-%d") if pub_dt else "",
            "content":   content,
        })

        if len(articles) >= MAX_ENTRIES_PER_FEED:
            break

    return articles


def fetch_all_feeds() -> list[dict[str, str]]:
    """Fetch all RSS sources sequentially, logging per-source counts."""
    all_articles: list[dict[str, str]] = []
    for source in RSS_SOURCES:
        articles = fetch_feed(source)
        log.info("%-18s  %d articles", source["name"], len(articles))
        all_articles.extend(articles)
    log.info("Total articles fetched: %d", len(all_articles))
    return all_articles


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def build_user_prompt(articles: list[dict[str, str]]) -> str:
    portfolio_ctx = {
        "holdings":        PORTFOLIO_HOLDINGS,
        "watchlist":       PORTFOLIO_WATCHLIST,
        "current_thesis":  CURRENT_THESIS,
        "known_gaps":      KNOWN_GAPS,
    }

    schema_example = {
        "generated_at":             "YYYY-MM-DD",
        "portfolio_alignment_score": "0-100 integer",
        "actionable_move":          "single most actionable move this week as a string",
        "top_gaps":                 ["gap1", "gap2", "gap3"],
        "theses": [
            {
                "name":            "thesis name",
                "signal_strength": "Early Signal | Emerging | Consensus",
                "core_claim":      "1-2 sentence core claim",
                "sources":         ["source1"],
                "supports":        ["TICKER"],
                "threatens":       ["TICKER"],
                "not_in_portfolio": ["TICKER"],
                "what_to_believe": "what needs to be true to act on this",
            }
        ],
    }

    return (
        "## RAW ARTICLE CONTENT\n\n"
        + json.dumps(articles, indent=2)
        + "\n\n## PORTFOLIO CONTEXT\n\n"
        + json.dumps(portfolio_ctx, indent=2)
        + "\n\n## TASK\n\n"
        "Extract 3-7 high-signal investment theses from the articles above. "
        "Score each against the portfolio. Return ONLY valid JSON matching this schema exactly:\n\n"
        + json.dumps(schema_example, indent=2)
        + "\n\nDo not include commentary outside the JSON object."
    )


def call_gemini(articles: list[dict[str, str]]) -> dict[str, Any]:
    """Send articles to Gemini Flash and return parsed JSON thesis output."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    user_prompt = build_user_prompt(articles)
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
        },
    }

    log.info("Calling Gemini Flash for thesis synthesis...")
    resp = requests.post(
        GEMINI_URL,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()

    data = resp.json()
    try:
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ValueError(f"Unexpected Gemini response structure: {exc}\n{data}") from exc

    return parse_gemini_response(raw_text)


def parse_gemini_response(response_text: str) -> dict[str, Any]:
    """Strip markdown fencing and parse JSON from Gemini response."""
    text = response_text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^```\s*",     "", text, flags=re.MULTILINE)
    text = re.sub(r"```$",        "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse JSON from Gemini response: {exc}") from exc

    raise ValueError("No JSON object found in Gemini response")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_theses(result: dict[str, Any]) -> None:
    result.setdefault("generated_at", date.today().isoformat())
    with open(THESES_FILE, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)


def write_error_theses(message: str) -> None:
    fallback = {
        "generated_at":             date.today().isoformat(),
        "portfolio_alignment_score": 0,
        "actionable_move":          "Pipeline error - see logs",
        "top_gaps":                 [],
        "theses":                   [],
        "error":                    message,
    }
    with open(THESES_FILE, "w", encoding="utf-8") as fh:
        json.dump(fallback, fh, indent=2, ensure_ascii=False)
    log.error("Wrote error theses.json: %s", message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_agent() -> None:
    # 1. Fetch RSS feeds
    articles = fetch_all_feeds()

    if not articles:
        write_error_theses("No articles fetched from any RSS source")
        return

    # 2. Call Gemini
    try:
        result = call_gemini(articles)
    except requests.HTTPError as exc:
        write_error_theses(f"Gemini HTTP error: {exc}")
        return
    except ValueError as exc:
        write_error_theses(str(exc))
        return
    except Exception as exc:
        write_error_theses(f"Unexpected error during Gemini call: {exc}")
        return

    # 3. Write output
    write_theses(result)

    n_theses = len(result.get("theses", []))
    score    = result.get("portfolio_alignment_score", "n/a")
    log.info("Wrote theses.json - %d theses, alignment score %s", n_theses, score)


if __name__ == "__main__":
    run_agent()
