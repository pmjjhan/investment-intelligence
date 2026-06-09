"""
orchestrate.py - Main pipeline entry point.
Runs the full investment intelligence pipeline in sequence:
  feeds -> scraper -> agent -> trade_evaluator -> render
"""

import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrate")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

RAW_CONTENT_PATH = DATA_DIR / "raw_content.json"
THESES_PATH = DATA_DIR / "theses.json"
TOP10_PATH = DATA_DIR / "top10.json"
STOCKS_DIR = DATA_DIR / "stocks"
TRADE_REVIEW_PATH = DATA_DIR / "trade_review.json"


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _run_step(name: str, fn, *args, **kwargs):
    """Run a pipeline step with timing and error handling."""
    log.info("=== STEP: %s ===", name)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        log.info("DONE %s  (%.1fs)", name, elapsed)
        return result, elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        log.error("FAILED %s after %.1fs: %s", name, elapsed, exc, exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Step 1 - RSS / Substack feeds
# ---------------------------------------------------------------------------

def step_feeds() -> list:
    """Fetch RSS content, return list of article dicts."""
    # Import here so import errors surface cleanly.
    try:
        from pipeline import feeds  # type: ignore
    except ModuleNotFoundError:
        import feeds  # type: ignore

    articles = feeds.fetch_all()
    log.info("Feeds returned %d articles", len(articles))
    return articles


# ---------------------------------------------------------------------------
# Step 2 - Exa deep search scraper
# ---------------------------------------------------------------------------

def step_scraper() -> list:
    """Fetch Exa search results, return list of article dicts."""
    try:
        from pipeline import scraper  # type: ignore
    except ModuleNotFoundError:
        import scraper  # type: ignore

    results = scraper.fetch_all()
    log.info("Scraper returned %d results", len(results))
    return results


# ---------------------------------------------------------------------------
# Step 3 - Agent analysis
# ---------------------------------------------------------------------------

def step_agent(raw_content: list) -> dict:
    """Run Claude analysis on raw content; returns parsed agent output."""
    try:
        from pipeline import agent  # type: ignore
    except ModuleNotFoundError:
        import agent  # type: ignore

    output = agent.run(raw_content)
    log.info(
        "Agent produced %d theses, %d top10 stocks",
        len(output.get("theses", [])),
        len(output.get("top10", [])),
    )
    return output


# ---------------------------------------------------------------------------
# Step 4 - Trade evaluator
# ---------------------------------------------------------------------------

def step_trade_evaluator() -> list:
    """Evaluate current portfolio trades against benchmarks."""
    try:
        from pipeline import trade_evaluator  # type: ignore
    except ModuleNotFoundError:
        import trade_evaluator  # type: ignore

    reviews = trade_evaluator.run()
    log.info("Trade evaluator produced %d reviews", len(reviews))
    return reviews


# ---------------------------------------------------------------------------
# Step 5 - Render static site
# ---------------------------------------------------------------------------

def step_render() -> None:
    """Render all HTML pages into dist/."""
    try:
        from pipeline import render  # type: ignore
    except ModuleNotFoundError:
        import render  # type: ignore

    render.main()


# ---------------------------------------------------------------------------
# Data I/O helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log.info("Wrote %s", path)


def _save_agent_output(output: dict) -> None:
    """Persist agent output to individual JSON files."""
    # theses.json
    theses_payload = {
        "generated_at": output.get("generated_at", ""),
        "portfolio_alignment_score": output.get("portfolio_alignment_score", 0),
        "actionable_move": output.get("actionable_move", ""),
        "top_gaps": output.get("top_gaps", []),
        "theses": output.get("theses", []),
    }
    _write_json(THESES_PATH, theses_payload)

    # top10.json
    _write_json(TOP10_PATH, output.get("top10", []))

    # stocks/TICKER.json
    STOCKS_DIR.mkdir(exist_ok=True)
    for stock in output.get("stocks", []):
        ticker = stock.get("ticker", "UNKNOWN")
        _write_json(STOCKS_DIR / f"{ticker}.json", stock)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log.info("Pipeline started")
    pipeline_start = time.time()
    timings: dict[str, float] = {}
    errors: list[str] = []

    # ---- Step 1: Feeds ----
    articles = []
    try:
        articles, timings["feeds"] = _run_step("RSS Feeds", step_feeds)
    except Exception:
        errors.append("feeds")
        log.warning("Continuing without RSS articles")

    # ---- Step 2: Scraper ----
    scraped = []
    try:
        scraped, timings["scraper"] = _run_step("Exa Scraper", step_scraper)
    except Exception:
        errors.append("scraper")
        log.warning("Continuing without Exa results")

    # Merge raw content and persist
    raw_content = articles + scraped
    _write_json(RAW_CONTENT_PATH, raw_content)
    log.info("raw_content.json has %d total items", len(raw_content))

    # ---- Step 3: Agent ----
    try:
        agent_output, timings["agent"] = _run_step("Claude Agent", step_agent, raw_content)
        _save_agent_output(agent_output)
    except Exception:
        errors.append("agent")
        log.error("Agent step failed - downstream steps may fail")

    # ---- Step 4: Trade evaluator ----
    try:
        trade_reviews, timings["trade_evaluator"] = _run_step(
            "Trade Evaluator", step_trade_evaluator
        )
        _write_json(TRADE_REVIEW_PATH, trade_reviews)
    except Exception:
        errors.append("trade_evaluator")
        log.warning("Trade evaluator failed - trade_review.html may be empty")

    # ---- Step 5: Render ----
    try:
        _, timings["render"] = _run_step("Render", step_render)
    except Exception:
        errors.append("render")

    # ---- Summary ----
    total = time.time() - pipeline_start
    log.info("")
    log.info("=" * 50)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 50)
    for step, t in timings.items():
        status = "FAIL" if step in errors else "OK  "
        log.info("  %s  %-20s  %.1fs", status, step, t)
    log.info("  Total: %.1fs", total)
    if errors:
        log.error("Failed steps: %s", ", ".join(errors))
        return 1
    log.info("All steps completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
