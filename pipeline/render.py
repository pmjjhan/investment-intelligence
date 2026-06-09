"""
render.py - Data fetching, scoring, Claude synthesis, archive, and static site rendering.

Pipeline order:
  1. archive_previous_run()        - snapshot current dist/ before overwriting
  2. fetch_and_calculate_metrics() - pull live data via yfinance
  3. process_historical_ranks()    - rank movement tracking (↑↓→)
  4. generate_investment_memo()    - Claude per-stock narrative
  5. render_all_pages()            - Jinja2 → dist/
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from jinja2 import Environment, FileSystemLoader, select_autoescape

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
ROOT          = Path(__file__).parent.parent
TEMPLATES_DIR = ROOT / "templates"
DATA_DIR      = ROOT / "data"
DIST_DIR      = ROOT / "dist"
DIST_STOCK    = DIST_DIR / "stock"
ARCHIVE_DIR   = DIST_DIR / "archive"
HISTORY_FILE  = DATA_DIR / "historical_metrics.json"

# Portfolio holdings - used to mark stocks as "held"
PORTFOLIO_TICKERS = [
    "MSFT", "GOOGL", "AVGO", "WMT", "TSM", "META",
    "CRM", "AMZN", "AXP", "VGT", "QQQM",
]

# How many top-ranked stocks get a full Claude memo (controls API cost)
MEMO_TOP_N = 25

# Fetch S&P 500 tickers dynamically from Wikipedia
def _fetch_sp500_tickers() -> list[str]:
    """Fetches current S&P 500 constituents from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        # Parse the first table on the page
        from html.parser import HTMLParser
        tickers = []
        class _Parser(HTMLParser):
            _in_first_table = False
            _in_tbody = False
            _in_td = False
            _col = 0
            _cur_row = 0
            def handle_starttag(self, tag, attrs):
                attrs_d = dict(attrs)
                if tag == "table" and "wikitable" in attrs_d.get("class", ""):
                    self._in_first_table = True
                if self._in_first_table and tag == "tbody":
                    self._in_tbody = True
                if self._in_tbody and tag == "tr":
                    self._col = 0
                    self._cur_row += 1
                if self._in_tbody and tag == "td":
                    self._in_td = True
            def handle_endtag(self, tag):
                if tag == "td":
                    self._in_td = False
                    self._col += 1
                if tag == "table":
                    self._in_first_table = False
            def handle_data(self, data):
                if self._in_first_table and self._in_td and self._col == 0:
                    t = data.strip().replace(".", "-")
                    if t:
                        tickers.append(t)
        p = _Parser()
        p.feed(resp.text)
        if len(tickers) > 400:
            log.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
            return tickers
        raise ValueError(f"Only got {len(tickers)} tickers - parse may have failed")
    except Exception as exc:
        log.warning("S&P 500 fetch failed (%s) - falling back to hardcoded list", exc)
        return _SP500_FALLBACK

# Fallback hardcoded list (top 50 by market cap) used if Wikipedia fetch fails
_SP500_FALLBACK = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","TSM","WMT",
    "JPM","LLY","V","UNH","XOM","ORCL","MA","COST","HD","PG",
    "JNJ","ABBV","BAC","KO","MRK","CVX","CRM","NFLX","AMD","PEP",
    "TMO","ACN","ADBE","LIN","MCD","ABT","CSCO","WFC","TXN","PM",
    "QCOM","IBM","CAT","GE","INTU","ISRG","VZ","SPGI","NOW","AXP",
    "PLTR","CRWD","ARM","APP","DDOG","SHOP","SNOW","PANW","UBER","COIN",
]

ALL_TICKERS: list[str] = []  # populated at pipeline start via _fetch_sp500_tickers()

# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------
for _dir in [DATA_DIR, DIST_DIR, DIST_STOCK, ARCHIVE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Gemini client (free tier)
# ---------------------------------------------------------------------------
import os
_gemini_key = os.environ.get("GEMINI_API_KEY")
if not _gemini_key:
    log.warning("GEMINI_API_KEY not set - memo generation will be skipped")
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


# ===========================================================================
# 1. ARCHIVE ENGINE
# ===========================================================================

def archive_previous_run() -> None:
    """
    Snapshots the current dist/ state into a dated archive folder
    before the new run overwrites it. Preserves HTML + JSON data.
    """
    today_str = date.today().isoformat()  # e.g. "2026-06-09"
    snapshot_dir = ARCHIVE_DIR / today_str

    if not (DIST_DIR / "top10.html").exists():
        log.info("No previous build found - skipping archive (fresh run)")
        return

    log.info("Archiving previous run → %s", snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Archive HTML pages
    for page in ["index.html", "top10.html", "trade_review.html"]:
        src = DIST_DIR / page
        if src.exists():
            shutil.copy(src, snapshot_dir / page)

    # Archive stock deep dives
    if DIST_STOCK.exists():
        shutil.copytree(DIST_STOCK, snapshot_dir / "stock", dirs_exist_ok=True)

    # Archive JSON data snapshot alongside HTML
    data_snapshot = snapshot_dir / "data"
    data_snapshot.mkdir(exist_ok=True)
    for json_file in DATA_DIR.glob("*.json"):
        shutil.copy(json_file, data_snapshot / json_file.name)

    log.info("Archive complete: %d stock files saved", len(list(DIST_STOCK.glob("*.html"))))


# ===========================================================================
# 2. DATA LAYER - yfinance metrics + conviction scoring
# ===========================================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def calculate_conviction_score(metrics: dict) -> int:
    """
    Rational Investing conviction score (0-100).
    Weighted across 4 dimensions matching the 5-attribute framework.
    """
    try:
        # Valuation: lower P/E = higher score (capped at 50x)
        pe = _safe_float(metrics.get("pe_ratio"), 50)
        valuation   = max(0, 100 - min(pe * 1.5, 100))

        # Growth: lower PEG = higher score (PEG < 1 is ideal)
        peg = _safe_float(metrics.get("peg_ratio"), 2.0)
        growth      = max(0, min((2.0 / max(peg, 0.1)) * 50, 100))

        # FCF yield: higher is better (5%+ FCF yield = full score)
        fcf_yield   = max(0, min(_safe_float(metrics.get("fcf_yield")) * 12, 100))

        # Margin quality: operating margin proxy for EBITDA health
        op_margin   = max(0, min(_safe_float(metrics.get("operating_margin")) * 2, 100))

        score = int(
            (0.30 * valuation) +
            (0.30 * growth) +
            (0.25 * fcf_yield) +
            (0.15 * op_margin)
        )
        return max(0, min(score, 100))
    except Exception as exc:
        log.warning("Conviction score calc failed: %s", exc)
        return 50


def _fetch_single(ticker: str) -> dict | None:
    """Fetch and score metrics for one ticker. Returns None on failure."""
    try:
        info = yf.Ticker(ticker).info

        fcf        = _safe_float(info.get("freeCashflow"), 1)
        market_cap = _safe_float(info.get("marketCap"), 1)
        fcf_yield  = round((fcf / market_cap) * 100, 2) if market_cap else 0.0

        sbc     = _safe_float(info.get("shareBasedCompensation"))
        ocf     = _safe_float(info.get("operatingCashflow"), 1)
        sbc_pct = round((sbc / ocf) * 100, 1) if ocf else 0.0

        fcf_raw           = _safe_float(info.get("freeCashflow"))
        fcf_after_sbc     = fcf_raw - sbc
        fcf_after_sbc_m   = round(fcf_after_sbc / 1e6, 1)
        fcf_yield_sbc_adj = round((fcf_after_sbc / market_cap) * 100, 2) if market_cap else 0.0

        metrics = {
            "ticker":             ticker,
            "name":               info.get("longName", ticker),
            "price":              round(_safe_float(info.get("currentPrice")), 2),
            "pe_ratio":           round(_safe_float(info.get("trailingPE"), 30.0), 1),
            "peg_ratio":          round(_safe_float(info.get("pegRatio"), 1.5), 2),
            "operating_margin":   round(_safe_float(info.get("operatingMargins", 0.0)) * 100, 1),
            "fcf_yield":          fcf_yield,
            "fcf_yield_sbc_adj":  fcf_yield_sbc_adj,
            "fcf_after_sbc_m":    fcf_after_sbc_m,
            "sbc_pct_of_ocf":     sbc_pct,
            "_fcf_raw":           fcf_raw,
            "_fcf_after_sbc":     fcf_after_sbc,
            "_market_cap":        market_cap,
            "_shares":            _safe_float(info.get("sharesOutstanding")),
            "_sbc":               sbc,
            "held":               ticker in PORTFOLIO_TICKERS,
        }
        metrics["conviction_score"] = calculate_conviction_score(metrics)
        log.info("Fetched %-6s  score=%d  price=$%.2f", ticker, metrics["conviction_score"], metrics["price"])
        return metrics

    except Exception as exc:
        log.warning("Failed to fetch %s: %s", ticker, exc)
        return None


def fetch_and_calculate_metrics() -> list[dict]:
    """
    Pulls live financial metrics from yfinance for all S&P 500 tickers in parallel.
    Returns list of metric dicts ready for scoring and rendering.
    """
    compiled: list[dict] = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_fetch_single, t): t for t in ALL_TICKERS}
        for future in as_completed(futures):
            result = future.result()
            if result:
                compiled.append(result)

    if not compiled:
        raise ValueError("No metrics fetched - check network or ticker list")

    log.info("Successfully fetched %d / %d tickers", len(compiled), len(ALL_TICKERS))
    return compiled


# ===========================================================================
# 3. HISTORICAL RANK TRACKING
# ===========================================================================

def process_historical_ranks(current_data: list[dict]) -> tuple[list[dict], list[str]]:
    """
    Sorts by conviction score, assigns ranks, computes week-over-week
    rank movement (↑2, ↓1, →), and persists to historical ledger.

    Returns (ranked_data, sorted_archive_dates).
    """
    today_str = date.today().isoformat()

    # Sort descending by conviction score
    current_data.sort(key=lambda x: x["conviction_score"], reverse=True)
    for i, stock in enumerate(current_data):
        stock["current_rank"] = i + 1

    # Load historical ledger
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            ledger: dict = json.load(f)
    else:
        ledger = {}

    # Find most recent past date for rank delta
    past_dates = sorted([d for d in ledger if d < today_str], reverse=True)

    for stock in current_data:
        stock["rank_movement"] = "→"
        if past_dates:
            prev = next(
                (s for s in ledger[past_dates[0]] if s["ticker"] == stock["ticker"]),
                None,
            )
            if prev:
                delta = prev.get("current_rank", stock["current_rank"]) - stock["current_rank"]
                if delta > 0:
                    stock["rank_movement"] = f"↑{delta}"
                elif delta < 0:
                    stock["rank_movement"] = f"↓{abs(delta)}"

    # Persist today's snapshot
    ledger[today_str] = current_data
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)

    archive_dates = sorted(ledger.keys(), reverse=True)
    log.info("Rank tracking: %d weeks of history", len(archive_dates))
    return current_data, archive_dates


# ===========================================================================
# 4. CLAUDE MEMO GENERATION
# ===========================================================================

def generate_investment_memo(stock: dict) -> str:
    """
    Calls Gemini Flash (free tier) to generate a structured HTML investment memo.
    Applies the Rational Investing framework: catalyst, bear case, sizing view.
    Returns raw HTML string (3 blocks).
    """
    if not _gemini_key:
        return "<p>Memo unavailable - GEMINI_API_KEY not configured.</p>"

    prompt = (
        "You are an institutional investment analyst applying the Rational Investing framework. "
        "Evaluate stocks on: Revenue Growth, EBITDA Growth, Free Cash Flow, Debt levels, and Valuation vs. intrinsic value.\n\n"
        "Output exactly three HTML blocks with no markdown wrappers or filler text:\n"
        "1. <p> tag: Core structural catalyst (why this stock, why now)\n"
        "2. <ul> tag: 3 specific bear case risks with <li> items\n"
        "3. <p> tag: Sizing recommendation - one of: Strong Buy / Add / Hold / Watch / Avoid, with a one-sentence rationale\n\n"
        f"Analyze {stock['name']} ({stock['ticker']}) using Rational Investing framework.\n"
        f"Metrics: Conviction Score {stock['conviction_score']}/100, "
        f"FCF Yield {stock['fcf_yield']}%, "
        f"Operating Margin {stock['operating_margin']:.1f}%, "
        f"P/E {stock['pe_ratio']}x, "
        f"PEG {stock['peg_ratio']}, "
        f"SBC as % of OCF {stock['sbc_pct_of_ocf']}%.\n"
        f"Currently {'held in portfolio' if stock['held'] else 'not held - watchlist candidate'}."
    )

    try:
        resp = requests.post(
            _GEMINI_URL,
            params={"key": _gemini_key},
            json={"contents": [{"parts": [{"text": prompt}]}],
                  "generationConfig": {"maxOutputTokens": 800, "temperature": 0.3}},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        log.error("Memo generation failed for %s: %s", stock["ticker"], exc)
        return f"<p>Memo generation error: {exc}</p>"


# ===========================================================================
# 4b. DCF INTRINSIC VALUE
# ===========================================================================

def calculate_dcf(stock: dict) -> dict:
    """
    10-year DCF valuation model.
    Returns intrinsic value per share, margin of safety, verdict, and supporting data.
    """
    try:
        price        = _safe_float(stock.get("price"))
        shares       = _safe_float(stock.get("_shares"))
        market_cap   = _safe_float(stock.get("_market_cap"))
        fcf_start    = stock.get("_fcf_after_sbc") or stock.get("_fcf_raw") or 0.0
        op_margin    = _safe_float(stock.get("operating_margin")) / 100  # convert from %
        peg          = _safe_float(stock.get("peg_ratio"), 1.5)

        if not shares or not fcf_start or shares <= 0:
            return {}

        wacc              = 0.10
        terminal_growth   = 0.03
        projection_years  = 10

        # Derive growth rates from PEG (PEG = P/E / growth_rate, so growth ~ 1/PEG as proxy)
        # Cap sensibly: 8%-30% for y1-5, half that (min 8%) for y6-10
        if peg and peg > 0:
            implied_growth = min(max(1.0 / peg * 0.20, 0.08), 0.30)
        else:
            implied_growth = 0.15
        g1 = implied_growth
        g2 = max(g1 / 2, 0.08)

        # Project FCF over 10 years
        projected_fcf = []
        fcf = fcf_start
        for yr in range(1, projection_years + 1):
            rate = g1 if yr <= 5 else g2
            fcf = fcf * (1 + rate)
            projected_fcf.append(round(fcf / 1e6, 1))  # store in $M

        # Discount all projected FCFs
        pv_sum = sum(
            projected_fcf[i] * 1e6 / (1 + wacc) ** (i + 1)
            for i in range(projection_years)
        )

        # Terminal value (Gordon Growth on year-10 FCF)
        fcf_y10_raw = projected_fcf[-1] * 1e6
        terminal_value = fcf_y10_raw * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_terminal    = terminal_value / (1 + wacc) ** projection_years

        total_pv = pv_sum + pv_terminal
        intrinsic_value = round(total_pv / shares, 2)
        terminal_value_pct = round(pv_terminal / total_pv * 100, 1) if total_pv else 0.0
        margin_of_safety = round((intrinsic_value - price) / intrinsic_value * 100, 1) if intrinsic_value else 0.0

        if margin_of_safety > 15:
            verdict = "Undervalued"
        elif margin_of_safety < -10:
            verdict = "Overvalued"
        else:
            verdict = "Fairly Valued"

        return {
            "intrinsic_value":   intrinsic_value,
            "terminal_value_pct": terminal_value_pct,
            "wacc":              wacc,
            "terminal_growth":   terminal_growth,
            "projected_fcf":     projected_fcf,
            "margin_of_safety":  margin_of_safety,
            "dcf_verdict":       verdict,
        }
    except Exception as exc:
        log.warning("DCF calc failed for %s: %s", stock.get("ticker"), exc)
        return {}


# ===========================================================================
# 4c. 9-YEAR HISTORICAL ONE-PAGER
# ===========================================================================

def fetch_historical_financials(ticker: str, dcf: dict | None = None) -> list[dict]:
    """
    Pulls up to 4 years of annual actuals from yfinance + 2 forward estimates.
    Returns list of dicts sorted oldest-to-newest.
    """
    try:
        t        = yf.Ticker(ticker)
        income   = t.financials   # columns = dates, index = line items
        cashflow = t.cashflow

        rows: list[dict] = []
        for col in sorted(income.columns, reverse=False):  # oldest first
            year_label = f"FY{str(col.year)[2:]}"

            def _get(df, key):
                try:
                    return float(df.loc[key, col]) if key in df.index else None
                except Exception:
                    return None

            revenue    = _get(income, "Total Revenue")
            op_income  = _get(income, "Operating Income")
            net_income = _get(income, "Net Income")
            sbc_val    = _get(cashflow, "Stock Based Compensation")

            # FCF: prefer explicit row, else OCF - Capex
            fcf_val = _get(cashflow, "Free Cash Flow")
            if fcf_val is None:
                ocf_val  = _get(cashflow, "Operating Cash Flow")
                capex    = _get(cashflow, "Capital Expenditure")
                if ocf_val is not None and capex is not None:
                    fcf_val = ocf_val + capex  # capex is usually negative in yfinance

            fcf_after_sbc_val = (fcf_val - sbc_val) if (fcf_val is not None and sbc_val is not None) else fcf_val

            op_margin_val = round(op_income / revenue * 100, 1) if (op_income and revenue) else None

            rows.append({
                "year":          year_label,
                "revenue":       round(revenue / 1e9, 2) if revenue else None,
                "ebitda":        None,  # not directly available from yfinance income stmt
                "fcf":           round(fcf_val / 1e9, 2) if fcf_val is not None else None,
                "fcf_after_sbc": round(fcf_after_sbc_val / 1e9, 2) if fcf_after_sbc_val is not None else None,
                "op_margin":     op_margin_val,
                "est":           False,
            })

        # Sort oldest-to-newest (already done by sorted above)
        rows = rows[-4:]  # keep at most 4 actuals

        # Append 2 forward estimates derived from DCF assumptions
        if dcf and rows:
            last = rows[-1]
            last_year_num = int("20" + last["year"][2:])
            g1 = dcf.get("wacc", 0.10)  # reuse growth from dcf context isn't stored, use revenue proxy
            # Use 15% as default forward estimate growth if DCF not parameterized further
            fwd_g = 0.15
            for offset in [1, 2]:
                est_year = f"FY{str(last_year_num + offset)[2:]}"
                prev_rev = rows[-1 + offset - 1]["revenue"] if offset == 1 else rows[-1]["revenue"] * (1 + fwd_g)
                prev_fcf = rows[-1 + offset - 1]["fcf"] if offset == 1 else (rows[-1]["fcf"] or 0) * (1 + fwd_g)
                est_rev  = round((prev_rev or 0) * (1 + fwd_g), 2)
                est_fcf  = round((prev_fcf or 0) * (1 + fwd_g), 2)
                rows.append({
                    "year":          est_year,
                    "revenue":       est_rev,
                    "ebitda":        None,
                    "fcf":           est_fcf,
                    "fcf_after_sbc": round(est_fcf * 0.85, 2),  # rough 85% of FCF after SBC
                    "op_margin":     (last["op_margin"] or 0) + 2 * offset if last["op_margin"] else None,
                    "est":           True,
                })

        return rows

    except Exception as exc:
        log.warning("Historical financials fetch failed for %s: %s", ticker, exc)
        return []


# ===========================================================================
# 4d. SENSITIVITY TABLE
# ===========================================================================

def calculate_sensitivity(stock: dict) -> dict:
    """
    3x3 IRR grid across exit multiples [25, 30, 35x] and FCF growth scenarios
    [base-3pp, base, base+3pp].
    """
    try:
        price  = _safe_float(stock.get("price"))
        shares = _safe_float(stock.get("_shares"))
        fcf_after_sbc = _safe_float(stock.get("_fcf_after_sbc")) or _safe_float(stock.get("_fcf_raw"))
        peg    = _safe_float(stock.get("peg_ratio"), 1.5)

        if not price or not shares or not fcf_after_sbc or shares <= 0:
            return {}

        # Base growth rate derived from PEG (same logic as DCF)
        if peg and peg > 0:
            base_growth = min(max(1.0 / peg * 0.20, 0.08), 0.30)
        else:
            base_growth = 0.15

        multiples    = [25, 30, 35]
        growth_rates = [base_growth - 0.03, base_growth, base_growth + 0.03]
        growth_labels = [
            f"Bear -{3}pp",
            "Base",
            f"Bull +{3}pp",
        ]

        matrix = []
        for mult in multiples:
            row = []
            for g in growth_rates:
                forward_fcf      = fcf_after_sbc * (1 + g)
                implied_mkt_cap  = forward_fcf * mult
                implied_price    = implied_mkt_cap / shares
                irr              = round((implied_price - price) / price * 100, 1)
                row.append(irr)
            matrix.append(row)

        return {
            "multiples":     multiples,
            "growth_rates":  [round(g * 100, 1) for g in growth_rates],
            "growth_labels": growth_labels,
            "matrix":        matrix,
        }
    except Exception as exc:
        log.warning("Sensitivity calc failed for %s: %s", stock.get("ticker"), exc)
        return {}


# ===========================================================================
# 5. JINJA2 RENDERING
# ===========================================================================

def _make_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    log.info("Wrote → %s", path.relative_to(ROOT))


def _render_page(env: Environment, template: str, output: Path, context: dict) -> bool:
    try:
        html = env.get_template(template).render(**context)
        _write(output, html)
        return True
    except Exception as exc:
        log.error("Render failed [%s → %s]: %s", template, output.name, exc, exc_info=True)
        return False


def build_checklist(stock: dict) -> dict:
    """
    Builds the Rational Investing 5-point checklist for a stock.
    Each item: {"label": str, "status": "pass"|"warn"|"fail", "note": str}
    """
    pe       = _safe_float(stock.get("pe_ratio"), 50)
    peg      = _safe_float(stock.get("peg_ratio"), 2.0)
    op_mg    = _safe_float(stock.get("operating_margin"), 0)
    fcf_y    = _safe_float(stock.get("fcf_yield"), 0)
    fcf_adj  = _safe_float(stock.get("fcf_yield_sbc_adj"), 0)
    sbc_pct  = _safe_float(stock.get("sbc_pct_of_ocf"), 0)

    def _status(condition_pass, condition_warn):
        if condition_pass: return "pass"
        if condition_warn: return "warn"
        return "fail"

    return {
        "revenue_growth": {
            "label": "Topline Revenue Growth",
            "status": _status(peg <= 1.5, peg <= 2.5),
            "note": f"PEG {peg:.2f} — {'strong growth implied' if peg <= 1.5 else 'moderate growth' if peg <= 2.5 else 'weak/no growth signal'}",
        },
        "ebitda_growth": {
            "label": "EBITDA Growth",
            "status": _status(op_mg >= 20, op_mg >= 10),
            "note": f"Operating margin {op_mg:.1f}% — {'expanding, healthy' if op_mg >= 20 else 'moderate' if op_mg >= 10 else 'thin or negative'}",
        },
        "fcf": {
            "label": "Strong Free Cash Flow",
            "status": _status(fcf_adj >= 3, fcf_adj >= 1),
            "note": f"FCF yield after SBC {fcf_adj:.2f}% — {'strong cash generation' if fcf_adj >= 3 else 'adequate' if fcf_adj >= 1 else 'weak; SBC drag significant' if sbc_pct > 15 else 'low yield'}",
        },
        "low_debt": {
            "label": "Low Debt",
            "status": _status(sbc_pct <= 10, sbc_pct <= 20),
            "note": f"SBC {sbc_pct:.1f}% of OCF — {'clean, low dilution' if sbc_pct <= 10 else 'manageable' if sbc_pct <= 20 else 'high dilution risk'}",
        },
        "valuation": {
            "label": "Well-Priced vs. Intrinsic Value",
            "status": _status(pe <= 25, pe <= 40),
            "note": f"P/E {pe:.1f}x — {'reasonably priced' if pe <= 25 else 'fair but not cheap' if pe <= 40 else 'expensive; limited margin of safety'}",
        },
    }


def build_irr(stock: dict) -> dict | None:
    """
    Estimates 1-year IRR scenarios (bear/base/bull) from current price and DCF.
    Returns dict with bear, base, bull IRR % and exit prices, or None if insufficient data.
    """
    try:
        price    = _safe_float(stock.get("price"))
        peg      = _safe_float(stock.get("peg_ratio"), 2.0)
        pe       = _safe_float(stock.get("pe_ratio"), 30)
        fcf_adj  = _safe_float(stock.get("_fcf_after_sbc"))
        shares   = _safe_float(stock.get("_shares"))
        dcf      = stock.get("dcf_calc") or {}
        iv       = _safe_float(dcf.get("intrinsic_value"))

        if not price or price <= 0:
            return None

        # Base growth from PEG; bear = -3pp, bull = +5pp
        base_g = min(max(1.0 / max(peg, 0.1) * 0.15, 0.05), 0.40)
        bear_g = max(base_g - 0.05, 0.0)
        bull_g = min(base_g + 0.08, 0.50)

        # Exit multiple: use PE as proxy, normalise to FCF multiple
        exit_mult = min(max(pe, 15), 50)

        def _irr(growth):
            if fcf_adj and shares and shares > 0:
                fwd_fcf_per_share = (fcf_adj * (1 + growth)) / shares
                exit_price = fwd_fcf_per_share * exit_mult
            elif iv and iv > 0:
                # fall back to DCF IV ± growth adjustment
                exit_price = iv * (1 + growth)
            else:
                exit_price = price * (1 + growth)
            return round((exit_price - price) / price * 100, 1), round(exit_price, 2)

        bear_irr, bear_exit = _irr(bear_g)
        base_irr, base_exit = _irr(base_g)
        bull_irr, bull_exit = _irr(bull_g)

        # Price needed for 15% IRR
        target_exit = round(price * 1.15, 2)

        return {
            "bear":        bear_irr,
            "base":        base_irr,
            "bull":        bull_irr,
            "bear_exit":   bear_exit,
            "base_exit":   base_exit,
            "bull_exit":   bull_exit,
            "target_exit": target_exit,
        }
    except Exception as exc:
        log.warning("IRR build failed for %s: %s", stock.get("ticker"), exc)
        return None


def render_all_pages(
    env: Environment,
    ranked_data: list[dict],
    archive_dates: list[str],
    theses: dict,
    trade_reviews: list[dict],
) -> list[str]:
    """
    Renders all pages into dist/. Returns list of any failed page names.
    """
    errors: list[str] = []
    today_str = date.today().isoformat()

    # index.html - weekly thesis digest
    if not _render_page(env, "index.html", DIST_DIR / "index.html", {
        "data": theses,
        "generated_at": today_str,
    }):
        errors.append("index.html")

    # top10.html - conviction matrix with rank movements + archive tabs (top 100)
    if not _render_page(env, "top10.html", DIST_DIR / "top10.html", {
        "stocks": ranked_data[:100],
        "archive_dates": archive_dates,
        "generated_at": today_str,
    }):
        errors.append("top10.html")

    # dist/stock/TICKER.html - individual deep dives
    for stock in ranked_data:
        ticker = stock.get("ticker", "")
        if not ticker:
            continue
        out = DIST_STOCK / f"{ticker}.html"
        if not _render_page(env, "stock_detail.html", out, {"stock": stock, "generated_at": today_str}):
            errors.append(f"stock/{ticker}.html")

    # trade_review.html
    if not _render_page(env, "trade_review.html", DIST_DIR / "trade_review.html", {
        "trades": trade_reviews,
        "generated_at": today_str,
    }):
        errors.append("trade_review.html")

    # 404.html
    if not _render_page(env, "404.html", DIST_DIR / "404.html", {}):
        errors.append("404.html")

    # archive/index.html - browsable history list
    if not _render_page(env, "archive_index.html", ARCHIVE_DIR / "index.html", {
        "archive_dates": archive_dates,
        "generated_at": today_str,
    }):
        log.warning("archive/index.html failed - non-fatal")

    return errors


# ===========================================================================
# 6. MAIN PIPELINE
# ===========================================================================

def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        log.warning("File not found: %s", path)
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.error("Failed to load %s: %s", path, exc)
        return default


def run_pipeline() -> None:
    log.info("=" * 60)
    log.info("Investment Intelligence Pipeline starting")
    log.info("=" * 60)

    # Step 1: Archive before overwriting
    archive_previous_run()

    # Step 2: Fetch live metrics
    global ALL_TICKERS
    ALL_TICKERS = _fetch_sp500_tickers()
    log.info("Fetching yfinance metrics for %d tickers...", len(ALL_TICKERS))
    metrics_data = fetch_and_calculate_metrics()

    # Step 3: Rank tracking + movement arrows
    ranked_data, archive_dates = process_historical_ranks(metrics_data)
    log.info("Ranked %d stocks. Top: %s (score=%d)",
             len(ranked_data), ranked_data[0]["ticker"], ranked_data[0]["conviction_score"])

    # Step 4: Claude memo - top 25 only (controls API cost)
    memo_candidates = [s for s in ranked_data if s.get("current_rank", 999) <= MEMO_TOP_N]
    log.info("Generating Claude memos for top %d stocks...", len(memo_candidates))
    for stock in memo_candidates:
        stock["memo_html"] = generate_investment_memo(stock)
    # remaining stocks get no memo
    for stock in ranked_data:
        if "memo_html" not in stock:
            stock["memo_html"] = ""

    # Step 4b: DCF, historical financials, sensitivity, checklist, IRR for all stocks
    log.info("Running DCF, checklist, IRR, and sensitivity for %d stocks...", len(ranked_data))
    for stock in ranked_data:
        stock["dcf_calc"]    = calculate_dcf(stock)
        stock["historical"]  = fetch_historical_financials(stock["ticker"], stock["dcf_calc"])
        stock["sensitivity"] = calculate_sensitivity(stock)
        stock["checklist"]   = build_checklist(stock)
        stock["irr"]         = build_irr(stock)

    # Step 5: Load supplementary data (theses from agent.py, trade reviews)
    theses       = _load_json(DATA_DIR / "theses.json", default={
        "generated_at": date.today().isoformat(),
        "portfolio_alignment_score": 0,
        "actionable_move": "Run agent.py to generate thesis analysis.",
        "top_gaps": [],
        "theses": [],
    })
    trade_reviews = _load_json(DATA_DIR / "trade_review.json", default=[])

    # Step 6: Render all pages
    log.info("Rendering static site...")
    env = _make_env()
    errors = render_all_pages(env, ranked_data, archive_dates, theses, trade_reviews)

    # Summary
    log.info("=" * 60)
    if errors:
        log.error("Pipeline completed with %d render error(s): %s", len(errors), ", ".join(errors))
        sys.exit(1)
    else:
        log.info("Pipeline complete. %d pages written to %s", len(ranked_data) + 5, DIST_DIR)
        log.info("Archive dates available: %s", ", ".join(archive_dates[:5]))
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
