# Investment Intelligence Agent

## Project Overview

A private tech portfolio monitor that surfaces leading-edge structural investment frameworks and capital reallocation theses BEFORE they reach mainstream distribution.

---

## Architecture

### Data Ingestion (feeds.py + search.py)

All source fetching happens before Claude analysis - Claude does NOT search at analysis time.

**RSS sources (feedparser):**
- a16z: https://a16z.com/feed
- Sequoia: https://medium.com/feed/sequoia-capital
- Not Boring (Packy McCormick): https://notboring.substack.com/feed
- The Generalist (Mario Gabriele): https://thegeneralist.substack.com/feed
- Tomasz Tunguz: https://www.tomasztunguz.com/index.xml
- Elad Gil: https://eladgil.substack.com/feed
- Lenny's Newsletter: https://www.lennysnewsletter.com/feed
- Invest Like the Best: joincolossus.com (migrated from Megaphone)

**Exa deep search (no RSS available):**
- BVP Atlas: site:bvp.com/atlas
- BVP Roadmaps: site:bvp.com/roadmaps
- Coatue: "Coatue" "AI" filetype:pdf (drops large public PDF decks)
- Lightspeed: site:lsvp.com/stories
- Goldman Sachs: goldmansachs.com/insights ("Top of Mind", "Blue Paper" series)
- Morgan Stanley: morganstanley.com/ideas
- Twitter/X: @benedictevans, @packyM, @bgurley, @sarahtavel, @TurnerNovak

**Exa sub-page targeting (Sequoia):** Prioritize Sonya Huang (AI/infra), Pat Grady (enterprise)

### Memory (ChromaDB)

Stores previously flagged theses as exclusion list - prevents re-surfacing known signals.

### Analysis (agent.py)

**Model:** claude-opus-4-5
**Extended thinking:** enabled, 16,000 token budget
**Max output tokens:** 8,000
**Tools at analysis time:** none

---

## Claude System Prompt

```
You are a Principal Investment Intelligence Agent running an elite private tech portfolio monitor.
Your objective is to identify leading-edge structural investment frameworks and capital 
reallocation theses BEFORE they reach mainstream distribution.

CRITERIA FOR A HIGH-SIGNAL THESIS - look specifically for:
1. Value Capture Shifts: Where profit margins are migrating (model layer vs. data layer, 
   seat licenses vs. outcome-based pricing)
2. Defensibility & Moats: New frameworks on distribution advantages (systems of record 
   converting to systems of intelligence)
3. Margin Compression/Expansion: How compute costs, fine-tuning expense, or retrieval 
   costs alter 80% software gross margins
4. TAM Re-engineering: Software shifting from IT budget line item to replacing enterprise 
   labor budgets
5. Platform Consolidation Signals: Which workflow incumbents are becoming the default 
   AI deployment layer

IGNORE: basic product announcements, funding rounds, generic AI hype, earnings summaries.
```

---

## User Prompt Structure (4 dynamic components)

1. **Raw content** - all RSS posts + Exa search results as JSON
2. **Exclusion list** - previously flagged theses from ChromaDB (do not re-surface)
3. **Portfolio context** - holdings, current thesis, watchlist, known gaps
4. **Task instructions** - extract theses, score, return JSON

---

## Output Format

Claude returns valid JSON. Dashboard renders as HTML with:
- Thesis cards (expandable): name, core claim, signal strength, supports/threatens holdings, not-in-portfolio implications, what to believe to act
- Portfolio Alignment Score (0-100)
- Single most actionable move this week
- Top 3 gaps

**Dashboard:** dashboard.html served at localhost:3456

---

## Portfolio

As of 6/2/2026 - ~$943K total

| Ticker | Name | Shares | Price (6/2) | Value | % |
|--------|------|--------|-------------|-------|---|
| MSFT | Microsoft | 376.1 | $441.31 | $165,977 | 18.3% |
| ZBGRIT | Russell 1000 ETF (401k) | 1114.83 | $98.87 | $110,227 | 12.1% |
| VOO | S&P 500 ETF | 147.6 | $698.26 | $103,063 | 11.4% |
| GOOGL | Google | 257 | $361.85 | $92,995 | 10.2% |
| AVGO | Broadcom | 170 | $481.57 | $81,867 | 9.0% |
| WMT | Walmart | 737.63 | $119.00 | $87,778 | - |
| TSM | TSMC | 94.8 | $446.69 | $42,346 | 4.7% |
| WMU60 | 2060 Target Date (401k) | 1138.29 | $30.24 | $34,419 | 3.8% |
| 005930 | Samsung (KRX) | - | - | ~$30,000 | 3.3% |
| Korea ETF | Korea ETF | - | - | ~$30,000 | 3.3% |
| VGT | Vanguard Info Tech ETF | 220 | $125.77 | $27,669 | 3.0% |
| QQQM | Nasdaq-100 ETF | 74 | $307.23 | $22,735 | 2.5% |
| META | Meta | 35 | $597.63 | $20,917 | 2.3% |
| SPAXX | Fidelity MMF | 20,169 | $1.00 | $20,169 | 2.2% |
| AMZN | Amazon | 52 | $256.52 | $13,339 | 1.5% |
| AXP | American Express | 39 | $310.97 | $12,128 | 1.3% |
| CRM | Salesforce | 37 | $200.84 | $7,431 | 0.8% |
| Cash | - | - | - | $40,000 | - |
| **Total** | | | | **$943,060** | **100%** |

**Current thesis:** AI distribution moat - companies owning workflow + data layer

**Watchlist:** NOW, ADBE, XLV, XLE

**Known gaps:**
- Healthcare
- Energy
- International developed markets (Samsung + Korea ETF partially fills but still concentrated)
- AI agent infrastructure
- Pure-play workflow automation
- Security (CRWD)

**Previously surfaced actionable move:** Build NOW (ServiceNow) - appeared across BVP, Goldman, a16z as clearest pure-play on Systems of Action thesis

---

## Rational Investing Strategy

The core stock evaluation framework applied to all analysis.

### 5 Key Attributes Checklist
Every prospective stock must pass all five:
1. **Topline Revenue Growth** - consistently growing overall sales
2. **EBITDA Growth** - core earnings and margins rising, proving revenue translates to operational profitability
3. **Strong Free Cash Flow** - healthy actual FCF; funds reinvestment, buybacks, or dividends without relying on debt
4. **Low Debt** - clean balance sheet that can withstand economic downturns
5. **Well-Priced** - stock trades at a discount to intrinsic value

### Financial Evaluation Process
- Analyze 3-statement financial models: Income Statement, Balance Sheet, Cash Flow
- **The One-Pager** - 9-year historical financial data condensed into a single summary sheet
- **Forecasting** - project EBITDA and cash flows forward

### Valuation and Return Calculation
- **Intrinsic Value** - DCF framework to determine fundamental worth
- **Target IRR** - 15% annualized hurdle rate
- **Estimate Return** - combine future stock price forecasts with DCF to confirm target rate of return is achievable

---

## Design Decisions

- **Exa runs first, Claude second** - cheaper (Exa per query vs. Claude per search), more reliable (controlled input), faster (one Claude call vs. tool-call loops)
- **Extended thinking on** - thesis synthesis benefits from reasoning depth
- **ChromaDB exclusion list** - prevents stale re-surfacing, keeps signal fresh week over week
- **No tools at Claude analysis time** - all retrieval is pre-done; Claude only synthesizes
