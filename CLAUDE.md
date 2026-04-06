# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose
LLM-powered portfolio analysis tool — proof of concept for a Founder's Associate application at Performativ.com (B2B SaaS wealth management platform).

## Commands

```bash
npm start        # production: node server.js
npm run dev      # development: node --watch server.js (auto-restarts on file change)
```

Requires a `.env` file with `OPENAI_API_KEY=sk-...`.

## Architecture

Two-file project — no build step, no framework:

- **[server.js](server.js)** — Express backend with four endpoints (see below). All price data comes from `yahoo-finance2` with a 1-hour in-memory cache (`priceCache` Map). OpenAI `gpt-4o-mini` handles all natural language output.
- **[public/index.html](public/index.html)** — Single-file vanilla JS frontend. State lives in `positions[]` and `lastCorrelations[]`. No bundler, no framework.

### Backend endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/status` | Health check — uptime, cache state, OpenAI key presence |
| `GET` | `/search?q=` | Ticker autocomplete via `yahooFinance.search()` — returns `[{ticker, name}]`, EQUITY only, max 6 |
| `GET` | `/correlate?a=&b=` | Fetches 24 months of closes for two tickers, computes Pearson ρ, returns `{a, b, rho, level, lookUpNote}` where `lookUpNote` is a 1-2 sentence LLM explanation |
| `POST` | `/analyze` | Main analysis — see flow below |

### POST /analyze flow

1. **Fetch prices & metadata** — `fetchCloses()` and `fetchMeta()` run in parallel per ticker (both cached 1hr). Failed tickers are skipped into `corrWarnings` rather than aborting.
2. **Compute metrics** — `alignPrices()` intersects trading dates across all tickers (handles US/EU calendar differences). `pearson()` runs on aligned daily returns per pair. `volatility()` computes annualised σ = `stddev(returns) × √252`, expressed as %.
3. **Portfolio volatility** — Weighted daily return series built from all aligned tickers; annualised σ mapped to `overallRisk` via `riskLevel()`.
4. **Build LLM message** — Inlines verified sector/country from Yahoo (`metaSummary`), per-ticker volatility with `positionRisk()` label (`volSummary`), and pairwise correlations (`corrSummary`).
5. **Call OpenAI** — `gpt-4o-mini` via function calling (`analyzeTool`). The model outputs per-position risk narrative and portfolio summary text; `riskLevel` is constrained to `High | Mid | Low` via the tool schema enum.
6. **Merge & return** — Server-computed `sectors`, `geographies` (from `computeBreakdown()`), `overallRisk`, `correlations`, and `portfolioVolatility` are merged into the LLM response before sending to the client.

### Key shared functions

- `fetchCloses(ticker)` — Yahoo Finance historical fetch + cache. Returns `{ dates[], closes[] }`. Used by both `/analyze` and `/correlate`.
- `fetchMeta(ticker)` — Yahoo Finance `assetProfile` fetch + cache. Returns `{ sector, country }`. Degrades silently to nulls — the LLM falls back to training knowledge.
- `alignPrices(priceMap)` — date intersection across tickers. Critical for mixed US/EU portfolios.
- `pearson(a, b)` — correlation on daily returns, returns `[-1, 1]` rounded to 2dp or `null`.
- `volatility(closes)` — annualised σ in %, used to ground LLM risk ratings in real data.
- `computeBreakdown(positions, metaMap, field)` — aggregates position weights by `sector` or `country` into `[{ name, percentage }]`. Ungrouped positions fall under `"Other"`.
- `corrLevel(rho)` — maps ρ to `Very high / High / Moderate / Low / Negative` for frontend styling.
- `riskLevel(sigma)` — maps portfolio-level σ to a 5-level label (`Aggressive → Conservative`). Used only for `overallRisk`.
- `positionRisk(sigma)` — maps individual-stock σ to `High | Mid | Low`. Kept separate from `riskLevel()` — "Aggressive" is portfolio language, not stock language. Thresholds: `< 15% → Low`, `15–30% → Mid`, `> 30% → High` (professional equity risk standards). Injected server-side into each position after the LLM call — not part of the LLM schema.

### Frontend structure

- `renderPositions()` — builds position rows using DOM (not innerHTML) so `attachAutocomplete()` can reference elements directly.
- `attachAutocomplete(input, fetchFn)` — generic debounced dropdown (300ms). Used with `yahooSearch` (Yahoo Finance via `/search`) for position inputs and lookup inputs. Handles keyboard nav (arrows, Enter, Escape) and mousedown-before-blur ordering.
- `analyze()` — POSTs to `/analyze`, calls `renderResults()`.
- `renderResults()` — populates metric cards, position risk pills, sector/geo bar charts, correlation rows, LLM recommendation.
- `corrRowHTML(c)` — correlation bar spanning ρ -1 to +1. Center tick at 50% = ρ 0. `margin-left = 50 + min(0, rho)*50`, `width = |rho|*50`.
- `lookupCorr()` — async, calls `GET /correlate`, shows inline spinner, renders bar + LLM note.

## LLM JSON contract

`analyzeTool` in `server.js` enforces what the LLM outputs. `sectors`, `geographies`, and `overallRisk` are **not** in the LLM schema — they are computed server-side and merged in before the response is sent.

```json
{
  "positions": [
    {
      "ticker": "string",
      "riskLevel": "High | Mid | Low",
      "riskRreason": "string (1-2 sentences, e.g. 'Annualised volatility of 17.4% reflects...')",
      "keyRiskFactor": "string (2-4 words)"
    }
  ],
  "portfolioSummary": {
    "topConcentration": "string (e.g. '55% Technology — over-exposed')",
    "recommendation": "string (2-3 sentences)",
    "correlationNote": "string (2-3 sentences, references specific tickers and ρ values)"
  }
}
```

The frontend parses `topConcentration` with a regex to extract the leading percentage for the metric card.

## Input format

Position labels use `TICKER — Name` format (e.g. `AAPL — Apple Inc.`). `parsePosition()` splits on `—` to extract ticker and name. Autocomplete fills this format automatically on selection.

## Key constraints
- API key in `.env` only — never hardcoded
- Deployed on Digital Ocean
- Keep frontend in `public/index.html` — no framework, no bundler
- MVC or multi-file restructuring is intentionally deferred — the two-file constraint is right for the current scope
- `setMonth` is used for the 24-month lookback — be aware it can overflow on month-end dates; prefer `setFullYear(getFullYear() - N)` for changes to the lookback period
