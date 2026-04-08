# Portfolio Intelligence Tool

LLM-powered portfolio analysis tool that takes a set of stock positions, fetches live price data from the NPM module "yahoo-finance2", and returns a risk breakdown with correlation analysis and an AI-generated recommendation.

**Live demo:** [portfoliointel.xyz](https://portfoliointel.xyz/)

---

## Why I built this

Built as part of a Founder's Associate application at Performativ. I used it as an opportunity to go hands-on with LLM integration in a real-world context, combining live market data with structured model output to produce something actually useful.

---

## How it works?

1. User enters stock positions with weights (e.g. AAPL 40%, MSFT 30%, NESN 30%)
2. The backend fetches 24 months of price history from Yahoo Finance and computes annualised volatility and pairwise Pearson correlations across all tickers
3. Sector and geography breakdowns are aggregated from Yahoo metadata
4. A structured OpenAI function call generates per-position risk narratives and a portfolio summary, constrained by the real volatility numbers so the model can't hallucinate risk levels
5. Everything is merged server-side and returned as a single JSON response

---

## Tech decisions worth noting

- **No framework or bundler** the scope didn't justify the overhead. Vanilla JS and a single Express server keeps the whole project readable in one sitting.
- **OpenAI function calling for structured output** the entire response shape is defined as a JSON Schema, so field types and allowed values are enforced at the API level rather than through prompt instructions or post-processing. The model can't return a risk or correlation level that isn't in the schema.
- **Server-side metric computation, LLM for narrative only** volatility, correlations, sector weights, and overall risk level are all computed from real data. The model only writes the text. This separation makes the output auditable.
- **Yahoo-Finance2 npm module with a 1-hour in-memory cache** avoids redundant API calls during a session without needing a database.


