// ═══════════════════════════════════════════════════════════════
// server.js — Express backend for Portfolio Intelligence
//
// Single endpoint: POST /analyze
//   1. Fetches ~24 months of daily closes for each ticker (yahoo-finance2)
//   2. Computes pairwise Pearson correlations on daily returns
//   3. Calls gpt-4o-mini with positions + correlation data
//   4. Returns { positions, portfolio_summary, correlations, corrWarnings }
// ═══════════════════════════════════════════════════════════════

require('dotenv').config();
const express = require('express');
const OpenAI = require('openai');
const YahooFinance = require('yahoo-finance2').default;
const yahooFinance = new YahooFinance(); // new is needed beacuse it is a class.

// ── App & client setup ──
const app = express(); //uses the express.
app.use(express.json()); //express should uese
app.use(express.static('public')); // serves public/index.html as the frontend (index.html is the default)

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY }); // the Open AI API key

// ── Caches (1hr TTL each) ──
const priceCache = new Map(); // { dates[], closes[] }
const metaCache  = new Map(); // { sector, country }

// Fetches real sector + country for a ticker via quoteSummary assetProfile.
// Degrades to nulls silently — the LLM falls back to training knowledge for that ticker.
async function fetchMeta(ticker) {
  const key = ticker.toUpperCase();
  const hit = metaCache.get(key); // holds the datapoints called from Yahoo. 
  if (hit && Date.now() - hit.ts < 3_600_000) return hit.data; //collects new data if the cache has expired (1-hr) & is true 
 
  try {
    const summary = await yahooFinance.quoteSummary(key, { modules: ['assetProfile'] }); //Collects data from the JSON-object assetProfile
    const profile = summary.assetProfile || {}; 

    //gets data about sector and country, so Open AI doesn't have to guess.
    const data = { 
      sector:  profile.sector  || null,
      country: profile.country || null
    };
    //sets the metacache to contain key/ticker, date now and data (sector and country)
    metaCache.set(key, { ts: Date.now(), data });
    return data;
  } catch {
    return { sector: null, country: null }; // non-critical — don't break the analysis
  }
}

// Fetches last ~24 months of daily closes for a single ticker.
// Returns { dates: string[], closes: number[] }, sorted ascending.
async function fetchCloses(ticker) {
  const key = ticker.toUpperCase();
  const hit = priceCache.get(key);
  if (hit && Date.now() - hit.ts < 3_600_000) return hit.data; // cache hit

  const period2 = new Date(); // End date - today
  const period1 = new Date(); // start date - 24 days from today
  period1.setMonth(period1.getMonth() - 24); // defines period now - 24 months

  // the actual API-call
  let quotes;
   // build-in method that handles http-req function by alling historical function and gets the data on the stocks in the interval we need the stocks
  try {
    quotes = await yahooFinance.historical(key, {
      period1,
      period2,
      interval: '1d'
    });
  } catch (err) {
    // Yahoo Finance returns HTTP 429 when rate-limited — surface it with a clear message
    if (err.message?.includes('429') || err.statusCode === 429) {
      const e = new Error(`Yahoo Finance rate limit hit for ${key} — try again in a moment`);
      e.statusCode = 429;
      throw e;
    }
    throw err;
  }

  //min. data requirements for rho (pearson function)
  const valid = quotes.filter(q => q.close !== null); // looping over every stock quotes, and filtering for null. 
  if (valid.length < 3) throw new Error(`Not enough price data for ${key}`);


  const dates = valid.map(q => q.date.toISOString().split('T')[0]); //gets the date without the time (splits at T). e.g. "2024-04-03T00:00:00.000Z"
  const closes = valid.map(q => q.close); //close price
  const data = { dates, closes }; //necessary data to calculate 
  priceCache.set(key, { ts: Date.now(), data }); //the timestamp (ts) of when the data was cached
  return data;
}


// Intersects trading dates across any number of tickers so calculations run on the same days.
// Used for pairwise correlations (2 tickers) and portfolio volatility (all tickers).
// Returns { [ticker]: closes[] } aligned to common dates only.
function alignPrices(priceMap) {
  const tickers = Object.keys(priceMap); //ticker names from the input priceMap
  const sets = tickers.map(t => new Set(priceMap[t].dates)); //fast, removes duplicates and unsorted (finds every useful dates), converts it into a =(1).has() lookups (Prepares filterquery - Power Automate) in a new array.
  const common = [...sets[0]].filter(d => sets.every(s => s.has(d))).sort(); //finds the dates where all tickers has the same a value (open market days)
  const aligned = {};
  for (const t of tickers) {
    const lookup = Object.fromEntries(priceMap[t].dates.map((d, i) => [d, priceMap[t].closes[i]])); //Potential output[["2025-01-02", 182.5], ["2025-01-03", 199.6]] - finds date and close price for each ticker.
    aligned[t] = common.map(d => lookup[d]); //creates arrray with all the close prices that is in the common (where both tickers are open.)
  }
  return aligned; // returns both tickers close prices.
}

// Computes Pearson correlation coefficient between two price series.
// Converts closes to daily returns first (% change), then correlates.
// Returns a value in [-1, 1] rounded to 2 decimal places, or null if too few points.
function pearson(a, b) { //input close price for each ticker - a = [170.00, 172.50, 171.00, 174.00, 173.50]
  const n = Math.min(a.length, b.length);
  if (n < 3) return null; //minimum amount for calculating rho-value.
  const ra = [], rb = [];

   //finds the percentage for each stock.
  for (let i = 1; i < n; i++) {
    ra.push((a[i] - a[i - 1]) / a[i - 1]);
    rb.push((b[i] - b[i - 1]) / b[i - 1]);
  }

  //find the mean/avg. value
  const ma = ra.reduce((s, v) => s + v, 0) / ra.length; 
  const mb = rb.reduce((s, v) => s + v, 0) / rb.length;

  let num = 0, sa = 0, sb = 0;
  for (let i = 0; i < ra.length; i++) {
    num += (ra[i] - ma) * (rb[i] - mb); //covariance numerator
    //variance for each ticker
    sa += (ra[i] - ma) ** 2;
    sb += (rb[i] - mb) ** 2;
  }

  const denom = Math.sqrt(sa * sb); //the sum of the two Standard Deviation
  return denom < 1e-10 ? 0 : Math.round((num / denom) * 100) / 100; // returns the Pearson value if the variance is not zero
}

// Maps a rho value to a human-readable risk label used by the frontend for styling.
// Computes annualised volatility (σ) from a closes array.
// Formula: stddev of daily returns × √252 (trading days per year), expressed as %.
// This is the standard measure of a stock's price risk.
function volatility(closes) {
  const n = closes.length; 
  if (n < 3) return null; // //minimum amount for calculating volatility-value.
  const returns = []; 

    //finds the percentage for each stock.
  for (let i = 1; i < n; i++) {
    returns.push((closes[i] - closes[i - 1]) / closes[i - 1])
  };
  //find the mean/avg. value
  const mean = returns.reduce((s, v) => s + v, 0) / returns.length;

  //calculates the variance
  const variance = returns.reduce((s, v) => s + (v - mean) ** 2, 0) / (returns.length - 1);

  //returns the volatility
  return Math.round(Math.sqrt(variance) * Math.sqrt(252) * 10000) / 100; // annualised %, 2dp
}

// Aggregates position weights by a metadata field (sector or country) into
// a sorted [{ name, percentage }] array. Positions missing the field are
// grouped under "Other". Percentages are rounded to 1dp and sum to 100%.
function computeBreakdown(positions, metaMap, field) {
  const totals = {};
  let totalWeight = 0;
  for (const p of positions) {
    const key = p.ticker.toUpperCase();
    const name = (metaMap[key] && metaMap[key][field]) || 'Other';
    totals[name] = (totals[name] || 0) + p.weight;
    totalWeight += p.weight;
  }
  if (totalWeight === 0) return [];
  return Object.entries(totals)
    .map(([name, w]) => ({ name, percentage: Math.round((w / totalWeight) * 1000) / 10 }))
    .sort((a, b) => b.percentage - a.percentage);
}

//Rating for each rho value.
function corrLevel(rho) {
  if (rho >= 0.8) return 'Very high';
  if (rho >= 0.5) return 'High';
  if (rho >= 0.2) return 'Moderate';
  if (rho >= -0.2) return 'Low';
  return 'Negative';
}

// Maps annualised portfolio volatility (σ%) to a 5-level risk label.
// Thresholds are calibrated for diversified portfolios (not individual stocks).
// S&P 500 long-run σ ~16% intentionally lands in "Moderate".
function riskLevel(sigma) {
  if (sigma > 25) return 'Aggressive';
  if (sigma >= 18) return 'Moderate-Aggressive';
  if (sigma >= 12) return 'Moderate';
  if (sigma >= 8)  return 'Moderate-Conservative';
  return 'Conservative';
}

// Maps annualised individual-stock volatility (σ%) to the 3-level schema used by the LLM.
// Kept separate from riskLevel() — "Aggressive" is portfolio language, not stock language.
// Thresholds: < 15% → Low, 15–30% → Mid, > 30% → High (professional equity risk standards).
function positionRisk(sigma) {
  if (sigma > 30) return 'High';
  if (sigma >= 15) return 'Mid';
  return 'Low';
}

// Maps sector weight (%) to a concentration label.
// Thresholds: < 10% → Under-weight, 10–25% → Balanced, 25–40% → Concentrated, > 40% → Over-exposed.
function sectorConcentration(pct) {
  if (pct > 40) return 'Over-exposed';
  if (pct >= 25) return 'Concentrated';
  if (pct >= 10) return 'Balanced';
  return 'Under-weight';
}

// ── LLM system prompt ──
// Keeps the analyst persona and σ-based risk rules.
// The output schema is enforced via function calling (analyzeTool below), not the prompt.
const systemPrompt = `You are a portfolio risk analyst. You will receive stock positions with verified sector and country metadata fetched directly from Yahoo Finance — treat these as ground truth and use them for the sector and geography breakdowns. Each position includes its annualised volatility (σ) — use this as the primary basis for your risk narrative.`;

// ── GET /search ──
// Thin wrapper around yahoo-finance2's search() method.
// Returns up to 6 equity matches as [{ ticker, name }] for the autocomplete dropdown.
app.get('/search', async (req, res) => {
  const q = (req.query.q || '').trim();
  if (q.length < 2) return res.json([]);
  try {
    const result = await yahooFinance.search(q);
    const hits = (result.quotes || [])
      .filter(r => r.quoteType === 'EQUITY' && r.symbol && r.longname)
      .slice(0, 6)
      .map(r => ({ ticker: r.symbol, name: r.longname }));
    res.json(hits);
  } catch (err) {
    res.json([]); // degrade silently — search is non-critical
  }
});

// ── GET /correlate ──
// Fetches 24 months of closes for two tickers, computes Pearson rho,
// then asks gpt-4o-mini for a 1-2 sentence plain-English explanation.
// Query params: ?a=AAPL&b=TSLA
app.get('/correlate', async (req, res) => {
  const a = (req.query.a || '').trim().toUpperCase(); // potential output: /correlate?a=AAPL
  const b = (req.query.b || '').trim().toUpperCase();
  if (!a || !b) return res.status(400).json({ error: 'Provide ?a=X&b=Y' });

  let dataA, dataB;
  try {
    [dataA, dataB] = await Promise.all([fetchCloses(a), fetchCloses(b)]); // gets close prices for each ticker
  } catch (err) {
    return res.status(err.statusCode || 400).json({ error: err.message }); 
  }

  const aligned = alignPrices({ [a]: dataA, [b]: dataB }); // sends the close prices for each ticker to the alignPairs for it to run the function and stored in the variable.
  const rho = pearson(aligned[a], aligned[b]); //finds the rho value for the two stocks using the pearson function and stores it into rho.
  if (rho === null) return res.status(400).json({ error: 'Not enough overlapping price data' });
  const level = corrLevel(rho); 

  // Ask the LLM for a concise plain-English reason for the correlation.
  // Uses function calling so the model can request verified sector/country metadata
  // from Yahoo Finance rather than relying on training-data guesses.
  let lookUpNote = '';
  try {
    const tools = [{
      type: 'function',
      function: {
        name: 'get_ticker_info',
        description: 'Returns verified sector and country for a stock ticker from Yahoo Finance.', //when to use this tool (when the AI is unsure if it's right or wrong.)
        parameters: {
          type: 'object',
          properties: {
            ticker: { type: 'string', description: 'The stock ticker symbol, e.g. MAERSK-B.CO' }
          },
          required: ['ticker']
        }
      }
    }];

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: `${a} and ${b} have a Pearson correlation of ${rho} (${level}) based on 24 months of daily returns. Use get_ticker_info to look up both tickers, then in 1-2 sentences explain why their price movements are correlated to this degree. Be specific about sector, macro, or business drivers. No filler phrases.` }
    ];

    // First call — model may request ticker metadata via tool calls
    let response = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages,
      tools,
      temperature: 0.3,
      max_tokens: 200
    });

    // Resolve all tool calls the model requested (typically 1-2 for the two tickers)
    while (response.choices[0].finish_reason === 'tool_calls') {
      const assistantMsg = response.choices[0].message; //takes the first response's message when the ticker has been found
      messages.push(assistantMsg); //pushes the message into the response.messages

      for (const call of assistantMsg.tool_calls) {
        const { ticker } = JSON.parse(call.function.arguments); //Gets ticker on the two stocks that is being paired
        const meta = await fetchMeta(ticker);//returns the sector and country and cashes for 1 hour.
        messages.push({
          role: 'tool',
          tool_call_id: call.id, //find the right tool_call from the ticker
          content: JSON.stringify({ ticker, sector: meta.sector, country: meta.country }) //prepares the data for the repsonse (second call)
        });
      }

      // Second call — model now has grounded metadata and writes the explanation
      response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages,
        tools,
        temperature: 0.3,
        max_tokens: 120
      });
    }

    lookUpNote = response.choices[0].message.content.trim(); //The final plaintext that is ready to be used on the application.
  } catch (err) {
    if (err.status === 429) return res.status(429).json({ error: 'OpenAI rate limit hit — try again in a moment' });
    // other LLM errors are non-critical; return rho without lookUpNote
  }

  res.json({ a, b, rho, level, lookUpNote });
});



// ── Function-calling tool for /analyze ──
// Forces the model to return a strictly-typed portfolio analysis object.
// tool_choice pins it to this function so the response always comes back as tool_calls.
const analyzeTool = {
  type: 'function',
  function: {
    name: 'submit_portfolio_analysis',
    description: 'Submit the complete portfolio risk analysis result.',
    parameters: {
      type: 'object',
      properties: {
        positions: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              ticker:          { type: 'string' },
              risk_reason:     { type: 'string', description: "1-2 sentences about why the risk reason. e.g. Annualised volatility of 17.4% reflects stable earnings and strong buyback program." },
              key_risk_factor: { type: 'string', description: '2-4 words' }
            },
            required: ['ticker', 'risk_reason', 'key_risk_factor']
          }
        },
        portfolio_summary: {
          type: 'object',
          properties: {
            recommendation:   { type: 'string', description: '2-3 sentences about what the investor should be aware of, and recommend different assets' },
            correlation_note: { type: 'string', description: '2-3 sentences about the most important correlation risks, mentioning specific tickers and rho values if available' }
          },
          required: ['recommendation', 'correlation_note']
        }
      },
      required: ['positions', 'portfolio_summary']
    }
  }
};

// ── POST /analyze ──
// Main route: receives { positions: [{ ticker, name, weight }] } from the frontend.
app.post('/analyze', async (req, res) => {
  const { positions } = req.body; // takes the positions from the user input
  if (!positions || !Array.isArray(positions) || positions.length === 0) { // must be an array and exists
    return res.status(400).json({ error: 'positions must be a non-empty array' });
  }

  // ── Step 1: Fetch prices, metadata & compute pairwise correlations ──
  // fetchCloses and fetchMeta run in parallel per ticker.
  // Failed tickers are skipped (corrWarnings) rather than aborting the whole request.
  let correlations = [];
  const corrWarnings = [];
  let volMap  = {};
  let metaMap = {};
  let portfolioVolatility = null;
  let overallRisk = null;

  {
    const [priceResults, metaResults] = await Promise.all([
      Promise.allSettled(positions.map(p => fetchCloses(p.ticker))), //calls and uses the fetchCloses function to find the close prices.
      Promise.allSettled(positions.map(p => fetchMeta(p.ticker))) //calls and uses the metaCloses function to find sector and country. 
    ]);

    const priceMap = {};

    positions.forEach((p, i) => { //loops over the investors portfolio
      const key = p.ticker.toUpperCase(); 
      if (priceResults[i].status === 'fulfilled') { //only accepts tickers with a fulfilled status
        priceMap[key] = priceResults[i].value; // creates obejct for each ticker
        const vol = volatility(priceResults[i].value.closes); //uses the volatility for each ticker
        if (vol !== null) volMap[key] = vol;
      } else {
        const err = priceResults[i].reason;
        if (err.statusCode === 429) return res.status(429).json({ error: err.message });
        corrWarnings.push(`${p.ticker}: ${err.message}`);
      }
      // metaResults never rejects (fetchMeta swallows errors), so always fulfilled
      metaMap[key] = metaResults[i].value;
    });

    const tickers = Object.keys(priceMap);
    if (tickers.length >= 2) {
      // Align each pair independently — maximises data points per pair
      // (global alignment loses dates whenever any single ticker didn't trade)
      for (let i = 0; i < tickers.length; i++) {
        for (let j = i + 1; j < tickers.length; j++) {
          const a = tickers[i], b = tickers[j];
          const aligned = alignPrices({ [a]: priceMap[a], [b]: priceMap[b] });
          const rho = pearson(aligned[a], aligned[b]);
          if (rho !== null) {
            correlations.push({ pair: [a, b], rho, level: corrLevel(rho) }); //output [{ pair: ['AAPL', 'MSFT'], rho: 0.87,  level: 'Very high' }]
          }
        }
      }
      correlations.sort((a, b) => b.rho - a.rho); // highest rho first
    }

    // ── Portfolio-level volatility ──
    // Align all available tickers globally, build a weighted daily return series, annualise σ.
    const totalWeight = positions.reduce((s, p) => s + p.weight, 0);
    if (tickers.length >= 1 && totalWeight > 0) {
      const aligned = alignPrices(Object.fromEntries(tickers.map(t => [t, priceMap[t]])));
      const n = aligned[tickers[0]].length; //amount of common days
      if (n >= 3) {
        const portReturns = [];
        for (let i = 1; i < n; i++) {
          let r = 0; // weighted portfolio return for a single day
          for (const t of tickers) {
            const pos = positions.find(p => p.ticker.toUpperCase() === t);
            const w = pos ? pos.weight / totalWeight : 0;
            r += w * (aligned[t][i] - aligned[t][i - 1]) / aligned[t][i - 1];
          }
          portReturns.push(r); // accumulates weighted daily return for each trading day
        }
        const mean = portReturns.reduce((s, v) => s + v, 0) / portReturns.length;
        const variance = portReturns.reduce((s, v) => s + (v - mean) ** 2, 0) / (portReturns.length - 1);
        portfolioVolatility = Math.round(Math.sqrt(variance) * Math.sqrt(252) * 10000) / 100; //252 is the avg. amount of trading days. Quite precise estimate, as it differs from 250 to 253
        overallRisk = riskLevel(portfolioVolatility);
      }
    }
  }

  // ── Step 2: Build the LLM user message ──
  // Order: positions → metadata (what) → volatility (individual risk) → correlations (portfolio interaction)

  // Real sector + country from Yahoo assetProfile injected per position.
  // The LLM is instructed to treat these as ground truth over its training knowledge.
  const metaSummary = positions.map(p => { //potential output: "AAPL: sector: Technology, country: United States;
    const key = p.ticker.toUpperCase();
    const m = metaMap[key] || {};
    const parts = [];
    if (m.sector)  parts.push(`sector: ${m.sector}`);
    if (m.country) parts.push(`country: ${m.country}`);
    return parts.length ? `${key}: ${parts.join(', ')}` : null;
  }).filter(Boolean).join('; ');

  const volSummary = Object.keys(volMap).length //
    ? '\n\nAnnualised volatility (24-month daily returns): ' +
      Object.entries(volMap).map(([t, v]) => `${t} σ=${v}% (${positionRisk(v)})`).join(', ')
    : '';

  // Inlines correlation data so the LLM can reference specific rho values in its narrative.
  const corrSummary = correlations.length
    ? `\n\nPairwise correlations (~24 months daily returns): ${correlations
        .map(c => `${c.pair[0]}/${c.pair[1]} ρ=${c.rho} (${c.level})`)
        .join(', ')}`
    : '\n\nNo correlation data available — reason based on general sector knowledge.';

  const userMessage = `Analyze this portfolio:\n${positions
    .map(p => `- ${p.ticker} (${p.name}): ${p.weight}% weight`)
    .join('\n')}${metaSummary ? `\n\nVerified metadata from Yahoo Finance (use as ground truth for sector and geography): ${metaSummary}` : ''}${volSummary}${corrSummary}`;

  // ── Step 3: Call OpenAI and parse the JSON response ──
  let raw; // unparsed JSON string returned by the LLM
  try {
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage }
      ],
      //calls the analyze function
      tools: [analyzeTool],
      tool_choice: { type: 'function', function: { name: 'submit_portfolio_analysis' } },
      temperature: 0.3 //most likely answer - lower the risk for hallucination
    });
    raw = completion.choices[0].message.tool_calls[0].function.arguments; // raw data on each ticker. 
  } catch (err) {
    console.error('OpenAI API error:', err.message);
    if (err.status === 429) return res.status(429).json({ error: 'OpenAI rate limit hit — try again in a moment' });
    return res.status(502).json({ error: 'OpenAI API request failed', details: err.message });
  }

  let parsed; // raw parsed into a JS object — safe to access properties on
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    console.error('JSON parse error:', raw);
    return res.status(500).json({ error: 'LLM returned invalid JSON', raw });
  }

  // Compute sectors + geographies from real Yahoo metadata.
  const sectors     = computeBreakdown(positions, metaMap, 'sector')
    .map(s => ({ ...s, concentration: sectorConcentration(s.percentage) }));
  const geographies = computeBreakdown(positions, metaMap, 'country');

  const positionsWithMeta = (parsed.positions || []).map(p => ({
    ...p,
    risk_level: positionRisk(volMap[p.ticker.toUpperCase()] ?? 0),
    sector:     metaMap[p.ticker.toUpperCase()]?.sector  || p.sector,
    country:    metaMap[p.ticker.toUpperCase()]?.country || null
  }));

  res.json({
    ...parsed,
    positions: positionsWithMeta,
    portfolio_summary: {
      ...parsed.portfolio_summary,
      overall_risk: overallRisk,
      sectors,
      geographies,
      top_sector: sectors[0] ?? null,
    },
    correlations,
    corrWarnings,
    portfolioVolatility
  });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));
