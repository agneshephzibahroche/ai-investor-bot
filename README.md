---
title: AI Investor Bot
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 5000
pinned: false
---

# AI Investor

A stock research web app with live market data, LSTM price forecasts with
Monte Carlo confidence bands, strategy backtests, a technical scorecard and
news sentiment. It works for any stock, ETF, index or crypto that Yahoo
Finance lists.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/agneshephzibahroche/ai-investor-bot)

> For educational use only. This is not financial advice.

## Features

- **Markets dashboard**: live ticker tape; tiles with sparklines for indices, crypto, gold, oil, yields and FX; a sector heatmap; top gainers and losers among 40 mega-caps; a watchlist saved in your browser; and the tickers most forecast on the site.
- **Stock pages** (`/stock/NVDA`)
  - Candlestick or line chart with MA50/MA200, volume and an RSI or MACD panel.
  - Key stats: 52-week range, volatility, Sharpe, beta vs the S&P 500, drawdowns and trailing returns.
- **AI forecast**: a 1–12 week LSTM path drawn on top of a Monte Carlo cone (50% and 90% ranges), with a Buy/Hold/Sell signal, plain-English reasoning and the model's live track record against a naive baseline.
- **Technical scorecard**: seven classic signals combined into a −100…+100 gauge, each one explained.
- **Strategy backtest**: golden cross, MACD crossover, RSI mean reversion and trend following against buy & hold, over 1–8 years. Trading costs are included, and signals act on the next session, so there is no look-ahead.
- **News sentiment**: recent headlines scored with VADER plus a finance-specific word list.
- **Compare** (`/compare`): rebased performance for up to six tickers, a risk/return table and a correlation matrix, with presets such as Big tech or Crypto vs S&P.
- **Portfolio** (`/portfolio`): add holdings with an optional average cost to see total value, today's change, gain/loss, allocation, performance against the S&P 500, volatility, beta and diversification. Holdings are saved in your browser.
- **UI**: dark and light themes, a mobile layout, ticker search with autocomplete, and `/` to focus search.

## How it works

| Piece | File |
| --- | --- |
| Market data: Yahoo Finance JSON endpoints via `requests`, cached in-process; optional Twelve Data fallback | `src/market_data.py` |
| Indicators (MA50/200, MACD, RSI) | `src/feature_engineering.py` |
| Training data: 60-day windows normalised to the last close, next-day log return as the target | `src/preprocessing.py` |
| Model training (TensorFlow, offline) | `src/train.py` |
| Keras → NumPy export, scored on the held-out test split | `scripts/export_models.py` |
| LSTM forward pass in pure NumPy, used at runtime | `src/nn.py` |
| Recursive multi-day forecast | `src/inference.py` |
| Monte Carlo, backtests, risk stats, scorecard, compare | `src/analytics.py` |
| Signal and explanation, headline sentiment | `src/advisor.py` |
| Flask pages and JSON API | `src/webapp/` |

**Why NumPy at runtime?** TensorFlow is over 1 GB installed, far above
Vercel's 250 MB function limit. The trained weights are exported to
`models/universal.npz` (120 KB), and `src/nn.py` replays the LSTM in NumPy. It
matches Keras to within 1e-7, and the deployed dependencies come to about 125 MB.

**The model.** One "universal" LSTM (64 → 32 units) trained on ten years of 20
large caps and ETFs. Inputs are scale-free, so it can be applied to any ticker.
On its held-out test set it gets next-day direction right about 52% of the time
and matches the naive "tomorrow equals today" price error. That is realistic
for daily stock prediction, and the app says so instead of hiding it. Each
forecast also shows a Monte Carlo range and the model's recent track record.

### API

All endpoints return JSON and send `Cache-Control: s-maxage` headers, so
Vercel's CDN absorbs repeat traffic.

| Endpoint | |
| --- | --- |
| `GET /api/market` | Overview, sectors, movers, model stats, trending |
| `GET /api/quotes?symbols=AAPL,MSFT` | Quotes with intraday sparklines |
| `GET /api/search?q=nvidia` | Ticker search |
| `GET /api/analysis/<sym>` | History + indicators, stats, returns, scorecard |
| `GET /api/forecast/<sym>?weeks=4` | LSTM + Monte Carlo forecast, signal, advice |
| `GET /api/backtest/<sym>?years=3` | Strategy equity curves and stats |
| `GET /api/news/<sym>` | Headlines with sentiment |
| `GET /api/compare?symbols=A,B&range=1y` | Rebased series, stats, correlations |
| `GET /health` | Health check |

## Deploy to Vercel

1. Push this repo to GitHub, then import it at [vercel.com/new](https://vercel.com/new), or click the button above.
2. Keep the defaults. `vercel.json` routes every request to the Flask app in `api/index.py`, and Vercel installs `requirements.txt` on Python 3.12.
3. Optional environment variables:

| Variable | Purpose |
| --- | --- |
| `DATABASE_URL` | Persist the "trending" forecast log (e.g. a Neon/Supabase Postgres URL; add `psycopg[binary]` to `requirements.txt`). Without it, SQLite in `/tmp` is used and resets on cold starts. |
| `TWELVEDATA_API_KEY` | Fallback price history if Yahoo is unreachable. |
| `NEWSAPI_KEY` | Extra news source for sentiment when Yahoo has no headlines. |
| `SECRET_KEY` | Flask session key (required when `FLASK_ENV=production`). |

Or from the CLI:

```bash
npm i -g vercel
vercel --prod
```

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
flask --app src.webapp.app run
```

Then open http://127.0.0.1:5000.

## Retrain the model

Training needs TensorFlow, so it uses a separate requirements file:

```bash
pip install -r requirements-train.txt
python -m src.preprocessing           # 10 years of the 20-ticker universe
python -m src.train                   # writes models/lstm_universal.h5
python scripts/export_models.py       # writes models/universal.npz (what the app loads)
```

To give one ticker its own model, pass `--ticker NVDA` to the first two
commands and then run `python scripts/export_models.py NVDA`. The app prefers
a ticker's own model over the universal one.

## Tests

```bash
pip install pytest
pytest
```

The tests mock all market data, so they run offline.

## Other hosts

- **Docker**: `SECRET_KEY=change-me docker compose up --build`
- **Render**: `render.yaml` describes the service. Choose **New → Blueprint** and pick this repo.
- **Hugging Face Spaces**: the front matter at the top of this file configures a Docker Space.
