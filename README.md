---
title: AI Investor Bot
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 5000
pinned: false
---

# AI Investor Bot

A Flask web app that forecasts a stock's closing price with a per-ticker LSTM
and turns the forecast into a Buy / Hold / Sell signal, explained with
technical indicators (MA50/MA200, MACD, RSI) and optional news sentiment.

> For educational use only. This is not financial advice.

## How it works

1. `src/feature_engineering.py` downloads price history from Yahoo Finance and computes the indicators.
2. `src/preprocessing.py` scales the 11 features and builds 60-day sequences.
3. `src/train.py` trains an LSTM per ticker and saves it to `models/`.
4. `src/webapp/` serves the form, runs a day-by-day forecast (`src/inference.py`) and renders the advice (`src/advisor.py`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
```

## Train models

The app only offers tickers that have a trained model in `models/`.

```bash
# one ticker
python -m src.preprocessing --ticker AAPL
python -m src.train --ticker AAPL

# ten popular tickers
python scripts/batch_train_sp500_common.py
```

## Run

```bash
flask --app src.webapp.app run
```

Then open http://127.0.0.1:5000.

| Variable       | Purpose                                                        |
| -------------- | -------------------------------------------------------------- |
| `SECRET_KEY`   | Flask session key. Required when `FLASK_ENV=production`.       |
| `NEWSAPI_KEY`  | Optional. Enables news sentiment via [NewsAPI](https://newsapi.org). |
| `DATABASE_URL` | Optional. Defaults to a local SQLite file.                     |

### Docker

```bash
SECRET_KEY=change-me docker compose up --build
```

`models/` and `data/` are mounted read-only from the host, so train first.

## Tests

```bash
pip install pytest
pytest
```

## Deploy to Hugging Face Spaces

The block at the top of this file configures a Docker Space. Train the models
you want to offer first (they are copied into the image from `models/`), create
a Space with the **Docker** SDK, add `SECRET_KEY` (and optionally `NEWSAPI_KEY`)
under *Settings → Variables and secrets*, then upload the project (needs `pip install huggingface_hub` and `hf auth login`):

```bash
python -c "from huggingface_hub import upload_folder; upload_folder(repo_id='<username>/<space-name>', repo_type='space', folder_path='.', ignore_patterns=['.venv/*', '.git/*', 'data/*', '**/__pycache__/*', 'instance/*', 'src/webapp/instance/*'])"
```

The Space's disk is reset on every restart, so the query log does not persist.

## Deploy to Render

`render.yaml` describes the service. Commit the trained models in `models/`,
push to GitHub, then in Render choose **New → Blueprint** and pick this repo.
`SECRET_KEY` is generated automatically; add `NEWSAPI_KEY` under *Environment*
if you want news sentiment.
