# src/webapp/routes.py

import os
import re
from datetime import timedelta
from functools import wraps

import pandas as pd
from flask import abort, current_app, jsonify, redirect, render_template, request, url_for
from newsapi import NewsApiClient

from src import analytics
from src.advisor import (compute_last_indicators, fetch_news_sentiment, format_advice,
                         get_signal, score_headline)
from src.feature_engineering import compute_indicators
from src.inference import (SEQ_LENGTH, UNIVERSAL, available_models, fetch_and_prepare,
                           load_resources, model_for, recursive_forecast)
from src.preprocessing import normalize_window
from src.market_data import (MarketDataError, SymbolNotFound, get_history, get_news,
                             get_profile, get_quotes, search)
from src.webapp.app import app
from src.webapp.models import Forecast, db

MAX_WEEKS        = 12   # each week is 5 sequential model calls
SIGNAL_THRESHOLD = 0.02
MAX_COMPARE      = 6
MAX_HOLDINGS     = 25
SYMBOL_RE        = re.compile(r"^[A-Z0-9^=.\-]{1,20}$")

MARKET_OVERVIEW = [
    ("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq"), ("^DJI", "Dow Jones"), ("^RUT", "Russell 2000"),
    ("^VIX", "VIX"), ("^TNX", "10Y Yield"), ("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"),
    ("GC=F", "Gold"), ("CL=F", "Crude Oil"), ("EURUSD=X", "EUR/USD"),
]
SECTORS = [
    ("XLK", "Technology"), ("XLC", "Communication"), ("XLY", "Consumer Disc."),
    ("XLF", "Financials"), ("XLV", "Health Care"), ("XLI", "Industrials"),
    ("XLP", "Consumer Staples"), ("XLE", "Energy"), ("XLB", "Materials"),
    ("XLU", "Utilities"), ("XLRE", "Real Estate"),
]
MOVERS_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "JNJ", "NFLX",
    "ABBV", "BAC", "CRM", "AMD", "KO", "PEP", "ORCL", "WMT", "CVX", "MRK",
    "ADBE", "TMO", "DIS", "INTC", "CSCO", "QCOM", "PLTR", "UBER", "NKE", "PYPL",
]

NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '')
newsapi     = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None


# ---------------------------------------------------------------- helpers

def get_available_tickers():
    """Tickers with a dedicated model (the universal model covers the rest)."""
    return [m for m in available_models() if m != UNIVERSAL]


def clean_symbol(raw):
    symbol = (raw or "").strip().upper()
    if not SYMBOL_RE.match(symbol):
        abort(400, description="Invalid ticker symbol.")
    return symbol


def cached(seconds):
    """Let Vercel's CDN (or any shared cache) serve repeat requests."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            resp = fn(*args, **kwargs)
            resp = current_app.make_response(resp)
            if resp.status_code == 200:
                resp.headers["Cache-Control"] = (
                    f"public, max-age=30, s-maxage={seconds}, stale-while-revalidate={seconds * 5}")
            return resp
        return wrapper
    return decorator


def series_json(s, digits=2):
    return [None if pd.isna(v) else round(float(v), digits) for v in s]


def parse_weeks(raw):
    try:
        weeks = int(raw)
    except (TypeError, ValueError):
        raise ValueError("Horizon must be a whole number of weeks.")
    if not 1 <= weeks <= MAX_WEEKS:
        raise ValueError(f"Horizon must be between 1 and {MAX_WEEKS} weeks.")
    return weeks


def headline_sentiment(symbol):
    """Yahoo headlines scored with VADER; NewsAPI is used too when configured."""
    try:
        items = get_news(symbol)
    except MarketDataError:
        items = []
    scored = [dict(n, sentiment=round(score_headline(n["title"]), 3)) for n in items]
    avg = sum(n["sentiment"] for n in scored) / len(scored) if scored else None
    if avg is None:
        try:
            avg = fetch_news_sentiment(newsapi, symbol)
        except Exception:
            current_app.logger.exception("NewsAPI lookup failed")
    return scored, avg


@app.errorhandler(MarketDataError)
def market_data_error(e):
    return jsonify(error=str(e)), 404 if isinstance(e, SymbolNotFound) else 502


@app.errorhandler(400)
@app.errorhandler(404)
def client_error(e):
    if request.path.startswith("/api/"):
        return jsonify(error=e.description), e.code
    return render_template("error.html", message=e.description, code=e.code), e.code


@app.errorhandler(500)
def server_error(e):
    if request.path.startswith("/api/"):
        return jsonify(error="Something went wrong. Please try again."), 500
    return render_template("error.html", message="Something went wrong.", code=500), 500


@app.context_processor
def inject_globals():
    return {"model_tickers": get_available_tickers(), "max_weeks": MAX_WEEKS}


# ---------------------------------------------------------------- pages

@app.route('/health')
def health():
    return {'status': 'ok', 'models': len(get_available_tickers())}, 200


@app.route('/')
def index():
    # Old links and the no-JS search form submit ?ticker=
    if request.args.get('ticker'):
        return redirect(url_for('stock', symbol=clean_symbol(request.args['ticker'])))
    return render_template('index.html')


@app.route('/stock/<path:symbol>')
def stock(symbol):
    symbol = clean_symbol(symbol)
    if symbol != request.view_args['symbol']:
        return redirect(url_for('stock', symbol=symbol))
    return render_template('stock.html', symbol=symbol)


@app.route('/compare')
def compare_page():
    return render_template('compare.html')


@app.route('/portfolio')
def portfolio_page():
    return render_template('portfolio.html')


# ---------------------------------------------------------------- API

@app.route('/api/search')
@cached(3600)
def api_search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify(results=[])
    return jsonify(results=search(q[:40]))


@app.route('/api/quotes')
@cached(60)
def api_quotes():
    symbols = [clean_symbol(s) for s in request.args.get('symbols', '').split(',') if s.strip()]
    return jsonify(quotes=get_quotes(symbols[:40]))


@app.route('/api/market')
@cached(60)
def api_market():
    overview = dict(MARKET_OVERVIEW)
    sectors  = dict(SECTORS)
    quotes   = {q["symbol"]: q for q in get_quotes(list(overview) + list(sectors) + MOVERS_UNIVERSE)}

    def labelled(names):
        return [dict(quotes[s], label=label) for s, label in names.items() if s in quotes]

    movers = sorted(({k: v for k, v in quotes[s].items() if k != "spark"} for s in MOVERS_UNIVERSE
                     if s in quotes and quotes[s]["change_pct"] is not None),
                    key=lambda q: q["change_pct"], reverse=True)

    models = [{"name": m, **load_resources(m)[1]} for m in available_models()]

    recent = (db.session.query(Forecast.ticker, db.func.count(Forecast.id).label("n"))
              .group_by(Forecast.ticker).order_by(db.desc("n")).limit(8).all())
    return jsonify(
        overview=labelled(overview),
        sectors=labelled(sectors),
        gainers=movers[:5],
        losers=movers[::-1][:5],
        models=models,
        trending=[{"symbol": t, "count": n} for t, n in recent],
    )


@app.route('/api/analysis/<path:symbol>')
@cached(300)
def api_analysis(symbol):
    symbol = clean_symbol(symbol)
    raw, meta = get_history(symbol, "5y")
    if len(raw) < 30:
        abort(404, description=f"Not enough price history for {symbol}.")

    df = compute_indicators(raw)
    quote = next(iter(get_quotes([symbol])), None) or {}
    try:
        spy = get_history("SPY", "5y")[0]["Close"] if symbol != "SPY" else None
    except MarketDataError:
        spy = None

    full = df.dropna()
    return jsonify(
        symbol=symbol,
        quote=quote,
        profile=get_profile(symbol),
        model=model_for(symbol),
        history={
            "dates":  [d.strftime("%Y-%m-%d") for d in df.index],
            "open":   series_json(df["Open"]),
            "high":   series_json(df["High"]),
            "low":    series_json(df["Low"]),
            "close":  series_json(df["Close"]),
            "volume": [None if pd.isna(v) else int(v) for v in df["Volume"]],
            "ma50":   series_json(df["MA50"]),
            "ma200":  series_json(df["MA200"]),
            "rsi":    series_json(df["RSI"], 1),
            "macd":   series_json(df["MACD"], 3),
            "signal": series_json(df["Signal"], 3),
        },
        returns=analytics.period_returns(df["Close"]),
        risk=analytics.risk_stats(df["Close"], spy),
        scorecard=analytics.technical_scorecard(full) if len(full) > 22 else None,
        currency=meta.get("currency") or quote.get("currency"),
    )


@app.route('/api/forecast/<path:symbol>')
@cached(600)
def api_forecast(symbol):
    symbol = clean_symbol(symbol)
    try:
        weeks = parse_weeks(request.args.get('weeks', 4))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    days = weeks * 5

    raw, _ = get_history(symbol, "5y")
    df = compute_indicators(raw)
    if df["MA200"].notna().sum() < 1:
        abort(404, description=f"Not enough price history to forecast {symbol}.")
    close = df["Close"].dropna()

    future = pd.bdate_range(start=close.index[-1] + timedelta(days=1), periods=days)
    mc = analytics.monte_carlo(close, days)
    out = {"symbol": symbol, "weeks": weeks,
           "dates": [d.strftime("%Y-%m-%d") for d in future],
           "monte_carlo": mc, "lstm": None}

    current = float(close.iloc[-1])
    target  = mc["p50"][-1]
    model_name = "monte_carlo"

    name = model_for(symbol)
    if name:
        model, meta = load_resources(name)
        raw, frame = fetch_and_prepare(symbol)
        preds = recursive_forecast(raw, model, days, meta["target_scale"])
        out["lstm"] = {
            "name": name,
            "path": [round(p, 2) for p in preds],
            "meta": meta,
            "in_training_set": symbol in meta["tickers"],
            "track_record": analytics.lstm_track_record(frame, model, normalize_window,
                                                        meta["target_scale"], SEQ_LENGTH),
        }
        target, model_name = preds[-1], "lstm"

    signal, pct = get_signal(current, target, SIGNAL_THRESHOLD)
    _, sentiment = headline_sentiment(symbol)
    indicators = compute_last_indicators(df.dropna())
    out.update(
        current=round(current, 2), target=round(float(target), 2), pct_change=round(pct, 4),
        signal=signal, model=model_name, sentiment=sentiment,
        advice=format_advice(current, target, pct, signal, indicators, sentiment, SIGNAL_THRESHOLD),
    )

    try:
        db.session.add(Forecast(ticker=symbol, weeks=weeks, recommendation=signal,
                                model=model_name, pct_change=pct))
        db.session.commit()
    except Exception:
        db.session.rollback()
        current_app.logger.exception("Could not save forecast")

    return jsonify(out)


@app.route('/api/backtest/<path:symbol>')
@cached(3600)
def api_backtest(symbol):
    symbol = clean_symbol(symbol)
    years = request.args.get('years', 3, type=int)
    years = min(max(years, 1), 8)
    raw, _ = get_history(symbol, "10y")
    df = compute_indicators(raw).dropna()
    if len(df) < 60:
        abort(404, description=f"Not enough price history to backtest {symbol}.")
    return jsonify(symbol=symbol, years=years, **analytics.backtest(df, years))


@app.route('/api/news/<path:symbol>')
@cached(900)
def api_news(symbol):
    symbol = clean_symbol(symbol)
    items, avg = headline_sentiment(symbol)
    return jsonify(symbol=symbol, sentiment=avg, items=items)


@app.route('/api/compare')
@cached(600)
def api_compare():
    symbols = [clean_symbol(s) for s in request.args.get('symbols', '').split(',') if s.strip()]
    symbols = list(dict.fromkeys(symbols))[:MAX_COMPARE]
    if len(symbols) < 2:
        return jsonify(error="Pick at least two tickers to compare."), 400
    range_ = request.args.get('range', '1y')
    if range_ not in {"6mo", "1y", "2y", "5y", "10y", "ytd"}:
        range_ = "1y"
    closes, missing = {}, []
    for s in symbols:
        try:
            closes[s] = get_history(s, range_)[0]["Close"]
        except MarketDataError:
            missing.append(s)
    return jsonify(range=range_, missing=missing, **analytics.compare(closes))


def parse_holdings(raw):
    """'AAPL:10,MSFT:2.5' → {'AAPL': 10.0, 'MSFT': 2.5}; raises ValueError."""
    holdings = {}
    for part in filter(None, (p.strip() for p in raw.split(','))):
        symbol, _, qty = part.partition(':')
        symbol = clean_symbol(symbol)
        try:
            shares = float(qty)
        except ValueError:
            raise ValueError(f"Shares for {symbol} must be a number.")
        if not 0 < shares < 1e9:
            raise ValueError(f"Shares for {symbol} must be positive.")
        holdings[symbol] = holdings.get(symbol, 0) + shares
    if not holdings:
        raise ValueError("Add at least one holding.")
    if len(holdings) > MAX_HOLDINGS:
        raise ValueError(f"A portfolio can have at most {MAX_HOLDINGS} holdings.")
    return holdings


@app.route('/api/portfolio')
@cached(600)
def api_portfolio():
    try:
        holdings = parse_holdings(request.args.get('holdings', ''))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    range_ = request.args.get('range', '1y')
    if range_ not in {"6mo", "1y", "2y", "5y", "ytd"}:
        range_ = "1y"
    closes, missing = {}, []
    for s in holdings:
        try:
            closes[s] = get_history(s, range_)[0]["Close"]
        except MarketDataError:
            missing.append(s)
    try:
        spy = get_history("SPY", range_)[0]["Close"]
    except MarketDataError:
        spy = None
    result = analytics.portfolio(closes, {s: holdings[s] for s in closes}, spy)
    return jsonify(range=range_, missing=missing, **result)
