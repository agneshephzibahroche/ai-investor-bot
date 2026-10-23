# src/market_data.py
"""
Live market data from Yahoo Finance's public JSON endpoints.

These are the same endpoints yfinance wraps, called directly with `requests`
so the web app stays small enough for serverless hosts. Responses are cached
in-process for a short time; on Vercel the API routes also set CDN cache
headers so most visitors never reach Yahoo at all.

If Yahoo is unreachable and TWELVEDATA_API_KEY is set, price history falls
back to Twelve Data (https://twelvedata.com, free tier).
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import pandas as pd
import requests

HOSTS   = ("https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com")
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"),
    "Accept": "application/json",
}
TIMEOUT    = 8
SPARK_MAX  = 20  # Yahoo rejects larger spark batches

_session = requests.Session()
_session.headers.update(HEADERS)


class MarketDataError(Exception):
    """Raised when no provider could return data for a request."""


class SymbolNotFound(MarketDataError):
    """The provider answered, but knows no such symbol."""


def ttl_cache(seconds):
    """Memoize a function's result per argument tuple for `seconds`."""
    def decorator(fn):
        store, lock = {}, threading.Lock()

        @wraps(fn)
        def wrapper(*args):
            now = time.monotonic()
            with lock:
                hit = store.get(args)
                if hit and now - hit[0] < seconds:
                    return hit[1]
            value = fn(*args)
            with lock:
                store[args] = (now, value)
                if len(store) > 512:  # keep warm serverless instances small
                    store.pop(next(iter(store)))
            return value

        wrapper.cache_clear = store.clear
        return wrapper
    return decorator


def _yahoo(path, params):
    last = None
    for host in HOSTS:
        try:
            r = _session.get(host + path, params=params, timeout=TIMEOUT)
            if r.status_code == 404:
                raise SymbolNotFound("Symbol not found")
            r.raise_for_status()
            return r.json()
        except MarketDataError:
            raise
        except (requests.RequestException, ValueError) as e:
            last = e
    raise MarketDataError(f"Yahoo Finance unavailable: {last}")


# ---------------------------------------------------------------- history

def _frame_from_chart(result):
    ts = result.get("timestamp") or []
    q  = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {"Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
         "Close": q.get("close"), "Volume": q.get("volume")},
        index=pd.to_datetime(ts, unit="s"),
        dtype=float,
    )
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    if adj is not None:
        # Same as yfinance's auto_adjust=True: scale OHLC by the split/dividend factor
        ratio = pd.Series(adj, index=df.index, dtype=float) / df["Close"]
        for col in ("Open", "High", "Low"):
            df[col] = df[col] * ratio
        df["Close"] = adj
    df = df.dropna(subset=["Close"])
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "Date"
    return df


def _twelvedata_history(symbol):
    key = os.getenv("TWELVEDATA_API_KEY")
    if not key:
        return None
    r = _session.get("https://api.twelvedata.com/time_series", timeout=TIMEOUT, params={
        "symbol": symbol.replace("-", "."), "interval": "1day",
        "outputsize": 5000, "apikey": key, "order": "ASC",
    })
    data = r.json()
    if data.get("status") != "ok":
        return None
    df = pd.DataFrame(data["values"]).set_index("datetime")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.rename(columns=str.capitalize)[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df


@ttl_cache(300)
def get_history(symbol, range_="5y"):
    """Daily, split/dividend-adjusted OHLCV history as a DataFrame."""
    symbol = symbol.upper()
    not_found = False
    try:
        data = _yahoo(f"/v8/finance/chart/{symbol}",
                      {"range": range_, "interval": "1d", "events": "div,split",
                       "includeAdjustedClose": "true"})
        result = (data.get("chart", {}).get("result") or [None])[0]
        if result and result.get("timestamp"):
            return _frame_from_chart(result), result.get("meta", {})
        not_found = True
    except SymbolNotFound:
        not_found = True
    except MarketDataError:
        pass
    fallback = _twelvedata_history(symbol)
    if fallback is not None and not fallback.empty:
        return fallback, {"symbol": symbol}
    if not_found:
        raise SymbolNotFound(f"We couldn't find a ticker called {symbol}.")
    raise MarketDataError(f"Market data for {symbol} is temporarily unavailable.")


# ---------------------------------------------------------------- quotes

def _quote_from_meta(meta, closes=None):
    price = meta.get("regularMarketPrice")
    prev  = meta.get("chartPreviousClose") or meta.get("previousClose")
    change = (price - prev) if price is not None and prev else None
    return {
        "symbol":   meta.get("symbol"),
        "name":     meta.get("longName") or meta.get("shortName") or meta.get("symbol"),
        "price":    price,
        "prev_close": prev,
        "change":   change,
        "change_pct": (change / prev * 100) if change is not None else None,
        "currency": meta.get("currency"),
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
        "type":     meta.get("instrumentType"),
        "day_high": meta.get("regularMarketDayHigh"),
        "day_low":  meta.get("regularMarketDayLow"),
        "volume":   meta.get("regularMarketVolume"),
        "high_52w": meta.get("fiftyTwoWeekHigh"),
        "low_52w":  meta.get("fiftyTwoWeekLow"),
        "market_time": meta.get("regularMarketTime"),
        "spark":    [c for c in (closes or []) if c is not None],
    }


@ttl_cache(60)
def _spark_batch(symbols):
    data = _yahoo("/v7/finance/spark",
                  {"symbols": ",".join(symbols), "range": "1d", "interval": "5m"})
    out = {}
    for item in (data.get("spark", {}).get("result") or []):
        resp = (item.get("response") or [None])[0]
        if not resp or "meta" not in resp:
            continue
        closes = resp.get("indicators", {}).get("quote", [{}])[0].get("close")
        out[item["symbol"]] = _quote_from_meta(resp["meta"], closes)
    return out


def get_quotes(symbols):
    """Latest price, daily change and an intraday sparkline for each symbol."""
    symbols = [s.upper() for s in dict.fromkeys(symbols) if s]
    chunks  = [tuple(symbols[i:i + SPARK_MAX]) for i in range(0, len(symbols), SPARK_MAX)]
    quotes  = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch in pool.map(_spark_batch, chunks):
            quotes.update(batch)
    return [quotes[s] for s in symbols if s in quotes]


def get_quote(symbol):
    quotes = get_quotes([symbol])
    if not quotes:
        raise MarketDataError(f"No quote for {symbol}")
    return quotes[0]


# ---------------------------------------------------------------- search & news

SEARCHABLE = {"EQUITY", "ETF", "INDEX", "CRYPTOCURRENCY", "MUTUALFUND", "CURRENCY", "FUTURE"}


@ttl_cache(3600)
def search(query):
    data = _yahoo("/v1/finance/search",
                  {"q": query, "quotesCount": 8, "newsCount": 0, "listsCount": 0})
    return [
        {"symbol": q["symbol"],
         "name": q.get("shortname") or q.get("longname") or q["symbol"],
         "exchange": q.get("exchDisp"),
         "type": q.get("typeDisp") or q.get("quoteType")}
        for q in data.get("quotes", [])
        if q.get("symbol") and q.get("quoteType") in SEARCHABLE
    ]


@ttl_cache(900)
def get_news(symbol, count=10):
    data = _yahoo("/v1/finance/search",
                  {"q": symbol, "quotesCount": 0, "newsCount": count, "listsCount": 0})
    items = []
    for n in data.get("news", []):
        if not n.get("title"):
            continue
        thumbs = (n.get("thumbnail") or {}).get("resolutions") or []
        small  = min(thumbs, key=lambda t: t.get("width", 1e9), default=None)
        items.append({
            "title":     n["title"],
            "publisher": n.get("publisher"),
            "link":      n.get("link"),
            "time":      n.get("providerPublishTime"),
            "thumbnail": small["url"] if small else None,
        })
    return items


@ttl_cache(3600)
def get_profile(symbol):
    """Sector / industry from the search endpoint (no crumb needed)."""
    try:
        data = _yahoo("/v1/finance/search", {"q": symbol, "quotesCount": 3, "newsCount": 0})
    except MarketDataError:
        return {}
    for q in data.get("quotes", []):
        if q.get("symbol", "").upper() == symbol.upper():
            return {"sector": q.get("sectorDisp") or q.get("sector"),
                    "industry": q.get("industryDisp") or q.get("industry"),
                    "type": q.get("typeDisp")}
    return {}
