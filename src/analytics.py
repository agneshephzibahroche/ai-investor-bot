# src/analytics.py
"""
Pure pandas/NumPy analytics behind the dashboard: risk statistics, a Monte
Carlo forecast cone, strategy backtests, a technical scorecard and
multi-ticker comparison. Everything here takes DataFrames from
`compute_indicators` and returns plain JSON-friendly values.
"""

import math

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _f(x, digits=4):
    """Round to a JSON-safe float (NaN/inf → None)."""
    if x is None:
        return None
    x = float(x)
    return round(x, digits) if math.isfinite(x) else None


def _dates(index):
    return [d.strftime("%Y-%m-%d") for d in index]


# ---------------------------------------------------------------- risk

def max_drawdown(close: pd.Series) -> float:
    peak = close.cummax()
    return float((close / peak - 1).min())


def period_returns(close: pd.Series) -> dict:
    """Trailing returns over common windows, as fractions."""
    last_date, last = close.index[-1], close.iloc[-1]
    out = {}
    for label, offset in [("1W", pd.DateOffset(weeks=1)), ("1M", pd.DateOffset(months=1)),
                          ("3M", pd.DateOffset(months=3)), ("6M", pd.DateOffset(months=6)),
                          ("1Y", pd.DateOffset(years=1)), ("3Y", pd.DateOffset(years=3)),
                          ("5Y", pd.DateOffset(years=5))]:
        past = close[close.index <= last_date - offset]
        out[label] = _f(last / past.iloc[-1] - 1) if len(past) else None
    ytd = close[close.index < pd.Timestamp(year=last_date.year, month=1, day=1)]
    out["YTD"] = _f(last / ytd.iloc[-1] - 1) if len(ytd) else None
    return out


def risk_stats(close: pd.Series, benchmark: pd.Series = None, risk_free: float = 0.04) -> dict:
    """Annualised volatility, Sharpe/Sortino, drawdown and beta over the last year."""
    year  = close[close.index > close.index[-1] - pd.DateOffset(years=1)]
    rets  = year.pct_change().dropna()
    vol   = rets.std() * math.sqrt(TRADING_DAYS)
    ann   = (1 + rets.mean()) ** TRADING_DAYS - 1
    down  = rets[rets < 0].std() * math.sqrt(TRADING_DAYS)
    stats = {
        "volatility": _f(vol),
        "sharpe":     _f((ann - risk_free) / vol if vol else None, 2),
        "sortino":    _f((ann - risk_free) / down if down else None, 2),
        "max_drawdown_1y": _f(max_drawdown(year)),
        "max_drawdown_5y": _f(max_drawdown(close)),
        "beta": None,
    }
    if benchmark is not None and len(benchmark):
        b = benchmark.pct_change().reindex(rets.index).dropna()
        joined = pd.concat([rets, b], axis=1, join="inner").dropna()
        if len(joined) > 20 and joined.iloc[:, 1].var():
            stats["beta"] = _f(joined.cov().iloc[0, 1] / joined.iloc[:, 1].var(), 2)
    return stats


# ---------------------------------------------------------------- forecasting

def monte_carlo(close: pd.Series, days: int, paths: int = 2000,
                lookback: int = TRADING_DAYS, seed: int = 7) -> dict:
    """
    Geometric-Brownian-motion forecast cone using the last year's drift and
    volatility. Returns percentile bands per future business day.
    """
    logret = np.log(close).diff().dropna().tail(lookback)
    mu, sigma = float(logret.mean()), float(logret.std())
    rng   = np.random.default_rng(seed)
    shock = rng.normal(mu, sigma, size=(paths, days))
    sims  = float(close.iloc[-1]) * np.exp(np.cumsum(shock, axis=1))

    bands = np.percentile(sims, [5, 25, 50, 75, 95], axis=0)
    final = sims[:, -1]
    last  = float(close.iloc[-1])
    return {
        "p5":  [round(v, 2) for v in bands[0]],
        "p25": [round(v, 2) for v in bands[1]],
        "p50": [round(v, 2) for v in bands[2]],
        "p75": [round(v, 2) for v in bands[3]],
        "p95": [round(v, 2) for v in bands[4]],
        "prob_up":       _f((final > last).mean(), 3),
        "expected_return": _f(final.mean() / last - 1),
        "daily_vol":     _f(sigma),
    }


def lstm_track_record(features: pd.DataFrame, model, normalize, target_scale: float,
                      seq_length: int, days: int = 120) -> dict:
    """
    Walk-forward check of the LSTM: for each of the last `days` sessions,
    predict that day's close from the 60 days before it and compare, next to
    a naive "same as yesterday" guess.
    """
    values = features.values
    n = min(days, len(values) - seq_length)
    if n <= 0:
        return None
    starts = range(len(values) - n - seq_length, len(values) - seq_length)
    X = np.stack([normalize(values[s:s + seq_length]) for s in starts])
    prev   = values[-n - 1:-1, 0]
    actual = values[-n:, 0]
    pred   = prev * np.exp(model.predict(X).ravel() / target_scale)
    moved  = actual != prev
    return {
        "dates":  _dates(features.index[-n:]),
        "actual": [round(float(v), 2) for v in actual],
        "predicted": [round(float(v), 2) for v in pred],
        "mape": _f(np.mean(np.abs(pred - actual) / actual)),
        "naive_mape": _f(np.mean(np.abs(prev - actual) / actual)),
        "direction_acc": _f(np.mean(np.sign(pred - prev)[moved] == np.sign(actual - prev)[moved]), 3),
    }


# ---------------------------------------------------------------- backtesting

STRATEGIES = {
    "golden_cross": ("Golden cross", "Hold while MA50 is above MA200, otherwise cash."),
    "macd":         ("MACD crossover", "Hold while MACD is above its signal line."),
    "rsi":          ("RSI mean reversion", "Buy when RSI < 30, sell when RSI > 70."),
    "trend":        ("Price above MA50", "Hold while the close is above its 50-day average."),
}


def _positions(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "golden_cross":
        return (df["MA50"] > df["MA200"]).astype(float)
    if name == "macd":
        return (df["MACD"] > df["Signal"]).astype(float)
    if name == "trend":
        return (df["Close"] > df["MA50"]).astype(float)
    if name == "rsi":
        pos, holding = [], 0.0
        for rsi in df["RSI"]:
            if rsi < 30:
                holding = 1.0
            elif rsi > 70:
                holding = 0.0
            pos.append(holding)
        return pd.Series(pos, index=df.index)
    raise KeyError(name)


def _curve_stats(equity: pd.Series, position: pd.Series = None) -> dict:
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    rets  = equity.pct_change().dropna()
    vol   = rets.std() * math.sqrt(TRADING_DAYS)
    stats = {
        "total_return": _f(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr":         _f((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1),
        "max_drawdown": _f(max_drawdown(equity)),
        "sharpe":       _f(rets.mean() * TRADING_DAYS / vol if vol else None, 2),
    }
    if position is not None:
        stats["exposure"] = _f(position.mean(), 3)
        stats["trades"]   = int((position.diff().abs() > 0).sum())
    return stats


def backtest(df: pd.DataFrame, years: int = 3, cost: float = 0.001) -> dict:
    """
    Compare simple indicator strategies with buy-and-hold over the last
    `years`. Signals act on the next session's return (no look-ahead) and each
    position change pays `cost` (10 bps by default).
    """
    df = df[df.index > df.index[-1] - pd.DateOffset(years=years)]
    daily = df["Close"].pct_change().fillna(0)

    # Sample curves to ~250 points so the payload stays small
    step = max(1, len(df) // 250)
    keep = df.index[::step].append(df.index[-1:]).unique()

    bh = (1 + daily).cumprod()
    out = {
        "start": df.index[0].strftime("%Y-%m-%d"),
        "dates": _dates(keep),
        "strategies": [{
            "key": "buy_hold", "name": "Buy & hold", "description": "Buy on day one and never sell.",
            "equity": [round(float(v), 4) for v in bh.loc[keep]],
            **_curve_stats(bh),
        }],
    }
    for key, (name, description) in STRATEGIES.items():
        signal = _positions(df, key)
        pos    = signal.shift(1).fillna(0)  # trade on the next bar
        trades = pos.diff().abs().fillna(pos.iloc[0])
        equity = (1 + pos * daily - trades * cost).cumprod()
        out["strategies"].append({
            "key": key, "name": name, "description": description,
            "equity": [round(float(v), 4) for v in equity.loc[keep]],
            "in_market": bool(signal.iloc[-1]),
            **_curve_stats(equity, pos),
        })
    return out


# ---------------------------------------------------------------- scorecard

RATINGS = [(50, "Strong Buy"), (15, "Buy"), (-15, "Neutral"), (-50, "Sell")]


def technical_scorecard(df: pd.DataFrame) -> dict:
    """
    Combine a handful of classic signals into a score from -100 (bearish) to
    +100 (bullish). Each signal reports its own reading so the score is
    explainable.
    """
    last, prev = df.iloc[-1], df.iloc[-2]
    close = df["Close"]
    signals = []

    def add(name, value, vote, detail):
        signals.append({"name": name, "value": value, "vote": vote, "detail": detail})

    add("Price vs MA50", _f(last.Close / last.MA50 - 1),
        1 if last.Close > last.MA50 else -1,
        f"Close is {'above' if last.Close > last.MA50 else 'below'} its 50-day average")
    add("Price vs MA200", _f(last.Close / last.MA200 - 1),
        1 if last.Close > last.MA200 else -1,
        f"Close is {'above' if last.Close > last.MA200 else 'below'} its 200-day average")

    crossed = (prev.MA50 <= prev.MA200) != (last.MA50 <= last.MA200)
    add("MA50 vs MA200", _f(last.MA50 / last.MA200 - 1),
        1 if last.MA50 > last.MA200 else -1,
        ("Golden cross" if last.MA50 > last.MA200 else "Death cross")
        + (" just happened" if crossed else " in effect"))

    add("MACD", _f(last.MACD - last.Signal, 3),
        1 if last.MACD > last.Signal else -1,
        f"MACD is {'above' if last.MACD > last.Signal else 'below'} its signal line")

    rsi = float(last.RSI)
    rsi_vote = 1 if rsi < 30 else -1 if rsi > 70 else (0.5 if rsi > 50 else -0.5)
    zone = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
    add("RSI (14)", _f(rsi, 1), rsi_vote, f"RSI {rsi:.1f} is in the {zone} zone")

    mom = close.iloc[-1] / close.iloc[-22] - 1 if len(close) > 22 else 0
    add("1-month momentum", _f(mom), 1 if mom > 0.02 else -1 if mom < -0.02 else 0,
        f"{mom:+.1%} over the last month")

    hi = close.tail(TRADING_DAYS).max()
    off_high = close.iloc[-1] / hi - 1
    add("Distance from 52w high", _f(off_high),
        1 if off_high > -0.05 else -1 if off_high < -0.25 else 0,
        f"{abs(off_high):.1%} below the 52-week high" if off_high < 0 else "At a 52-week high")

    score = round(100 * sum(s["vote"] for s in signals) / len(signals))
    rating = next((label for floor, label in RATINGS if score > floor), "Strong Sell")
    return {"score": score, "rating": rating, "signals": signals}


# ---------------------------------------------------------------- compare

def compare(closes: dict, risk_free: float = 0.04) -> dict:
    """Rebased performance, stats and return correlations for several tickers."""
    frame = pd.DataFrame(closes).dropna()
    if frame.empty:
        return {"dates": [], "series": {}, "stats": {}, "correlation": {}}
    rebased = frame / frame.iloc[0] * 100
    step = max(1, len(rebased) // 300)
    keep = rebased.index[::step].append(rebased.index[-1:]).unique()
    rets = frame.pct_change().dropna()
    stats = {}
    for col in frame:
        r = rets[col]
        vol = r.std() * math.sqrt(TRADING_DAYS)
        ann = (1 + r.mean()) ** TRADING_DAYS - 1
        stats[col] = {
            "return": _f(frame[col].iloc[-1] / frame[col].iloc[0] - 1),
            "volatility": _f(vol),
            "sharpe": _f((ann - risk_free) / vol if vol else None, 2),
            "max_drawdown": _f(max_drawdown(frame[col])),
        }
    corr = rets.corr().round(2)
    return {
        "dates": _dates(keep),
        "series": {c: [round(float(v), 2) for v in rebased.loc[keep, c]] for c in frame},
        "stats": stats,
        "correlation": {c: {k: _f(v, 2) for k, v in corr[c].items()} for c in corr},
    }


# ---------------------------------------------------------------- portfolio

def portfolio(closes: dict, shares: dict, benchmark: pd.Series = None) -> dict:
    """
    Value history and risk of a fixed set of holdings, as if they had been
    held for the whole period, next to the benchmark scaled to the same
    starting value.
    """
    # Crypto trades on weekends and stocks don't: carry the last price forward
    frame = pd.DataFrame(closes).sort_index().ffill().dropna()
    if frame.empty:
        return {"dates": [], "value": [], "benchmark": [], "stats": {}, "weights": {}}
    value = sum(frame[s] * shares[s] for s in frame)

    step = max(1, len(value) // 300)
    keep = value.index[::step].append(value.index[-1:]).unique()
    out = {
        "dates": _dates(keep),
        "value": [round(float(v), 2) for v in value.loc[keep]],
        "benchmark": [],
        # Weekends are flat fills for stocks; leaving them in would understate volatility
        "stats": {"return": _f(value.iloc[-1] / value.iloc[0] - 1),
                  **risk_stats(value[value.index.dayofweek < 5], benchmark)},
        "weights": {s: _f(frame[s].iloc[-1] * shares[s] / value.iloc[-1]) for s in frame},
    }
    if benchmark is not None:
        bench = benchmark.reindex(value.index).ffill().dropna()
        if len(bench):
            rebased = bench / bench.iloc[0] * value.iloc[0]
            out["benchmark"] = [round(float(v), 2) for v in rebased.reindex(keep).ffill()]
            out["stats"]["benchmark_return"] = _f(bench.iloc[-1] / bench.iloc[0] - 1)

    rets = frame[frame.index.dayofweek < 5].pct_change().dropna()
    if frame.shape[1] > 1 and len(rets) > 20:
        corr = rets.corr().values
        pairs = corr[np.triu_indices_from(corr, k=1)]
        out["stats"]["avg_correlation"] = _f(np.nanmean(pairs), 2)
    return out
