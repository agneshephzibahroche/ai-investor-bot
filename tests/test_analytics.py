import numpy as np
import pandas as pd
import pytest

from src import analytics
from src.feature_engineering import compute_indicators
from tests.test_feature_engineering import make_prices


def long_prices(n=900, seed=3):
    idx = pd.bdate_range("2022-01-03", periods=n)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    return pd.DataFrame({"Close": close, "High": close * 1.01, "Low": close * 0.99,
                         "Open": close, "Volume": 1e6}, index=idx)


def test_monte_carlo_bands_are_ordered():
    close = long_prices()["Close"]
    mc = analytics.monte_carlo(close, 20)
    assert all(len(mc[k]) == 20 for k in ("p5", "p25", "p50", "p75", "p95"))
    for lo, mid, hi in zip(mc["p5"], mc["p50"], mc["p95"]):
        assert lo <= mid <= hi
    assert 0 <= mc["prob_up"] <= 1
    # The cone widens with the horizon
    assert mc["p95"][-1] - mc["p5"][-1] > mc["p95"][0] - mc["p5"][0]


def test_backtest_buy_and_hold_matches_price_change():
    df = compute_indicators(long_prices()).dropna()
    bt = analytics.backtest(df, years=2)
    names = [s["key"] for s in bt["strategies"]]
    assert names[0] == "buy_hold" and set(analytics.STRATEGIES) <= set(names)
    window = df[df.index > df.index[-1] - pd.DateOffset(years=2)]["Close"]
    assert bt["strategies"][0]["total_return"] == pytest.approx(window.iloc[-1] / window.iloc[0] - 1, abs=1e-4)
    for s in bt["strategies"]:
        assert len(s["equity"]) == len(bt["dates"])
        assert s["max_drawdown"] <= 0


def test_backtest_has_no_lookahead():
    # A signal seen at day k's close can only earn day k+1's return.
    df = compute_indicators(long_prices()).dropna()
    window = df[df.index > df.index[-1] - pd.DateOffset(years=1)]
    k = 100
    ma50 = window["MA200"] - 1
    ma50.iloc[k] = window["MA200"].iloc[k] + 1  # golden cross on day k only
    df.loc[window.index, "MA50"] = ma50
    gc = next(s for s in analytics.backtest(df, years=1, cost=0)["strategies"] if s["key"] == "golden_cross")
    close = window["Close"]
    assert gc["total_return"] == pytest.approx(close.iloc[k + 1] / close.iloc[k] - 1, abs=1e-4)
    assert gc["trades"] == 2


def test_scorecard_is_bounded_and_explained():
    sc = analytics.technical_scorecard(compute_indicators(make_prices()).dropna())
    assert -100 <= sc["score"] <= 100
    assert sc["rating"] in {"Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"}
    assert len(sc["signals"]) == 7 and all(s["detail"] for s in sc["signals"])


def test_risk_stats_and_returns():
    df = long_prices()
    stats = analytics.risk_stats(df["Close"], df["Close"])
    assert stats["beta"] == pytest.approx(1.0)
    assert stats["volatility"] > 0 and stats["max_drawdown_5y"] <= 0
    rets = analytics.period_returns(df["Close"])
    assert set(rets) >= {"1W", "1M", "1Y", "YTD"} and rets["5Y"] is None


def test_compare_rebases_and_correlates():
    a, b = long_prices(seed=1)["Close"], long_prices(seed=2)["Close"]
    out = analytics.compare({"A": a, "B": b, "A2": a * 2})
    assert out["series"]["A"][0] == 100 and out["series"]["A2"] == out["series"]["A"]
    assert out["correlation"]["A"]["A2"] == pytest.approx(1.0)
    assert out["correlation"]["A"]["A"] == 1.0


def test_lstm_track_record_compares_against_naive():
    class Flat:
        def predict(self, X, verbose=0):
            return np.zeros((len(X), 1))

    from src.feature_engineering import FEATURES
    from src.preprocessing import normalize_window
    feats = compute_indicators(long_prices()).dropna()[FEATURES]
    tr = analytics.lstm_track_record(feats, Flat(), normalize_window, 10.0, 60, days=50)
    assert len(tr["dates"]) == 50
    # Predicting zero return is exactly the naive baseline
    assert tr["mape"] == pytest.approx(tr["naive_mape"])


def test_portfolio_values_holdings_and_benchmark():
    a, b = long_prices(seed=1)["Close"], long_prices(seed=2)["Close"]
    out = analytics.portfolio({"A": a, "B": b}, {"A": 2, "B": 1}, benchmark=a)
    assert out["value"][0] == pytest.approx(2 * a.iloc[0] + b.iloc[0], abs=0.01)
    assert out["value"][-1] == pytest.approx(2 * a.iloc[-1] + b.iloc[-1], abs=0.01)
    assert out["benchmark"][0] == pytest.approx(out["value"][0], abs=0.01)  # same starting value
    assert sum(out["weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert -1 <= out["stats"]["avg_correlation"] <= 1


def test_portfolio_fills_weekend_gaps():
    stock = long_prices(n=300)["Close"]
    crypto_idx = pd.date_range(stock.index[0], stock.index[-1], freq="D")
    crypto = pd.Series(np.linspace(1, 2, len(crypto_idx)), index=crypto_idx)
    out = analytics.portfolio({"S": stock, "C": crypto}, {"S": 1, "C": 1})
    assert None not in out["value"] and len(out["dates"]) > 0
