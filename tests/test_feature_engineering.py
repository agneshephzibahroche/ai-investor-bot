import numpy as np
import pandas as pd

from src.feature_engineering import FEATURES, compute_indicators


def make_prices(n=300, multiindex=False):
    idx = pd.bdate_range("2023-01-02", periods=n)
    close = 100 + np.sin(np.arange(n) / 5.0) * 10 + np.arange(n) * 0.1
    df = pd.DataFrame(
        {
            "Close": close,
            "High": close + 1,
            "Low": close - 1,
            "Open": close - 0.5,
            "Volume": 1_000_000.0,
        },
        index=idx,
    )
    df.index.name = "Date"
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["TEST"]])
    return df


def test_compute_indicators_adds_all_features():
    out = compute_indicators(make_prices()).dropna()
    assert set(FEATURES) <= set(out.columns)
    assert len(out) == 300 - 199  # MA200 needs 200 rows
    assert out["RSI"].between(0, 100).all()


def test_compute_indicators_does_not_mutate_input():
    df = make_prices()
    compute_indicators(df)
    assert list(df.columns) == ["Close", "High", "Low", "Open", "Volume"]


def test_compute_indicators_handles_yfinance_multiindex():
    flat = compute_indicators(make_prices())
    multi = compute_indicators(make_prices(multiindex=True))
    assert list(multi.columns) == list(flat.columns)
    pd.testing.assert_frame_equal(multi, flat)
