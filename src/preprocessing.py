# src/preprocessing.py
"""
Build LSTM training data.

Each sample is a 60-day window of the 11 engineered features, normalised
relative to the window itself (see `normalize_window`), and the target is the
next day's log return. Because nothing depends on the absolute price level,
one model can be trained on many tickers and applied to any of them.
"""

import argparse
import os

import numpy as np

from src.feature_engineering import FEATURES, compute_indicators
from src.market_data import get_history

ROOT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR     = os.path.join(ROOT_DIR, "data", "processed")
SEQ_LENGTH   = 60
TARGET_SCALE = 10.0  # log returns are ~±0.03; scale them to a friendlier range

# Large, liquid names across sectors plus two index ETFs
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V",
    "LLY", "XOM", "UNH", "JNJ", "WMT", "PG", "HD", "KO", "SPY", "QQQ",
]

PRICE_COLS = [FEATURES.index(c) for c in ("Close", "High", "Low", "Open", "MA50", "MA200")]
VOL_COL    = FEATURES.index("Volume")
RET_COL    = FEATURES.index("Return")
MACD_COLS  = [FEATURES.index(c) for c in ("MACD", "Signal")]
RSI_COL    = FEATURES.index("RSI")


def normalize_window(window: np.ndarray) -> np.ndarray:
    """
    Scale-free view of one (SEQ_LENGTH, 11) window of raw features: prices
    relative to the last close, volume relative to its window average, MACD
    as a fraction of price and RSI centred on zero.
    """
    w = np.asarray(window, dtype=float)
    out = np.empty_like(w)
    last_close = w[-1, 0]
    out[:, PRICE_COLS] = w[:, PRICE_COLS] / last_close - 1
    vol = w[:, VOL_COL]
    out[:, VOL_COL] = np.log((vol + 1) / (vol.mean() + 1))
    out[:, RET_COL] = w[:, RET_COL] * 10
    out[:, MACD_COLS] = w[:, MACD_COLS] / last_close * 10
    out[:, RSI_COL] = w[:, RSI_COL] / 100 - 0.5
    return out


def build_sequences(features: np.ndarray):
    """(X, y, prev_close) for every window that has a next-day close."""
    X, y, prev = [], [], []
    for i in range(SEQ_LENGTH, len(features)):
        window = features[i - SEQ_LENGTH:i]
        X.append(normalize_window(window))
        y.append(np.log(features[i, 0] / window[-1, 0]) * TARGET_SCALE)
        prev.append(window[-1, 0])
    return np.array(X), np.array(y), np.array(prev)


def preprocess(tickers, name):
    """Chronological 70/15/15 split per ticker, concatenated across tickers."""
    os.makedirs(DATA_DIR, exist_ok=True)
    parts = {"train": [], "val": [], "test": []}
    for ticker in tickers:
        raw, _ = get_history(ticker, "10y")
        feats = compute_indicators(raw).dropna()[FEATURES].values
        X, y, prev = build_sequences(feats)
        n = len(X)
        a, b = int(0.7 * n), int(0.85 * n)
        for split, sl in (("train", slice(0, a)), ("val", slice(a, b)), ("test", slice(b, n))):
            parts[split].append((X[sl], y[sl], prev[sl]))
        print(f"{ticker}: {n} samples")

    for split, chunks in parts.items():
        X = np.concatenate([c[0] for c in chunks])
        y = np.concatenate([c[1] for c in chunks])
        prev = np.concatenate([c[2] for c in chunks])
        np.save(os.path.join(DATA_DIR, f"X_{split}_{name}.npy"), X.astype(np.float32))
        np.save(os.path.join(DATA_DIR, f"y_{split}_{name}.npy"), y.astype(np.float32))
        np.save(os.path.join(DATA_DIR, f"prev_{split}_{name}.npy"), prev)
        print(f"{split}: X {X.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ticker", help="Train data for one ticker (default: the whole universe)")
    args = parser.parse_args()
    if args.ticker:
        preprocess([args.ticker.upper()], args.ticker.upper())
    else:
        preprocess(UNIVERSE, "universal")
