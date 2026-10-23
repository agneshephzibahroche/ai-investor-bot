# src/inference.py

import os
from functools import lru_cache

import numpy as np
import pandas as pd

from src.feature_engineering import FEATURES, compute_indicators
from src.market_data import get_history
from src.nn import load_bundle
from src.preprocessing import SEQ_LENGTH, normalize_window

ROOT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MODEL_DIR    = os.path.join(ROOT_DIR, "models")
UNIVERSAL    = "universal"
OHLCV        = ["Open", "High", "Low", "Close", "Volume"]
MAX_DAILY_MOVE = 0.10  # cap each predicted day at ±10%
HISTORY_ROWS = 300     # MA200 + the model window, with room to spare


def available_models():
    """Names of exported NumPy bundles in models/ (tickers and/or 'universal')."""
    if not os.path.isdir(MODEL_DIR):
        return []
    return sorted(fn[:-len(".npz")] for fn in os.listdir(MODEL_DIR) if fn.endswith(".npz"))


def model_for(ticker: str):
    """A ticker's own model if one was trained, otherwise the universal one."""
    models = available_models()
    if ticker in models:
        return ticker
    return UNIVERSAL if UNIVERSAL in models else None


@lru_cache(maxsize=16)
def load_resources(name: str):
    """(model, metadata) for a bundle in models/ (see src/nn.py)."""
    return load_bundle(os.path.join(MODEL_DIR, f"{name}.npz"))


def fetch_and_prepare(ticker: str):
    """Raw OHLCV history and the model-ready feature frame for a ticker."""
    raw, _ = get_history(ticker, "5y")
    if raw is None or raw.empty:
        raise ValueError(f"No price data returned for {ticker}")

    features = compute_indicators(raw).dropna()[FEATURES]
    if len(features) < SEQ_LENGTH:
        raise ValueError(f"Not enough data for {ticker}")
    return raw, features


def predict_return(model, window: np.ndarray, target_scale: float) -> float:
    """Next-day log return predicted from one raw (SEQ_LENGTH, 11) window."""
    X = normalize_window(window)[np.newaxis]
    return float(model.predict(X, verbose=0)[0, 0]) / target_scale


def recursive_forecast(raw: pd.DataFrame, model, horizon_days: int,
                       target_scale: float = 10.0):
    """
    Forecast closes day by day: predict tomorrow's return, append that day to
    the price history, recompute the indicators and repeat.
    """
    hist = raw[OHLCV].dropna().tail(HISTORY_ROWS).copy()
    last_close = float(hist["Close"].iloc[-1])
    avg_volume = float(hist["Volume"].tail(20).mean())
    future = pd.bdate_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=horizon_days)
    preds = []

    for day in future:
        window = compute_indicators(hist).dropna()[FEATURES].values[-SEQ_LENGTH:]
        r = np.clip(predict_return(model, window, target_scale), -MAX_DAILY_MOVE, MAX_DAILY_MOVE)
        # Keep the whole path within ⅓× and 3× the last actual close
        close = float(np.clip(window[-1, 0] * np.exp(r), last_close * 0.33, last_close * 3.0))
        preds.append(close)
        hist.loc[day] = [close, close, close, close, avg_volume]

    return preds
