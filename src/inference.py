import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

from src.feature_engineering import compute_indicators

ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SEQ_LENGTH  = 60
MODEL_DIR   = os.path.join(ROOT_DIR, "models")


def load_resources(ticker: str):
    """
    Load the trained scaler(s) and model for the given ticker.
    Returns:
        scaler_all   : MinMaxScaler fitted on all features
        scaler_close : MinMaxScaler fitted only on Close column
        model        : Trained LSTM model
    """
    scaler_all   = joblib.load(os.path.join(MODEL_DIR, f"scaler_{ticker}.save"))
    scaler_close = joblib.load(os.path.join(MODEL_DIR, f"scaler_close_{ticker}.save"))
    model        = load_model(os.path.join(MODEL_DIR, f"lstm_{ticker}.h5"))
    return scaler_all, scaler_close, model


def fetch_and_prepare(ticker: str):
    """
    Download stock history, compute indicators, drop NaNs,
    and return last SEQ_LENGTH rows of features.
    """
    lookback = SEQ_LENGTH + 200
    days = lookback * 2
    end   = datetime.today()
    start = end - timedelta(days=days)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True
    )

    df = compute_indicators(df)
    df = df.dropna()

    if len(df) < SEQ_LENGTH:
        raise ValueError(f"Not enough data for {ticker}")

    window_df = df.tail(SEQ_LENGTH)
    return window_df.values, df


def recursive_forecast(features: np.ndarray,
                       scaler_all,
                       scaler_close,
                       model,
                       horizon_days: int):
    window = features.copy()
    preds  = []

    for _ in range(horizon_days):
        scaled_win  = scaler_all.transform(window)
        X           = scaled_win.reshape(1, SEQ_LENGTH, -1)
        scaled_pred = model.predict(X, verbose=0)[0, 0]

        # ✅ Inverse transform with clipping
        inv = scaler_close.inverse_transform([[scaled_pred]])[0, 0]
        inv = max(0, inv)  # no negatives allowed

        preds.append(inv)

        # Update window with new predicted close
        new_row = window[-1].copy()
        new_row[0] = inv
        window = np.vstack([window[1:], new_row])

    return preds