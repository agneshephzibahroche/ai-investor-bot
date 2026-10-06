# src/inference.py

import os
import joblib
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

from src.feature_engineering import FEATURES, compute_indicators

ROOT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SEQ_LENGTH   = 60
MODEL_DIR    = os.path.join(ROOT_DIR, "models")
HISTORY_DAYS = 5 * 365  # enough for MA200 + the model window, plus chart history


def load_resources(ticker: str):
    """Load scalers and LSTM model for a specific ticker."""
    # Imported here so the web app and tests start without loading TensorFlow.
    from tensorflow.keras.models import load_model

    scaler_all   = joblib.load(os.path.join(MODEL_DIR, f"scaler_{ticker}.save"))
    scaler_close = joblib.load(os.path.join(MODEL_DIR, f"scaler_close_{ticker}.save"))
    model = load_model(os.path.join(MODEL_DIR, f"lstm_{ticker}.h5"), compile=False)
    return scaler_all, scaler_close, model


def fetch_and_prepare(ticker: str):
    """Download data and compute features for the given ticker."""
    end   = datetime.today()
    start = end - timedelta(days=HISTORY_DAYS)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True
    )
    if df is None or df.empty:
        raise ValueError(f"No price data returned for {ticker}")

    df = compute_indicators(df).dropna()
    df = df[FEATURES]

    if len(df) < SEQ_LENGTH:
        raise ValueError(f"Not enough data for {ticker}")

    window_df = df.tail(SEQ_LENGTH)
    return window_df.values, df


def recursive_forecast(features: np.ndarray,
                       scaler_all,
                       scaler_close,
                       model,
                       horizon_days: int):
    """
    Forecast future prices day by day using the trained LSTM.
    Adds sanity checks to avoid negative or unrealistic outputs.
    """
    window = features.copy()
    preds  = []

    last_close = window[-1, 0]  # last actual close price

    for _ in range(horizon_days):
        scaled_win  = scaler_all.transform(window)
        X           = scaled_win.reshape(1, SEQ_LENGTH, -1)

        scaled_pred = model.predict(X, verbose=0)[0, 0]
        inv         = scaler_close.inverse_transform([[scaled_pred]])[0, 0]

        # Clamp to between ⅓× and 3× the last actual close
        inv = np.clip(inv, last_close * 0.33, last_close * 3.0)

        preds.append(float(inv))

        # update for next step
        new_row = window[-1].copy()
        new_row[0] = inv
        window = np.vstack([window[1:], new_row])

    return preds
