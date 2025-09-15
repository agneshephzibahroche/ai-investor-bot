import os
import argparse
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from src.feature_engineering import compute_indicators

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR   = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
SEQ_LENGTH = 60

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_ticker(ticker: str):
    # 1. Download history
    df = yf.download(ticker, period="10y", auto_adjust=True, progress=False)
    df = compute_indicators(df).dropna()

    # 2. Select all 11 engineered features
    features = [
        "Close", "High", "Low", "Open", "Volume",
        "Return", "MA50", "MA200", "MACD", "Signal", "RSI"
    ]
    df = df[features]

    # 3. Fit scalers
    scaler_all   = MinMaxScaler()
    scaler_all.fit(df.values)
    joblib.dump(scaler_all, os.path.join(MODEL_DIR, f"scaler_{ticker}.save"))

    # Close-only scaler (for inverse transform of predictions)
    scaler_close = MinMaxScaler()
    scaler_close.fit(df[["Close"]].values)
    joblib.dump(scaler_close, os.path.join(MODEL_DIR, f"scaler_close_{ticker}.save"))

    # 4. Transform full dataset
    scaled = scaler_all.transform(df.values)

    # 5. Build sequences (X, y)
    X, y = [], []
    for i in range(SEQ_LENGTH, len(scaled)):
        X.append(scaled[i-SEQ_LENGTH:i])
        y.append(scaled[i, 0])  # predict Close only
    X, y = np.array(X), np.array(y)

    # 6. Split train/val/test
    n = len(X)
    train_end = int(0.7 * n)
    val_end   = int(0.85 * n)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val     = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test   = X[val_end:], y[val_end:]

    # 7. Save
    np.save(os.path.join(DATA_DIR, f"X_train_{ticker}.npy"), X_train)
    np.save(os.path.join(DATA_DIR, f"y_train_{ticker}.npy"), y_train)
    np.save(os.path.join(DATA_DIR, f"X_val_{ticker}.npy"), X_val)
    np.save(os.path.join(DATA_DIR, f"y_val_{ticker}.npy"), y_val)
    np.save(os.path.join(DATA_DIR, f"X_test_{ticker}.npy"), X_test)
    np.save(os.path.join(DATA_DIR, f"y_test_{ticker}.npy"), y_test)

    print(f"✅ Preprocessing complete for {ticker} | X_train: {X_train.shape}, y_train: {y_train.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. AAPL)")
    args = parser.parse_args()
    preprocess_ticker(args.ticker)
