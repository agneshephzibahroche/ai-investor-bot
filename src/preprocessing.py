# src/preprocessing.py
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.feature_engineering import compute_indicators
from src.inference import ROOT_DIR, SEQ_LENGTH

def preprocess(ticker: str):
    raw_path  = os.path.join(ROOT_DIR, "data", "raw", f"{ticker}.csv")
    proc_path = os.path.join(ROOT_DIR, "data", "processed", f"{ticker}_features.csv")
    model_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    df = pd.read_csv(proc_path, index_col=0, parse_dates=True)

    feature_cols = df.columns.tolist()
    X, y = [], []

    # build sliding windows
    for i in range(len(df) - SEQ_LENGTH):
        X.append(df.iloc[i:i+SEQ_LENGTH].values)
        y.append(df.iloc[i+SEQ_LENGTH]["Close"])
    X, y = np.array(X), np.array(y)

    # scale ALL features
    scaler_all = MinMaxScaler()
    scaler_all.fit(df[feature_cols])
    joblib.dump(scaler_all, os.path.join(model_dir, f"scaler_{ticker}.save"))

    # scale ONLY Close
    scaler_close = MinMaxScaler()
    scaler_close.fit(df[["Close"]])
    joblib.dump(scaler_close, os.path.join(model_dir, f"scaler_close_{ticker}.save"))

    # split train/val/test
    n = len(X)
    n_train, n_val = int(0.7*n), int(0.85*n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_val], y[n_train:n_val]
    X_test,  y_test  = X[n_val:], y[n_val:]

    np.save(os.path.join(ROOT_DIR, "data", "processed", f"X_train_{ticker}.npy"), X_train)
    np.save(os.path.join(ROOT_DIR, "data", "processed", f"y_train_{ticker}.npy"), y_train)
    np.save(os.path.join(ROOT_DIR, "data", "processed", f"X_val_{ticker}.npy"), X_val)
    np.save(os.path.join(ROOT_DIR, "data", "processed", f"y_val_{ticker}.npy"), y_val)
    np.save(os.path.join(ROOT_DIR, "data", "processed", f"X_test_{ticker}.npy"), X_test)
    np.save(os.path.join(ROOT_DIR, "data", "processed", f"y_test_{ticker}.npy"), y_test)

    print(f"✅ Preprocessing complete for {ticker}")
    print(f"Scalers saved to {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()
    preprocess(args.ticker)
