# src/feature_engineering.py

import os
import argparse

import pandas as pd
import yfinance as yf
import numpy as np


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # Daily returns
    df['Return'] = df['Close'].pct_change()

    # Moving averages
    df['MA50']  = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # MACD + Signal
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()   # ✅ NEW

    # RSI (14-day)
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0)
    loss  = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs  = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def main():
    p = argparse.ArgumentParser(
        description="Download raw data and compute features for a ticker"
    )
    p.add_argument('--ticker', default='^GSPC')
    args = p.parse_args()
    ticker = args.ticker.upper()

    # paths
    ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    RAW_DIR   = os.path.join(ROOT, 'data', 'raw')
    PROC_DIR  = os.path.join(ROOT, 'data', 'processed')
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROC_DIR, exist_ok=True)

    RAW_CSV   = os.path.join(RAW_DIR, f'{ticker}.csv')
    PROC_CSV  = os.path.join(PROC_DIR, f'{ticker}_features.csv')

    # 1) Download *all* history (auto_adjust=True gives adjusted Close)
    df_raw = yf.download(
        ticker,
        period='max',
        progress=False,
        auto_adjust=True
    )
    # flatten multi-index if present
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)

    df_raw.to_csv(RAW_CSV)
    print(f"Raw data saved to {RAW_CSV} ({len(df_raw)} rows)")

    # 2) Compute indicators
    df_feat = compute_indicators(df_raw)

    # 3) Drop initial NaNs, save
    df_feat.dropna(inplace=True)
    df_feat.to_csv(PROC_CSV)
    print(f"Features saved to {PROC_CSV} ({len(df_feat)} rows)")


if __name__ == '__main__':
    main()
