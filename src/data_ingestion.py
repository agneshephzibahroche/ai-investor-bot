# src/data_ingestion.py

import yfinance as yf
import pandas as pd
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'raw')

def download_sp500(start='2010-01-01', end='2025-01-01'):
    os.makedirs(RAW_DIR, exist_ok=True)
    df = yf.download('^GSPC', start=start, end=end)
    csv_path = os.path.join(RAW_DIR, 'sp500.csv')
    df.to_csv(csv_path)
    print(f"Saved raw data to {csv_path}")

if __name__ == '__main__':
    download_sp500()
