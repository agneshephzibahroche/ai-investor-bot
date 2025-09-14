# scripts/batch_train_sp500.py

import sys
import subprocess
import traceback

import pandas as pd

def get_sp500_tickers() -> list[str]:
    """
    Scrape Wikipedia for the current S&P 500 constituent list.
    """
    url   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]
    # the column is called 'Symbol'
    tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers

def run_cmd(cmd: list[str]):
    """
    Run a subprocess command, streaming output to console.
    """
    print(f"\n>>> Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, check=True)

def main():
    python = sys.executable  # your venv’s python
    tickers = get_sp500_tickers()
    total   = len(tickers)
    print(f"Found {total} tickers in S&P 500.")

    for i, tkr in enumerate(tickers, start=1):
        print(f"\n[{i}/{total}] Processing {tkr}...")
        try:
            # 1) feature engineering
            run_cmd([python, "src/feature_engineering.py", "--ticker", tkr])
            # 2) preprocessing
            run_cmd([python, "src/preprocessing.py",    "--ticker", tkr])
            # 3) training
            run_cmd([python, "src/train.py",           "--ticker", tkr])
        except subprocess.CalledProcessError:
            print(f"Error processing {tkr}! See trace:")
            traceback.print_exc()
            # optionally: continue to next ticker
            continue

    print("\n Batch training complete.")

if __name__ == "__main__":
    main()
