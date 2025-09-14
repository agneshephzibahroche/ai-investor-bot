#!/usr/bin/env python3
# scripts/batch_train_sp500_common.py

"""
Batch‐trains feature/model pipelines only for a small set of popular S&P500 tickers.
"""

import sys
import subprocess
import traceback

# You can adjust this list as you like
COMMON_TICKERS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "AMZN",  # Amazon
    "GOOGL", # Alphabet Class A
    "BRK-B", # Berkshire Hathaway
    "NVDA",  # NVIDIA
    "META",  # Meta Platforms
    "TSLA",  # Tesla
    "JPM",   # JPMorgan Chase
    "V"      # Visa
]

def run_cmd(cmd):
    """Run a subprocess and raise on error."""
    print(f">>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    python = sys.executable
    total  = len(COMMON_TICKERS)
    print(f"Batch‐training {total} tickers: {', '.join(COMMON_TICKERS)}")

    for i, tkr in enumerate(COMMON_TICKERS, start=1):
        print(f"\n[{i}/{total}] Processing {tkr}...")
        try:
            run_cmd([python, "-m", "src.feature_engineering", "--ticker", tkr])
            run_cmd([python, "-m", "src.preprocessing",     "--ticker", tkr])
            run_cmd([python, "-m", "src.train",             "--ticker", tkr])
        except subprocess.CalledProcessError:
            print(f"Error processing {tkr}, skipping.")
            traceback.print_exc()
            continue

    print("\n✅ Batch training of common tickers complete.")

if __name__ == "__main__":
    main()