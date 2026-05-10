"""
data_loader.py
Downloads and validates daily market data for all assets.
Saves raw data to data/raw/
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
TICKERS = ["CRM", "VZ", "WMT", "INTC", "UNH", "MRK", "BA", "NKE", "CVX", "AMGN"]
BENCHMARKS = ["^VIX", "^GSPC"]  # VIX and S&P500
START_DATE = "2014-07-01"
RAW_PATH = Path("data/raw")


def download_prices(tickers, start, end=None):
    """Download daily close prices from yfinance."""
    print(f"Downloading {tickers}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data


def validate_data(df, name):
    """Validate downloaded data for missing values and zero prices."""
    missing = df.isnull().sum()
    if missing.any():
        print(f"WARNING — Missing values in {name}:")
        print(missing[missing > 0])

    zeros = (df == 0).sum()
    if zeros.any():
        raise ValueError(f"Zero prices found in {name}")

    print(f"{name} — OK. Shape: {df.shape}, "
          f"From: {df.index[0].date()} To: {df.index[-1].date()}")
    return True


def save_data(df, filename):
    """Save DataFrame to CSV in data/raw/"""
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    filepath = RAW_PATH / filename
    df.to_csv(filepath)
    print(f"Saved: {filepath}")


def main():
    # Download stock prices
    prices = download_prices(TICKERS, start=START_DATE)
    validate_data(prices, "Stock prices")
    save_data(prices, "prices.csv")

    # Download benchmarks
    benchmarks = download_prices(BENCHMARKS, start=START_DATE)
    benchmarks.columns = ["SP500", "VIX"]
    validate_data(benchmarks, "Benchmarks")
    save_data(benchmarks, "benchmarks.csv")

    print("\ndata_loader.py completed successfully.")


if __name__ == "__main__":
    main()