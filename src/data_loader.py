"""
data_loader.py

Downloads and validates daily market data for all assets used in the pipeline.
Saves two CSV files to data/raw/:
  - prices.csv    : daily close prices for the 10 portfolio stocks
  - benchmarks.csv: daily VIX and S&P500 prices (used for VRP calculation)

This script must be run first in the pipeline before any other script.
Raw data is never modified after download — all transformations happen downstream.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
# 10 stocks selected from Dow Jones 30 using minimum average pairwise
# correlation criterion + liquidity check (>1M daily volume).
# See notebooks/stock_selection.py for the selection procedure.
TICKERS = ["CRM", "VZ", "WMT", "INTC", "UNH", "MRK", "BA", "NKE", "CVX", "AMGN"]

# VIX: implied volatility of S&P500 options (forward-looking)
# ^GSPC: S&P500 index prices (used to compute realized volatility for VRP)
BENCHMARKS = ["^VIX", "^GSPC"]

# Start 6 months before 2015-01-01 to allow for burn-in period.
# Computing 126-day realized volatility on 2015-01-02 requires
# 126 trading days of history before that date.
START_DATE = "2014-07-01"

# All raw data is saved here — never modified after download
RAW_PATH = Path("data/raw")


# ── Functions ──────────────────────────────────────────────────────────────
def download_prices(tickers, start, end=None):
    """
    Download daily close prices from Yahoo Finance using yfinance.

    Args:
        tickers: list of ticker symbols (e.g. ["WMT", "BA"])
        start:   start date string (e.g. "2014-07-01")
        end:     end date string or None (defaults to today)

    Returns:
        DataFrame with Date index and one column per ticker.

    Note:
        auto_adjust=True adjusts prices for stock splits and dividends,
        ensuring comparability across the full history.
    """
    print(f"Downloading {tickers}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    return data


def validate_data(df, name):
    """
    Validate downloaded data for data quality issues.

    Checks:
    1. Missing values (NaN) — acceptable with warning, common for non-trading days
    2. Zero prices — not acceptable, raise error (economically meaningless
       and would cause division by zero in log return calculation)

    Args:
        df:   DataFrame to validate
        name: label for print messages (e.g. "Stock prices")

    Returns:
        True if validation passes (raises ValueError if zero prices found)
    """
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"WARNING — Missing values in {name}:")
        print(missing[missing > 0])

    # Check for zero prices — these would cause log(0) = -inf in returns
    zeros = (df == 0).sum()
    if zeros.any():
        raise ValueError(f"Zero prices found in {name}")

    print(f"{name} — OK. Shape: {df.shape}, "
          f"From: {df.index[0].date()} To: {df.index[-1].date()}")
    return True


def save_data(df, filename):
    """
    Save DataFrame to CSV in data/raw/.

    Creates the directory if it does not exist.
    Raw data is immutable after saving — never overwrite with modified data.

    Args:
        df:       DataFrame to save
        filename: output filename (e.g. "prices.csv")
    """
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    filepath = RAW_PATH / filename
    df.to_csv(filepath)
    print(f"Saved: {filepath}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Step 1: Download daily close prices for the 10 portfolio stocks
    prices = download_prices(TICKERS, start=START_DATE)
    validate_data(prices, "Stock prices")
    save_data(prices, "prices.csv")

    # Step 2: Download VIX and S&P500 for VRP calculation
    # Note: yfinance returns columns in alphabetical order — ^GSPC before ^VIX
    # so we rename accordingly: first column = SP500, second = VIX
    benchmarks = download_prices(BENCHMARKS, start=START_DATE)
    benchmarks.columns = ["SP500", "VIX"]
    validate_data(benchmarks, "Benchmarks")
    save_data(benchmarks, "benchmarks.csv")

    print("\ndata_loader.py completed successfully.")


if __name__ == "__main__":
    main()