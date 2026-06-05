"""
volatility.py

Computes realized volatility for each stock across 6 time horizons.
Realized volatility is the rolling standard deviation of log returns,
annualized by multiplying by sqrt(252) — the number of trading days per year.

Output: data/processed/realized_volatility.csv
  - Shape: (n_days, 60) — 6 horizons x 10 stocks
  - Columns: CRM_vol_5, CRM_vol_10, ..., AMGN_vol_126
  - Starts from 2014-12-30 (after 126-day burn-in period)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
# 6 time horizons in trading days:
#   5   = 1 week   : captures immediate short-term risk
#   10  = 2 weeks  : smooths daily noise while remaining reactive
#   21  = 1 month  : standard horizon — matches VIX window
#   42  = 2 months : medium-term dynamics, earnings cycles
#   63  = 1 quarter: classic realized volatility horizon in literature
#   126 = 6 months : structural risk level, less sensitive to spikes
HORIZONS = [5, 10, 21, 42, 63, 126]

RAW_PATH       = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

# Number of trading days per year — used for annualization
TRADING_DAYS_PER_YEAR = 252


# ── Functions ──────────────────────────────────────────────────────────────
def load_prices():
    """
    Load raw daily close prices from data/raw/prices.csv.

    Returns:
        DataFrame with Date index and one column per stock ticker.
    """
    filepath = RAW_PATH / "prices.csv"
    prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded prices — Shape: {prices.shape}")
    return prices


def compute_log_returns(prices):
    """
    Compute daily log returns from price series.

    Formula: r(t) = log( P(t) / P(t-1) )

    We use log returns instead of arithmetic returns because:
    1. Additivity: sum of daily log returns = total period return
    2. Symmetry: +50% and -50% have equal absolute magnitude
    3. Standard in financial econometrics literature (Engle, 1982)

    The first row is dropped (NaN) since there is no P(t-1) for day 1.

    Returns:
        DataFrame of daily log returns, same shape as prices minus 1 row.
    """
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    print(f"Log returns computed — Shape: {returns.shape}")
    return returns


def compute_realized_volatility(returns, horizons):
    """
    Compute annualized realized volatility across multiple time horizons.

    For each horizon N and each day t:
        vol(t, N) = std( r(t-N+1), ..., r(t) ) * sqrt(252)

    The rolling standard deviation measures how dispersed returns were
    over the past N days. Multiplying by sqrt(252) converts from daily
    to annual scale — the market convention.

    The first (max_horizon - 1) rows are dropped because not all horizons
    have enough history. With max horizon = 126, the first 125 rows are NaN.

    Output columns are named: {ticker}_vol_{horizon}
    Example: CRM_vol_5, CRM_vol_21, ..., AMGN_vol_126

    Args:
        returns:  DataFrame of daily log returns (Date x tickers)
        horizons: list of rolling window sizes in trading days

    Returns:
        DataFrame with 6 * n_tickers columns, one per (ticker, horizon) pair.
    """
    vol_dict = {}

    for horizon in horizons:
        # Rolling std over horizon days, annualized
        vol = returns.rolling(horizon).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # Rename columns to include horizon: e.g. "CRM" -> "CRM_vol_5"
        vol.columns = [f"{col}_vol_{horizon}" for col in vol.columns]
        vol_dict[horizon] = vol

    # Concatenate all horizons into a single wide DataFrame
    vol_all = pd.concat(vol_dict.values(), axis=1)

    # Drop rows where any horizon has NaN (first 125 rows due to 126-day window)
    vol_all = vol_all.dropna()

    print(f"Realized volatility computed — Shape: {vol_all.shape}")
    print(f"Horizons: {horizons}")
    print(f"From: {vol_all.index[0].date()} To: {vol_all.index[-1].date()}")

    return vol_all


def save_volatility(vol_df, filename):
    """
    Save realized volatility DataFrame to data/processed/.

    Args:
        vol_df:   DataFrame with realized volatility
        filename: output filename (e.g. "realized_volatility.csv")
    """
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_PATH / filename
    vol_df.to_csv(filepath)
    print(f"Saved: {filepath}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Step 1: Load raw prices from data/raw/
    prices = load_prices()

    # Step 2: Compute daily log returns
    # r(t) = log( P(t) / P(t-1) )
    returns = compute_log_returns(prices)

    # Step 3: Compute realized volatility on 6 horizons
    # Result: 60 time series (6 horizons x 10 stocks)
    vol = compute_realized_volatility(returns, HORIZONS)

    # Step 4: Save to data/processed/realized_volatility.csv
    save_volatility(vol, "realized_volatility.csv")

    print("\nvolatility.py completed successfully.")


if __name__ == "__main__":
    main()