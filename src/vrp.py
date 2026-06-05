"""
vrp.py

Computes the Variance Risk Premium (VRP) for each trading day.

VRP = VIX(t) - realized_vol_SP500(t, 21 days)

The VIX measures implied volatility — what the options market expects
the S&P500 to do over the next 30 calendar days (~21 trading days).

Realized volatility measures what the S&P500 actually did over the
past 21 trading days.

The difference (VRP) captures how much the market is paying in excess
of what actually materializes:
  - High positive VRP: market very nervous, pricing in risk preemptively
  - Low or negative VRP: realized volatility exceeds expectations (crisis)

VRP is used as an exogenous input to XGBoost — it provides forward-looking
market sentiment that the stock-specific betas cannot capture alone.
Reference: Carr and Wu (2009), "Variance Risk Premiums"

Input:  data/raw/benchmarks.csv (VIX and S&P500 prices)
Output: data/processed/vrp.csv  (daily VRP series)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
# 21 trading days ≈ 30 calendar days — matches the VIX measurement window.
# The VIX measures expected volatility over the next 30 calendar days.
# We use 21 trading days for realized vol to make the comparison consistent.
HORIZON = 21

RAW_PATH       = Path("data/raw")
PROCESSED_PATH = Path("data/processed")

# Annualization factor — same as in volatility.py
TRADING_DAYS_PER_YEAR = 252


# ── Functions ──────────────────────────────────────────────────────────────
def load_benchmarks():
    """
    Load VIX and S&P500 daily prices from data/raw/benchmarks.csv.

    Columns:
        SP500: S&P500 index daily close prices
        VIX:   CBOE VIX index — implied volatility of S&P500 options
               expressed as annualized percentage (e.g. 18.5 = 18.5% per year)

    Returns:
        DataFrame with Date index and columns [SP500, VIX].
    """
    filepath = RAW_PATH / "benchmarks.csv"
    benchmarks = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded benchmarks — Shape: {benchmarks.shape}")
    return benchmarks


def compute_vrp(benchmarks, horizon):
    """
    Compute the Variance Risk Premium (VRP) for each trading day.

    Formula:
        VRP(t) = VIX(t) / 100  -  std(r_SP500(t-20:t)) * sqrt(252)

    Steps:
    1. Compute daily log returns of S&P500
    2. Compute rolling 21-day realized volatility (annualized)
    3. Convert VIX from percentage to decimal (divide by 100)
    4. Subtract: VRP = VIX_decimal - realized_vol

    Scale note:
        VIX from yfinance is in percentage points (e.g. 18.5 means 18.5%)
        Realized vol from rolling std * sqrt(252) is in decimal (e.g. 0.185)
        We divide VIX by 100 to put both on the same decimal scale.

    Args:
        benchmarks: DataFrame with SP500 and VIX columns
        horizon:    rolling window for realized volatility (21 trading days)

    Returns:
        Series of daily VRP values with Date index.
    """
    # Step 1: Log returns of S&P500
    # r(t) = log(P(t) / P(t-1))
    sp500_returns = np.log(benchmarks["SP500"] / benchmarks["SP500"].shift(1))

    # Step 2: Realized volatility of S&P500 — annualized
    # Same formula as volatility.py but applied to S&P500
    sp500_vol = sp500_returns.rolling(horizon).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # Step 3: Convert VIX from percentage to decimal
    # VIX = 18.5 → 0.185 (same scale as realized vol)
    vix = benchmarks["VIX"] / 100

    # Step 4: VRP = implied volatility - realized volatility
    # Positive: market more nervous than realized → risk premium
    # Negative: realized volatility exceeded expectations → crisis in progress
    vrp = vix - sp500_vol
    vrp = vrp.dropna()
    vrp.name = "VRP"

    print(f"VRP computed — Shape: {vrp.shape}")
    print(f"From: {vrp.index[0].date()} To: {vrp.index[-1].date()}")
    print(f"Mean VRP: {vrp.mean():.4f}  (historical avg ~3-5%)")
    print(f"Min VRP:  {vrp.min():.4f}  (negative = crisis in progress)")
    print(f"Max VRP:  {vrp.max():.4f}  (high = market very nervous)")

    return vrp


def save_vrp(vrp, filename):
    """
    Save VRP Series to data/processed/.

    Args:
        vrp:      Series of daily VRP values
        filename: output filename (e.g. "vrp.csv")
    """
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_PATH / filename
    vrp.to_csv(filepath)
    print(f"Saved: {filepath}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Step 1: Load VIX and S&P500 prices
    benchmarks = load_benchmarks()

    # Step 2: Compute VRP = VIX - realized vol S&P500 (21 days)
    vrp = compute_vrp(benchmarks, HORIZON)

    # Step 3: Show sample to verify values are in expected range
    print("\nSample VRP (first 5 rows):")
    print(vrp.head())

    # Step 4: Save to data/processed/vrp.csv
    save_vrp(vrp, "vrp.csv")

    print("\nvrp.py completed successfully.")


if __name__ == "__main__":
    main()