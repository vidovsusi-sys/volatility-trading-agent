"""
vrp.py
Computes the Variance Risk Premium (VRP) for each day.
VRP = VIX - realized volatility of S&P500 (21-day horizon)
Saves results to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
HORIZON = 21  # 21 trading days = ~30 calendar days (matches VIX window)
RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")


def load_benchmarks():
    """Load VIX and S&P500 from data/raw/benchmarks.csv"""
    filepath = RAW_PATH / "benchmarks.csv"
    benchmarks = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded benchmarks — Shape: {benchmarks.shape}")
    return benchmarks


def compute_vrp(benchmarks, horizon):
    """
    Compute VRP = VIX - realized volatility of S&P500.
    VIX is already in annualized % terms.
    Realized vol is computed as rolling std of log returns x sqrt(252).
    """
    # Log returns of S&P500
    sp500_returns = np.log(benchmarks["SP500"] / benchmarks["SP500"].shift(1))

    # Realized volatility of S&P500 — annualized
    sp500_vol = sp500_returns.rolling(horizon).std() * np.sqrt(252)

    # VIX is in % terms — divide by 100 to match realized vol scale
    vix = benchmarks["VIX"] / 100

    # VRP
    vrp = vix - sp500_vol

    vrp = vrp.dropna()
    vrp.name = "VRP"

    print(f"VRP computed — Shape: {vrp.shape}")
    print(f"From: {vrp.index[0].date()} To: {vrp.index[-1].date()}")
    print(f"Mean VRP: {vrp.mean():.4f}")
    print(f"Min VRP:  {vrp.min():.4f}")
    print(f"Max VRP:  {vrp.max():.4f}")

    return vrp


def save_vrp(vrp, filename):
    """Save VRP to data/processed/"""
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_PATH / filename
    vrp.to_csv(filepath)
    print(f"Saved: {filepath}")


def main():
    # Load benchmarks
    benchmarks = load_benchmarks()

    # Compute VRP
    vrp = compute_vrp(benchmarks, HORIZON)

    # Show sample
    print("\nSample VRP (first 5 rows):")
    print(vrp.head())

    # Save
    save_vrp(vrp, "vrp.csv")

    print("\nvrp.py completed successfully.")


if __name__ == "__main__":
    main()