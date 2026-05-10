"""
volatility.py
Computes realized volatility for each stock across 6 time horizons.
Saves results to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
HORIZONS = [5, 10, 21, 42, 63, 126]
RAW_PATH = Path("data/raw")
PROCESSED_PATH = Path("data/processed")


def load_prices():
    """Load raw prices from data/raw/prices.csv"""
    filepath = RAW_PATH / "prices.csv"
    prices = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded prices — Shape: {prices.shape}")
    return prices


def compute_log_returns(prices):
    """Compute daily log returns."""
    returns = np.log(prices / prices.shift(1))
    returns = returns.dropna()
    print(f"Log returns computed — Shape: {returns.shape}")
    return returns


def compute_realized_volatility(returns, horizons):
    """
    Compute realized volatility for each stock across multiple horizons.
    Annualized by sqrt(252).

    Returns a MultiIndex DataFrame with columns (ticker, horizon).
    """
    vol_dict = {}

    for horizon in horizons:
        vol = returns.rolling(horizon).std() * np.sqrt(252)
        vol.columns = [f"{col}_vol_{horizon}" for col in vol.columns]
        vol_dict[horizon] = vol

    # Combine all horizons
    vol_all = pd.concat(vol_dict.values(), axis=1)
    vol_all = vol_all.dropna()

    print(f"Realized volatility computed — Shape: {vol_all.shape}")
    print(f"Horizons: {horizons}")
    print(f"From: {vol_all.index[0].date()} To: {vol_all.index[-1].date()}")

    return vol_all


def save_volatility(vol_df, filename):
    """Save volatility DataFrame to data/processed/"""
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_PATH / filename
    vol_df.to_csv(filepath)
    print(f"Saved: {filepath}")


def main():
    # Load prices
    prices = load_prices()

    # Compute log returns
    returns = compute_log_returns(prices)

    # Compute realized volatility
    vol = compute_realized_volatility(returns, HORIZONS)

    # Save
    save_volatility(vol, "realized_volatility.csv")

    print("\nvolatility.py completed successfully.")


if __name__ == "__main__":
    main()