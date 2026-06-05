"""
nelson_siegel.py

Fits the Nelson-Siegel model to the realized volatility term structure
for each stock and each day.

The model compresses 6 volatility values (one per horizon) into 3 parameters
with precise economic meaning:
  B0: long-run volatility level    — value the curve converges to as tau → ∞
  B1: slope                        — difference between short and long-term vol
  B2: curvature                    — hump in the middle section of the curve

Formula:
  σ(τ) = B0 + B1·f1(τ) + B2·f2(τ)

  f1(τ) = (1 - exp(-λτ)) / (λτ)
  f2(τ) = f1(τ) - exp(-λτ)

Lambda is fixed at 0.04 so that the peak of f2 falls at τ = 1/λ = 25 days,
the geometric midpoint of our 6 horizons: (5·10·21·42·63·126)^(1/6) ≈ 25.

Input:  data/processed/realized_volatility.csv  (2872 x 60)
Output: data/processed/betas.csv                (2872 x 30)
  Columns: CRM_B0, CRM_B1, CRM_B2, VZ_B0, ..., AMGN_B2
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
HORIZONS = [5, 10, 21, 42, 63, 126]

# Lambda fixed at 0.04 — peak of f2 at tau = 1/0.04 = 25 days
# 25 days is the geometric midpoint of our 6 horizons:
# (5 × 10 × 21 × 42 × 63 × 126)^(1/6) ≈ 25
# This ensures β2 captures curvature in the central section of the curve.
LAMBDA = 0.04

PROCESSED_PATH = Path("data/processed")
TICKERS = ["CRM", "VZ", "WMT", "INTC", "UNH", "MRK", "BA", "NKE", "CVX", "AMGN"]


# ── Functions ──────────────────────────────────────────────────────────────
def nelson_siegel_factors(tau, lam):
    """
    Compute the Nelson-Siegel loading factors f1 and f2.

    These factors are fixed for a given lambda and set of horizons.
    They define how each beta parameter affects the volatility curve:
      - f1 starts at 1 (short end) and decays to 0 (long end) → drives slope
      - f2 starts at 0, peaks at tau = 1/lambda, returns to 0 → drives curvature

    Args:
        tau: array of time horizons in trading days (e.g. [5, 10, 21, 42, 63, 126])
        lam: lambda parameter (controls where f2 peaks)

    Returns:
        f1, f2: arrays of same length as tau
    """
    tau = np.array(tau, dtype=float)
    f1 = (1 - np.exp(-lam * tau)) / (lam * tau)
    f2 = f1 - np.exp(-lam * tau)
    return f1, f2


def fit_nelson_siegel(sigma_obs, tau, lam):
    """
    Fit Nelson-Siegel model to observed volatility term structure using OLS.

    With lambda fixed, f1 and f2 are known constants for each horizon.
    The problem becomes a linear regression:

        sigma_obs = X @ [B0, B1, B2]

    where X = [1, f1(tau), f2(tau)] — a 6x3 matrix.

    We solve using numpy least squares (lstsq) which minimizes:
        sum( (sigma_obs - sigma_fitted)^2 )

    The system is overdetermined (6 equations, 3 unknowns) — lstsq finds
    the best fit in the least squares sense.

    Args:
        sigma_obs: array of 6 observed volatilities (one per horizon)
        tau:       array of 6 horizons in trading days
        lam:       lambda parameter

    Returns:
        array [B0, B1, B2] or [nan, nan, nan] if fitting fails
    """
    try:
        f1, f2 = nelson_siegel_factors(tau, lam)

        # Design matrix X: 6 rows (horizons) x 3 columns [1, f1, f2]
        X = np.column_stack([np.ones(len(tau)), f1, f2])

        # Solve: [B0, B1, B2] = argmin ||sigma_obs - X @ beta||^2
        betas, _, _, _ = np.linalg.lstsq(X, sigma_obs, rcond=None)
        return betas

    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def load_volatility():
    """
    Load realized volatility from data/processed/realized_volatility.csv.

    Returns:
        DataFrame with Date index and 60 columns (6 horizons x 10 stocks).
    """
    filepath = PROCESSED_PATH / "realized_volatility.csv"
    vol = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded volatility — Shape: {vol.shape}")
    return vol


def fit_all_betas(vol_df, tickers, horizons, lam):
    """
    Fit Nelson-Siegel model for every stock and every day.

    For each stock:
      - Extract the 6 volatility columns for that stock
      - For each day, fit Nelson-Siegel to the 6 observed values
      - Store B0, B1, B2

    The result compresses 60 noisy time series into 30 clean series
    with precise economic meaning.

    Args:
        vol_df:   DataFrame with realized volatility (Date x 60 columns)
        tickers:  list of stock ticker symbols
        horizons: list of time horizons in trading days
        lam:      lambda parameter for Nelson-Siegel

    Returns:
        DataFrame with 30 columns: {ticker}_B0, {ticker}_B1, {ticker}_B2
        for each of the 10 stocks.
    """
    results = {}

    for ticker in tickers:
        print(f"Fitting {ticker}...")

        # Extract the 6 volatility columns for this stock
        # e.g. ["CRM_vol_5", "CRM_vol_10", ..., "CRM_vol_126"]
        cols = [f"{ticker}_vol_{h}" for h in horizons]
        ticker_vol = vol_df[cols].values  # shape: (n_days, 6)

        B0_list, B1_list, B2_list = [], [], []

        for row in ticker_vol:
            if np.any(np.isnan(row)):
                # Skip days with missing volatility values
                B0_list.append(np.nan)
                B1_list.append(np.nan)
                B2_list.append(np.nan)
            else:
                # Fit Nelson-Siegel to the 6 observed volatilities
                betas = fit_nelson_siegel(row, horizons, lam)
                B0_list.append(betas[0])
                B1_list.append(betas[1])
                B2_list.append(betas[2])

        results[f"{ticker}_B0"] = B0_list
        results[f"{ticker}_B1"] = B1_list
        results[f"{ticker}_B2"] = B2_list

    # Build DataFrame with same index as input
    betas_df = pd.DataFrame(results, index=vol_df.index)

    # Drop rows where any beta is NaN
    betas_df = betas_df.dropna()

    print(f"\nBetas computed — Shape: {betas_df.shape}")
    print(f"From: {betas_df.index[0].date()} To: {betas_df.index[-1].date()}")

    return betas_df


def save_betas(betas_df, filename):
    """
    Save betas DataFrame to data/processed/.

    Args:
        betas_df: DataFrame with B0, B1, B2 for each stock and day
        filename: output filename (e.g. "betas.csv")
    """
    filepath = PROCESSED_PATH / filename
    betas_df.to_csv(filepath)
    print(f"Saved: {filepath}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Step 1: Load realized volatility (60 series)
    vol = load_volatility()

    # Step 2: Fit Nelson-Siegel for all stocks and all days
    # Compresses 60 series → 30 series with economic meaning
    print(f"\n--- Fitting Nelson-Siegel (lambda={LAMBDA}) ---")
    betas = fit_all_betas(vol, TICKERS, HORIZONS, LAMBDA)

    # Step 3: Show sample to verify results
    print("\nSample betas (first 3 rows):")
    print(betas.head(3).round(4).to_string())

    # Step 4: Save to data/processed/betas.csv
    save_betas(betas, "betas.csv")

    print(f"\nnelson_siegel.py completed successfully. Lambda used: {LAMBDA}")


if __name__ == "__main__":
    main()