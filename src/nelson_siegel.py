"""
nelson_siegel.py
Fits Nelson-Siegel model to realized volatility term structure.
Lambda fixed at 0.04 — peak of f2 at tau=25 days (geometric midpoint of horizons 5-126).
Extracts B0 (level), B1 (slope), B2 (curvature) for each stock each day.
Saves results to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
HORIZONS = [5, 10, 21, 42, 63, 126]
LAMBDA = 0.04  # Fixed — peak of f2 at tau=25 days (geometric midpoint of horizons 5-126)
PROCESSED_PATH = Path("data/processed")
TICKERS = ["CRM", "VZ", "WMT", "INTC", "UNH", "MRK", "BA", "NKE", "CVX", "AMGN"]


def nelson_siegel_factors(tau, lam):
    """Compute f1 and f2 factors for given horizons and lambda."""
    tau = np.array(tau, dtype=float)
    f1 = (1 - np.exp(-lam * tau)) / (lam * tau)
    f2 = f1 - np.exp(-lam * tau)
    return f1, f2


def fit_nelson_siegel(sigma_obs, tau, lam):
    """
    Fit Nelson-Siegel model using OLS.
    Returns [B0, B1, B2] or [nan, nan, nan] if fitting fails.
    """
    try:
        f1, f2 = nelson_siegel_factors(tau, lam)
        X = np.column_stack([np.ones(len(tau)), f1, f2])
        betas, _, _, _ = np.linalg.lstsq(X, sigma_obs, rcond=None)
        return betas
    except Exception:
        return np.array([np.nan, np.nan, np.nan])


def load_volatility():
    """Load realized volatility from data/processed/"""
    filepath = PROCESSED_PATH / "realized_volatility.csv"
    vol = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded volatility — Shape: {vol.shape}")
    return vol


def fit_all_betas(vol_df, tickers, horizons, lam):
    """
    Fit Nelson-Siegel for each stock and each day.
    Returns DataFrame with columns: ticker_B0, ticker_B1, ticker_B2
    """
    results = {}

    for ticker in tickers:
        print(f"Fitting {ticker}...")
        cols = [f"{ticker}_vol_{h}" for h in horizons]
        ticker_vol = vol_df[cols].values

        B0_list, B1_list, B2_list = [], [], []

        for row in ticker_vol:
            if np.any(np.isnan(row)):
                B0_list.append(np.nan)
                B1_list.append(np.nan)
                B2_list.append(np.nan)
            else:
                betas = fit_nelson_siegel(row, horizons, lam)
                B0_list.append(betas[0])
                B1_list.append(betas[1])
                B2_list.append(betas[2])

        results[f"{ticker}_B0"] = B0_list
        results[f"{ticker}_B1"] = B1_list
        results[f"{ticker}_B2"] = B2_list

    betas_df = pd.DataFrame(results, index=vol_df.index)
    betas_df = betas_df.dropna()

    print(f"\nBetas computed — Shape: {betas_df.shape}")
    print(f"From: {betas_df.index[0].date()} To: {betas_df.index[-1].date()}")

    return betas_df


def save_betas(betas_df, filename):
    """Save betas to data/processed/"""
    filepath = PROCESSED_PATH / filename
    betas_df.to_csv(filepath)
    print(f"Saved: {filepath}")


def main():
    # Load volatility
    vol = load_volatility()

    # Fit all betas using fixed lambda
    print(f"\n--- Fitting Nelson-Siegel (lambda={LAMBDA}) ---")
    betas = fit_all_betas(vol, TICKERS, HORIZONS, LAMBDA)

    # Show sample
    print("\nSample betas (first 3 rows):")
    print(betas.head(3).round(4).to_string())

    # Save
    save_betas(betas, "betas.csv")

    print(f"\nnelson_siegel.py completed successfully. Lambda used: {LAMBDA}")


if __name__ == "__main__":
    main()
