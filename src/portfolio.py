"""
portfolio.py
Compute portfolio weights using three different methods:
  - Method A: Risk Parity with forecasted B0
  - Method B: Shape Trading with normalized B0, B1, B2     [TODO]
  - Method C: Momentum / forecasted volatility              [TODO]

Inputs:  data/processed/predictions_xgboost.csv (or predictions_arma.csv)
         data/raw/prices.csv (for Method C)
Outputs: data/processed/weights_method_a.csv
         data/processed/weights_method_b.csv [TODO]
         data/processed/weights_method_c.csv [TODO]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

# ── Configuration ──────────────────────────────────────────────────────────
TICKERS = ["CRM", "VZ", "WMT", "INTC", "UNH", "MRK", "BA", "NKE", "CVX", "AMGN"]
PROCESSED_PATH = Path("data/processed")

# Constraints on portfolio weights
MIN_WEIGHT = 0.05    # 5% minimum per stock
MAX_WEIGHT = 0.25    # 25% maximum per stock


# ── Function 1: Load forecasts ─────────────────────────────────────────────
def load_forecasts(model="xgboost"):
    """
    Load Nelson-Siegel beta forecasts from CSV.

    Args:
        model: "xgboost" or "arma" — which model's forecasts to load.

    Returns:
        Tuple (b0, b1, b2): three DataFrames with TICKERS as columns
        and Date as index.
    """
    # Build path to the file
    filename = f"predictions_{model}.csv"
    filepath = PROCESSED_PATH / filename

    # Load CSV
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)

    # Split columns by Nelson-Siegel parameter
    b0 = df[[f"{ticker}_B0" for ticker in TICKERS]]
    b1 = df[[f"{ticker}_B1" for ticker in TICKERS]]
    b2 = df[[f"{ticker}_B2" for ticker in TICKERS]]

    # Rename columns by stripping the _B0/_B1/_B2 suffix
    b0.columns = TICKERS
    b1.columns = TICKERS
    b2.columns = TICKERS

    print(f"Loaded {model} forecasts — Shape: {df.shape}")
    print(f"From: {df.index[0].date()} To: {df.index[-1].date()}")

    return b0, b1, b2


# ── Function 2: Method A — Risk Parity with forecasted B0 ──────────────────
def method_a_risk_parity(b0_forecast):
    """
    Method A: Risk Parity using forecasted B0.

    Weights are inversely proportional to forecasted volatility:
        weight_i = (1 / B0_i) / sum_j (1 / B0_j)

    Stocks with high forecasted B0 (high expected volatility) get less weight.
    Stocks with low forecasted B0 (low expected volatility) get more weight.

    Args:
        b0_forecast: DataFrame with Date index and TICKERS columns.
                     Contains forecasted B0 for each stock on each day.

    Returns:
        weights: DataFrame with same shape as b0_forecast,
                 where each row sums to 1.
    """
    # SAFETY: replace non-positive B0 with the per-stock median.
    # XGBoost can occasionally produce non-positive forecasts, which are
    # economically meaningless (volatility cannot be negative).
    b0_clean = b0_forecast.where(b0_forecast > 0)
    b0_clean = b0_clean.fillna(b0_clean.median())

    # Compute 1 / B0 element-wise
    inv_b0 = 1.0 / b0_clean

    # Normalize row by row (axis=1 = per date)
    # weight_i = inv_b0_i / row sum
    weights = inv_b0.div(inv_b0.sum(axis=1), axis=0)

    print(f"Method A weights computed — Shape: {weights.shape}")
    print(f"Row sums (should be ~1.0):")
    print(weights.sum(axis=1).describe())

    return weights
# ── Function 3: Apply weight constraints via scipy QP ──────────────────────
def _solve_one_day(target_weights, min_w, max_w):
    """
    Solve the constrained optimization for a single day.

    Find weights w that minimize the squared distance to target_weights,
    subject to:
        - sum(w) = 1                 (fully invested)
        - min_w <= w_i <= max_w      (per-stock bounds)

    Args:
        target_weights: 1D numpy array of target weights (e.g., raw Risk Parity).
        min_w: minimum allowed weight per stock.
        max_w: maximum allowed weight per stock.

    Returns:
        1D numpy array of optimized weights, or NaN array if optimization fails.
    """
    n = len(target_weights)

    # Objective function: squared distance from target weights
    def objective(w):
        return np.sum((w - target_weights) ** 2)

    # Equality constraint: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds: each weight in [min_w, max_w]
    bounds = [(min_w, max_w)] * n

    # Initial guess: clipped target (good starting point)
    x0 = np.clip(target_weights, min_w, max_w)
    x0 = x0 / x0.sum()  # normalize so the starting point is feasible-ish

    # Solve
    result = minimize(
        objective, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 200},
    )

    if result.success:
        return result.x
    else:
        # Fallback: clip and renormalize. Not as clean as the optimizer,
        # but guarantees a feasible-ish answer.
        fallback = np.clip(target_weights, min_w, max_w)
        return fallback / fallback.sum()


def apply_constraints(weights, min_w=MIN_WEIGHT, max_w=MAX_WEIGHT):
    """
    Apply min/max weight constraints day by day using quadratic optimization.

    For each date, solve a small QP:
        minimize    sum( (w_i - target_i)^2 )
        subject to  sum(w) = 1
                    min_w <= w_i <= max_w

    The target weights are the raw (unconstrained) input weights.
    This guarantees that the bounds are strictly respected.

    Args:
        weights: DataFrame with raw weights (rows = dates, cols = TICKERS).
        min_w: minimum weight per stock (default: MIN_WEIGHT).
        max_w: maximum weight per stock (default: MAX_WEIGHT).

    Returns:
        DataFrame with constrained weights, each row summing to 1
        and each weight within [min_w, max_w].
    """
    print(f"Applying constraints via scipy (this may take ~30-90s)...")

    # Allocate output array
    constrained_array = np.full(weights.shape, np.nan)

    # Solve one day at a time
    for i, (date, row) in enumerate(weights.iterrows()):
        target = row.values
        if np.isnan(target).any():
            continue  # skip dates with missing forecasts
        constrained_array[i, :] = _solve_one_day(target, min_w, max_w)

    # Wrap back into a DataFrame
    constrained = pd.DataFrame(
        constrained_array,
        index=weights.index,
        columns=weights.columns,
    )

    # Diagnostics
    n_failed = constrained.isna().any(axis=1).sum()
    print(f"Constraints applied — min: {constrained.min().min():.4f}, "
          f"max: {constrained.max().max():.4f}")
    print(f"Row sums (should be 1.0): "
          f"mean={constrained.sum(axis=1).mean():.6f}")
    if n_failed > 0:
        print(f"WARNING: optimizer failed on {n_failed} days "
              f"(out of {len(constrained)}).")

    return constrained

# ── Method B: Shape Trading ────────────────────────────────────────────────
def load_historical_betas():
    """
    Load historical Nelson-Siegel betas and compute their historical
    standard deviations per stock.

    These std values serve as scaling constants for z-score normalization
    in Method B. We use the full historical sample (2014-2026) since std
    is a structural property of the series, not a forecast.

    Returns:
        std_b0: Series with std of B0 for each stock (10 values)
        std_b1: Series with std of B1 for each stock (10 values)
        std_b2: Series with std of B2 for each stock (10 values)
    """
    path = PROCESSED_PATH / "betas.csv"
    betas = pd.read_csv(path, index_col=0, parse_dates=True)

    # Build column lists for each parameter
    b0_cols = [f"{ticker}_B0" for ticker in TICKERS]
    b1_cols = [f"{ticker}_B1" for ticker in TICKERS]
    b2_cols = [f"{ticker}_B2" for ticker in TICKERS]

    # Compute std per stock (across all available history)
    std_b0 = betas[b0_cols].std()
    std_b1 = betas[b1_cols].std()
    std_b2 = betas[b2_cols].std()

    # Rename index to use ticker symbols only (e.g., "CRM_B0" -> "CRM")
    std_b0.index = TICKERS
    std_b1.index = TICKERS
    std_b2.index = TICKERS

    print(f"Loaded historical std from {len(betas)} days of betas.csv")
    print(f"  std_B0 range: [{std_b0.min():.4f}, {std_b0.max():.4f}]")
    print(f"  std_B1 range: [{std_b1.min():.4f}, {std_b1.max():.4f}]")
    print(f"  std_B2 range: [{std_b2.min():.4f}, {std_b2.max():.4f}]")

    return std_b0, std_b1, std_b2

def method_b_shape_trading(b0_forecast, b1_forecast, b2_forecast,
                            std_b0, std_b1, std_b2):
    """
    Compute Method B weights via shape-aware z-score scoring.

    The score combines three normalized terms:
      score_i = (B0_i / std_B0_i) + (|B1_i| / std_B1_i) + (|B2_i| / std_B2_i)

    Each term is a z-score-like measure of how anomalous the forecasted
    parameter is relative to its historical variability. Absolute values
    on B1 and B2 ensure that both directions of slope/curvature deviation
    are penalized (any irregular shape signals risk).

    Weights are inversely proportional to the score (Risk-Parity-style):
      weight_i = (1 / score_i) / sum_j(1 / score_j)

    Args:
        b0_forecast: DataFrame (Date x TICKERS) with forecasted B0
        b1_forecast: DataFrame (Date x TICKERS) with forecasted B1
        b2_forecast: DataFrame (Date x TICKERS) with forecasted B2
        std_b0: Series with historical std of B0 per stock
        std_b1: Series with historical std of B1 per stock
        std_b2: Series with historical std of B2 per stock

    Returns:
        weights: DataFrame (Date x TICKERS) where each row sums to 1.
    """
    # SAFETY: replace non-positive B0 with the per-stock median.
    # XGBoost can occasionally produce non-positive forecasts, which are
    # economically meaningless (volatility cannot be negative).
    b0_clean = b0_forecast.where(b0_forecast > 0)
    b0_clean = b0_clean.fillna(b0_clean.median())

    # Compute the three normalized terms (broadcasting: divide each column
    # by the std of that stock). |B1| and |B2| capture deviation magnitude.
    term_b0 = b0_clean.div(std_b0, axis=1)
    term_b1 = b1_forecast.abs().div(std_b1, axis=1)
    term_b2 = b2_forecast.abs().div(std_b2, axis=1)

    # Composite score: sum of the three normalized terms
    score = term_b0 + term_b1 + term_b2

    # Inverse-score weights: higher score (more anomalous) -> less weight
    inv_score = 1.0 / score

    # Normalize per row so each day's weights sum to 1
    weights = inv_score.div(inv_score.sum(axis=1), axis=0)

    print(f"Method B weights computed — Shape: {weights.shape}")
    print(f"Row sums (should be ~1.0):")
    print(weights.sum(axis=1).describe())

    return weights
# ── Main orchestration ─────────────────────────────────────────────────────
def compute_method_a(model="xgboost"):
    """
    Full pipeline for Method A:
      1. Load forecasts (xgboost or arma).
      2. Compute raw Risk Parity weights from forecasted B0.
      3. Apply min/max constraints via scipy QP.
      4. Save final weights to data/processed/weights_method_a_{model}.csv.

    Args:
        model: "xgboost" or "arma".

    Returns:
        DataFrame of constrained weights.
    """
    print(f"\n{'='*60}")
    print(f"METHOD A — Risk Parity with {model.upper()} forecasts")
    print(f"{'='*60}\n")

    # Step 1: load forecasts
    b0, _, _ = load_forecasts(model=model)

    # Step 2: compute raw Risk Parity weights
    w_raw = method_a_risk_parity(b0)

    # Step 3: apply constraints via scipy
    w = apply_constraints(w_raw)

    # Step 4: save to CSV
    output_path = PROCESSED_PATH / f"weights_method_a_{model}.csv"
    w.to_csv(output_path)
    print(f"\nSaved: {output_path}")

    return w

def compute_method_b(model="xgboost"):
    """
    Full pipeline for Method B (Shape Trading):
      1. Load historical betas and compute std for each parameter.
      2. Load forecasts (xgboost or arma).
      3. Compute raw shape-trading weights via z-score scoring.
      4. Apply min/max constraints via scipy QP.
      5. Save final weights to data/processed/weights_method_b_{model}.csv.

    Args:
        model: "xgboost" or "arma".

    Returns:
        DataFrame of constrained weights.
    """
    print(f"\n{'='*60}")
    print(f"METHOD B — Shape Trading with {model.upper()} forecasts")
    print(f"{'='*60}\n")

    # Step 1: load historical betas and compute std
    std_b0, std_b1, std_b2 = load_historical_betas()

    # Step 2: load forecasts
    b0, b1, b2 = load_forecasts(model=model)

    # Step 3: compute raw shape-trading weights
    w_raw = method_b_shape_trading(b0, b1, b2, std_b0, std_b1, std_b2)

    # Step 4: apply constraints via scipy
    w = apply_constraints(w_raw)

    # Step 5: save to CSV
    output_path = PROCESSED_PATH / f"weights_method_b_{model}.csv"
    w.to_csv(output_path)
    print(f"\nSaved: {output_path}")

    return w

def main():
    """Entry point: run Methods A and B for available forecast models."""
    # ── Method A (Risk Parity) ─────────────────────────────────────────
    # Run for XGBoost forecasts (primary model)
    compute_method_a(model="xgboost")

    # ── Method B (Shape Trading) ───────────────────────────────────────
    # Run for XGBoost forecasts (primary model)
    compute_method_b(model="xgboost")

    # Run for ARMA forecasts (benchmark used for RQ1)
    # NOTE: at the time of writing, predictions_arma.csv contains only NaN
    # (forecasting.py ARMA pipeline issue, to be fixed upstream).
    # Once ARMA forecasts are valid, uncomment the line below.
    # compute_method_a(model="arma")
    print("\n[INFO] Skipping ARMA: predictions file contains only NaN. "
          "Re-enable once forecasting.py produces valid ARMA forecasts.")

    print("\nportfolio.py completed successfully.")