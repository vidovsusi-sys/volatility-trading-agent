"""
forecasting.py

Walk-forward forecasting of Nelson-Siegel betas using two models:
  1. XGBoost — main forecaster, captures non-linear relationships
  2. ARMA    — linear benchmark for comparison (Research Question 1)

Walk-forward methodology:
  - To predict day t, only data up to day t-GAP_DAYS is used for training
  - Model is retrained every RETRAIN_FREQ days (monthly) for efficiency
  - Minimum MIN_TRAIN_DAYS of history required before first prediction
  - This ensures no future data leaks into predictions (no look-ahead bias)

Inputs:
  data/processed/betas.csv — Nelson-Siegel betas (B0, B1, B2 per stock per day)
  data/processed/vrp.csv   — Variance Risk Premium (daily)

Outputs:
  data/processed/predictions_xgboost.csv — XGBoost predicted betas (2018 onwards)
  data/processed/predictions_arma.csv    — ARMA predicted betas (2018 onwards)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
TICKERS = ["CRM", "VZ", "WMT", "INTC", "UNH", "MRK", "BA", "NKE", "CVX", "AMGN"]
PROCESSED_PATH = Path("data/processed")

# Walk-forward parameters
MIN_TRAIN_DAYS = 500   # Minimum days of history before first prediction
                        # With data from 2015 and backtest from 2018,
                        # this gives ~3 years of training before first forecast
GAP_DAYS       = 5     # Gap between training end and prediction date
                        # Prevents autocorrelation from inflating model accuracy
RETRAIN_FREQ   = 21    # Retrain every 21 trading days (~monthly)
                        # Balances model freshness vs computational cost
BACKTEST_START = "2018-01-01"  # First date for which we generate predictions

# XGBoost hyperparameters
# These are standard conservative values that avoid overfitting
# on financial time series of this size (~2800 observations)
XGB_PARAMS = {
    "n_estimators":    200,   # number of trees
    "max_depth":       4,     # shallow trees to avoid overfitting
    "learning_rate":   0.05,  # small learning rate for stable convergence
    "subsample":       0.8,   # row subsampling — regularization
    "colsample_bytree":0.8,   # column subsampling — regularization
    "random_state":    42,    # reproducibility
    "verbosity":       0,     # suppress XGBoost output
}

# Lag structure for feature engineering
# We use lags 1, 2, 5 to capture:
#   lag 1: yesterday's betas (strongest predictor)
#   lag 2: two days ago (short-term momentum)
#   lag 5: one week ago (weekly pattern)
# Lags 3 and 4 are omitted to reduce feature count and overfitting risk
LAGS = [1, 2, 5]


# ── Functions ──────────────────────────────────────────────────────────────
def load_data():
    """
    Load Nelson-Siegel betas and VRP from data/processed/.

    Returns:
        betas: DataFrame (Date x 30) — B0, B1, B2 for each of 10 stocks
        vrp:   DataFrame (Date x 1)  — daily Variance Risk Premium
    """
    betas = pd.read_csv(PROCESSED_PATH / "betas.csv", index_col=0, parse_dates=True)
    vrp   = pd.read_csv(PROCESSED_PATH / "vrp.csv",   index_col=0, parse_dates=True)
    vrp.columns = ["VRP"]
    print(f"Betas: {betas.shape}, VRP: {vrp.shape}")
    return betas, vrp


def build_features(betas, vrp, lags):
    """
    Build the feature matrix X for XGBoost.

    Features for each date t:
      - Lagged betas: B0, B1, B2 for each stock at t-1, t-2, t-5
        Total: 30 betas x 3 lags = 90 columns
      - Lagged VRP: VRP at t-1, t-2, t-5
        Total: 3 columns
      Grand total: 93 feature columns

    Using lagged values ensures no look-ahead bias — we only use
    information available before the prediction date.

    Args:
        betas: DataFrame of daily Nelson-Siegel betas
        vrp:   DataFrame of daily VRP values
        lags:  list of lag periods (e.g. [1, 2, 5])

    Returns:
        X: DataFrame with 93 feature columns and same Date index as betas
    """
    features = {}

    # Lagged betas — shift by lag days
    for lag in lags:
        shifted = betas.shift(lag)
        shifted.columns = [f"{col}_lag{lag}" for col in betas.columns]
        features[f"betas_lag{lag}"] = shifted

    # Lagged VRP — forward-looking market sentiment signal
    for lag in lags:
        features[f"vrp_lag{lag}"] = vrp.shift(lag).rename(
            columns={"VRP": f"VRP_lag{lag}"}
        )

    X = pd.concat(features.values(), axis=1)
    return X


def walk_forward_xgboost(X, y_col, betas, min_train, gap, retrain_freq,
                          backtest_start, xgb_params):
    """
    Walk-forward XGBoost forecasting with expanding training window.

    For each prediction date t (starting from backtest_start):
      1. Training data: all observations up to t - gap
      2. If enough data (>= min_train) and retrain due: fit a new XGBoost model
      3. Predict beta for date t using the current model

    The expanding window ensures the model learns from all available
    history, while the gap prevents look-ahead bias from autocorrelation.

    Args:
        X:              feature matrix (Date x 93 features)
        y_col:          target column name (e.g. "CRM_B0")
        betas:          DataFrame of observed betas (for training targets)
        min_train:      minimum training days before first prediction
        gap:            gap in days between training end and prediction
        retrain_freq:   retrain model every N days
        backtest_start: first prediction date
        xgb_params:     XGBoost hyperparameters dict

    Returns:
        Series of predicted values with Date index (backtest_start onwards)
    """
    backtest_start = pd.Timestamp(backtest_start)
    all_dates  = X.index
    pred_dates = all_dates[all_dates >= backtest_start]

    predictions    = {}
    last_train_idx = -1
    model          = None

    for date in pred_dates:
        t = all_dates.get_loc(date)
        train_end = t - gap  # training data ends gap days before prediction

        # Skip if not enough training data
        if train_end < min_train:
            continue

        # Retrain every retrain_freq days
        if last_train_idx == -1 or (t - last_train_idx) >= retrain_freq:
            X_train = X.iloc[:train_end].dropna()
            y_train = betas[y_col].iloc[:train_end].loc[X_train.index]
            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            last_train_idx = t

        # Predict for current date
        X_pred = X.loc[[date]].dropna()
        if len(X_pred) == 0 or model is None:
            continue

        predictions[date] = model.predict(X_pred)[0]

    return pd.Series(predictions, name=y_col)


def walk_forward_arma(betas_col, min_train, gap, backtest_start, retrain_freq):
    """
    Walk-forward ARMA(1,0,1) forecasting — linear benchmark.

    ARMA(1,0,1) = one autoregressive term + one moving average term.
    This captures:
      - AR(1): yesterday's beta is the strongest predictor of today's
      - MA(1): the forecast error from yesterday informs today's prediction

    ARMA is used as the linear benchmark for Research Question 1:
    "Does XGBoost outperform ARMA in forecasting and portfolio performance?"

    ARMA generates more stable predictions than XGBoost because it does
    not react to every small fluctuation in the feature matrix. This
    results in lower portfolio turnover and lower transaction costs.

    Args:
        betas_col:      Series of observed betas for one stock/parameter
        min_train:      minimum training days before first prediction
        gap:            gap in days between training end and prediction
        backtest_start: first prediction date
        retrain_freq:   retrain model every N days

    Returns:
        Series of predicted values with Date index (backtest_start onwards)
    """
    backtest_start = pd.Timestamp(backtest_start)
    all_dates  = betas_col.index
    pred_dates = all_dates[all_dates >= backtest_start]

    predictions    = {}
    last_train_idx = -1
    model_result   = None

    for date in pred_dates:
        t = all_dates.get_loc(date)
        train_end = t - gap
        if train_end < min_train:
            continue

        # Retrain every retrain_freq days
        if last_train_idx == -1 or (t - last_train_idx) >= retrain_freq:
            train_data = betas_col.iloc[:train_end].dropna()
            try:
                model = ARIMA(train_data, order=(1, 0, 1))
                model_result = model.fit()
                last_train_idx = t
            except Exception as e:
                print(f"FIT ERROR {date}: {e}")
                model_result = None

        # Forecast gap days ahead and take the last value
        # This gives us the prediction for date t
        try:
            if model_result is not None:
                predictions[date] = model_result.forecast(steps=gap).iloc[-1]
            else:
                predictions[date] = np.nan
        except Exception as e:
            print(f"FORECAST ERROR {date}: {e}")
            predictions[date] = np.nan

    return pd.Series(predictions, name=betas_col.name)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Step 1: Load betas and VRP
    betas, vrp = load_data()

    # Step 2: Build feature matrix
    # X contains lagged betas and lagged VRP for each date
    print("\nBuilding features...")
    X = build_features(betas, vrp, LAGS)

    xgb_predictions  = {}
    arma_predictions = {}

    # Step 3: Walk-forward forecasting for each stock and each beta parameter
    # Total: 10 stocks x 3 parameters = 30 series to forecast
    for ticker in TICKERS:
        for param in ["B0", "B1", "B2"]:
            col = f"{ticker}_{param}"
            print(f"Forecasting {col}...")

            # XGBoost — non-linear, uses all 93 features
            xgb_predictions[col] = walk_forward_xgboost(
                X, col, betas,
                MIN_TRAIN_DAYS, GAP_DAYS, RETRAIN_FREQ,
                BACKTEST_START, XGB_PARAMS
            )

            # ARMA — linear benchmark, uses only the beta series itself
            arma_predictions[col] = walk_forward_arma(
                betas[col], MIN_TRAIN_DAYS, GAP_DAYS,
                BACKTEST_START, RETRAIN_FREQ
            )

    # Step 4: Combine predictions into DataFrames
    xgb_df  = pd.concat(xgb_predictions,  axis=1)
    xgb_df.columns  = list(xgb_predictions.keys())
    arma_df = pd.concat(arma_predictions, axis=1)
    arma_df.columns = list(arma_predictions.keys())

    print(f"\nXGBoost predictions — Shape: {xgb_df.shape}")
    print(f"ARMA predictions    — Shape: {arma_df.shape}")
    print(f"From: {xgb_df.index[0].date()} To: {xgb_df.index[-1].date()}")

    # Step 5: Save to data/processed/
    xgb_df.to_csv(PROCESSED_PATH  / "predictions_xgboost.csv")
    arma_df.to_csv(PROCESSED_PATH / "predictions_arma.csv")
    print("\nSaved: predictions_xgboost.csv and predictions_arma.csv")

    print("\nforecasting.py completed successfully.")


if __name__ == "__main__":
    main()