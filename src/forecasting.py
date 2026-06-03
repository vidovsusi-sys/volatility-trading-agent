"""
forecasting.py
Walk-forward forecasting of Nelson-Siegel betas using XGBoost and ARMA.
Inputs: betas + VRP
Outputs: predicted betas for each day from 2018 onwards
Saves results to data/processed/
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
MIN_TRAIN_DAYS = 500
GAP_DAYS       = 5
RETRAIN_FREQ   = 21
BACKTEST_START = "2018-01-01"

# XGBoost parameters
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}

LAGS = [1, 2, 5]


def load_data():
    """Load betas and VRP."""
    betas = pd.read_csv(PROCESSED_PATH / "betas.csv", index_col=0, parse_dates=True)
    vrp   = pd.read_csv(PROCESSED_PATH / "vrp.csv",   index_col=0, parse_dates=True)
    vrp.columns = ["VRP"]
    print(f"Betas: {betas.shape}, VRP: {vrp.shape}")
    return betas, vrp


def build_features(betas, vrp, lags):
    """Build feature matrix with lagged betas and lagged VRP."""
    features = {}
    for lag in lags:
        shifted = betas.shift(lag)
        shifted.columns = [f"{col}_lag{lag}" for col in betas.columns]
        features[f"betas_lag{lag}"] = shifted
    for lag in lags:
        features[f"vrp_lag{lag}"] = vrp.shift(lag).rename(columns={"VRP": f"VRP_lag{lag}"})
    X = pd.concat(features.values(), axis=1)
    return X


def walk_forward_xgboost(X, y_col, betas, min_train, gap, retrain_freq, backtest_start, xgb_params):
    """Walk-forward XGBoost forecasting with monthly retraining."""
    backtest_start = pd.Timestamp(backtest_start)
    all_dates = X.index
    pred_dates = all_dates[all_dates >= backtest_start]

    predictions = {}
    last_train_idx = -1
    model = None

    for date in pred_dates:
        t = all_dates.get_loc(date)
        train_end = t - gap
        if train_end < min_train:
            continue

        if last_train_idx == -1 or (t - last_train_idx) >= retrain_freq:
            X_train = X.iloc[:train_end].dropna()
            y_train = betas[y_col].iloc[:train_end].loc[X_train.index]
            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            last_train_idx = t

        X_pred = X.loc[[date]].dropna()
        if len(X_pred) == 0 or model is None:
            continue

        predictions[date] = model.predict(X_pred)[0]

    return pd.Series(predictions, name=y_col)


def walk_forward_arma(betas_col, min_train, gap, backtest_start, retrain_freq):
    """Walk-forward ARMA forecasting with monthly retraining."""
    backtest_start = pd.Timestamp(backtest_start)
    all_dates = betas_col.index
    pred_dates = all_dates[all_dates >= backtest_start]

    predictions = {}
    last_train_idx = -1
    model_result = None

    for date in pred_dates:
        t = all_dates.get_loc(date)
        train_end = t - gap
        if train_end < min_train:
            continue

        if last_train_idx == -1 or (t - last_train_idx) >= retrain_freq:
            train_data = betas_col.iloc[:train_end].dropna()
            try:
                model = ARIMA(train_data, order=(1, 0, 1))
                model_result = model.fit()
                last_train_idx = t
            except Exception as e:
                print(f"FIT ERROR {date}: {e}")
                model_result = None

        try:
            if model_result is not None:
                predictions[date] = model_result.forecast(steps=gap).iloc[-1]
            else:
                predictions[date] = np.nan
        except Exception as e:
            print(f"FORECAST ERROR {date}: {e}")
            predictions[date] = np.nan

    return pd.Series(predictions, name=betas_col.name)


def main():
    betas, vrp = load_data()

    print("\nBuilding features...")
    X = build_features(betas, vrp, LAGS)

    xgb_predictions = {}
    arma_predictions = {}

    for ticker in TICKERS:
        for param in ["B0", "B1", "B2"]:
            col = f"{ticker}_{param}"
            print(f"Forecasting {col}...")

            xgb_predictions[col] = walk_forward_xgboost(
                X, col, betas,
                MIN_TRAIN_DAYS, GAP_DAYS, RETRAIN_FREQ,
                BACKTEST_START, XGB_PARAMS
            )

            arma_predictions[col] = walk_forward_arma(
                betas[col], MIN_TRAIN_DAYS, GAP_DAYS,
                BACKTEST_START, RETRAIN_FREQ
            )

    xgb_df = pd.concat(xgb_predictions, axis=1)
    xgb_df.columns = list(xgb_predictions.keys())
    arma_df = pd.concat(arma_predictions, axis=1)
    arma_df.columns = list(arma_predictions.keys())

    print(f"\nXGBoost predictions — Shape: {xgb_df.shape}")
    print(f"ARMA predictions    — Shape: {arma_df.shape}")
    print(f"From: {xgb_df.index[0].date()} To: {xgb_df.index[-1].date()}")

    xgb_df.to_csv(PROCESSED_PATH / "predictions_xgboost.csv")
    arma_df.to_csv(PROCESSED_PATH / "predictions_arma.csv")
    print("\nSaved: predictions_xgboost.csv and predictions_arma.csv")

    print("\nforecasting.py completed successfully.")


if __name__ == "__main__":
    main()