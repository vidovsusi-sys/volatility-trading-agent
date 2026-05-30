"""
backtest.py
Backtesting framework for the 5 portfolio strategies.

Strategies compared:
  1. Equal Weighted              — fixed 10% per stock (passive benchmark)
  2. Historical Risk Parity      — weights from past 21-day realized volatility
  3. Method A + XGBoost          — weights from XGBoost-forecasted B0
  4. Method B + XGBoost          — weights from normalized score (B0, B1, B2)
  5. Method C + XGBoost          — weights from momentum / forecasted B0
  (6) Method A + ARMA            — placeholder, skipped if NaN

Anti-look-ahead bias convention:
  - Weights w(t) are applied to returns r(t)
  - For Historical Risk Parity: shift(1) on realized volatility
  - XGBoost weights are NOT shifted (forecasts are already walk-forward)
  - Transaction costs: 10 basis points on daily turnover

Backtest period: 2018-01-02 to most recent available date.

Inputs:  data/raw/prices.csv
         data/processed/realized_volatility.csv
         data/processed/weights_method_{a,b,c}_xgboost.csv
         data/processed/weights_method_a_arma.csv (placeholder)

Outputs: data/processed/backtest_results.csv   (metrics per strategy)
         data/processed/equity_curves.csv      (daily equity curves)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
TICKERS = ["CRM", "VZ", "WMT", "INTC", "UNH", "MRK", "BA", "NKE", "CVX", "AMGN"]
PROCESSED_PATH = Path("data/processed")
RAW_PATH = Path("data/raw")

# Transaction costs: 10 basis points on daily turnover
TRANSACTION_COST_BPS = 0.001    # 10 bps = 0.001

# Stress periods for separate analysis
STRESS_PERIODS = {
    "COVID_crash_2020":   ("2020-03-01", "2020-04-30"),
    "Bear_market_2022":   ("2022-09-01", "2022-12-31"),
}

# Trading days per year (for annualization)
TRADING_DAYS = 252


# ── Step 2: Loaders ────────────────────────────────────────────────────────
def load_prices():
    """Load daily close prices for the 10 stocks."""
    path = RAW_PATH / "prices.csv"
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices = prices[TICKERS]
    print(f"Loaded prices — Shape: {prices.shape}")
    print(f"  From: {prices.index.min().date()} To: {prices.index.max().date()}")
    return prices


def compute_log_returns(prices):
    """Compute daily log returns from prices."""
    returns = np.log(prices / prices.shift(1))
    print(f"Computed log returns — Shape: {returns.shape}")
    return returns


def load_realized_vol_21d():
    """Load 21-day realized volatility for the 10 stocks."""
    path = PROCESSED_PATH / "realized_volatility.csv"
    rv = pd.read_csv(path, index_col=0, parse_dates=True)
    vol_21_cols = [f"{t}_vol_21" for t in TICKERS]
    vol_21 = rv[vol_21_cols].copy()
    vol_21.columns = TICKERS
    print(f"Loaded 21d realized volatility — Shape: {vol_21.shape}")
    print(f"  From: {vol_21.index.min().date()} To: {vol_21.index.max().date()}")
    return vol_21


def load_weights(method, model="xgboost"):
    """Load pre-computed portfolio weights for a method/model combination."""
    filename = f"weights_method_{method}_{model}.csv"
    path = PROCESSED_PATH / filename
    weights = pd.read_csv(path, index_col=0, parse_dates=True)
    weights = weights[TICKERS]
    if weights.isna().all().all():
        print(f"  [WARNING] {filename} contains only NaN — skipping.")
        return None
    print(f"Loaded weights — Method {method.upper()} ({model}) — "
          f"Shape: {weights.shape}")
    return weights


# ── Step 3: Build strategies ───────────────────────────────────────────────
def build_equal_weighted(dates):
    """Equal-weighted strategy: each stock gets 1/N at all times."""
    n = len(TICKERS)
    return pd.DataFrame(1.0 / n, index=dates, columns=TICKERS)


def build_historical_risk_parity(vol_21, dates):
    """
    Historical Risk Parity: weights inversely proportional to past 21-day
    realized volatility, with shift(1) to avoid look-ahead bias.
    """
    vol_lagged = vol_21.shift(1)
    inv_vol = 1.0 / vol_lagged
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights.reindex(dates)


def build_all_strategies(returns, vol_21, w_a, w_b, w_c, w_a_arma):
    """Build the dictionary of all strategies' weight DataFrames."""
    common_start = max(w_a.index.min(), w_b.index.min(), w_c.index.min())
    common_end   = min(returns.index.max(), w_a.index.max(),
                       w_b.index.max(), w_c.index.max())
    dates = returns.loc[common_start:common_end].index

    print(f"Backtest period: {dates.min().date()} -> {dates.max().date()}")
    print(f"Trading days: {len(dates)}")

    strategies = {}
    strategies["Equal_Weighted"] = build_equal_weighted(dates)
    strategies["Historical_Risk_Parity"] = build_historical_risk_parity(
        vol_21, dates)
    strategies["Method_A_XGBoost"] = w_a.reindex(dates)
    strategies["Method_B_XGBoost"] = w_b.reindex(dates)
    strategies["Method_C_XGBoost"] = w_c.reindex(dates).shift(1)

    if w_a_arma is not None:
        strategies["Method_A_ARMA"] = w_a_arma.reindex(dates)
    else:
        print("[INFO] Method_A_ARMA skipped (NaN forecasts).")

    return strategies, dates


# ── Step 4: Simulate strategies ────────────────────────────────────────────
def simulate_strategy(weights, returns, transaction_cost_bps):
    """
    Run the daily portfolio simulation for one strategy.

    For each day t we compute:
      - simple_returns(t) = exp(log_returns(t)) - 1
      - gross_return(t)   = sum_i( w_i(t) * simple_i(t) )
      - turnover(t)       = sum_i( |w_i(t) - w_i(t-1)| )
      - cost(t)           = transaction_cost_bps * turnover(t)
      - net_return(t)     = gross_return(t) - cost(t)
      - equity(t)         = product of (1 + net_return) up to t

    The first day's turnover is the sum of initial weights (1.0): we model
    this as the cost of building the portfolio from cash.

    Args:
        weights: DataFrame (Date x TICKERS) of portfolio weights.
        returns: DataFrame (Date x TICKERS) of log returns.
        transaction_cost_bps: cost per unit of turnover (e.g. 0.001 = 10 bps).

    Returns:
        DataFrame indexed by Date with columns:
        gross_return, turnover, cost, net_return, equity.
    """
    # Align dates between weights and returns
    common_idx = weights.index.intersection(returns.index)
    w = weights.loc[common_idx].copy()
    r = returns.loc[common_idx].copy()

    # Convert log returns to simple returns (additive across stocks)
    simple_r = np.exp(r) - 1.0

    # Daily gross portfolio return: dot product of weights and stock returns
    gross_return = (w * simple_r).sum(axis=1)

    # Turnover: L1 distance between consecutive weight vectors
    turnover = (w - w.shift(1)).abs().sum(axis=1)
    # First day: portfolio is built from cash, so turnover = sum of weights
    turnover.iloc[0] = w.iloc[0].abs().sum()

    # Transaction costs and net return
    cost = transaction_cost_bps * turnover
    net_return = gross_return - cost

    # Equity curve, normalized to start at 1.0
    equity = (1.0 + net_return).cumprod()

    return pd.DataFrame({
        "gross_return": gross_return,
        "turnover":     turnover,
        "cost":         cost,
        "net_return":   net_return,
        "equity":       equity,
    })


def simulate_all_strategies(strategies, returns,
                            transaction_cost_bps=TRANSACTION_COST_BPS):
    """Run the simulation for every strategy and return a dict of results."""
    sim_results = {}
    for name, w in strategies.items():
        print(f"  Simulating {name} ...")
        sim_results[name] = simulate_strategy(w, returns, transaction_cost_bps)
    return sim_results
# ── Step 5: Compute risk and performance metrics ───────────────────────────
def compute_metrics(sim_df):
    """
    Compute Sharpe ratio, Max Drawdown, CAGR and Calmar ratio
    for a single strategy's simulation DataFrame.

    Conventions:
      - Risk-free rate = 0 (standard for short backtests)
      - Annualization factor: 252 trading days
      - Returns are net of transaction costs

    Args:
        sim_df: output of simulate_strategy(), with 'net_return' and 'equity'.

    Returns:
        dict with keys: sharpe, max_drawdown, cagr, calmar.
    """
    net_ret = sim_df["net_return"].dropna()
    equity  = sim_df["equity"].dropna()

    # Sharpe ratio (annualized)
    mean_daily = net_ret.mean()
    std_daily  = net_ret.std()
    sharpe = (mean_daily / std_daily) * np.sqrt(TRADING_DAYS)

    # Max Drawdown: largest peak-to-trough loss
    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    max_dd      = drawdown.min()    # negative number, e.g. -0.30 = -30%

    # CAGR: compound annual growth rate
    n_days   = len(net_ret)
    n_years  = n_days / TRADING_DAYS
    cagr     = equity.iloc[-1] ** (1.0 / n_years) - 1.0

    # Calmar ratio: CAGR per unit of drawdown
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "cagr":         cagr,
        "calmar":       calmar,
    }


def compute_all_metrics(sim_results):
    """
    Compute metrics for every strategy and return a tidy DataFrame.

    Rows = strategies, Columns = metrics.
    """
    rows = {name: compute_metrics(df) for name, df in sim_results.items()}
    metrics_df = pd.DataFrame(rows).T
    metrics_df.index.name = "strategy"
    return metrics_df
# ── Step 6: Stress period analysis ─────────────────────────────────────────
def compute_stress_metrics(sim_df, start, end):
    """
    Compute performance metrics for a single strategy on a stress sub-period.

    Differences vs compute_metrics():
      - Total Return (NOT annualized) instead of CAGR, since extrapolating
        a 2-month return to a yearly figure is misleading.
      - Max Drawdown is computed WITHIN the window (local peaks/troughs).

    Args:
        sim_df: full backtest output for one strategy.
        start, end: inclusive bounds of the stress window (e.g. "2020-03-01").

    Returns:
        dict with keys: total_return, vol_annual, sharpe, max_drawdown, n_days.
    """
    window = sim_df.loc[start:end]
    if window.empty:
        return None

    net_ret = window["net_return"].dropna()
    equity  = window["equity"].dropna()

    # Total return over the period (NOT annualized)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0

    # Annualized volatility and Sharpe ratio
    vol_ann = net_ret.std() * np.sqrt(TRADING_DAYS)
    sharpe  = (net_ret.mean() / net_ret.std()) * np.sqrt(TRADING_DAYS)

    # Max Drawdown within the window
    running_max = equity.cummax()
    drawdown    = (equity - running_max) / running_max
    max_dd      = drawdown.min()

    return {
        "total_return": total_return,
        "vol_annual":   vol_ann,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "n_days":       len(window),
    }


def compute_all_stress_metrics(sim_results, periods=STRESS_PERIODS):
    """
    Compute stress metrics for every strategy across every stress period.

    Returns:
        dict {period_name: DataFrame}, where each DataFrame has
        strategies as rows and metrics as columns.
    """
    results = {}
    for period_name, (start, end) in periods.items():
        rows = {}
        for strat_name, sim_df in sim_results.items():
            metrics = compute_stress_metrics(sim_df, start, end)
            if metrics is not None:
                rows[strat_name] = metrics
        df = pd.DataFrame(rows).T
        df.index.name = "strategy"
        results[period_name] = df
    return results
# ── Step 7: Save outputs ───────────────────────────────────────────────────
def save_outputs(metrics_df, sim_results, stress_results,
                 out_path=PROCESSED_PATH):
    """
    Save backtest results to CSV files for downstream use (LaTeX tables,
    plotting, dashboards).

    Files produced:
      - backtest_results.csv: full-period metrics per strategy
      - equity_curves.csv:    daily equity curve per strategy
      - stress_<period>.csv:  one file per stress period with local metrics

    Args:
        metrics_df:     output of compute_all_metrics().
        sim_results:    output of simulate_all_strategies().
        stress_results: output of compute_all_stress_metrics().
        out_path:       directory where to write the CSVs.
    """
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Full-period metrics
    fp = out_path / "backtest_results.csv"
    metrics_df.to_csv(fp)
    print(f"Saved metrics       -> {fp}")

    # 2. Equity curves (one column per strategy)
    equity_df = pd.DataFrame(
        {name: df["equity"] for name, df in sim_results.items()}
    )
    fp = out_path / "equity_curves.csv"
    equity_df.to_csv(fp)
    print(f"Saved equity curves -> {fp}  shape={equity_df.shape}")

    # 3. Stress period results (one CSV per period)
    for period_name, df in stress_results.items():
        fp = out_path / f"stress_{period_name}.csv"
        df.to_csv(fp)
        print(f"Saved stress period -> {fp}")

# ── Main orchestration ─────────────────────────────────────────────────────
def main():
    """Entry point: run the full backtest on all 5 strategies."""
    print(f"\n{'='*60}")
    print(f"BACKTESTING — 5 portfolio strategies")
    print(f"{'='*60}\n")

    # ── Step 2: Load all inputs ────────────────────────────────────────────
    print("\n--- Step 2: Loading inputs ---\n")
    prices  = load_prices()
    returns = compute_log_returns(prices)
    vol_21  = load_realized_vol_21d()

    print()
    weights_a_xgb  = load_weights("a", "xgboost")
    weights_b_xgb  = load_weights("b", "xgboost")
    weights_c_xgb  = load_weights("c", "xgboost")
    weights_a_arma = load_weights("a", "arma")

    # ── Step 3: Build strategies ───────────────────────────────────────────
    print("\n--- Step 3: Building strategies ---\n")
    strategies, dates = build_all_strategies(
        returns, vol_21,
        weights_a_xgb, weights_b_xgb, weights_c_xgb, weights_a_arma,
    )

    print("\nVerification — each strategy's row sums should be ~1.0:")
    for name, w in strategies.items():
        row_sums = w.sum(axis=1)
        print(f"  {name:30s} | shape={w.shape} | "
              f"row sum mean={row_sums.mean():.4f} (std={row_sums.std():.6f})")

    # ── Step 4: Simulate strategies ────────────────────────────────────────
    print("\n--- Step 4: Simulating strategies ---\n")
    sim_results = simulate_all_strategies(strategies, returns)

    print("\nSummary — final equity (1.0 = initial capital):")
    for name, df in sim_results.items():
        final_eq    = df["equity"].iloc[-1]
        total_cost  = df["cost"].sum()
        avg_turnov  = df["turnover"].mean()
        print(f"  {name:30s} | final eq = {final_eq:6.4f} | "
              f"total cost = {total_cost:.4f} | "
              f"avg daily turnover = {avg_turnov:.4f}")

    # ── Step 5: Compute risk and performance metrics ───────────────────────
    print("\n--- Step 5: Computing risk and performance metrics ---\n")
    metrics_df = compute_all_metrics(sim_results)

    # Pretty print with percentages where appropriate
    print(f"{'Strategy':30s} | {'Sharpe':>7s} | {'MaxDD':>8s} | "
          f"{'CAGR':>7s} | {'Calmar':>7s}")
    print("-" * 75)
    for name, row in metrics_df.iterrows():
        print(f"{name:30s} | {row['sharpe']:7.3f} | "
              f"{row['max_drawdown']*100:7.2f}% | "
              f"{row['cagr']*100:6.2f}% | {row['calmar']:7.3f}")

    # ── Step 6: Stress period analysis ─────────────────────────────────────
    print("\n--- Step 6: Stress period analysis ---")
    stress_results = compute_all_stress_metrics(sim_results)

    for period_name, df in stress_results.items():
        start, end = STRESS_PERIODS[period_name]
        print(f"\n{period_name}  ({start} -> {end})")
        print("-" * 85)
        print(f"{'Strategy':30s} | {'TotRet':>8s} | {'Vol(ann)':>9s} | "
              f"{'Sharpe':>7s} | {'MaxDD':>8s}")
        print("-" * 85)
        for name, row in df.iterrows():
            print(f"{name:30s} | {row['total_return']*100:7.2f}% | "
                  f"{row['vol_annual']*100:8.2f}% | {row['sharpe']:7.3f} | "
                  f"{row['max_drawdown']*100:7.2f}%")
    # ── Step 7: Save outputs to CSV ────────────────────────────────────────
    print("\n--- Step 7: Saving outputs ---\n")
    save_outputs(metrics_df, sim_results, stress_results)
    print("\n[Step 7] Outputs saved successfully. Backtest complete")
    



if __name__ == "__main__":
    main()