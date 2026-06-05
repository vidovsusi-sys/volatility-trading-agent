# Methodology

## Overview

This project forecasts the volatility term structure of 10 stocks using Nelson-Siegel decomposition and XGBoost, and uses those forecasts to dynamically optimize portfolio weights every day.

---

## 1. Stock Selection

10 stocks selected from the Dow Jones 30 using a greedy minimum average pairwise correlation algorithm with a liquidity check (>1M daily volume).

**Selected stocks:** CRM, VZ, WMT, INTC, UNH, MRK, BA, NKE, CVX, AMGN  
**Average pairwise correlation:** 0.2781  
**Data period:** 2014-07-01 to today

---

## 2. Realized Volatility

For each stock and each day, realized volatility is computed on 6 horizons:

    vol(t, N) = std( r(t-N+1), ..., r(t) ) × sqrt(252)

Where r(t) = log( P(t) / P(t-1) ) are daily log returns.

**Horizons:** 5, 10, 21, 42, 63, 126 trading days  
**Output:** 60 time series (6 horizons × 10 stocks)

---

## 3. Nelson-Siegel Decomposition

The Nelson-Siegel model compresses the 6-point volatility term structure into 3 parameters:

    σ(τ) = β₀ + β₁·f₁(τ) + β₂·f₂(τ)
    f₁(τ) = (1 - exp(-λτ)) / (λτ)
    f₂(τ) = f₁(τ) - exp(-λτ)

**Parameters:**
- β₀: long-run volatility level
- β₁: slope (short vs long term difference)
- β₂: curvature (hump in the middle of the curve)
- λ = 0.04 — fixed so that the peak of f₂ falls at τ = 25 days, the geometric midpoint of the 6 horizons: (5×10×21×42×63×126)^(1/6) ≈ 25

**Output:** 30 time series (3 parameters × 10 stocks)

---

## 4. Variance Risk Premium (VRP)

    VRP(t) = VIX(t) / 100 - realized_vol_SP500(t, 21 days)

The VRP captures how much the market pays in excess of realized volatility. Positive VRP means the market is nervous and pricing in risk preemptively. Negative VRP means realized volatility exceeded expectations — a crisis in progress. VRP is used as an exogenous input to XGBoost alongside the lagged betas.

---

## 5. Walk-Forward Forecasting

Both XGBoost and ARMA are trained using an expanding window walk-forward approach to avoid look-ahead bias. To predict day t, only data available up to day t-5 is used for training.

**Parameters:**
- Minimum training days: 500
- Gap: 5 days between training end and prediction date
- Retrain frequency: every 21 trading days
- Backtest start: 2018-01-01

**XGBoost features:** lagged betas at lag 1, 2, 5 and lagged VRP at lag 1, 2, 5 — total 93 features  
**ARMA:** ARIMA(1,0,1) on each beta series independently — linear benchmark for RQ1

---

## 6. Portfolio Methods

**Method A — Risk Parity**

    weight_i = (1 / β₀_i) / Σⱼ (1 / β₀_j)

Weights inversely proportional to predicted volatility level β₀.

**Method B — Shape Trading**

    score_i  = (β₀_i / std_β₀) + (|β₁_i| / std_β₁) + (|β₂_i| / std_β₂)
    weight_i = (1 / score_i) / Σⱼ (1 / score_j)

Uses the full term structure shape, normalized by historical standard deviations. Absolute values on β₁ and β₂ ensure both directions of slope and curvature deviation are penalized.

**Method C — Momentum**

    momentum_i = (P_t - P_{t-63}) / P_{t-63}
    score_i    = momentum_i / β₀_i
    weight_i   = max(score_i, 0) / Σⱼ max(score_j, 0)

Combines 63-day momentum with predicted volatility. Stocks with negative momentum receive zero weight.

**Weight constraints:** min 5%, max 25% per stock — enforced via scipy SLSQP optimizer  
**Weight smoothing:** 5-day rolling mean applied before backtesting to reduce daily turnover

---

## 7. Backtesting

**Period:** 2018-01-02 to today  
**Transaction costs:** 10 basis points per unit of daily turnover  
**Metrics:** Sharpe ratio, Max Drawdown, CAGR, Calmar ratio  
**Stress periods:** COVID crash (Mar–Apr 2020), Bear market (Sep–Dec 2022)

**Strategies compared:**

| Strategy | Description |
|----------|-------------|
| Equal Weighted | Fixed 10% per stock — passive benchmark |
| Historical Risk Parity | Weights from past 21-day realized volatility |
| Method A + XGBoost | Risk Parity with XGBoost-forecasted β₀ |
| Method B + XGBoost | Shape Trading with XGBoost forecasts |
| Method C + XGBoost | Momentum with XGBoost-forecasted β₀ |
| Method A + ARMA | Risk Parity with ARMA-forecasted β₀ |

---

## References

*To be completed in the LaTeX documentation.*