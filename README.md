# Volatility Trading Agent

A trading agent that forecasts the volatility term structure of 10 stocks using Nelson-Siegel and XGBoost, then uses those forecasts to dynamically optimize portfolio weights every day.

**Course:** Programming in Finance II — USI 2026  
**Project:** 2.6 Algorithmic Trading  
**Team:** Luca, Stefan, Arnel

---

## What this project does

Most trading systems try to predict whether a stock will go up or down. We take a different approach, we predict how volatile each stock will be tomorrow, then use that forecast to allocate portfolio weights.

The key insight is that volatility has memory. If today's market is agitated, tomorrow it tends to be agitated again. This is called volatility clustering (Engle, 1982). We model this with Nelson-Siegel and forecast it with XGBoost.

---

## Setup and installation

**Requirements:** Python 3.10+, Git

```bash
git clone https://github.com/vidovsusi-sys/volatility-trading-agent.git
cd volatility-trading-agent
pip install -r requirements.txt
```

---

## How to run

```bash
python src/data_loader.py    # Download market data
python src/volatility.py     # Compute realized volatility
python src/nelson_siegel.py  # Fit Nelson-Siegel model
python src/forecasting.py    # Forecast betas with XGBoost
python src/portfolio.py      # Compute portfolio weights
python src/backtesting.py    # Run backtesting
streamlit run dashboard/app.py  # Launch dashboard
```
---

## Architecture

The pipeline has 6 stages that feed into each other sequentially:

**Stage 1** — Raw prices downloaded from yfinance  
**Stage 2** — Realized volatility on 6 horizons: 5, 10, 21, 42, 63, 126 days  
**Stage 3** — Nelson-Siegel fitting → β0 (level), β1 (slope), β2 (curvature)  
**Stage 3b** — VRP computed as VIX minus realized volatility of S&P500, used as exogenous signal in XGBoost  
**Stage 4** — XGBoost walk-forward → forecasts tomorrow's betas using only past data 
**Stage 5** — Portfolio optimization → Methods A, B, C  
**Stage 6** — Backtesting → 5 strategies compared from 2018 to today  

**Two AI agents automate the pipeline:**
- **Data Agent** — downloads and validates market data, commits to GitHub
- **Trading Agent** — runs the full pipeline and opens a Pull Request with results

---

## Folder structure

| Path | Description |
|------|-------------|
| `src/data_loader.py` | Downloads prices from yfinance |
| `src/volatility.py` | Realized volatility on 6 horizons |
| `src/nelson_siegel.py` | Nelson-Siegel fitting, extracts β0 β1 β2 |
| `src/forecasting.py` | XGBoost and ARMA walk-forward |
| `src/portfolio.py` | Portfolio weights — Method A, B, C |
| `src/backtesting.py` | Backtesting engine and metrics |
| `dashboard/app.py` | Streamlit dashboard |
| `data/raw/` | Immutable raw prices from yfinance |
| `data/processed/` | Computed betas, forecasts, weights |
| `agents/` | Agent descriptions |
| `outputs/` | Results, metrics, plots |

## Data sources

| Dataset | Source | Ticker | Period |
|---------|--------|--------|--------|
| Stock prices | yfinance | 10 stocks from Dow Jones 30 | 2014-07-01 → today |
| VIX | yfinance | ^VIX | 2014-07-01 → today |
| S&P 500 | yfinance | ^GSPC | 2014-07-01 → today |
| VRP | Computed internally | VIX − realized vol S&P500 | 2015-01-02 → today |

Raw data is saved in `data/raw/` and never modified after download.

---

## AI tools used

Claude (Anthropic), used for project design, methodology, code development, and documentation.