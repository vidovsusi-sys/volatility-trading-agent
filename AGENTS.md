# AGENTS.md — Volatility Trading Agent

## Project context

Quantitative finance research project. The pipeline forecasts volatility 
term structures of 10 stocks using Nelson-Siegel and XGBoost, and uses 
those forecasts to optimize portfolio weights dynamically.

## Tools

AGENTS MAY use without asking: python, pip, git add, git commit, git push,
read any file in the repository.

AGENTS MUST ask before: deleting files, modifying raw data, changing 
the list of stocks, changing research questions.

## Data rules

- `data/raw/` is IMMUTABLE. NEVER modify, overwrite, or delete raw data.
- All computed outputs go to `data/processed/` or `outputs/`.
- Always validate downloaded data for missing values before saving.

## Code standards

- Language: Python 3.10+
- All scripts must run from the repository root.
- Each script must have a clear docstring explaining what it does.
- Use meaningful variable names — no single letters except loop indices.

## Financial standards

- Volatility is always annualized: multiply daily std by sqrt(252).
- Walk-forward only: never use future data to train or normalize.
- Transaction costs: always apply 10bps per unit of daily turnover.

## Modes

The pipeline supports two modes:
- `--mode backtest` — runs full historical pipeline from 2015 to today
- `--mode live` — adds today's data and updates forecasts and weights

## Git rules

- Commit after each meaningful step — not after every line.
- Commit message format: [component] description
  Examples: [data] add data_loader.py, [model] implement XGBoost walk-forward
- NEVER commit raw data files or model weights.

## Output format

- All results saved as CSV in `outputs/backtesting/`.
- All plots saved as PNG in `outputs/plots/`.
- Performance metrics must include: Sharpe ratio, Max Drawdown, Calmar ratio.