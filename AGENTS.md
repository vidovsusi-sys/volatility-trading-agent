# AGENTS.md — Volatility Trading Agent

## Project context
Quantitative finance research project. The pipeline forecasts volatility
term structures of 10 stocks using Nelson-Siegel and XGBoost, and uses
those forecasts to optimize portfolio weights dynamically.

**Team:** Luca Anselmi · Stefan Vidovic · Arnel Hodza  
**Course:** Programming in Finance II — USI 2026

---

## Tools

AGENTS MAY use without asking:
- `python`, `pip`, `git add`, `git commit`, `git push`
- Read any file in the repository
- Run any script in `src/` or `agents/`

AGENTS MUST ask before:
- Deleting any file
- Modifying raw data in `data/raw/`
- Changing the list of 10 stocks
- Changing the research questions
- Changing lambda in `nelson_siegel.py`
- Changing transaction costs or smoothing window

AGENTS MUST NOT:
- Commit API keys, tokens, or credentials
- Use future data in training or normalization (look-ahead bias)
- Modify `data/raw/` after initial download

---

## Data rules

- `data/raw/` is IMMUTABLE — NEVER modify, overwrite, or delete files here
- All computed outputs go to `data/processed/` or `outputs/`
- MUST validate downloaded data for missing values before saving

---

## Pipeline order

Scripts MUST run in this exact order — each depends on the previous output:

`src/data_loader.py` → `src/volatility.py` → `src/nelson_siegel.py` → `src/vrp.py` → `src/forecasting.py` → `src/portfolio.py` → `src/backtest.py`

---

## Financial standards

- Volatility MUST always be annualized: `std * sqrt(252)`
- Walk-forward ONLY — NEVER use future data to train or normalize
- Transaction costs: ALWAYS apply 10bps per unit of daily turnover
- Portfolio weights MUST be smoothed with a 5-day rolling mean before backtesting
- Weight constraints: minimum 5%, maximum 25% per stock
- Lambda in Nelson-Siegel is fixed at 0.04 — do NOT change without discussion
- ARMA model order is fixed at (1,0,1) — do NOT change without discussion

---

## Code standards

- Language: Python 3.10+
- All scripts MUST run from the repository root
- Each script MUST have a docstring explaining inputs, outputs, and purpose
- All functions MUST have docstrings with Args and Returns
- Use meaningful variable names — no single letters except loop indices

---

## Git rules

- MUST commit after each meaningful step
- Commit message format: `[component] description`
  - Examples: `[data] add data_loader.py`, `[backtest] add 5-day smoothing`
- NEVER commit API keys, GitHub tokens, or credentials
- The Trading Agent MUST open a Pull Request after each pipeline run

---

## Output standards

- All metrics saved as CSV in `data/processed/`
- All plots saved as PNG in `outputs/figures/`
- Performance metrics MUST include: Sharpe, Max Drawdown, CAGR, Calmar
- Stress analysis MUST cover: COVID crash 2020, Bear market 2022