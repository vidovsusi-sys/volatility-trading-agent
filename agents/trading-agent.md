# Trading Agent

## Role
Runs the full pipeline and produces portfolio weights and performance metrics.

## Trigger
Run after Data Agent completes successfully.

## Modes

**Backtest mode** (`--mode backtest`)
Runs full historical pipeline from 2015 to today.
Produces complete backtesting results for all 5 strategies.

**Live mode** (`--mode live`)
Adds today's data and updates forecasts and weights.
Appends new row to results without rerunning full history.

## Actions
1. Compute realized volatility on 6 horizons
2. Fit Nelson-Siegel model — extract β0, β1, β2 per stock per day
3. Forecast tomorrow's betas with XGBoost walk-forward
4. Compute portfolio weights — Method A, B, C
5. Run backtesting — compare 5 strategies with transaction costs
6. Save results to `outputs/backtesting/` and plots to `outputs/plots/`
7. Open a Pull Request on GitHub with results summary

## MUST NOT
- Use future data at any point in the pipeline
- Modify files in `data/raw/`
- Skip transaction costs in backtesting
- Open PR if pipeline fails