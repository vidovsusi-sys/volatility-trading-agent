# Data Agent

## Role
Downloads and validates daily market data for all assets.

## Trigger
Run manually or scheduled after market close.

## Actions
1. Download prices for 10 stocks, VIX, S&P500 using yfinance
2. Validate data — check for missing values and zero prices
3. Save to `data/raw/` — overwrites existing files with latest data
4. Commit to GitHub with message: [data-agent] updated prices YYYY-MM-DD

## MUST NOT
- Modify any existing file in `data/raw/`
- Run during market hours
- Commit if validation fails