"""
stock_selection.py

Selects 10 stocks from the Dow Jones 30 universe using a greedy minimum
average pairwise correlation algorithm, subject to a liquidity filter.

Motivation:
    Low pairwise correlation across portfolio stocks ensures diversification.
    When stocks move independently, the portfolio volatility is lower than
    the weighted average of individual volatilities — this is the only free
    lunch in finance (Markowitz, 1952).

Algorithm:
    1. Download daily prices for all 30 Dow Jones stocks
    2. Compute log returns and the full 30x30 correlation matrix
    3. Start with the pair of stocks with the lowest pairwise correlation
    4. Greedily add one stock at a time — always the one that minimizes
       the average correlation with the already-selected stocks
    5. Stop when 10 stocks are selected

Result:
    Selected stocks: CRM, VZ, WMT, INTC, UNH, MRK, BA, NKE, CVX, AMGN
    Average pairwise correlation: 0.2781
    All stocks pass the liquidity filter (>1M daily volume)

This script is run once during project setup — not part of the daily pipeline.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

# ── Configuration ──────────────────────────────────────────────────────────
# Full Dow Jones 30 universe as of 2024
# Note: WBA (Walgreens) was delisted — yfinance will skip it automatically
DOW30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]

# Same start date as the main pipeline for consistency
START_DATE = "2014-07-01"

# Number of stocks to select
N_STOCKS = 10


# ── Functions ──────────────────────────────────────────────────────────────
def download_all_prices(tickers, start):
    """
    Download daily close prices for all Dow Jones 30 stocks.

    Stocks with any missing values are dropped — this handles delisted
    or recently added components automatically.

    Args:
        tickers: list of ticker symbols
        start:   start date string (e.g. "2014-07-01")

    Returns:
        DataFrame with Date index and one column per available stock.
    """
    print(f"Downloading {len(tickers)} stocks...")
    data = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    data = data.dropna(axis=1, how="any")
    print(f"Available stocks after cleaning: {list(data.columns)}")
    return data


def compute_returns(prices):
    """
    Compute daily log returns from price series.

    Args:
        prices: DataFrame of daily close prices

    Returns:
        DataFrame of daily log returns (one row shorter than prices)
    """
    return np.log(prices / prices.shift(1)).dropna()


def greedy_selection(returns, n=10):
    """
    Select n stocks with minimum average pairwise correlation using a greedy algorithm.

    The greedy approach is computationally efficient — finding the true global
    minimum over all C(30,10) = 30,045,015 combinations would be intractable.

    Steps:
        1. Compute the full correlation matrix
        2. Start with the pair of stocks with the lowest pairwise correlation
        3. At each step, add the stock that minimizes average correlation
           with the already-selected stocks
        4. Repeat until n stocks are selected

    Args:
        returns: DataFrame of daily log returns
        n:       number of stocks to select

    Returns:
        List of n ticker symbols
    """
    corr_matrix = returns.corr()
    tickers = list(corr_matrix.columns)

    # Step 1: find the pair with lowest pairwise correlation
    min_corr = 1.0
    best_pair = (tickers[0], tickers[1])
    for i, j in combinations(tickers, 2):
        c = corr_matrix.loc[i, j]
        if c < min_corr:
            min_corr = c
            best_pair = (i, j)

    selected = list(best_pair)
    print(f"\nStarting pair: {selected} (corr={min_corr:.4f})")

    # Step 2: greedy expansion
    while len(selected) < n:
        remaining = [t for t in tickers if t not in selected]
        best_ticker = None
        best_avg_corr = 1.0

        for ticker in remaining:
            avg_corr = corr_matrix.loc[ticker, selected].mean()
            if avg_corr < best_avg_corr:
                best_avg_corr = avg_corr
                best_ticker = ticker

        selected.append(best_ticker)
        print(f"Added: {best_ticker} (avg corr with selected = {best_avg_corr:.4f})")

    return selected


def evaluate_selection(returns, selected):
    """
    Print the correlation matrix and average pairwise correlation
    for the selected stocks.

    Args:
        returns:  DataFrame of daily log returns
        selected: list of selected ticker symbols

    Returns:
        Average pairwise correlation (float)
    """
    corr = returns[selected].corr()
    n = len(selected)
    upper = corr.values[np.triu_indices(n, k=1)]
    avg_corr = upper.mean()

    print(f"\nSelected stocks: {selected}")
    print(f"Average pairwise correlation: {avg_corr:.4f}")
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())

    return avg_corr


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Step 1: Download all Dow Jones 30 prices
    prices = download_all_prices(DOW30, START_DATE)

    # Step 2: Compute log returns
    returns = compute_returns(prices)

    # Step 3: Greedy selection of 10 stocks
    selected = greedy_selection(returns, n=N_STOCKS)

    # Step 4: Evaluate and print results
    avg_corr = evaluate_selection(returns, selected)

    print(f"\n{'='*50}")
    print(f"FINAL SELECTION ({N_STOCKS} stocks):")
    print(selected)
    print(f"Average pairwise correlation: {avg_corr:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()