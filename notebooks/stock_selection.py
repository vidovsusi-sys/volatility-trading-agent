"""
stock_selection.py
Selects 10 stocks from Dow Jones 30 with minimum average pairwise correlation.
Uses a greedy approach for computational efficiency.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from itertools import combinations

# ── Dow Jones 30 tickers ───────────────────────────────────────────────────
DOW30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]

START_DATE = "2014-07-01"
N_STOCKS = 10


def download_all_prices(tickers, start):
    """Download prices for all Dow Jones 30 stocks."""
    print(f"Downloading {len(tickers)} stocks...")
    data = yf.download(tickers, start=start, auto_adjust=True)["Close"]
    data = data.dropna(axis=1, how="any")
    print(f"Available stocks after cleaning: {list(data.columns)}")
    return data


def compute_returns(prices):
    """Compute log returns."""
    return np.log(prices / prices.shift(1)).dropna()


def greedy_selection(returns, n=10):
    """
    Greedy algorithm to select n stocks with minimum average pairwise correlation.
    Start with the pair of stocks with lowest correlation,
    then add one stock at a time that minimizes average correlation with selected.
    """
    corr_matrix = returns.corr()
    tickers = list(corr_matrix.columns)

    # Start with the pair with lowest correlation
    min_corr = 1.0
    best_pair = (tickers[0], tickers[1])
    for i, j in combinations(tickers, 2):
        c = corr_matrix.loc[i, j]
        if c < min_corr:
            min_corr = c
            best_pair = (i, j)

    selected = list(best_pair)
    print(f"\nStarting pair: {selected} (corr={min_corr:.4f})")

    # Greedy expansion
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
    """Print correlation matrix and average correlation for selected stocks."""
    corr = returns[selected].corr()
    n = len(selected)
    upper = corr.values[np.triu_indices(n, k=1)]
    avg_corr = upper.mean()
    print(f"\nSelected stocks: {selected}")
    print(f"Average pairwise correlation: {avg_corr:.4f}")
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())
    return avg_corr


def main():
    # Download all Dow Jones 30 prices
    prices = download_all_prices(DOW30, START_DATE)

    # Compute log returns
    returns = compute_returns(prices)

    # Greedy selection
    selected = greedy_selection(returns, n=N_STOCKS)

    # Evaluate
    avg_corr = evaluate_selection(returns, selected)

    print(f"\n{'='*50}")
    print(f"FINAL SELECTION ({N_STOCKS} stocks):")
    print(selected)
    print(f"Average pairwise correlation: {avg_corr:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()