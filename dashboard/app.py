"""
app.py

Streamlit dashboard for the Volatility Trading Agent project.
Provides an interactive interface to explore backtesting results,
portfolio weights, stress period analysis, and AI-generated insights.

Five tabs:
  1. Performance    — metrics table + Sharpe and MaxDD bar charts
  2. Equity Curves  — interactive portfolio value over time
  3. Portfolio Weights — current and historical weights per method
  4. Stress Periods — COVID 2020 and Bear market 2022 analysis
  5. AI Insights    — Claude API analysis of backtesting results

Run with:
    streamlit run dashboard/app.py

Requirements:
    pip install streamlit plotly anthropic

For AI Insights tab, set:
    export ANTHROPIC_API_KEY=your_key
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import anthropic
import os

# ── Configuration ──────────────────────────────────────────────────────────
PROCESSED_PATH = Path("data/processed")
OUTPUTS_PATH   = Path("outputs")

TICKERS = ["CRM", "VZ", "WMT", "INTC", "UNH", "MRK", "BA", "NKE", "CVX", "AMGN"]

# Consistent colors for each strategy across all charts
STRATEGY_COLORS = {
    "Equal_Weighted":         "#636EFA",
    "Historical_Risk_Parity": "#EF553B",
    "Method_A_XGBoost":       "#00CC96",
    "Method_B_XGBoost":       "#AB63FA",
    "Method_C_XGBoost":       "#FFA15A",
    "Method_A_ARMA":          "#19D3F3",
}

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Volatility Trading Agent",
    page_icon="📈",
    layout="wide"
)

# ── Data loaders ───────────────────────────────────────────────────────────
# @st.cache_data caches the result in memory after the first load.
# This avoids re-reading CSV files from disk on every user interaction.

@st.cache_data
def load_metrics():
    """Load backtesting performance metrics for all strategies."""
    return pd.read_csv(PROCESSED_PATH / "backtest_results.csv")

@st.cache_data
def load_equity_curves():
    """Load daily equity curves for all strategies."""
    return pd.read_csv(
        PROCESSED_PATH / "equity_curves.csv",
        index_col=0, parse_dates=True
    )

@st.cache_data
def load_weights(method):
    """
    Load portfolio weights for a given method.

    Args:
        method: filename suffix (e.g. "method_a_xgboost")
    """
    filename = f"weights_{method}.csv"
    return pd.read_csv(
        PROCESSED_PATH / filename,
        index_col=0, parse_dates=True
    )

@st.cache_data
def load_betas():
    """Load Nelson-Siegel betas for all stocks."""
    return pd.read_csv(
        PROCESSED_PATH / "betas.csv",
        index_col=0, parse_dates=True
    )

@st.cache_data
def load_stress(period):
    """
    Load stress period metrics.

    Args:
        period: period name (e.g. "COVID_crash_2020")
    """
    filename = f"stress_{period}.csv"
    return pd.read_csv(PROCESSED_PATH / filename)


# ── AI Insights ────────────────────────────────────────────────────────────
def generate_ai_insights(metrics_df):
    """
    Generate AI analysis of backtesting results using Claude API.

    Sends a structured prompt to Claude describing the system architecture
    and the backtesting results. Claude returns a 3-paragraph analysis
    covering key findings, risk analysis, and economic interpretation.

    Requires ANTHROPIC_API_KEY environment variable.
    Returns an error message string if the API key is not set or the
    call fails — does not raise exceptions.

    Args:
        metrics_df: DataFrame with backtesting metrics per strategy

    Returns:
        String with the AI-generated analysis
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return ("⚠️ Set the ANTHROPIC_API_KEY environment variable to enable AI insights.\n\n"
                "Run: `export ANTHROPIC_API_KEY=your_key`")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        metrics_str = metrics_df.to_string()

        prompt = f"""You are a quantitative finance analyst. Analyze these backtesting results 
from a volatility forecasting trading system and provide a concise 3-paragraph analysis.

The system forecasts volatility term structures using Nelson-Siegel decomposition and XGBoost/ARMA 
walk-forward models. Portfolio weights are computed using three methods:
- Method A: Risk Parity on predicted B0 (volatility level)
- Method B: Shape Trading using full term structure (B0, B1, B2)
- Method C: Momentum combined with predicted volatility

Portfolio weights are smoothed with a 5-day moving average to reduce transaction costs.
Transaction costs: 10 basis points per unit of daily turnover.

Results:
{metrics_str}

Provide:
1. Key findings — which strategy performed best and why
2. Risk analysis — drawdown and volatility management across strategies
3. Economic interpretation — what the results tell us about volatility forecasting

Be specific and reference the actual numbers. Keep it concise."""

        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except Exception as e:
        return f"Error generating insights: {e}"


# ── Main dashboard ─────────────────────────────────────────────────────────
def main():
    """Main Streamlit app — renders the full dashboard."""

    # Header
    st.title("📈 Volatility Trading Agent")
    st.markdown("**Volatility Term Structure Forecasting for Dynamic Portfolio Optimization**")
    st.markdown(
        "*Programming in Finance II — USI 2026 — "
        "Luca Anselmi · Stefan Vidovic · Arnel Hodza*"
    )
    st.divider()

    # Load core data — show error if pipeline has not been run yet
    try:
        metrics       = load_metrics()
        equity_curves = load_equity_curves()
    except FileNotFoundError:
        st.error(
            "Pipeline output not found. "
            "Run: `python agents/trading_agent.py --mode backtest`"
        )
        return

    # Derive backtest period dynamically from data
    backtest_start = equity_curves.index.min().date()
    backtest_end   = equity_curves.index.max().date()

    #