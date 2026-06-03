"""
app.py
Streamlit dashboard for Volatility Trading Agent.
Shows backtesting results, equity curves, portfolio weights, and AI insights.
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

STRATEGY_COLORS = {
    "Equal_Weighted":     "#636EFA",
    "Historical_Risk_Parity": "#EF553B",
    "Method_A_XGBoost":   "#00CC96",
    "Method_B_XGBoost":   "#AB63FA",
    "Method_C_XGBoost":   "#FFA15A",
    "Method_A_ARMA":      "#19D3F3",
}

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Volatility Trading Agent",
    page_icon="📈",
    layout="wide"
)

# ── Load data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    return pd.read_csv(PROCESSED_PATH / "backtest_results.csv")

@st.cache_data
def load_equity_curves():
    return pd.read_csv(PROCESSED_PATH / "equity_curves.csv", index_col=0, parse_dates=True)

@st.cache_data
def load_weights(method):
    filename = f"weights_{method}.csv"
    return pd.read_csv(PROCESSED_PATH / filename, index_col=0, parse_dates=True)

@st.cache_data
def load_betas():
    return pd.read_csv(PROCESSED_PATH / "betas.csv", index_col=0, parse_dates=True)

@st.cache_data
def load_stress(period):
    filename = f"stress_{period}.csv"
    return pd.read_csv(PROCESSED_PATH / filename)

# ── AI Insights ────────────────────────────────────────────────────────────
def generate_ai_insights(metrics_df):
    """Generate AI insights using Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "Set ANTHROPIC_API_KEY environment variable to enable AI insights."

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

Results:
{metrics_str}

Provide:
1. Key findings — which strategy performed best and why
2. Risk analysis — drawdown and volatility management
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
    st.title("📈 Volatility Trading Agent")
    st.markdown("**Volatility Term Structure Forecasting for Dynamic Portfolio Optimization**")
    st.markdown("*Programming in Finance II — USI 2026 — Luca Anselmi · Stefan Vidovic · Arnel Hodza*")
    st.divider()

    # Load data
    try:
        metrics = load_metrics()
        equity_curves = load_equity_curves()
    except FileNotFoundError:
        st.error("Run the pipeline first: `python agents/trading_agent.py --mode backtest`")
        return

    # ── Tab layout ──────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance", "📈 Equity Curves", "⚖️ Portfolio Weights",
        "🔥 Stress Periods", "🤖 AI Insights"
    ])

    # ── Tab 1: Performance metrics ──────────────────────────────────────────
    with tab1:
        st.header("Performance Metrics")
        st.markdown("Backtesting period: **2018-01-02 to 2026-05-22** | Transaction costs: **10 bps**")

        # Metrics table
        display_metrics = metrics.copy()
        display_metrics["max_drawdown"] = display_metrics["max_drawdown"].apply(lambda x: f"{x:.2%}")
        display_metrics["cagr"]         = display_metrics["cagr"].apply(lambda x: f"{x:.2%}")
        display_metrics["sharpe"]       = display_metrics["sharpe"].round(3)
        display_metrics["calmar"]       = display_metrics["calmar"].round(3)
        display_metrics.columns = ["Strategy", "Sharpe", "Max DD", "CAGR", "Calmar"]

        st.dataframe(display_metrics, use_container_width=True, hide_index=True)

        # Bar charts
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(metrics, x="strategy", y="sharpe",
                        title="Sharpe Ratio by Strategy",
                        color="strategy",
                        color_discrete_map={k: v for k, v in STRATEGY_COLORS.items()})
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(metrics, x="strategy", y="max_drawdown",
                        title="Maximum Drawdown by Strategy",
                        color="strategy",
                        color_discrete_map={k: v for k, v in STRATEGY_COLORS.items()})
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Equity curves ────────────────────────────────────────────────
    with tab2:
        st.header("Equity Curves")
        st.markdown("Portfolio value over time (starting from 1.0)")

        fig = go.Figure()
        for col in equity_curves.columns:
            color = STRATEGY_COLORS.get(col, "#888888")
            fig.add_trace(go.Scatter(
                x=equity_curves.index,
                y=equity_curves[col],
                name=col,
                line=dict(color=color, width=2)
            ))

        fig.update_layout(
            title="Portfolio Equity Curves — 2018 to 2026",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Portfolio weights ────────────────────────────────────────────
    with tab3:
        st.header("Portfolio Weights")

        method_options = {
            "Method A — XGBoost": "method_a_xgboost",
            "Method B — XGBoost": "method_b_xgboost",
            "Method C — XGBoost": "method_c_xgboost",
            "Method A — ARMA":    "method_a_arma",
        }
        selected = st.selectbox("Select method", list(method_options.keys()))
        weights = load_weights(method_options[selected])

        # Current weights
        st.subheader("Current Weights (latest)")
        latest = weights.iloc[-1]
        fig = px.bar(
            x=latest.index,
            y=latest.values,
            title=f"Portfolio Weights — {selected} — {weights.index[-1].date()}",
            labels={"x": "Stock", "y": "Weight"},
            color=latest.index,
        )
        fig.add_hline(y=0.10, line_dash="dash", line_color="gray",
                     annotation_text="Equal weight (10%)")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Weights over time
        st.subheader("Weights Over Time")
        ticker = st.selectbox("Select stock", TICKERS)
        fig = px.line(weights, x=weights.index, y=ticker,
                     title=f"{ticker} weight over time — {selected}")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Stress periods ───────────────────────────────────────────────
    with tab4:
        st.header("Stress Period Analysis")

        stress_options = {
            "COVID Crash 2020": "COVID_crash_2020",
            "Bear Market 2022": "Bear_market_2022",
        }
        selected_stress = st.selectbox("Select period", list(stress_options.keys()))
        stress_data = load_stress(stress_options[selected_stress])

        st.dataframe(stress_data, use_container_width=True, hide_index=True)

        fig = px.bar(stress_data, x="strategy", y="sharpe",
                    title=f"Sharpe Ratio — {selected_stress}",
                    color="strategy")
        fig.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: AI Insights ──────────────────────────────────────────────────
    with tab5:
        st.header("🤖 AI Insights")
        st.markdown("Powered by **Claude** (Anthropic)")

        if st.button("Generate AI Analysis", type="primary"):
            with st.spinner("Generating analysis..."):
                insights = generate_ai_insights(metrics)
            st.markdown(insights)
        else:
            st.info("Click the button to generate an AI analysis of the backtesting results.")

if __name__ == "__main__":
    main()