"""
plots.py
Generate publication-quality figures for the backtest results.

Reads pre-computed CSVs from data/processed/ and saves PNG figures to
outputs/figures/. Designed for inclusion in the academic LaTeX document.

Figures produced:
  1. equity_curves.png       — all 5 strategies over the full backtest
  2. drawdowns.png           — drawdown curves for all 5 strategies
  3. stress_covid.png        — zoom on the COVID 2020 crash
  4. stress_bear.png         — zoom on the 2022 bear market

Style: academic black-and-white-friendly, sober palette, no chartjunk.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────
PROCESSED_PATH = Path("data/processed")
FIGURES_PATH   = Path("outputs/figures")

# Stress periods (must match backtest.py)
STRESS_PERIODS = {
    "COVID_crash_2020":   ("2020-03-01", "2020-04-30"),
    "Bear_market_2022":   ("2022-09-01", "2022-12-31"),
}

# Color palette: sober, colorblind-friendly, distinct in grayscale.
# Each strategy keeps the same color across all plots for consistency.
STRATEGY_COLORS = {
    "Equal_Weighted":         "#000000",   # black — the benchmark
    "Historical_Risk_Parity": "#7F7F7F",   # gray
    "Method_A_XGBoost":       "#1F77B4",   # blue (the "good" ML strategy)
    "Method_B_XGBoost":       "#D62728",   # red
    "Method_C_XGBoost":       "#FF7F0E",   # orange
}

# Line styles per strategy: makes the figure readable when printed in B&W
STRATEGY_STYLES = {
    "Equal_Weighted":         "-",     # solid, the most important reference
    "Historical_Risk_Parity": "--",    # dashed
    "Method_A_XGBoost":       "-",     # solid (key strategy)
    "Method_B_XGBoost":       ":",     # dotted
    "Method_C_XGBoost":       "-.",    # dash-dot
}


def setup_matplotlib_style():
    """
    Apply an academic-style matplotlib configuration:
      - white background, no gridlines noise
      - serif font (LaTeX-friendly)
      - large enough fonts to be readable in a PDF
      - thin axis spines, clean look
    """
    plt.rcParams.update({
        # Figure & layout
        "figure.figsize":   (8, 4.5),
        "figure.dpi":       100,
        "savefig.dpi":      300,        # high-res when saved
        "savefig.bbox":     "tight",
        "figure.facecolor": "white",
        "axes.facecolor":   "white",

        # Fonts
        "font.family":      "serif",
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "legend.fontsize":  10,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,

        # Axes / spines
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,

        # Grid: subtle, only horizontal
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linewidth":    0.5,

        # Lines
        "lines.linewidth":   1.5,
    })
# ── Step 2: Equity curves ──────────────────────────────────────────────────
def plot_equity_curves(equity_df, out_path):
    """
    Plot all strategies' equity curves on a single figure.

    Each strategy uses a consistent color and line style across all plots,
    so the reader can identify it at a glance in any figure.

    Args:
        equity_df: DataFrame (Date x strategy_names) with daily equity values.
        out_path:  Path object pointing to the directory where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for strat in equity_df.columns:
        ax.plot(
            equity_df.index,
            equity_df[strat],
            label=strat.replace("_", " "),
            color=STRATEGY_COLORS.get(strat, "black"),
            linestyle=STRATEGY_STYLES.get(strat, "-"),
            linewidth=1.5,
        )

    # Horizontal reference line at 1.0 (initial capital)
    ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="-")

    ax.set_title("Portfolio Equity Curves (2018-01-02 to 2026-05-22)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (initial capital = 1.0)")
    ax.legend(loc="upper left", frameon=False)

    fp = out_path / "equity_curves.png"
    plt.savefig(fp)
    plt.close(fig)
    print(f"Saved -> {fp}")
    # ── Step 3: Drawdowns ──────────────────────────────────────────────────────
def plot_drawdowns(equity_df, out_path):
    """
    Plot drawdown curves for all strategies.

    Drawdown(t) = (equity(t) - running_max(t)) / running_max(t)

    It is always <= 0; large negative values indicate severe peak-to-trough
    losses. The figure is shaded below zero (fill_between) to emphasize the
    "underwater" nature of the curve — a standard convention in finance.

    Args:
        equity_df: DataFrame (Date x strategy_names) with daily equity values.
        out_path:  Path where to save the PNG.
    """
    # Compute drawdowns for every column
    running_max = equity_df.cummax()
    drawdowns   = (equity_df - running_max) / running_max

    fig, ax = plt.subplots(figsize=(9, 5))

    for strat in drawdowns.columns:
        ax.plot(
            drawdowns.index,
            drawdowns[strat] * 100,    # show as percentage
            label=strat.replace("_", " "),
            color=STRATEGY_COLORS.get(strat, "black"),
            linestyle=STRATEGY_STYLES.get(strat, "-"),
            linewidth=1.2,
        )

    # Zero reference line (the "high water mark")
    ax.axhline(y=0.0, color="black", linewidth=0.5, linestyle="-")

    ax.set_title("Strategy Drawdowns (2018-01-02 to 2026-05-22)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=False)

    fp = out_path / "drawdowns.png"
    plt.savefig(fp)
    plt.close(fig)
    print(f"Saved -> {fp}")
# ── Step 4: Stress period zooms ────────────────────────────────────────────
def plot_stress_period(equity_df, start, end, title, filename, out_path):
    """
    Zoom on a stress period, showing how each strategy behaved.

    Equity is rebased to 1.0 at the start of the window so that all curves
    share a common starting point — this makes relative performance during
    the crisis directly comparable.

    Args:
        equity_df: DataFrame (Date x strategies) with daily equity values.
        start, end: window bounds (strings or Timestamps).
        title:     plot title (string).
        filename:  output PNG filename.
        out_path:  output directory (Path).
    """
    # Filter the window
    window = equity_df.loc[start:end].copy()
    if window.empty:
        print(f"  [WARNING] No data in {start} -> {end}, skipping.")
        return

    # Rebase to 1.0 at the start of the window for fair visual comparison
    window = window.div(window.iloc[0], axis=1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for strat in window.columns:
        ax.plot(
            window.index,
            window[strat],
            label=strat.replace("_", " "),
            color=STRATEGY_COLORS.get(strat, "black"),
            linestyle=STRATEGY_STYLES.get(strat, "-"),
            linewidth=1.6,
        )

    # Reference line at 1.0 (start-of-period level)
    ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="-")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (rebased to 1.0 at start)")
    ax.legend(loc="best", frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fp = out_path / filename
    plt.savefig(fp)
    plt.close(fig)
    print(f"Saved -> {fp}")


def plot_all_stress_periods(equity_df, out_path):
    """
    Generate one zoom plot per stress period defined in STRESS_PERIODS.
    """
    titles = {
        "COVID_crash_2020":  "COVID-19 Crash (Mar - Apr 2020)",
        "Bear_market_2022":  "Bear Market (Sep - Dec 2022)",
    }
    filenames = {
        "COVID_crash_2020":  "stress_covid.png",
        "Bear_market_2022":  "stress_bear.png",
    }

    for period_name, (start, end) in STRESS_PERIODS.items():
        plot_stress_period(
            equity_df, start, end,
            title=titles.get(period_name, period_name),
            filename=filenames.get(period_name, f"{period_name}.png"),
            out_path=out_path,
        )
# ── Main orchestration ─────────────────────────────────────────────────────
def main():
    """Entry point: generate all backtest figures."""
    print(f"\n{'='*60}")
    print(f"PLOTS — generating backtest figures")
    print(f"{'='*60}\n")

    # Ensure output directory exists
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {FIGURES_PATH.resolve()}")

    # Apply academic plotting style
    setup_matplotlib_style()
    print("Matplotlib style: academic (serif, white bg, minimal grid)")

    # ── Step 2: Equity curves ──────────────────────────────────────────────
    print("\n--- Step 2: Equity curves ---")
    equity_df = pd.read_csv(PROCESSED_PATH / "equity_curves.csv",
                            index_col=0, parse_dates=True)
    print(f"Loaded equity curves — shape: {equity_df.shape}")
    plot_equity_curves(equity_df, FIGURES_PATH)
    # ── Step 3: Drawdowns ──────────────────────────────────────────────────
    print("\n--- Step 3: Drawdowns ---")
    plot_drawdowns(equity_df, FIGURES_PATH)
    # ── Step 4: Stress period zooms ────────────────────────────────────────
    print("\n--- Step 4: Stress period zoom plots ---")
    plot_all_stress_periods(equity_df, FIGURES_PATH)

    print("\n[Step 4] All plots saved. Visualization complete.")


if __name__ == "__main__":
    main()
