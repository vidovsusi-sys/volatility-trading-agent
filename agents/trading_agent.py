"""
trading_agent.py

The Trading Agent is the agentic component of the Volatility Trading Agent project.
It orchestrates the full pipeline and opens an automatic Pull Request on GitHub.

This satisfies the agentic project requirement:
  - AGENTS.md defines the rules for AI agents
  - This script provides at least one automated Pull Request (required by the prof)

Usage:
    python agents/trading_agent.py --mode backtest
    python agents/trading_agent.py --mode live

Modes:
    backtest: runs the full historical pipeline from 2014 to today
    live:     same as backtest — downloads latest data and updates all results

Pipeline steps (in order):
    1. data_loader.py    — download prices for 10 stocks, VIX, S&P500
    2. volatility.py     — compute realized volatility on 6 horizons
    3. nelson_siegel.py  — fit Nelson-Siegel, extract B0, B1, B2
    4. vrp.py            — compute Variance Risk Premium
    5. forecasting.py    — XGBoost and ARMA walk-forward forecasting
    6. portfolio.py      — compute weights (Method A, B, C)
    7. backtest.py       — backtest 6 strategies, compute metrics

After the pipeline:
    - Results are committed to GitHub (main branch)
    - A Pull Request is opened automatically with a performance summary
"""

import subprocess
import argparse
import os
from pathlib import Path
from datetime import date

# ── Configuration ──────────────────────────────────────────────────────────
# GitHub token is read from environment variable for security.
# Set it with: export GITHUB_TOKEN=your_token
# Never hardcode the token in the source code.
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME    = "vidovsusi-sys/volatility-trading-agent"
BASE_BRANCH  = "main"


# ── Functions ──────────────────────────────────────────────────────────────
def run_script(script_path):
    """
    Execute a Python script as a subprocess and stream its output.

    If the script exits with a non-zero return code (i.e. fails),
    raises RuntimeError to stop the pipeline immediately.
    This prevents downstream scripts from running on corrupted data.

    Args:
        script_path: path to the Python script (e.g. "src/data_loader.py")
    """
    print(f"\n>>> Running {script_path}...")
    result = subprocess.run(
        ["python", script_path],
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script {script_path} failed with return code {result.returncode}.")
    print(f">>> {script_path} completed successfully.")


def run_pipeline(mode):
    """
    Run the full pipeline in sequence.

    Each script must complete successfully before the next one starts.
    The order is critical — each script depends on the output of the previous:
      data_loader → volatility → nelson_siegel → vrp → forecasting → portfolio → backtest

    Args:
        mode: "backtest" or "live" — currently both run the same pipeline.
              Future extension: live mode could skip historical recomputation
              and only update the most recent days.
    """
    print(f"\n{'='*60}")
    print(f"TRADING AGENT — mode: {mode.upper()}")
    print(f"{'='*60}")

    # Execute each pipeline script in order
    run_script("src/data_loader.py")    # Step 1: download market data
    run_script("src/volatility.py")     # Step 2: realized volatility
    run_script("src/nelson_siegel.py")  # Step 3: Nelson-Siegel fitting
    run_script("src/vrp.py")            # Step 4: Variance Risk Premium
    run_script("src/forecasting.py")    # Step 5: XGBoost + ARMA forecasting
    run_script("src/portfolio.py")      # Step 6: portfolio weights
    run_script("src/backtest.py")       # Step 7: backtesting + metrics

    print("\nPipeline completed successfully.")


def git_commit_results():
    """
    Stage and commit all updated pipeline outputs to the main branch.

    Commits:
      - data/processed/ : all computed CSVs (betas, predictions, weights, metrics)
      - outputs/        : figures and summary files
      - data/raw/       : updated market prices

    The commit message includes today's date for traceability.
    """
    today = date.today().isoformat()

    subprocess.run(["git", "add",
                    "data/processed/",
                    "outputs/",
                    "data/raw/"], check=True)

    subprocess.run(["git", "commit", "-m",
                    f"[trading-agent] pipeline results {today}"],
                   check=True)

    subprocess.run(["git", "push", "origin", BASE_BRANCH], check=True)
    print(f"\nResults committed to GitHub: {today}")


def open_pull_request():
    """
    Create a new branch, commit a results summary, and open a Pull Request.

    This is the automated PR required by the project agentic specification.
    The PR includes a Markdown table with the backtesting performance metrics
    for all strategies, making results visible directly on GitHub.

    Requires GITHUB_TOKEN environment variable to be set.
    If not set, the PR step is skipped gracefully.
    """
    try:
        from github import Github, Auth
    except ImportError:
        print("PyGithub not installed. Run: pip install PyGithub")
        return

    if not GITHUB_TOKEN:
        print("WARNING: GITHUB_TOKEN not set. Skipping PR creation.")
        print("Set it with: export GITHUB_TOKEN=your_token")
        return

    today = date.today().isoformat()
    branch_name = f"agent-results-{today}"

    # Create a new branch from main
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)

    # Build the PR body from backtest results
    try:
        import pandas as pd
        results = pd.read_csv("data/processed/backtest_results.csv")
        best = results.loc[results["sharpe"].idxmax()]

        # Markdown table with performance metrics
        summary = f"# Pipeline Results — {today}\n\n"
        summary += "## Performance Summary\n\n"
        summary += "| Strategy | Sharpe | MaxDD | CAGR |\n"
        summary += "|----------|--------|-------|------|\n"
        for _, row in results.iterrows():
            summary += (f"| {row['strategy']} | {row['sharpe']:.3f} | "
                       f"{row['max_drawdown']:.2%} | {row['cagr']:.2%} |\n")
        summary += f"\n**Best Strategy:** {best['strategy']} — Sharpe {best['sharpe']:.3f}\n"
        summary += f"\n*Generated automatically by the Trading Agent.*\n"

        with open("outputs/pipeline_summary.md", "w") as f:
            f.write(summary)

        body = summary

    except Exception as e:
        body = f"Automated pipeline run completed on {today}.\n\nError reading results: {e}"
        with open("outputs/pipeline_summary.md", "w") as f:
            f.write(body)

    # Commit the summary file on the new branch
    subprocess.run(["git", "add", "outputs/pipeline_summary.md"], check=True)
    subprocess.run(["git", "commit", "-m",
                    f"[trading-agent] add pipeline summary {today}"], check=True)
    subprocess.run(["git", "push", "origin", branch_name], check=True)

    # Return to main branch
    subprocess.run(["git", "checkout", BASE_BRANCH], check=True)

    # Open the Pull Request via GitHub API
    auth = Auth.Token(GITHUB_TOKEN)
    g = Github(auth=auth)
    repo = g.get_repo(REPO_NAME)

    pr = repo.create_pull(
        title=f"[Trading Agent] Pipeline results — {today}",
        body=body,
        head=branch_name,
        base=BASE_BRANCH
    )

    print(f"\nPull Request opened: {pr.html_url}")
    return pr.html_url


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    """Entry point: parse arguments, run pipeline, commit, open PR."""
    parser = argparse.ArgumentParser(
        description="Trading Agent — runs the full volatility forecasting pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "live"],
        default="backtest",
        help="Pipeline mode: 'backtest' runs full history, 'live' updates with latest data"
    )
    args = parser.parse_args()

    # Step 1: Run the full pipeline
    run_pipeline(args.mode)

    # Step 2: Commit results to GitHub
    git_commit_results()

    # Step 3: Open automatic Pull Request with results summary
    open_pull_request()


if __name__ == "__main__":
    main()