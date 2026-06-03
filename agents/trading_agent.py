"""
trading_agent.py
Runs the full pipeline and opens a Pull Request on GitHub with results.
This is the agentic component required by the project specifications.

Usage:
    python agents/trading_agent.py --mode backtest
    python agents/trading_agent.py --mode live
"""

import subprocess
import argparse
import os
from pathlib import Path
from datetime import date

# ── Configuration ──────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME    = "vidovsusi-sys/volatility-trading-agent"
BASE_BRANCH  = "main"


def run_script(script_path):
    """Run a Python script and print output."""
    print(f"\n>>> Running {script_path}...")
    result = subprocess.run(
        ["python", script_path],
        capture_output=False,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script {script_path} failed.")
    print(f">>> {script_path} completed successfully.")


def run_pipeline(mode):
    """Run the full pipeline in backtest or live mode."""
    print(f"\n{'='*60}")
    print(f"TRADING AGENT — mode: {mode.upper()}")
    print(f"{'='*60}")

    # Always run these steps
    run_script("src/data_loader.py")
    run_script("src/volatility.py")
    run_script("src/nelson_siegel.py")
    run_script("src/vrp.py")
    run_script("src/forecasting.py")
    run_script("src/portfolio.py")
    run_script("src/backtest.py")

    print("\nPipeline completed successfully.")


def git_commit_results():
    """Commit updated results to GitHub."""
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
    """Open an automatic Pull Request on GitHub with results summary."""
    try:
        from github import Github, Auth
    except ImportError:
        print("PyGithub not installed. Run: pip install PyGithub")
        return

    if not GITHUB_TOKEN:
        print("WARNING: GITHUB_TOKEN not set. Skipping PR creation.")
        return

    today = date.today().isoformat()
    branch_name = f"agent-results-{today}"

    # Create branch, add a summary file, commit, push
    subprocess.run(["git", "checkout", "-b", branch_name], check=True)

    # Write summary file
    try:
        import pandas as pd
        results = pd.read_csv("data/processed/backtest_results.csv")
        best = results.loc[results["sharpe"].idxmax()]

        summary = f"# Pipeline Results — {today}\n\n"
        summary += "## Performance Summary\n\n"
        summary += "| Strategy | Sharpe | MaxDD | CAGR |\n"
        summary += "|----------|--------|-------|------|\n"
        for _, row in results.iterrows():
            summary += f"| {row['strategy']} | {row['sharpe']:.3f} | {row['max_drawdown']:.2%} | {row['cagr']:.2%} |\n"
        summary += f"\n**Best Strategy:** {best['strategy']} — Sharpe {best['sharpe']:.3f}\n"
        summary += f"\n*Generated automatically by the Trading Agent.*\n"

        with open("outputs/pipeline_summary.md", "w") as f:
            f.write(summary)

        body = summary

    except Exception as e:
        body = f"Automated pipeline run completed on {today}.\n\nError reading results: {e}"
        with open("outputs/pipeline_summary.md", "w") as f:
            f.write(body)

    # Commit summary file on new branch
    subprocess.run(["git", "add", "outputs/pipeline_summary.md"], check=True)
    subprocess.run(["git", "commit", "-m",
                    f"[trading-agent] add pipeline summary {today}"], check=True)
    subprocess.run(["git", "push", "origin", branch_name], check=True)

    # Back to main
    subprocess.run(["git", "checkout", BASE_BRANCH], check=True)

    # Create PR
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


def main():
    parser = argparse.ArgumentParser(description="Trading Agent")
    parser.add_argument("--mode", choices=["backtest", "live"],
                        default="backtest",
                        help="Pipeline mode: backtest or live")
    args = parser.parse_args()

    # Run pipeline
    run_pipeline(args.mode)

    # Commit results
    git_commit_results()

    # Open PR
    open_pull_request()


if __name__ == "__main__":
    main()