import pathlib

import pandas as pd

from data_loader import fetch_tsla_prices, get_tsla_q1_2025_window
from models import fit_garch, fit_markov_regime_switching
from risk import backtest_var


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_tsla_q1_2025_pipeline() -> None:
    """
    End-to-end pipeline for:
    - Fetching TSLA data
    - Fitting GARCH and regime-switching models
    - Computing VaR and backtesting in Q1 2025

    Results are saved into the outputs/ directory for use in the dashboard.
    """
    # 1. Fetch and cache data
    prices = fetch_tsla_prices()

    # 2. Split into estimation vs Q1 2025 evaluation
    estimation, q1 = get_tsla_q1_2025_window()

    # 3. Fit baseline GARCH on estimation window
    garch_res = fit_garch(estimation["ret"])

    # For simplicity, align conditional volatility with Q1 dates
    # by refitting including Q1 and then slicing (you may refine this
    # logic inside your notebook).
    garch_all = fit_garch(prices["ret"])
    garch_sigma = garch_all.volatility.reindex(q1["date"])
    garch_sigma.name = "garch_sigma"

    # 4. Fit regime-switching model on estimation window
    regime_res = fit_markov_regime_switching(estimation["ret"], n_states=2)

    # Extend regime labels to full sample by refitting on full sample
    regime_all = fit_markov_regime_switching(prices["ret"], n_states=2)
    regime_states_q1 = regime_all.states.reindex(q1["date"])
    regime_probs_q1 = regime_all.state_probs.reindex(q1["date"])

    # 5. Backtest VaR for GARCH-based volatility in Q1 2025
    q1_returns = q1.set_index("date")["ret"]
    garch_bt = backtest_var(q1_returns, garch_sigma)

    # 6. Save combined results for the dashboard
    result_df = pd.DataFrame(
        {
            "ret": q1_returns,
            "garch_sigma": garch_sigma,
            "state": regime_states_q1,
        }
    )
    if regime_probs_q1 is not None:
        result_df = result_df.join(regime_probs_q1)

    result_df.to_csv(OUTPUT_DIR / "tsla_q1_2025_results.csv", index_label="date")

    # 7. Save backtest summary as a small CSV for convenience
    bt_summary = pd.DataFrame(
        [
            {
                "alpha": garch_bt.alpha,
                "n_obs": garch_bt.n_obs,
                "n_violations": garch_bt.n_violations,
                "kupiec_lr": garch_bt.kupiec_lr,
                "kupiec_pvalue": garch_bt.kupiec_pvalue,
                "christoffersen_lr": garch_bt.christoffersen_lr,
                "christoffersen_pvalue": garch_bt.christoffersen_pvalue,
            }
        ]
    )
    bt_summary.to_csv(OUTPUT_DIR / "tsla_q1_2025_backtest_summary.csv", index=False)


if __name__ == "__main__":
    run_tsla_q1_2025_pipeline()

