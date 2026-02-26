import pathlib
from typing import Dict, List, Tuple

import pandas as pd

from data_loader import (
    ASSET_UNIVERSE,
    fetch_multi_asset_prices,
    fetch_tsla_prices,
    get_tsla_q1_2025_window,
    get_universe_q1_2025_window,
)
from macro_sentiment import (
    fetch_macro_data,
    fetch_sentiment_data,
)
from models import fit_garch, fit_markov_regime_switching, fit_ms_garch_like, forecast_garch_vol
from risk import backtest_var


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def rolling_vol_forecasts(
    returns: pd.Series,
    test_start: str = "2025-01-01",
    test_end: str = "2025-03-31",
) -> Tuple[pd.Series, pd.Series]:
    """
    Rolling 1-step-ahead volatility forecasts for GARCH and MS-GARCH-like models.

    For each date t in [test_start, test_end], the models are estimated using
    data up to t-1 only (expanding window). The returned series are indexed
    by the test dates.
    """
    r = returns.dropna().sort_index()
    test = r.loc[test_start:test_end]
    if test.empty:
        raise ValueError("No test data available in the specified window.")

    garch_sigmas = []
    ms_sigmas = []
    test_index = []

    for t in test.index:
        train_slice = r.loc[: t - pd.Timedelta(days=1)]
        if len(train_slice) < 50:
            # Skip very early dates with insufficient history
            continue

        # GARCH forecast
        garch_res = fit_garch(train_slice)
        sigma_g = forecast_garch_vol(garch_res, horizon=1)

        # MS-GARCH-like "forecast": use latest conditional MS volatility
        ms_res = fit_ms_garch_like(train_slice)
        sigma_ms = float(ms_res.ms_volatility.iloc[-1])

        garch_sigmas.append(sigma_g)
        ms_sigmas.append(sigma_ms)
        test_index.append(t)

    garch_series = pd.Series(garch_sigmas, index=test_index, name="garch_sigma")
    ms_series = pd.Series(ms_sigmas, index=test_index, name="ms_garch_sigma")
    return garch_series, ms_series


def run_tsla_q1_2025_pipeline() -> None:
    """
    Legacy single-asset TSLA pipeline kept for backward compatibility.
    """
    prices = fetch_tsla_prices()
    _, q1 = get_tsla_q1_2025_window()

    # Use date index for returns so that model outputs align with calendar dates
    returns_full = prices.set_index("date")["ret"]

    # Rolling out-of-sample forecasts for Q1 2025
    garch_sigma, ms_sigma = rolling_vol_forecasts(returns_full)

    # Regime states and probabilities from full-sample MS fit (for interpretation)
    ms_res_full = fit_ms_garch_like(returns_full, n_states=2)
    regime_states_q1 = ms_res_full.regime.states.reindex(garch_sigma.index)
    regime_probs_q1 = ms_res_full.regime.state_probs.reindex(garch_sigma.index)

    q1_returns = returns_full.reindex(garch_sigma.index)
    garch_bt = backtest_var(q1_returns, garch_sigma)
    ms_bt = backtest_var(q1_returns, ms_sigma)

    result_df = pd.DataFrame(
        {
            "ret": q1_returns,
            "garch_sigma": garch_sigma,
            "ms_garch_sigma": ms_sigma,
            "state": regime_states_q1,
        }
    )
    if regime_probs_q1 is not None:
        result_df = result_df.join(regime_probs_q1)

    result_df.to_csv(OUTPUT_DIR / "tsla_q1_2025_results.csv", index_label="date")

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
                "model": "GARCH",
            },
            {
                "alpha": ms_bt.alpha,
                "n_obs": ms_bt.n_obs,
                "n_violations": ms_bt.n_violations,
                "kupiec_lr": ms_bt.kupiec_lr,
                "kupiec_pvalue": ms_bt.kupiec_pvalue,
                "christoffersen_lr": ms_bt.christoffersen_lr,
                "christoffersen_pvalue": ms_bt.christoffersen_pvalue,
                "model": "MS-GARCH-like",
            },
        ]
    )
    bt_summary.to_csv(OUTPUT_DIR / "tsla_q1_2025_backtest_summary.csv", index=False)


def run_multi_asset_q1_2025_pipeline(
    tickers: List[str] = None,
) -> None:
    """
    Multi-asset pipeline:
    - Fetches prices for a high-volatility universe (TSLA, NVDA, BTC, AI ETF, S&P 500).
    - Fits GARCH and regime-switching models per asset.
    - Computes VaR and backtesting in Q1 2025.

    Outputs:
    - outputs/multi_asset_q1_2025_results.csv
    - outputs/multi_asset_q1_2025_backtest_summary.csv
    """
    if tickers is None:
        tickers = ASSET_UNIVERSE

    # 1. Fetch and cache multi-asset prices (full history 2020–2025)
    prices = fetch_multi_asset_prices(tickers=tickers)

    results: List[pd.DataFrame] = []
    bt_rows: List[Dict] = []

    for ticker in tickers:
        all_t = prices[prices["ticker"] == ticker].copy()
        if all_t.empty:
            continue

        # Full return history with date index
        returns_full = all_t.set_index("date")["ret"]

        # Rolling out-of-sample volatility forecasts for Q1 2025
        garch_sigma_q1, ms_sigma_q1 = rolling_vol_forecasts(returns_full)

        # Regime states/probabilities from full-sample MS fit for interpretation
        ms_res_full = fit_ms_garch_like(returns_full, n_states=2)
        regime_states_q1 = ms_res_full.regime.states.reindex(garch_sigma_q1.index)
        regime_probs_q1 = ms_res_full.regime.state_probs.reindex(garch_sigma_q1.index)

        q1_returns = returns_full.reindex(garch_sigma_q1.index)
        garch_bt = backtest_var(q1_returns, garch_sigma_q1)
        ms_bt = backtest_var(q1_returns, ms_sigma_q1)

        df_t = pd.DataFrame(
            {
                "ticker": ticker,
                "ret": q1_returns,
                "garch_sigma": garch_sigma_q1,
                "ms_garch_sigma": ms_sigma_q1,
                "state": regime_states_q1,
            }
        )
        if regime_probs_q1 is not None:
            df_t = df_t.join(regime_probs_q1)

        results.append(df_t)

        bt_rows.extend(
            [
                {
                    "ticker": ticker,
                    "model": "GARCH",
                    "alpha": garch_bt.alpha,
                    "n_obs": garch_bt.n_obs,
                    "n_violations": garch_bt.n_violations,
                    "kupiec_lr": garch_bt.kupiec_lr,
                    "kupiec_pvalue": garch_bt.kupiec_pvalue,
                    "christoffersen_lr": garch_bt.christoffersen_lr,
                    "christoffersen_pvalue": garch_bt.christoffersen_pvalue,
                },
                {
                    "ticker": ticker,
                    "model": "MS-GARCH-like",
                    "alpha": ms_bt.alpha,
                    "n_obs": ms_bt.n_obs,
                    "n_violations": ms_bt.n_violations,
                    "kupiec_lr": ms_bt.kupiec_lr,
                    "kupiec_pvalue": ms_bt.kupiec_pvalue,
                    "christoffersen_lr": ms_bt.christoffersen_lr,
                    "christoffersen_pvalue": ms_bt.christoffersen_pvalue,
                },
            ]
        )

    if results:
        all_results = pd.concat(results)

        # Attach macro and sentiment context
        macro = fetch_macro_data()
        sentiment = fetch_sentiment_data(tickers)

        merged = (
            all_results.reset_index()
            .merge(macro, on="date", how="left")
            .merge(sentiment, on=["date", "ticker"], how="left")
            .set_index("date")
            .sort_index()
        )

        all_results = merged
        all_results.to_csv(
            OUTPUT_DIR / "multi_asset_q1_2025_results.csv",
            index_label="date",
        )

    if bt_rows:
        bt_summary = pd.DataFrame(bt_rows)
        bt_summary.to_csv(
            OUTPUT_DIR / "multi_asset_q1_2025_backtest_summary.csv", index=False
        )


if __name__ == "__main__":
    # Run both the legacy TSLA-only pipeline and the multi-asset pipeline.
    run_tsla_q1_2025_pipeline()
    run_multi_asset_q1_2025_pipeline()

