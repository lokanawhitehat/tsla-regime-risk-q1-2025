import pathlib
from typing import Dict, List, Tuple

import numpy as np
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
from models import (
    fit_egarch,
    fit_garch,
    fit_garch_student_t,
    fit_gjr_garch,
    fit_markov_regime_switching,
    fit_ms_garch_like,
    forecast_garch_vol,
    forecast_vol_from_result,
)
from risk import (
    backtest_var,
    diebold_mariano_test,
    mean_qlike_loss,
    mean_squared_error_variance,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Multiple evaluation windows for robustness (COVID crash, 2022 tightening, Q1 2025)
EVAL_WINDOWS: List[Tuple[str, str, str]] = [
    ("COVID_crash_2020", "2020-03-01", "2020-05-31"),
    ("2022_tightening", "2022-01-01", "2022-06-30"),
    ("Q1_2025", "2025-01-01", "2025-03-31"),
]


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


def rolling_vol_forecasts_multi_model(
    returns: pd.Series,
    test_start: str,
    test_end: str,
    models: List[str] | None = None,
) -> Dict[str, pd.Series]:
    """
    Rolling 1-step-ahead volatility forecasts for GARCH, MS-GARCH-like, EGARCH, GJR-GARCH.
    Returns dict: model_name -> sigma_series (indexed by test dates).
    """
    if models is None:
        models = ["GARCH", "MS-GARCH-like", "EGARCH", "GJR-GARCH", "Student-t GARCH"]
    r = returns.dropna().sort_index()
    test = r.loc[test_start:test_end]
    if test.empty:
        return {}

    out: Dict[str, List[Tuple[pd.Timestamp, float]]] = {m: [] for m in models}
    test_index: List[pd.Timestamp] = []

    for t in test.index:
        train_slice = r.loc[: t - pd.Timedelta(days=1)]
        if len(train_slice) < 50:
            continue
        row: List[float] = []
        for m in models:
            try:
                if m == "GARCH":
                    res = fit_garch(train_slice)
                    sigma = forecast_garch_vol(res, horizon=1)
                elif m == "MS-GARCH-like":
                    res = fit_ms_garch_like(train_slice)
                    sigma = float(res.ms_volatility.iloc[-1])
                elif m == "EGARCH":
                    res = fit_egarch(train_slice)
                    sigma = forecast_vol_from_result(res, horizon=1)
                elif m == "GJR-GARCH":
                    res = fit_gjr_garch(train_slice)
                    sigma = forecast_vol_from_result(res, horizon=1)
                elif m == "Student-t GARCH":
                    res = fit_garch_student_t(train_slice)
                    sigma = forecast_vol_from_result(res, horizon=1)
                else:
                    sigma = np.nan
            except Exception:
                sigma = np.nan
            out[m].append((t, sigma))
        test_index.append(t)

    result: Dict[str, pd.Series] = {}
    for m in models:
        if out[m]:
            idx = [x[0] for x in out[m]]
            vals = [x[1] for x in out[m]]
            result[m] = pd.Series(vals, index=idx, name=m)
    return result


def evaluate_window(
    returns: pd.Series,
    window_name: str,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    """
    Run multi-model rolling forecasts, backtests, MSE/QLIKE, and DM tests vs GARCH.
    Returns one row per model with backtest and forecast accuracy metrics.
    """
    forecasts = rolling_vol_forecasts_multi_model(returns, test_start, test_end)
    if not forecasts or "GARCH" not in forecasts:
        return pd.DataFrame()

    rows = []
    garch_sigma = forecasts["GARCH"]
    test_returns = returns.reindex(garch_sigma.index).dropna()
    garch_sigma = garch_sigma.reindex(test_returns.index).dropna()
    test_returns = test_returns.reindex(garch_sigma.index).dropna()
    if test_returns.empty:
        return pd.DataFrame()

    for model_name, sigma_series in forecasts.items():
        sigma_aligned = sigma_series.reindex(test_returns.index).dropna()
        ret_aligned = test_returns.reindex(sigma_aligned.index).dropna()
        sigma_aligned = sigma_aligned.reindex(ret_aligned.index).dropna()
        if sigma_aligned.empty or ret_aligned.empty:
            rows.append({
                "window": window_name,
                "model": model_name,
                "n_obs": 0,
                "n_violations": np.nan,
                "kupiec_pvalue": np.nan,
                "christoffersen_pvalue": np.nan,
                "var_mse": np.nan,
                "var_qlike": np.nan,
                "dm_vs_garch_mse": np.nan,
                "dm_pvalue_mse": np.nan,
                "dm_vs_garch_qlike": np.nan,
                "dm_pvalue_qlike": np.nan,
            })
            continue
        bt = backtest_var(ret_aligned, sigma_aligned)
        mse = mean_squared_error_variance(ret_aligned, sigma_aligned)
        qlike = mean_qlike_loss(ret_aligned, sigma_aligned)
        common_idx = ret_aligned.index.intersection(garch_sigma.index).intersection(sigma_aligned.index)
        r_dm = ret_aligned.reindex(common_idx).dropna()
        g_dm = garch_sigma.reindex(common_idx).dropna()
        s_dm = sigma_aligned.reindex(common_idx).dropna()
        idx_final = r_dm.index.intersection(g_dm.index).intersection(s_dm.index)
        r_dm = r_dm.loc[idx_final]
        g_dm = g_dm.loc[idx_final]
        s_dm = s_dm.loc[idx_final]
        if len(r_dm) > 1:
            dm_mse, p_mse = diebold_mariano_test(r_dm, g_dm, s_dm, loss="mse")
            dm_qlike, p_qlike = diebold_mariano_test(r_dm, g_dm, s_dm, loss="qlike")
        else:
            dm_mse = dm_qlike = p_mse = p_qlike = np.nan
        rows.append({
            "window": window_name,
            "model": model_name,
            "n_obs": bt.n_obs,
            "n_violations": bt.n_violations,
            "kupiec_pvalue": bt.kupiec_pvalue,
            "christoffersen_pvalue": bt.christoffersen_pvalue,
            "var_mse": mse,
            "var_qlike": qlike,
            "dm_vs_garch_mse": dm_mse,
            "dm_pvalue_mse": p_mse,
            "dm_vs_garch_qlike": dm_qlike,
            "dm_pvalue_qlike": p_qlike,
        })
    return pd.DataFrame(rows)


def run_multi_window_evaluation(asset: str = "TSLA") -> pd.DataFrame:
    """
    Run backtests and forecast accuracy (MSE, QLIKE, DM) over COVID, 2022, and Q1 2025.
    Saves results to outputs/multi_window_evaluation.csv and returns the DataFrame.
    """
    if asset == "TSLA":
        prices = fetch_tsla_prices()
    else:
        prices = fetch_multi_asset_prices()
        if asset not in prices.columns.get_level_values(0):
            prices = fetch_tsla_prices()
            asset = "TSLA"
    if hasattr(prices, "columns") and isinstance(prices.columns, pd.MultiIndex):
        ret = prices[asset]["ret"].droplevel(0) if asset in prices.columns.get_level_values(0) else prices.iloc[:, 0]
    else:
        ret = prices.set_index("date")["ret"]
    if hasattr(ret, "index") and ret.index.name != "date":
        ret.index = pd.to_datetime(ret.index)
    ret = ret.rename(None)
    all_rows = []
    for wname, t_start, t_end in EVAL_WINDOWS:
        df = evaluate_window(ret, wname, t_start, t_end)
        if not df.empty:
            df["asset"] = asset
            all_rows.append(df)
    if not all_rows:
        return pd.DataFrame()
    out_df = pd.concat(all_rows, ignore_index=True)
    out_path = OUTPUT_DIR / "multi_window_evaluation.csv"
    out_df.to_csv(out_path, index=False)
    return out_df


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
        # Ensure index has name "date" so reset_index() yields a "date" column for merge
        all_results.index.name = "date"

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
    # Multi-window evaluation (COVID, 2022, Q1 2025) with EGARCH/GJR/Student-t and DM/QLIKE/MSE
    run_multi_window_evaluation(asset="TSLA")

