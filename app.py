import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@st.cache_data
def load_results() -> pd.DataFrame:
    """
    Prefer multi-asset results; fall back to TSLA-only if needed.
    """
    multi_path = OUTPUT_DIR / "multi_asset_q1_2025_results.csv"
    tsla_path = OUTPUT_DIR / "tsla_q1_2025_results.csv"

    if multi_path.exists():
        df = pd.read_csv(multi_path, parse_dates=["date"])
    elif tsla_path.exists():
        df = pd.read_csv(tsla_path, parse_dates=["date"])
        df["ticker"] = "TSLA"
    else:
        raise FileNotFoundError(
            "No results file found. Run `python pipeline.py` first to generate results."
        )

    df = df.set_index("date").sort_index()
    return df


@st.cache_data
def load_backtest_summary() -> pd.DataFrame:
    multi_path = OUTPUT_DIR / "multi_asset_q1_2025_backtest_summary.csv"
    tsla_path = OUTPUT_DIR / "tsla_q1_2025_backtest_summary.csv"
    if multi_path.exists():
        return pd.read_csv(multi_path)
    if tsla_path.exists():
        bt = pd.read_csv(tsla_path)
        bt["ticker"] = "TSLA"
        bt["model"] = "GARCH"
        return bt
    return pd.DataFrame()


def compute_regime_duration(states: pd.Series, selected_idx: int) -> int:
    """
    Number of consecutive days the current regime has been active up to selected_idx.
    """
    if states.isna().all():
        return 0
    current_state = states.iloc[selected_idx]
    if np.isnan(current_state):
        return 0

    duration = 1
    for i in range(selected_idx - 1, -1, -1):
        if states.iloc[i] == current_state:
            duration += 1
        else:
            break
    return duration


def compute_transition_matrix(states: pd.Series) -> pd.DataFrame:
    """
    Empirical 2x2 transition matrix for regimes.
    """
    s = states.dropna().astype(int).values
    if len(s) < 2:
        return pd.DataFrame([[np.nan, np.nan], [np.nan, np.nan]], columns=[0, 1], index=[0, 1])

    counts = np.zeros((2, 2))
    for i in range(1, len(s)):
        prev_, curr_ = s[i - 1], s[i]
        if 0 <= prev_ < 2 and 0 <= curr_ < 2:
            counts[prev_, curr_] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = counts.sum(axis=1, keepdims=True)
        probs = np.divide(counts, row_sums, where=row_sums != 0)

    return pd.DataFrame(probs, columns=[0, 1], index=[0, 1])


def compute_risk_score(
    state: int,
    prob: float,
    garch_sigma: float,
    sigma_series: pd.Series,
) -> int:
    """
    Heuristic 0-100 risk score based on regime, probability, and relative volatility.
    """
    if np.isnan(garch_sigma):
        return 50

    score = 50

    # High-vol regime boosts risk
    if state == 1:
        score += 20
    else:
        score -= 10

    # Strong regime probability magnifies effect
    if prob is not None:
        if prob > 0.8:
            score += 20
        elif prob > 0.6:
            score += 10

    # Volatility relative to its historical distribution in Q1 2025
    if len(sigma_series.dropna()) > 0:
        q50 = sigma_series.quantile(0.5)
        q90 = sigma_series.quantile(0.9)
        if garch_sigma > q90:
            score += 15
        elif garch_sigma > q50:
            score += 5
        else:
            score -= 5

    return int(max(0, min(100, score)))


def main() -> None:
    st.set_page_config(page_title="Regime-Aware Risk Dashboard — Q1 2025", layout="wide")
    st.title("AI Volatility Regime Detection Engine")
    st.subheader("Dynamic Risk Dashboard for High-Volatility Assets (Q1 2025)")

    try:
        df = load_results()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    bt_summary = load_backtest_summary()

    assets = sorted(df["ticker"].unique())

    st.sidebar.header("Controls")
    asset = st.sidebar.selectbox("Asset", assets, index=assets.index("TSLA") if "TSLA" in assets else 0)

    df_asset = df[df["ticker"] == asset].copy()
    dates = df_asset.index.unique()
    default_date = dates.max()

    selected_date = st.sidebar.date_input(
        "Select date in Q1 2025",
        value=default_date,
        min_value=dates.min(),
        max_value=dates.max(),
    )
    selected_date = pd.to_datetime(selected_date)
    if selected_date not in dates:
        selected_date = dates[dates <= selected_date].max()
        st.sidebar.caption(f"Using nearest previous trading day: {selected_date.date()}")

    row = df_asset.loc[selected_date]

    # Identify most likely regime on selected date
    state = int(row["state"]) if not np.isnan(row["state"]) else None
    prob = None
    if state is not None:
        prob_col = f"state_{state}_prob"
        if prob_col in df_asset.columns:
            prob = float(row[prob_col])

    # Compute regime duration
    idx_pos = list(dates).index(selected_date)
    duration_days = compute_regime_duration(df_asset["state"], idx_pos) if state is not None else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Asset", asset)

    with col2:
        st.metric("Selected Date", selected_date.strftime("%Y-%m-%d"))

    with col3:
        if state is not None:
            label = "High-Vol Regime" if state == 1 else "Low-Vol Regime"
            st.metric(
                "Current Regime",
                f"{label} (state {state})",
                help="Regime inferred from a Markov-switching model on daily returns.",
            )
        else:
            st.metric("Current Regime", "N/A")

    with col4:
        if prob is not None:
            st.metric("Regime Probability", f"{prob*100:.1f}%")
        else:
            st.metric("Regime Probability", "N/A")

    # Volatility, VaR and risk score
    st.markdown("### Volatility, VaR, and Risk Score (GARCH vs MS-GARCH-like)")
    garch_sigma = float(row["garch_sigma"]) if not np.isnan(row["garch_sigma"]) else np.nan
    ms_sigma = (
        float(row["ms_garch_sigma"]) if "ms_garch_sigma" in row.index and not np.isnan(row["ms_garch_sigma"]) else np.nan
    )
    sigma_series = df_asset["garch_sigma"]

    if not np.isnan(garch_sigma):
        var_95 = 1.645 * garch_sigma
        var_99 = 2.326 * garch_sigma
        risk_score = compute_risk_score(state, prob, garch_sigma, sigma_series)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("GARCH Volatility (σ)", f"{garch_sigma:.4f}")
        with c2:
            st.metric("1-day 95% VaR (GARCH)", f"{var_95:.4f}")
        with c3:
            st.metric("1-day 99% VaR (GARCH)", f"{var_99:.4f}")
        with c4:
            st.metric("Risk Score (0–100)", risk_score)

        # MS-GARCH-like metrics if available
        if not np.isnan(ms_sigma):
            var_95_ms = 1.645 * ms_sigma
            var_99_ms = 2.326 * ms_sigma
            st.metric("1-day 95% VaR (MS-GARCH-like)", f"{var_95_ms:.4f}")
            st.metric("1-day 99% VaR (MS-GARCH-like)", f"{var_99_ms:.4f}")

        # Panic alerts
        if risk_score >= 80:
            st.error("Panic alert: Very high risk regime. Consider reducing exposure or hedging.")
        elif risk_score >= 60:
            st.warning("Elevated risk: Monitor closely and review hedging/position limits.")
        else:
            st.success("Normal risk conditions relative to this asset's history in Q1 2025.")
    else:
        st.info("No GARCH volatility available for this date.")

    # Macro & sentiment context
    st.markdown("### Macro & Sentiment Context")
    macro_cols = ["cpi", "fed_funds", "vix"]
    cpi = float(row["cpi"]) if "cpi" in row.index and not np.isnan(row["cpi"]) else None
    fed = float(row["fed_funds"]) if "fed_funds" in row.index and not np.isnan(row["fed_funds"]) else None
    vix = float(row["vix"]) if "vix" in row.index and not np.isnan(row["vix"]) else None
    sent = (
        float(row["sentiment_score"])
        if "sentiment_score" in row.index and not np.isnan(row["sentiment_score"])
        else None
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("CPI (index)", f"{cpi:.2f}" if cpi is not None else "N/A")
    with c2:
        st.metric("Fed Funds Rate (%)", f"{fed:.2f}" if fed is not None else "N/A")
    with c3:
        st.metric("VIX", f"{vix:.2f}" if vix is not None else "N/A")
    with c4:
        st.metric("News Sentiment", f"{sent:.2f}" if sent is not None else "0.00 (neutral)")

    # Regime duration and transition matrix
    st.markdown("### Regime Persistence and Transition Matrix")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Current Regime Duration", f"{duration_days} trading days")
    with c2:
        tm = compute_transition_matrix(df_asset["state"])
        st.write("Empirical 2-state transition matrix (rows: from, cols: to):")
        st.dataframe(tm.style.format("{:.2f}"))

    # Time series plots
    st.markdown("### Price Dynamics in Q1 2025")
    price_series = (1 + df_asset["ret"].fillna(0)).cumprod()
    price_df = price_series.to_frame("Cumulative Return (indexed to 1.0)")
    if not price_df.dropna().empty:
        price_plot_df = price_df.reset_index().rename(columns={"index": "date"})
        fig = px.line(
            price_plot_df,
            x="date",
            y="Cumulative Return (indexed to 1.0)",
            labels={"date": "Date", "Cumulative Return (indexed to 1.0)": "Cumulative Return"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No price data available to plot for this asset in Q1 2025.")

    st.markdown("### Regime Probabilities Over Time")
    prob_cols = [c for c in df_asset.columns if c.startswith("state_") and c.endswith("_prob")]
    if prob_cols:
        st.area_chart(df_asset[prob_cols])
    else:
        st.info("Regime probability columns not found in results.")

    # Backtest summary
    st.markdown("### VaR Backtest Summary (GARCH, Q1 2025)")
    if not bt_summary.empty:
        if "ticker" in bt_summary.columns:
            bt_asset = bt_summary[bt_summary["ticker"] == asset]
        else:
            bt_asset = bt_summary

        if not bt_asset.empty:
            st.dataframe(bt_asset)
            row_bt = bt_asset.iloc[0]
            text = (
                f"For {asset} in Q1 2025 (α={row_bt['alpha']:.2f}), we observed "
                f"{int(row_bt['n_violations'])} VaR violations out of {int(row_bt['n_obs'])} days. "
                f"Kupiec p-value={row_bt['kupiec_pvalue']:.3f}, "
                f"Christoffersen p-value={row_bt['christoffersen_pvalue']:.3f}."
            )
            st.write(text)
        else:
            st.info(f"No backtest summary found for {asset}.")
    else:
        st.info("Backtest summary not found. Run `python pipeline.py` to generate it.")

    # Decision suggestion
    st.markdown("### Decision Support")
    recommendation = "Hold position."
    if not np.isnan(garch_sigma) and state is not None:
        if state == 1 and (prob is not None and prob > 0.7):
            recommendation = (
                "Regime 1 (high-vol) with high probability: reduce exposure, "
                "consider hedging with options, or increase cash allocation."
            )
        elif state == 0 and (prob is not None and prob > 0.6):
            recommendation = (
                "Regime 0 (lower-vol) with stable probability: within risk limits, "
                "a modest increase in exposure may be acceptable."
            )
    st.info(recommendation)


if __name__ == "__main__":
    main()

