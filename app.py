import pathlib

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


@st.cache_data
def load_results() -> pd.DataFrame:
    path = OUTPUT_DIR / "tsla_q1_2025_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python pipeline.py` first to generate results."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    return df


@st.cache_data
def load_backtest_summary() -> pd.DataFrame:
    path = OUTPUT_DIR / "tsla_q1_2025_backtest_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    st.set_page_config(page_title="TSLA Regime & VaR — Q1 2025", layout="wide")
    st.title("Improving Risk Management with Regime-Switching Models")
    st.subheader("Case Study: Tesla (TSLA) in Q1 2025")

    try:
        df = load_results()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    bt_summary = load_backtest_summary()

    dates = df.index
    default_date = dates.max()

    st.sidebar.header("Controls")
    selected_date = st.sidebar.date_input(
        "Select date in Q1 2025", value=default_date, min_value=dates.min(), max_value=dates.max()
    )
    if pd.to_datetime(selected_date) not in df.index:
        st.warning("Selected date not in trading calendar. Showing nearest previous trading day.")
        selected_date = df.index[df.index <= pd.to_datetime(selected_date)].max()

    row = df.loc[pd.to_datetime(selected_date)]

    # Identify most likely regime on selected date
    state = int(row["state"]) if not np.isnan(row["state"]) else None
    prob = None
    if state is not None:
        prob_col = f"state_{state}_prob"
        if prob_col in df.columns:
            prob = float(row[prob_col])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Selected Date", selected_date.strftime("%Y-%m-%d"))

    with col2:
        if state is not None:
            label = "High-Vol Regime" if state == 1 else "Low-Vol Regime"
            st.metric(
                "Current Regime",
                f"{label} (state {state})",
                help="Regime inferred from a Markov-switching model on TSLA returns.",
            )
        else:
            st.metric("Current Regime", "N/A")

    with col3:
        if prob is not None:
            st.metric("Regime Probability", f"{prob*100:.1f}%")
        else:
            st.metric("Regime Probability", "N/A")

    # Volatility and simple VaR display
    st.markdown("### Volatility and Simple VaR (GARCH-based)")
    garch_sigma = float(row["garch_sigma"]) if not np.isnan(row["garch_sigma"]) else None
    if garch_sigma is not None:
        var_95 = 1.645 * garch_sigma
        var_99 = 2.326 * garch_sigma

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("GARCH Volatility (σ)", f"{garch_sigma:.4f}")
        with c2:
            st.metric("1-day 95% VaR", f"{var_95:.4f}")
        with c3:
            st.metric("1-day 99% VaR", f"{var_99:.4f}")
    else:
        st.info("No GARCH volatility available for this date.")

    # Time series plots
    st.markdown("### TSLA Price and Regime States in Q1 2025")
    price_series = (df["ret"] + 1).cumprod()
    plot_df = pd.DataFrame(
        {
            "Cumulative Return (indexed to 1.0)": price_series,
            "Regime State": df["state"],
        }
    )
    st.line_chart(plot_df[["Cumulative Return (indexed to 1.0)"]])

    st.markdown("### Regime Probabilities Over Time")
    prob_cols = [c for c in df.columns if c.startswith("state_") and c.endswith("_prob")]
    if prob_cols:
        st.area_chart(df[prob_cols])
    else:
        st.info("Regime probability columns not found in results.")

    # Backtest summary
    st.markdown("### VaR Backtest Summary (GARCH, Q1 2025)")
    if not bt_summary.empty:
        st.dataframe(bt_summary)

        row_bt = bt_summary.iloc[0]
        text = (
            f"In Q1 2025, with α={row_bt['alpha']:.2f}, we observed "
            f"{int(row_bt['n_violations'])} VaR violations out of {int(row_bt['n_obs'])} days. "
            f"Kupiec p-value={row_bt['kupiec_pvalue']:.3f}, "
            f"Christoffersen p-value={row_bt['christoffersen_pvalue']:.3f}."
        )
        st.write(text)
    else:
        st.info("Backtest summary not found. Run `python pipeline.py` to generate it.")

    # Simple decision suggestion
    st.markdown("### Simple Risk Recommendation")
    recommendation = "Hold position."
    if garch_sigma is not None and state is not None:
        if state == 1 and (prob is not None and prob > 0.7):
            recommendation = (
                "High-Vol regime with elevated probability: consider reducing exposure or hedging."
            )
        elif state == 0 and (prob is not None and prob > 0.6):
            recommendation = (
                "Low-Vol regime with stable probability: risk is lower, modest increase in exposure "
                "may be acceptable within risk limits."
            )
    st.info(recommendation)


if __name__ == "__main__":
    main()

