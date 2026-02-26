import pathlib
from typing import Tuple

import pandas as pd
import yfinance as yf


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_tsla_prices(
    start: str = "2024-01-01",
    end: str = "2025-04-01",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download TSLA daily data using yfinance and compute log-returns.

    Parameters
    ----------
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    cache : bool
        If True, save a cached CSV under data/tsla_prices.csv.

    Returns
    -------
    pd.DataFrame
        Columns: ['Date', 'Adj Close', 'Return'] indexed by Date.
    """
    DATA_DIR.mkdir(exist_ok=True)
    df = yf.download("TSLA", start=start, end=end)
    if df.empty:
        raise ValueError("No data returned for TSLA. Check dates or internet connection.")

    df = df[["Adj Close"]].copy()
    df["Return"] = df["Adj Close"].pct_change().dropna()
    df = df.dropna().reset_index()
    df.rename(columns={"Date": "date", "Adj Close": "adj_close", "Return": "ret"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    if cache:
        df.to_csv(DATA_DIR / "tsla_prices.csv", index=False)

    return df


def load_tsla_prices_from_cache() -> pd.DataFrame:
    """
    Load cached TSLA prices if available.

    Returns
    -------
    pd.DataFrame
        Cached TSLA data with columns ['date', 'adj_close', 'ret'].
    """
    path = DATA_DIR / "tsla_prices.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fetch_tsla_prices() first or execute pipeline.py."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def get_tsla_q1_2025_window() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Helper to split TSLA data into:
    - estimation window (before 2025-01-01)
    - Q1 2025 evaluation window (2025-01-01 to 2025-03-31)

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (estimation_df, q1_df)
    """
    df = load_tsla_prices_from_cache()
    df = df.sort_values("date")

    estimation = df[df["date"] < "2025-01-01"].copy()
    q1 = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-03-31")].copy()
    if estimation.empty or q1.empty:
        raise ValueError("Estimation or Q1 2025 window is empty. Check date ranges and raw data.")
    return estimation, q1

