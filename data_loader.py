import pathlib
from typing import Iterable, Tuple

import pandas as pd
import yfinance as yf


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Core high-volatility asset universe
ASSET_UNIVERSE = [
    "TSLA",      # Tesla
    "NVDA",      # NVIDIA
    "BTC-USD",   # Bitcoin
    "AI",        # AI ETF (ticker may vary by broker)
    "^GSPC",     # S&P 500 index
]


def fetch_tsla_prices(
    start: str = "2020-01-01",
    end: str = "2025-04-01",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download TSLA daily data using yfinance and compute returns.
    """
    DATA_DIR.mkdir(exist_ok=True)
    df = yf.download("TSLA", start=start, end=end)
    if df.empty:
        raise ValueError("No data returned for TSLA. Check dates or internet connection.")

    # yfinance can return different column formats depending on version;
    # handle both single-index and MultiIndex columns robustly.
    if isinstance(df.columns, pd.MultiIndex):
        # Try typical (price_field, ticker) layout
        price_series = None
        for field in ["Adj Close", "Close"]:
            try:
                price_series = df[field].iloc[:, 0]
                break
            except Exception:
                continue
        if price_series is None:
            raise KeyError("Could not find an Adj Close or Close column for TSLA.")
        price_series.name = "Adj Close"
        df = price_series.to_frame()
    else:
        price_col = None
        for field in ["Adj Close", "Close"]:
            if field in df.columns:
                price_col = field
                break
        if price_col is None:
            raise KeyError("Could not find an Adj Close or Close column for TSLA.")
        df = df[[price_col]].copy()
        df.rename(columns={price_col: "Adj Close"}, inplace=True)

    df["Return"] = df["Adj Close"].pct_change()
    df = df.dropna().reset_index()
    df.rename(columns={"Date": "date", "Adj Close": "adj_close", "Return": "ret"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])

    if cache:
        df.to_csv(DATA_DIR / "tsla_prices.csv", index=False)

    return df


def load_tsla_prices_from_cache() -> pd.DataFrame:
    """
    Load cached TSLA prices if available.
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
    """
    df = load_tsla_prices_from_cache()
    df = df.sort_values("date")

    estimation = df[df["date"] < "2025-01-01"].copy()
    q1 = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-03-31")].copy()
    if estimation.empty or q1.empty:
        raise ValueError("Estimation or Q1 2025 window is empty. Check date ranges and raw data.")
    return estimation, q1


def fetch_multi_asset_prices(
    tickers: Iterable[str] = ASSET_UNIVERSE,
    start: str = "2020-01-01",
    end: str = "2025-04-01",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily data for a list of tickers and compute returns.

    Returns a long-format DataFrame with columns:
    ['date', 'ticker', 'adj_close', 'ret'].
    """
    tickers = list(tickers)
    DATA_DIR.mkdir(exist_ok=True)

    df = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=False)
    if df.empty:
        raise ValueError("No data returned for requested tickers. Check dates or internet.")

    all_rows = []
    for ticker in tickers:
        # yfinance structure differs for single vs multi ticker; be defensive
        try:
            sub = df[ticker][["Adj Close"]].copy()
        except KeyError:
            # Fallback: maybe this is a single-ticker download
            if "Adj Close" in df.columns:
                sub = df[["Adj Close"]].copy()
            else:
                continue

        sub["Return"] = sub["Adj Close"].pct_change()
        sub = sub.dropna().reset_index()
        sub.rename(
            columns={"Date": "date", "Adj Close": "adj_close", "Return": "ret"},
            inplace=True,
        )
        sub["date"] = pd.to_datetime(sub["date"])
        sub["ticker"] = ticker
        all_rows.append(sub)

    if not all_rows:
        raise ValueError("No valid price series could be constructed for the given tickers.")

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["ticker", "date"])

    if cache:
        out.to_csv(DATA_DIR / "multi_asset_prices.csv", index=False)

    return out


def load_multi_asset_prices_from_cache() -> pd.DataFrame:
    """
    Load cached multi-asset prices if available.
    """
    path = DATA_DIR / "multi_asset_prices.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fetch_multi_asset_prices() first or execute pipeline.py."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def get_universe_q1_2025_window() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the multi-asset universe into:
    - estimation window (before 2025-01-01)
    - Q1 2025 evaluation window (2025-01-01 to 2025-03-31)

    Returns long-format (date, ticker, adj_close, ret) DataFrames.
    """
    df = load_multi_asset_prices_from_cache()
    df = df.sort_values(["ticker", "date"])

    estimation = df[df["date"] < "2025-01-01"].copy()
    q1 = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2025-03-31")].copy()
    if estimation.empty or q1.empty:
        raise ValueError("Estimation or Q1 2025 window is empty for multi-asset universe.")
    return estimation, q1

