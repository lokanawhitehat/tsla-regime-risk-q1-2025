import pathlib
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from pandas_datareader import data as pdr  # type: ignore
except Exception:
    # On Python 3.12+, pandas_datareader may fail due to missing distutils.
    # In that case we fall back to using only VIX and leaving CPI/Fed as NaN.
    pdr = None


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_macro_data(
    start: str = "2024-01-01",
    end: str = "2025-04-01",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch macro series relevant for risk regimes:
    - CPI (CPIAUCSL, FRED)
    - Effective Fed Funds Rate (FEDFUNDS, FRED)
    - VIX (^VIX, via yfinance)

    Returns a daily DataFrame with columns: ['date', 'cpi', 'fed_funds', 'vix'].
    """
    DATA_DIR.mkdir(exist_ok=True)

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    # CPI and Fed Funds from FRED via pandas_datareader (if available)
    if pdr is not None:
        fred = pdr.DataReader(["CPIAUCSL", "FEDFUNDS"], "fred", start_dt, end_dt)
        fred = fred.rename(columns={"CPIAUCSL": "cpi", "FEDFUNDS": "fed_funds"})
    else:
        # Create empty frame; CPI and Fed funds will remain NaN.
        fred = pd.DataFrame(index=pd.date_range(start=start_dt, end=end_dt, freq="D"))

    # VIX from yfinance (handle both single and MultiIndex column formats)
    vix = yf.download("^VIX", start=start, end=end)
    if vix.empty:
        vix_series = pd.Series(dtype=float, name="vix")
    else:
        if isinstance(vix.columns, pd.MultiIndex):
            price_series = None
            for field in ["Adj Close", "Close"]:
                try:
                    price_series = vix[field].iloc[:, 0]
                    break
                except Exception:
                    continue
            if price_series is None:
                # fall back to the first column
                price_series = vix.iloc[:, 0]
            vix_series = price_series.rename("vix")
        else:
            price_col = None
            for field in ["Adj Close", "Close"]:
                if field in vix.columns:
                    price_col = field
                    break
            if price_col is None:
                price_col = vix.columns[0]
            vix_series = vix[price_col].rename("vix")

    vix_df = vix_series.to_frame()

    # Convert to daily and forward-fill
    macro = fred.join(vix_df, how="outer")
    macro = macro.sort_index().resample("D").ffill()
    macro = macro.reset_index().rename(columns={"index": "date", "DATE": "date"})
    macro["date"] = pd.to_datetime(macro["date"])

    if cache:
        macro.to_csv(DATA_DIR / "macro_data.csv", index=False)

    return macro


def load_macro_from_cache() -> pd.DataFrame:
    path = DATA_DIR / "macro_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fetch_macro_data() first or execute pipeline.py."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def fetch_sentiment_data(
    tickers: Iterable[str],
    start: str = "2024-01-01",
    end: str = "2025-04-01",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Placeholder for daily news sentiment scores per asset.

    For now, this returns a neutral sentiment score (0.0) for each
    trading day and ticker in the requested range. This keeps the
    code simple while leaving a clear place to plug in a real NLP
    pipeline (e.g. FinBERT or VADER on news headlines).
    """
    DATA_DIR.mkdir(exist_ok=True)
    tickers = list(tickers)

    dates = pd.date_range(start=start, end=end, freq="B")
    rows = []
    for d in dates:
        for t in tickers:
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "sentiment_score": 0.0,
                }
            )
    sentiment = pd.DataFrame(rows)

    if cache:
        sentiment.to_csv(DATA_DIR / "sentiment_data.csv", index=False)

    return sentiment


def load_sentiment_from_cache() -> pd.DataFrame:
    path = DATA_DIR / "sentiment_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fetch_sentiment_data() first or execute pipeline.py."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df

