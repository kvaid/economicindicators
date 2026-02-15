"""
Download bond yield data from FRED and ETF sector proxies into one weekly CSV.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf
from fredapi import Fred

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_FILE = DATA_DIR / "bondyields.csv"
ERROR_FILE = DATA_DIR / "bond_download_errors.csv"

START_DATE_FRED = "2000-01-01"
START_DATE_ETF = "2005-01-01"
WEEKLY_RULE = "W-FRI"
ETF_YIELD_WINDOW_WEEKS = 52  # Set to 8 for a shorter rolling window.
FRED_API_KEY = "69da3d502e36febadb1d149b360b8464"

FRED_SERIES: dict[str, str] = {
    "ice_bofa_us_corporate_effective_yield": "BAMLC0A0CMEY",
    "ice_bofa_us_high_yield_effective_yield": "BAMLH0A0HYM2EY",
    "ice_bofa_aaa_us_corporate_effective_yield": "BAMLC0A1CAAAEY",
    "ice_bofa_b_us_corporate_effective_yield": "BAMLC0A4CBBBEY",
}

SECTOR_ETFS: dict[str, str] = {
    "AAA_CLO": "JAAA",
    "SENIOR_LOANS": "BKLN",
    "AGENCY_MBS": "MBB",
    "CMBS": "CMBS",
    "IG_MUNIS": "MUB",
    "HY_MUNIS": "HYD",
}


def round_etf_proxy_columns(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    etf_cols = [f"{sector}:{ticker}" for sector, ticker in SECTOR_ETFS.items()]
    existing_cols = [col for col in etf_cols if col in out.columns]
    if existing_cols:
        out[existing_cols] = out[existing_cols].round(digits)
    return out


def download_fred_weekly(start_date: str, end_date: str) -> pd.DataFrame:
    fred_client = Fred(api_key=FRED_API_KEY)
    frames: list[pd.DataFrame] = []
    for out_col, series_id in FRED_SERIES.items():
        series = fred_client.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
        if series is None or series.empty:
            continue
        df = series.rename(out_col).to_frame().reset_index()
        df = df.rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df[out_col] = pd.to_numeric(df[out_col], errors="coerce")
        df = df.dropna(subset=["date"])
        frames.append(df[["date", out_col]])

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="date", how="outer")

    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    merged = merged.set_index("date").resample(WEEKLY_RULE).last().ffill()
    merged.index.name = "date"
    merged = normalize_date_index(merged)
    return merged


def download_etf_series(ticker: str, start_date: str) -> tuple[pd.Series, pd.Series]:
    t = yf.Ticker(ticker)
    hist = t.history(start=start_date, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No price history for {ticker}")

    adj = hist["Adj Close"].dropna()
    adj.index = pd.to_datetime(adj.index)

    div = t.dividends
    if div is None:
        div = pd.Series(dtype=float)
    div = div.dropna()
    div.index = pd.to_datetime(div.index)
    return adj, div


def compute_weekly_ttm_yield(adj_close: pd.Series, dividends: pd.Series) -> pd.Series:
    px_w = adj_close.resample(WEEKLY_RULE).last().dropna()
    div_w = dividends.resample(WEEKLY_RULE).sum().reindex(px_w.index, fill_value=0.0)
    rolling_div = div_w.rolling(ETF_YIELD_WINDOW_WEEKS, min_periods=1).sum()
    return (rolling_div / px_w) * 100.0


def download_sector_weekly(start_date: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    series_list: list[pd.Series] = []
    errors: list[dict[str, str]] = []

    for sector, ticker in SECTOR_ETFS.items():
        try:
            adj, div = download_etf_series(ticker, start_date=start_date)
            yld = compute_weekly_ttm_yield(adj, div)
            yld.name = f"{sector}:{ticker}"
            series_list.append(yld)
        except Exception as exc:
            errors.append({"sector": sector, "ticker": ticker, "error": str(exc)})

    if not series_list:
        return pd.DataFrame(), errors

    out = pd.concat(series_list, axis=1, sort=False).sort_index()
    out.index.name = "date"
    out = normalize_date_index(out)
    return out, errors


def normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is not None:
            out.index = out.index.tz_convert("UTC").tz_localize(None)
        out.index = out.index.normalize()
    return out


def download_all_bond_data() -> tuple[pd.DataFrame, list[dict[str, str]]]:
    end_date = datetime.now().strftime("%Y-%m-%d")

    print(f"Downloading FRED weekly series ({START_DATE_FRED} to {end_date})...")
    fred_weekly = download_fred_weekly(start_date=START_DATE_FRED, end_date=end_date)

    print(f"Downloading ETF sector weekly series ({START_DATE_ETF} to {end_date})...")
    etf_weekly, errors = download_sector_weekly(start_date=START_DATE_ETF)

    combined = fred_weekly.join(etf_weekly, how="outer").sort_index()
    return combined, errors


if __name__ == "__main__":
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        combined_df, download_errors = download_all_bond_data()
        combined_df = round_etf_proxy_columns(combined_df, digits=2)
        combined_df.reset_index().to_csv(
            OUTPUT_FILE,
            index=False,
            date_format="%Y-%m-%d",
        )

        if download_errors:
            pd.DataFrame(download_errors).to_csv(ERROR_FILE, index=False)
            print(f"Wrote {len(download_errors)} download errors to {ERROR_FILE}")

        print(f"OK Saved {len(combined_df)} rows to {OUTPUT_FILE}")
        if not combined_df.empty:
            print(f"Date range: {combined_df.index.min().date()} to {combined_df.index.max().date()}")
    except Exception as exc:
        print(f"ERROR: {exc}")
