"""
Download bond yield data from FRED and ETF sector proxies into one weekly CSV.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_FILE = DATA_DIR / "bondyields.csv"
ERROR_FILE = DATA_DIR / "bond_download_errors.csv"

START_DATE_ETF = "2005-01-01"
DAILY_RULE = "D"
ETF_YIELD_WINDOW_DAYS = 91
ETF_YIELD_SMOOTH_DAYS = 28

ETF_SERIES: dict[str, dict[str, str]] = {
    "US_IG_CORP_PROXY": {"ticker": "LQD", "out_col": "IG_CORP:LQD"},
    "AAA_CORP_PROXY": {"ticker": "QLTA", "out_col": "AAA_CORP:QLTA"},
    "US_HY_CORP_PROXY": {"ticker": "HYG", "out_col": "HY_CORP:HYG"},
    "AAA_CLO": {"ticker": "JAAA", "out_col": "AAA_CLO:JAAA"},
    "SENIOR_LOANS": {"ticker": "BKLN", "out_col": "SENIOR_LOANS:BKLN"},
    "AGENCY_MBS": {"ticker": "MBB", "out_col": "AGENCY_MBS:MBB"},
    "EM_SOV_HARD": {"ticker": "EMB", "out_col": "EM_SOV_HARD:EMB"},
    "EM_SOV_LOCAL": {"ticker": "ELD", "out_col": "EM_SOV_LOCAL:ELD"},
    "MONEY_MARKET": {"ticker": "SGOV", "out_col": "MONEY_MARKET:SGOV"},
    "US_AGENCY": {"ticker": "AGZ", "out_col": "US_AGENCY:AGZ"},
    "IG_MUNIS": {"ticker": "MUB", "out_col": "IG_MUNIS:MUB"},
    "HY_MUNIS": {"ticker": "HYD", "out_col": "HY_MUNIS:HYD"},
}

def round_all_yield_columns(df: pd.DataFrame, digits: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    candidate_cols = [c for c in out.columns if c != "date"]
    for col in candidate_cols:
        try:
            coerced = pd.to_numeric(out[col], errors="coerce")
            if coerced.notna().any():
                out[col] = coerced.round(digits)
        except Exception:
            continue
    return out


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
    px_d = adj_close.resample(DAILY_RULE).last().ffill().dropna()
    div_d = dividends.resample(DAILY_RULE).sum().reindex(px_d.index, fill_value=0.0)
    rolling_div = div_d.rolling(ETF_YIELD_WINDOW_DAYS, min_periods=1).sum()
    annualized_yield = (rolling_div / px_d) * (365.0 / ETF_YIELD_WINDOW_DAYS) * 100.0
    return annualized_yield.rolling(ETF_YIELD_SMOOTH_DAYS, min_periods=1).mean()


def download_sector_weekly(start_date: str) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    series_list: list[pd.Series] = []
    errors: list[dict[str, str]] = []

    for sector, spec in ETF_SERIES.items():
        ticker = spec["ticker"]
        out_col = spec["out_col"]
        try:
            adj, div = download_etf_series(ticker, start_date=start_date)
            yld = compute_weekly_ttm_yield(adj, div)
            yld.name = out_col
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

    print(f"Downloading ETF sector daily series ({START_DATE_ETF} to {end_date})...")
    etf_weekly, errors = download_sector_weekly(start_date=START_DATE_ETF)

    return etf_weekly.sort_index(), errors


if __name__ == "__main__":
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        combined_df, download_errors = download_all_bond_data()
        combined_df = round_all_yield_columns(combined_df, digits=2)
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
