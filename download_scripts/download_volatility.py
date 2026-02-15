"""
download_volatility.py

Downloads cross-asset volatility / stress indicators (2007+) from:
- FRED (macro/credit/vol indices + yields)
- Yahoo Finance via yfinance (MOVE, SKEW, VVIX)

Outputs:
- data/volatility.csv

Install:
  pip install pandas fredapi yfinance requests
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf
from fredapi import Fred


# -----------------------------
# Config
# -----------------------------

START_DATE = "2007-01-01"
END_DATE = None  # None => today
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR.parent / "data"
OUT_CSV_COMBINED = OUT_DIR / "volatility.csv"
FRED_API_KEY = "69da3d502e36febadb1d149b360b8464"
ROUND_DIGITS = 2
Z_SCORE_WINDOW = 90
VOLATILITY_OUTPUT_ORDER = [
    "vix",
    "vxn",
    "move",
    "ig_oas",
    "hy_oas",
    "gvz",
    "ovx",
    "stlfsi",
]


# -----------------------------
# FRED series (daily/weekly/mixed)
# -----------------------------
# These IDs are standard FRED mnemonics.
FRED_SERIES: Dict[str, str] = {
    # Equity implied vol
    "vix": "VIXCLS",
    "vxn": "VXNCLS",

    # Commodity implied vol
    "gvz": "GVZCLS",   # gold vol index
    "ovx": "OVXCLS",   # oil vol index

    # Broad financial stress / conditions
    "stlfsi": "STLFSI4",  # St. Louis Fed Financial Stress Index

    # Credit spreads (ICE BofA OAS)
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",

}


# -----------------------------
# Yahoo Finance tickers (via yfinance)
# -----------------------------
YF_TICKERS: Dict[str, str] = {
    "move": "^MOVE",
}


# -----------------------------
# Helpers
# -----------------------------

def _ensure_out_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def fetch_fred_series(series_map: Dict[str, str],
                      start: str,
                      end: Optional[str] = None) -> pd.DataFrame:
    """
    Pull multiple FRED series into one DataFrame.
    """
    fred_client = Fred(api_key=FRED_API_KEY)
    frames: list[pd.DataFrame] = []

    for out_col, series_id in series_map.items():
        series = fred_client.get_series(
            series_id,
            observation_start=start,
            observation_end=end,
        )
        if series is None or series.empty:
            continue
        df = series.rename(out_col).to_frame()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1, sort=False).sort_index()
    return out


def fetch_yfinance_close(ticker_map: Dict[str, str],
                         start: str,
                         end: Optional[str] = None) -> pd.DataFrame:
    """
    Pull Adjusted Close (preferred) where available; fallback to Close.
    """
    tickers = list(ticker_map.values())
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
    )

    # yfinance structure differs for single vs multi ticker.
    # We standardize to a DataFrame of "Adj Close" if present, else "Close".
    if isinstance(raw.columns, pd.MultiIndex):
        if ("Adj Close" in raw.columns.get_level_values(0)):
            px = raw["Adj Close"].copy()
        else:
            px = raw["Close"].copy()
        px = px.rename(columns={v: k for k, v in ticker_map.items()})
    else:
        # single ticker
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        px = raw[[col]].rename(columns={col: list(ticker_map.keys())[0]})

    return px


def add_curve_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds common derived features (curve slopes, real/nominal deltas).
    """
    out = df.copy()

    # Curve slopes
    if {"ust_10y", "ust_2y"}.issubset(out.columns):
        out["curve_10y2y"] = out["ust_10y"] - out["ust_2y"]
    if {"ust_10y", "ust_3m"}.issubset(out.columns):
        out["curve_10y3m"] = out["ust_10y"] - out["ust_3m"]

    return out


def build_rolling_zscore_dataset(df: pd.DataFrame, window: int = Z_SCORE_WINDOW) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = pd.to_numeric(out[col], errors="coerce")
        valid = series.dropna()
        roll_mean = valid.rolling(window=window, min_periods=window).mean()
        roll_std = valid.rolling(window=window, min_periods=window).std()
        z_valid = (valid - roll_mean) / roll_std.replace(0, pd.NA)
        out[col] = z_valid.reindex(series.index)
    return out


# -----------------------------
# Optional OFR FSI stub
# -----------------------------
def fetch_ofr_fsi_stub() -> pd.DataFrame:
    """
    OFR FSI page is interactive/JS-driven; a stable direct CSV endpoint isn't
    exposed in the extracted HTML view. If you identify a direct CSV/JSON endpoint,
    implement it here and merge into the final dataset.
    """
    # Example placeholder:
    # url = "https://<your-direct-endpoint>.csv"
    # df = pd.read_csv(url, parse_dates=["date"]).set_index("date")
    # df = df.rename(columns={"value": "ofr_fsi"})
    # return df[["ofr_fsi"]]
    return pd.DataFrame()


# -----------------------------
# Main pipeline
# -----------------------------
def build_indicator_dataset(start: str = START_DATE,
                            end: Optional[str] = END_DATE) -> pd.DataFrame:
    # 1) FRED
    fred = fetch_fred_series(FRED_SERIES, start, end)

    # 2) Yahoo Finance indices (MOVE/SKEW/VVIX)
    yf_idx = fetch_yfinance_close(YF_TICKERS, start, end)

    # 3) Optional OFR (stub)
    ofr = fetch_ofr_fsi_stub()

    # 4) Merge + feature engineering
    df = pd.concat([fred, yf_idx, ofr], axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = add_curve_features(df)

    # Optional: keep from 2007 onward even if some sources start later
    df = df.loc[pd.to_datetime(start):]

    return df


def main() -> None:
    _ensure_out_dir(OUT_CSV_COMBINED)

    df_original = build_indicator_dataset()
    ordered_original_cols = [c for c in VOLATILITY_OUTPUT_ORDER if c in df_original.columns]
    remaining_original_cols = [c for c in df_original.columns if c not in ordered_original_cols]
    df_original = df_original[ordered_original_cols + remaining_original_cols]
    df_zscore = build_rolling_zscore_dataset(df_original, window=Z_SCORE_WINDOW)

    numeric_cols_original = df_original.select_dtypes(include="number").columns
    if len(numeric_cols_original) > 0:
        df_original.loc[:, numeric_cols_original] = df_original.loc[:, numeric_cols_original].round(ROUND_DIGITS)

    numeric_cols_zscore = df_zscore.select_dtypes(include="number").columns
    if len(numeric_cols_zscore) > 0:
        df_zscore.loc[:, numeric_cols_zscore] = df_zscore.loc[:, numeric_cols_zscore].round(ROUND_DIGITS)

    df_combined = pd.concat(
        [
            df_original,
            df_zscore.add_suffix("_zscore"),
        ],
        axis=1,
        sort=False,
    )

    # Save
    df_combined.to_csv(OUT_CSV_COMBINED, index_label="date")

    print(f"[OK] Wrote {len(df_combined):,} rows x {df_combined.shape[1]:,} cols")
    print(f"     Combined CSV: {OUT_CSV_COMBINED}")


if __name__ == "__main__":
    main()
