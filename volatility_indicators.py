"""
volatility_indicators.py

Downloads cross-asset volatility / stress indicators (2007+) from:
- FRED (macro/credit/vol indices + yields)
- Yahoo Finance via yfinance (MOVE, SKEW, VVIX)

Outputs:
- data/indicators.csv
- data/indicators.parquet

Install:
  pip install pandas fredapi yfinance pyarrow requests
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fredapi import Fred


# -----------------------------
# Config
# -----------------------------

START_DATE = "2007-01-01"
END_DATE = None  # None => today
OUT_DIR = "data"
OUT_CSV = os.path.join(OUT_DIR, "volatility_indicators.csv")
OUT_PARQUET = os.path.join(OUT_DIR, "volatility_indicators.parquet")
FRED_API_KEY = "69da3d502e36febadb1d149b360b8464"


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
    "nfci": "NFCI",       # Chicago Fed National Financial Conditions Index
    "anfci": "ANFCI",     # Adjusted NFCI

    # Credit spreads (ICE BofA OAS)
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",

    # Funding stress (legacy; discontinued but useful historically)
    "ted_spread": "TEDRATE",

    # Rates + curve
    "ust_3m": "DGS3MO",
    "ust_2y": "DGS2",
    "ust_10y": "DGS10",

    # Real rates + breakevens
    "tips_10y_real": "DFII10",  # 10Y TIPS real yield
    "breakeven_10y": "T10YIE",  # 10Y breakeven inflation
}


# -----------------------------
# Yahoo Finance tickers (via yfinance)
# -----------------------------
YF_TICKERS: Dict[str, str] = {
    "move": "^MOVE",
    "skew": "^SKEW",
    "vvix": "^VVIX",
}


# Optional: realized vol proxies (price-based, cross-asset)
# You can comment this out if you only want the index series.
RV_PROXIES: Dict[str, str] = {
    "spy": "SPY",
    "tlt": "TLT",
    "hyg": "HYG",
    "lqd": "LQD",
    "gld": "GLD",
    "uso": "USO",
    "uup": "UUP",
}


# -----------------------------
# Helpers
# -----------------------------

def _ensure_out_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


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

    out = pd.concat(frames, axis=1).sort_index()
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


def realized_vol(price: pd.Series, window: int) -> pd.Series:
    """
    Annualized realized vol from log returns.
    Assumes ~252 trading days.
    """
    r = (price.astype(float).replace(0, pd.NA)).dropna()
    lr = (r / r.shift(1)).apply(lambda x: pd.NA if pd.isna(x) else np.log(x))
    rv = lr.rolling(window).std() * (252 ** 0.5) * 100.0
    return rv.rename(f"{price.name}_rv_{window}d")


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

    # Real-rate gap (nominal - real) should approximate breakeven; useful sanity check
    if {"ust_10y", "tips_10y_real"}.issubset(out.columns):
        out["breakeven_proxy_10y"] = out["ust_10y"] - out["tips_10y_real"]

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
                            end: Optional[str] = END_DATE,
                            include_realized_vol: bool = True) -> pd.DataFrame:
    # 1) FRED
    fred = fetch_fred_series(FRED_SERIES, start, end)

    # 2) Yahoo Finance indices (MOVE/SKEW/VVIX)
    yf_idx = fetch_yfinance_close(YF_TICKERS, start, end)

    # 3) Optional: realized vol proxies (prices)
    yf_px = pd.DataFrame()
    rv = pd.DataFrame()
    if include_realized_vol:
        yf_px = fetch_yfinance_close(RV_PROXIES, start, end)
        # realized vols (20d, 60d, 252d)
        rvs = []
        for col in yf_px.columns:
            rvs.append(realized_vol(yf_px[col], 20))
            rvs.append(realized_vol(yf_px[col], 60))
            rvs.append(realized_vol(yf_px[col], 252))
        rv = pd.concat(rvs, axis=1)

    # 4) Optional OFR (stub)
    ofr = fetch_ofr_fsi_stub()

    # 5) Merge + feature engineering
    df = pd.concat([fred, yf_idx, yf_px.add_prefix("px_"), rv, ofr], axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = add_curve_features(df)

    # Optional: keep from 2007 onward even if some sources start later
    df = df.loc[pd.to_datetime(start):]

    return df


def main() -> None:
    _ensure_out_dir(OUT_CSV)
    _ensure_out_dir(OUT_PARQUET)

    df = build_indicator_dataset()

    # Save
    df.to_csv(OUT_CSV, index_label="date")
    try:
        df.to_parquet(OUT_PARQUET, index=True)
    except Exception as e:
        print(f"[WARN] Parquet save failed ({e}). CSV saved to {OUT_CSV}.")

    print(f"[OK] Wrote {len(df):,} rows x {df.shape[1]:,} cols")
    print(f"     CSV:     {OUT_CSV}")
    print(f"     Parquet: {OUT_PARQUET}")


if __name__ == "__main__":
    main()
