"""
download_volatility.py

Downloads cross-asset volatility / stress indicators (2007+) from:
- FRED (macro/credit/vol indices + yields)
- Yahoo Finance via yfinance (MOVE, SKEW)

Outputs:
- data/volatility.csv

Install:
  pip install pandas fredapi yfinance
"""

from __future__ import annotations

import os
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
CACHE_DIR = BASE_DIR.parent / ".cache" / "yfinance"
ROUND_DIGITS = 2
VOLATILITY_OUTPUT_ORDER = [
    "vix",
    "vxn",
    "rvx",
    "vxeem",
    "skew",
    "move",
    "dxy",
    "cnn_fear_greed",
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
    "rvx": "RVXCLS",
    "vxeem": "VXEEMCLS",

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
    "skew": "^SKEW",
    "dxy": "DX-Y.NYB",
}

PROXY_HELPER_TICKERS: Dict[str, str] = {
    "spy": "SPY",
    "tlt": "TLT",
    "hyg": "HYG",
    "lqd": "LQD",
    "xly": "XLY",
    "xlp": "XLP",
}


# -----------------------------
# Helpers
# -----------------------------

def _ensure_out_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _configure_yfinance_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache_dir))
    except Exception as exc:
        print(f"[WARN] Could not set yfinance cache location: {exc}")


def fetch_fred_series(series_map: Dict[str, str],
                      start: str,
                      end: Optional[str] = None) -> pd.DataFrame:
    """
    Pull multiple FRED series into one DataFrame.
    """
    fred_api_key = os.environ.get("FRED_API_KEY", "").strip()
    if not fred_api_key:
        raise RuntimeError("Missing FRED_API_KEY environment variable.")
    fred_client = Fred(api_key=fred_api_key)
    frames: list[pd.DataFrame] = []

    for out_col, series_id in series_map.items():
        try:
            series = fred_client.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
        except Exception as exc:
            print(f"[WARN] Skipping FRED series {series_id} ({out_col}): {exc}")
            continue
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
        threads=False,
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


def _rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.rolling(window=window, min_periods=min_periods).mean()
    sigma = s.rolling(window=window, min_periods=min_periods).std()
    z = (s - mu) / sigma
    return z.clip(-3, 3)


def _z_to_score(z: pd.Series, invert: bool = False) -> pd.Series:
    score = 50.0 + (z / 3.0) * 50.0
    score = score.clip(0, 100)
    return 100.0 - score if invert else score


def build_cnn_fear_greed_proxy(base_df: pd.DataFrame, helper_px: pd.DataFrame) -> pd.Series:
    """
    CNN Fear & Greed-style proxy (0-100) using 7 weighted components.
    This is an approximation, not CNN's proprietary exact methodology.
    """
    components: dict[str, pd.Series] = {}

    # 1) Market volatility (lower VIX => more greed)
    if "vix" in base_df.columns:
        components["market_volatility"] = _z_to_score(_rolling_zscore(base_df["vix"]), invert=True)

    # 2) Nasdaq volatility (lower VXN => more greed)
    if "vxn" in base_df.columns:
        components["nasdaq_volatility"] = _z_to_score(_rolling_zscore(base_df["vxn"]), invert=True)

    if {"spy"}.issubset(helper_px.columns):
        spy = pd.to_numeric(helper_px["spy"], errors="coerce")
        # 3) Market momentum vs 125D trend
        trend_125 = (spy / spy.rolling(window=125, min_periods=60).mean()) - 1.0
        components["market_momentum"] = _z_to_score(_rolling_zscore(trend_125), invert=False)
        # 4) Short-term momentum proxy
        mom_20 = spy.pct_change(20)
        components["price_strength_proxy"] = _z_to_score(_rolling_zscore(mom_20), invert=False)

    # 5) Junk bond demand proxy
    if {"hyg", "lqd"}.issubset(helper_px.columns):
        hyg_lqd = pd.to_numeric(helper_px["hyg"], errors="coerce") / pd.to_numeric(helper_px["lqd"], errors="coerce")
        components["junk_bond_demand"] = _z_to_score(_rolling_zscore(hyg_lqd), invert=False)

    # 6) Safe-haven demand proxy: SPY/TLT (higher => greedier)
    if {"spy", "tlt"}.issubset(helper_px.columns):
        spy_tlt = pd.to_numeric(helper_px["spy"], errors="coerce") / pd.to_numeric(helper_px["tlt"], errors="coerce")
        components["safe_haven_demand"] = _z_to_score(_rolling_zscore(spy_tlt), invert=False)

    # 7) Stock/beta risk appetite proxy
    if {"xly", "xlp"}.issubset(helper_px.columns):
        xly_xlp = pd.to_numeric(helper_px["xly"], errors="coerce") / pd.to_numeric(helper_px["xlp"], errors="coerce")
        components["risk_appetite"] = _z_to_score(_rolling_zscore(xly_xlp), invert=False)

    if not components:
        return pd.Series(index=base_df.index, dtype="float64", name="cnn_fear_greed")

    # Equal-weight 7-component blend with daily re-normalization for missing values.
    weights = {
        "market_volatility": 1 / 7,
        "nasdaq_volatility": 1 / 7,
        "market_momentum": 1 / 7,
        "price_strength_proxy": 1 / 7,
        "junk_bond_demand": 1 / 7,
        "safe_haven_demand": 1 / 7,
        "risk_appetite": 1 / 7,
    }

    comp_df = pd.concat(components.values(), axis=1)
    comp_df.columns = list(components.keys())
    weight_s = pd.Series(weights, dtype="float64")
    weight_s = weight_s.reindex(comp_df.columns).fillna(0.0)

    weighted_sum = comp_df.mul(weight_s, axis=1).sum(axis=1, skipna=True)
    available_weight = comp_df.notna().mul(weight_s, axis=1).sum(axis=1)
    min_components = 4
    score = weighted_sum / available_weight.replace(0.0, pd.NA)
    score = score.where(comp_df.notna().sum(axis=1) >= min_components)
    score = score.clip(0, 100)
    score.name = "cnn_fear_greed"
    return score


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

    # 2) Yahoo Finance indices (MOVE/SKEW/DXY)
    yf_idx = fetch_yfinance_close(YF_TICKERS, start, end)
    helper_px = fetch_yfinance_close(PROXY_HELPER_TICKERS, start, end)

    # 3) Optional OFR (stub)
    ofr = fetch_ofr_fsi_stub()

    # 4) Merge + feature engineering
    df = pd.concat([fred, yf_idx, ofr], axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    df = add_curve_features(df)
    df["cnn_fear_greed"] = build_cnn_fear_greed_proxy(df, helper_px)

    # Keep a weekday-only calendar:
    # if a source reports on weekend dates, shift to prior Friday.
    weekend_mask = df.index.dayofweek >= 5
    if weekend_mask.any():
        shifted_index = df.index.copy()
        shifted_index = shifted_index.where(~weekend_mask, shifted_index - pd.to_timedelta(shifted_index.dayofweek - 4, unit="D"))
        df.index = pd.DatetimeIndex(shifted_index)
        df = df.sort_index()
        df = df.groupby(level=0).last()

    # Final guard: drop any residual weekend dates.
    df = df.loc[df.index.dayofweek < 5]

    # Optional: keep from 2007 onward even if some sources start later
    df = df.loc[pd.to_datetime(start):]

    return df


def main() -> None:
    _ensure_out_dir(OUT_CSV_COMBINED)
    _configure_yfinance_cache(CACHE_DIR)

    df_original = build_indicator_dataset()
    ordered_original_cols = [c for c in VOLATILITY_OUTPUT_ORDER if c in df_original.columns]
    remaining_original_cols = [c for c in df_original.columns if c not in ordered_original_cols]
    df_original = df_original[ordered_original_cols + remaining_original_cols]

    numeric_cols_original = df_original.select_dtypes(include="number").columns
    if len(numeric_cols_original) > 0:
        df_original.loc[:, numeric_cols_original] = df_original.loc[:, numeric_cols_original].round(ROUND_DIGITS)

    # Save
    df_original.to_csv(OUT_CSV_COMBINED, index_label="date")

    print(f"[OK] Wrote {len(df_original):,} rows x {df_original.shape[1]:,} cols")
    print(f"     Combined CSV: {OUT_CSV_COMBINED}")


if __name__ == "__main__":
    main()
