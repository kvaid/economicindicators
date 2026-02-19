"""
Download macroeconomic indicator data (Fed rate, core PCE, and unemployment)
into a single CSV: data/macroindicators.csv.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

import pandas as pd
from fredapi import Fred

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_FILE = DATA_DIR / "macroindicators.csv"
ERROR_FILE = DATA_DIR / "macro_download_errors.csv"

TARGET_SERIES_CUTOFF = pd.Timestamp("2008-12-16")

FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
YOY_ANCHOR_START = "1989-01-01"
YOY_ANCHOR_END = "1990-12-31"
INFLATION_SERIES = "PCEPILFE"
UNRATE_SERIES = "UNRATE"
U6RATE_SERIES = "U6RATE"
NROU_SERIES = "NROU"
FED_RATE_SERIES = "DFEDTAR"
FED_RATE_ALT_SERIES = "DFEDTARU"
SOFR_SERIES = "SOFR"
START_DATE_FULL = "2000-01-01"


def _needs_full_backfill(existing: pd.DataFrame, value_col: str) -> bool:
    if existing.empty or value_col not in existing.columns:
        return True
    s = pd.to_numeric(existing[value_col], errors="coerce")
    if s.dropna().empty:
        return True
    first_idx = s.first_valid_index()
    if first_idx is None:
        return True
    first_date = pd.to_datetime(existing.loc[first_idx, "date"], errors="coerce")
    if pd.isna(first_date):
        return True
    return first_date > pd.to_datetime(START_DATE_FULL)


def get_fred_api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing FRED_API_KEY environment variable.")
    return key


def load_existing() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        return df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    except Exception:
        return pd.DataFrame()


def fetch_fred_series(
    fred_client: Fred, series_id: str, out_col: str, start_date: str, end_date: str
) -> pd.DataFrame:
    series = fred_client.get_series(
        series_id,
        observation_start=start_date,
        observation_end=end_date,
    )
    if series is None or series.empty:
        return pd.DataFrame(columns=["observation_date", out_col])
    df = series.rename(out_col).to_frame()
    df.index.name = "observation_date"
    return df.reset_index()


def download_cpi_data(existing: pd.DataFrame) -> pd.DataFrame | None:
    if _needs_full_backfill(existing, "PCE"):
        fetch_start = START_DATE_FULL
        print("Full mode: no existing inflation history found.")
    else:
        last_date = existing["date"].max()
        fetch_start = (last_date - pd.DateOffset(months=24)).strftime("%Y-%m-%d")
        print(f"Incremental mode: inflation last date {last_date.date()}, fetch start {fetch_start}")

    fetch_end = datetime.now().strftime("%Y-%m-%d")
    pce_url = f"{FRED_BASE_URL}?id={INFLATION_SERIES}&cosd={fetch_start}&coed={fetch_end}"
    anchor_url = f"{FRED_BASE_URL}?id={INFLATION_SERIES}&cosd={YOY_ANCHOR_START}&coed={YOY_ANCHOR_END}"

    df = pd.read_csv(pce_url)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.rename(columns={"observation_date": "date", INFLATION_SERIES: "PCE"})
    df["PCE"] = pd.to_numeric(df["PCE"], errors="coerce")
    new_pce = df[["date", "PCE"]].drop_duplicates(subset=["date"], keep="last").set_index("date")
    new_pce = new_pce.sort_index().resample("MS").last()

    anchor_df = pd.read_csv(anchor_url)
    anchor_df["observation_date"] = pd.to_datetime(anchor_df["observation_date"])
    anchor_df = anchor_df.rename(columns={"observation_date": "date", INFLATION_SERIES: "PCE"})
    anchor_df["PCE"] = pd.to_numeric(anchor_df["PCE"], errors="coerce")
    anchor_pce = anchor_df[["date", "PCE"]].drop_duplicates(subset=["date"], keep="last").set_index("date")
    anchor_pce = anchor_pce.sort_index().resample("MS").last()

    if not existing.empty and {"PCE"}.issubset(existing.columns):
        existing_pce = existing[["date", "PCE"]].drop_duplicates(subset=["date"], keep="last").set_index("date")
        # Existing macro CSV is daily-aligned; convert back to monthly points for correct YoY math.
        existing_pce = existing_pce.sort_index().resample("MS").last()
        # Do not let synthetic forward-filled months from existing daily data
        # extend beyond the latest true FRED observation month.
        if not new_pce.empty:
            existing_pce = existing_pce[existing_pce.index <= new_pce.index.max()]
        merged_pce = anchor_pce.combine_first(existing_pce)
        merged_pce = new_pce.combine_first(merged_pce)
    else:
        merged_pce = new_pce.combine_first(anchor_pce)

    merged_pce = merged_pce.sort_index()
    merged_pce["PCE"] = pd.to_numeric(merged_pce["PCE"], errors="coerce")
    merged_pce["PCE_YoY"] = (merged_pce["PCE"].pct_change(periods=12) * 100).round(2)
    merged_pce = merged_pce.reset_index()
    merged_pce = merged_pce[merged_pce["date"] >= START_DATE_FULL]
    return merged_pce[["date", "PCE", "PCE_YoY"]].reset_index(drop=True)


def download_unemployment_data(existing: pd.DataFrame) -> pd.DataFrame | None:
    fetch_end = datetime.now().strftime("%Y-%m-%d")
    if _needs_full_backfill(existing, UNRATE_SERIES):
        fetch_start = START_DATE_FULL
        print("Full mode: no existing unemployment history found.")
    else:
        last_date = existing["date"].max()
        fetch_start = (last_date - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
        print(f"Incremental mode: unemployment last date {last_date.date()}, fetch start {fetch_start}")

    def _download_series(series_id: str) -> pd.DataFrame:
        url = f"{FRED_BASE_URL}?id={series_id}&cosd={fetch_start}&coed={fetch_end}"
        df = pd.read_csv(url)
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        return df.rename(columns={"observation_date": "date", series_id: series_id}).set_index("date")

    unrate = _download_series(UNRATE_SERIES)
    u6rate = _download_series(U6RATE_SERIES)
    nrou = _download_series(NROU_SERIES)

    combined = unrate.join(u6rate, how="outer").join(nrou, how="outer").sort_index()
    for col in [UNRATE_SERIES, U6RATE_SERIES, NROU_SERIES]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Keep one monthly row and forward-fill NROU across months.
    combined = combined.resample("MS").last()
    combined["NROU"] = combined["NROU"].ffill()
    combined = combined.reset_index()
    combined = combined[["date", UNRATE_SERIES, U6RATE_SERIES, NROU_SERIES]]

    if not existing.empty:
        existing_u = existing.copy()
        # Backward compatibility: older or transitional files may use U3RATE.
        if "UNRATE" not in existing_u.columns and "U3RATE" in existing_u.columns:
            existing_u = existing_u.rename(columns={"U3RATE": "UNRATE"})
        for col in [UNRATE_SERIES, U6RATE_SERIES, NROU_SERIES]:
            if col not in existing_u.columns:
                existing_u[col] = pd.NA
        existing_idx = (
            existing_u[["date", UNRATE_SERIES, U6RATE_SERIES, NROU_SERIES]]
            .drop_duplicates(subset=["date"], keep="last")
            .set_index("date")
        )
        new_idx = combined.drop_duplicates(subset=["date"], keep="last").set_index("date")
        merged = new_idx.combine_first(existing_idx).sort_index().reset_index()
        return merged[["date", UNRATE_SERIES, U6RATE_SERIES, NROU_SERIES]].reset_index(drop=True)

    return combined.reset_index(drop=True)


def download_fed_rate_data() -> pd.DataFrame | None:
    fetch_start = START_DATE_FULL
    fetch_end = datetime.now().strftime("%Y-%m-%d")
    print(f"Fetching Fed and SOFR data from {fetch_start} to {fetch_end}")
    fred_client = Fred(api_key=get_fred_api_key())

    old_rate = fetch_fred_series(
        fred_client,
        FED_RATE_SERIES,
        "FED_RATE",
        fetch_start,
        fetch_end,
    ).set_index("observation_date")
    old_rate = old_rate[old_rate.index < TARGET_SERIES_CUTOFF]

    new_rate = fetch_fred_series(
        fred_client,
        FED_RATE_ALT_SERIES,
        "FED_RATE",
        fetch_start,
        fetch_end,
    ).set_index("observation_date")
    new_rate = new_rate[new_rate.index >= TARGET_SERIES_CUTOFF]

    fed = pd.concat([old_rate, new_rate], sort=False).sort_index().resample("D").ffill()
    if fed.empty:
        return None

    sofr = fetch_fred_series(fred_client, SOFR_SERIES, "SOFR", fetch_start, fetch_end).set_index("observation_date")
    if not sofr.empty:
        fed = fed.join(sofr[["SOFR"]], how="left")
        fed["SOFR"] = pd.to_numeric(fed["SOFR"], errors="coerce").ffill()

    fed = fed.reset_index()
    fed = fed.rename(columns={"observation_date": "date"})
    fed = fed[fed["date"] >= START_DATE_FULL]
    fed["FED_RATE"] = pd.to_numeric(fed["FED_RATE"], errors="coerce")
    if "SOFR" in fed.columns:
        fed["SOFR"] = pd.to_numeric(fed["SOFR"], errors="coerce")
    return fed[["date", "FED_RATE", "SOFR"]].reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_existing()

    errors: list[dict[str, str]] = []
    print(f"Downloading macro indicators into {OUTPUT_FILE}...")

    fed_df = None
    infl_df = None
    unemp_df = None

    try:
        fed_df = download_fed_rate_data()
    except Exception as exc:
        errors.append({"section": "fedrate", "error": str(exc)})

    try:
        infl_df = download_cpi_data(existing)
    except Exception as exc:
        errors.append({"section": "inflation", "error": str(exc)})

    try:
        unemp_df = download_unemployment_data(existing)
    except Exception as exc:
        errors.append({"section": "unemployment", "error": str(exc)})

    if fed_df is None and infl_df is None and unemp_df is None:
        raise RuntimeError("No macro indicator data was downloaded.")

    frames = []
    for frame in (fed_df, infl_df, unemp_df):
        if frame is not None and not frame.empty:
            frames.append(frame.set_index("date"))

    if not frames:
        raise RuntimeError("No macro indicator data to combine.")

    macro_df = pd.concat(frames, axis=1, join="outer", sort=False).sort_index()
    macro_df = macro_df[~macro_df.index.duplicated(keep="last")]

    numeric_cols = [col for col in macro_df.columns if col != "date"]
    macro_df[numeric_cols] = macro_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    # Forward-fill monthly macro series across the daily calendar.
    monthly_cols = [col for col in ["PCE", "PCE_YoY", "UNRATE", "U6RATE", "NROU"] if col in macro_df.columns]
    if monthly_cols:
        macro_df[monthly_cols] = macro_df[monthly_cols].ffill()
    macro_df = macro_df.round(2).reset_index()

    macro_df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d",
    )
    print(f"OK Saved {len(macro_df)} rows to {OUTPUT_FILE}")

    if errors:
        pd.DataFrame(errors).to_csv(ERROR_FILE, index=False)
        print(f"Wrote {len(errors)} section errors to {ERROR_FILE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
