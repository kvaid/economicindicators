"""
Download Federal Reserve target rate data using fredapi and save to fedrate.csv.
Builds a full authoritative monthly history each run for data integrity.
"""
from datetime import datetime
import os
from pathlib import Path

import pandas as pd
from fredapi import Fred

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_FILE = DATA_DIR / "fedrate.csv"
TARGET_SERIES_CUTOFF = pd.Timestamp("2008-12-16")

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


def download_fed_rate_data() -> pd.DataFrame | None:
    fetch_start = "1990-01-01"
    fetch_end = datetime.now().strftime("%Y-%m-%d")
    print(f"Full rebuild mode: fetch range {fetch_start} to {fetch_end}")

    try:
        fred_client = Fred(api_key=get_fred_api_key())

        df_old = fetch_fred_series(fred_client, "DFEDTAR", "FED_RATE", fetch_start, fetch_end).set_index("observation_date")
        df_old = df_old[df_old.index < TARGET_SERIES_CUTOFF]

        df_new = fetch_fred_series(fred_client, "DFEDTARU", "FED_RATE", fetch_start, fetch_end).set_index("observation_date")
        df_new = df_new[df_new.index >= TARGET_SERIES_CUTOFF]

        combined = pd.concat([df_old, df_new]).sort_index()
        if combined.empty:
            return None

        sofr = fetch_fred_series(fred_client, "SOFR30DAYAVG", "SOFR", fetch_start, fetch_end).set_index("observation_date")
        if not sofr.empty:
            combined = combined.join(sofr[["SOFR"]], how="outer")

        combined = combined.resample("D").ffill()
        monthly = combined.resample("ME").last().reset_index()
        monthly = monthly.rename(columns={"observation_date": "date"})
        monthly = monthly[monthly["date"] >= "1990-01-01"]

        print(f"OK Built {len(monthly)} monthly records")
        return monthly.reset_index(drop=True)
    except Exception as exc:
        print(f"ERROR downloading data: {exc}")
        return None


if __name__ == "__main__":
    print("Downloading Federal Reserve Interest Rate Data from FRED...\n")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    before_rows = len(load_existing())
    df = download_fed_rate_data()

    if df is not None:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nOK Data saved to {OUTPUT_FILE}")
        print(f"Rows added/updated: {len(df) - before_rows}")
    else:
        print("\nERROR Failed to download data")
