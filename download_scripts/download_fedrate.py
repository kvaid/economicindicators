"""
Download Federal Reserve target rate data using fredapi and save to fedrate.csv.
Builds a full authoritative monthly history each run for data integrity.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd
from fredapi import Fred

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_FILE = DATA_DIR / "fedrate.csv"
TARGET_SERIES_CUTOFF = pd.Timestamp("2008-12-16")
FRED_API_KEY = "69da3d502e36febadb1d149b360b8464"


def load_existing() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        return df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    except Exception:
        return pd.DataFrame()


def fetch_fred_series(
    fred_client: Fred, series_id: str, start_date: str, end_date: str
) -> pd.DataFrame:
    series = fred_client.get_series(
        series_id,
        observation_start=start_date,
        observation_end=end_date,
    )
    if series is None or series.empty:
        return pd.DataFrame(columns=["observation_date", "FED_RATE"])
    df = series.rename("FED_RATE").to_frame()
    df.index.name = "observation_date"
    return df.reset_index()


def download_fed_rate_data() -> pd.DataFrame | None:
    fetch_start = "1990-01-01"
    fetch_end = datetime.now().strftime("%Y-%m-%d")
    print(f"Full rebuild mode: fetch range {fetch_start} to {fetch_end}")

    try:
        fred_client = Fred(api_key=FRED_API_KEY)

        df_old = fetch_fred_series(fred_client, "DFEDTAR", fetch_start, fetch_end).set_index("observation_date")
        df_old = df_old[df_old.index < TARGET_SERIES_CUTOFF]

        df_new = fetch_fred_series(fred_client, "DFEDTARU", fetch_start, fetch_end).set_index("observation_date")
        df_new = df_new[df_new.index >= TARGET_SERIES_CUTOFF]

        combined = pd.concat([df_old, df_new]).sort_index()
        if combined.empty:
            return None

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
