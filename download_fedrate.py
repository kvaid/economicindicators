"""
Download Federal Reserve target rate data from FRED and save to fedrate.csv.
Builds a full authoritative monthly history each run for data integrity.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_FILE = Path(__file__).resolve().parent / "fedrate.csv"
TARGET_SERIES_CUTOFF = pd.Timestamp("2008-12-16")


def load_existing() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        return df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    except Exception:
        return pd.DataFrame()


def download_fed_rate_data() -> pd.DataFrame | None:
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    fetch_start = "1990-01-01"
    fetch_end = datetime.now().strftime("%Y-%m-%d")
    print(f"Full rebuild mode: fetch range {fetch_start} to {fetch_end}")

    try:
        url_old = f"{base_url}?id=DFEDTAR&cosd={fetch_start}&coed={fetch_end}"
        df_old = pd.read_csv(url_old)
        df_old["observation_date"] = pd.to_datetime(df_old["observation_date"])
        df_old = df_old.rename(columns={"DFEDTAR": "FED_RATE"}).set_index("observation_date")
        df_old["FED_RATE"] = pd.to_numeric(df_old["FED_RATE"], errors="coerce")
        df_old = df_old[df_old.index < TARGET_SERIES_CUTOFF]

        url_new = f"{base_url}?id=DFEDTARU&cosd={fetch_start}&coed={fetch_end}"
        df_new = pd.read_csv(url_new)
        df_new["observation_date"] = pd.to_datetime(df_new["observation_date"])
        df_new = df_new.rename(columns={"DFEDTARU": "FED_RATE"}).set_index("observation_date")
        df_new["FED_RATE"] = pd.to_numeric(df_new["FED_RATE"], errors="coerce")
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
    before_rows = len(load_existing())
    df = download_fed_rate_data()

    if df is not None:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nOK Data saved to {OUTPUT_FILE}")
        print(f"Rows added/updated: {len(df) - before_rows}")
    else:
        print("\nERROR Failed to download data")
