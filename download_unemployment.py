"""
Incrementally download unemployment data from FRED and save to unemployment.csv.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_FILE = Path(__file__).resolve().parent / "unemployment.csv"


def load_existing() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        return df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    except Exception:
        return pd.DataFrame()


def _download_series(base_url: str, series_id: str, fetch_start: str, fetch_end: str) -> pd.DataFrame:
    url = f"{base_url}?id={series_id}&cosd={fetch_start}&coed={fetch_end}"
    df = pd.read_csv(url)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    return df.rename(columns={"observation_date": "date", series_id: series_id}).set_index("date")


def download_unemployment_data() -> pd.DataFrame | None:
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    fetch_end = datetime.now().strftime("%Y-%m-%d")
    existing = load_existing()

    if existing.empty:
        fetch_start = "1990-01-01"
        print("Full mode: no existing unemployment.csv found.")
    else:
        last_date = existing["date"].max()
        # Overlap protects quarterly NROU forward-fill transitions.
        fetch_start = (last_date - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
        print(f"Incremental mode: last saved date {last_date.date()}, fetch start {fetch_start}")

    try:
        print("Fetching UNRATE, U6RATE, and NROU...")
        unrate = _download_series(base_url, "UNRATE", fetch_start, fetch_end)
        u6rate = _download_series(base_url, "U6RATE", fetch_start, fetch_end)
        nrou = _download_series(base_url, "NROU", fetch_start, fetch_end)

        combined = unrate.join(u6rate, how="outer").join(nrou, how="outer").sort_index()
        for col in ["UNRATE", "U6RATE", "NROU"]:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

        # Keep one monthly row and then forward-fill NROU across months.
        combined = combined.resample("MS").last()
        combined["NROU"] = combined["NROU"].ffill()
        combined = combined.reset_index()
        combined = combined[combined["date"] >= "1990-01-01"]

        new_slice = combined[["date", "UNRATE", "U6RATE", "NROU"]]
        if not existing.empty:
            existing_idx = (
                existing[["date", "UNRATE", "U6RATE", "NROU"]]
                .drop_duplicates(subset=["date"], keep="last")
                .set_index("date")
            )
            new_idx = new_slice.drop_duplicates(subset=["date"], keep="last").set_index("date")
            # Prefer fresh non-null values, but keep existing values where new data is null.
            merged = new_idx.combine_first(existing_idx).sort_index().reset_index()
            return merged[["date", "UNRATE", "U6RATE", "NROU"]].reset_index(drop=True)
        return new_slice.reset_index(drop=True)
    except Exception as exc:
        print(f"ERROR downloading data: {exc}")
        return None


if __name__ == "__main__":
    print("Downloading Unemployment Data from FRED...\n")
    before_rows = len(load_existing())
    df = download_unemployment_data()

    if df is not None:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nOK Data saved to {OUTPUT_FILE}")
        print(f"Rows added/updated: {len(df) - before_rows}")
    else:
        print("\nERROR Failed to download data")
