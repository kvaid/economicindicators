"""
Incrementally download Core PCE from FRED, compute YoY inflation, and save to inflation.csv.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_FILE = Path(__file__).resolve().parent / "inflation.csv"
SERIES_ID = "PCEPILFE"
YOY_ANCHOR_START = "1989-01-01"
YOY_ANCHOR_END = "1990-12-31"


def load_existing() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        return df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    except Exception:
        return pd.DataFrame()


def download_cpi_data() -> pd.DataFrame | None:
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    existing = load_existing()

    if existing.empty:
        fetch_start = "1989-01-01"
        print("Full mode: no existing inflation.csv found.")
    else:
        last_date = existing["date"].max()
        # YoY needs prior 12 months. Keep a wider overlap to avoid boundary NaNs.
        fetch_start = (last_date - pd.DateOffset(months=24)).strftime("%Y-%m-%d")
        print(f"Incremental mode: last saved date {last_date.date()}, fetch start {fetch_start}")

    fetch_end = datetime.now().strftime("%Y-%m-%d")
    url = f"{base_url}?id={SERIES_ID}&cosd={fetch_start}&coed={fetch_end}"
    anchor_url = f"{base_url}?id={SERIES_ID}&cosd={YOY_ANCHOR_START}&coed={YOY_ANCHOR_END}"

    try:
        df = pd.read_csv(url)
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df = df.rename(columns={"observation_date": "date", SERIES_ID: "PCE"})
        df["PCE"] = pd.to_numeric(df["PCE"], errors="coerce")
        new_pce = df[["date", "PCE"]].drop_duplicates(subset=["date"], keep="last").set_index("date")

        anchor_df = pd.read_csv(anchor_url)
        anchor_df["observation_date"] = pd.to_datetime(anchor_df["observation_date"])
        anchor_df = anchor_df.rename(columns={"observation_date": "date", SERIES_ID: "PCE"})
        anchor_df["PCE"] = pd.to_numeric(anchor_df["PCE"], errors="coerce")
        anchor_pce = anchor_df[["date", "PCE"]].drop_duplicates(subset=["date"], keep="last").set_index("date")

        if not existing.empty:
            existing_pce = existing[["date", "PCE"]].drop_duplicates(subset=["date"], keep="last").set_index("date")
            merged_pce = anchor_pce.combine_first(existing_pce)
            # Prefer fresh non-null values, but never overwrite with null.
            merged_pce = new_pce.combine_first(merged_pce)
        else:
            merged_pce = new_pce.combine_first(anchor_pce)

        merged_pce = merged_pce.sort_index()
        merged_pce["PCE"] = pd.to_numeric(merged_pce["PCE"], errors="coerce")
        merged_pce["PCE_YoY"] = (merged_pce["PCE"].pct_change(periods=12) * 100).round(2)
        merged_pce = merged_pce.reset_index()
        merged_pce = merged_pce[merged_pce["date"] >= "1990-01-01"]
        return merged_pce[["date", "PCE", "PCE_YoY"]].reset_index(drop=True)
    except Exception as exc:
        print(f"ERROR downloading {SERIES_ID}: {exc}")
        return None


if __name__ == "__main__":
    print("Downloading Core PCE Inflation Data from FRED...\n")
    before_rows = len(load_existing())
    df = download_cpi_data()

    if df is not None:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nOK Data saved to {OUTPUT_FILE}")
        print(f"Rows added/updated: {len(df) - before_rows}")
    else:
        print("\nERROR Failed to download data")
