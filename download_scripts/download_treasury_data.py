"""
Incrementally download US Treasury yield data from FRED and save to ust.csv.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

SERIES_IDS = {
    "BC_1YEAR": "DGS1",
    "BC_2YEAR": "DGS2",
    "BC_5YEAR": "DGS5",
    "BC_7YEAR": "DGS7",
    "BC_10YEAR": "DGS10",
    "BC_20YEAR": "DGS20",
    "BC_30YEAR": "DGS30",
}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
OUTPUT_FILE = DATA_DIR / "ust.csv"
OUTPUT_COLUMNS = [
    "date",
    "BC_1YEAR",
    "BC_2YEAR",
    "BC_5YEAR",
    "BC_7YEAR",
    "BC_10YEAR",
    "BC_20YEAR",
    "BC_30YEAR",
]


def load_existing() -> pd.DataFrame:
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[OUTPUT_COLUMNS]
    except Exception:
        return pd.DataFrame()


def enforce_daily_timeline(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty:
        return df
    value_cols = [c for c in df.columns if c != date_col]
    out = df.set_index(date_col).sort_index()
    full_index = pd.date_range(start=out.index.min(), end=out.index.max(), freq="D")
    out = out.reindex(full_index)
    out.index.name = date_col
    out[value_cols] = out[value_cols].ffill()
    return out.reset_index()


def download_treasury_data() -> pd.DataFrame | None:
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    existing = load_existing()

    if existing.empty:
        incremental_start = "1990-01-01"
        print("Full mode: no existing ust.csv found.")
    else:
        last_date = existing["date"].max()
        incremental_start = (last_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Incremental mode: last saved date {last_date.date()}, fetch start {incremental_start}")

    fetch_end = datetime.now().strftime("%Y-%m-%d")
    merged = existing.copy()
    if merged.empty:
        merged = pd.DataFrame(columns=["date"])

    for col_name, series_id in SERIES_IDS.items():
        needs_backfill = (
            existing.empty
            or col_name not in existing.columns
            or existing[col_name].notna().sum() == 0
        )
        series_start = "1990-01-01" if needs_backfill else incremental_start
        mode = "full backfill" if needs_backfill else "incremental"
        print(f"Downloading {col_name} ({series_id}) [{mode}]...")
        url = f"{base_url}?id={series_id}&cosd={series_start}&coed={fetch_end}"
        try:
            df = pd.read_csv(url)
            df["observation_date"] = pd.to_datetime(df["observation_date"])
            df = df.rename(columns={"observation_date": "DATE", series_id: col_name})
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            series_df = df.rename(columns={"DATE": "date"})[["date", col_name]]
            if col_name in merged.columns:
                merged = merged.merge(
                    series_df,
                    on="date",
                    how="outer",
                    suffixes=("", "__new"),
                )
                merged[col_name] = pd.to_numeric(merged[f"{col_name}__new"], errors="coerce").combine_first(
                    pd.to_numeric(merged[col_name], errors="coerce")
                )
                merged = merged.drop(columns=[f"{col_name}__new"])
            else:
                merged = merged.merge(series_df, on="date", how="outer")
            print(f"  OK Downloaded {len(df)} records")
        except Exception as exc:
            print(f"  ERROR downloading {series_id}: {exc}")

    if merged.empty:
        print("No data downloaded.")
        return None

    merged = merged.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    merged = enforce_daily_timeline(merged, date_col="date")
    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[OUTPUT_COLUMNS]
    print(f"Final dataset: {len(merged)} rows from {merged['date'].min()} to {merged['date'].max()}")
    return merged


if __name__ == "__main__":
    print("Downloading US Treasury Yield Data from FRED...\n")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    before_rows = len(load_existing())
    df = download_treasury_data()

    if df is not None:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nOK Data saved to {OUTPUT_FILE}")
        print(f"Rows added/updated: {len(df) - before_rows}")
    else:
        print("\nERROR Failed to download data")
