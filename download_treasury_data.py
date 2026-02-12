"""
Download US Treasury yield data from FRED (Federal Reserve Economic Data)
and save to ust.csv in the required format.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path

# FRED series IDs for Treasury Constant Maturity rates
# Using daily series and will resample to weekly
SERIES_IDS = {
    'BC_1YEAR': 'DGS1',    # 1-Year Treasury Constant Maturity Rate
    'BC_2YEAR': 'DGS2',    # 2-Year Treasury Constant Maturity Rate
    'BC_5YEAR': 'DGS5',    # 5-Year Treasury Constant Maturity Rate
    'BC_10YEAR': 'DGS10',  # 10-Year Treasury Constant Maturity Rate
    'BC_30YEAR': 'DGS30',  # 30-Year Treasury Constant Maturity Rate
}

def download_treasury_data():
    """Download Treasury yield data from FRED."""
    
    # FRED provides CSV downloads directly without API key for public data
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    
    all_data = []
    
    for col_name, series_id in SERIES_IDS.items():
        print(f"Downloading {col_name} ({series_id})...")
        
        # Construct URL for CSV download
        url = f"{base_url}?id={series_id}&cosd=1990-01-01&coed={datetime.now().strftime('%Y-%m-%d')}"
        
        try:
            # Read the CSV directly from FRED
            df = pd.read_csv(url)
            
            # FRED uses 'observation_date' as column name, parse it after reading
            df['observation_date'] = pd.to_datetime(df['observation_date'])
            df = df.rename(columns={'observation_date': 'DATE', series_id: col_name})
            
            # Convert to numeric, handling '.' as NaN
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            
            all_data.append(df)
            print(f"  OK Downloaded {len(df)} records")
            
        except Exception as e:
            print(f"  ERROR downloading {series_id}: {e}")
            continue
    
    if not all_data:
        print("No data downloaded!")
        return None
    
    # Merge all series on DATE
    print("\nMerging data...")
    merged = all_data[0]
    for df in all_data[1:]:
        merged = merged.merge(df, on='DATE', how='outer')
    
    # Sort by date
    merged = merged.sort_values('DATE').reset_index(drop=True)

    # Ensure every row is a consecutive calendar date, then fill missing
    # weekend/holiday rates from the previous available business day.
    rate_cols = [c for c in merged.columns if c != 'DATE']
    merged = merged.set_index('DATE')
    full_daily_index = pd.date_range(start=merged.index.min(), end=merged.index.max(), freq='D')
    merged = merged.reindex(full_daily_index)
    merged.index.name = 'DATE'
    merged[rate_cols] = merged[rate_cols].ffill()
    merged = merged.reset_index()
    
    # Rename DATE to date for consistency with app
    merged = merged.rename(columns={'DATE': 'date'})
    
    print(f"\nFinal dataset: {len(merged)} rows from {merged['date'].min()} to {merged['date'].max()}")
    print(f"Columns: {list(merged.columns)}")
    
    return merged

if __name__ == "__main__":
    print("Downloading US Treasury Yield Data from FRED...\n")
    
    df = download_treasury_data()
    
    if df is not None:
        # Save to CSV
        output_file = Path(__file__).resolve().parent / "ust.csv"
        df.to_csv(output_file, index=False)
        print(f"\nOK Data saved to {output_file}")
        
        # Show preview
        print("\nData preview:")
        print(df.head(10))
        print("\n...")
        print(df.tail(10))
        
        # Show data info
        print(f"\nData summary:")
        print(f"  Total rows: {len(df)}")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Missing values per column:")
        for col in df.columns:
            if col != 'date':
                missing = df[col].isna().sum()
                pct = (missing / len(df)) * 100
                print(f"    {col}: {missing} ({pct:.1f}%)")
    else:
        print("\nERROR Failed to download data")
