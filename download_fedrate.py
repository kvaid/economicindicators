"""
Download Federal Reserve Interest Rate data from FRED
and save to fedrate.csv in the required format.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path

def download_fed_rate_data():
    """Download Federal Reserve interest rate data from FRED."""
    
    # FRED series IDs for Target Rate
    # DFEDTAR: Federal Funds Target Rate (Discontinued 2008-12-16)
    # DFEDTARU: Federal Funds Target Range - Upper Limit (Started 2008-12-16)
    
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    
    print("Downloading Federal Funds Target Rate history...")
    
    try:
        # Download DFEDTAR (Pre-2008)
        url_old = f"{base_url}?id=DFEDTAR"
        df_old = pd.read_csv(url_old)
        df_old['observation_date'] = pd.to_datetime(df_old['observation_date'])
        df_old = df_old.rename(columns={'DFEDTAR': 'FED_RATE'}).set_index('observation_date')
        
        # Download DFEDTARU (Post-2008)
        url_new = f"{base_url}?id=DFEDTARU"
        df_new = pd.read_csv(url_new)
        df_new['observation_date'] = pd.to_datetime(df_new['observation_date'])
        df_new = df_new.rename(columns={'DFEDTARU': 'FED_RATE'}).set_index('observation_date')
        
        # Combine them
        # Use combine_first to prefer df_new (upper limit) if overlap, but they shouldn't overlap much
        # Actually, let's just concatenate and sort
        df_combined = pd.concat([df_old, df_new]).sort_index()
        
        # Determine the rate for each month
        # We want the rate that was in effect for the month.
        # Since rates can change mid-month, taking the last value of the month is a good representation of where the month ended.
        # Or we can take the mode? No, last value is safer for "current status".
        # Ensure we cover the full range
        
        # forward fill to fill weekends/holidays if daily data has gaps
        df_combined = df_combined.resample('D').ffill()
        
        # Resample to month end, taking the last valid observation
        df_monthly = df_combined.resample('ME').last().reset_index()
        
        df_monthly = df_monthly.rename(columns={'observation_date': 'date'})
        
        # Convert to numeric, handling '.' as NaN
        df_monthly['FED_RATE'] = pd.to_numeric(df_monthly['FED_RATE'], errors='coerce')
        
        # Filter from 1990 onwards for our app
        df_monthly = df_monthly[df_monthly['date'] >= '1990-01-01']
        
        print(f"  OK Downloaded and combined {len(df_monthly)} monthly records")
        
        return df_monthly
        
    except Exception as e:
        print(f"  ERROR downloading data: {e}")
        return None

if __name__ == "__main__":
    print("Downloading Federal Reserve Interest Rate Data from FRED...\n")
    
    # Needs pandas version compatible with ME offset, or fallback to M
    try:
        pd.tseries.offsets.MonthEnd()
    except:
        print("Warning: Pandas version might be old, check resample compatibility.")

    df = download_fed_rate_data()
    
    if df is not None:
        # Save to CSV
        output_file = Path(__file__).resolve().parent / "fedrate.csv"
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
        missing = df['FED_RATE'].isna().sum()
        pct = (missing / len(df)) * 100
        print(f"  Missing values: {missing} ({pct:.1f}%)")
    else:
        print("\nERROR Failed to download data")
