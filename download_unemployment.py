"""
Download Unemployment Rate (U-3) data from FRED (BLS source)
and save to unemployment.csv.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path

def download_unemployment_data():
    """Download U-3 Unemployment Rate and Natural Rate of Unemployment."""
    
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    start_date = '1990-01-01'
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    print("Downloading Unemployment Data...")
    
    try:
        # 1. Download UNRATE (Monthly)
        print("  Fetching UNRATE (U-3)...")
        url_unrate = f"{base_url}?id=UNRATE&cosd={start_date}&coed={end_date}"
        df_unrate = pd.read_csv(url_unrate)
        df_unrate['observation_date'] = pd.to_datetime(df_unrate['observation_date'])
        df_unrate = df_unrate.rename(columns={'observation_date': 'date', 'UNRATE': 'UNRATE'}).set_index('date')
        
        # 2. Download NROU (Natural Rate of Unemployment - Quarterly)
        print("  Fetching NROU (Natural Rate)...")
        url_nrou = f"{base_url}?id=NROU&cosd={start_date}&coed={end_date}"
        df_nrou = pd.read_csv(url_nrou)
        df_nrou['observation_date'] = pd.to_datetime(df_nrou['observation_date'])
        df_nrou = df_nrou.rename(columns={'observation_date': 'date', 'NROU': 'NROU'}).set_index('date')
        
        # 3. Combine and Resample
        # NROU is quarterly, so we join it to the monthly UNRATE and forward fill the NROU values
        # to apply the quarterly estimate to each month in that quarter.
        
        # Merge on index (date)
        # using 'left' join to keep UNRATE's monthly timeline
        df_combined = df_unrate.join(df_nrou, how='outer') # Outer to catch everything first
        
        # Resample to daily to fill gaps if any, then back to monthly?
        # Actually, simpler: just forward fill NROU.
        df_combined['NROU'] = df_combined['NROU'].ffill()
        
        # Filter back to just the months we have UNRATE for, or resample to MS (Month Start)
        df_combined = df_combined.resample('MS').first() # Take the first value (which is the data value)
        
        # Reset index to make date a column
        df_combined = df_combined.reset_index()
        
        # Ensure we filter to our start date again just in case
        df_combined = df_combined[df_combined['date'] >= start_date]

        # Convert cols to numeric
        df_combined['UNRATE'] = pd.to_numeric(df_combined['UNRATE'], errors='coerce')
        df_combined['NROU'] = pd.to_numeric(df_combined['NROU'], errors='coerce')
        
        print(f"  OK Combined {len(df_combined)} records")
        
        return df_combined[['date', 'UNRATE', 'NROU']]
        
    except Exception as e:
        print(f"  ERROR downloading data: {e}")
        return None

if __name__ == "__main__":
    print("Downloading Unemployment Rate Data from FRED...\n")
    
    df = download_unemployment_data()
    
    if df is not None:
        output_file = Path(__file__).resolve().parent / "unemployment.csv"
        df.to_csv(output_file, index=False)
        print(f"\nOK Data saved to {output_file}")
        
        print("\nData preview:")
        print(df.head())
        print("...")
        print(df.tail())
    else:
        print("\nERROR Failed to download data")
