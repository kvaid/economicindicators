"""
Download CPI (Consumer Price Index) data from FRED
and calculate Year-over-Year inflation rate.
Save to inflation.csv.
"""
import pandas as pd
from datetime import datetime
from pathlib import Path

def download_cpi_data():
    """Download Core PCE data and calculate inflation."""
    
    # FRED series ID for Core PCE (Personal Consumption Expenditures Excluding Food and Energy)
    series_id = 'PCEPILFE'
    
    # We need data starting earlier to calculate 1-year change for 1990
    start_date = '1989-01-01'
    
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    
    print(f"Downloading Core PCE data ({series_id})...")
    
    url = f"{base_url}?id={series_id}&cosd={start_date}&coed={datetime.now().strftime('%Y-%m-%d')}"
    
    try:
        df = pd.read_csv(url)
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        df = df.rename(columns={'observation_date': 'date', series_id: 'PCE'})
        
        # Calculate Year-over-Year Inflation Rate (%)
        # formula: ((current_index / index_12_months_ago) - 1) * 100
        df['PCE_YoY'] = df['PCE'].pct_change(periods=12) * 100
        
        # Round to 2 decimal places
        df['PCE_YoY'] = df['PCE_YoY'].round(2)
        
        # Filter to start from 1990-01-01 match other data
        df = df[df['date'] >= '1990-01-01']
        
        print(f"  OK Downloaded and processed {len(df)} monthly records")
        
        return df[['date', 'PCE', 'PCE_YoY']]
        
    except Exception as e:
        print(f"  ERROR downloading {series_id}: {e}")
        return None

if __name__ == "__main__":
    print("Downloading CPI Inflation Data from FRED...\n")
    
    df = download_cpi_data()
    
    if df is not None:
        output_file = Path(__file__).resolve().parent / "inflation.csv"
        df.to_csv(output_file, index=False)
        print(f"\nOK Data saved to {output_file}")
        
        print("\nData preview:")
        print(df.head())
        print("...")
        print(df.tail())
    else:
        print("\nERROR Failed to download data")
