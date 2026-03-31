import pandas as pd
import sys
import io
import requests
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.db_manager import DatabaseManager

def sync_sp500():
    # Official State Street (SSGA) SPY Holdings URL
    # This is a verifiable institutional source
    ssga_url = "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
    
    try:
        # Headers to prevent 403 Forbidden (mimicking a browser request)
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(ssga_url, headers=headers)
        
        # Read Excel, skipping the SSGA header rows (usually first 4 rows)
        df = pd.read_excel(io.BytesIO(response.content), skiprows=4)
        
        # Clean the dataframe: keep only valid tickers
        df = df.dropna(subset=['Ticker'])
        
        db = DatabaseManager()
        with db.get_connection() as conn:
            for _, row in df.iterrows():
                # Alpaca uses '-' instead of '.' for share classes (e.g., BRK-B) [cite: 247]
                symbol = str(row['Ticker']).replace('.', '-')
                name = row['Name']
                sector = row['Sector'] if 'Sector' in row else "Unknown"
                
                conn.execute('''
                    INSERT OR IGNORE INTO tickers (symbol, company_name, sector)
                    VALUES (?, ?, ?)
                ''', (symbol, name, sector))
            conn.commit()
        print(f"Successfully synced {len(df)} tickers from State Street SPY Holdings.")
        
    except Exception as e:
        print(f"Error syncing from SSGA: {e}")

if __name__ == "__main__":
    sync_sp500()