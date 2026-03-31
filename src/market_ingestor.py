import time
import sqlite3
import pandas as pd
import requests
import io
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import src.market_config as config
from src.market_db import MarketDB

class MarketIngestor:
    def __init__(self):
        """
        Initializes the Alpaca Client and ensures the database 
        tables are ready before starting ingestion.
        """
        self.client = StockHistoricalDataClient(config.ALPACA_KEY, config.ALPACA_SECRET)
        self.db_manager = MarketDB()
        self.db_manager.create_tables()

    def get_sp500_list(self):
        """
        Retrieves the S&P 500 tickers from iShares (BlackRock).
        Cleans ticker symbols for Alpaca compatibility (e.g., BRK.B -> BRK-B).
        """
        print("Connecting to iShares for the S&P 500 constituent list...")
        response = requests.get(config.SP500_URL)
        
        # Skip the first 9 rows of iShares metadata to reach the CSV header
        df = pd.read_csv(io.StringIO(response.text), skiprows=9)
        
        # Extract and clean tickers
        tickers = df['Ticker'].dropna().unique().tolist()
        clean_tickers = [str(t).strip().replace('.', '-') for t in tickers]
        
        print(f"Successfully retrieved {len(clean_tickers)} unique tickers.")
        return clean_tickers

    def download_prices(self, tickers):
        """
        Downloads 5 years of daily OHLCV data in chunks of 50 tickers.
        Includes trade_count and vwap to match the updated database schema.
        """
        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            print(f"--- Processing Batch {i//chunk_size + 1}: {chunk[0]} to {chunk[-1]} ---")

            request_params = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=config.START_DATE,
                end=config.END_DATE,
                adjustment='all' # Adjusts for stock splits and dividends
            )

            try:
                # Fetch historical bars from Alpaca
                bars_response = self.client.get_stock_bars(request_params)
                bars = bars_response.df
                
                # Format the DataFrame for the SQLite 'price_history' table
                bars = bars.reset_index()
                bars.rename(columns={'symbol': 'ticker'}, inplace=True)
                
                # Define exactly what we want to save (matching market_db.py)
                cols_to_save = [
                    'ticker', 'timestamp', 'open', 'high', 'low', 
                    'close', 'volume', 'trade_count', 'vwap'
                ]
                
                # Ensure we only try to save columns that actually exist in the response
                final_df = bars[[c for c in cols_to_save if c in bars.columns]]
                
                # Save to the market_data.db
                with sqlite3.connect(config.DB_PATH) as conn:
                    final_df.to_sql('price_history', conn, if_exists='append', index=False)
                
                print(f"✓ Saved {len(final_df)} records for this batch.")
                
                # Sleep to respect Alpaca's free-tier rate limits (approx 200/min)
                time.sleep(1.2)
                
            except Exception as e:
                print(f"Error processing batch starting with {chunk[0]}: {e}")
                # Brief pause before the next attempt
                time.sleep(2.0)

if __name__ == "__main__":
    ingestor = MarketIngestor()
    
    # 1. Fetch the S&P 500 list
    tickers = ingestor.get_sp500_list()
    
    # 2. Start the download process
    print(f"Commencing data ingestion into {config.DB_PATH}...")
    start_time = time.time()
    
    ingestor.download_prices(tickers)
    
    duration = (time.time() - start_time) / 60
    print(f"\n--- Ingestion Complete ---")
    print(f"Total time: {duration:.2f} minutes")
                
