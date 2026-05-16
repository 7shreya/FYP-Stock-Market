import time
import pandas as pd
import requests
import io
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame  
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
import src.config as config
from src.database_manager import DatabaseManager

class DataIngestor:
    def __init__(self):
        self.history_client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        self.trading_client = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        self.db = DatabaseManager()
        self.db.initialize_db()

    def get_sp500_tickers(self):
        """Retrieves tickers from iShares institutional source."""
        response = requests.get(config.SP500_URL)
        df = pd.read_csv(io.StringIO(response.text), skiprows=9)
        tickers = df['Ticker'].dropna().unique().tolist()
        return [t.replace('.', '-') for t in tickers if isinstance(t, str)]

    def fetch_and_store_prices(self, tickers):
        """Downloads price data and saves to SQLite."""
        chunk_size = 50
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i + chunk_size]
            request_params = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day, 
                start=config.TRAIN_START_DATE,
                end=config.TRAIN_END_DATE,
                adjustment='all'
            )
            try:
                bars = self.history_client.get_stock_bars(request_params).df
                bars = bars.reset_index().rename(columns={'symbol': 'ticker'})
                self.db.save_prices(bars)
                print(f"Ingested: {chunk[0]} and batch...")
                time.sleep(1.0) 
            except Exception as e:
                print(f"Error in batch {i}: {e}")

if __name__ == "__main__":
    ingestor = DataIngestor()
    tickers = ingestor.get_sp500_tickers()
    ingestor.fetch_and_store_prices(tickers)