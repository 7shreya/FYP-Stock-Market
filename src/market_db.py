import sqlite3
import os
from src import market_config as config

class MarketDB:
    def __init__(self):
        self.db_path = config.DB_PATH

    def create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Added trade_count and vwap to the schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    ticker TEXT,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    trade_count INTEGER,
                    vwap REAL,
                    PRIMARY KEY (ticker, timestamp)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_headlines (
                    ticker TEXT,
                    headline TEXT,
                    published_at DATETIME,
                    effective_trading_day DATE,
                    PRIMARY KEY (ticker, published_at, headline)
                )
            ''')
            conn.commit()
            print(f"Database schema updated at: {self.db_path}")

if __name__ == "__main__":
    db = MarketDB()
    db.create_tables()