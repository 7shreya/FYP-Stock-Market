import sqlite3
import sys
from pathlib import Path

# Fix for module loading: Ensure root is in the path
sys.path.append(str(Path(__file__).parent.parent))
from config import DB_PATH

class DatabaseManager:
    def __init__(self):
        self.db_path = DB_PATH
        self.initialize_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row 
        return conn

    def initialize_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Tickers Metadata [cite: 312]
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tickers (
                    symbol TEXT PRIMARY KEY,
                    company_name TEXT,
                    sector TEXT
                )
            ''')

            # OHLCV + Log Returns (Stationarity Fix) [cite: 245, 299]
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    timestamp DATETIME,
                    symbol TEXT,
                    open REAL, high REAL, low REAL, close REAL, volume INTEGER,
                    log_return REAL,
                    PRIMARY KEY (timestamp, symbol),
                    FOREIGN KEY (symbol) REFERENCES tickers (symbol)
                )
            ''')

            # 3-Vector Sentiment (Information Loss Fix) [cite: 97, 323, 380]
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    timestamp DATETIME,
                    symbol TEXT,
                    prob_positive REAL, prob_negative REAL, prob_neutral REAL,
                    headline_count INTEGER,
                    PRIMARY KEY (timestamp, symbol),
                    FOREIGN KEY (symbol) REFERENCES tickers (symbol)
                )
            ''')
            
            # Optimization Indexes [cite: 312]
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_mkt_sym ON market_data (symbol)')
            print(f"DATABASE INITIALIZED AT: {self.db_path}")

if __name__ == "__main__":
    DatabaseManager()