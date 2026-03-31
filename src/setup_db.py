import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

db_path = os.getenv('DB_PATH', '../data/stock_data.db')

def create_tables():
    print(f"initializing database at {db_path}")
    
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # price table for OHLCV and calculated returns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS price_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        trade_date DATE NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        log_return REAL,
        UNIQUE(ticker, trade_date)
    )
    ''')

    # news table to store raw text and finbert tensors
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS news_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        published_at DATETIME NOT NULL,
        headline TEXT NOT NULL,
        summary TEXT,
        sentiment_pos REAL,
        sentiment_neg REAL,
        sentiment_neu REAL,
        UNIQUE(ticker, published_at, headline)
    )
    ''')

    conn.commit()
    conn.close()
    print("Database tables created")

if __name__ == "__main__":
    create_tables()