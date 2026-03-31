import pandas as pd
import os
import time
import sqlite3
import pytz
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
DB_PATH = os.getenv('DB_PATH', '../data/stock_data.db')

BASE_URL = 'https://paper-api.alpaca.markets'

# timezone setup for the 16:00 cutoff logic
eastern = pytz.timezone('US/Eastern')
start_date = '2020-01-01'
end_date = '2025-12-31'

# strict data quality threshold defined in the research methodology
MIN_HEADLINES_THRESHOLD = 50

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_tickers_from_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM price_data")
    tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tickers

def calculate_effective_date(created_at):
    # alpaca returns a pandas Timestamp. we ensure it is utc timezone aware.
    dt_utc = pd.to_datetime(created_at)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.tz_localize('UTC')
        
    # convert to us/eastern market time
    dt_est = dt_utc.astimezone(eastern)
    
    effective_date = dt_est.date()
    
    # temporal alignment rule 1: shift post-market news to the next day
    if dt_est.hour >= 16:
        effective_date += timedelta(days=1)
        
    # temporal alignment rule 2: shift weekend news to monday
    while effective_date.weekday() > 4:
        effective_date += timedelta(days=1)
        
    return effective_date.strftime('%Y-%m-%d')



def fetch_news_data(api, ticker):
    try:
        # fetch historical news. limit=10000 ensures we get the full 5-year history if available.
        news_items = api.get_news(
            symbol=ticker, 
            start=start_date, 
            end=end_date, 
            limit=10000,
            include_content=False
        )
        
        # enforce the data quality threshold
        headline_count = len(news_items) if news_items else 0
        if headline_count < MIN_HEADLINES_THRESHOLD:
            print(f"Rejected {ticker}: insufficient news volume ({headline_count} headlines). Minimum required is {MIN_HEADLINES_THRESHOLD}.")
            return
            
        records = []
        for item in news_items:
            effective_date = calculate_effective_date(item.created_at)
            headline = item.headline
            summary = item.summary if item.summary else ""
            
            records.append((
                ticker, 
                effective_date, 
                headline, 
                summary
            ))
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.executemany('''
            INSERT OR IGNORE INTO news_data (ticker, published_at, headline, summary)
            VALUES (?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        conn.close()
        print(f"Inserted {len(records)} temporally aligned news records for {ticker}")
        
    except Exception as e:
        print(f"failed to fetch news for {ticker} error: {e}")

def run_news_extraction():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("API keys missing")
        return

    tickers = get_tickers_from_db()
    if not tickers:
        print("No tickers found in database")
        return

    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')
    
    print(f"starting news extraction and alignment for {len(tickers)} assets...")
    
    for count, ticker in enumerate(tickers, 1):
        print(f"[{count}/{len(tickers)}] processing {ticker}...", end=" ")
        fetch_news_data(api, ticker)
        time.sleep(1)
        
    print("News extraction pipeline complete")

if __name__ == '__main__':
    run_news_extraction()