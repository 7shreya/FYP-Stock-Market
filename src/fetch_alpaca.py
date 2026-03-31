import os
import time
import sqlite3
import requests
import pandas as pd
from io import StringIO
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
DB_PATH = os.getenv('DB_PATH', '../data/stock_data.db')

BASE_URL = 'https://paper-api.alpaca.markets'

start_date = '2020-01-01'
end_date = '2025-12-31'

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_sp100_tickers():
    print("Requesting S&P 100 constituents from BlackRock iShares...")
    
    # robust browser user-agent to bypass standard 403 blocks
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    url = "https://www.ishares.com/us/products/239707/ishares-sp-100-etf/1467271812596.ajax?fileType=csv&fileName=OEF_holdings&dataType=fund"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # skip the legal boilerplate at the top of the csv
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data, skiprows=9)
        
        # clean the dataframe
        df = df.dropna(subset=['Ticker'])
        tickers = df['Ticker'].tolist()
        
        clean_tickers = []
        for ticker in tickers:
            # filter out cash, usd equivalents, or derivatives
            if ticker in ['USD', 'CASH'] or len(str(ticker)) > 5:
                continue
            
            # format for alpaca -
            clean_ticker = str(ticker).replace('.', '-')
            clean_tickers.append(clean_ticker)
            
        print(f"successfully extracted {len(clean_tickers)} tickers")
        clean_tickers = clean_tickers[:100]
        
        # add the macro indicator to the end of the list
        clean_tickers.append('SPY')
        return clean_tickers
        
    except Exception as e:
        print(f"fatal error fetching constituents: {e}")
        return []

def fetch_price_data(api, ticker):
    try:
        # fetch historical daily bars, adjusted for splits
        bars = api.get_bars(
            ticker, 
            TimeFrame.Day, 
            start_date, 
            end_date, 
            adjustment='all'
        ).df
        
        if bars.empty:
            print(f"Warning: no price data returned for {ticker} - skipping")
            return
            
        bars = bars.reset_index()
        
        records = []
        for index, row in bars.iterrows():
            trade_date = row['timestamp'].strftime('%Y-%m-%d')
            records.append((
                ticker, 
                trade_date, 
                row['open'], 
                row['high'], 
                row['low'], 
                row['close'], 
                row['volume']
            ))
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # ignore duplicates so we can rerun safely if the script crashes halfway
        cursor.executemany('''
            INSERT OR IGNORE INTO price_data (ticker, trade_date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        conn.close()
        print(f"Inserted {len(records)} price records for {ticker}")
        
    except Exception as e:
        print(f"Failed to fetch prices for {ticker}. error: {e}")

def run_extraction():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Alpaca API keys missing")
        return

    tickers = get_sp100_tickers()
    if not tickers:
        print("ticker extraction failed")
        return

    api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')
    
    print(f"starting price extraction for {len(tickers)} assets...")
    
    for count, ticker in enumerate(tickers, 1):
        print(f"[{count}/{len(tickers)}] processing {ticker}...", end=" ")
        fetch_price_data(api, ticker)
        
        # strict rate limiting for the free tier
        time.sleep(1)
        
    print("price extraction pipeline complete")

if __name__ == '__main__':
    run_extraction()