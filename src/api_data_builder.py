import alpaca_trade_api as tradeapi
import pandas as pd
import requests
import io
import time
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Configuration and API Keys
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

if not API_KEY or not SECRET_KEY:
    raise ValueError("Alpaca API keys not found")

START_DATE = "2020-01-01"
END_DATE = "2025-01-01"
DATA_DIR = os.path.join("data", "training_raw")

def fetch_sp500_tickers_spdr():
    """
    Extracts the official S&P 500 list from State Street Global Advisors
    """
    print("Fetching S&P 500 constituents from State Street...")
    url = "https://www.ssga.com/us/en/intermediary/etfs/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_excel(io.BytesIO(response.content), skiprows=4)
        df = df.dropna(subset=['Ticker'])
        
        # Clean formatting for Alpaca compatibility
        clean_tickers = [str(t).replace('.', '-') for t in df['Ticker'].tolist() if str(t) != 'nan']
        return clean_tickers[:500]
    except Exception as e:
        print(f"Error fetching S&P 500 list: {e}")
        return []

def chunked_date_ranges(start, end, chunk_days=180):
    """
    Splits a date range into smaller chunks to prevent API server timeouts
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    ranges = []
    current = start_dt
    
    while current < end_dt:
        next_date = min(current + pd.Timedelta(days=chunk_days), end_dt)
        ranges.append((current.strftime('%Y-%m-%d'), next_date.strftime('%Y-%m-%d')))
        current = next_date
        
    return ranges

def fetch_alpaca_data(api, ticker):
    """
    Fetches OHLCV prices and news from Alpaca, merging them chronologically
    """
    # 1. Fetch Price Data
    try:
        bars = api.get_bars(ticker, tradeapi.rest.TimeFrame.Day, start=START_DATE, end=END_DATE, adjustment='all').df
        if bars.empty:
            return False
            
        bars.index = bars.index.tz_convert('America/New_York')
        bars['date'] = bars.index.date
    except Exception as e:
        print(f"Price data error for {ticker}: {e}")
        return False

    # 2. Fetch News Data in Chunks
    all_news = []
    date_chunks = chunked_date_ranges(START_DATE, END_DATE, chunk_days=180)
    
    for s, e in date_chunks:
        try:
            news_chunk = api.get_news(ticker, start=s, end=e, limit=10000)
            for n in news_chunk:
                all_news.append({
                    'date': pd.to_datetime(n.created_at).tz_convert('America/New_York').date(),
                    'headline': n.headline
                })
            time.sleep(0.3)
        except Exception:
            pass
            
    # 3. Merge Data
    if all_news:
        news_df = pd.DataFrame(all_news)
        grouped_news = news_df.groupby('date')['headline'].apply(lambda x: ' | '.join(x)).reset_index()
    else:
        grouped_news = pd.DataFrame(columns=['date', 'headline'])

    merged_df = pd.merge(bars, grouped_news, on='date', how='left')
    merged_df['headline'] = merged_df['headline'].fillna("No significant news reported today.")
    
    final_df = merged_df[['Open', 'High', 'Low', 'Close', 'Volume', 'headline']]
    
    save_path = os.path.join(DATA_DIR, f"{ticker}_raw.csv")
    final_df.to_csv(save_path, index=False)
    
    return True

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL)
    sp500_tickers = fetch_sp500_tickers_spdr()
    
    if not sp500_tickers:
        print("Failed to initialize ticker list. Exiting.")
        exit()
        
    print(f"Starting Alpaca data extraction for {len(sp500_tickers)} ")
    success_count = 0
    
    for idx, ticker in enumerate(sp500_tickers, 1):
        print(f"Processing [{idx}/{len(sp500_tickers)}]: {ticker}...", end=" ")
        
        if fetch_alpaca_data(api, ticker):
            print("Completed.")
            success_count += 1
        else:
            print("Skipped due to missing data.")
            
    print(f"Extraction finished. Successfully generated {success_count} datasets")