# src/api_data_builder.py
import os
import time
import pandas as pd
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST, TimeFrame

# 1. Credentials & Setup
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets"

try:
    api = REST(key_id=API_KEY, secret_key=SECRET_KEY, base_url=BASE_URL, api_version='v2')
except Exception as e:
    print(f"Connection failed: {e}")
    exit()

def fetch_liquid_tickers(api, limit=500):
    assets = api.list_assets(status='active', asset_class='us_equity')
    valid = [a.symbol for a in assets if a.tradable and a.marginable and a.fractionable and a.exchange in ['NYSE', 'NASDAQ']]
    return valid[:limit]

def build_dataset(ticker, start_date, end_date):
    start_iso, end_iso = f"{start_date}T00:00:00Z", f"{end_date}T23:59:59Z"
    
    try:
        news_items = api.get_news(ticker, start=start_iso, end=end_iso, limit=10000)
    except Exception:
        return False

    if len(news_items) < 50: return False
        
    print(f"[{ticker}] Processing. Applying Strict Temporal Alignment...")
    
    # --- ALIGNMENT ALGORITHM (Zero Look-Ahead Bias) ---
    aligned_news = []
    for item in news_items:
        dt = pd.to_datetime(item.created_at)
        if dt.tzinfo is None: dt = dt.tz_localize('UTC')
        dt_est = dt.tz_convert('US/Eastern')
        
        # Shift >= 4:00 PM to tomorrow
        if dt_est.hour >= 16:
            dt_est += pd.Timedelta(days=1)
            
        # Shift Weekends to Monday
        if dt_est.dayofweek == 5: # Saturday
            dt_est += pd.Timedelta(days=2)
        elif dt_est.dayofweek == 6: # Sunday
            dt_est += pd.Timedelta(days=1)
            
        aligned_news.append({'date': dt_est.date(), 'headline': item.headline})
        
    news_df = pd.DataFrame(aligned_news)
    if not news_df.empty:
        news_df = news_df.groupby('date')['headline'].apply(lambda x: '. '.join(x)).reset_index()
        news_df.set_index('date', inplace=True)
    else:
        news_df = pd.DataFrame(columns=['headline'])

    # --- STOCK SPLIT / DIVIDEND ADJUSTMENT ---
    try:
        bars = api.get_bars(ticker, TimeFrame.Day, start_iso, end_iso, adjustment='all').df
        if bars.empty: return False
        bars.index = bars.index.date
        bars.index.name = 'date'
    except Exception:
        return False

    # Merge and Export
    combined_df = bars.join(news_df, how='left')
    combined_df['headline'] = combined_df['headline'].fillna("No significant news reported today.")
    
    req_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in combined_df.columns for col in req_cols): return False
            
    final_df = combined_df[['open', 'high', 'low', 'close', 'volume', 'headline']]
    final_df.to_csv(f"data/training_raw/{ticker}_raw.csv")
    return True

if __name__ == "__main__":
    os.makedirs("data/training_raw", exist_ok=True)
    tickers = fetch_liquid_tickers(api, limit=500)
    for ticker in tickers:
        build_dataset(ticker, "2020-01-01", "2025-12-31")
        time.sleep(0.5)