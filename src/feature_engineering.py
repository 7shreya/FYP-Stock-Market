# src/feature_engineering.py
import pandas as pd
import sqlite3
from src.config import DB_PATH
from datetime import timedelta

def load_data(ticker):
    """Loads raw price and news data from the database."""
    conn = sqlite3.connect(DB_PATH)
    
    # Load Prices
    print(f"   Loading prices for {ticker}...")
    df_price = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM prices WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,)
    )
    
    # Load News
    print(f"   Loading news for {ticker}...")
    df_news = pd.read_sql(
        "SELECT date, sentiment_score FROM news WHERE ticker = ?",
        conn, params=(ticker,)
    )
    
    conn.close()
    return df_price, df_news

def align_news_to_trading_hours(df_news):
    """
    Applies strict time alignment to prevent Look-Ahead Bias.
    1. News > 16:00 -> Next Day
    2. Weekends -> Next Monday
    """
    if df_news.empty:
        return df_news
    
    # Convert to datetime
    df_news['date'] = pd.to_datetime(df_news['date'], utc=True)
    
    # 1. SHIFT AFTER-HOURS NEWS
    # Identify news that happened after 16:00 (4 PM)
    # We add 1 day to these dates effectively moving them to "Tomorrow"
    after_hours_mask = df_news['date'].dt.hour >= 16
    df_news.loc[after_hours_mask, 'date'] = df_news.loc[after_hours_mask, 'date'] + timedelta(days=1)
    
    # 2. NORMALIZE TO DATE ONLY
    # Now that we shifted the time, we only care about the Date
    df_news['aligned_date'] = df_news['date'].dt.date
    
    # 3. HANDLE WEEKENDS
    # If the (shifted) date is a Saturday (5) -> Move to Monday (+2 days)
    # If the (shifted) date is a Sunday (6)   -> Move to Monday (+1 day)
    df_news['weekday'] = df_news['date'].dt.weekday
    
    mask_sat = df_news['weekday'] == 5
    mask_sun = df_news['weekday'] == 6
    
    df_news.loc[mask_sat, 'aligned_date'] += timedelta(days=2)
    df_news.loc[mask_sun, 'aligned_date'] += timedelta(days=1)
    
    return df_news

def create_training_set(ticker):
    print(f"Engineering features for {ticker}...")
    
    # 1. Load Raw Data
    df_price, df_news = load_data(ticker)
    
    if df_price.empty:
        print(f"No price data found for {ticker}. Did you run data_loader?")
        return None

    # 2. Process Prices
    # Ensure standard UTC datetime and set index
    df_price['date'] = pd.to_datetime(df_price['date'], utc=True).dt.date
    df_price.set_index('date', inplace=True)
    
    # 3. Process & Align News
    if not df_news.empty:
        df_news = align_news_to_trading_hours(df_news)
        
        # Aggregate: If multiple articles exist for one day, take the MEAN score
        # FIX: We add .rename('sentiment_score') so pandas knows the column name
        daily_sentiment = df_news.groupby('aligned_date')['sentiment_score'].mean().rename('sentiment_score')
    else:
        # FIX: We explicitly give the empty series a name
        daily_sentiment = pd.Series(dtype=float, name='sentiment_score')
    
    # 4. Merge (Left Join)
    # We keep all Price days. If there's news, we attach it.
    full_df = df_price.join(daily_sentiment, how='left')
    
    # 5. Handle Missing Sentiment (Null News Days)
    # If no news happened that day, assume Neutral (0.0)
    full_df['sentiment_score'] = full_df['sentiment_score'].fillna(0.0)
    
    # 6. Final Cleanup
    full_df.dropna(inplace=True)
    full_df.sort_index(inplace=True)
    
    print(f"✅ Training set ready: {len(full_df)} rows. Range: {full_df.index.min()} to {full_df.index.max()}")
    return full_df

if __name__ == "__main__":
    stock = "NVDA" 
    df = create_training_set(stock)
    
    if df is not None:
        print("\nSample Data (First 5 rows):")
        print(df.head())