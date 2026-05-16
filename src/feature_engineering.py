# src/feature_engineering.py
import pandas as pd
import sqlite3
from src.config import DB_PATH
from datetime import timedelta

def load_data(ticker):
    """Pull price + news rows out of the db for a given ticker."""
    conn = sqlite3.connect(DB_PATH)
    
    #prices first
    print(f"   Loading prices for {ticker}...")
    df_price = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM prices WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,)
    )
    
    print(f"   Loading news for {ticker}")
    df_news = pd.read_sql(
        "SELECT date, sentiment_score FROM news WHERE ticker = ?",
        conn, params=(ticker,)
    )
    
    conn.close()
    return df_price, df_news
#look ahead bias prevention 
def align_news_to_trading_hours(df_news):
    """
    Realigns news timestamps to prevent look-ahead bias.
    Post-market news can't affect that day's price action so we push it forward.
    Weekends roll to Monday for the same reason.
    """
    if df_news.empty:
        return df_news
    
    df_news['date'] = pd.to_datetime(df_news['date'], utc=True)
    
    #anything after 4pm is next-day 
    after_hours_mask = df_news['date'].dt.hour >= 16
    df_news.loc[after_hours_mask, 'date'] = df_news.loc[after_hours_mask, 'date'] + timedelta(days=1)
    
    #drop the time, only the date matters from here
    df_news['aligned_date'] = df_news['date'].dt.date
    
    df_news['weekday'] = df_news['date'].dt.weekday
    
    #sat=5, sun=6 - no trading so roll to Monday
    mask_sat = df_news['weekday'] == 5
    mask_sun = df_news['weekday'] == 6
    
    df_news.loc[mask_sat, 'aligned_date'] += timedelta(days=2)
    df_news.loc[mask_sun, 'aligned_date'] += timedelta(days=1)
    
    return df_news

def create_training_set(ticker):
    print(f"Engineering features for {ticker}...")
    
    df_price, df_news = load_data(ticker)
    
    if df_price.empty:
        print(f"No price data found for {ticker}. Did you run data_loader?")
        return None

    #dates only, then index on them
    df_price['date'] = pd.to_datetime(df_price['date'], utc=True).dt.date
    df_price.set_index('date', inplace=True)
    
    if not df_news.empty:
        df_news = align_news_to_trading_hours(df_news)
        
        #multiple articles same day -> average them out
        daily_sentiment = df_news.groupby('aligned_date')['sentiment_score'].mean().rename('sentiment_score')
    else:
        daily_sentiment = pd.Series(dtype=float, name='sentiment_score')  #empty but named so the join doesn't break
    
    #left join so every price row survives -sentiment only where it exists
    #forward-filling old scores would introduce look-ahead bias, so we don't
    full_df = df_price.join(daily_sentiment, how='left')
    
    #no news that day = neutral, don't carry forward a previous score
    full_df['sentiment_score'] = full_df['sentiment_score'].fillna(0.0)
    
    full_df.dropna(inplace=True)
    full_df.sort_index(inplace=True)
    
    print(f" Training set ready: {len(full_df)} rows. Range: {full_df.index.min()} to {full_df.index.max()}")
    return full_df

if __name__ == "__main__":
    stock = "NVDA" 
    df = create_training_set(stock)
    
    if df is not None:
        print("\nSample Data (First 5 rows):")
        print(df.head())