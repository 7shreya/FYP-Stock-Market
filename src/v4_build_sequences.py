import os
import sqlite3
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv('DB_PATH', '../data/stock_data.db')
SEQUENCE_LENGTH = 60 
HORIZON = 5  

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def build_v4_sequences():
    print("--- INITIATING V4 SEQUENCE BUILDER (VOLATILITY CROSS-ATTENTION) ---")
    print("Extracting raw engineered features and sentiment...")
    conn = get_db_connection()
    
    price_query = "SELECT * FROM model_features ORDER BY ticker, trade_date ASC"
    price_df = pd.read_sql_query(price_query, conn)
    
    news_query = """
        SELECT ticker, published_at as trade_date, 
               AVG(sentiment_pos) as avg_pos, 
               AVG(sentiment_neg) as avg_neg, 
               AVG(sentiment_neu) as avg_neu
        FROM news_data 
        WHERE sentiment_pos IS NOT NULL
        GROUP BY ticker, published_at
    """
    news_df = pd.read_sql_query(news_query, conn)
    conn.close()

    merged_df = pd.merge(price_df, news_df, on=['ticker', 'trade_date'], how='left')
    
    merged_df['avg_pos'] = merged_df['avg_pos'].fillna(0.0)
    merged_df['avg_neg'] = merged_df['avg_neg'].fillna(0.0)
    merged_df['avg_neu'] = merged_df['avg_neu'].fillna(1.0)
    
    merged_df['true_target_return'] = merged_df['log_return']
    
    feature_columns = ['log_return', 'rsi_14', 'macd', 'macd_signal', 'bb_width', 'spy_log_return']
    sentiment_columns = ['avg_pos', 'avg_neg', 'avg_neu']
    
    print("Skipping global scaling to prevent look-ahead bias. Generating raw sequences...")
    
    X_price, X_sentiment, y_target = [], [], []
    target_dates, meta_tickers = [], [] 
    
    tickers = merged_df['ticker'].unique()
    
    for ticker in tickers:
        ticker_data = merged_df[merged_df['ticker'] == ticker].sort_values('trade_date')
        
        price_vals = ticker_data[feature_columns].values
        sent_vals = ticker_data[sentiment_columns].values
        targets = ticker_data['true_target_return'].values 
        dates = ticker_data['trade_date'].values
        
        max_idx = len(ticker_data) - SEQUENCE_LENGTH - HORIZON
        
        if max_idx < 0:
            continue 
            
        for i in range(max_idx + 1):
            X_price.append(price_vals[i : i + SEQUENCE_LENGTH])
            X_sentiment.append(sent_vals[i : i + SEQUENCE_LENGTH])
            
            future_5_day_returns = targets[i + SEQUENCE_LENGTH : i + SEQUENCE_LENGTH + HORIZON]
            swing_target = np.sum(future_5_day_returns)
            y_target.append(swing_target)
            
            target_dates.append(dates[i + SEQUENCE_LENGTH - 1]) 
            meta_tickers.append(ticker) 
            
    sorted_indices = np.argsort(target_dates)
    
    X_price = np.array(X_price)[sorted_indices]
    X_sentiment = np.array(X_sentiment)[sorted_indices]
    y_target = np.array(y_target)[sorted_indices]
    
    meta_tickers = np.array(meta_tickers)[sorted_indices]
    meta_dates = np.array(target_dates)[sorted_indices]
    meta_data = np.column_stack((meta_tickers, meta_dates))
    
    print(f"\nTotal V4 Sequences Built: {len(y_target)}")
    
    #loaded from the google drive folder 
    os.makedirs('../data', exist_ok=True)
    np.save('../data/v4_X_price.npy', X_price)
    np.save('../data/v4_X_sentiment.npy', X_sentiment)
    np.save('../data/v4_y_target.npy', y_target)
    np.save('../data/v4_meta_data.npy', meta_data)
    
    print("Complete! Tensors saved to ../data/")

if __name__ == '__main__':
    build_v4_sequences()