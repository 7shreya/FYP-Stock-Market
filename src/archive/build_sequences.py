import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv('DB_PATH', '../data/stock_data.db')
SEQUENCE_LENGTH = 60 

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def build_training_sequences():
    print("extracting engineered features and aggregating daily sentiment...")
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
    
    # --- THE FIX IS HERE ---
    # Save the true, unscaled log returns specifically for the target array
    merged_df['true_target_return'] = merged_df['log_return']
    
    feature_columns = ['log_return', 'rsi_14', 'macd', 'macd_signal', 'bb_width', 'spy_log_return']
    sentiment_columns = ['avg_pos', 'avg_neg', 'avg_neu']
    
    # Switch to StandardScaler for robust financial normalization
    scaler = StandardScaler()
    merged_df[feature_columns] = scaler.fit_transform(merged_df[feature_columns])
    
    print("scaling complete. generating 60-day sequences...")
    
    X_price, X_sentiment, y_target = [], [], []
    target_dates, meta_tickers = [], [] 
    
    tickers = merged_df['ticker'].unique()
    
    for ticker in tickers:
        ticker_data = merged_df[merged_df['ticker'] == ticker].sort_values('trade_date')
        
        price_vals = ticker_data[feature_columns].values
        sent_vals = ticker_data[sentiment_columns].values
        # Use the unscaled true returns for the target
        targets = ticker_data['true_target_return'].values 
        dates = ticker_data['trade_date'].values
        
        for i in range(len(ticker_data) - SEQUENCE_LENGTH):
            X_price.append(price_vals[i : i + SEQUENCE_LENGTH])
            X_sentiment.append(sent_vals[i : i + SEQUENCE_LENGTH])
            y_target.append(targets[i + SEQUENCE_LENGTH])
            target_dates.append(dates[i + SEQUENCE_LENGTH]) 
            meta_tickers.append(ticker) 
            
    sorted_indices = np.argsort(target_dates)
    
    X_price = np.array(X_price)[sorted_indices]
    X_sentiment = np.array(X_sentiment)[sorted_indices]
    y_target = np.array(y_target)[sorted_indices]
    
    meta_tickers = np.array(meta_tickers)[sorted_indices]
    meta_dates = np.array(target_dates)[sorted_indices]
    meta_data = np.column_stack((meta_tickers, meta_dates))
    
    np.save('../data/X_price.npy', X_price)
    np.save('../data/X_sentiment.npy', X_sentiment)
    np.save('../data/y_target.npy', y_target)
    np.save('../data/meta_data.npy', meta_data)
    
    print("tensors successfully saved")

if __name__ == '__main__':
    build_training_sequences()