import os
import sqlite3
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv('DB_PATH', '../data/stock_data.db')

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def setup_features_table():
    print("setting up engineered features table...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # creating a new table strictly for the neural network input
    # this separates raw data from engineered data
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        trade_date DATE NOT NULL,
        close REAL,
        log_return REAL,
        rsi_14 REAL,
        macd REAL,
        macd_signal REAL,
        bb_width REAL,
        spy_log_return REAL,
        UNIQUE(ticker, trade_date)
    )
    ''')
    conn.commit()
    conn.close()

def extract_macro_indicator():
    print("calculating SPY macro market baseline...")
    conn = get_db_connection()
    
    # pull raw spy data
    query = "SELECT trade_date, close FROM price_data WHERE ticker = 'SPY' ORDER BY trade_date ASC"
    spy_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if spy_df.empty:
        raise ValueError("SPY data missing from database - cannot calculate macro indicator")
        
    # calculate daily log return for the entire market
    # formula: ln(current_close / previous_close)
    spy_df['spy_log_return'] = np.log(spy_df['close'] / spy_df['close'].shift(1))
    
    # drop the raw close price, we only want the momentum indicator
    spy_df = spy_df[['trade_date', 'spy_log_return']].dropna()
    return spy_df

def calculate_technical_indicators(df):
    # 1. log returns (solves the stationarity problem)
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. rsi - 14 day relative strength index (momentum)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 3. macd (trend direction)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 4. bollinger band width (volatility regime)
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    upper_band = sma_20 + (std_20 * 2)
    lower_band = sma_20 - (std_20 * 2)
    # width is normalised by the sma so it is comparable across stocks of different prices
    df['bb_width'] = (upper_band - lower_band) / sma_20
    
    return df

def process_and_load_features():
    setup_features_table()
    spy_macro_df = extract_macro_indicator()
    
    conn = get_db_connection()
    
    # get the list of all distinct assets, excluding spy as it is our baseline
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT ticker FROM price_data WHERE ticker != 'SPY'")
    tickers = [row[0] for row in cursor.fetchall()]
    
    print(f"Beginning feature engineering for {len(tickers)} assets...")
    
    for count, ticker in enumerate(tickers, 1):
        print(f"[{count}/{len(tickers)}] calculating quant features for {ticker}...")
        
        query = f"SELECT ticker, trade_date, close FROM price_data WHERE ticker = '{ticker}' ORDER BY trade_date ASC"
        df = pd.read_sql_query(query, conn)
        
        # calculate company-specific features
        df = calculate_technical_indicators(df)
        
        # merge the macro market indicator based on the exact trading date
        df = pd.merge(df, spy_macro_df, on='trade_date', how='left')
        
        # rolling windows create null values at the start of the dataset 
        # (e.g., a 26-day EMA requires 26 days of data to start)
        # we must drop these to prevent neural network corruption
        df = df.dropna()
        
        # prepare for database insertion
        records = df.to_dict('records')
        insert_data = [
            (
                row['ticker'], row['trade_date'], row['close'], 
                row['log_return'], row['rsi_14'], row['macd'], 
                row['macd_signal'], row['bb_width'], row['spy_log_return']
            ) for row in records
        ]
        
        # insert or ignore maintains our idempotent, crash-proof pipeline
        cursor.executemany('''
            INSERT OR IGNORE INTO model_features 
            (ticker, trade_date, close, log_return, rsi_14, macd, macd_signal, bb_width, spy_log_return)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', insert_data)
        
    conn.commit()
    conn.close()
    print("Feature engineering pipeline complete - database updated")

if __name__ == '__main__':
    process_and_load_features()