# src/data_loader.py
import sqlite3
import pandas as pd
import yfinance as yf
from src.config import DB_PATH, TRAIN_START_DATE, TRAIN_END_DATE

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    return conn

def create_tables():
    """Creates the necessary tables for the project if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create PRICES Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prices (
        date TEXT,
        ticker TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume INTEGER,
        PRIMARY KEY (date, ticker)
    )
    ''')
    
    # 2. Create NEWS Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        ticker TEXT,
        headline TEXT,
        source TEXT,
        sentiment_score REAL,
        sentiment_label TEXT
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f" Database tables checked/created at: {DB_PATH}")

def save_prices_to_db(df, ticker):
    """Saves a dataframe of price data to the database."""
    conn = get_db_connection()
    
    # --- FIX FOR YFINANCE TUPLE ISSUE ---
    # If columns are MultiIndex (Tuples), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # ------------------------------------

    # Reset index so 'Date' becomes a column
    df = df.reset_index()
    
    # Rename columns to lowercase to match SQL schema
    df.columns = [str(c).lower() for c in df.columns]
    
    # Add the ticker column
    df['ticker'] = ticker
    
    # Convert Timestamp to string
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Keep only necessary columns
    cols_to_keep = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    
    # Filter columns ensuring they exist
    final_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[final_cols]
    
    try:
        df.to_sql('prices', conn, if_exists='append', index=False)
        print(f"   Saved {len(df)} rows for {ticker}")
    except sqlite3.IntegrityError:
        print(f"    Data for {ticker} already exists. Skipping insertion.")
    except Exception as e:
        print(f"    Error saving {ticker}: {e}")
    finally:
        conn.close()

def get_stock_data(ticker):
    """
    The smart fetcher:
    1. Checks if we already have data for 'ticker' in the DB.
    2. If yes, returns it.
    3. If no, downloads it from Yahoo Finance, saves it, and then returns it.
    """
    conn = get_db_connection()
    
    # 1. Check Database First
    print(f" Checking database for {ticker}...")
    query = "SELECT * FROM prices WHERE ticker = ? ORDER BY date"
    df = pd.read_sql(query, conn, params=(ticker,))
    conn.close()
    
    if not df.empty:
        print(f"   Found {len(df)} rows in DB.")
        return df

    # 2. If missing, Fetch Live
    print(f"Not in DB. Downloading from Yahoo Finance...")
    try:
        # Download data
        raw_df = yf.download(ticker, start=TRAIN_START_DATE, end=TRAIN_END_DATE, progress=False)
        
        if raw_df.empty:
            print(f"Stock '{ticker}' not found on Yahoo Finance.")
            return None
            
        # Save to DB (Cache it for next time)
        save_prices_to_db(raw_df, ticker)
        
        # 3. Recursive Call
        # We call this function again. Now that data is saved, it will hit step #1 and return the clean data.
        return get_stock_data(ticker)
        
    except Exception as e:
        print(f"API Error: {e}")
        return None

if __name__ == "__main__":
    # Ensure tables exist first
    create_tables()
    
    # TEST: Try to fetch a random stock
    test_ticker = "NVDA"
    print(f"\n--- Testing Dynamic Fetch for {test_ticker} ---")
    
    data = get_stock_data(test_ticker)
    
    if data is not None:
        print(f"\nSUCCESS! Loaded data for {test_ticker}")
        print(data.head())
    else:
        print("Failed to load data.")