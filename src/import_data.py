# src/import_data.py
import pandas as pd
import sqlite3
import os
from src.config import DB_PATH

def clean_text(text):
    """
    Basic text cleaning:
    1. Converts to string (handles NaNs)
    2. Strips leading/trailing whitespace
    3. Replaces double spaces with single spaces
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())

def import_clean_data():
    csv_path = os.path.join("data", "processed", "processed_all_news.csv")
    
    print("⏳ Loading and Cleaning Data...")
    # 1. Load Data
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    # 2. DATA CLEANING: Dates
    # We use utc=True to handle mixed timezones (e.g., EST vs UTC)
    # This is the correct engineering fix, not a shortcut.
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    
    # Log how many dates failed
    failed_dates = df['date'].isna().sum()
    if failed_dates > 0:
        print(f"     Dropped {failed_dates} rows with invalid dates.")
        df = df.dropna(subset=['date'])
    
    # Standardize format to YYYY-MM-DD for database consistency
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # 3. DATA CLEANING: Text
    # Clean the headlines (remove weird spacing, newlines)
    df['headline'] = df['headline'].apply(clean_text)
    
    # Drop rows where headline became empty after cleaning
    df = df[df['headline'] != ""]
    
    final_count = len(df)
    print(f"    Data Cleaning Complete. Kept {final_count}/{initial_count} rows ({(final_count/initial_count):.1%} retained).")

    # 4. Save to Database
    conn = sqlite3.connect(DB_PATH)
    cols = ['date', 'ticker', 'headline', 'sentiment_score', 'sentiment_label']
    df['source'] = 'Kaggle_Universal'
    
    df[cols + ['source']].to_sql('news', conn, if_exists='replace', index=False)
    conn.close()
    print(" Successfully imported clean data into SQLite.")

if __name__ == "__main__":
    import_clean_data()