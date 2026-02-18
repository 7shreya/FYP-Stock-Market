# src/clean_duplicates.py
import sqlite3
import pandas as pd
from src.config import DB_PATH

def clean_duplicates():
    conn = sqlite3.connect(DB_PATH)
    
    df = pd.read_sql("SELECT * FROM news", conn)
    original_count = len(df)
    
    # 1. IDENTIFY DUPLICATES
    # We look for rows where the Ticker, Date, and Headline are ALL the same.
    # keeping='first' means we keep one and delete the rest.
    print(f"   Current rows: {original_count}")
    
    df_clean = df.drop_duplicates(subset=['ticker', 'date', 'headline'], keep='first')
    
    new_count = len(df_clean)
    removed_count = original_count - new_count
    
    if removed_count > 0:
        print(f"Found and removed {removed_count} duplicate articles.")
        
        # 2. SAVE BACK TO DB
        # We replace the old table with this clean version
        print("   Saving clean data back to SQLite...")
        df_clean.to_sql('news', conn, if_exists='replace', index=False)
        print(f"Database updated. Now holds {new_count} unique articles.")
    else:
        print("No duplicates found. Data is already clean!")
    
    conn.close()

if __name__ == "__main__":
    clean_duplicates()