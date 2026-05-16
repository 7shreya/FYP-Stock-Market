# src/import_from_colab.py

import pandas as pd
import sqlite3
import os
from src.config import DB_PATH

def import_colab_data():
    csv_path = os.path.join("data", "processed", "scored_news_from_colab.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    print("Reading Scored Data...")
    df = pd.read_csv(csv_path)
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)#utc
    df = df.dropna(subset=['date'])
    
    # Format strictly as YYYY-MM-DD HH:MM:SS for the alignment logic
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to SQLite
    conn = sqlite3.connect(DB_PATH)
    
    # Select columns
    cols = ['date', 'ticker', 'headline', 'sentiment_score']
    if 'sentiment_label' in df.columns:
        cols.append('sentiment_label')
        
    df['source'] = 'Kaggle_FinBERT_Colab'
    
    # Write to database
    df[cols + ['source']].to_sql('news', conn, if_exists='append', index=False)
    
    conn.close()
    print(f"Success! Imported {len(df)} scored headlines into the database.")

if __name__ == "__main__":
    import_colab_data()