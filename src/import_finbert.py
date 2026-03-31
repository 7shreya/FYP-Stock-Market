import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv('DB_PATH', '../data/stock_data.db')
SCORED_NEWS_CSV = '../data/finbert_scored_news.csv'

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def import_scored_news():
    if not os.path.exists(SCORED_NEWS_CSV):
        print(f"error: could not find {SCORED_NEWS_CSV}")
        return

    print("loading finbert scores from csv...")
    df = pd.read_csv(SCORED_NEWS_CSV)
    
    # drop any rows that failed processing just in case
    df = df.dropna(subset=['sentiment_pos', 'sentiment_neg', 'sentiment_neu'])
    
    records = list(df[['sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'id']].itertuples(index=False, name=None))
    
    print(f"preparing to update {len(records)} records in the database...")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # update the existing rows with the new sentiment tensors
    cursor.executemany('''
        UPDATE news_data 
        SET sentiment_pos = ?, sentiment_neg = ?, sentiment_neu = ?
        WHERE id = ?
    ''', records)
    
    conn.commit()
    conn.close()
    print("successfully imported finbert sentiment scores into sqlite")

if __name__ == '__main__':
    import_scored_news()