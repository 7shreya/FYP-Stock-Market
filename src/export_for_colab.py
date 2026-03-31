import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
DB_PATH = os.getenv('DB_PATH', '../data/stock_data.db')

def export_unprocessed_news():
    print("Exporting unprocessed news to csv")
    conn = sqlite3.connect(DB_PATH)
    
    # query only the news that hasn't been scored yet
    query = """
        SELECT id, ticker, published_at, headline 
        FROM news_data 
        WHERE sentiment_pos IS NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No unprocessed news found in the database")
        return
        
    output_path = '../data/unprocessed_news.csv'
    df.to_csv(output_path, index=False)
    print(f"successfully exported {len(df)} rows to {output_path}")

if __name__ == '__main__':
    export_unprocessed_news()