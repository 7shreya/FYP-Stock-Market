# src/ingest_news.py
import pandas as pd
import sqlite3
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from src.config import DB_PATH, FINBERT_MODEL, MIN_HEADLINES_THRESHOLD
import os

def load_and_filter_news():
    csv_path = os.path.join("data", "raw", "raw_news.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return None

    print("Loading raw news CSV...")
    df = pd.read_csv(csv_path)
    
    # Standardize column names to ensure compatibility
    if 'title' in df.columns:
        df.rename(columns={'title': 'headline'}, inplace=True)
    if 'stock' in df.columns:
        df.rename(columns={'stock': 'ticker'}, inplace=True)

    initial_count = len(df)
    print(f"   Raw Dataset: {initial_count} rows")

    # --- DYNAMIC DISCOVERY ---
    # 1. Count headlines per ticker
    ticker_counts = df['ticker'].value_counts()
    
    # 2. Identify Valid Tickers (those with enough data)
    valid_tickers = ticker_counts[ticker_counts >= MIN_HEADLINES_THRESHOLD].index.tolist()
    
    print(f"   Found {len(ticker_counts)} unique tickers in raw data.")
    print(f"   Applying Quality Filter: Keeping tickers with >= {MIN_HEADLINES_THRESHOLD} headlines.")
    print(f"   Qualifying Tickers: {len(valid_tickers)}")
    
    if len(valid_tickers) > 0:
        print(f"   Examples: {valid_tickers[:5]}...")

    # 3. Filter the dataset to keep only valid tickers
    df = df[df['ticker'].isin(valid_tickers)].copy()
    
    dropped_rows = initial_count - len(df)
    print(f"   Dropped {dropped_rows} rows of sparse data.")
    print(f"   Final Dataset to Process: {len(df)} rows")
    
    return df

def process_sentiment(df):
    print("Loading AI Model...")
    tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
    model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1) # device=-1 for CPU

    print(f"Starting Sentiment Analysis on {len(df)} headlines...")
    print("This may take a while depending on your hardware...")
    
    # Batch processing is safer for large datasets
    # If you run out of memory, reduce batch_size (e.g., to 16 or 8)
    results = nlp(df['headline'].tolist(), batch_size=32, truncation=True, max_length=512)
    
    sentiment_scores = []
    labels = []
    
    for res in results:
        label = res['label']
        score = res['score']
        labels.append(label)
        
        # Convert label to numeric score
        if label == 'positive':
            sentiment_scores.append(score)
        elif label == 'negative':
            sentiment_scores.append(-score)
        else:
            sentiment_scores.append(0.0)
            
    df['sentiment_score'] = sentiment_scores
    df['sentiment_label'] = labels
    return df

def save_news_to_db(df):
    conn = sqlite3.connect(DB_PATH)
    
    # --- TIME ALIGNMENT PREPARATION ---
    # We must preserve the specific time (HH:MM:SS) to later handle 
    # the 16:00 market close logic correctly.
    
    # 1. Convert to UTC datetime objects
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    
    # 2. Drop rows with invalid dates
    df = df.dropna(subset=['date'])

    # 3. Format as string including TIME
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Select columns matching our DB schema
    cols = ['date', 'ticker', 'headline', 'sentiment_score', 'sentiment_label']
    df['source'] = 'Kaggle'
    
    # Save to database
    df[cols + ['source']].to_sql('news', conn, if_exists='append', index=False)
    conn.close()
    print("Success! Data saved to DB.")

if __name__ == "__main__":
    df = load_and_filter_news()
    if df is not None and not df.empty:
        # Process the sentiment and save
        processed_df = process_sentiment(df) 
        save_news_to_db(processed_df)