# src/live_news.py
import feedparser
import pandas as pd
from datetime import datetime
from src.sentiment import predict_sentiment, convert_score_to_numeric

def fetch_live_news(ticker):
    """
    1. Fetches real-time headlines from Google News RSS.
    2. Runs FinBERT sentiment analysis on them immediately.
    3. Returns a DataFrame ready for the LSTM model.
    """
    # Ask Google for news about the stock
    rss_url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    
    print(f"📡 Fetching live news for {ticker}...")
    feed = feedparser.parse(rss_url)
    
    if not feed.entries:
        print(f"⚠️ No recent news found for {ticker}.")
        return pd.DataFrame()

    # Extract headlines
    headlines = []
    dates = []
    
    # Process up to 20 newest articles
    for entry in feed.entries[:20]:
        headlines.append(entry.title)
        # Use today's date for live prediction context
        dates.append(datetime.now().strftime('%Y-%m-%d'))

    # Run Live Sentiment Analysis (Using your existing sentiment engine)
    print(f"🧠 Analyzing sentiment for {len(headlines)} headlines...")
    bert_results = predict_sentiment(headlines)
    
    # Format data
    scores = []
    for res in bert_results:
        scores.append(convert_score_to_numeric(res))
        
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'ticker': ticker,
        'headline': headlines,
        'sentiment_score': scores
    })
    
    # Aggregate to get one score per day (Mean sentiment)
    # This is what the LSTM needs (one row per day)
    daily_sentiment = df.groupby('date')['sentiment_score'].mean().reset_index()
    
    print(f"✅ Generated live sentiment score: {daily_sentiment['sentiment_score'].values[0]:.4f}")
    return daily_sentiment

if __name__ == "__main__":
    # Test with a stock NOT in your database
    df = fetch_live_news("NFLX")
    print(df)