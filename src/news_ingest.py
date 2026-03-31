import time
import sqlite3
import pandas as pd
from datetime import timedelta
from alpaca.data.news import NewsClient
from alpaca.data.requests import NewsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
import src.market_config as config
from src.market_db import MarketDB

class NewsIngestor:
    def __init__(self):
        self.news_client = NewsClient(config.ALPACA_KEY, config.ALPACA_SECRET)
        self.trading_client = TradingClient(config.ALPACA_KEY, config.ALPACA_SECRET)
        self.db_manager = MarketDB()

    def get_trading_days(self):
        """Fetches the official market calendar to align news timestamps."""
        print("Fetching market calendar for temporal alignment...")
        request = GetCalendarRequest(start=config.START_DATE, end=config.END_DATE)
        calendar = self.trading_client.get_calendar(request)
        return [c.date for c in calendar]

    def ingest_news(self, tickers, trading_days):
        """
        Fetches headlines and shifts timestamps to the effective trading day.
        Ensures news released after 16:00 ET is mapped to the next open market day.
        """
        for ticker in tickers:
            try:
                # Look back 3 days before the start date to catch weekend news
                start_dt = pd.to_datetime(config.START_DATE) - timedelta(days=3)
                
                request_params = NewsRequest(
                    symbols=ticker,
                    start=start_dt,
                    end=config.END_DATE,
                    limit=10000
                )
                
                news_response = self.news_client.get_news(request_params)
                articles = news_response.news
                
                processed_news = []
                for art in articles:
                    # Convert to Eastern Time for accurate close-of-day logic
                    created_at = art.created_at.tz_convert('America/New_York')
                    
                    # Shift Logic: News after 16:00 (4:00 PM) belongs to the next day
                    effective_date = created_at.date()
                    if created_at.hour >= 16:
                        effective_date += timedelta(days=1)
                    
                    # Align to the next valid trading day (handles weekends/holidays)
                    actual_day = next((d for d in trading_days if d >= effective_date), None)
                    
                    if actual_day:
                        processed_news.append({
                            "ticker": ticker,
                            "headline": art.headline,
                            "published_at": created_at,
                            "effective_trading_day": actual_trading_day
                        })
                
                if processed_news:
                    df = pd.DataFrame(processed_news)
                    with sqlite3.connect(config.DB_PATH) as conn:
                        df.to_sql('news_headlines', conn, if_exists='append', index=False)
                    print(f"Stored {len(processed_news)} headlines for {ticker}.")
                
                # Respect API rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching news for {ticker}: {e}")

if __name__ == "__main__":
    from src.market_ingestor import MarketIngestor
    
    ingestor = NewsIngestor()
    ticker_source = MarketIngestor()
    
    tickers = ticker_source.get_sp500_list()
    calendar = ingestor.get_trading_days()
    
    print(f"Starting news ingestion for {len(tickers)} tickers...")
    ingestor.ingest_news(tickers, calendar)