# src/prepare_for_colab.py
import pandas as pd
import os
from src.config import MIN_HEADLINES_THRESHOLD

def prepare_csv():
    # 1. Load Raw Data
    raw_path = os.path.join("data", "raw", "raw_news.csv")
    df = pd.read_csv(raw_path)
    
    # Standardize columns
    if 'title' in df.columns: df.rename(columns={'title': 'headline'}, inplace=True)
    if 'stock' in df.columns: df.rename(columns={'stock': 'ticker'}, inplace=True)
    
    # 2. Filter by Threshold (The Dynamic Logic)
    counts = df['ticker'].value_counts()
    valid_tickers = counts[counts >= MIN_HEADLINES_THRESHOLD].index.tolist()
    df_filtered = df[df['ticker'].isin(valid_tickers)].copy()
    
    # 3. Save for Colab
    output_path = os.path.join("data", "processed", "upload_to_colab.csv")
    df_filtered.to_csv(output_path, index=False)
    print(f"Saved {len(df_filtered)} rows to {output_path}")

if __name__ == "__main__":
    prepare_csv()