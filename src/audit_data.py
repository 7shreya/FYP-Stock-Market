# src/audit_data.py
import pandas as pd
import os

def audit_raw_data():
    # Load the raw file as pure strings (don't let pandas guess types yet)
    csv_path = os.path.join("data", "processed", "processed_all_news.csv")
    df = pd.read_csv(csv_path, dtype=str)
    
    print(f"🧐 Auditing {len(df)} rows of raw data...")

    # --- CHECK 1: DATE FORMATS ---
    # Try to convert to datetime, but keep the failures (NaT)
    # We use utc=True because financial news often comes from different timezones
    temp_dates = pd.to_datetime(df['date'], errors='coerce', utc=True)
    
    # Find rows that failed
    failed_rows = df[temp_dates.isna()]
    
    if len(failed_rows) > 0:
        print(f"\n⚠️  Found {len(failed_rows)} unparseable dates!")
        print("   Samples of bad data:")
        print(failed_rows['date'].unique()[:10]) # Show top 10 unique bad formats
    else:
        print("\n✅ Date Audit Passed: All dates are readable (with utc=True).")

    # --- CHECK 2: MISSING HEADLINES ---
    # Check for empty or "nan" headlines
    empty_headlines = df[df['headline'].isna() | (df['headline'].str.strip() == "")]
    if len(empty_headlines) > 0:
        print(f"⚠️  Found {len(empty_headlines)} rows with missing headlines.")
    else:
        print("✅ Headline Audit Passed: No missing text.")

if __name__ == "__main__":
    audit_raw_data()