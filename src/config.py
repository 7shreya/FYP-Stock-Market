# src/config.py
import os

# --- PATHS ---
# Automatically detect the project root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# --- DATA SETTINGS ---
# QUALITY CONTROL:
# Only process stocks that have at least this many news articles in the dataset.
# This prevents the model from training on "noise" (stocks with insufficient data).
# Set to 50 or higher for better quality.
MIN_HEADLINES_THRESHOLD = 50

# Date range for training data
TRAIN_START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'

# --- MODEL HYPERPARAMETERS ---
SEQ_LENGTH = 60  # Lookback window (days)
PREDICT_DAYS = 1 # Predict 1 day into the future

# --- SENTIMENT CONFIG ---
# FinBERT model name from Hugging Face
FINBERT_MODEL = "ProsusAI/finbert"


# config.py
"""
Static configuration for the ashboard.
Curated list of major S&P equities by secto.
"""

# config.py

#this list is for the market explorer page frontend- for the buttons labels only 
SP100_SECTORS = {
    "Information Technology": [("NVDA", "NVIDIA"), ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("AVGO", "Broadcom"), ("ORCL", "Oracle"), ("MU", "Micron Technology"), ("AMD", "AMD"), ("CSCO", "Cisco Systems"), ("CRM", "Salesforce"), ("AMAT", "Applied Materials")],
    "Financials": [("BRK.B", "Berkshire Hathaway"), ("JPM", "JPMorgan Chase"), ("V", "Visa"), ("MA", "Mastercard"), ("BAC", "Bank of America"), ("MS", "Morgan Stanley"), ("GS", "Goldman Sachs"), ("C", "Citigroup"), ("WFC", "Wells Fargo"), ("BLK", "BlackRock")],
    "Health Care": [("LLY", "Eli Lilly"), ("JNJ", "Johnson & Johnson"), ("UNH", "UnitedHealth Group"), ("ABBV", "AbbVie"), ("MRK", "Merck & Co."), ("AMGN", "Amgen"), ("TMO", "Thermo Fisher Scientific"), ("ABT", "Abbott Laboratories"), ("PFE", "Pfizer"), ("ISRG", "Intuitive Surgical")],
    "Consumer Discretionary": [("AMZN", "Amazon"), ("TSLA", "Tesla"), ("HD", "Home Depot"), ("MCD", "McDonald's"), ("LOW", "Lowe's"), ("BKNG", "Booking Holdings"), ("TJX", "TJX Companies"), ("SBUX", "Starbucks"), ("NKE", "Nike"), ("TGT", "Target")],
    "Communication Services": [("GOOGL", "Alphabet A"), ("GOOG", "Alphabet C"), ("META", "Meta Platforms"), ("NFLX", "Netflix"), ("DIS", "Disney"), ("TMUS", "T-Mobile US"), ("VZ", "Verizon"), ("T", "AT&T"), ("CMCSA", "Comcast"), ("CHTR", "Charter")],
    "Industrials": [("CAT", "Caterpillar"), ("GE", "GE Aerospace"), ("UNP", "Union Pacific"), ("HON", "Honeywell"), ("RTX", "RTX"), ("BA", "Boeing"), ("FDX", "FedEx"), ("UPS", "United Parcel Service"), ("LMT", "Lockheed Martin"), ("MMM", "3M")],
    "Consumer Staples": [("WMT", "Walmart"), ("COST", "Costco"), ("PG", "Procter & Gamble"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"), ("PM", "Philip Morris"), ("MDLZ", "Mondelez"), ("MO", "Altria"), ("CL", "Colgate-Palmolive"), ("TGT", "Target")],
    "Energy": [("XOM", "ExxonMobil"), ("CVX", "Chevron"), ("COP", "ConocoPhillips"), ("SLB", "Schlumberger"), ("EOG", "EOG Resources"), ("MPC", "Marathon Petroleum"), ("PSX", "Phillips 66"), ("VLO", "Valero"), ("OXY", "Occidental"), ("WMB", "Williams Cos")],
    "Utilities": [("NEE", "NextEra Energy"), ("DUK", "Duke Energy"), ("SO", "Southern Co"), ("AEP", "American Electric Power"), ("D", "Dominion Energy"), ("EXC", "Exelon"), ("SRE", "Sempra"), ("PEG", "PSEG"), ("XEL", "Xcel Energy"), ("ED", "Consolidated Edison")],
    "Real Estate": [("AMT", "American Tower"), ("PLD", "Prologis"), ("EQIX", "Equinix"), ("CCI", "Crown Castle"), ("SPG", "Simon Property"), ("WELL", "Welltower"), ("PSA", "Public Storage"), ("DLR", "Digital Realty"), ("O", "Realty Income"), ("CSGP", "CoStar Group")],
    "Materials": [("LIN", "Linde"), ("FCX", "Freeport-McMoRan"), ("SHW", "Sherwin-Williams"), ("APD", "Air Products"), ("ECL", "Ecolab"), ("DOW", "Dow"), ("NEM", "Newmont"), ("DD", "DuPont"), ("NUE", "Nucor"), ("CTVA", "Corteva")]
}