import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- DYNAMIC FOLDER SEPARATION ---
# DB moves to local Downloads to avoid OneDrive sync lag/errors
CODE_ROOT = Path(__file__).parent.resolve()
DOWNLOADS_DIR = Path.home() / "Downloads"
DB_PATH = str(DOWNLOADS_DIR / "fyp_stock_data.db")

# --- API KEYS ---
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# --- RESEARCH PARAMETERS ---
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
MARKET_CLOSE_EST = "16:00" # Crucial for eliminating look-ahead bias 