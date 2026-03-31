import os
from dotenv import load_dotenv

load_dotenv()

# --- PROJECT PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


HOME_DIR = os.path.expanduser("~")
DB_PATH = os.path.join(HOME_DIR, 'Downloads', 'market_data.db')
# --- DATA SOURCES ---
# Official iShares S&P 500 ETF (IVV) Holdings
SP500_URL = "https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?fileType=csv&fileName=IVV_holdings&dataType=fund"

# --- TIMELINES ---
START_DATE = '2020-01-01'
END_DATE = '2025-12-31' 

# --- API KEYS ---
ALPACA_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

