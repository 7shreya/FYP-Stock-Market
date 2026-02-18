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