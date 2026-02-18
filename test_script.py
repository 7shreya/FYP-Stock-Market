import tensorflow as tf
from transformers import pipeline

# Check if TensorFlow is working
print("TensorFlow version:", tf.__version__)

# Test FinBERT loading (This will download a small model first time)
print("Loading FinBERT sentiment analyzer...")
sentiment_analysis = pipeline("sentiment-analysis", model="ProsusAI/finbert")
result = sentiment_analysis("Stock prices reached a record high today!")
print(f"Test Result: {result}")