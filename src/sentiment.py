# src/sentiment.py
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from src.config import FINBERT_MODEL

# Load the model ONCE when this module is imported.
# This makes the app faster because we don't reload the AI for every single request.
print("⏳ Loading FinBERT Model...")
tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("✅ FinBERT Loaded Successfully.")

def predict_sentiment(text_list):
    """
    Takes a list of headlines (strings) and returns sentiment results.
    """
    if not text_list:
        return []
    # Truncate text to 512 tokens to prevent crashing on long sentences
    return nlp(text_list, truncation=True, max_length=512)

def convert_score_to_numeric(result):
    """
    Converts FinBERT output to a single number:
    Positive: +score
    Negative: -score
    Neutral:  0
    """
    label = result['label']
    score = result['score']
    
    if label == 'positive':
        return score
    elif label == 'negative':
        return -score
    else:
        return 0.0