
#predict_live to power the frontend 

import os
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import tensorflow as tf
import joblib
import difflib
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from transformers import pipeline
from dotenv import load_dotenv

#custom neiral network layers class 

@tf.keras.utils.register_keras_serializable()
def directional_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.cast(tf.equal(tf.sign(y_true), tf.sign(y_pred)), tf.float32)) * 100

@tf.keras.utils.register_keras_serializable()
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='att_w', shape=(input_shape[-1], 1), initializer='random_normal')
        self.b = self.add_weight(name='att_b', shape=(input_shape[1], 1), initializer='zeros')
        super(TemporalAttention, self).build(input_shape)
    def call(self, x):
        e = K.tanh(tf.matmul(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)
    def get_config(self): return super(TemporalAttention, self).get_config()


#Fuzzy search functionality for tickers 
def resolve_ticker(user_input, api):
    """Matches company names or typos to the closest valid Alpaca ticker."""
    user_input = user_input.upper().strip()
    try:
        #Fetch active assets to avoid overwhelming the search
        assets = api.list_assets(status='active', asset_class='us_equity')
        ticker_list = [a.symbol for a in assets]
        name_map = {a.name.upper(): a.symbol for a in assets if len(a.name) > 3}
        
        #Exact Match
        if user_input in ticker_list: return user_input
        
        #Fuzzy match by company name 
        matches = difflib.get_close_matches(user_input, name_map.keys(), n=1, cutoff=0.5)
        if matches: return name_map[matches[0]]
        
        #Fuzzy match by ticker typo
        t_matches = difflib.get_close_matches(user_input, ticker_list, n=1, cutoff=0.6)
        return t_matches[0] if t_matches else user_input
    except: 
        return user_input


# DATA PIPELINE & FEATURE ENGINEERING

def fetch_and_align_data(ticker):
    load_dotenv()
    api = tradeapi.REST(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), os.getenv('APCA_API_BASE_URL'), api_version='v2')
    
    final_ticker = resolve_ticker(ticker, api)
    
    #settings - 20 mins delayayed data is fetched from the API
    now_est = pd.Timestamp.now(tz='America/New_York')
    end_date = now_est - pd.Timedelta(minutes=20)
    start_date = (end_date - pd.Timedelta(days=150)).isoformat()
    
    try:
        bars = api.get_bars(final_ticker, tradeapi.TimeFrame.Day, start=start_date, end=end_date.isoformat()).df
        spy = api.get_bars('SPY', tradeapi.TimeFrame.Day, start=start_date, end=end_date.isoformat()).df
        news = api.get_news(final_ticker, start=(end_date - pd.Timedelta(days=40)).isoformat(), limit=40)
        
        if bars.empty or spy.empty: return None, None, f"Data unavailable for {final_ticker}", None, final_ticker
        bars.columns = [c.lower() for c in bars.columns]
        spy.columns = [c.lower() for c in spy.columns]
    except Exception as e: 
        return None, None, str(e), None, final_ticker

    #Align indices & strip timezones to fix mapping errors
    for frame in [bars, spy]: frame.index = frame.index.normalize().tz_localize(None)

    #FEATURE ENGINEERING
    df = pd.DataFrame(index=bars.index)
    
    #preserve the raw close price for the stock graph
    df['close'] = bars['close'] 
    
    df['log_return'] = np.log(bars['close'] / bars['close'].shift(1))
    df['spy_log_return'] = np.log(spy['close'] / spy['close'].shift(1))
    
    #Technical Indicators
    delta = bars['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + (gain / loss)))
    
    exp1 = bars['close'].ewm(span=12, adjust=False).mean()
    exp2 = bars['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['bb_width'] = (bars['close'].rolling(20).std() * 4) / bars['close'].rolling(20).mean()
    
    #Slice the final 60 days including the close price
    price_slice = df[['close', 'log_return', 'rsi_14', 'macd', 'macd_signal', 'bb_width', 'spy_log_return']].dropna().tail(60)

    #NLP SENTIMENT STREAM
    nlp = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    sent_map, tally = {}, {"POS": 0, "NEG": 0, "NEU": 0}
    headlines_ui = []

    for article in news:
        ts = pd.Timestamp(article.created_at).tz_convert('America/New_York')
        target_date = (ts.normalize() if ts.hour < 16 else (ts + pd.Timedelta(days=1)).normalize()).tz_localize(None)
        
        scores = {s['label']: s['score'] for s in nlp(article.headline[:512])[0]}
        label = "POS" if scores['positive'] > 0.5 else "NEG" if scores['negative'] > 0.5 else "NEU"
        tally[label] += 1
        
        if target_date not in sent_map: sent_map[target_date] = []
        sent_map[target_date].append([scores['positive'], scores['negative'], scores['neutral']])
        
        #Get the URL for the news stories links to the external websites
        headlines_ui.append({
            "ts": ts.strftime('%b %d, %H:%M'), 
            "text": article.headline, 
            "sentiment": label,
            "url": getattr(article, 'url', '#') 
        })

    sent_aligned = [np.mean(sent_map[d], axis=0) if d in sent_map else [0.0, 0.0, 1.0] for d in price_slice.index]
    
    return price_slice, np.array(sent_aligned), headlines_ui, tally, final_ticker


# NEURAL INFERENCE ENGINE

def run_live_v4_prediction(ticker):
    price_df, sent_arr, headlines, tally, final_ticker = fetch_and_align_data(ticker)
    if price_df is None: return {"error": tally}

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler = joblib.load(os.path.join(base, 'data', 'scaler.pkl'))
    model = tf.keras.models.load_model(
        os.path.join(base, 'data', 'models', 'v4_champion_model.keras'), 
        custom_objects={'TemporalAttention': TemporalAttention, 'directional_accuracy': directional_accuracy}
    )
    
    # Feature pre-processing (Isolate the 6 neural features from the frontend 'close' price)
    model_features = price_df[['log_return', 'rsi_14', 'macd', 'macd_signal', 'bb_width', 'spy_log_return']]
    
    #LAMPING: Prevents Sigmoid Saturation / feature dominance
    scaled_data = np.clip(scaler.transform(model_features.values), -3.0, 3.0)
    
    #Raw Prediction
    raw_pred = model.predict([scaled_data.reshape(1, 60, 6), sent_arr.reshape(1, 60, 3)], verbose=0)[0][0]
    
    #Calibration layer
    news_bias = (tally['POS'] - tally['NEG']) / (sum(tally.values()) + 1)
    adjusted_prob = np.clip(raw_pred + (news_bias * 0.1), 0.01, 0.99)
    
    latest = price_df.iloc[-1]
    rel_strength = latest['log_return'] - latest['spy_log_return']

    return {
        "ticker": final_ticker,
        "signal": "BULLISH" if adjusted_prob > 0.5 else "BEARISH",
        "confidence": adjusted_prob if adjusted_prob > 0.5 else 1 - adjusted_prob,
        "history": price_df, # Contains the 'close' column for Plotly chart 
        "volatility": latest['bb_width'],
        "rsi": latest['rsi_14'],
        "macd": latest['macd'] - latest['macd_signal'],
        "rel_strength": rel_strength,
        "headlines": headlines,
        "tally": tally
    }