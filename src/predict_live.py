import os
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from transformers import pipeline
from dotenv import load_dotenv
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. LOAD CUSTOM LAYERS & METRICS ---
@tf.keras.utils.register_keras_serializable()
def directional_accuracy(y_true, y_pred):
    y_true_sign = tf.sign(y_true)
    y_pred_sign = tf.sign(y_pred)
    correct = tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)
    return tf.reduce_mean(correct) * 100

@tf.keras.utils.register_keras_serializable()
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), 
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), 
                                 initializer='zeros', trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(tf.matmul(x, self.W) + self.b)
        a = K.softmax(e, axis=1) 
        output = x * a
        return K.sum(output, axis=1) 

    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        return config

# --- 2. DATA PIPELINE ---
def get_live_data(ticker, lookback_days=400):
    load_dotenv()
    api = tradeapi.REST(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), 
                        os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'), api_version='v2')
    
    end_date = pd.Timestamp.now(tz='America/New_York') - pd.Timedelta(minutes=20)
    start_date = end_date - pd.Timedelta(days=lookback_days)
    
    try:
        barset = api.get_bars(ticker, tradeapi.TimeFrame.Day, start=start_date.isoformat(), end=end_date.isoformat()).df
        spy_barset = api.get_bars('SPY', tradeapi.TimeFrame.Day, start=start_date.isoformat(), end=end_date.isoformat()).df
    except Exception as e:
        return None, None, str(e)
        
    barset.index = barset.index.normalize()
    spy_barset.index = spy_barset.index.normalize()
    
    df = pd.DataFrame(index=barset.index)
    df['close'] = barset['close']
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    df['spy_close'] = spy_barset['close']
    df['spy_log_return'] = np.log(df['spy_close'] / df['spy_close'].shift(1))
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['bb_width'] = (sma_20 + (std_20 * 2) - (sma_20 - (std_20 * 2))) / sma_20
    
    df = df.dropna()
    price_df = df[['log_return', 'rsi_14', 'macd', 'macd_signal', 'bb_width', 'spy_log_return']]
    
    # --- NLP PIPELINE ---
    target_dates = price_df.index[-60:]
    news_items = api.get_news(ticker, start=target_dates[0].isoformat(), end=end_date.isoformat(), limit=50)
    
    nlp = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
    
    sentiment_records = []
    recent_headlines = [] # NEW: We will save the actual articles here
    
    for article in news_items:
        date = pd.Timestamp(article.created_at).tz_convert('America/New_York').normalize()
        text = f"{article.headline}. {article.summary}"[:512]
        
        scores = nlp(text)[0]
        score_dict = {s['label']: s['score'] for s in scores}
        
        sentiment_records.append({
            'date': date,
            'pos': score_dict.get('positive', 0.0),
            'neg': score_dict.get('negative', 0.0),
            'neu': score_dict.get('neutral', 0.0)
        })
        
        # Save headline info for the UI
        recent_headlines.append({
            "date": date.strftime('%Y-%m-%d'),
            "headline": article.headline,
            "url": article.url,
            "sentiment": "Positive" if score_dict.get('positive') > 0.4 else "Negative" if score_dict.get('negative') > 0.4 else "Neutral"
        })
        
    news_df = pd.DataFrame(sentiment_records)
    if not news_df.empty:
        news_df = news_df.groupby('date').mean()
    
    sent_list = []
    for d in target_dates:
        if not news_df.empty and d in news_df.index:
            sent_list.append(news_df.loc[d].values)
        else:
            sent_list.append([0.0, 0.0, 1.0]) 
            
    sentiment_df = pd.DataFrame(sent_list, index=target_dates, columns=['avg_pos', 'avg_neg', 'avg_neu'])
    
    return price_df, sentiment_df, recent_headlines

# --- 3. INFERENCE ENGINE FOR WEB APP ---
def run_live_v4_prediction(ticker):
    price_df, sentiment_df, recent_headlines = get_live_data(ticker)
    
    if price_df is None or len(price_df) < 60:
        return {"error": f"Failed to fetch enough market data. Error: {recent_headlines}"}
        
    scaler = StandardScaler()
    scaled_price_full = scaler.fit_transform(price_df.values)
    
    X_price_live = scaled_price_full[-60:].reshape(1, 60, 6)
    X_sent_live = sentiment_df.values.reshape(1, 60, 3)
    
    model_path = 'data/models/v4_champion_model.keras'
    if not os.path.exists(model_path):
        model_path = '../data/models/v4_champion_model.keras'
        
    model = tf.keras.models.load_model(model_path, custom_objects={
        'TemporalAttention': TemporalAttention,
        'directional_accuracy': directional_accuracy
    })
    
    predicted_log_return = model.predict([X_price_live, X_sent_live], verbose=0)[0][0]
    predicted_pct_change = (np.exp(predicted_log_return) - 1) * 100
    
    direction = "BULLISH (UP)" if predicted_pct_change > 0 else "BEARISH (DOWN)"
    
    today_sent = sentiment_df.iloc[-1]
    dominant_vibe = "Neutral"
    if today_sent['avg_pos'] > 0.4: dominant_vibe = "Positive"
    if today_sent['avg_neg'] > 0.4: dominant_vibe = "Negative"

    return {
        "success": True,
        "ticker": ticker,
        "direction": direction,
        "prediction_pct": predicted_pct_change,
        "vibe": dominant_vibe,
        "pos_score": today_sent['avg_pos'],
        "neg_score": today_sent['avg_neg'],
        "price_data": price_df,
        "sentiment_data": sentiment_df,
        "headlines": recent_headlines[:5] # Return the top 5 most recent articles
    }