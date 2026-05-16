import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import joblib
import alpaca_trade_api as tradeapi
import difflib
from transformers import pipeline
from dotenv import load_dotenv
import plotly.graph_objects as go
import os
from src.config import SP100_SECTORS

st.set_page_config(
    page_title="Stock Market Deep Learning Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

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

def fuzzy_ticker_search(user_ticker, api):
    try:
        assets = api.list_assets(status='active', asset_class='us_equity')
        ticker_list = [a.symbol for a in assets]
        name_map = {a.name.upper(): a.symbol for a in assets}
        user_ticker = user_ticker.upper()
        
        if user_ticker in ticker_list: return user_ticker
        
        name_matches = difflib.get_close_matches(user_ticker, name_map.keys(), n=1, cutoff=0.5)
        if name_matches: return name_map[name_matches[0]]
        
        ticker_matches = difflib.get_close_matches(user_ticker, ticker_list, n=1, cutoff=0.6)
        return ticker_matches[0] if ticker_matches else user_ticker
    except Exception:
        return user_ticker.upper()

def generate_heuristic_narrative(signal, news_tally, relative_strength, rsi_numerical, current_ticker):
    total_news = sum(news_tally.values()) + 1 
    mood_status = "Optimistic" if news_tally['POS'] > news_tally['NEG'] else "Cautious" if news_tally['NEG'] > news_tally['POS'] else "Neutral"
    technical_pace = "heated (overbought)" if rsi_numerical > 70 else "cooled (oversold)" if rsi_numerical < 30 else "stable"
    perf_status = "outperforming" if relative_strength > 0 else "underperforming"
    
    narrative = f"✦ The model maintains a <b>{'Strong Bullish' if signal == 'BULLISH' else 'Strong Bearish'} 5-Day Neural Outlook</b> for <b>{current_ticker}</b>. Over the next 5 trading days, the system projects this trajectory based on deep temporal price patterns. This call is currently contextualised by a <b>{mood_status}</b> market mood, driven by <b>{news_tally['POS']}</b> positive catalysts in recent headlines. The stock has an RSI of <b>{rsi_numerical:.1f}</b> which indicates a <b>{technical_pace}</b> pace. Notably, <b>{current_ticker}</b> is currently <b>{perf_status}</b> the S&P 500 benchmark by {abs(relative_strength):.2f}%, confirming its relative sector strength. ✦"
    
    return narrative

# CSS styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #c9d1d9; }
    

    .stMetric { 
        border: 1px solid #30363d; 
        background: #161b22; 
        border-radius: 8px; 
        padding: 20px 15px; 
        height: 120px; 
        display: flex; 
        flex-direction: column; 
        justify-content: flex-start; 
        gap: 8px; 
    }
    
    /* Skeleton Loading */
    @keyframes skeleton-loading {
        0% { background-color: #161b22; border-color: #30363d; }
        50% { background-color: #21262d; border-color: #484f58; }
        100% { background-color: #161b22; border-color: #30363d; }
    }
    .skeleton-box {
        animation: skeleton-loading 1.5s infinite;
        border-radius: 8px;
        width: 100%;
        border: 1px solid #30363d;
    }
    
    /* Global Secondary Button Defaults */
    .stButton > button { 
        border-radius: 8px; 
        border: 1px solid #30363d; 
        background-color: #0d1117; 
        color: #c9d1d9; 
        font-weight: bold; 
        transition: all 0.3s ease-in-out;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start !important;
        text-align: left !important;
        padding-left: 15px !important;
        width: 100% !important;
    }

    section[data-testid="stSidebar"] .stButton > button div {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }
            
            
    
    /* Button Run Analysis */
    [data-testid="stBaseButton-primary"] {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important; /* Standard gray/white text */
    }
    
    /* Hover effect */
    [data-testid="stBaseButton-primary"]:hover {
        background-color: #161b22 !important;
        border: 1px solid #40e0d0 !important;
        color: #ffffff !important;
        box-shadow: 0 0 10px rgba(64, 224, 208, 0.3) !important;
    }
    
    a { color: #58a6ff; text-decoration: none; }
    </style>
    """, unsafe_allow_html=True)


#Stage mgmt & caching 
if 'current_ticker' not in st.session_state:
    st.session_state['current_ticker'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "Dashboard"
if 'analysis_cache' not in st.session_state:
    st.session_state['analysis_cache'] = {}

#Navigation menu 
with st.sidebar:
    st.title("Stock Market Deep Learning Model")
    if st.button("Stock Analysis Dashboard", icon=":material/dashboard:", use_container_width=True):
        st.session_state['current_page'] = "Dashboard"
        st.rerun()
    if st.button("Explore Stocks", icon=":material/explore:", use_container_width=True):
        st.session_state['current_page'] = "Explorer"
        st.rerun()

api = tradeapi.REST(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), os.getenv('APCA_API_BASE_URL'), api_version='v2')


# EXPLORE STOCK PAGE 
if st.session_state['current_page'] == "Explorer":
    st.title("S&P 100 Market Explorer")
    st.caption("Select a sector to view the top equities")
    
    sector_choice = st.selectbox("Select Market Sector", list(SP100_SECTORS.keys()))
    
    st.markdown(f"### Top 10 Stocks in <span style='color: #40e0d0;'>{sector_choice}</span>", unsafe_allow_html=True)
    st.caption("✦ Click a ticker to initialize deep neural analysis and load historical price streams")

    target_height = "80px" 

    st.markdown(f"""
        <style>
        [data-baseweb="select"] > div:focus-within {{
            border-color: #40e0d0 !important;
            box-shadow: 0 0 0 1px #40e0d0 !important;
        }}
        [data-testid="stMain"] [data-testid="stBaseButton-secondary"] {{
            min-height: {target_height} !important;
            height: {target_height} !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 2px 5px !important;
            background: #0d1117 !important;
            border: 1px solid #30363d !important;
            transition: all 0.25s ease !important;
        }}
        [data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover {{
            border-color: #40e0d0 !important;
            background: #161b22 !important;
            box-shadow: 0 0 10px rgba(64, 224, 208, 0.3) !important;
        }}
        [data-testid="stMain"] [data-testid="stBaseButton-secondary"] div[data-testid="stMarkdownContainer"] p {{
            white-space: pre-line !important;
            text-align: center !important;
            line-height: 1.1 !important;
            font-size: 14px !important; 
            margin: 0 !important;
            background: linear-gradient(to bottom, #ffffff 60%, #40e0d0 60%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        [data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover p {{
            background: linear-gradient(to bottom, #ffffff 60%, #ffffff 60%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        </style>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    for j, (tick, name) in enumerate(SP100_SECTORS[sector_choice]):
        with cols[j % 5]:
            if st.button(f"{name}\n({tick})", key=f"btn_{sector_choice}_{tick}", use_container_width=True):
                st.session_state['current_ticker'] = tick
                st.session_state['current_page'] = "Dashboard"
                st.rerun()


#Analytics DASHBOARD page 
elif st.session_state['current_page'] == "Dashboard":
    st.title("Stock Analytics Dashboard")
    st.caption("Deep Learning framework for multimodal market analysis")
    st.divider()

    # INLINE SEARCH BAR
    _, search_col, _ = st.columns([1, 2, 1])
    with search_col:
        st.markdown("<p style='font-size: 14px; margin-bottom: 5px; color: #c9d1d9; font-weight: bold;'>Search Ticker or Company Name</p>", unsafe_allow_html=True)
        col_input, col_btn = st.columns([3, 1.2]) 
        
        with col_input:
            initial_val = st.session_state['current_ticker'] if st.session_state['current_ticker'] else ""
            ticker_input = st.text_input("Search", value=initial_val, placeholder="e.g. NVDA or NVIDIA", label_visibility="collapsed").strip()
            
        with col_btn:
            analyze_btn = st.button("Run analysis", type="primary", use_container_width=True)

    current_ticker = fuzzy_ticker_search(ticker_input, api) if ticker_input else None

    if not analyze_btn and current_ticker not in st.session_state['analysis_cache']:
        _, console_col, _ = st.columns([1, 2, 1])
        with console_col:
            if not ticker_input and not st.session_state.get('current_ticker'):
                st.markdown("""
                    <div style='background-color: rgba(64, 224, 208, 0.05); border: 1px solid rgba(64, 224, 208, 0.3); border-radius: 8px; padding: 25px; text-align: left;'>
                        <h4 style='color: #40e0d0; margin: 0 0 8px 0; font-size: 18px; font-weight: bold; line-height: 1.2;'>
                            Predictive Engine
                        </h4>
                         <p style='color: #8b949e; font-size: 14px; margin: 0; line-height: 1.6;'>
                            Initialise the multimodal stream by entering a valid equity ticker or company name. 
                             The system will synchronise 60 days of historical pricing with real-time NLP sentiment analysis.
                         </p>
                    </div>
          
                    
                """, unsafe_allow_html=True)
                st.stop()
            else:
                # State 2 of the ready to Run 
                target = st.session_state.get('current_ticker') or ticker_input
                st.markdown(f"""
                    <div style='background-color: #161b22; border: 1px solid #40e0d0; border-left: 5px solid #40e0d0; border-radius: 8px; padding: 20px;'>
                        <div style='display: flex; align-items: center;'>
                            <div>
                                <h4 style='color: #ffffff; margin: 0; font-size: 16px;'>Ready to Analyze: <span style='color: #40e0d0;'>{target}</span></h4>
                                <p style='color: #8b949e; font-size: 13px; margin: 5px 0 0 0;'>Configuration: [Dual-Stream LSTM/GRU] + [FinBERT Sentiment Engine]</p>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.stop()
            
    
    if analyze_btn or (current_ticker and current_ticker not in st.session_state['analysis_cache']):
        
        if not current_ticker:
            st.stop()
            
        st.session_state['current_ticker'] = current_ticker
        
        ui_placeholder = st.empty()
        with ui_placeholder.container():
            st.markdown(f"### Building analysis of market trends for {current_ticker}...")
            c1, c2, c3 = st.columns(3)
            for col in [c1, c2, c3]:
                col.markdown('<div class="skeleton-box" style="height: 120px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)
            st.markdown('<div class="skeleton-box" style="height: 450px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)
            c_news, c_sum = st.columns([1.5, 1])
            c_news.markdown('<div class="skeleton-box" style="height: 350px;"></div>', unsafe_allow_html=True)
            c_sum.markdown('<div class="skeleton-box" style="height: 350px;"></div>', unsafe_allow_html=True)

        with st.spinner(f"Synchronising neural price and news streams for {current_ticker}..."):
            now_est = pd.Timestamp.now(tz='America/New_York')
            end_date = now_est - pd.Timedelta(minutes=20) 
            start_date = (end_date - pd.Timedelta(days=150)).isoformat()
            
            try:
                bars = api.get_bars(current_ticker, tradeapi.TimeFrame.Day, start=start_date, end=end_date.isoformat()).df
                spy = api.get_bars('SPY', tradeapi.TimeFrame.Day, start=start_date, end=end_date.isoformat()).df
                news = api.get_news(current_ticker, start=(end_date - pd.Timedelta(days=40)).isoformat(), limit=40)
                
                if bars.empty or spy.empty:
                    ui_placeholder.empty()
                    st.error(f"❌ Market data unavailable for **{current_ticker}**. Please try a different symbol.")
                    st.stop()
                
                bars.columns = [c.lower() for c in bars.columns]
                spy.columns = [c.lower() for c in spy.columns]
            except Exception as e:
                ui_placeholder.empty()
                st.error(f"API Error: {str(e)}")
                st.stop()

            for frame in [bars, spy]:
                frame.index = frame.index.normalize().tz_localize(None)

            df = pd.DataFrame(index=bars.index)
            df['open'] = bars['open']
            df['high'] = bars['high']
            df['low'] = bars['low']
            df['close'] = bars['close'] 
            df['log_return'] = np.log(bars['close'] / bars['close'].shift(1))
            df['spy_log_return'] = np.log(spy['close'] / spy['close'].shift(1))
            
            delta = bars['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi_14'] = 100 - (100 / (1 + (gain / loss)))
            
            exp1 = bars['close'].ewm(span=12, adjust=False).mean()
            exp2 = bars['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['bb_width'] = (bars['close'].rolling(20).std() * 4) / bars['close'].rolling(20).mean()
            
            price_slice = df[['close', 'log_return', 'rsi_14', 'macd', 'macd_signal', 'bb_width', 'spy_log_return']].dropna().tail(60)

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
                
                headlines_ui.append({
                    "ts": ts.strftime('%b %d'), 
                    "text": article.headline, 
                    "sentiment": label,
                    "url": getattr(article, 'url', '#') 
                })

            sent_aligned = [np.mean(sent_map[d], axis=0) if d in sent_map else [0.0, 0.0, 1.0] for d in price_slice.index]
            
            scaler = joblib.load(os.path.join('data', 'scaler.pkl'))
            model = tf.keras.models.load_model(
                os.path.join('data', 'models', 'v4_champion_model.keras'), 
                custom_objects={'TemporalAttention': TemporalAttention, 'directional_accuracy': directional_accuracy}
            )
            
            model_features = price_slice[['log_return', 'rsi_14', 'macd', 'macd_signal', 'bb_width', 'spy_log_return']]
            scaled_features = np.clip(scaler.transform(model_features.values), -3.0, 3.0)
            
            prob = model.predict([scaled_features.reshape(1, 60, 6), np.array(sent_aligned).reshape(1, 60, 3)], verbose=0)[0][0]
            
            news_bias = (tally['POS'] - tally['NEG']) / (sum(tally.values()) + 1)
            adjusted_prob = np.clip(prob + (news_bias * 0.1), 0.01, 0.99)
            latest = price_slice.iloc[-1]
            
            outlook = "BULLISH" if adjusted_prob > 0.5 else "BEARISH"
            intensity = "HIGH CONFIDENCE" if adjusted_prob > 0.85 else "MODERATE" if adjusted_prob > 0.65 else "NOISY"
            vol_state = "STABLE / TRENDING" if latest['bb_width'] < 0.08 else "Volatile"

            st.session_state['analysis_cache'][current_ticker] = {
                'bars_df': df, 
                'latest': latest,
                'outlook': outlook,
                'intensity': intensity,
                'vol_state': vol_state,
                'adjusted_prob': adjusted_prob,
                'headlines_ui': headlines_ui,
                'tally': tally
            }
            
            st.toast(f"Synchronised analytics for {current_ticker}")
            ui_placeholder.empty()

    #RENDER the UI FROM CACHE
    if current_ticker in st.session_state['analysis_cache']:
        cached = st.session_state['analysis_cache'][current_ticker]
        bars_df = cached['bars_df']
        latest = cached['latest']
        outlook = cached['outlook']
        intensity = cached['intensity']
        vol_state = cached['vol_state']
        adjusted_prob = cached['adjusted_prob']
        headlines_ui = cached['headlines_ui']
        tally = cached['tally']
        
        st.subheader(":material/memory: Market sentiment & outlook summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            color = "#2ea043" if outlook == "BULLISH" else "#e74c3c"
            st.markdown(f"""
            <div class="stMetric">
                <p style='margin:0; font-size:13px; color:#8b949e'>5-DAY NEURAL OUTLOOK <span class='tooltip-icon' title='Neural prediction of directional bias over next 5 trading days.'>&#9432;</span></p>
                <p style='margin:0; font-size:22px; font-weight:bold; color:{color};'>{outlook} {'🟢' if outlook == 'BULLISH' else '🔴'}</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stMetric">
                <p style='margin:0; font-size:13px; color:#8b949e'>PATTERN ALIGNMENT STRENGTH <span class='tooltip-icon' title='Measures the Signal-to-Noise ratio of the neural forecast. 'HIGH CONFIDENCE' indicates strong alignment between price patterns and news sentiment. 'NOISY' indicates ambiguity'>&#9432;</span></p>
                <p style='margin:0; font-size:22px; font-weight:bold;'>Tier: {intensity}</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            color = "#2ea043" if vol_state == "STABLE / TRENDING" else "#c9d1d9"
            st.markdown(f"""
            <div class="stMetric">
                <p style='margin:0; font-size:13px; color:#8b949e'>MARKET ENVIRONMENT <span class='tooltip-icon' title='Evaluates the current price environment. 'STABLE / TRENDING' suggests the stock is moving smoothly within its historical bounds, while 'Volatile' warns of aggressive, unpredictable price jumps'>&#9432;</span></p>
                <p style='margin:0; font-size:22px; font-weight:bold; color:{color};'>{vol_state}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown(f"### :material/candlestick_chart: <span style='color:#40e0d0;'>{current_ticker}</span> Price momentum graph (60-Day)", unsafe_allow_html=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Candlestick(
            x=bars_df.index[-60:], open=bars_df['open'][-60:], high=bars_df['high'][-60:],
            low=bars_df['low'][-60:], close=bars_df['close'][-60:], name="Price Action"
        ))

        fig.add_trace(go.Scatter(
            x=bars_df.index[-60:], y=bars_df['close'][-60:],
            mode='lines',
            line=dict(color='rgba(201, 209, 217, 0.4)', width=1.5), 
            name="Closing Trend",
            hoverinfo='skip'
        ))

        fig.update_layout(
            template="plotly_dark", 
            height=450, 
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis_title="Trading Date", 
            yaxis_title="Asset Price (USD)",
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(":material/feed: Market Intelligence Stream")
        
        col_news, col_summary = st.columns([1.5, 1])
        
        with col_news:
            st.markdown("**:material/newspaper: Latest headlines for this stock**")
            st.caption("✦ Key: 🟢 Positive | 🔴 Negative | ⚪ Neutral")
            st.caption("Click to view source")

            with st.container(height=350, border=True):
                for h in headlines_ui:
                    icon = "🟢" if h['sentiment'] == "POS" else "🔴" if h['sentiment'] == "NEG" else "⚪"
                    st.markdown(f"{icon} [{h['text']}]({h['url']}) <span style='color:#8b949e; font-size:12px;'>{h['ts']}</span>", unsafe_allow_html=True)

        with col_summary:
            st.markdown("**:material/smart_toy: Model Inference Summary**")
            
            st.caption("Disclaimer: This summary is generated by the deep learning model Inference from the dual stream architechture")

            #narrative box
            narrative_text = generate_heuristic_narrative(outlook, tally, (latest['log_return'] - latest['spy_log_return']) * 100, latest['rsi_14'], current_ticker)
            with st.container(height=350, border=True):
                st.markdown(narrative_text, unsafe_allow_html=True)
            
            #referncing my model
            st.markdown("<p style='color: #8b949e; font-size: 11px; font-style: italic;'>Neural Multimodal Fusion Engine v4.0</p>", unsafe_allow_html=True)
           
        st.divider()

        st.subheader(":material/query_stats: Secondary Indicator Benchmarks")
        
        met1, met2, met3 = st.columns(3)
        with met1:
            rel_perc = (latest['log_return'] - latest['spy_log_return']) * 100
            rel_color = "#2ea043" if rel_perc > 0 else "#da3633"
            st.markdown(f"""
            <div class="stMetric">
                <p style='margin:0; font-size:13px; color:#8b949e'>PERFORMANCE VS S&P 500 <span class='tooltip-icon' title='Daily outperformance/underperformance relative to the broader market index.'>&#9432;</span></p>
                <p style='margin:0; font-size:22px; font-weight:bold; color:{rel_color};'>{rel_perc:+.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
        with met2:
            st.markdown(f"""
            <div class="stMetric">
                <p style='margin:0; font-size:13px; color:#8b949e'>TECHNICAL RSI <span class='tooltip-icon' title='Relative Strength Index. Values > 70 are overbought, < 30 are oversold.'>&#9432;</span></p>
                <p style='margin:0; font-size:22px; font-weight:bold;'>{latest['rsi_14']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        with met3:
            st.markdown(f"""
            <div class="stMetric">
                <p style='margin:0; font-size:13px; color:#8b949e'>BB WIDTH <span class='tooltip-icon' title='Bollinger Band Width. Quantitative measure of market volatility/squeeze.'>&#9432;</span></p>
                <p style='margin:0; font-size:22px; font-weight:bold;'>{latest['bb_width']:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.subheader("Learning Centre")
        st.caption("Understanding model logic and technical terminology")
        
        learn1, learn2, learn3 = st.columns(3)
        with learn1:
            with st.expander("How multimodal forecasting works"):
                st.markdown("""
                The core engine synthesizes 60 days of historical data through a **dual-stream architecture**. 
                By fusing **quantitative price action** with **qualitative sentiment**, the model identifies 
                non-linear patterns to project a 5-day directional bias.
                """)
            
        with learn2:
            with st.expander("Deciphering Technical Ratings"):
                st.markdown("""
                Technical ratings evaluate statistical price trajectories and volume trends. 
                Unlike fundamental analysis, these metrics identify **behavioral patterns** and 
                **market exhaustion points** to gauge short-term directional strength.
                """)
        with learn3:
            with st.expander("Interpreting Market Oscillators"):
                st.markdown("""
                Oscillators like the Relative Strength Index (RSI) measure the velocity of price changes. We use this to identify 'Overbought' or 'Oversold' conditions, indicating when a price trend may be overextended and due for a correction
                """)
        
        st.divider()

        with st.expander("About the model engine behind the analysis  - Neural architecture"):
            st.subheader("Multimodal Dual-Stream Fusion")
            st.markdown("""
            This platform utilizes a **Custom LSTM & GRU Neural Network**, taking into account historical price data and current news headlines to determine directional bias. The system uses the following data:
            1. **Quantitative Matrix:** A 6-feature temporal array tracking Log Returns, RSI, MACD, and Volatility
            2. **Semantic Vector:** FinBERT NLP embeddings translating live news headlines into mathematical sentiment
            """)
            st.write("To prevent 'Overconfidence Bias' (Sigmoid Saturation), extreme quantitative probabilities are translated into a qualitative 'Pattern Alignment Strength' tier, ensuring investors understand this is a directional bias, not a mathematical certainty.")

        st.caption("This dashboard is for educational and research purposes only. Do your own research before making investment decisions. **This does not constitute to financial advice**")