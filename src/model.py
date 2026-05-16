# src/model.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate

def build_dual_stream_model(seq_length=60):
    """
    Constructs a true Dual-Stream Hybrid Neural Network using the Keras Functional API.
    Isolates price / volume data from sentiment data to prevent feature drowning,
    fusing them late in the network for a unified prediction.
    """
    
    # ---------------------------------------------------------
    # STREAM 1: Technical Analysis (Price & Volume - 5 features)
    # ---------------------------------------------------------
    price_input = Input(shape=(seq_length, 5), name="Price_Volume_Input")
    
    # Deep LSTM for complex price pattern recognition
    lstm_price_1 = LSTM(units=128, return_sequences=True)(price_input)
    drop_price_1 = Dropout(0.2)(lstm_price_1)
    
    lstm_price_2 = LSTM(units=64, return_sequences=False)(drop_price_1)
    price_features = Dropout(0.2)(lstm_price_2)

    # ---------------------------------------------------------
    # STREAM 2: Fundamental Analysis (FinBERT Sentiment - 1 feature)
    # ---------------------------------------------------------
    sentiment_input = Input(shape=(seq_length, 1), name="Sentiment_Input")
    
    # Lighter LSTM to track the momentum/trend of the news mood over 60 days
    lstm_sentiment = LSTM(units=32, return_sequences=False)(sentiment_input)
    sentiment_features = Dropout(0.2)(lstm_sentiment)

    # ---------------------------------------------------------
    # LATE FUSION: Combining the Streams
    # ---------------------------------------------------------
    merged_layer = Concatenate(name="Feature_Fusion")([price_features, sentiment_features])
    
    # Fully Connected Dense Layers to interpret the fused data
    dense_1 = Dense(units=32, activation='relu')(merged_layer)
    
    # Output Layer - A single node predicting the Day 61 Close Price
    final_output = Dense(units=1, activation='linear', name="Price_Prediction")(dense_1)

    # Compile the comprehensive model
    model = Model(inputs=[price_input, sentiment_input], outputs=final_output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    # Test the architecture to ensure it compiles correctly
    test_model = build_dual_stream_model()
    test_model.summary()
    print("\nSUCCESS: Dual-Stream Hybrid Architecture Compiled.")