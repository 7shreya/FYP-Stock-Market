# src/train_model.py
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from src.feature_engineering import create_training_set
from src.config import SEQ_LENGTH, MODELS_DIR

# Ensure directories exist for saving our outputs
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("results", exist_ok=True) 

def prepare_lstm_data(df, ticker):
    """
    Transforms tabular data into 3D sequences for the LSTM
    Enforces a chronological split
    """
    print(f"Starting data preprocessing for: {ticker}")
    
    # We use multiple features to provide full market context
    features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']
    data = df[features].values
    
    # Scale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler for later (to convert predictions back to real prices)
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    target_col_index = features.index('close')
    
    X, y = [], []
    # Create the sliding window (eg look at 60 days to predict day 61)
    for i in range(SEQ_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LENGTH:i])
        y.append(scaled_data[i, target_col_index])
        
    X, y = np.array(X), np.array(y)
    
    # Strict 80/20 Chronological Split (No random splitting)
    split_index = int(len(X) * 0.8)
    
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    print(f"Data formatting complete. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, X_test, y_train, y_test, scaler

def build_advanced_lstm(input_shape):
    """
    Builds the neural network with built-in defenses against overfitting.
    """
    print("Initialising Advanced LSTM Architecture...")
    model = Sequential()
    
    # Layer 1: LSTM with L2 Regularization
    model.add(LSTM(
        units=100, 
        return_sequences=True, 
        input_shape=(input_shape[1], input_shape[2]),
        kernel_regularizer=l2(0.001) # Penalizes extreme weights to force generalization
    ))
    model.add(Dropout(0.3)) # 30% dropout 
    
    # Layer 2: Deep LSTM layer
    model.add(LSTM(
        units=50, 
        return_sequences=False,
        kernel_regularizer=l2(0.001)
    ))
    model.add(Dropout(0.3))
    
    # Layer 3: Fully connected Dense layers
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1, activation='linear')) # Linear output for regression (price prediction)
    
    # Custom Adam optimizer 
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

def plot_training_history(history, ticker):
    """
    Saves a graph of the training process
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title(f'Model Training History: {ticker}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join("results", f"{ticker}_training_history.png")
    plt.savefig(plot_path)
    print(f"Training history plot saved successfully to: {plot_path}")

def train_and_save_model(ticker):
    df = create_training_set(ticker)
    
    if df is None or len(df) < SEQ_LENGTH * 2:
        print(f"Error: Insufficient data for {ticker} Minimum required days: {SEQ_LENGTH * 2}")
        return None
        
    X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(df, ticker)
    model = build_advanced_lstm(X_train.shape)
    
    model_path = os.path.join(MODELS_DIR, f"{ticker}_lstm_model.keras")
    
    # 1. Checkpoint: Only save the model if validation loss improves
    checkpoint = ModelCheckpoint(
        filepath=model_path, 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
    
    # 2. Early Stopping: Halt training if the model stops learning for 10 epochs
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    # 3. Learning Rate Scheduler: Take smaller learning steps if progress stalls
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=5, 
        min_lr=0.00001,
        verbose=1
    )
    
    print("Initiating model training sequence...")
    history = model.fit(
        X_train, y_train,
        batch_size=32, 
        epochs=100, # Set high, but EarlyStopping will catch it before 100
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping, lr_scheduler],
        verbose=1
    )
    
    # Generate the graph for your report
    plot_training_history(history, ticker)
    
    print("Training sequence complete. Optimal model weights have been saved")
    return model

if __name__ == "__main__":
    # Test the pipeline with a data-rich stock like NVIDIA or Apple
    train_and_save_model("NVDA")