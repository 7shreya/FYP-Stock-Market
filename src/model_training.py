import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Concatenate, BatchNormalization, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.model_selection import TimeSeriesSplit

# Force CPU if local CUDA is not configured, preventing driver crashes
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- 1. CUSTOM METRICS & LAYERS ---

@tf.keras.utils.register_keras_serializable()
def directional_accuracy(y_true, y_pred):
    """Calculates the percentage of time the model correctly guesses the market direction (+/-)"""
    y_true_sign = tf.sign(y_true)
    y_pred_sign = tf.sign(y_pred)
    correct = tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)
    return tf.reduce_mean(correct) * 100

@tf.keras.utils.register_keras_serializable()
class TemporalAttention(Layer):
    """Custom Attention mechanism to provide Explainable AI (XAI) capabilities"""
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


# --- 2. DATA LOADING & MODEL ARCHITECTURE ---

def load_tensors():
    print("loading chronologically sorted tensors from disk...")
    if not os.path.exists('../data/X_price.npy'):
        raise FileNotFoundError("Tensors not found. Run build_sequences.py first.")
        
    X_price = np.load('../data/X_price.npy')
    X_sentiment = np.load('../data/X_sentiment.npy')
    y_target = np.load('../data/y_target.npy')
    return X_price, X_sentiment, y_target

def build_dual_stream_model(price_shape, sentiment_shape):
    # Stream 1: Price Dynamics (LSTM)
    price_input = Input(shape=price_shape, name='price_input')
    x1 = LSTM(64, return_sequences=True)(price_input)
    x1 = Dropout(0.2)(x1)
    x1 = TemporalAttention(name='price_attention')(x1)
    x1 = BatchNormalization()(x1)
    
    # Stream 2: News Sentiment (GRU)
    sentiment_input = Input(shape=sentiment_shape, name='sentiment_input')
    x2 = GRU(32, return_sequences=True)(sentiment_input)
    x2 = Dropout(0.2)(x2)
    x2 = TemporalAttention(name='sentiment_attention')(x2)
    x2 = BatchNormalization()(x2)
    
    # Late Fusion
    merged = Concatenate()([x1, x2])
    z = Dense(32, activation='relu')(merged)
    z = Dropout(0.2)(z)
    output = Dense(1, activation='linear', name='target_return')(z)
    
    model = Model(inputs=[price_input, sentiment_input], outputs=output)
    
    # Huber Loss protects against black-swan anomalies
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss=tf.keras.losses.Huber(), 
                  metrics=['mae', directional_accuracy])
    return model


# --- 3. THE TRAINING ENGINE ---

def train_network():
    X_price, X_sentiment, y_target = load_tensors()
    
    # 5-Fold Walk-Forward Validation to test multiple economic regimes
    tscv = TimeSeriesSplit(n_splits=5)
    
    fold = 1
    fold_scores = []
    
    print("\n--- INITIATING WALK-FORWARD VALIDATION ---")
    
    for train_idx, val_idx in tscv.split(X_price):
        print(f"\n[ Training Fold {fold}/5 ]")
        
        tf.keras.backend.clear_session() # Prevent RAM memory leaks between folds
        
        X_price_train, X_price_val = X_price[train_idx], X_price[val_idx]
        X_sent_train, X_sent_val = X_sentiment[train_idx], X_sentiment[val_idx]
        y_train, y_val = y_target[train_idx], y_target[val_idx]
        
        model = build_dual_stream_model(
            price_shape=(X_price.shape[1], X_price.shape[2]),
            sentiment_shape=(X_sentiment.shape[1], X_sentiment.shape[2])
        )
        
        # Callbacks: Stop if overfitting, reduce learning rate if plateaued
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]
        
        model.fit(
            x=[X_price_train, X_sent_train],
            y=y_train,
            validation_data=([X_price_val, X_sent_val], y_val),
            epochs=30,
            batch_size=64,
            callbacks=callbacks,
            shuffle=False, # STRICTLY FALSE to prevent time-series data leakage
            verbose=1
        )
        
        # Evaluate Fold
        val_loss, val_mae, dir_acc = model.evaluate([X_price_val, X_sent_val], y_val, verbose=0)
        print(f"Fold {fold} Results -> Huber Loss: {val_loss:.6f} | Dir Acc: {dir_acc:.2f}%")
        fold_scores.append(dir_acc)
        
        # --- DISSERTATION ARTIFACT GENERATION (Final Fold Only) ---
        if fold == 5:
            os.makedirs('../models', exist_ok=True)
            model.save('../models/dual_stream_attention.keras')
            print("\nFinal model saved to disk.")
            
            print("Exporting prediction artifacts for dissertation analysis...")
            predictions = model.predict([X_price_val, X_sent_val]).flatten()
            
            results_df = pd.DataFrame({
                'Actual_Log_Return': y_val,
                'Predicted_Log_Return': predictions,
                'Correct_Direction': np.sign(y_val) == np.sign(predictions)
            })
            
            results_df.to_csv('../data/final_fold_predictions.csv', index=False)
            print("Saved final_fold_predictions.csv. Use this for your methodology graphs.")
            
        fold += 1
        
    print("\n=== FINAL OVERALL PERFORMANCE ===")
    print(f"Average Directional Accuracy Across All Regimes: {np.mean(fold_scores):.2f}%")

if __name__ == '__main__':
    train_network()