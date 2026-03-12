# src/processor.py
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

INPUT_DIR = "data/training_processed"
OUTPUT_DIR = "data/processed_arrays"
MODELS_DIR = "models/scalers"
SEQUENCE_LENGTH = 60
TRAIN_SPLIT = 0.80

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def create_sequences(data, seq_length):
    X, y = [], []
    close_idx = 3 
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length, close_idx])
    return np.array(X), np.array(y)

def process_watertight_pipeline():
    csv_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']
    
    train_dfs, val_dfs = [], []
    
    for file in tqdm(csv_files, desc="Temporal Splitting"):
        df = pd.read_csv(file).dropna(subset=features)
        if len(df) <= SEQUENCE_LENGTH + 10: continue
            
        split_idx = int(len(df) * TRAIN_SPLIT)
        train_dfs.append(df.iloc[:split_idx][features])
        val_dfs.append(df.iloc[split_idx:][features])

    print("Fitting Scaler STRICTLY on Training Data...")
    master_train_df = pd.concat(train_dfs, ignore_index=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(master_train_df)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "global_scaler.pkl"))
    
    print("Scaling and Building Sequences...")
    X_train_list, y_train_list, X_val_list, y_val_list = [], [], [], []
    
    for t_df, v_df in tqdm(zip(train_dfs, val_dfs), total=len(train_dfs)):
        scaled_train = scaler.transform(t_df)
        scaled_val = scaler.transform(v_df)
        
        X_t, y_t = create_sequences(scaled_train, SEQUENCE_LENGTH)
        X_v, y_v = create_sequences(scaled_val, SEQUENCE_LENGTH)
        
        if len(X_t) > 0:
            X_train_list.append(X_t)
            y_train_list.append(y_t)
        if len(X_v) > 0:
            X_val_list.append(X_v)
            y_val_list.append(y_v)

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), np.vstack(X_train_list))
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), np.concatenate(y_train_list))
    np.save(os.path.join(OUTPUT_DIR, "X_val.npy"), np.vstack(X_val_list))
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), np.concatenate(y_val_list))
    print("SUCCESS: 4 isolated arrays created.")

if __name__ == "__main__":
    process_watertight_pipeline()