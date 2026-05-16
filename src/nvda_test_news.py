import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
# Registering your custom layers
from src.predict_live import TemporalAttention, directional_accuracy

# 1. DEFINE UPDATED ONEDRIVE PATHS
base_path = r"C:\Users\mistr\OneDrive - Aston University\data"
models_path = os.path.join(base_path, "models")

print("--- INITIATING CHAMPION VISUALIZATION: NVDA ---")

# 2. LOAD V4 CHAMPION MODEL
model_path = os.path.join(models_path, "v4_champion_model.keras")
model = tf.keras.models.load_model(
    model_path, 
    custom_objects={'TemporalAttention': TemporalAttention, 'directional_accuracy': directional_accuracy}
)

# 3. LOAD TENSORS AND METADATA
# Price/Sentiment are in 'data', metadata is in 'models'
X_price = np.load(os.path.join(base_path, "v4_X_price.npy"))
X_sent = np.load(os.path.join(base_path, "v4_X_sentiment.npy"))
metadata = np.load(os.path.join(models_path, "v4_meta_data.npy"), allow_pickle=True)

# 4. FILTER FOR NVDA
nvda_indices = np.where(metadata == 'NVDA')[0]

if len(nvda_indices) > 0:
    # We take the last available NVDA sample (most recent training state)
    target_idx = nvda_indices[-1]
    print(f"✅ NVDA Tensors found. Analyzing sample index: {target_idx}")

    # 5. CONSTRUCT ATTENTION EXTRACTOR
    # Using confirmed layer name from Index 9 of your inspector
    attention_layer = model.get_layer('price_temporal_attention')
    debug_model = Model(inputs=model.input, outputs=attention_layer.output)
    
    # Extract the weights (Forward Pass)
    sample_input = [X_price[target_idx:target_idx+1], X_sent[target_idx:target_idx+1]]
    attention_weights = debug_model.predict(sample_input)

    # 6. GENERATE FINAL DISSERTATION GRAPH
    plt.figure(figsize=(15, 3))
    
    # Plotting the 60-day window (T-60 on left, Today/T-0 on right)
    sns.heatmap(
        attention_weights.reshape(1, 60), 
        cmap="YlGnBu", 
        cbar=True, 
        cbar_kws={'label': 'Neural Importance'}
    )
    
    plt.title("NVDA: Neural Attention Heatmap (V4 Cross-Attention Engine)")
    plt.xlabel("Days Ago (Lookback Window: T-60 to T-0)")
    plt.ylabel("Attention")
    
    # Display results
    plt.tight_layout()
    plt.show()
    print("✅ Heatmap generation complete.")
else:
    print("❌ Error: NVDA not found in the metadata mapping.")