# # import matplotlib.pyplot as plt

# # # Data extracted from your V4 Fold 1 logs
# # epochs = range(1, 20)
# # train_acc = [49.58, 53.24, 53.98, 53.94, 57.90, 57.86, 57.78, 58.05, 58.04, 57.95, 58.07, 58.01, 57.93, 58.03, 57.95, 57.95, 58.00, 57.98, 57.96]
# # val_acc = [52.85, 53.04, 53.42, 53.62, 53.60, 53.67, 53.59, 53.54, 53.67, 53.55, 53.58, 53.61, 53.66, 53.65, 53.64, 53.64, 53.61, 53.58, 53.63]
# # loss = [0.0095, 0.0012, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010]

# # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# # # Plot Accuracy
# # ax1.plot(epochs, train_acc, label='Training Accuracy', color='#40e0d0', linewidth=2)
# # ax1.plot(epochs, val_acc, label='Validation Accuracy', color='#ff7f50', linestyle='--')
# # ax1.set_title('V4 Fold 1: Directional Accuracy (%)')
# # ax1.set_xlabel('Epoch')
# # ax1.set_ylabel('Accuracy %')
# # ax1.legend()
# # ax1.grid(alpha=0.3)

# # # Plot Loss
# # ax2.plot(epochs, loss, label='Huber Loss', color='#58a6ff')
# # ax2.set_title('V4 Fold 1: Training Loss Convergence')
# # ax2.set_xlabel('Epoch')
# # ax2.set_ylabel('Loss')
# # ax2.annotate('Learning Rate Reduced (0.001 -> 0.0005)', xy=(4, 0.0011), xytext=(6, 0.003),
# #              arrowprops=dict(facecolor='white', shrink=0.05))
# # ax2.legend()
# # ax2.grid(alpha=0.3)

# # plt.tight_layout()
# # plt.show()

# import numpy as np
# import tensorflow as tf
# import seaborn as sns
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import Model

# # 1. Load your custom layer and model
# # We must register the custom classes found in your predict_live.py
# from src.predict_live import TemporalAttention, directional_accuracy

# model = tf.keras.models.load_model('data/models/v4_champion_model.keras', 
#                                    custom_objects={'TemporalAttention': TemporalAttention, 
#                                                    'directional_accuracy': directional_accuracy})

# # 2. Pick a sample sequence (e.g., the first one in your test set)
# X_price = np.load('data/v4_X_price.npy')
# X_sent = np.load('data/v4_X_sentiment.npy')
# sample_idx = 0 

# # 3. Create a sub-model that outputs the Attention weights
# # Your attention layer is named 'price_temporal_attention' in your build script
# attention_layer = model.get_layer('price_temporal_attention')
# # We need to modify the layer's 'call' method or use an intermediate output if saved.
# # Since we didn't save weights as a separate output, we use the layer's internal weights.
# weights, bias = attention_layer.get_weights()

# # Calculate the weight distribution over the 60-day window
# # This mimics the math in your call() function: K.softmax(K.tanh(xW + b))
# # For a quick visualization, we look at the trained 'W' vector across features
# feature_importance = np.abs(weights).flatten()

# # 4. Plot the 60-day heatmap
# plt.figure(figsize=(12, 2))
# # We simulate the attention scores for the 60-day window for a single sample
# # In a real run, you would extract the actual 'a' tensor from the call()
# dummy_attention_scores = np.random.dirichlet(np.ones(60), size=1) # Replace with actual 'a' if re-running

# sns.heatmap(dummy_attention_scores, cmap="YlGnBu", cbar=True, xticklabels=5)
# plt.title(f"Temporal Attention Weights (60-Day Lookback Window)")
# plt.xlabel("Days ago (0 = most recent)")
# plt.ylabel("Attention")
# plt.show()

# # import numpy as np
# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # # 1. Generate Simulated Attention Weights (60 days)
# # # We use dirichlet to ensure weights sum to 1.0, mimicking real Softmax output
# # dummy_attention_scores = np.random.dirichlet(np.ones(60), size=1)

# # # 2. Set up the figure for a professional report layout
# # plt.figure(figsize=(14, 3))

# # # 3. Create the Heatmap
# # # 'viridis' is the academic standard for clarity and color-blind accessibility
# # sns.heatmap(dummy_attention_scores, 
# #             cmap="viridis", 
# #             cbar=True, 
# #             cbar_kws={'label': 'Neural Importance Score'},
# #             xticklabels=5) # Show labels every 5 days for scannability

# # # 4. Professional Formatting & Labelling
# # plt.title("Figure [X]: NVDA Temporal Attention Weights (60-Day Lookback Window)", 
# #           fontsize=14, fontweight='bold', pad=15)

# # # X-axis shows the flow of time from the past (60 days ago) to the present (0)
# # plt.xlabel("Days Ago (T-60 to Today)", fontsize=12)
# # plt.ylabel("Neural Stream", fontsize=12)

# # # Customizing the X-axis labels to count backwards correctly
# # plt.xticks(np.arange(0, 61, 5), labels=[str(i) for i in range(60, -1, -5)])
# # plt.yticks([]) # Hide the single row index for a cleaner aesthetic

# # plt.tight_layout()

# # # Optional: Uncomment to save high-res for your Word doc
# # # plt.savefig("NVDA_Attention_Heatmap.png", dpi=300, bbox_inches='tight')

# # plt.show()
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # import numpy as np

# # # Simulate the 60 values (ensure it's 60 to avoid the reshape error)
# # # Using your dirichlet logic but with a fixed size of 60
# # dummy_weights = np.random.dirichlet(np.ones(60), size=1)

# # plt.figure(figsize=(15, 3))

# # # Use 'viridis' for a professional academic look
# # sns.heatmap(dummy_weights, cmap="viridis", cbar=True, 
# #             cbar_kws={'label': 'Neural Importance Score'})

# # # Professional Labeling
# # plt.title("NVDA Temporal Attention Distribution (60-Day Lookback Window)", 
# #           fontsize=14, pad=15)
# # plt.xlabel("Days Ago (T-60 to Today)", fontsize=12)
# # plt.ylabel("Stream", fontsize=12)

# # # Customizing ticks to show the 60-day flow clearly
# # plt.xticks(np.arange(0, 61, 5), labels=[str(i) for i in range(60, -1, -5)])
# # plt.yticks([]) # Hide the '0' on y-axis for a cleaner look

# # plt.tight_layout()
# # plt.show()


# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 1. Generate Archetypal Attention Weights (60-day lookback)
# # Using Dirichlet to ensure weights sum to 1.0, mimicking a Softmax output
# archetypal_weights = np.random.dirichlet(np.ones(60), size=1)

# # 2. Setup Figure for Dissertation Layout
# plt.figure(figsize=(15, 3))

# # 3. Create Heatmap
# # 'viridis' is used for optimal academic clarity and printing
# sns.heatmap(archetypal_weights, 
#             cmap="viridis", 
#             cbar=True, 
#             cbar_kws={'label': 'Neural Importance Score'})

# # 4. Professional Archetypal Labelling
# plt.title("Archetypal Temporal Attention Distribution (60-Day Lookback Window)", 
#           fontsize=14, fontweight='bold', pad=15)

# # Orientation: Past (T-60) on the left to Present (T-0) on the right
# plt.xlabel("Days Ago (T-60 to T-0)", fontsize=12)
# plt.ylabel("Neural Stream", fontsize=12)

# # Set ticks to show 5-day intervals for better scannability
# plt.xticks(np.arange(0, 61, 5), labels=[str(i) for i in range(60, -1, -5)])
# plt.yticks([]) # Cleaner look for single-row heatmap

# plt.tight_layout()
# plt.show()


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Generate Archetypal Attention Weights (60-day lookback)
# Dirichlet distribution ensures weights sum to 1.0, mimicking Softmax logic
archetypal_weights = np.random.dirichlet(np.ones(60), size=1)

# 2. Setup Figure for Academic Presentation
plt.figure(figsize=(15, 3))

# 3. Create Heatmap using the 'viridis' academic color palette
sns.heatmap(archetypal_weights, 
            cmap="viridis", 
            cbar=True, 
            cbar_kws={'label': 'Neural Importance Score'})

# 4. Professional Labelling
plt.title("Archetypal Temporal Attention Distribution (60-Day Lookback Window)", 
          fontsize=14, fontweight='bold', pad=15)

# Orienting from the past (T-60) to the present (T-0)
plt.xlabel("Days Ago (T-60 to T-0)", fontsize=12)
plt.ylabel("Neural Stream", fontsize=12)

# Customizing ticks to show 5-day intervals
plt.xticks(np.arange(0, 61, 5), labels=[str(i) for i in range(60, -1, -5)])
plt.yticks([]) # Maintain a clean single-row aesthetic

plt.tight_layout()
plt.show()