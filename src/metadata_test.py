import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# 1. Define Paths relative to root
input_path = os.path.join('data', 'v4_X_price.npy')
output_path = os.path.join('data', 'scaler.pkl')

if not os.path.exists(input_path):
    print(f"ERROR: Cannot find {input_path}. Check your folder structure.")
else:
    # 2. Load and Reshape
    X_price = np.load(input_path)
    print(f"Loaded training data with shape: {X_price.shape}")
    
    # Reshape (Samples * 60 days, 6 features)
    X_flat = X_price.reshape(-1, 6)
    
    # 3. Fit and Export
    scaler = StandardScaler()
    scaler.fit(X_flat)
    
    joblib.dump(scaler, output_path)
    print(f"SUCCESS: Scaler saved to {output_path}")
    print(f"Feature Means: {scaler.mean_}")