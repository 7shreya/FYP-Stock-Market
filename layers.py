import tensorflow as tf
from src.predict_live import TemporalAttention, directional_accuracy

model = tf.keras.models.load_model(
    r"C:\Users\mistr\OneDrive - Aston University\data\models\v4_champion_model.keras", 
    custom_objects={'TemporalAttention': TemporalAttention, 'directional_accuracy': directional_accuracy}
)

print("--- Listing All Model Layers ---")
for i, layer in enumerate(model.layers):
    print(f"Index {i}: Name = '{layer.name}', Type = {type(layer)}")