import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-darkgrid')

def analyze_confidence():
    file_path = '../data/final_fold_classifier_predictions.csv'
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    df = pd.read_csv(file_path)
    probs = df['Predicted_Probability']

    print("\n=== AI CONFIDENCE SPREAD ===")
    print(f"Lowest Confidence Prediction:  {probs.min():.4f} ({probs.min()*100:.2f}%)")
    print(f"Highest Confidence Prediction: {probs.max():.4f} ({probs.max()*100:.2f}%)")
    print(f"Average Prediction:            {probs.mean():.4f} ({probs.mean()*100:.2f}%)")
    
    # Calculate how many predictions are "High Conviction" (e.g., > 55% or < 45%)
    high_conviction_up = len(df[probs >= 0.55])
    high_conviction_down = len(df[probs <= 0.45])
    print(f"\nTotal High Conviction 'UP' Signals (>55%): {high_conviction_up}")
    print(f"Total High Conviction 'DOWN' Signals (<45%): {high_conviction_down}")

    # Plot the spread
    plt.figure(figsize=(10, 6))
    sns.histplot(probs, bins=50, kde=True, color='#2ca02c')
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='50/50 Guess')
    
    plt.title('Distribution of AI Predicted Probabilities (Fold 5)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Probability of Going UP')
    plt.ylabel('Number of Predictions')
    plt.legend()
    
    os.makedirs('../images', exist_ok=True)
    plt.savefig('../images/confidence_spread.png', dpi=300, bbox_inches='tight')
    print("\nSaved histogram to images/confidence_spread.png")
    
    plt.show()

if __name__ == '__main__':
    analyze_confidence()