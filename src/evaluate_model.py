# src/evaluate_model.py
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from src.feature_engineering import create_training_set
from src.config import SEQ_LENGTH, MODELS_DIR

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

def calculate_max_drawdown(cumulative_returns):
    """
    Calculates the Maximum Drawdown (MDD) of a return series.
    MDD measures the largest peak-to-trough drop in a portfolio.
    """
    running_max = np.maximum.accumulate(cumulative_returns + 1)
    drawdown = ((cumulative_returns + 1) / running_max) - 1
    return np.min(drawdown)

def simulate_trading_strategy(actual_prices, predicted_prices):
    """
    Simulates a trading strategy and calculates Risk-Adjusted Returns.
    """
    # Calculate daily returns of the asset
    actual_returns = np.diff(actual_prices) / actual_prices[:-1]
    
    # AI Signal: +1 (Buy/Long) if predicting price goes up, -1 (Short) if down
    predicted_directions = np.sign(np.diff(predicted_prices))
    
    # Strategy Return
    strategy_returns = predicted_directions * actual_returns
    
    # Calculate cumulative compounding returns
    buy_and_hold_cumulative = np.cumprod(1 + actual_returns) - 1
    ai_strategy_cumulative = np.cumprod(1 + strategy_returns) - 1
    
    # Calculate Annualized Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
    # 252 is the standard number of trading days in a year
    bh_sharpe = (np.mean(actual_returns) / np.std(actual_returns)) * np.sqrt(252) if np.std(actual_returns) != 0 else 0
    ai_sharpe = (np.mean(strategy_returns) / np.std(strategy_returns)) * np.sqrt(252) if np.std(strategy_returns) != 0 else 0
    
    # Calculate Maximum Drawdown
    bh_mdd = calculate_max_drawdown(buy_and_hold_cumulative)
    ai_mdd = calculate_max_drawdown(ai_strategy_cumulative)
    
    return {
        "bh_cum": buy_and_hold_cumulative,
        "ai_cum": ai_strategy_cumulative,
        "bh_sharpe": bh_sharpe,
        "ai_sharpe": ai_sharpe,
        "bh_mdd": bh_mdd,
        "ai_mdd": ai_mdd
    }

def evaluate_model_performance(ticker):
    print(f"Initiating formal evaluation and backtesting for: {ticker}")
    
    # 1. Load Model and Scaler
    model_path = os.path.join(MODELS_DIR, f"{ticker}_lstm_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Error: Required files for {ticker} not found. Execute train_model.py first.")
        return
        
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    
    # 2. Data Preparation
    df = create_training_set(ticker)
    features = ['open', 'high', 'low', 'close', 'volume', 'sentiment_score']
    scaled_data = scaler.transform(df[features].values)
    target_col_index = features.index('close')
    
    X, y_actual_scaled = [], []
    for i in range(SEQ_LENGTH, len(scaled_data)):
        X.append(scaled_data[i-SEQ_LENGTH:i])
        y_actual_scaled.append(scaled_data[i, target_col_index])
        
    X = np.array(X)
    split_index = int(len(X) * 0.8)
    
    X_test = X[split_index:]
    y_test_scaled = np.array(y_actual_scaled)[split_index:]
    test_dates = df.index[split_index + SEQ_LENGTH:]
    
    # 3. Model Inference
    print("Executing model inference on test dataset...")
    predicted_scaled = model.predict(X_test, verbose=0)
    
    # 4. Inverse Transformation
    dummy_pred = np.zeros((len(predicted_scaled), len(features)))
    dummy_pred[:, target_col_index] = predicted_scaled[:, 0]
    predicted_prices = scaler.inverse_transform(dummy_pred)[:, target_col_index]
    
    dummy_actual = np.zeros((len(y_test_scaled), len(features)))
    dummy_actual[:, target_col_index] = y_test_scaled
    actual_prices = scaler.inverse_transform(dummy_actual)[:, target_col_index]
    
    # 5. Baseline Calculation (Naive Benchmark: P_t = P_t-1)
    naive_predictions = actual_prices[:-1]
    naive_actuals = actual_prices[1:]
    naive_mae = mean_absolute_error(naive_actuals, naive_predictions)
    
    # 6. Academic Metrics
    lstm_mae = mean_absolute_error(actual_prices, predicted_prices)
    lstm_mape = mean_absolute_percentage_error(actual_prices, predicted_prices) * 100
    
    actual_diff = np.diff(actual_prices)
    pred_diff = np.diff(predicted_prices)
    correct_directions = np.sum(np.sign(actual_diff) == np.sign(pred_diff))
    da = (correct_directions / len(actual_diff)) * 100
    
    # 7. Trading Simulation (ROI & Risk)
    sim_results = simulate_trading_strategy(actual_prices, predicted_prices)
    bh_cum = sim_results["bh_cum"]
    ai_cum = sim_results["ai_cum"]
    final_bh_roi = bh_cum[-1] * 100
    final_ai_roi = ai_cum[-1] * 100

    # 8. Console Reporting and Logging
    print("\n" + "=" * 60)
    print(f"OUT-OF-SAMPLE EVALUATION: {ticker}")
    print("=" * 60)
    print("PREDICTIVE ACCURACY METRICS:")
    print(f"Naive Baseline MAE:      ${naive_mae:.2f}")
    print(f"LSTM Model MAE:          ${lstm_mae:.2f}")
    print(f"Edge over Baseline:      {((naive_mae - lstm_mae) / naive_mae) * 100:.2f}% improvement")
    print(f"LSTM MAPE:               {lstm_mape:.2f}%")
    print(f"Directional Accuracy:    {da:.2f}%")
    print("-" * 60)
    print("FINANCIAL PERFORMANCE METRICS (AI vs Buy & Hold):")
    print(f"Cumulative ROI:          AI: {final_ai_roi:.2f}%  |  B&H: {final_bh_roi:.2f}%")
    print(f"Annualized Sharpe Ratio: AI: {sim_results['ai_sharpe']:.2f}    |  B&H: {sim_results['bh_sharpe']:.2f}")
    print(f"Maximum Drawdown (MDD):  AI: {sim_results['ai_mdd']*100:.2f}%  |  B&H: {sim_results['bh_mdd']*100:.2f}%")
    print("=" * 60)

    # Export metrics to CSV for report generation
    metrics_df = pd.DataFrame([{
        "Ticker": ticker,
        "LSTM_MAE": lstm_mae,
        "Directional_Accuracy": da,
        "AI_ROI": final_ai_roi,
        "AI_Sharpe": sim_results['ai_sharpe'],
        "AI_MDD": sim_results['ai_mdd']
    }])
    metrics_path = os.path.join("results", "evaluation_metrics_log.csv")
    metrics_df.to_csv(metrics_path, mode='a', header=not os.path.exists(metrics_path), index=False)
    
    # 9. Advanced Visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(test_dates, actual_prices, label="Actual Price", color='black', linewidth=1.5)
    ax1.plot(test_dates, predicted_prices, label="LSTM Prediction", color='blue', linestyle='--', linewidth=1.5)
    ax1.set_title(f"Model Forecasting: {ticker} (Price Tracking)", fontsize=14)
    ax1.set_ylabel("Price (USD)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    return_dates = test_dates[1:]
    ax2.plot(return_dates, bh_cum * 100, label="Buy & Hold Portfolio", color='gray')
    ax2.plot(return_dates, ai_cum * 100, label="AI Trading Portfolio", color='green')
    ax2.set_title("Simulated Algorithmic Trading Performance (Cumulative ROI %)", fontsize=12)
    ax2.set_ylabel("Return (%)")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join("results", f"{ticker}_advanced_evaluation.png")
    plt.savefig(plot_path)
    print(f"\nAdvanced visual analysis saved to: {plot_path}")

if __name__ == "__main__":
    evaluate_model_performance("NVDA")