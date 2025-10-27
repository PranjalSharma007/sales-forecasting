# Hybrid SARIMAX + LightGBM Optimization Script

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# --- 1. SMAPE Calculation Function ---
def calculate_smape(y_true, y_pred):
    """
    Calculates SMAPE (Symmetric Mean Absolute Percentage Error).
    Handles division-by-zero cases gracefully.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = np.where(denominator == 0, 0, 2 * numerator / denominator)
    
    return np.mean(smape) * 100


# --- 2. Define the Data using your actual model outputs ---
# Replace these variable names with your actual notebook variables
# Example: from your previous Phase 3
# y_test      = actual sales in test data
# sarimax_pred = SARIMAX forecast for same period
# lgbm_pred    = LightGBM forecast for same period

y_true = np.array(y_test)
y_sarimax_preds = np.array(sarimax_pred)
y_lightgbm_preds = np.array(lgbm_pred)

# --- 2.1 Ensure equal lengths ---
min_len = min(len(y_true), len(y_sarimax_preds), len(y_lightgbm_preds))
y_true = y_true[:min_len]
y_sarimax_preds = y_sarimax_preds[:min_len]
y_lightgbm_preds = y_lightgbm_preds[:min_len]


# --- 3. Define the Objective Function for Optimization ---
def objective_function(w, y_true, y_sarimax, y_lgbm):
    """
    Returns SMAPE for the given SARIMAX weight 'w'
    (LightGBM weight will be (1 - w)).
    """
    y_hybrid = (w * y_sarimax) + ((1 - w) * y_lgbm)
    return calculate_smape(y_true, y_hybrid)


# --- 4. Run the Optimization (SARIMAX weight > LightGBM) ---
optimization_result = minimize_scalar(
    objective_function,
    bounds=(0.5, 1.0),  # SARIMAX weight between 0.5 and 1.0
    args=(y_true, y_sarimax_preds, y_lightgbm_preds),
    method='bounded'
)


# --- 5. Display the Results ---
if optimization_result.success:
    optimal_w = optimization_result.x
    min_smape = optimization_result.fun
    
    print("\n--- Optimization Successful ---")
    print(f"✅ Optimal SARIMAX Weight:   {optimal_w:.4f}")
    print(f"✅ Optimal LightGBM Weight:  {1 - optimal_w:.4f}")
    print(f"✅ Lowest Hybrid SMAPE:      {min_smape:.2f}%")
    
    # Compare individual model SMAPEs
    sarimax_smape = calculate_smape(y_true, y_sarimax_preds)
    lgbm_smape = calculate_smape(y_true, y_lightgbm_preds)
    
    print("\n--- Model Comparison ---")
    print(f"SARIMAX SMAPE:   {sarimax_smape:.2f}%")
    print(f"LightGBM SMAPE:  {lgbm_smape:.2f}%")
else:
    print("❌ Optimization failed.")
    print(optimization_result.message)


# --- 6. Compute and Plot Final Hybrid Forecast ---
y_hybrid_final = (optimal_w * y_sarimax_preds) + ((1 - optimal_w) * y_lightgbm_preds)

plt.figure(figsize=(12, 6))
plt.plot(y_true, label='True Sales', color='black', linewidth=2)
plt.plot(y_sarimax_preds, label='SARIMAX', linestyle='--')
plt.plot(y_lightgbm_preds, label='LightGBM', linestyle='--')
plt.plot(y_hybrid_final, label='Hybrid Forecast', color='green', linewidth=3)
plt.title('Hybrid SARIMAX + LightGBM Forecast Comparison')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# --- 7. Optional: Print final SMAPE of hybrid model again ---
final_hybrid_smape = calculate_smape(y_true, y_hybrid_final)
print(f"\n✅ Final Hybrid Model SMAPE: {final_hybrid_smape:.2f}%")

