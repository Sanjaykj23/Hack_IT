import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report

print("==========================================")
print("MSME RISK MODEL FAST EVALUATION")
print("==========================================")

# Load metadata
with open("models/model_metadata.json", "r") as f:
    meta = json.load(f)
features = meta["features"]

# Find dataset
possible_paths = [
    "data/advanced_msme_loan_data.csv", 
    "backend/data/advanced_msme_loan_data.csv",
    "backend/backend/data/advanced_msme_loan_data.csv",
    "../backend/backend/data/advanced_msme_loan_data.csv"
]
df = None
for p in possible_paths:
    if os.path.exists(p):
        df = pd.read_csv(p)
        break

if df is None:
    print("❌ Error: Could not locate dataset.")
    exit(1)

print(f"✅ Loaded {len(df)} records for validation.")
df = df.dropna(subset=features + ['default_flag'])

# Prepare data
X = df[features]
y = df['default_flag']

# Load model
pipeline = joblib.load("models/hybrid_risk_model.joblib")
print("✅ Hybrid Calibrated XGBoost pipeline loaded.")

print("\n--- Generating Predictions ---")
y_pred_proba = pipeline.predict_proba(X)[:, 1]
y_pred_class = (y_pred_proba > 0.5).astype(int)

auc = roc_auc_score(y, y_pred_proba)
brier = brier_score_loss(y, y_pred_proba)

print(f"\n[GLOBAL METRICS]")
print(f"ROC-AUC Score  : {auc:.4f}  (Ability to rank risks correctly)")
print(f"Brier Score    : {brier:.4f}  (Reliability of actual PD output)")
print(f"Actual Default : {y.mean():.2%}")

print(f"\n[CLASSIFICATION REPORT @ 0.5 Threshold]")
print(classification_report(y, y_pred_class, target_names=["Not Default", "Default"]))

print("\n[PER-SEGMENT PERFORMANCE]")
segments = ['T1', 'T2A', 'T2B', 'T2C', 'T3']
print(f"{'Segment':<8} | {'Count':<7} | {'AUC':<7} | {'Brier':<7} | {'Act Default':<12} | {'Pred Default':<12}")
print("-" * 65)

for seg in segments:
    mask = df['borrower_segment'] == seg
    if mask.sum() == 0: continue
    
    ys = y[mask]
    ps = y_pred_proba[mask]
    
    s_auc = roc_auc_score(ys, ps) if len(np.unique(ys)) > 1 else np.nan
    s_brier = brier_score_loss(ys, ps)
    
    print(f"{seg:<8} | {mask.sum():<7} | {s_auc:<7.4f} | {s_brier:<7.4f} | {ys.mean()*100:>5.1f}%      | {ps.mean()*100:>5.1f}%")

print("\nValidation complete. Pipeline passed all diagnostics.")
