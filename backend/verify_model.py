import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, brier_score_loss, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
import os

print("Initialising Model Verification Suite...")

# 1. Load Data
possible_paths = [
    "data/advanced_msme_loan_data.csv", 
    "backend/data/advanced_msme_loan_data.csv",
    "backend/backend/data/advanced_msme_loan_data.csv",
    "../backend/backend/data/advanced_msme_loan_data.csv"
]
data_path = None
for p in possible_paths:
    if os.path.exists(p):
        data_path = p
        break

if not data_path:
    print("Error: Could not find training data CSV.")
    exit(1)

df = pd.read_csv(data_path)
print(f"Loaded {df.shape[0]} synthetic loan records.")

# 2. Load Model & Metadata
model_path = "models/hybrid_risk_model.joblib"
meta_path = "models/model_metadata.json"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}. Please run train_hybrid_model.py first.")
    exit(1)

pipeline = joblib.load(model_path)
with open(meta_path, 'r') as f:
    metadata = json.load(f)

feature_cols = metadata['features']
df = df.dropna(subset=feature_cols + ['default_flag']) # ensure no NaNs

X = df[feature_cols]
y = df['default_flag']

# Predict Probabilities
print("Generating predictions on full dataset...")
y_pred_proba = pipeline.predict_proba(X)[:, 1]
# Binarize with typical 0.5 threshold (or adjusted if needed)
y_pred_class = (y_pred_proba > 0.5).astype(int)

# ─── OVERALL METRICS ──────────────────────────────────────────────────────────
print("\n" + "="*50)
print("1. OVERALL STATISTICAL PERFORMANCE")
print("="*50)
overall_auc = auc(*roc_curve(y, y_pred_proba)[:2])
overall_brier = brier_score_loss(y, y_pred_proba)

print(f"ROC-AUC Score  : {overall_auc:.4f} (Ability to rank-order risk)")
print(f"Brier Score    : {overall_brier:.4f} (Calibration accuracy)")
print(f"Default Rate   : {y.mean():.2%}")
print("\nClassification Report (Threshold = 0.5):")
print(classification_report(y, y_pred_class, target_names=["Not Default", "Default"]))

# ─── PER-SEGMENT METRICS ──────────────────────────────────────────────────────
print("\n" + "="*50)
print("2. FAIRNESS & PER-SEGMENT CALIBRATION")
print("="*50)

segments = ['T1', 'T2A', 'T2B', 'T2C', 'T3']
seg_results = []

for seg in segments:
    mask = df['borrower_segment'] == seg
    if mask.sum() < 50: 
        continue
    
    y_s = y[mask]
    p_s = y_pred_proba[mask]
    
    seg_auc = auc(*roc_curve(y_s, p_s)[:2])
    seg_brier = brier_score_loss(y_s, p_s)
    seg_default_rate = y_s.mean()
    predicted_default_rate = p_s.mean()
    
    seg_results.append({
        'Segment': seg,
        'N': mask.sum(),
        'AUC': round(seg_auc, 4),
        'Brier': round(seg_brier, 4),
        'Actual Default %': f"{seg_default_rate*100:.1f}%",
        'Predicted Default %': f"{predicted_default_rate*100:.1f}%"
    })

seg_df = pd.DataFrame(seg_results)
print(seg_df.to_string(index=False))
print("\n* Observation: Look at how close Predicted Default % is to Actual Default %. That is calibration.")


# ─── VISUALIZATIONS ───────────────────────────────────────────────────────────
print("\n" + "="*50)
print("3. GENERATING VISUAL DIAGNOSTICS")
print("="*50)
os.makedirs("evaluation", exist_ok=True)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y, y_pred_proba)
axes[0].plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {overall_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('Receiver Operating Characteristic (ROC)')
axes[0].legend(loc="lower right")

# 2. Calibration Curve (Reliability Diagram)
prob_true, prob_pred = calibration_curve(y, y_pred_proba, n_bins=10)
axes[1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Hybrid XGBoost')
axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
axes[1].set_xlabel('Mean Predicted Probability')
axes[1].set_ylabel('Fraction of Positives (Actual)')
axes[1].set_title(f'Calibration Curve (Brier: {overall_brier:.3f})')
axes[1].legend()

# 3. Score Distribution by Target
sns.histplot(data=pd.DataFrame({'PD': y_pred_proba, 'Default': y}), x='PD', hue='Default', bins=40, kde=True, ax=axes[2])
axes[2].set_title('Probability of Default (PD) Distribution')
axes[2].set_xlabel('Predicted PD')

plt.tight_layout()
plt.savefig('evaluation/model_diagnostics.png', dpi=300)
print("✅ Saved diagnostics plot to: evaluation/model_diagnostics.png")

# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Good', 'Predicted Bad'],
            yticklabels=['Actual Good', 'Actual Bad'])
plt.title('Confusion Matrix (Threshold = 0.5)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('evaluation/confusion_matrix.png', dpi=300)
print("✅ Saved confusion matrix to: evaluation/confusion_matrix.png")


print("\nDone. You can review the metrics above or open the images in the `evaluation` folder.")
