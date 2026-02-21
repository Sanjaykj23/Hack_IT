import pandas as pd
import numpy as np
import joblib
import json
import shap
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import os

print("Initializing Phase 2 v2: Training 5-Segment Hybrid Risk Engine...")

data_path = "data/advanced_msme_loan_data.csv" if os.path.exists("data/advanced_msme_loan_data.csv") else "backend/data/advanced_msme_loan_data.csv"
df = pd.read_csv(data_path)
print(f"Loaded dataset: {df.shape[0]} rows from {data_path}")
print(f"\nSegment distribution in training data:")
print(df['borrower_segment'].value_counts(normalize=True).round(3))

# ─── FEATURE COLUMNS ──────────────────────────────────────────────────────────
# All original features +
# 3 new features: stability_score, income_confidence_encoded, stability proxies
feature_cols = [
    # Static & Structural
    'enterprise_category', 'business_type', 'industry_sector',
    'years_in_operation', 'annual_turnover', 'monthly_revenue',
    'gst_registered', 'udyam_registered',
    'property_value', 'machinery_value', 'inventory_value', 'total_asset_value',

    # Advanced Temporal Intelligence
    'revenue_growth_rate', 'revenue_momentum', 'volatility_index',
    'payment_delay_trend_slope', 'seasonality_coefficient',

    # Financials & Behavioral Proxies
    'existing_loans', 'existing_emi', 'bank_account_years', 'credit_history_length',
    'delayed_payments', 'past_defaults', 'gst_delay_months',
    'utility_consistency_pct', 'upi_volatility_high',

    # ── NEW: Stability & Segment Signals ──────────────────────────────────────
    'stability_score',              # 0–100; non-zero primarily for T3
    'income_confidence_encoded',    # 2=high, 1=medium, 0=low
    'years_at_location',            # Location stability proxy
    'employee_count',               # Business scale proxy

    # Proposed Loan
    'loan_amount_requested', 'loan_tenure_months', 'proposed_emi', 'total_emi', 'emi_to_revenue_ratio',

    # Macroeconomic
    'macro_stress_factor',

    # Segment (categorical — model learns segment-specific patterns)
    'borrower_segment',
]

X = df[feature_cols]
y = df['default_flag']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ─── PREPROCESSOR ─────────────────────────────────────────────────────────────
numerical_cols = [
    'years_in_operation', 'annual_turnover', 'monthly_revenue',
    'property_value', 'machinery_value', 'inventory_value', 'total_asset_value',
    'revenue_growth_rate', 'revenue_momentum', 'volatility_index',
    'payment_delay_trend_slope', 'seasonality_coefficient',
    'existing_loans', 'existing_emi', 'bank_account_years', 'credit_history_length',
    'delayed_payments', 'past_defaults', 'gst_delay_months',
    'utility_consistency_pct',
    'stability_score', 'income_confidence_encoded',
    'years_at_location', 'employee_count',
    'loan_amount_requested', 'loan_tenure_months', 'proposed_emi', 'total_emi', 'emi_to_revenue_ratio',
]

categorical_cols = ['enterprise_category', 'business_type', 'industry_sector', 'macro_stress_factor', 'borrower_segment']
binary_cols      = ['gst_registered', 'udyam_registered', 'upi_volatility_high']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('bin', 'passthrough', binary_cols)
    ]
)

# ─── MODEL ────────────────────────────────────────────────────────────────────
hgb_model = HistGradientBoostingClassifier(
    max_iter=150,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

calibrated_hgb = CalibratedClassifierCV(estimator=hgb_model, method='sigmoid', cv=3)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', calibrated_hgb)
])

print("\nTraining Calibrated Hybrid Risk Model (5-segment, lightweight v3.1)...")
print("Est. 2–4 minutes (3-fold CV × 150 estimators)...\n")
pipeline.fit(X_train, y_train)

# ─── EVALUATION ───────────────────────────────────────────────────────────────
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
auc   = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)

print(f"\n{'='*50}")
print("Evaluation Metrics (Test Set):")
print(f"  ROC-AUC Score  : {auc:.4f}   (higher = better discrimination)")
print(f"  Brier Score    : {brier:.4f}  (lower = better calibration)")
print(f"{'='*50}")

# Per-segment evaluation
test_idx = X_test.index
print("\nPer-Segment AUC (where computable):")
for seg in ['T1', 'T2A', 'T2B', 'T2C', 'T3']:
    seg_mask = df.loc[test_idx, 'borrower_segment'] == seg
    if seg_mask.sum() < 10: continue
    try:
        seg_auc = roc_auc_score(y_test[seg_mask], y_pred_proba[seg_mask])
        print(f"  {seg}: AUC={seg_auc:.4f}  (n={int(seg_mask.sum())})")
    except Exception:
        pass

# ─── EXPLAINABILITY ───────────────────────────────────────────────────────────
print("\nPhase 3: Pre-computing SHAP Explainability mapping (this takes a few minutes)...")

preprocessor.fit(X)
ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(ohe_features) + binary_cols

shap_model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42).fit(
    preprocessor.transform(X), y
)
explainer    = shap.TreeExplainer(shap_model)
X_transformed = preprocessor.transform(X)

# Use a sample for SHAP to save compute (1000 random rows)
sample_idx = np.random.choice(len(X_transformed), min(3000, len(X_transformed)), replace=False)
shap_values = explainer.shap_values(X_transformed[sample_idx])

# SHAP returned for RandomForestClassifier binary classification is often a list of arrays [shap_values_class0, shap_values_class1]
# or an array of shape (n_samples, n_features, 2). 
shap_values_target = shap_values[1] if isinstance(shap_values, list) else (shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values)

feature_importance = {}
for i, feature in enumerate(all_feature_names):
    mean_abs_shap = float(np.mean(np.abs(shap_values_target[:, i])))
    if mean_abs_shap > 0:
        corr = np.corrcoef(X_transformed[sample_idx, i].flatten(), shap_values_target[:, i].flatten())[0, 1]
        direction = "increases" if corr > 0 else "decreases"
        feature_importance[feature] = {
            "impact_magnitude": float(np.round(mean_abs_shap * 10, 1)),
            "direction": direction,
            "raw_importance": mean_abs_shap
        }

sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1]['raw_importance'], reverse=True)

print("\nTop 15 features by SHAP importance:")
for feat, stats in sorted_importance[:15]:
    sign = '+' if stats['direction'] == 'increases' else '-'
    print(f"  {sign} {feat:40s}  {stats['raw_importance']:.4f}")

explainability_mapping = {}
for feature, stats in sorted_importance:
    direction_sign = "+" if stats['direction'] == "increases" else "-"
    explainability_mapping[feature] = {
        "text_template": f"{direction_sign}{{val}}% risk due to {{feature_name}}",
        "magnitude": stats["impact_magnitude"],
        "direction": direction_sign
    }

# ─── SAVE ARTIFACTS ───────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/hybrid_risk_model.joblib")
with open("models/explainability_map.json", "w") as f:
    json.dump(explainability_mapping, f, indent=4)

# Save model metadata
metadata = {
    "model_version": "v3.1-5segment-fast",
    "features": feature_cols,
    "num_features": len(numerical_cols),
    "cat_features": categorical_cols,
    "segments_supported": ["T1", "T2A", "T2B", "T2C", "T3"],
    "training_samples": len(X_train),
    "roc_auc": round(auc, 4),
    "brier_score": round(brier, 4),
    "new_features_v3": ["stability_score", "income_confidence_encoded", "years_at_location", "employee_count", "borrower_segment"]
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Saved: models/hybrid_risk_model.joblib")
print("✅ Saved: models/explainability_map.json")
print("✅ Saved: models/model_metadata.json")
print(f"\nModel v3.0-5segment ready. AUC={auc:.4f}, Brier={brier:.4f}")
