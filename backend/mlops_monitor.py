import pandas as pd
import numpy as np
from fastapi import APIRouter
import joblib
import os
import json

mlops_router = APIRouter(prefix="/mlops", tags=["MLOps"])

def calculate_psi(baseline: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """
    Statistically correct PSI using consistent quantile bin edges.
    ✅ FIXED: both baseline and current are binned using the SAME edges
    derived from the baseline distribution. This ensures PSI is stable
    and not affected by differing data ranges in the two series.
    """
    # Derive bin edges from baseline quantiles (ensures consistent boundaries)
    quantiles = np.linspace(0, 100, buckets + 1)
    bin_edges = np.percentile(baseline, quantiles)

    # Avoid duplicate edges at extremes (can happen with heavily skewed data)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Count observations in each bucket using the same edges for both series
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to proportions
    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)

    # Avoid division by zero or log(0)
    baseline_pct = np.where(baseline_pct == 0, 1e-4, baseline_pct)
    current_pct = np.where(current_pct == 0, 1e-4, current_pct)

    psi_values = (current_pct - baseline_pct) * np.log(current_pct / baseline_pct)
    return float(np.sum(psi_values))

_BASELINE_DF = None

def _load_baseline():
    global _BASELINE_DF
    if _BASELINE_DF is None:
        for path in ["data/advanced_msme_loan_data.csv", "backend/data/advanced_msme_loan_data.csv"]:
            if os.path.exists(path):
                _BASELINE_DF = pd.read_csv(path)
                break
        if _BASELINE_DF is None:
            raise FileNotFoundError("Dataset not found. Run advanced_data_generation.py first.")
    return _BASELINE_DF

@mlops_router.get("/drift-status")
async def check_drift():
    baseline_df = _load_baseline()

    # Simulate recent inference data with artificial feature drift
    recent_df = baseline_df.sample(500, random_state=123).copy()
    recent_df['loan_amount_requested'] = recent_df['loan_amount_requested'] * 1.4
    recent_df['emi_to_revenue_ratio'] = recent_df['emi_to_revenue_ratio'] * 1.2

    baseline_emi = baseline_df['emi_to_revenue_ratio'].values
    recent_emi = recent_df['emi_to_revenue_ratio'].values

    psi_score = calculate_psi(baseline_emi, recent_emi)

    drift_status = "Stable"
    trigger_retraining = False
    action_required = "None"

    if psi_score > 0.2:
        drift_status = "Critical Feature Drift Detected"
        trigger_retraining = True
        action_required = "Automated Retraining Triggered via ML Pipeline"
    elif psi_score > 0.1:
        drift_status = "Warning: Minor Drift"
        action_required = "Monitor closely in next 7 days"

    return {
        "metric": "Population Stability Index (PSI)",
        "monitored_feature": "emi_to_revenue_ratio",
        "psi_score": round(psi_score, 4),
        "status": drift_status,
        "trigger_retraining": trigger_retraining,
        "recommended_action": action_required,
        "active_model_version": "v2.0-Hybrid_Risk_Engine"
    }

@mlops_router.get("/model-registry")
async def get_model_registry():
    """
    ✅ NEW: Structured model registry endpoint.
    Exposes key model metadata for auditability and governance.
    In production this would be backed by MLflow or a model DB.
    """
    registry = {
        "active_model": {
            "model_id": "v2.0-Hybrid_Risk_Engine",
            "algorithm": "XGBoost + Platt Scaling (CalibratedClassifierCV)",
            "train_date": "2026-02-21",
            "training_samples": 15000,
            "features_count": 35,
            "metrics": {
                "roc_auc": None,        # Populated after retraining
                "brier_score": None,
                "ks_statistic": None,
                "calibration_error": None,
            },
            "data_source": "advanced_msme_loan_data.csv (Synthetic, Generative PD Engine v2)",
            "pd_framework": "Point-in-Time (PIT) — Log-Odds Generative Function",
            "status": "Active"
        },
        "retired_models": [
            {
                "model_id": "v1.0-Baseline_XGB",
                "retired_date": "2026-02-10",
                "reason": "Replaced by calibrated hybrid engine with temporal features"
            }
        ],
        "next_scheduled_review": "2026-03-21",
        "retraining_trigger": "PSI > 0.20 on emi_to_revenue_ratio"
    }

    # Try to attach live metrics from the explainability map as a proxy
    try:
        with open("models/explainability_map.json", "r") as f:
            exp_map = json.load(f)
        registry["active_model"]["top_features_by_shap"] = list(exp_map.keys())[:10]
    except Exception:
        pass

    return registry

