from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import json
import asyncio
from typing import Dict, List, Optional
from uuid import uuid4

# Import routers
# from portfolio_manager import portfolio_router # Dashboard Removed
from mlops_monitor import mlops_router
from segment_router import segment_router, SEGMENT_POLICY
from feature_extractor import feature_extractor_router
from stability_scorecard import stability_router

app = FastAPI(title="AI-Powered MSME Lending Risk Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phase 4: Portfolio simulation layer
# app.include_router(portfolio_router) # Dashboard Removed
# Phase 5: MLOps monitoring
app.include_router(mlops_router)
# Phase 6 (NEW): 5-Tier segment engine
app.include_router(segment_router)
app.include_router(feature_extractor_router)
app.include_router(stability_router)

# Lazy loading: model will be loaded on first prediction request
risk_pipeline = None
explainability_map = {}


class MSMEApplicantData(BaseModel):
    # Static
    enterprise_category: str
    business_type: str
    industry_sector: str
    years_in_operation: int
    annual_turnover: float
    monthly_revenue: float
    gst_registered: int
    udyam_registered: int
    property_value: float = 0
    machinery_value: float = 0
    inventory_value: float = 0
    
    # Advanced Temporal Intelligence
    revenue_growth_rate: float
    revenue_momentum: float
    volatility_index: float
    payment_delay_trend_slope: float
    seasonality_coefficient: float
    
    # Financials & Proxies
    existing_loans: int
    existing_emi: float
    bank_account_years: int
    credit_history_length: int
    delayed_payments: int
    past_defaults: int
    gst_delay_months: int
    utility_consistency_pct: float
    upi_volatility_high: int
    
    # New v3.1 Stability & Segment Signals
    stability_score: float = 0.0
    income_confidence_encoded: int = 2
    years_at_location: float = 0.0
    employee_count: int = 0
    
    # Banking Intelligence (Tier 1 / Bank Statement derived)
    avg_monthly_balance: float = 0.0
    cheque_bounce_count: int = 0
    banking_vintage_months: int = 0
    
    # Proposed Loan Data

    loan_amount_requested: float
    loan_tenure_months: int

    # Macroeconomic
    macro_stress_factor: str = "Baseline"

    # Segment (optional — set by the gateway before form submission)
    borrower_segment: Optional[str] = "T1"  # T1, T2A, T2B, T2C, T3

class RiskPredictionResponse(BaseModel):
    layer_1_behavioral_pd: float
    layer_2_policy_status: str
    policy_rejection_flag: bool
    layer_3_final_pd: float
    risk_based_interest_rate_pct: float
    expected_loss_amt: float
    monthly_emi: float
    top_driver_explanations: List[str]
    credit_score_mapped: int
    borrower_segment: str
    segment_label: str
    income_confidence: str


def calculate_emi(principal, tenure_months, annual_rate=0.12):
    monthly_rate = annual_rate / 12
    return (principal * monthly_rate * ((1 + monthly_rate) ** tenure_months)) / (((1 + monthly_rate) ** tenure_months) - 1)

def map_pd_to_credit_score(pd_val: float, caps: List[int] = None) -> int:
    """
    Map PD to 300-900 score with banker-centric hard caps.
    """
    pd_val = max(0.01, min(0.99, pd_val))
    score = 300 + (600 / (1 + np.exp(5 * (pd_val - 0.5))))
    
    if caps:
        score = min(score, *caps)
        
    return int(round(score))


@app.post("/predict_risk", response_model=RiskPredictionResponse)
async def predict_risk(data: MSMEApplicantData):
    global risk_pipeline, explainability_map
    if risk_pipeline is None:
        try:
            print("Lazy loading Hybrid Risk Model Pipeline and Explainability mapping...")
            risk_pipeline = joblib.load("models/hybrid_risk_model.joblib")
            with open("models/explainability_map.json", "r", encoding="utf-8") as f:
                explainability_map = json.load(f)
            print("[OK] Models loaded successfully.")
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {e}")

    # Feature Engineering
    proposed_emi = calculate_emi(data.loan_amount_requested, data.loan_tenure_months)
    total_emi = proposed_emi + data.existing_emi
    emi_to_revenue_ratio = total_emi / data.monthly_revenue
    total_asset_value = data.property_value + data.machinery_value + data.inventory_value

    input_dict = data.dict()
    input_dict['proposed_emi'] = proposed_emi
    input_dict['total_emi'] = total_emi
    input_dict['emi_to_revenue_ratio'] = emi_to_revenue_ratio
    input_dict['total_asset_value'] = total_asset_value

    input_df = pd.DataFrame([input_dict])

    # ---------------------------------------------------------
    # LAYER 1: Behavioral Model (Calibrated PD Prediction)
    # ---------------------------------------------------------
    layer_1_pd = float(risk_pipeline.predict_proba(input_df)[0, 1])

    # ---------------------------------------------------------
    # LAYER 2: Policy Filter & Scoring Caps (Banker-Centric)
    # ---------------------------------------------------------
    policy_status = "Pass"
    policy_rejected = False
    score_caps = []

    # 1. Hard Rejections
    if emi_to_revenue_ratio > 0.60:
        policy_status = "Reject: Critical EMI-to-Income Ratio (>60%)"
        policy_rejected = True
    elif data.past_defaults > 0:
        policy_status = "Reject: History of Credit Defaults"
        policy_rejected = True
    elif data.industry_sector in ['Speculative Trading', 'Crypto']:
        policy_status = "Reject: Blacklisted Industry"
        policy_rejected = True
    elif data.borrower_segment == "T1" and data.cheque_bounce_count > 2:
        policy_status = "Reject: Multiple Cheque Bounces in Bank Statement"
        policy_rejected = True

    # 2. Strict Scoring Caps
    if emi_to_revenue_ratio > 0.50:
        score_caps.append(600)  # High burden limit
    elif emi_to_revenue_ratio > 0.35:
        score_caps.append(699)  # Medium burden threshold
        
    if data.years_in_operation < 2:
        score_caps.append(650)  # Business vintage cap
        
    if data.volatility_index > 0.40:
        score_caps.append(600)  # Excessive volatility cap

    # 3. Soft Review Flags
    if not policy_rejected:
        if data.existing_loans > 5:
            policy_status = "Review: High Multi-Lending Over-leverage"
        elif data.years_in_operation < 1:
            policy_status = "Review: Insufficient Business Age"
            # Soft statistical penalty
            logit_pd = np.log(layer_1_pd / (1 - layer_1_pd))
            layer_1_pd = float(np.clip(1 / (1 + np.exp(-(logit_pd + 0.40))), 0.01, 0.99))



    # ---------------------------------------------------------
    # LAYER 3: Portfolio Adjustment
    # ---------------------------------------------------------
    # Mocking a portfolio constraint: e.g. "We have too many Micro loans, increase risk premium slightly to curb growth"
    portfolio_multiplier = 1.0
    if data.enterprise_category == 'Micro':
        portfolio_multiplier = 1.05
    
    final_pd = min(0.99, layer_1_pd * portfolio_multiplier)

    # ---------------------------------------------------------
    # Risk-Based Pricing Engine (Segment-Aware)
    # ---------------------------------------------------------
    segment = data.borrower_segment or "T1"
    seg_policy = SEGMENT_POLICY.get(segment, SEGMENT_POLICY["T1"])
    segment_premium = seg_policy["interest_premium"]
    segment_label = seg_policy["label"]

    # Apply PD floor for lower-confidence segments
    pd_floor = seg_policy["pd_floor"]
    final_pd = max(pd_floor, min(0.99, layer_1_pd * portfolio_multiplier))

    # Apply max tenure policy
    if data.loan_tenure_months > seg_policy["max_tenure"]:
        policy_status = f"Review: Tenure exceeds {seg_policy['max_tenure']}m limit for {segment_label} segment"
        policy_rejected = True

    base_rate = 10.0
    risk_premium = final_pd * 100 * 0.3
    final_interest_rate = base_rate + risk_premium + segment_premium

    # Income confidence level from segment
    confidence_map = {"T1": "high", "T2A": "high", "T2B": "medium", "T2C": "medium", "T3": "low"}
    income_confidence = confidence_map.get(segment, "high")

    # ---------------------------------------------------------
    # Expected Loss (EL) Calculation
    # ---------------------------------------------------------
    exposure_at_default = data.loan_amount_requested
    collateral_ratio = total_asset_value / exposure_at_default if exposure_at_default > 0 else 0

    # ✅ IMPROVED: industry-aware LGD with macro multiplier
    from portfolio_manager import get_lgd
    lgd = get_lgd(collateral_ratio, data.industry_sector)

    expected_loss = final_pd * lgd * exposure_at_default

    # ---------------------------------------------------------
    # Intelligent Explainability Engine (Precomputed Lookup)
    # ---------------------------------------------------------
    top_explanations = []
    # Find top 4 impactful features based on applicant data logic vs the map
    # A simple runtime heuristic: check if severe features are present
    if emi_to_revenue_ratio > 0.5:
        top_explanations.append(f"⚠️ +15% risk due to High EMI Burden ({emi_to_revenue_ratio*100:.0f}%)")
    if data.gst_delay_months > 2:
        val = explainability_map.get('gst_delay_months', {}).get('magnitude', 8.0)
        top_explanations.append(f"⚠️ +{val}% risk due to GST Filing Delays")
    if data.utility_consistency_pct < 0.7:
        val = explainability_map.get('utility_consistency_pct', {}).get('magnitude', 6.0)
        top_explanations.append(f"⚠️ +{val}% risk due to Erratic Utility Payments")
    if data.revenue_growth_rate > 0.1:
        val = explainability_map.get('revenue_growth_rate', {}).get('magnitude', 5.0)
        top_explanations.append(f"✅ -{val}% risk due to Strong Revenue Growth")

    # Map back to old score format for visual familiarty
    credit_score = map_pd_to_credit_score(final_pd, caps=score_caps)


    return RiskPredictionResponse(
        layer_1_behavioral_pd=round(layer_1_pd, 6),
        layer_2_policy_status=policy_status,
        policy_rejection_flag=policy_rejected,
        layer_3_final_pd=round(final_pd, 6),
        risk_based_interest_rate_pct=round(final_interest_rate, 2),
        expected_loss_amt=round(expected_loss, 2),
        monthly_emi=round(proposed_emi, 2),
        top_driver_explanations=top_explanations,
        credit_score_mapped=credit_score,
        borrower_segment=segment,
        segment_label=segment_label,
        income_confidence=income_confidence
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
