import numpy as np
from fastapi import APIRouter
import json
import os

segment_router = APIRouter(prefix="/segment", tags=["Segment Engine"])

SECTOR_BENCHMARKS_PATH = os.path.join(os.path.dirname(__file__), "data", "sector_benchmarks.json")

def _load_benchmarks():
    for path in [SECTOR_BENCHMARKS_PATH, "data/sector_benchmarks.json", "backend/data/sector_benchmarks.json"]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return {}

BENCHMARKS = _load_benchmarks()

# Segment policy caps — applied in the policy layer of predict_risk
SEGMENT_POLICY = {
    "T1":  {"label": "Full Financial",       "max_ltv": 1.00, "interest_premium": 0.0, "max_tenure": 60, "pd_floor": 0.01},
    "T2A": {"label": "GST + UPI Validated",  "max_ltv": 1.00, "interest_premium": 0.0, "max_tenure": 60, "pd_floor": 0.01},
    "T2B": {"label": "GST Declared Only",    "max_ltv": 0.80, "interest_premium": 1.5, "max_tenure": 48, "pd_floor": 0.05},
    "T2C": {"label": "Digital Informal",     "max_ltv": 0.70, "interest_premium": 3.0, "max_tenure": 36, "pd_floor": 0.08},
    "T3":  {"label": "Stability-Only",       "max_ltv": 0.50, "interest_premium": 5.0, "max_tenure": 24, "pd_floor": 0.18},
}

def classify_segment(has_gst: bool, has_upi: bool, has_bank_statement: bool) -> str:
    """
    Routes borrower to the correct tier based on 3 boolean signals.
    Returns tier code: T1, T2A, T2B, T2C, or T3.
    """
    if has_bank_statement and has_gst:
        return "T1"
    if has_gst and has_upi:
        return "T2A"
    if has_gst and not has_upi:
        return "T2B"
    if has_upi and not has_gst:
        return "T2C"
    return "T3"

@segment_router.post("/classify")
async def classify_borrower_segment(
    has_gst: bool = False,
    has_upi: bool = False,
    has_bank_statement: bool = False
):
    """
    Gateway classifier. Takes 3 boolean flags and returns the borrower tier.
    Called immediately when user answers the 3 gateway questions in the UI.
    """
    tier = classify_segment(has_gst, has_upi, has_bank_statement)
    policy = SEGMENT_POLICY[tier]
    return {
        "tier": tier,
        "label": policy["label"],
        "max_ltv_pct": policy["max_ltv"] * 100,
        "interest_premium_pct": policy["interest_premium"],
        "max_tenure_months": policy["max_tenure"],
        "pd_floor": policy["pd_floor"],
        "data_collection_required": _get_required_inputs(tier)
    }

def _get_required_inputs(tier: str) -> dict:
    """Returns the list of form sections the UI should render for this tier."""
    base = ["business_identity", "asset_values", "loan_request"]
    inputs = {
        "T1":  base + ["gst_or_bank_statement_upload"],
        "T2A": base + ["gst_turnover_12m", "upi_history_6m", "utility_bills_6m"],
        "T2B": base + ["gst_turnover_12m", "utility_bills_6m", "stability_partial"],
        "T2C": base + ["upi_history_6m", "utility_bills_6m", "stability_partial"],
        "T3":  base + ["utility_bills_6m", "stability_full"],
    }
    return inputs.get(tier, base)

@segment_router.get("/sector-benchmarks/{industry_sector}")
async def get_sector_benchmarks(industry_sector: str):
    """
    Returns sector-specific defaults for a given industry.
    These are auto-populated as read-only chips in the form
    when the user selects their industry sector.
    """
    benchmark = BENCHMARKS.get(industry_sector)
    if not benchmark:
        # Fallback to General Manufacturing averages
        benchmark = BENCHMARKS.get("General Manufacturing", {
            "revenue_growth_rate": 0.06,
            "seasonality_coefficient": 0.20,
            "digital_payment_ratio": 0.30,
            "under_reporting_adjustment": 1.10,
            "sector_daily_revenue_floor": 3000,
            "avg_volatility_index": 0.20,
            "description": "General sector average"
        })

    return {
        "industry_sector": industry_sector,
        "benchmark": benchmark,
        "source": "RBI MSME Sector Intelligence / Dun & Bradstreet India (Simulated)",
        "note": "These values are auto-filled. The underwriter may override them based on applicant evidence."
    }

@segment_router.get("/sector-benchmarks")
async def list_all_benchmarks():
    """Returns benchmarks for all sectors."""
    return {
        "sectors": list(BENCHMARKS.keys()),
        "benchmarks": BENCHMARKS
    }
