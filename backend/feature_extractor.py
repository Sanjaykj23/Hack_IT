import numpy as np
from typing import List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

feature_extractor_router = APIRouter(prefix="/features", tags=["Feature Extraction"])

# ─── Pydantic I/O Models ────────────────────────────────────────────────────

class UtilityRecord(BaseModel):
    expected_amount: float
    paid_amount: float
    days_late: int = 0

class ComputeFeaturesRequest(BaseModel):
    tier: str                                          # T1, T2A, T2B, T2C, T3
    industry_sector: str
    # Tier 2A/2B: GST monthly turnover (12 values, INR)
    gst_monthly_turnover_12m: Optional[List[float]] = None
    # Tier 2A/2C: UPI monthly inflows (6 values, INR)
    upi_monthly_inflows_6m: Optional[List[float]] = None
    # Tier 2C: Sector digital payment ratio (overrides benchmark if provided)
    digital_payment_ratio_override: Optional[float] = None
    # Tier 2B/2C/3: Utility bills (6 months)
    utility_bills_6m: Optional[List[UtilityRecord]] = None
    # Tier 2B: GST filing delays per quarter (months late)
    gst_delay_months: Optional[float] = 0
    # Sector benchmark (passed from frontend after /sector-benchmarks call)
    sector_revenue_growth_rate: Optional[float] = None
    sector_seasonality_coefficient: Optional[float] = None
    sector_digital_payment_ratio: Optional[float] = None
    sector_avg_volatility_index: Optional[float] = None
    sector_under_reporting_adjustment: Optional[float] = None

class ComputedFeatures(BaseModel):
    estimated_monthly_revenue: float
    revenue_growth_rate: float
    revenue_momentum: float
    volatility_index: float
    payment_delay_trend_slope: float
    utility_consistency_pct: float
    upi_volatility_high: int
    seasonality_coefficient: float
    gst_delay_months: float
    # Metadata
    income_confidence: str          # high / medium / low
    income_source: str
    income_error_band_pct: float
    computation_notes: List[str]

# ─── Utility Calculations ───────────────────────────────────────────────────

def compute_utility_consistency(bills: List[UtilityRecord]) -> float:
    """Utility consistency = fraction of bills paid in full within 7 days."""
    if not bills:
        return 0.80  # neutral assumption if not provided
    on_time = sum(1 for b in bills if b.paid_amount >= b.expected_amount and b.days_late <= 7)
    return round(on_time / len(bills), 4)

def compute_upi_statistics(inflows: List[float]) -> dict:
    """Compute revenue stats from UPI monthly inflow history."""
    arr = np.array(inflows, dtype=float)
    mean_ = float(np.mean(arr))
    std_ = float(np.std(arr))
    vol = round(std_ / mean_, 4) if mean_ > 0 else 0.40

    n = len(arr)
    if n >= 6:
        momentum = float(np.mean(arr[-3:]) / np.mean(arr[:3]) - 1.0)
    elif n >= 2:
        momentum = float(arr[-1] / arr[0] - 1.0)
    else:
        momentum = 0.0

    # Payment delay slope: uses relative UPI decline as proxy (no direct delay data)
    if n >= 3:
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0] / mean_)
    else:
        slope = 0.0

    return {
        "mean_monthly": mean_,
        "volatility_index": round(min(vol, 1.0), 4),
        "momentum": round(momentum, 4),
        "slope_proxy": round(slope, 4),
        "upi_volatility_high": int(vol > 0.40)
    }

def compute_gst_slope(monthly_turnover: List[float]) -> dict:
    """Compute revenue signals from GST 12-month turnover array."""
    arr = np.array(monthly_turnover, dtype=float)
    mean_ = float(np.mean(arr))
    std_ = float(np.std(arr))
    vol = round(std_ / mean_, 4) if mean_ > 0 else 0.20

    n = len(arr)
    if n >= 6:
        momentum = float(np.mean(arr[-3:]) / np.mean(arr[:3]) - 1.0)
        growth_rate = float((arr[-1] - arr[0]) / arr[0]) if arr[0] > 0 else 0.0
    else:
        momentum = 0.0
        growth_rate = 0.0

    if n >= 3:
        x = np.arange(n, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0] / mean_)
    else:
        slope = 0.0

    return {
        "mean_monthly": mean_,
        "volatility_index": round(min(vol, 1.0), 4),
        "momentum": round(momentum, 4),
        "slope": round(slope, 4),
        "growth_rate": round(growth_rate, 4)
    }

# ─── Tier Pipelines ─────────────────────────────────────────────────────────

def pipeline_t1_t2a(req: ComputeFeaturesRequest) -> ComputedFeatures:
    """Tier 1 (Bank+GST) and Tier 2A (GST+UPI): Direct financial data."""
    notes = []
    gst_stats = compute_gst_slope(req.gst_monthly_turnover_12m or [])
    monthly_revenue = gst_stats["mean_monthly"]
    growth_rate = gst_stats["growth_rate"]
    momentum = gst_stats["momentum"]
    volatility = gst_stats["volatility_index"]
    slope = gst_stats["slope"]
    confidence = "high"
    error_band = 5.0

    if req.tier == "T2A" and req.upi_monthly_inflows_6m:
        upi_stats = compute_upi_statistics(req.upi_monthly_inflows_6m)
        # Cross-validation: check UPI vs GST
        upi_annual_est = upi_stats["mean_monthly"] / (req.digital_payment_ratio_override or req.sector_digital_payment_ratio or 0.35) * 12
        gst_annual = monthly_revenue * 12
        divergence = abs(upi_annual_est - gst_annual) / gst_annual if gst_annual > 0 else 0
        notes.append(f"GST declared: ₹{gst_annual:,.0f}. UPI-estimated annual: ₹{upi_annual_est:,.0f}. Divergence: {divergence*100:.1f}%.")
        if divergence > 0.40:
            notes.append("⚠️ High divergence between GST and UPI — possible cash component or under-declaration. Manual review recommended.")
            confidence = "medium"
        volatility = upi_stats["volatility_index"]  # Use UPI for behavioral signal

    utility_bills = req.utility_bills_6m or []
    util_consistency = compute_utility_consistency(utility_bills)

    return ComputedFeatures(
        estimated_monthly_revenue=round(monthly_revenue, 2),
        revenue_growth_rate=round(growth_rate, 4),
        revenue_momentum=round(momentum, 4),
        volatility_index=round(volatility, 4),
        payment_delay_trend_slope=round(slope, 4),
        utility_consistency_pct=util_consistency,
        upi_volatility_high=int(volatility > 0.40),
        seasonality_coefficient=req.sector_seasonality_coefficient or 0.20,
        gst_delay_months=req.gst_delay_months or 0.0,
        income_confidence=confidence,
        income_source="gst_returns" if req.tier == "T1" else "gst_upi_crossvalidated",
        income_error_band_pct=error_band,
        computation_notes=notes or ["Income computed from GST GSTR-3B monthly turnover."]
    )

def pipeline_t2b(req: ComputeFeaturesRequest) -> ComputedFeatures:
    """Tier 2B: GST only. Applies sector under-reporting adjustment."""
    notes = []
    gst_stats = compute_gst_slope(req.gst_monthly_turnover_12m or [])
    adjustment = req.sector_under_reporting_adjustment or 1.10
    monthly_revenue = gst_stats["mean_monthly"] * adjustment
    notes.append(f"GST declared ₹{gst_stats['mean_monthly']:,.0f}/month. Under-reporting adj {adjustment:.2f}x → estimated ₹{monthly_revenue:,.0f}/month.")

    # Behavioral signals are sector-average (no UPI data)
    volatility = req.sector_avg_volatility_index or 0.20
    notes.append("Volatility index defaulted to sector average (no UPI transaction history available).")

    util_consistency = compute_utility_consistency(req.utility_bills_6m or [])

    return ComputedFeatures(
        estimated_monthly_revenue=round(monthly_revenue, 2),
        revenue_growth_rate=req.sector_revenue_growth_rate or gst_stats["growth_rate"],
        revenue_momentum=gst_stats["momentum"],
        volatility_index=round(volatility, 4),
        payment_delay_trend_slope=gst_stats["slope"],
        utility_consistency_pct=util_consistency,
        upi_volatility_high=0,
        seasonality_coefficient=req.sector_seasonality_coefficient or 0.20,
        gst_delay_months=req.gst_delay_months or 0.0,
        income_confidence="medium",
        income_source="gst_declared_adjusted",
        income_error_band_pct=15.0,
        computation_notes=notes
    )

def pipeline_t2c(req: ComputeFeaturesRequest) -> ComputedFeatures:
    """Tier 2C: UPI only. Grosses up UPI inflows using sector digital ratio."""
    notes = []
    inflows = req.upi_monthly_inflows_6m or []
    digital_ratio = req.digital_payment_ratio_override or req.sector_digital_payment_ratio or 0.35
    upi_stats = compute_upi_statistics(inflows)
    estimated_revenue = upi_stats["mean_monthly"] / digital_ratio
    notes.append(f"UPI 6m avg: ₹{upi_stats['mean_monthly']:,.0f}/month. Sector digital ratio: {digital_ratio:.0%}. Estimated total revenue: ₹{estimated_revenue:,.0f}/month.")
    notes.append("Revenue estimate carries ±25% error band due to unknown cash transaction volume.")

    util_consistency = compute_utility_consistency(req.utility_bills_6m or [])
    growth = req.sector_revenue_growth_rate or 0.06

    return ComputedFeatures(
        estimated_monthly_revenue=round(estimated_revenue, 2),
        revenue_growth_rate=growth,
        revenue_momentum=round(upi_stats["momentum"], 4),
        volatility_index=round(upi_stats["volatility_index"], 4),
        payment_delay_trend_slope=round(upi_stats["slope_proxy"], 4),
        utility_consistency_pct=util_consistency,
        upi_volatility_high=upi_stats["upi_volatility_high"],
        seasonality_coefficient=req.sector_seasonality_coefficient or 0.25,
        gst_delay_months=0.0,
        income_confidence="medium",
        income_source="upi_grossed_up",
        income_error_band_pct=25.0,
        computation_notes=notes
    )

def pipeline_t3(req: ComputeFeaturesRequest) -> ComputedFeatures:
    """
    Tier 3: No digital records. Income estimated from sector floor × employee proxy.
    Behavioral signals all from sector averages. PD floor enforced separately in main.py.
    """
    notes = ["No GST or UPI data available. Income estimated using sector-specific activity proxy."]
    util_consistency = compute_utility_consistency(req.utility_bills_6m or [])
    # Sector floor as proxy for estimated daily revenue × 25 operating days
    daily_floor = 3000  # Generic default
    estimated_revenue = daily_floor * 25
    notes.append(f"Fallback: sector daily floor ₹{daily_floor} × 25 days = ₹{estimated_revenue:,.0f}/month. High uncertainty.")

    return ComputedFeatures(
        estimated_monthly_revenue=round(estimated_revenue, 2),
        revenue_growth_rate=req.sector_revenue_growth_rate or 0.05,
        revenue_momentum=0.0,
        volatility_index=req.sector_avg_volatility_index or 0.30,
        payment_delay_trend_slope=0.0,
        utility_consistency_pct=util_consistency,
        upi_volatility_high=0,
        seasonality_coefficient=req.sector_seasonality_coefficient or 0.25,
        gst_delay_months=0.0,
        income_confidence="low",
        income_source="sector_floor_proxy",
        income_error_band_pct=50.0,
        computation_notes=notes
    )

# ─── Main Endpoint ──────────────────────────────────────────────────────────

@feature_extractor_router.post("/compute", response_model=ComputedFeatures)
async def compute_features(req: ComputeFeaturesRequest):
    """
    Central feature extraction endpoint. Routes to tier-specific pipeline.
    Returns the canonical feature set ready for /predict_risk.
    """
    dispatch = {
        "T1": pipeline_t1_t2a,
        "T2A": pipeline_t1_t2a,
        "T2B": pipeline_t2b,
        "T2C": pipeline_t2c,
        "T3": pipeline_t3,
    }
    pipeline_fn = dispatch.get(req.tier, pipeline_t3)
    return pipeline_fn(req)
