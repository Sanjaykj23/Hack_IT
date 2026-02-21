import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

stability_router = APIRouter(prefix="/stability", tags=["Stability Scorecard"])

# ─── Pydantic Models ─────────────────────────────────────────────────────────

class UtilityBillRecord(BaseModel):
    expected_amount: float
    paid_amount: float
    days_late: int = 0

class StabilityScorecardRequest(BaseModel):
    # Location & longevity
    years_at_current_location: float                    # Weight: 25%
    employee_count: int                                 # Weight: 20%
    years_with_primary_supplier: float                  # Weight: 15%
    owns_premises: bool = False                         # Weight: 10%
    trade_licence_registered: bool = False              # Weight: 10%
    trade_association_member: bool = False              # Weight:  5%
    # Utility discipline (replaces utility_consistency_pct)
    utility_bills_6m: Optional[List[UtilityBillRecord]] = None  # Weight: 15%
    # Optional enrichers (not scored but used for loan cap calculation)
    estimated_daily_customers: Optional[int] = None
    sector_daily_revenue_floor: Optional[float] = 3000.0
    operating_days_per_month: Optional[int] = 25

class StabilityScorecardResult(BaseModel):
    stability_score: float              # 0–100
    stability_grade: str                # A / B / C / D
    utility_consistency_pct: float      # Computed from bills
    estimated_monthly_revenue: float    # Sector floor × employee proxy
    pd_estimate: float                  # 0.18–0.65 based on score
    max_recommended_loan: float         # Score × sector cap multiplier
    score_breakdown: dict               # Component-by-component scores
    interpretation: str

# ─── Scoring Logic ───────────────────────────────────────────────────────────

def score_stability(req: StabilityScorecardRequest) -> StabilityScorecardResult:
    """
    Rule-based stability scorecard for Tier 3 borrowers.

    A high stability score does NOT guarantee low PD — income is unverified.
    It reduces PD within a constrained [0.18, 0.65] band.
    The band itself communicates the irreducible information uncertainty.
    """
    breakdown = {}

    # 1. Years at current location (25%) — capped at 10 years
    loc_score = min(req.years_at_current_location, 10.0) / 10.0 * 25.0
    breakdown["location_stability"] = round(loc_score, 2)

    # 2. Employee count (20%) — capped at 20 employees
    emp_score = min(req.employee_count, 20) / 20.0 * 20.0
    breakdown["employee_count"] = round(emp_score, 2)

    # 3. Supplier relationship duration (15%) — capped at 8 years
    supp_score = min(req.years_with_primary_supplier, 8.0) / 8.0 * 15.0
    breakdown["supplier_relationship"] = round(supp_score, 2)

    # 4. Premises ownership (10%)
    own_score = 10.0 if req.owns_premises else 0.0
    breakdown["premises_ownership"] = own_score

    # 5. Trade licence (10%)
    lic_score = 10.0 if req.trade_licence_registered else 0.0
    breakdown["trade_licence"] = lic_score

    # 6. Trade association membership (5%)
    assoc_score = 5.0 if req.trade_association_member else 0.0
    breakdown["trade_association"] = assoc_score

    # 7. Utility consistency (15%) — from bill records
    bills = req.utility_bills_6m or []
    if bills:
        on_time = sum(1 for b in bills if b.paid_amount >= b.expected_amount and b.days_late <= 7)
        util_pct = on_time / len(bills)
    else:
        util_pct = 0.70  # Neutral assumption if no bills provided
    util_score = util_pct * 15.0
    breakdown["utility_discipline"] = round(util_score, 2)

    total_score = sum(breakdown.values())
    breakdown["total"] = round(total_score, 2)

    # ── PD Mapping (constrained band: 0.18 to 0.65) ─────────────────────────
    # Even a perfect score doesn't go below 0.18 — irreducible information risk.
    # A zero score doesn't go above 0.65 — floor from sector stability.
    pd_estimate = round(0.65 - (total_score / 100.0) * 0.47, 4)
    pd_estimate = max(0.18, min(0.65, pd_estimate))

    # ── Grade ────────────────────────────────────────────────────────────────
    if total_score >= 75:
        grade = "A"
        interp = "Strong stability signals. Business is deeply embedded in its community and location. Proceed with conservative loan structure."
    elif total_score >= 55:
        grade = "B"
        interp = "Moderate stability. Business has established presence but lacks some formal verifications. Proceed with reduced LTV."
    elif total_score >= 35:
        grade = "C"
        interp = "Limited stability signals. High information asymmetry. Consider smaller pilot loan with short tenure."
    else:
        grade = "D"
        interp = "Insufficient verifiable stability. Recommend field visit or additional references before proceeding."

    # ── Revenue Estimate ─────────────────────────────────────────────────────
    # Sector floor × operating days, scaled by employee count as activity proxy
    employee_multiplier = 1.0 + min(req.employee_count, 20) * 0.05  # each employee adds 5% to floor
    estimated_revenue = req.sector_daily_revenue_floor * req.operating_days_per_month * employee_multiplier

    # ── Max Loan ─────────────────────────────────────────────────────────────
    # Cap: 3× estimated monthly revenue × stability_score factor
    score_factor = 0.5 + (total_score / 100.0) * 1.5   # ranges 0.5 to 2.0
    max_loan = estimated_revenue * 3.0 * score_factor

    return StabilityScorecardResult(
        stability_score=round(total_score, 2),
        stability_grade=grade,
        utility_consistency_pct=round(util_pct, 4),
        estimated_monthly_revenue=round(estimated_revenue, 2),
        pd_estimate=pd_estimate,
        max_recommended_loan=round(max_loan, 2),
        score_breakdown=breakdown,
        interpretation=interp
    )

# ─── Endpoint ────────────────────────────────────────────────────────────────

@stability_router.post("/score", response_model=StabilityScorecardResult)
async def compute_stability_score(req: StabilityScorecardRequest):
    """
    Tier 3 stability scorecard endpoint.
    Accepts field-verifiable proxies for businesses with no digital footprint.
    Returns a scored PD estimate within the constrained [0.18, 0.65] band.
    """
    return score_stability(req)
