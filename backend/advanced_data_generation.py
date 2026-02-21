import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import json

print("Initializing Phase 1 v2: 5-Segment Advanced Synthetic Data Engine...")

np.random.seed(42)
NUM_SAMPLES = 15000

# ─── SECTOR BENCHMARKS ───────────────────────────────────────────────────────
SECTOR_BENCHMARKS = {
    'Textile':             {'growth': 0.05, 'seasonality': 0.35, 'digital_ratio': 0.25, 'adj': 1.10, 'floor': 3000, 'vol': 0.22},
    'Food Processing':     {'growth': 0.08, 'seasonality': 0.45, 'digital_ratio': 0.30, 'adj': 1.15, 'floor': 4000, 'vol': 0.28},
    'Agriculture':         {'growth': 0.04, 'seasonality': 0.65, 'digital_ratio': 0.12, 'adj': 1.25, 'floor': 2000, 'vol': 0.38},
    'IT Services':         {'growth': 0.18, 'seasonality': 0.10, 'digital_ratio': 0.80, 'adj': 1.02, 'floor': 8000, 'vol': 0.12},
    'Auto Components':     {'growth': 0.07, 'seasonality': 0.20, 'digital_ratio': 0.38, 'adj': 1.12, 'floor': 3500, 'vol': 0.18},
    'General Manufacturing':{'growth': 0.06,'seasonality': 0.18, 'digital_ratio': 0.32, 'adj': 1.10, 'floor': 3000, 'vol': 0.20},
    'Retail':              {'growth': 0.09, 'seasonality': 0.50, 'digital_ratio': 0.55, 'adj': 1.20, 'floor': 5000, 'vol': 0.25},
    'Hospitality':         {'growth': 0.10, 'seasonality': 0.55, 'digital_ratio': 0.48, 'adj': 1.18, 'floor': 4500, 'vol': 0.30},
    'Healthcare':          {'growth': 0.12, 'seasonality': 0.15, 'digital_ratio': 0.42, 'adj': 1.08, 'floor': 6000, 'vol': 0.14},
    'Education Services':  {'growth': 0.11, 'seasonality': 0.40, 'digital_ratio': 0.60, 'adj': 1.05, 'floor': 5000, 'vol': 0.16},
}
SECTORS = list(SECTOR_BENCHMARKS.keys())

# ─── SEGMENT DISTRIBUTION ────────────────────────────────────────────────────
# Reflects realistic Indian MSME population composition
SEGMENT_DIST = {
    'T1':  0.10,   # GST + Bank — formal, fully documented
    'T2A': 0.20,   # GST + UPI — semi-formal, cross-validated
    'T2B': 0.15,   # GST only — declared income, traditional payments
    'T2C': 0.30,   # UPI only — digital informal, no GST
    'T3':  0.25,   # Cash only — fully informal, stability proxies
}
SEGMENT_NAMES = list(SEGMENT_DIST.keys())
SEGMENT_PROBS = list(SEGMENT_DIST.values())

SEGMENT_INTEREST_PREMIUM = {'T1': 0.0, 'T2A': 0.0, 'T2B': 1.5, 'T2C': 3.0, 'T3': 5.0}
SEGMENT_INCOME_CONFIDENCE = {'T1': 2, 'T2A': 2, 'T2B': 1, 'T2C': 1, 'T3': 0}  # 2=high,1=medium,0=low

print(f"Generating {NUM_SAMPLES} profiles across 5 borrower segments...")

# ─── SEGMENT ASSIGNMENT ───────────────────────────────────────────────────────
segments = np.random.choice(SEGMENT_NAMES, NUM_SAMPLES, p=SEGMENT_PROBS)

# ─── BASE FEATURES ─────────────────────────────────────────────────────────────
industry_sectors = np.random.choice(SECTORS, NUM_SAMPLES)
enterprise_categories = np.random.choice(['Micro', 'Small', 'Medium'], NUM_SAMPLES, p=[0.70, 0.25, 0.05])

# Segment → GST/UPI flags
gst_registered = np.array([1 if s in ('T1', 'T2A', 'T2B') else 0 for s in segments])
upi_active     = np.array([1 if s in ('T1', 'T2A', 'T2C') else 0 for s in segments])
udyam_registered = np.where(gst_registered == 1,
                             np.random.choice([0, 1], NUM_SAMPLES, p=[0.25, 0.75]),
                             np.random.choice([0, 1], NUM_SAMPLES, p=[0.65, 0.35]))

years_in_operation = np.random.randint(1, 25, NUM_SAMPLES)
macro_stress = np.random.choice(
    ['Baseline', 'Festival Season (Boom)', 'Inflation Spike (Stress)', 'Industry Slowdown (Severe Stress)'],
    NUM_SAMPLES, p=[0.60, 0.15, 0.15, 0.10]
)

# ─── REVENUE GENERATION (per segment) ─────────────────────────────────────────
def assign_turnover(cat, segment):
    """Revenue is segment-aware — T3 businesses are almost always Micro."""
    if segment == 'T3' or cat == 'Micro':
        return np.random.uniform(300000, 3000000)     # ₹3L–₹30L (very small)
    elif cat == 'Small':
        return np.random.uniform(5000000, 50000000)
    else:
        return np.random.uniform(50000000, 250000000)

annual_turnover = np.array([assign_turnover(enterprise_categories[i], segments[i]) for i in range(NUM_SAMPLES)])

# T2B: declared GST turnover is under-reported — actual revenue is higher
# T2C: UPI-inferred revenue is estimated (has noise)
actual_monthly_revenue = np.zeros(NUM_SAMPLES)
for i in range(NUM_SAMPLES):
    s = segments[i]
    bench = SECTOR_BENCHMARKS[industry_sectors[i]]
    base = annual_turnover[i] / 12
    if s == 'T2B':
        # GST declared is understated; actual revenue = declared × adjustment factor
        actual_monthly_revenue[i] = base * bench['adj'] * np.random.uniform(0.95, 1.05)
    elif s == 'T2C':
        # UPI/digital portion is known; gross up to total revenue
        upi_portion = base * bench['digital_ratio']
        actual_monthly_revenue[i] = upi_portion / bench['digital_ratio'] * np.random.uniform(0.90, 1.10)
    elif s == 'T3':
        # Sector floor × employee proxy (added later)
        actual_monthly_revenue[i] = bench['floor'] * 25 * np.random.uniform(0.8, 1.3)
    else:
        actual_monthly_revenue[i] = base * np.random.uniform(0.97, 1.03)

annual_turnover_declared = actual_monthly_revenue * 12  # Reconcile declared to actual

# ─── ASSETS ─────────────────────────────────────────────────────────────────────
property_value = annual_turnover_declared * np.random.uniform(0, 0.5, NUM_SAMPLES)
machinery_value = np.where(
    np.isin(np.array([{'Manufacturing':True}.get(b, False) for b in np.random.choice(['Manufacturing', 'Services', 'Trading'], NUM_SAMPLES, p=[0.3, 0.4, 0.3])]), [True]),
    annual_turnover_declared * np.random.uniform(0.1, 0.8, NUM_SAMPLES),
    annual_turnover_declared * np.random.uniform(0, 0.2, NUM_SAMPLES)
)
inventory_value = actual_monthly_revenue * np.random.uniform(0.5, 2.5, NUM_SAMPLES)
total_asset_value = property_value + machinery_value + inventory_value

business_types = np.random.choice(['Manufacturing', 'Services', 'Trading'], NUM_SAMPLES, p=[0.3, 0.4, 0.3])
machinery_value = np.where(
    business_types == 'Manufacturing',
    annual_turnover_declared * np.random.uniform(0.1, 0.8, NUM_SAMPLES),
    annual_turnover_declared * np.random.uniform(0, 0.2, NUM_SAMPLES)
)
inventory_value = np.where(
    np.isin(business_types, ['Manufacturing', 'Trading']),
    actual_monthly_revenue * np.random.uniform(0.5, 3, NUM_SAMPLES),
    actual_monthly_revenue * np.random.uniform(0, 0.5, NUM_SAMPLES)
)
total_asset_value = property_value + machinery_value + inventory_value

# ─── TEMPORAL INTELLIGENCE ──────────────────────────────────────────────────────
# Revenue growth: sector-seeded but with noise; T3 has more uncertainty
revenue_growth_rate = np.array([
    np.random.normal(SECTOR_BENCHMARKS[industry_sectors[i]]['growth'], 0.08) for i in range(NUM_SAMPLES)
])

revenue_momentum = np.random.normal(0.02, 0.10, NUM_SAMPLES)

# Volatility: T3 and T2C are inherently more volatile
volatility_index = np.array([
    np.random.uniform(
        SECTOR_BENCHMARKS[industry_sectors[i]]['vol'] * 0.8,
        SECTOR_BENCHMARKS[industry_sectors[i]]['vol'] * 1.8
    ) if segments[i] in ('T2C', 'T3') else
    np.random.uniform(
        SECTOR_BENCHMARKS[industry_sectors[i]]['vol'] * 0.5,
        SECTOR_BENCHMARKS[industry_sectors[i]]['vol'] * 1.2
    ) for i in range(NUM_SAMPLES)
])

payment_delay_trend_slope = np.random.normal(0, 0.5, NUM_SAMPLES)

seasonality_coefficient = np.array([
    np.random.normal(SECTOR_BENCHMARKS[industry_sectors[i]]['seasonality'], 0.10)
    for i in range(NUM_SAMPLES)
]).clip(0.05, 0.90)

# ─── BEHAVIORAL PROXIES ──────────────────────────────────────────────────────────
existing_loans = np.random.poisson(1.5, NUM_SAMPLES)
existing_emi   = actual_monthly_revenue * np.random.uniform(0, 0.4, NUM_SAMPLES)

# Bank account age: T3 borrowers have shorter or no history
bank_account_years = np.where(
    np.isin(segments, ['T3']),
    np.random.randint(0, 3, NUM_SAMPLES),
    np.minimum(years_in_operation, np.random.randint(1, 15, NUM_SAMPLES))
)
credit_history_length = (bank_account_years * 12 + np.random.randint(-6, 6, NUM_SAMPLES)).clip(min=0)

# Delayed payments / past defaults: only relevant for existing credit holders
delayed_payments = np.where(existing_loans > 0, np.random.poisson(1.2, NUM_SAMPLES), 0)
past_defaults    = np.where(existing_loans > 0, np.random.poisson(0.3, NUM_SAMPLES), 0)

# GST delay: 0 for non-GST segments (T2C / T3)
gst_delay_months = np.where(
    gst_registered == 1,
    np.random.poisson(0.8, NUM_SAMPLES),
    0  # Not GST registered → no GST delay applicable
)

# Utility consistency: T3 uses this as primary behavioral proxy
utility_consistency_pct = np.random.uniform(0.40, 1.0, NUM_SAMPLES)

# UPI volatility: directly computable for T2C/T2A; set to 0 for T3 (no UPI)
upi_volatility_high = np.where(
    np.isin(segments, ['T3', 'T2B']),
    0,
    (volatility_index > 0.40).astype(int)
)

# ─── NEW FEATURES: STABILITY PROXIES (for T3 segment) ───────────────────────────
# These are field-verifiable facts collected during field visits for T3 applicants
years_at_location        = np.where(segments == 'T3', np.random.uniform(1, 20, NUM_SAMPLES), years_in_operation)
employee_count           = np.where(segments == 'T3', np.random.randint(1, 30, NUM_SAMPLES), np.random.randint(0, 5, NUM_SAMPLES))
supplier_relationship_yrs= np.where(segments == 'T3', np.random.uniform(1, 15, NUM_SAMPLES), np.random.uniform(1, 10, NUM_SAMPLES))
owns_premises            = np.where(segments == 'T3', np.random.choice([0, 1], NUM_SAMPLES, p=[0.7, 0.3]), 0)
trade_licence            = np.where(segments == 'T3', np.random.choice([0, 1], NUM_SAMPLES, p=[0.4, 0.6]), gst_registered)
trade_association_member = np.where(segments == 'T3', np.random.choice([0, 1], NUM_SAMPLES, p=[0.6, 0.4]), 0)

# Stability score (0-100) — computed from proxy inputs
def compute_stability_score_batch(n):
    loc_s  = np.minimum(years_at_location, 10) / 10 * 25
    emp_s  = np.minimum(employee_count, 20) / 20 * 20
    sup_s  = np.minimum(supplier_relationship_yrs, 8) / 8 * 15
    own_s  = owns_premises * 10
    lic_s  = trade_licence * 10
    asc_s  = trade_association_member * 5
    util_s = utility_consistency_pct * 15
    return (loc_s + emp_s + sup_s + own_s + lic_s + asc_s + util_s).clip(0, 100)

stability_score = compute_stability_score_batch(NUM_SAMPLES)

# Income confidence encoded: 2=high (T1/T2A), 1=medium (T2B/T2C), 0=low (T3)
income_confidence_encoded = np.array([SEGMENT_INCOME_CONFIDENCE[s] for s in segments])

# ─── LOAN FEATURES ─────────────────────────────────────────────────────────────
# T3: smaller loans due to policy cap (max LTV 50%, shorter tenure)
loan_amount_multiplier = np.where(segments == 'T3', np.random.uniform(1, 4, NUM_SAMPLES), np.random.uniform(2, 12, NUM_SAMPLES))
loan_amount_requested = actual_monthly_revenue * loan_amount_multiplier

max_tenure_by_segment = {'T1': 60, 'T2A': 60, 'T2B': 48, 'T2C': 36, 'T3': 24}
loan_tenure_months = np.array([
    np.random.choice([t for t in [12, 24, 36, 48, 60] if t <= max_tenure_by_segment[segments[i]]])
    for i in range(NUM_SAMPLES)
])

def calculate_emi(principal, tenure_months, annual_rate=0.12):
    monthly_rate = annual_rate / 12
    return (principal * monthly_rate * ((1 + monthly_rate) ** tenure_months)) / (((1 + monthly_rate) ** tenure_months) - 1)

proposed_emi = calculate_emi(loan_amount_requested, loan_tenure_months)
total_emi    = existing_emi + proposed_emi
emi_to_revenue_ratio = total_emi / actual_monthly_revenue

# ─── GENERATIVE PD FUNCTION (5-SEGMENT AWARE) ────────────────────────────────────
print("Calculating Generative PD across all 5 segments...")

def calculate_generative_pd(i):
    """
    Segment-aware PD in log-odds space.
    - T1/T2A: full financial model (EMI ratio dominant)
    - T2B: GST compliance + EMI ratio   
    - T2C: UPI volatility + EMI ratio + income uncertainty
    - T3: Stability score drives PD within constrained [0.18, 0.65] band
    """
    s = segments[i]

    if s == 'T3':
        # Stability-based PD: no reliable income data.
        # Borrowers who self-apply have passed basic screening — realistic PD band [0.12, 0.40].
        # High stability score → PD approaches 0.12 (low-risk informal)
        # Low stability score → PD approaches 0.40 (high-risk informal)
        # PD > 0.40 for T3 means the bank shouldn't lend, so they don't reach application stage.
        stab = stability_score[i]
        base_pd = float(np.clip(0.40 - (stab / 100) * 0.28, 0.12, 0.40))
        # Macro stress escalation
        if macro_stress[i] == 'Industry Slowdown (Severe Stress)':
            base_pd = min(0.40, base_pd * 1.12)
        elif macro_stress[i] == 'Inflation Spike (Stress)':
            base_pd = min(0.40, base_pd * 1.06)
        # Add realistic noise (±3%)
        return float(np.clip(base_pd + np.random.normal(0, 0.03), 0.12, 0.42))

    # Tiers T1 / T2A / T2B / T2C: logit-space model
    base_pd = 0.05
    logit = np.log(base_pd / (1 - base_pd))  # logit(0.05) ≈ -2.944

    # ── EMI burden ─────────────────────────────────────────────
    emr = emi_to_revenue_ratio[i]
    if emr > 0.65:
        logit += 1.20
    elif emr > 0.50:
        logit += 0.35

    # ── GST compliance signals ──────────────────────────────────
    if gst_registered[i] == 1:
        if gst_delay_months[i] > 2:
            logit += 0.70
        if udyam_registered[i] == 1:
            logit -= 0.18
    else:
        # No GST = no formal verification → small uncertainty penalty
        logit += 0.15

    # ── Behavioral proxies ─────────────────────────────────────
    if utility_consistency_pct[i] < 0.75:
        logit += 0.55
    if upi_volatility_high[i] == 1:
        logit += 0.45

    # ── T2C income uncertainty premium ─────────────────────────
    if s == 'T2C':
        logit += 0.20   # UPI estimation carries ~25% error; adds uncertainty

    # ── Temporal signals ───────────────────────────────────────
    if payment_delay_trend_slope[i] > 0.2:
        logit += 0.30
    if revenue_growth_rate[i] < 0:
        logit += 0.22
    if volatility_index[i] > 0.25:
        logit += 0.25

    # ── Macro stress ───────────────────────────────────────────
    if macro_stress[i] == 'Inflation Spike (Stress)':
        logit += 0.35
    elif macro_stress[i] == 'Industry Slowdown (Severe Stress)':
        logit += 0.80
    elif macro_stress[i] == 'Festival Season (Boom)':
        logit -= 0.12

    pit_pd = 1 / (1 + np.exp(-logit))

    # Segment-specific PD floors
    pd_floor = {'T1': 0.01, 'T2A': 0.01, 'T2B': 0.05, 'T2C': 0.08}.get(s, 0.01)
    return float(np.clip(pit_pd, pd_floor, 0.99))

true_pd = np.array([calculate_generative_pd(i) for i in tqdm(range(NUM_SAMPLES), desc="Computing PDs")])

# Bernoulli sampling — use a fresh seeded RNG for reproducibility
rng = np.random.default_rng(seed=123)
default_flag = rng.binomial(n=1, p=true_pd).astype(int)


def assign_bucket(pd):
    if pd < 0.15: return 'Low'
    elif pd < 0.40: return 'Medium'
    else: return 'High'

risk_bucket = [assign_bucket(p) for p in true_pd]

# ─── ASSEMBLE DATAFRAME ─────────────────────────────────────────────────────────
print("Assembling dataset...")

df = pd.DataFrame({
    'loan_id': [f"LN-{100000+i}" for i in range(NUM_SAMPLES)],

    # Identity
    'enterprise_category': enterprise_categories,
    'business_type': business_types,
    'industry_sector': industry_sectors,
    'years_in_operation': years_in_operation,
    'gst_registered': gst_registered,
    'udyam_registered': udyam_registered,
    'macro_stress_factor': macro_stress,

    # Segment metadata
    'borrower_segment': segments,
    'income_confidence_encoded': income_confidence_encoded,

    # Financial
    'annual_turnover': annual_turnover_declared,
    'monthly_revenue': actual_monthly_revenue,
    'property_value': property_value,
    'machinery_value': machinery_value,
    'inventory_value': inventory_value,
    'total_asset_value': total_asset_value,

    # Temporal intelligence
    'revenue_growth_rate': revenue_growth_rate,
    'revenue_momentum': revenue_momentum,
    'volatility_index': volatility_index,
    'payment_delay_trend_slope': payment_delay_trend_slope,
    'seasonality_coefficient': seasonality_coefficient,

    # Behavioral
    'existing_loans': existing_loans,
    'existing_emi': existing_emi,
    'bank_account_years': bank_account_years,
    'credit_history_length': credit_history_length,
    'delayed_payments': delayed_payments,
    'past_defaults': past_defaults,
    'gst_delay_months': gst_delay_months,
    'utility_consistency_pct': utility_consistency_pct,
    'upi_volatility_high': upi_volatility_high,

    # Stability proxies (T3-specific, zero for other tiers)
    'stability_score': stability_score,
    'years_at_location': years_at_location,
    'employee_count': employee_count,
    'supplier_relationship_yrs': supplier_relationship_yrs,

    # Loan
    'loan_amount_requested': loan_amount_requested,
    'loan_tenure_months': loan_tenure_months,
    'proposed_emi': proposed_emi,
    'total_emi': total_emi,
    'emi_to_revenue_ratio': emi_to_revenue_ratio,

    # Targets
    'true_pd': true_pd,
    'default_flag': default_flag,
    'risk_bucket': risk_bucket
})

# ─── STATISTICS ─────────────────────────────────────────────────────────────────
print(f"\n✅ Dataset shape: {df.shape}")
print("\n📊 Segment Distribution:")
print(df['borrower_segment'].value_counts(normalize=True).round(3))
print("\n📊 Risk Bucket Distribution:")
print(df['risk_bucket'].value_counts(normalize=True).round(3))
print("\n📊 Mean PD by Segment:")
print(df.groupby('borrower_segment')['true_pd'].mean().round(4))
print("\n📊 Default Rate by Segment:")
print(df.groupby('borrower_segment')['default_flag'].mean().round(4))

os.makedirs('data', exist_ok=True)
os.makedirs('backend/data', exist_ok=True)

out_path = 'data/advanced_msme_loan_data.csv'
df.to_csv(out_path, index=False)
df.to_csv('backend/data/advanced_msme_loan_data.csv', index=False)
print(f"\n✅ Saved to {out_path}")
