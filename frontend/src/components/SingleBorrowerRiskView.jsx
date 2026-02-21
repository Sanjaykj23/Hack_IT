import React, { useState, useRef, useCallback, useEffect } from 'react';
import Papa from 'papaparse';
import axios from 'axios';
import useRiskStore from '../store/riskStore';
import {
    User, Activity, FileText, CheckCircle2, XCircle,
    UploadCloud, ClipboardList, Building2, TrendingUp,
    CreditCard, AlertTriangle, Info, Download, Wifi,
    MapPin, Users, ShieldCheck, Zap, ChevronRight,
    BarChart3, Landmark
} from 'lucide-react';

const API = 'http://localhost:8000';

// ─── Constants ───────────────────────────────────────────────────────────────
const ENTERPRISE_CATEGORIES = ['Micro', 'Small', 'Medium'];
const BUSINESS_TYPES = ['Manufacturing', 'Services', 'Trading'];
const INDUSTRY_SECTORS = [
    'Textile', 'Food Processing', 'Agriculture', 'IT Services',
    'Auto Components', 'General Manufacturing', 'Retail',
    'Hospitality', 'Healthcare', 'Education Services'
];
const LOAN_TENURES = [12, 24, 36, 48, 60];
const MACRO_FACTORS = ['Baseline', 'Festival Season (Boom)', 'Inflation Spike (Stress)', 'Industry Slowdown (Severe Stress)'];

const SEGMENT_META = {
    T1: { label: 'Full Financial', color: 'green', icon: Landmark, desc: 'GST + Bank Statement available' },
    T2A: { label: 'GST + UPI Validated', color: 'green', icon: ShieldCheck, desc: 'GST registered + accepts UPI' },
    T2B: { label: 'GST Declared Only', color: 'yellow', icon: FileText, desc: 'GST registered, no UPI records' },
    T2C: { label: 'Digital Informal', color: 'orange', icon: Wifi, desc: 'UPI-accepting, not GST registered' },
    T3: { label: 'Stability-Only', color: 'red', icon: MapPin, desc: 'No GST, no UPI — cash economy' },
};

const EMPTY_FORM = {
    enterprise_category: '', business_type: '', industry_sector: '',
    years_in_operation: '', annual_turnover: '', monthly_revenue: '',
    gst_registered: 0, udyam_registered: 0, property_value: '0',
    machinery_value: '0', inventory_value: '0', revenue_growth_rate: '',
    revenue_momentum: '0', volatility_index: '', payment_delay_trend_slope: '0',
    seasonality_coefficient: '', existing_loans: '0', existing_emi: '0',
    bank_account_years: '', credit_history_length: '0', delayed_payments: '0',
    past_defaults: '0', gst_delay_months: '0', utility_consistency_pct: '',
    upi_volatility_high: 0, loan_amount_requested: '', loan_tenure_months: '',
    macro_stress_factor: 'Baseline', borrower_segment: 'T1',
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
const ic = (err) => `w-full mt-1 px-3 py-2 border rounded-lg text-sm focus:outline-none focus:ring-2 transition-all ${err ? 'border-red-400 focus:ring-red-300 bg-red-50' : 'border-gray-200 focus:ring-blue-300 bg-white'}`;
const Label = ({ children, req: required }) => <label className="block text-xs font-semibold text-gray-500 uppercase tracking-wide">{children}{required && <span className="text-red-500 ml-1">*</span>}</label>;
const Err = ({ msg }) => msg ? <p className="text-xs text-red-500 mt-1">{msg}</p> : null;

const Toggle = ({ label, fieldKey, value, onChange }) => (
    <div className="flex items-center justify-between mt-1">
        <span className="text-sm text-gray-600">{label}</span>
        <button type="button" onClick={() => onChange(fieldKey, value === 1 ? 0 : 1)}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-300 ${value === 1 ? 'bg-blue-600' : 'bg-gray-300'}`}>
            <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${value === 1 ? 'translate-x-6' : 'translate-x-1'}`} />
        </button>
    </div>
);

const Chip = ({ label, value }) => (
    <div className="flex items-center justify-between bg-blue-50 border border-blue-100 rounded-lg px-3 py-2">
        <span className="text-xs text-blue-600 font-medium">{label}</span>
        <div className="text-right">
            <span className="text-sm font-bold text-blue-800">{value}</span>
            <span className="text-xs text-blue-400 ml-1">(auto)</span>
        </div>
    </div>
);

const Section = ({ icon: IconSection, title, color = 'blue', children }) => (
    <div className={`border border-gray-100 rounded-xl p-4 bg-gray-50/50 space-y-3`}>
        <h4 className={`text-sm font-bold text-${color}-800 flex items-center gap-2`}>
            {IconSection && <IconSection className="h-4 w-4" />} {title}
        </h4>
        <div className="grid grid-cols-2 gap-x-4 gap-y-3">{children}</div>
    </div>
);

// ─── STEP 1: Segment Gateway ──────────────────────────────────────────────────
const SegmentGateway = ({ onSegmentSelected }) => {
    const [gst, setGst] = useState(null);
    const [upi, setUpi] = useState(null);
    const [bank, setBank] = useState(null);
    const [loading, setLoading] = useState(false);
    const [segResult, setSegResult] = useState(null);

    const canClassify = gst !== null && upi !== null && bank !== null;

    const classify = async () => {
        setLoading(true);
        try {
            const res = await axios.post(`${API}/segment/classify?has_gst=${gst}&has_upi=${upi}&has_bank_statement=${bank}`);
            setSegResult(res.data);
        } catch {
            // Fallback to local logic
            let tier = 'T3';
            if (bank && gst) tier = 'T1';
            else if (gst && upi) tier = 'T2A';
            else if (gst) tier = 'T2B';
            else if (upi) tier = 'T2C';
            setSegResult({ tier, label: SEGMENT_META[tier].label, interest_premium_pct: { T1: 0, T2A: 0, T2B: 1.5, T2C: 3, T3: 5 }[tier] });
        } finally {
            setLoading(false);
        }
    };

    const Opt = ({ val, set, yes, no }) => (
        <div className="flex gap-2 mt-1">
            <button type="button" onClick={() => set(true)} className={`flex-1 py-2 rounded-lg text-sm font-semibold border transition-all ${val === true ? 'bg-green-600 text-white border-green-600' : 'border-gray-200 text-gray-600 hover:bg-gray-50'}`}>{yes || 'Yes'}</button>
            <button type="button" onClick={() => set(false)} className={`flex-1 py-2 rounded-lg text-sm font-semibold border transition-all ${val === false ? 'bg-red-500 text-white border-red-500' : 'border-gray-200 text-gray-600 hover:bg-gray-50'}`}>{no || 'No'}</button>
        </div>
    );

    const meta = segResult ? SEGMENT_META[segResult.tier] : null;

    return (
        <div className="space-y-5">
            <div className="bg-blue-50 border border-blue-100 rounded-xl p-4">
                <p className="text-sm font-semibold text-blue-800 flex items-center gap-2">
                    <Info className="h-4 w-4" /> Answer 3 questions to get the right form for your borrower
                </p>
                <p className="text-xs text-blue-600 mt-1">Different documents are required based on the borrower's digital footprint.</p>
            </div>

            <div className="space-y-4">
                <div>
                    <Label>1. Is the business GST registered?</Label>
                    <Opt val={gst} set={setGst} yes="Yes, has GSTIN" no="No GST registration" />
                </div>
                <div>
                    <Label>2. Does the business accept UPI payments?</Label>
                    <Opt val={upi} set={setUpi} yes="Yes, uses PhonePe/GPay/etc." no="No — cash only" />
                </div>
                <div>
                    <Label>3. Is a bank statement available?</Label>
                    <Opt val={bank} set={setBank} yes="Yes, can provide statement" no="No bank statement" />
                </div>
            </div>

            {canClassify && !segResult && (
                <button onClick={classify} disabled={loading}
                    className="w-full bg-blue-600 text-white font-bold py-3 rounded-xl hover:bg-blue-700 transition-all flex items-center justify-center gap-2 text-sm">
                    {loading ? <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" /> : <><ChevronRight className="h-4 w-4" />Identify Borrower Segment</>}
                </button>
            )}

            {segResult && meta && (
                <div className={`border-2 rounded-xl p-4 space-y-3 ${segResult.tier === 'T1' || segResult.tier === 'T2A' ? 'border-green-200 bg-green-50' :
                    segResult.tier === 'T2B' || segResult.tier === 'T2C' ? 'border-amber-200 bg-amber-50' :
                        'border-red-200 bg-red-50'}`}>
                    <div className="flex items-center gap-3">
                        <meta.icon className={`h-8 w-8 ${segResult.tier === 'T1' || segResult.tier === 'T2A' ? 'text-green-600' : segResult.tier === 'T3' ? 'text-red-600' : 'text-amber-600'}`} />
                        <div>
                            <p className="font-bold text-gray-800">{segResult.label || meta.label}</p>
                            <p className="text-xs text-gray-500">{meta.desc}</p>
                        </div>
                        {segResult.interest_premium_pct > 0 && (
                            <div className="ml-auto text-right">
                                <span className="text-xs font-semibold text-gray-500">Interest premium</span>
                                <p className="text-lg font-bold text-amber-600">+{segResult.interest_premium_pct}%</p>
                            </div>
                        )}
                    </div>
                    <button onClick={() => onSegmentSelected(segResult.tier)}
                        className="w-full bg-blue-600 text-white font-bold py-2.5 rounded-lg hover:bg-blue-700 transition-all flex items-center justify-center gap-2 text-sm">
                        <ClipboardList className="h-4 w-4" />Proceed to {segResult.tier} Application Form
                    </button>
                </div>
            )}
        </div>
    );
};

// ─── UPI History Input (Tier 2A/2C) ─────────────────────────────────────────
const UpiHistoryInput = ({ values, onChange, computedRevenue, digitalRatio }) => {
    const months = ['Month 1 (oldest)', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6 (latest)'];
    return (
        <div className="space-y-2">
            <p className="text-xs text-gray-500">Enter total UPI credit received each month (from PhonePe/GPay statement)</p>
            <div className="grid grid-cols-3 gap-2">
                {months.map((m, i) => (
                    <div key={i}>
                        <label className="text-xs text-gray-400">{m}</label>
                        <input type="number" value={values[i] || ''} onChange={e => { const v = [...values]; v[i] = parseFloat(e.target.value) || 0; onChange(v); }}
                            placeholder="₹ amount" className="w-full mt-0.5 px-2 py-1.5 border border-gray-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-300" />
                    </div>
                ))}
            </div>
            {computedRevenue > 0 && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-2 text-xs text-green-800 flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span>Estimated monthly revenue: <strong>₹{computedRevenue.toLocaleString('en-IN')}</strong> (UPI avg ÷ {(digitalRatio * 100).toFixed(0)}% sector digital ratio)</span>
                </div>
            )}
        </div>
    );
};

// ─── Utility Bills Input ───────────────────────────────────────────────────
const UtilityBillsInput = ({ values, onChange, consistency }) => {
    const months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Month 6'];
    return (
        <div className="space-y-2">
            <p className="text-xs text-gray-500">Enter electricity/utility bill details for last 6 months</p>
            <div className="grid grid-cols-1 gap-1.5">
                <div className="grid grid-cols-4 gap-2 text-xs font-semibold text-gray-400 px-1">
                    <span>Month</span><span>Expected (₹)</span><span>Paid (₹)</span><span>Days Late</span>
                </div>
                {months.map((m, i) => (
                    <div key={i} className="grid grid-cols-4 gap-2">
                        <span className="text-xs text-gray-500 flex items-center">{m}</span>
                        {['expected_amount', 'paid_amount', 'days_late'].map(field => (
                            <input key={field} type="number" value={values[i]?.[field] || ''} placeholder={field === 'days_late' ? '0' : '₹'}
                                onChange={e => { const v = [...values]; v[i] = { ...v[i], [field]: parseFloat(e.target.value) || 0 }; onChange(v); }}
                                className="px-2 py-1.5 border border-gray-200 rounded-lg text-xs focus:outline-none focus:ring-1 focus:ring-blue-300" />
                        ))}
                    </div>
                ))}
            </div>
            {consistency !== null && (
                <div className={`rounded-lg p-2 text-xs flex items-center gap-2 ${consistency >= 0.8 ? 'bg-green-50 border border-green-200 text-green-700' : consistency >= 0.5 ? 'bg-amber-50 border border-amber-200 text-amber-700' : 'bg-red-50 border border-red-200 text-red-700'}`}>
                    {consistency >= 0.8 ? <CheckCircle2 className="h-3.5 w-3.5" /> : <AlertTriangle className="h-3.5 w-3.5" />}
                    <span>Utility consistency: <strong>{(consistency * 100).toFixed(0)}%</strong> of bills paid on time</span>
                </div>
            )}
        </div>
    );
};

// ─── Stability Scorecard (Tier 3) ─────────────────────────────────────────
const StabilityForm = ({ stab, setStab, score }) => (
    <div className="space-y-3">
        <p className="text-xs text-gray-500">These field-verifiable facts replace financial documentation for cash-economy businesses.</p>
        <div className="grid grid-cols-2 gap-3">
            <div>
                <Label>Years at Current Location</Label>
                <input type="number" value={stab.years_at_current_location || ''} onChange={e => setStab(p => ({ ...p, years_at_current_location: parseFloat(e.target.value) || 0 }))} placeholder="e.g. 8" className={ic(false)} />
                <p className="text-xs text-gray-400 mt-0.5">Score weight: 25%</p>
            </div>
            <div>
                <Label>Employee Count</Label>
                <input type="number" value={stab.employee_count || ''} onChange={e => setStab(p => ({ ...p, employee_count: parseInt(e.target.value) || 0 }))} placeholder="e.g. 25" className={ic(false)} />
                <p className="text-xs text-gray-400 mt-0.5">Score weight: 20%</p>
            </div>
            <div>
                <Label>Years with Main Supplier</Label>
                <input type="number" value={stab.years_with_primary_supplier || ''} onChange={e => setStab(p => ({ ...p, years_with_primary_supplier: parseFloat(e.target.value) || 0 }))} placeholder="e.g. 6" className={ic(false)} />
                <p className="text-xs text-gray-400 mt-0.5">Score weight: 15%</p>
            </div>
            <div className="space-y-2">
                <div className="bg-white border border-gray-200 rounded-lg p-2.5">
                    <Label>Owns Premises</Label>
                    <Toggle label={stab.owns_premises ? 'Yes — owns shop' : 'No — rented'} fieldKey="owns_premises" value={stab.owns_premises ? 1 : 0} onChange={(_, v) => setStab(p => ({ ...p, owns_premises: v === 1 }))} />
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-2.5">
                    <Label>Trade Licence</Label>
                    <Toggle label={stab.trade_licence_registered ? 'Registered' : 'Not registered'} fieldKey="trade_licence_registered" value={stab.trade_licence_registered ? 1 : 0} onChange={(_, v) => setStab(p => ({ ...p, trade_licence_registered: v === 1 }))} />
                </div>
                <div className="bg-white border border-gray-200 rounded-lg p-2.5">
                    <Label>Trade Association</Label>
                    <Toggle label={stab.trade_association_member ? 'Member' : 'Not a member'} fieldKey="trade_association_member" value={stab.trade_association_member ? 1 : 0} onChange={(_, v) => setStab(p => ({ ...p, trade_association_member: v === 1 }))} />
                </div>
            </div>
        </div>
        {score !== null && (
            <div className={`rounded-xl p-3 border-2 ${score >= 75 ? 'border-green-300 bg-green-50' : score >= 55 ? 'border-amber-300 bg-amber-50' : score >= 35 ? 'border-orange-300 bg-orange-50' : 'border-red-300 bg-red-50'}`}>
                <div className="flex items-center justify-between">
                    <span className="text-sm font-bold text-gray-700">Stability Score</span>
                    <div className="flex items-center gap-2">
                        <span className={`text-2xl font-extrabold ${score >= 75 ? 'text-green-700' : score >= 55 ? 'text-amber-700' : 'text-red-700'}`}>{score.toFixed(0)}</span>
                        <span className="text-xs text-gray-500">/ 100</span>
                    </div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                    <div className={`h-2 rounded-full transition-all ${score >= 75 ? 'bg-green-500' : score >= 55 ? 'bg-amber-500' : 'bg-red-500'}`} style={{ width: `${score}%` }} />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                    Grade {score >= 75 ? 'A' : score >= 55 ? 'B' : score >= 35 ? 'C' : 'D'} · Est. PD range: {(0.65 - score / 100 * 0.47).toFixed(2)} (constrained {'>'}0.18)
                </p>
            </div>
        )}
    </div>
);

// ─── Main Component ───────────────────────────────────────────────────────────
const SingleBorrowerRiskView = () => {
    const { singleBorrowerResult, isLoading, submitSingleBorrower, error } = useRiskStore();
    const [step, setStep] = useState('gateway'); // 'gateway' | 'form'
    const [segment, setSegment] = useState('T1');
    const [tab, setTab] = useState('form'); // 'form' | 'csv'
    const [formData, setFormData] = useState({ ...EMPTY_FORM });
    const [errors, setErrors] = useState({});
    const [sectorBenchmark, setSectorBenchmark] = useState(null);

    // UPI + utility state
    const [upiValues, setUpiValues] = useState([0, 0, 0, 0, 0, 0]);
    const [utilityBills, setUtilityBills] = useState(Array(6).fill({ expected_amount: 0, paid_amount: 0, days_late: 0 }));
    const [stabilityData, setStabilityData] = useState({ years_at_current_location: 0, employee_count: 0, years_with_primary_supplier: 0, owns_premises: false, trade_licence_registered: false, trade_association_member: false });
    const [stabilityScore, setStabilityScore] = useState(null);

    // CSV
    const fileInputRef = useRef();
    const [dragging, setDragging] = useState(false);
    const [csvMsg, setCsvMsg] = useState('');

    const needsUpi = ['T2A', 'T2C'].includes(segment);
    const needsGst = ['T1', 'T2A', 'T2B'].includes(segment);
    const isT3 = segment === 'T3';

    // Auto-fetch sector benchmarks when sector changes
    useEffect(() => {
        if (!formData.industry_sector) return;
        axios.get(`${API}/segment/sector-benchmarks/${encodeURIComponent(formData.industry_sector)}`)
            .then(r => {
                setSectorBenchmark(r.data.benchmark);
                setFormData(prev => ({
                    ...prev,
                    revenue_growth_rate: r.data.benchmark.revenue_growth_rate,
                    seasonality_coefficient: r.data.benchmark.seasonality_coefficient,
                }));
            }).catch(() => setSectorBenchmark(null));
    }, [formData.industry_sector]);

    // Live stability score
    useEffect(() => {
        if (!isT3) { setStabilityScore(null); return; }
        const s = stabilityData;
        const loc = Math.min(s.years_at_current_location || 0, 10) / 10 * 25;
        const emp = Math.min(s.employee_count || 0, 20) / 20 * 20;
        const sup = Math.min(s.years_with_primary_supplier || 0, 8) / 8 * 15;
        const own = s.owns_premises ? 10 : 0;
        const lic = s.trade_licence_registered ? 10 : 0;
        const asc = s.trade_association_member ? 5 : 0;
        const bills = utilityBills.filter(b => b.expected_amount > 0);
        const util = bills.length > 0 ? bills.filter(b => b.paid_amount >= b.expected_amount && b.days_late <= 7).length / bills.length : 0.7;
        setStabilityScore(loc + emp + sup + own + lic + asc + util * 15);
    }, [stabilityData, utilityBills, isT3]);

    // Estimated revenue from UPI
    const upiMean = upiValues.filter(v => v > 0).length > 0 ? upiValues.reduce((a, b) => a + b, 0) / upiValues.filter(v => v > 0).length : 0;
    const digitalRatio = sectorBenchmark?.digital_payment_ratio || 0.35;
    const estimatedRevenue = needsUpi && upiMean > 0 ? Math.round(upiMean / digitalRatio) : 0;

    // Utility consistency
    const utilBills = utilityBills.filter(b => b.expected_amount > 0);
    const utilConsistency = utilBills.length > 0 ? utilBills.filter(b => b.paid_amount >= b.expected_amount && b.days_late <= 7).length / utilBills.length : null;

    const handleChange = (name, value) => {
        setFormData(prev => ({ ...prev, [name]: value }));
        if (errors[name]) setErrors(prev => { const e = { ...prev }; delete e[name]; return e; });
    };
    const handleInput = e => handleChange(e.target.name, e.target.value);

    const validateAndSubmit = async (e) => {
        e.preventDefault();
        const errs = {};
        if (!formData.enterprise_category) errs.enterprise_category = 'Required';
        if (!formData.business_type) errs.business_type = 'Required';
        if (!formData.industry_sector) errs.industry_sector = 'Required';
        if (!formData.loan_amount_requested || parseFloat(formData.loan_amount_requested) <= 0) errs.loan_amount_requested = 'Must be > 0';
        if (!formData.loan_tenure_months) errs.loan_tenure_months = 'Required';

        if (!isT3) {
            if (needsUpi && estimatedRevenue === 0) errs.upi_history = 'Enter at least 2 months of UPI data';
            else if (!needsUpi && (!formData.monthly_revenue || parseFloat(formData.monthly_revenue) <= 0)) errs.monthly_revenue = 'Required';
        }

        setErrors(errs);
        if (Object.keys(errs).length > 0) return;

        // Build payload
        const payload = { ...EMPTY_FORM };
        Object.keys(formData).forEach(k => { if (formData[k] !== '') payload[k] = formData[k]; });

        // Type coercion
        const nums = ['years_in_operation', 'annual_turnover', 'monthly_revenue', 'property_value', 'machinery_value', 'inventory_value', 'revenue_growth_rate', 'revenue_momentum', 'volatility_index', 'payment_delay_trend_slope', 'seasonality_coefficient', 'existing_loans', 'existing_emi', 'bank_account_years', 'credit_history_length', 'delayed_payments', 'past_defaults', 'gst_delay_months', 'utility_consistency_pct', 'loan_amount_requested', 'loan_tenure_months'];
        nums.forEach(k => { payload[k] = parseFloat(payload[k]) || 0; });
        payload.gst_registered = parseInt(payload.gst_registered) || 0;
        payload.udyam_registered = parseInt(payload.udyam_registered) || 0;
        payload.upi_volatility_high = parseInt(payload.upi_volatility_high) || 0;
        payload.borrower_segment = segment;

        // Use computed revenue for UPI-based tiers
        if (needsUpi && estimatedRevenue > 0) {
            payload.monthly_revenue = estimatedRevenue;
            payload.annual_turnover = estimatedRevenue * 12;
        }

        // Use utility consistency
        if (utilConsistency !== null) payload.utility_consistency_pct = utilConsistency;

        // Volatility from UPI
        if (needsUpi && upiValues.some(v => v > 0)) {
            const valid = upiValues.filter(v => v > 0);
            const mean = valid.reduce((a, b) => a + b, 0) / valid.length;
            const std = Math.sqrt(valid.reduce((a, b) => a + (b - mean) ** 2, 0) / valid.length);
            payload.volatility_index = parseFloat((std / mean).toFixed(4)) || 0.15;
            payload.upi_volatility_high = payload.volatility_index > 0.40 ? 1 : 0;
        }

        // T3: use stability floor for revenue + override PD via segment
        if (isT3) {
            const floor = sectorBenchmark?.sector_daily_revenue_floor || 3000;
            const empMultiplier = 1 + Math.min(stabilityData.employee_count || 0, 20) * 0.05;
            payload.monthly_revenue = Math.round(floor * 25 * empMultiplier);
            payload.annual_turnover = payload.monthly_revenue * 12;
            payload.utility_consistency_pct = utilConsistency ?? 0.70;
        }

        submitSingleBorrower(payload);
    };

    // CSV
    const parseCSV = (file) => {
        Papa.parse(file, {
            header: true, skipEmptyLines: true,
            complete: ({ data }) => {
                if (!data.length) { setCsvMsg('Empty CSV'); return; }
                const row = data[0];
                setFormData(prev => {
                    const next = { ...prev };
                    Object.keys(next).forEach(k => { if (k in row) next[k] = row[k]; });
                    return next;
                });
                setCsvMsg(`✅ "${file.name}" loaded. Switch to Full Form to review.`);
                setTab('form');
            },
            error: (e) => setCsvMsg(`Parse error: ${e.message}`)
        });
    };
    const onDrop = useCallback((e) => { e.preventDefault(); setDragging(false); parseCSV(e.dataTransfer.files[0]); }, []);

    // ─── Gateway Step ─────────────────────────────────────────────────────
    if (step === 'gateway') {
        return (
            <div className="grid grid-cols-1 xl:grid-cols-5 gap-8">
                <div className="xl:col-span-3 bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
                    <div className="px-6 pt-6 pb-4 border-b border-gray-100">
                        <h3 className="text-xl font-bold flex items-center gap-2 text-gray-800">
                            <User className="text-blue-600" /> Borrower Segment Classification
                        </h3>
                        <p className="text-sm text-gray-400 mt-1">Determine which loan pathway applies to this borrower before collecting data.</p>
                    </div>
                    <div className="p-6">
                        <SegmentGateway onSegmentSelected={(tier) => { setSegment(tier); setStep('form'); setFormData(prev => ({ ...prev, borrower_segment: tier })); }} />
                    </div>
                </div>
                <div className="xl:col-span-2 bg-white p-6 rounded-xl shadow-lg border border-gray-100 flex flex-col justify-center min-h-[350px]">
                    <div className="space-y-3 opacity-60 text-center">
                        <BarChart3 className="h-14 w-14 text-gray-300 mx-auto" />
                        <p className="text-sm text-gray-500">Complete the segment classification to proceed to the application form.</p>
                        <div className="space-y-2 text-left mt-4">
                            {Object.entries(SEGMENT_META).map(([key, m]) => (
                                <div key={key} className="flex items-center gap-2 text-xs text-gray-400">
                                    <m.icon className="h-3.5 w-3.5" /> <span className="font-semibold">{key}</span> — {m.desc}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    // ─── Form Step ────────────────────────────────────────────────────────
    const meta = SEGMENT_META[segment];

    return (
        <div className="grid grid-cols-1 xl:grid-cols-5 gap-8">
            <div className="xl:col-span-3 bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
                {/* Header */}
                <div className="px-6 pt-5 pb-0 border-b border-gray-100">
                    <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                            <meta.icon className="h-5 w-5 text-blue-600" />
                            <h3 className="text-lg font-bold text-gray-800">{meta.label} Application</h3>
                        </div>
                        <button onClick={() => { setStep('gateway'); }} className="text-xs text-blue-600 hover:underline flex items-center gap-1">
                            ← Change Segment
                        </button>
                    </div>
                    <div className="flex border-b border-gray-100">
                        <button onClick={() => setTab('form')} className={`flex items-center gap-2 px-4 py-2.5 text-sm font-semibold border-b-2 transition-colors ${tab === 'form' ? 'border-blue-600 text-blue-700' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>
                            <ClipboardList className="h-4 w-4" />Application Form
                        </button>
                        <button onClick={() => setTab('csv')} className={`flex items-center gap-2 px-4 py-2.5 text-sm font-semibold border-b-2 transition-colors ${tab === 'csv' ? 'border-blue-600 text-blue-700' : 'border-transparent text-gray-500 hover:text-gray-700'}`}>
                            <UploadCloud className="h-4 w-4" />CSV Upload
                        </button>
                    </div>
                </div>

                <div className="p-6 max-h-[78vh] overflow-y-auto space-y-5">
                    {/* CSV Tab */}
                    {tab === 'csv' && (
                        <div className="space-y-4">
                            <div onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}
                                onClick={() => fileInputRef.current.click()}
                                className={`flex flex-col items-center justify-center gap-3 border-2 border-dashed rounded-xl p-10 cursor-pointer transition-all ${dragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-gray-50 hover:border-blue-400'}`}>
                                <UploadCloud className={`h-12 w-12 ${dragging ? 'text-blue-500' : 'text-gray-400'}`} />
                                <div className="text-center">
                                    <p className="text-sm font-semibold text-gray-700">{dragging ? 'Drop CSV here' : 'Drag & drop CSV file'}</p>
                                    <p className="text-xs text-gray-400 mt-1">or click to browse</p>
                                </div>
                                <input ref={fileInputRef} type="file" accept=".csv" className="hidden" onChange={e => parseCSV(e.target.files[0])} />
                            </div>
                            {csvMsg && <div className="text-xs bg-green-50 border border-green-200 text-green-700 rounded-lg p-3">{csvMsg}</div>}
                            <a href="/sample_borrower_template.csv" download className="inline-flex items-center gap-1 text-xs text-blue-700 font-semibold hover:underline">
                                <Download className="h-3.5 w-3.5" />Download sample CSV template
                            </a>
                        </div>
                    )}

                    {/* Form Tab */}
                    {tab === 'form' && (
                        <form onSubmit={validateAndSubmit} noValidate className="space-y-5">
                            {/* Business Identity */}
                            <Section icon={Building2} title="Business Identity">
                                <div className="col-span-2 grid grid-cols-3 gap-3">
                                    <div>
                                        <Label req>Category</Label>
                                        <select name="enterprise_category" value={formData.enterprise_category} onChange={handleInput} className={ic(errors.enterprise_category)}>
                                            <option value="">Select…</option>
                                            {ENTERPRISE_CATEGORIES.map(c => <option key={c}>{c}</option>)}
                                        </select>
                                        <Err msg={errors.enterprise_category} />
                                    </div>
                                    <div>
                                        <Label req>Business Type</Label>
                                        <select name="business_type" value={formData.business_type} onChange={handleInput} className={ic(errors.business_type)}>
                                            <option value="">Select…</option>
                                            {BUSINESS_TYPES.map(t => <option key={t}>{t}</option>)}
                                        </select>
                                        <Err msg={errors.business_type} />
                                    </div>
                                    <div>
                                        <Label req>Industry Sector</Label>
                                        <select name="industry_sector" value={formData.industry_sector} onChange={handleInput} className={ic(errors.industry_sector)}>
                                            <option value="">Select…</option>
                                            {INDUSTRY_SECTORS.map(s => <option key={s}>{s}</option>)}
                                        </select>
                                    </div>
                                </div>
                                <div>
                                    <Label>Years in Operation</Label>
                                    <input type="number" name="years_in_operation" value={formData.years_in_operation} onChange={handleInput} placeholder="e.g. 5" className={ic(false)} />
                                </div>
                                <div className="col-span-2 grid grid-cols-2 gap-3">
                                    <div className="bg-white border border-gray-200 rounded-lg p-3">
                                        <Label>GST Registered</Label>
                                        <Toggle label={formData.gst_registered ? 'Yes' : 'No'} fieldKey="gst_registered" value={formData.gst_registered} onChange={handleChange} />
                                    </div>
                                    <div className="bg-white border border-gray-200 rounded-lg p-3">
                                        <Label>Udyam Registered</Label>
                                        <Toggle label={formData.udyam_registered ? 'Yes' : 'No'} fieldKey="udyam_registered" value={formData.udyam_registered} onChange={handleChange} />
                                    </div>
                                </div>
                            </Section>

                            {/* Sector Benchmarks (auto-filled) */}
                            {sectorBenchmark && (
                                <div className="rounded-xl border border-blue-100 p-4 bg-blue-50/30 space-y-2">
                                    <p className="text-xs font-bold text-blue-700 flex items-center gap-1"><Info className="h-3.5 w-3.5" />Sector Intelligence (auto-filled from RBI/MSME data)</p>
                                    <div className="grid grid-cols-2 gap-2">
                                        <Chip label="Revenue Growth Rate" value={`${(sectorBenchmark.revenue_growth_rate * 100).toFixed(1)}% p.a.`} />
                                        <Chip label="Seasonality Coefficient" value={sectorBenchmark.seasonality_coefficient.toFixed(2)} />
                                        {needsUpi && <Chip label="Digital Payment Ratio" value={`${(sectorBenchmark.digital_payment_ratio * 100).toFixed(0)}% via UPI`} />}
                                        {!needsUpi && needsGst && <Chip label="Under-Reporting Adj." value={`${sectorBenchmark.under_reporting_adjustment}×`} />}
                                    </div>
                                    <p className="text-xs text-blue-400">{sectorBenchmark.description}</p>
                                </div>
                            )}

                            {/* UPI History (T2A, T2C) */}
                            {needsUpi && (
                                <Section icon={Wifi} title="UPI Transaction History" color="purple">
                                    <div className="col-span-2">
                                        <UpiHistoryInput values={upiValues} onChange={setUpiValues} computedRevenue={estimatedRevenue} digitalRatio={digitalRatio} />
                                        <Err msg={errors.upi_history} />
                                    </div>
                                </Section>
                            )}

                            {/* GST Turnover / Manual Revenue (T1, T2B, non-UPI tiers) */}
                            {!needsUpi && !isT3 && (
                                <Section icon={CreditCard} title="Financials (INR)">
                                    <div>
                                        <Label req>Monthly Revenue</Label>
                                        <input type="number" name="monthly_revenue" value={formData.monthly_revenue} onChange={handleInput} placeholder="e.g. 100000" className={ic(errors.monthly_revenue)} />
                                        <Err msg={errors.monthly_revenue} />
                                    </div>
                                    <div>
                                        <Label>Annual Turnover</Label>
                                        <input type="number" name="annual_turnover" value={formData.annual_turnover} onChange={handleInput} placeholder="e.g. 1200000" className={ic(false)} />
                                    </div>
                                    <div>
                                        <Label>Existing EMI / month</Label>
                                        <input type="number" name="existing_emi" value={formData.existing_emi} onChange={handleInput} placeholder="0" className={ic(false)} />
                                    </div>
                                    <div>
                                        <Label>GST Delay (months)</Label>
                                        <input type="number" name="gst_delay_months" value={formData.gst_delay_months} onChange={handleInput} placeholder="0" className={ic(false)} />
                                    </div>
                                </Section>
                            )}

                            {/* Assets */}
                            {!isT3 && (
                                <Section icon={CreditCard} title="Assets & Credit">
                                    <div>
                                        <Label>Property Value (₹)</Label>
                                        <input type="number" name="property_value" value={formData.property_value} onChange={handleInput} placeholder="0" className={ic(false)} />
                                    </div>
                                    <div>
                                        <Label>Machinery Value (₹)</Label>
                                        <input type="number" name="machinery_value" value={formData.machinery_value} onChange={handleInput} placeholder="0" className={ic(false)} />
                                    </div>
                                    <div>
                                        <Label>Inventory Value (₹)</Label>
                                        <input type="number" name="inventory_value" value={formData.inventory_value} onChange={handleInput} placeholder="0" className={ic(false)} />
                                    </div>
                                    <div>
                                        <Label>Bank Account Age (yrs)</Label>
                                        <input type="number" name="bank_account_years" value={formData.bank_account_years} onChange={handleInput} placeholder="e.g. 3" className={ic(false)} />
                                    </div>
                                    {parseInt(formData.existing_loans) > 0 && (
                                        <>
                                            <div>
                                                <Label>Delayed Payments</Label>
                                                <input type="number" name="delayed_payments" value={formData.delayed_payments} onChange={handleInput} placeholder="0" className={ic(false)} />
                                            </div>
                                            <div>
                                                <Label>Past Defaults</Label>
                                                <input type="number" name="past_defaults" value={formData.past_defaults} onChange={handleInput} placeholder="0" className={ic(false)} />
                                            </div>
                                        </>
                                    )}
                                </Section>
                            )}

                            {/* Utility Bills (T2B, T2C, T3) */}
                            {(['T2B', 'T2C', 'T3'].includes(segment)) && (
                                <Section icon={Zap} title="Utility Payment History" color="amber">
                                    <div className="col-span-2">
                                        <UtilityBillsInput values={utilityBills} onChange={setUtilityBills} consistency={utilConsistency} />
                                    </div>
                                </Section>
                            )}

                            {/* Stability Scorecard (T3) */}
                            {isT3 && (
                                <Section icon={Users} title="Business Stability Credentials" color="purple">
                                    <div className="col-span-2">
                                        <StabilityForm stab={stabilityData} setStab={setStabilityData} score={stabilityScore} />
                                    </div>
                                </Section>
                            )}

                            {/* Loan Request */}
                            <Section icon={FileText} title="Loan Request">
                                <div>
                                    <Label req>Loan Amount (₹)</Label>
                                    <input type="number" name="loan_amount_requested" value={formData.loan_amount_requested} onChange={handleInput} placeholder="e.g. 500000" className={ic(errors.loan_amount_requested)} />
                                    <Err msg={errors.loan_amount_requested} />
                                </div>
                                <div>
                                    <Label req>Tenure (months)</Label>
                                    <select name="loan_tenure_months" value={formData.loan_tenure_months} onChange={handleInput} className={ic(errors.loan_tenure_months)}>
                                        <option value="">Select…</option>
                                        {LOAN_TENURES.filter(t => {
                                            const caps = { T1: 60, T2A: 60, T2B: 48, T2C: 36, T3: 24 };
                                            return t <= (caps[segment] || 60);
                                        }).map(t => <option key={t} value={t}>{t} months</option>)}
                                    </select>
                                    <Err msg={errors.loan_tenure_months} />
                                    {segment !== 'T1' && segment !== 'T2A' && (
                                        <p className="text-xs text-amber-600 mt-0.5">Max tenure for {segment}: {({ T2B: 48, T2C: 36, T3: 24 })[segment]}m</p>
                                    )}
                                </div>
                                <div className="col-span-2">
                                    <Label>Macro Stress Scenario</Label>
                                    <select name="macro_stress_factor" value={formData.macro_stress_factor} onChange={handleInput} className={ic(false)}>
                                        {MACRO_FACTORS.map(m => <option key={m}>{m}</option>)}
                                    </select>
                                </div>
                            </Section>

                            {/* Error Summary */}
                            {Object.keys(errors).length > 0 && (
                                <div className="flex items-start gap-2 bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-xs">
                                    <XCircle className="h-4 w-4 shrink-0 mt-0.5" />
                                    {Object.keys(errors).length} field(s) need attention.
                                </div>
                            )}
                            {error && <div className="text-xs bg-red-50 border border-red-200 text-red-700 rounded-lg p-3">API Error: {error}</div>}

                            <button type="submit" disabled={isLoading}
                                className="w-full bg-blue-600 text-white font-bold py-3 rounded-xl hover:bg-blue-700 active:scale-[0.98] transition-all disabled:opacity-50 flex items-center justify-center gap-2 text-sm">
                                {isLoading
                                    ? <><span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />Running Risk Engine…</>
                                    : <><Activity className="h-4 w-4" />Analyze Borrower Risk ({segment})</>}
                            </button>
                        </form>
                    )}
                </div>
            </div>

            {/* Result Panel */}
            <div className="xl:col-span-2 bg-white p-6 rounded-xl shadow-lg border border-gray-100 flex flex-col justify-center min-h-[400px]">
                {singleBorrowerResult ? (
                    <div className="space-y-4 animate-in slide-in-from-right fade-in duration-500">
                        <div className="flex justify-between items-start border-b pb-4">
                            <div>
                                <h3 className="text-xl font-bold text-gray-800">Risk Assessment</h3>
                                <p className="text-xs text-gray-400 mt-0.5">{singleBorrowerResult.segment_label || meta.label} · {singleBorrowerResult.income_confidence} confidence</p>
                                <p className="text-sm mt-1">
                                    {singleBorrowerResult.policy_rejection_flag
                                        ? <span className="text-red-600 font-bold">{singleBorrowerResult.layer_2_policy_status}</span>
                                        : <span className="text-green-600 font-bold">Approved ✓</span>}
                                </p>
                            </div>
                            <div className="text-center">
                                <div className={`text-3xl font-extrabold rounded-full h-20 w-20 flex items-center justify-center border-4 ${singleBorrowerResult.credit_score_mapped > 700 ? 'text-green-700 border-green-200 bg-green-50' : singleBorrowerResult.credit_score_mapped > 500 ? 'text-amber-700 border-amber-200 bg-amber-50' : 'text-red-700 border-red-200 bg-red-50'}`}>
                                    {singleBorrowerResult.credit_score_mapped}
                                </div>
                                <span className="text-xs font-semibold text-gray-400 uppercase mt-1 block">MSME Score</span>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-3">
                            <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
                                <span className="text-xs font-semibold text-gray-400 uppercase">Probability of Default</span>
                                <div className={`text-2xl font-bold mt-1 ${singleBorrowerResult.layer_3_final_pd > 0.4 ? 'text-red-600' : singleBorrowerResult.layer_3_final_pd > 0.2 ? 'text-amber-600' : 'text-green-600'}`}>
                                    {(singleBorrowerResult.layer_3_final_pd * 100).toFixed(2)}%
                                </div>
                            </div>
                            <div className="bg-gray-50 rounded-xl p-3 border border-gray-100">
                                <span className="text-xs font-semibold text-gray-400 uppercase">Risk-Based Rate</span>
                                <div className="text-2xl font-bold mt-1 text-gray-800">
                                    {singleBorrowerResult.risk_based_interest_rate_pct.toFixed(2)}%
                                </div>
                            </div>
                            <div className="bg-gray-50 rounded-xl p-3 border border-gray-100 col-span-2">
                                <span className="text-xs font-semibold text-gray-400 uppercase">Expected Loss</span>
                                <div className="text-xl font-bold mt-1 text-gray-800">
                                    ₹{singleBorrowerResult.expected_loss_amt.toLocaleString('en-IN')}
                                </div>
                            </div>
                        </div>

                        {/* Segment info strip */}
                        <div className={`rounded-lg px-3 py-2 text-xs flex items-center gap-2 ${singleBorrowerResult.income_confidence === 'high' ? 'bg-green-50 border border-green-200 text-green-700' : singleBorrowerResult.income_confidence === 'medium' ? 'bg-amber-50 border border-amber-200 text-amber-700' : 'bg-red-50 border border-red-200 text-red-700'}`}>
                            <Info className="h-3.5 w-3.5 shrink-0" />
                            <span>Segment: <strong>{singleBorrowerResult.borrower_segment}</strong> — Income confidence: <strong>{singleBorrowerResult.income_confidence}</strong></span>
                        </div>

                        {singleBorrowerResult.policy_rejection_flag && (
                            <div className="flex items-start gap-2 bg-red-50 border border-red-200 text-red-700 rounded-lg p-3 text-xs">
                                <XCircle className="h-4 w-4 shrink-0 mt-0.5" />
                                <span><strong>Policy Rejected:</strong> {singleBorrowerResult.layer_2_policy_status}. Statistical PD preserved for audit.</span>
                            </div>
                        )}

                        <div>
                            <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                                <Activity className="h-4 w-4" />Top Risk Drivers
                            </h4>
                            <ul className="space-y-2">
                                {singleBorrowerResult.top_driver_explanations.length > 0
                                    ? singleBorrowerResult.top_driver_explanations.map((exp, i) => (
                                        <li key={i} className="flex items-start gap-2 text-xs bg-gray-50 p-2.5 rounded-lg border border-gray-100">
                                            {exp.includes('✅') ? <CheckCircle2 className="h-3.5 w-3.5 text-green-500 mt-0.5 shrink-0" /> : <XCircle className="h-3.5 w-3.5 text-orange-500 mt-0.5 shrink-0" />}
                                            <span className="text-gray-700">{exp.replace(/[✅⚠️]/g, '').trim()}</span>
                                        </li>
                                    ))
                                    : <li className="text-xs text-gray-400 italic">Standard profile — no extreme risk drivers detected.</li>
                                }
                            </ul>
                        </div>
                        <div className="text-xs text-gray-300 border-t pt-2">
                            Behavioral PD (Layer 1): {(singleBorrowerResult.layer_1_behavioral_pd * 100).toFixed(3)}%
                        </div>
                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-center space-y-4 opacity-40">
                        <FileText className="h-14 w-14 text-gray-300" />
                        <p className="text-sm text-gray-500">Complete the application and click <strong>Analyze Borrower Risk</strong> to see results.</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SingleBorrowerRiskView;
