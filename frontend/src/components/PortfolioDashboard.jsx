import React, { useEffect } from 'react';
import useRiskStore from '../store/riskStore';
import { ShieldAlert, TrendingUp, AlertTriangle, Briefcase, Activity } from 'lucide-react';

const formatCurrency = (val) => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(val);

const PortfolioDashboard = () => {
    const { portfolioSummary, isLoading, fetchPortfolioSummary, scenarioStressFactor, setStressFactor } = useRiskStore();

    useEffect(() => {
        fetchPortfolioSummary();
    }, []);

    return (
        <div className="bg-white p-6 rounded-xl shadow-lg border border-gray-100">
            <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold flex items-center gap-2 text-gray-800">
                    <Briefcase className="text-blue-600" /> Enterprise Portfolio Risk
                </h2>
                {isLoading && <span className="text-sm text-blue-500 animate-pulse">Recalculating...</span>}
            </div>

            {portfolioSummary ? (
                <div className="space-y-8">
                    {/* Top Level KPIs */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="bg-blue-50 p-5 rounded-lg border border-blue-100">
                            <p className="text-blue-600 text-sm font-semibold uppercase tracking-wider">Total Exposure</p>
                            <h3 className="text-3xl font-bold text-blue-900 mt-2">{formatCurrency(portfolioSummary.total_exposure)}</h3>
                            <p className="text-blue-500 text-xs mt-1">Across {portfolioSummary.total_loans} active MSME loans</p>
                        </div>
                        <div className="bg-orange-50 p-5 rounded-lg border border-orange-100">
                            <p className="text-orange-600 text-sm font-semibold uppercase tracking-wider">Expected Loss (EL)</p>
                            <h3 className="text-3xl font-bold text-orange-900 mt-2">{formatCurrency(portfolioSummary.total_expected_loss)}</h3>
                            <p className="text-orange-500 text-xs mt-1">Weighted Avg PD: {(portfolioSummary.weighted_average_pd * 100).toFixed(2)}%</p>
                        </div>
                        <div className="bg-red-50 p-5 rounded-lg border border-red-100">
                            <div className="flex justify-between items-start">
                                <div>
                                    <p className="text-red-600 text-sm font-semibold uppercase tracking-wider">Value-at-Risk (95%)</p>
                                    <h3 className="text-3xl font-bold text-red-900 mt-2">{formatCurrency(portfolioSummary.var_95)}</h3>
                                </div>
                                <ShieldAlert className="text-red-400 opacity-50 h-10 w-10" />
                            </div>
                            <p className="text-red-500 text-xs mt-1">Capital Required: {formatCurrency(portfolioSummary.capital_required)}</p>
                        </div>
                    </div>

                    {/* Scenario Simulator */}
                    <div className="bg-gray-50 border border-gray-200 p-6 rounded-xl">
                        <h3 className="text-lg font-bold flex items-center gap-2 text-gray-800 mb-4">
                            <Activity className="text-purple-600" /> Macro Stress Simulator
                        </h3>
                        <div className="space-y-6">
                            <div>
                                <div className="flex justify-between mb-2">
                                    <label className="text-sm font-medium text-gray-700">Systemic Default Stress Factor</label>
                                    <span className="text-sm font-bold text-purple-600">+{Math.round(scenarioStressFactor * 100)}%</span>
                                </div>
                                <input
                                    type="range"
                                    min="0" max="0.5" step="0.01"
                                    value={scenarioStressFactor}
                                    onChange={(e) => setStressFactor(parseFloat(e.target.value))}
                                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                                />
                            </div>

                            {scenarioStressFactor > 0 && (
                                <div className="flex items-center gap-4 bg-purple-100 text-purple-800 p-4 rounded-lg animation-fade-in">
                                    <AlertTriangle className="h-6 w-6 text-purple-600" />
                                    <div>
                                        <p className="font-semibold">Stress Adjusted Expected Loss: {formatCurrency(portfolioSummary.stress_adjusted_el)}</p>
                                        <p className="text-sm opacity-80">Portfolio PD distribution shifts heavily right under this scenario.</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            ) : (
                <div className="h-64 flex items-center justify-center text-gray-400">Loading Portfolio Data...</div>
            )}
        </div>
    );
};

export default PortfolioDashboard;
