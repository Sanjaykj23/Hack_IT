import React from 'react';
import PortfolioDashboard from './components/PortfolioDashboard';
import SingleBorrowerRiskView from './components/SingleBorrowerRiskView';
import { Shield } from 'lucide-react';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <nav className="bg-blue-900 border-b border-blue-800 sticky top-0 z-10 shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-2">
              <Shield className="h-8 w-8 text-blue-400" />
              <span className="text-xl font-bold text-white tracking-wide">FlowBank <span className="text-blue-300 font-light">MSME Risk Orchestrator</span></span>
            </div>
            <div className="text-blue-200 text-sm font-medium">
              Hybrid Risk Engine Active
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Phase 5: Single Borrower Risk View */}
        <section>
          <div className="mb-4">
            <h2 className="text-lg font-bold text-gray-800">1. Individual Loan Underwriting</h2>
            <p className="text-sm text-gray-500">Layer 1 (Behavioral) & Layer 2 (Policy) Analysis</p>
          </div>
          <SingleBorrowerRiskView />
        </section>

        <hr className="border-gray-200" />

        {/* Phase 5: Portfolio Dashboard & Scenario Simulator */}
        <section>
          <div className="mb-4">
            <h2 className="text-lg font-bold text-gray-800">2. Enterprise Risk Management</h2>
            <p className="text-sm text-gray-500">Layer 3 (Portfolio Concentration) & Macro Stress Scenarios</p>
          </div>
          <PortfolioDashboard />
        </section>
      </main>
    </div>
  );
}

export default App;
