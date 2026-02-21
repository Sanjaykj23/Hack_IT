import { create } from 'zustand';
import axios from 'axios';

const useRiskStore = create((set, get) => ({
    portfolioSummary: null,
    singleBorrowerResult: null,
    isLoading: false,
    error: null,

    // Scenario Simulator State
    scenarioStressFactor: 0.0,

    setStressFactor: (val) => {
        set({ scenarioStressFactor: val });
        get().fetchPortfolioSummary();
    },

    fetchPortfolioSummary: async () => {
        set({ isLoading: true, error: null });
        try {
            const { scenarioStressFactor } = get();
            const response = await axios.get(`http://localhost:8000/portfolio/summary?stress_factor=${scenarioStressFactor}`);
            set({ portfolioSummary: response.data, isLoading: false });
        } catch (err) {
            set({ error: err.message, isLoading: false });
        }
    },

    submitSingleBorrower: async (borrowerData) => {
        set({ isLoading: true, error: null });
        try {
            const response = await axios.post('http://localhost:8000/predict_risk', borrowerData);
            set({ singleBorrowerResult: response.data, isLoading: false });
        } catch (err) {
            set({ error: err.response?.data?.detail || err.message, isLoading: false });
        }
    }
}));

export default useRiskStore;
