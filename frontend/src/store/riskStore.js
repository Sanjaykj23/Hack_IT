import { create } from 'zustand';
import axios from 'axios';

const useRiskStore = create((set, get) => ({
    // portfolioSummary: null, - REMOVED
    singleBorrowerResult: null,
    isLoading: false,
    error: null,

    // Scenario Simulator State - REMOVED

    /* 
    fetchPortfolioSummary: async () => {
        ...
    },
    */

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
