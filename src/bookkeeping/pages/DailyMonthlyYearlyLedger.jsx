import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Calendar, ArrowLeft } from 'lucide-react';

function DailyMonthlyYearlyLedger() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center">
      <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full mx-4">
        <div className="flex items-center justify-center mb-6">
          <Calendar className="w-12 h-12 text-orange-700 mr-4" />
          <h1 className="text-3xl font-bold text-gray-800 text-center">
            Daily/Monthly/Yearly Ledger
          </h1>
        </div>
        
        <div className="text-center text-gray-600 mb-8">
          <p className="mb-4">This section is under development.</p>
          <p>Future features will include:</p>
          <ul className="list-disc list-inside mt-4 text-left">
            <li>Daily Transaction Reports</li>
            <li>Monthly Financial Summaries</li>
            <li>Yearly Account Statements</li>
            <li>Period Comparisons</li>
            <li>Financial Trends Analysis</li>
          </ul>
        </div>

        <button
          onClick={() => navigate('/bookkeeping')}
          className="w-full bg-orange-600 hover:bg-orange-700 text-white px-6 py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Bookkeeping
        </button>
      </div>
    </div>
  );
}

export default DailyMonthlyYearlyLedger;
