import React from 'react';
import { useNavigate } from 'react-router-dom';
import { BarChart3, ArrowLeft } from 'lucide-react';

function SummaryReport() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center">
      <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full mx-4">
        <div className="flex items-center justify-center mb-6">
          <BarChart3 className="w-12 h-12 text-red-700 mr-4" />
          <h1 className="text-3xl font-bold text-gray-800">
            Summary Report
          </h1>
        </div>
        
        <div className="text-center text-gray-600 mb-8">
          <p className="mb-4">This section is under development.</p>
          <p>Future features will include:</p>
          <ul className="list-disc list-inside mt-4 text-left">
            <li>Financial Overview</li>
            <li>Revenue & Expense Charts</li>
            <li>Profit/Loss Statements</li>
            <li>Cash Flow Analysis</li>
            <li>Performance Metrics</li>
          </ul>
        </div>

        <button
          onClick={() => navigate('/bookkeeping')}
          className="w-full bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Bookkeeping
        </button>
      </div>
    </div>
  );
}

export default SummaryReport;
