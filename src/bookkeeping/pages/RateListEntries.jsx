import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { DollarSign, ArrowLeft, Edit3, Save, X, Plus, Trash2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function RateListEntries() {
  const navigate = useNavigate();

  // Company names - same as companies ledger
  const companyNames = [
    'ABC Maritime Training',
    'XYZ Shipping Company',
    'Marine Safety Institute',
    'Offshore Training Center',
    'Coastal Academy',
    'Deep Sea Training'
  ];

  // Default courses with sample rates - Updated to match certificate names
  const defaultCourses = [
    { id: 1, name: 'Basic Safety Training (STCW)', description: 'STCW Basic Safety Training Course' },
    { id: 2, name: 'MODU Survival Training', description: 'Mobile Offshore Drilling Unit Survival Training' },
    { id: 3, name: 'H2S Training', description: 'Hydrogen Sulfide Safety Training' },
    { id: 4, name: 'BOSIET Training', description: 'Basic Offshore Safety Induction and Emergency Training' }
  ];

  const [rateData, setRateData] = useState({});
  const [editingCell, setEditingCell] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // Load rate data from localStorage on component mount
  useEffect(() => {
    const loadRateData = () => {
      const savedRates = localStorage.getItem('courseRates');
      if (savedRates) {
        setRateData(JSON.parse(savedRates));
      } else {
        // Initialize with default rates
        const initialRates = {};
        companyNames.forEach(company => {
          initialRates[company] = {};
          defaultCourses.forEach(course => {
            // Generate sample rates between 15000-50000
            initialRates[company][course.name] = Math.floor(Math.random() * 35000) + 15000;
          });
        });
        setRateData(initialRates);
        localStorage.setItem('courseRates', JSON.stringify(initialRates));
      }
      setIsLoading(false);
    };

    loadRateData();
  }, []);

  // Save rate data to localStorage
  const saveRateData = (newRateData) => {
    localStorage.setItem('courseRates', JSON.stringify(newRateData));
    // Dispatch event to notify other components
    window.dispatchEvent(new CustomEvent('rateDataUpdated', { detail: newRateData }));
  };

  // Handle cell edit start
  const handleEditStart = (company, course, currentValue) => {
    setEditingCell(`${company}-${course}`);
    setEditValue(currentValue.toString());
  };

  // Handle cell edit save
  const handleEditSave = () => {
    if (editingCell) {
      const [company, course] = editingCell.split('-');
      const newValue = parseFloat(editValue) || 0;

      const newRateData = {
        ...rateData,
        [company]: {
          ...rateData[company],
          [course]: newValue
        }
      };

      setRateData(newRateData);
      saveRateData(newRateData);
      setEditingCell(null);
      setEditValue('');
    }
  };

  // Handle cell edit cancel
  const handleEditCancel = () => {
    setEditingCell(null);
    setEditValue('');
  };

  // Format currency
  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR'
    }).format(amount);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading rate data...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
      {/* Header */}
      <div className="bg-white/10 backdrop-blur-sm border-b border-white/20 p-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/bookkeeping')}
              className="flex items-center gap-2 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-lg transition-colors text-white"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </button>
            <div className="flex items-center gap-3">
              <DollarSign className="w-8 h-8 text-indigo-300" />
              <div>
                <h1 className="text-2xl font-bold text-white">Rate List Management</h1>
                <p className="text-indigo-200 text-sm">Manage course pricing for all companies</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto p-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl overflow-hidden"
        >
          {/* Table Header */}
          <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-6">
            <h2 className="text-xl font-bold text-white mb-2">Course Rate Matrix</h2>
            <p className="text-indigo-100 text-sm">Click on any rate to edit. Changes are saved automatically.</p>
          </div>

          {/* Rate Table */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sticky left-0 bg-gray-50 z-10">
                    Course / Company
                  </th>
                  {companyNames.map((company, index) => (
                    <th key={index} className="px-6 py-4 text-center text-xs font-medium text-gray-500 uppercase tracking-wider min-w-[180px]">
                      {company}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {defaultCourses.map((course, courseIndex) => (
                  <motion.tr
                    key={course.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: courseIndex * 0.05 }}
                    className="hover:bg-gray-50 transition-colors"
                  >
                    <td className="px-6 py-4 sticky left-0 bg-white z-10 border-r border-gray-200">
                      <div>
                        <div className="text-sm font-medium text-gray-900">{course.name}</div>
                        <div className="text-xs text-gray-500">{course.description}</div>
                      </div>
                    </td>
                    {companyNames.map((company, companyIndex) => {
                      const cellKey = `${company}-${course.name}`;
                      const isEditing = editingCell === cellKey;
                      const rate = rateData[company]?.[course.name] || 0;

                      return (
                        <td key={companyIndex} className="px-6 py-4 text-center">
                          {isEditing ? (
                            <div className="flex items-center justify-center gap-2">
                              <input
                                type="number"
                                value={editValue}
                                onChange={(e) => setEditValue(e.target.value)}
                                className="w-24 px-2 py-1 text-sm border border-indigo-300 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
                                autoFocus
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter') handleEditSave();
                                  if (e.key === 'Escape') handleEditCancel();
                                }}
                              />
                              <button
                                onClick={handleEditSave}
                                className="p-1 text-green-600 hover:text-green-800"
                              >
                                <Save className="w-4 h-4" />
                              </button>
                              <button
                                onClick={handleEditCancel}
                                className="p-1 text-red-600 hover:text-red-800"
                              >
                                <X className="w-4 h-4" />
                              </button>
                            </div>
                          ) : (
                            <div
                              onClick={() => handleEditStart(company, course.name, rate)}
                              className="group cursor-pointer flex items-center justify-center gap-2 p-2 rounded hover:bg-indigo-50 transition-colors"
                            >
                              <span className="text-sm font-medium text-gray-900">
                                {formatCurrency(rate)}
                              </span>
                              <Edit3 className="w-3 h-3 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                            </div>
                          )}
                        </td>
                      );
                    })}
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Footer */}
          <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <div className="text-sm text-gray-600">
                <span className="font-medium">{defaultCourses.length}</span> courses × <span className="font-medium">{companyNames.length}</span> companies = <span className="font-medium">{defaultCourses.length * companyNames.length}</span> rate entries
              </div>
              <div className="text-xs text-gray-500">
                All changes are saved automatically • Last updated: {new Date().toLocaleString()}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default RateListEntries;
