import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CreditCard, ArrowLeft, DollarSign, Receipt, X, Calendar, User, FileText } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

function PaymentReceiptEntries() {
  const navigate = useNavigate();

  // Receipt Entry Modal State (Main focus as per requirements)
  const [showReceiptModal, setShowReceiptModal] = useState(false);
  const [receiptData, setReceiptData] = useState({
    partyName: '',
    dateReceived: '',
    amountReceived: '',
    discount: '',
    paymentType: '',
    selectedCourses: []
  });

  // Certificate and company data
  const [availableCertificates, setAvailableCertificates] = useState([]);
  const [availableCompanies, setAvailableCompanies] = useState([]);
  const [rateData, setRateData] = useState({});
  const [calculatedAmount, setCalculatedAmount] = useState(0);
  const [rateWarning, setRateWarning] = useState('');

  // Payment types dropdown
  const paymentTypes = [
    { value: 'cash', label: 'Cash' },
    { value: 'neft', label: 'NEFT' },
    { value: 'gpay', label: 'GPay' }
  ];

  // Load data on component mount
  useEffect(() => {
    loadCertificateSelections();
    loadRateData();
    loadAvailableCompanies();
  }, []);

  // Load certificate selections from backend
  const loadCertificateSelections = async () => {
    try {
      console.log('[RECEIPT] Loading certificate selections from API...');
      const response = await fetch('http://localhost:5000/get-certificate-selections-for-receipt');

      if (response.ok) {
        const result = await response.json();
        const certificates = result.data || [];
        setAvailableCertificates(certificates);
        console.log('[RECEIPT] Successfully loaded certificate selections:', certificates);
        console.log('[RECEIPT] Number of certificates loaded:', certificates.length);

        // Log each certificate for debugging
        certificates.forEach((cert, index) => {
          console.log(`[RECEIPT] Certificate ${index + 1}:`, {
            id: cert.id,
            name: `${cert.firstName} ${cert.lastName} - ${cert.certificateName}`,
            company: cert.companyName || 'Unprocessed',
            amount: cert.amount
          });
        });
      } else {
        console.warn('[RECEIPT] Failed to load certificate selections. Status:', response.status);
        setAvailableCertificates([]);
      }
    } catch (error) {
      console.error('[RECEIPT] Error loading certificate selections:', error);
      setAvailableCertificates([]);
    }
  };

  // Load rate data from localStorage
  const loadRateData = () => {
    const savedRates = JSON.parse(localStorage.getItem('courseRates') || '{}');
    setRateData(savedRates);
    console.log('[RECEIPT] Loaded rate data:', savedRates);
  };

  // Load available companies from certificate selections
  const loadAvailableCompanies = async () => {
    try {
      const response = await fetch('http://localhost:5000/get-certificate-selections-for-receipt');
      if (response.ok) {
        const result = await response.json();
        const certificates = result.data || [];

        // Extract unique company names from certificates where companyName is not empty
        const companies = [...new Set(
          certificates
            .filter(cert => cert.companyName && cert.companyName.trim() !== '')
            .map(cert => cert.companyName)
        )];

        setAvailableCompanies(companies);
        console.log('[RECEIPT] Available companies from certificates:', companies);
      } else {
        console.warn('[RECEIPT] Failed to load companies from certificates');
        setAvailableCompanies([]);
      }
    } catch (error) {
      console.error('[RECEIPT] Error loading companies from certificates:', error);
      setAvailableCompanies([]);
    }
  };

  // Calculate amount when courses and company are selected
  useEffect(() => {
    calculateTotalAmount();
  }, [receiptData.selectedCourses, receiptData.partyName, rateData]);

  const calculateTotalAmount = () => {
    console.log('[RECEIPT] Calculating total amount...');
    console.log('[RECEIPT] Selected company:', receiptData.partyName);
    console.log('[RECEIPT] Selected courses:', receiptData.selectedCourses);

    if (!receiptData.partyName || receiptData.selectedCourses.length === 0) {
      console.log('[RECEIPT] No company or courses selected, setting amount to 0');
      setCalculatedAmount(0);
      setRateWarning('');
      return;
    }

    let totalAmount = 0;
    let warnings = [];
    const companyRates = rateData[receiptData.partyName] || {};
    console.log('[RECEIPT] Company rates:', companyRates);

    receiptData.selectedCourses.forEach(courseId => {
      const certificate = availableCertificates.find(cert => cert.id === courseId);
      if (certificate) {
        const rate = companyRates[certificate.certificateName] || 0;
        totalAmount += rate;
        console.log(`[RECEIPT] Course: ${certificate.certificateName}, Rate: ₹${rate}`);

        if (rate === 0) {
          warnings.push(`${certificate.certificateName}: Rate not found`);
        }
      } else {
        console.warn(`[RECEIPT] Certificate not found for ID: ${courseId}`);
      }
    });

    console.log('[RECEIPT] Total calculated amount:', totalAmount);
    setCalculatedAmount(totalAmount);
    setReceiptData(prev => ({ ...prev, amountReceived: totalAmount.toString() }));

    if (warnings.length > 0) {
      setRateWarning(`⚠️ ${warnings.join(', ')}`);
      console.log('[RECEIPT] Rate warnings:', warnings);
    } else {
      setRateWarning('');
    }
  };

  // Handle form input changes
  const handleReceiptInputChange = (field, value) => {
    setReceiptData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // Handle course selection
  const handleCourseSelection = (courseId) => {
    const certificate = availableCertificates.find(cert => cert.id === courseId);
    const isCurrentlySelected = receiptData.selectedCourses.includes(courseId);

    console.log(`[RECEIPT] ${isCurrentlySelected ? 'Deselecting' : 'Selecting'} course:`,
      certificate ? `${certificate.firstName} ${certificate.lastName} - ${certificate.certificateName}` : courseId);

    setReceiptData(prev => ({
      ...prev,
      selectedCourses: prev.selectedCourses.includes(courseId)
        ? prev.selectedCourses.filter(id => id !== courseId)
        : [...prev.selectedCourses, courseId]
    }));
  };

  // Reset modal
  const resetReceiptModal = () => {
    setShowReceiptModal(false);
    setReceiptData({
      partyName: '',
      dateReceived: '',
      amountReceived: '',
      discount: '',
      paymentType: '',
      selectedCourses: []
    });
    setCalculatedAmount(0);
    setRateWarning('');
  };

  // Handle receipt submission
  const handleReceiptSubmit = async () => {
    // Validate required fields
    if (!receiptData.partyName || !receiptData.dateReceived || receiptData.selectedCourses.length === 0) {
      alert('Please fill in all required fields and select at least one course.');
      return;
    }

    try {
      // Update certificate selections with company data
      await fetch('http://localhost:5000/update-certificate-company-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          certificateIds: receiptData.selectedCourses,
          companyName: receiptData.partyName,
          rateData: rateData
        }),
      });

      // Create receipt entry with individual candidate course details
      const receiptEntry = {
        id: Date.now(),
        partyName: receiptData.partyName,
        dateReceived: receiptData.dateReceived,
        amountReceived: parseFloat(receiptData.amountReceived) || 0,
        discount: parseFloat(receiptData.discount) || 0,
        finalAmount: (parseFloat(receiptData.amountReceived) || 0) - (parseFloat(receiptData.discount) || 0),
        paymentType: receiptData.paymentType,
        // Store individual candidate course details for detailed ledger display
        candidateCourses: receiptData.selectedCourses.map(courseId => {
          const cert = availableCertificates.find(c => c.id === courseId);
          return cert ? {
            candidateId: cert.id,
            candidateName: `${cert.firstName} ${cert.lastName}`,
            courseName: cert.certificateName,
            firstName: cert.firstName,
            lastName: cert.lastName
          } : {
            candidateId: courseId,
            candidateName: 'Unknown Candidate',
            courseName: 'Unknown Course',
            firstName: 'Unknown',
            lastName: 'Unknown'
          };
        }),
        // Keep legacy format for backward compatibility
        selectedCourses: receiptData.selectedCourses.map(courseId => {
          const cert = availableCertificates.find(c => c.id === courseId);
          return cert ? `${cert.firstName} ${cert.lastName} - ${cert.certificateName}` : 'Unknown Course';
        }),
        timestamp: new Date().toISOString()
      };

      // Save to localStorage and dispatch events for real-time sync
      const existingReceipts = JSON.parse(localStorage.getItem('receiptEntries') || '[]');
      existingReceipts.push(receiptEntry);
      localStorage.setItem('receiptEntries', JSON.stringify(existingReceipts));

      // Dispatch events to notify other components for real-time sync
      window.dispatchEvent(new CustomEvent('receiptDataUpdated', { detail: receiptEntry }));
      window.dispatchEvent(new CustomEvent('receiptEntryAdded', { detail: receiptEntry }));
      window.dispatchEvent(new CustomEvent('dataUpdated', { detail: { type: 'receipt', data: receiptEntry } }));

      console.log('[RECEIPT] Saved receipt entry:', receiptEntry);
      alert('Receipt entry saved successfully!');
      resetReceiptModal();

      // Reload certificate selections to reflect updated company data
      loadCertificateSelections();
    } catch (error) {
      console.error('[RECEIPT] Error saving receipt:', error);
      alert('Error saving receipt. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <CreditCard className="w-8 h-8 text-blue-700 mr-3" />
              <h1 className="text-3xl font-bold text-gray-800">
                Payment/Receipt Entries
              </h1>
            </div>
            <button
              onClick={() => navigate('/bookkeeping')}
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Bookkeeping
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

          {/* Payment Entries Section */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-6">
              <DollarSign className="w-8 h-8 text-green-700 mr-3" />
              <h2 className="text-2xl font-bold text-gray-800">
                Payment Entries
              </h2>
            </div>
            <p className="text-gray-600 mb-4">
              Manage outgoing payments and generate paid invoices
            </p>
            <button className="w-full bg-green-600 hover:bg-green-700 text-white py-3 px-4 rounded-lg transition-colors font-semibold">
              Create Payment Entry
            </button>
          </div>

          {/* Receipt Entry Section */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center mb-6">
              <Receipt className="w-8 h-8 text-blue-700 mr-3" />
              <h2 className="text-2xl font-bold text-gray-800">
                Receipt Entry
              </h2>
            </div>
            <p className="text-gray-600 mb-4">
              Process incoming receipts from certificate selections
            </p>
            <button
              onClick={() => setShowReceiptModal(true)}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg transition-colors font-semibold"
            >
              Create Receipt Entry
            </button>
          </div>

        </div>
      </div>

      {/* Receipt Entry Modal */}
      <AnimatePresence>
        {showReceiptModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
            onClick={(e) => e.target === e.currentTarget && resetReceiptModal()}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-y-auto"
            >
              {/* Modal Header */}
              <div className="flex items-center justify-between p-6 border-b border-gray-200">
                <div className="flex items-center">
                  <Receipt className="w-6 h-6 text-blue-600 mr-3" />
                  <h2 className="text-2xl font-bold text-gray-800">Receipt Entry</h2>
                </div>
                <button
                  onClick={resetReceiptModal}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              {/* Modal Content */}
              <div className="p-6">
                {/* Receipt Form */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                  {/* Party Name (Company Selection) */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      <User className="w-4 h-4 inline mr-1" />
                      Party Name (Company) *
                    </label>
                    <select
                      value={receiptData.partyName}
                      onChange={(e) => handleReceiptInputChange('partyName', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="">Select Company</option>
                      {availableCompanies.map(company => (
                        <option key={company} value={company}>
                          {company}
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Date Received */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      <Calendar className="w-4 h-4 inline mr-1" />
                      Date Received *
                    </label>
                    <input
                      type="date"
                      value={receiptData.dateReceived}
                      onChange={(e) => handleReceiptInputChange('dateReceived', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                  </div>

                  {/* Amount Received */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      <DollarSign className="w-4 h-4 inline mr-1" />
                      Amount Received (Main Amount)
                    </label>
                    <input
                      type="number"
                      value={receiptData.amountReceived}
                      onChange={(e) => handleReceiptInputChange('amountReceived', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Auto-calculated from rate list"
                      readOnly
                    />
                    {rateWarning && (
                      <p className="text-sm text-orange-600 mt-1">{rateWarning}</p>
                    )}
                  </div>

                  {/* Discount Amount */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      <DollarSign className="w-4 h-4 inline mr-1" />
                      Discount Amount
                    </label>
                    <input
                      type="number"
                      value={receiptData.discount}
                      onChange={(e) => handleReceiptInputChange('discount', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      placeholder="Enter discount amount"
                      min="0"
                      max={receiptData.amountReceived}
                    />
                  </div>

                  {/* Final Amount After Discount */}
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      <DollarSign className="w-4 h-4 inline mr-1" />
                      Final Amount After Discount
                    </label>
                    <div className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-green-50 text-green-800 font-bold text-lg">
                      ₹{((parseFloat(receiptData.amountReceived) || 0) - (parseFloat(receiptData.discount) || 0)).toLocaleString('en-IN')}
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      Main Amount (₹{(parseFloat(receiptData.amountReceived) || 0).toLocaleString('en-IN')}) - Discount (₹{(parseFloat(receiptData.discount) || 0).toLocaleString('en-IN')})
                    </p>
                  </div>



                  {/* Payment Type */}
                  <div className="md:col-span-2">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      <FileText className="w-4 h-4 inline mr-1" />
                      Payment Type *
                    </label>
                    <select
                      value={receiptData.paymentType}
                      onChange={(e) => handleReceiptInputChange('paymentType', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="">Select Payment Type</option>
                      {paymentTypes.map(type => (
                        <option key={type.value} value={type.value}>
                          {type.label}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Course Selection */}
                <div className="mb-8">
                  <h3 className="text-lg font-bold text-gray-800 mb-4">Select Courses *</h3>
                  {availableCertificates.length === 0 ? (
                    <div className="text-center py-8 text-gray-500">
                      <FileText className="w-16 h-16 mx-auto mb-4 opacity-50" />
                      <p className="text-lg font-medium">No certificates available</p>
                      <p className="text-sm">Generate certificates first to see them here</p>
                    </div>
                  ) : (
                    <div className="space-y-3 max-h-64 overflow-y-auto">
                      {availableCertificates.map((certificate) => (
                        <div
                          key={certificate.id}
                          className={`flex items-center justify-between p-4 border rounded-lg transition-colors ${
                            receiptData.selectedCourses.includes(certificate.id)
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:border-gray-300'
                          }`}
                        >
                          <div className="flex items-center">
                            <input
                              type="checkbox"
                              checked={receiptData.selectedCourses.includes(certificate.id)}
                              onChange={() => handleCourseSelection(certificate.id)}
                              className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
                            />
                            <div className="ml-3">
                              <h4 className="font-semibold text-gray-800">
                                {certificate.firstName} {certificate.lastName} - {certificate.certificateName}
                              </h4>
                              <p className="text-sm text-gray-600">
                                Generated: {new Date(certificate.timestamp).toLocaleDateString()}
                                {!certificate.companyName || certificate.companyName === '' ? (
                                  <span className="ml-2 px-2 py-1 bg-green-100 text-green-800 text-xs rounded">
                                    Unprocessed
                                  </span>
                                ) : (
                                  <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                                    {certificate.companyName}
                                  </span>
                                )}
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="font-bold text-blue-600">
                              ₹{receiptData.partyName && rateData[receiptData.partyName] && rateData[receiptData.partyName][certificate.certificateName]
                                ? rateData[receiptData.partyName][certificate.certificateName].toLocaleString()
                                : '0'}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Total Amount Display */}
                {calculatedAmount > 0 && (
                  <div className="mb-8 p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex justify-between items-center">
                      <span className="text-lg font-medium text-green-800">Total Amount:</span>
                      <span className="text-2xl font-bold text-green-600">₹{calculatedAmount.toLocaleString()}</span>
                    </div>
                  </div>
                )}

                {/* Submit Button */}
                <div className="flex gap-4 pt-4 border-t border-gray-200">
                  <button
                    onClick={resetReceiptModal}
                    className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 py-3 px-4 rounded-lg transition-colors font-semibold"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleReceiptSubmit}
                    className="flex-1 bg-blue-600 hover:bg-blue-700 text-white py-3 px-4 rounded-lg transition-colors font-semibold"
                  >
                    Save Receipt Entry
                  </button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default PaymentReceiptEntries;
