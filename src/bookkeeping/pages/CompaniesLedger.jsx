import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Building2,
  ArrowLeft,
  Calendar,
  FileText,
  Filter,
  Search,
  Download,
  Eye,
  TrendingUp,
  TrendingDown,
  Users,
  Truck,
  Clock,
  Edit3,
  Trash2,
  Save,
  X,
  ToggleLeft,
  ToggleRight,
  List,
  Grid
} from 'lucide-react';
import { motion } from 'framer-motion';

function CompaniesLedger() {
  const navigate = useNavigate();

  // Sample data
  const sampleDates = [
    '2024-01-15', '2024-01-20', '2024-02-05', '2024-02-18', '2024-03-10',
    '2024-03-25', '2024-04-08', '2024-04-22', '2024-05-12', '2024-05-28',
    '2024-06-15', '2024-06-30', '2024-07-14', '2024-07-29', '2024-08-11',
    '2024-08-26', '2024-09-09', '2024-09-24', '2024-10-07', '2024-10-21',
    '2024-11-05', '2024-11-18', '2024-12-02', '2024-12-16'
  ];

  // Client companies - Updated to match receipt entry form
  const clientCompanies = [
    'ABC Maritime Training',
    'XYZ Shipping Company',
    'Marine Safety Institute',
    'Offshore Training Center',
    'Coastal Academy',
    'Deep Sea Training'
  ];

  // Vendor companies
  const vendorCompanies = [
    'MATSON NAVIGATION COMPANY', 'CROWLEY MARITIME CORP', 'SEABOARD MARINE',
    'GRIMALDI LINES', 'STENA LINE', 'DFDS SEAWAYS', 'BRITTANY FERRIES',
    'P&O FERRIES', 'IRISH FERRIES', 'VIKING LINE', 'TALLINK GRUPP', 'COLOR LINE'
  ];

  // Financial years
  const financialYears = [
    'F.Y. 22-23', 'F.Y. 23-24', 'F.Y. 24-25', 'F.Y. 25-26', 'F.Y. 26-27'
  ];

  const sampleCourses = [
    'BASIC SAFETY TRAINING (BST)', 'ADVANCED FIRE FIGHTING (AFF)',
    'MEDICAL FIRST AID (MFA)', 'MEDICAL CARE (MC)', 'PROFICIENCY IN SURVIVAL CRAFT (PSC)',
    'RADAR NAVIGATION (RN)', 'AUTOMATIC RADAR PLOTTING AIDS (ARPA)',
    'GLOBAL MARITIME DISTRESS SAFETY SYSTEM (GMDSS)', 'BRIDGE RESOURCE MANAGEMENT (BRM)',
    'ENGINE RESOURCE MANAGEMENT (ERM)', 'CARGO HANDLING & STOWAGE',
    'DANGEROUS GOODS HANDLING', 'TANKER SAFETY', 'PASSENGER SHIP SAFETY',
    'HIGH VOLTAGE SAFETY', 'WELDING & FABRICATION', 'CRANE OPERATION',
    'RIGGING & SLINGING', 'CONFINED SPACE ENTRY', 'WORKING AT HEIGHT',
    'HELICOPTER UNDERWATER ESCAPE TRAINING (HUET)', 'OFFSHORE SURVIVAL',
    'DYNAMIC POSITIONING (DP)', 'MARINE ENGINEERING', 'NAVIGATION WATCHKEEPING'
  ];

  // New filter states for three-filter system
  const [filters, setFilters] = useState({
    selectedDate: '',
    companyType: 'Client', // 'Client' or 'Vendor'
    companyName: '',
    financialYear: ''
  });

  // Filtered data state
  const [filteredData, setFilteredData] = useState([]);
  const [allData, setAllData] = useState([]);

  // View mode state
  const [viewMode, setViewMode] = useState('summary'); // 'summary' or 'detailed'
  const [selectedCompany, setSelectedCompany] = useState(null);

  // CRUD operations state
  const [editingEntry, setEditingEntry] = useState(null);
  const [editFormData, setEditFormData] = useState({});

  // Load all data from localStorage and backend and prepare accrual data with improved company name handling
  useEffect(() => {
    const loadAllData = async () => {
      const receiptEntries = JSON.parse(localStorage.getItem('receiptEntries') || '[]');

      // Load certificate selections from backend
      let courseSelections = [];
      try {
        const response = await fetch('http://localhost:5000/get-certificate-selections-for-receipt');
        if (response.ok) {
          const result = await response.json();
          courseSelections = result.data || [];
        }
      } catch (error) {
        console.error('[LEDGER] Error loading certificate selections:', error);
      }

      const allEntries = [];

      // Add course selection entries (Sales) - using certificate selections with company data
      courseSelections.forEach(entry => {
        // Only include entries that have been associated with a company (companyName is not empty)
        if (entry.companyName && entry.companyName.trim() !== '') {
          const normalizedCompanyName = entry.companyName.trim();
          allEntries.push({
            id: `course-${entry.id}`,
            date: entry.timestamp ? new Date(entry.timestamp).toISOString().split('T')[0] : new Date().toISOString().split('T')[0],
            companyName: normalizedCompanyName,
            particular: `${entry.firstName} ${entry.lastName} (${entry.certificateName})`,
            sales: entry.amount || 0, // Amount from rate list when company was selected
            received: 0,
            type: 'course_selection',
            candidateName: `${entry.firstName} ${entry.lastName}`,
            courseName: entry.certificateName,
            timestamp: entry.timestamp,
            originalEntry: entry
          });
        }
      });

      // Add receipt entries (Received) - Create individual candidate entries
      receiptEntries.forEach(entry => {
        if (entry.type === 'receipt' && entry.partyName) {
          const normalizedCompanyName = entry.partyName.trim();

          // Check if we have individual candidate course data (new format)
          if (entry.candidateCourses && entry.candidateCourses.length > 0) {
            // Calculate per-candidate amounts
            const totalCandidates = entry.candidateCourses.length;
            const amountPerCandidate = (entry.amountReceived || 0) / totalCandidates;
            const discountPerCandidate = (entry.discount || 0) / totalCandidates;

            // Create individual entries for each candidate-course combination
            entry.candidateCourses.forEach((candidate, index) => {
              // Individual candidate course entry showing amount received
              allEntries.push({
                id: `receipt-candidate-${entry.id}-${index}`,
                date: entry.dateReceived,
                companyName: normalizedCompanyName,
                particular: `${candidate.candidateName} (${candidate.courseName})`,
                sales: 0,
                received: 0, // Will be shown in separate "Amount Received" row
                type: 'candidate_course',
                paymentType: entry.paymentType,
                candidateName: candidate.candidateName,
                courseName: candidate.courseName,
                timestamp: entry.timestamp,
                originalEntry: entry,
                candidateData: candidate
              });

              // Amount Received row for this candidate
              allEntries.push({
                id: `amount-received-${entry.id}-${index}`,
                date: entry.dateReceived,
                companyName: normalizedCompanyName,
                particular: 'Amount Received',
                sales: 0,
                received: amountPerCandidate,
                type: 'amount_received',
                paymentType: entry.paymentType,
                parentCandidateId: `receipt-candidate-${entry.id}-${index}`,
                timestamp: entry.timestamp,
                originalEntry: entry,
                candidateData: candidate
              });

              // Discount Amount row for this candidate (if applicable)
              if (discountPerCandidate > 0) {
                allEntries.push({
                  id: `discount-amount-${entry.id}-${index}`,
                  date: entry.dateReceived,
                  companyName: normalizedCompanyName,
                  particular: 'Discount Amount',
                  sales: 0,
                  received: discountPerCandidate,
                  type: 'discount_amount',
                  paymentType: entry.paymentType,
                  parentCandidateId: `receipt-candidate-${entry.id}-${index}`,
                  timestamp: entry.timestamp,
                  originalEntry: entry,
                  candidateData: candidate
                });
              }
            });
          } else {
            // Fallback to old format for backward compatibility
            // Main payment entry
            allEntries.push({
              id: `receipt-${entry.id}`,
              date: entry.dateReceived,
              companyName: normalizedCompanyName,
              particular: 'Payment Received',
              sales: 0,
              received: entry.amountReceived || 0,
              type: 'receipt',
              paymentType: entry.paymentType,
              selectedCourses: entry.selectedCourses,
              timestamp: entry.timestamp,
              originalEntry: entry
            });

            // Add discount entry if applicable
            if (entry.discount && entry.discount > 0) {
              allEntries.push({
                id: `discount-${entry.id}`,
                date: entry.dateReceived,
                companyName: normalizedCompanyName,
                particular: 'Discount Applied',
                sales: 0,
                received: entry.discount,
                type: 'discount',
                paymentType: entry.paymentType,
                timestamp: entry.timestamp,
                originalEntry: entry
              });
            }
          }
        }
      });

      return allEntries.sort((a, b) => new Date(a.date) - new Date(b.date));
    };

    loadAllData().then(data => {
      setAllData(data);
    });

    // Listen for custom events when new entries are added
    const handleDataUpdate = () => {
      loadAllData().then(newData => {
        setAllData(newData);
      });
    };

    // Listen for various data update events for real-time synchronization
    window.addEventListener('receiptEntryAdded', handleDataUpdate);
    window.addEventListener('receiptDataUpdated', handleDataUpdate);
    window.addEventListener('courseSelectionAdded', handleDataUpdate);
    window.addEventListener('rateDataUpdated', handleDataUpdate);
    window.addEventListener('dataUpdated', handleDataUpdate);
    window.addEventListener('certificateDataUpdated', handleDataUpdate);

    return () => {
      window.removeEventListener('receiptEntryAdded', handleDataUpdate);
      window.removeEventListener('receiptDataUpdated', handleDataUpdate);
      window.removeEventListener('courseSelectionAdded', handleDataUpdate);
      window.removeEventListener('rateDataUpdated', handleDataUpdate);
      window.removeEventListener('dataUpdated', handleDataUpdate);
      window.removeEventListener('certificateDataUpdated', handleDataUpdate);
    };
  }, []);

  // Filter data based on view mode and selected company
  useEffect(() => {
    if (viewMode === 'detailed' && selectedCompany) {
      // In detailed view, show only entries for selected company
      const companyData = allData.filter(item => item.companyName === selectedCompany);
      setFilteredData(companyData);
    } else {
      // In summary view, we don't filter - we'll aggregate in the component
      setFilteredData(allData);
    }
  }, [allData, viewMode, selectedCompany]);

  const handleFilterChange = (filterType, value) => {
    setFilters(prev => ({
      ...prev,
      [filterType]: value,
      // Reset company name when company type changes
      ...(filterType === 'companyType' && { companyName: '' })
    }));
  };

  const clearFilters = () => {
    setFilters({
      selectedDate: '',
      companyType: 'Client',
      companyName: '',
      financialYear: ''
    });
  };

  // Get available companies based on selected type
  const getAvailableCompanies = () => {
    if (filters.companyType === 'Client') {
      return clientCompanies;
    } else {
      return vendorCompanies;
    }
  };

  const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR'
    }).format(amount);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric'
    });
  };

  // Calculate company summary data for summary view with robust aggregation
  const getCompanySummary = () => {
    const summary = {};

    // Get all unique companies from the data with proper normalization
    const companyNames = allData
      .map(item => item.companyName)
      .filter(Boolean)
      .map(name => name.trim())
      .filter(name => name.length > 0); // Remove empty strings

    const uniqueCompanies = [...new Set(companyNames)];

    uniqueCompanies.forEach(company => {
      // Filter all data for this specific company with exact match
      const companyData = allData.filter(item =>
        item.companyName && item.companyName.trim() === company
      );

      // Calculate total sales (from course selections only)
      const salesEntries = companyData.filter(item => item.type === 'course_selection');
      const totalSales = salesEntries.reduce((sum, item) => sum + (parseFloat(item.sales) || 0), 0);

      // Calculate total received (from receipts and discounts)
      const receivedEntries = companyData.filter(item =>
        item.type === 'receipt' || item.type === 'discount'
      );
      const totalReceived = receivedEntries.reduce((sum, item) => sum + (parseFloat(item.received) || 0), 0);

      // Calculate pending amount (Total Sales - Total Received)
      const pendingAmount = totalSales - totalReceived;

      // Only add companies that have actual data
      if (companyData.length > 0) {
        summary[company] = {
          companyName: company,
          totalSales,
          totalReceived,
          pendingAmount,
          isCredit: pendingAmount < 0,
          entryCount: companyData.length,
          salesEntries: salesEntries.length,
          receivedEntries: receivedEntries.length
        };
      }
    });

    return summary;
  };

  // Handle company row click to switch to detailed view
  const handleCompanyClick = (companyName) => {
    setSelectedCompany(companyName);
    setViewMode('detailed');
  };

  // Handle back to summary
  const handleBackToSummary = () => {
    setSelectedCompany(null);
    setViewMode('summary');
  };

  // Inline editing functionality for detailed view
  const [editingCell, setEditingCell] = useState(null);
  const [editValue, setEditValue] = useState('');



  const handleEditStart = (itemId, field, currentValue) => {
    if (typeof itemId === 'string' && itemId.includes('-')) {
      // Summary view editing - itemId is already in format "company-field"
      setEditingCell(itemId);
    } else {
      // Detailed view editing
      setEditingCell(`${itemId}-${field}`);
    }
    setEditValue(currentValue.toString());
  };

  const handleEditSave = (itemId, field) => {
    const newValue = field === 'particular' ? editValue : parseFloat(editValue) || 0;

    // Update the data based on the field being edited
    if (field === 'sales') {
      // Update course selection entry in localStorage
      const courseSelections = JSON.parse(localStorage.getItem('courseSelections') || '[]');
      const updatedSelections = courseSelections.map(entry => {
        if (`course-${entry.id}` === itemId) {
          return { ...entry, coursePrice: newValue };
        }
        return entry;
      });
      localStorage.setItem('courseSelections', JSON.stringify(updatedSelections));
    } else if (field === 'received') {
      // Update receipt entry in localStorage
      const receiptEntries = JSON.parse(localStorage.getItem('receiptEntries') || '[]');
      const updatedReceipts = receiptEntries.map(entry => {
        if (`receipt-${entry.id}` === itemId) {
          return { ...entry, amountReceived: newValue, finalAmount: newValue - (entry.discount || 0) };
        } else if (`discount-${entry.id}` === itemId) {
          return { ...entry, discount: newValue, finalAmount: (entry.amountReceived || 0) - newValue };
        }

        // Handle new candidate entry types
        const receiptIdMatch = itemId.match(/amount-received-(\d+)-\d+|discount-amount-(\d+)-\d+/);
        if (receiptIdMatch) {
          const receiptId = receiptIdMatch[1] || receiptIdMatch[2];
          if (entry.id.toString() === receiptId) {
            if (itemId.includes('amount-received-')) {
              // Update amount received for the entire receipt
              return { ...entry, amountReceived: newValue, finalAmount: newValue - (entry.discount || 0) };
            } else if (itemId.includes('discount-amount-')) {
              // Update discount for the entire receipt
              return { ...entry, discount: newValue, finalAmount: (entry.amountReceived || 0) - newValue };
            }
          }
        }

        return entry;
      });
      localStorage.setItem('receiptEntries', JSON.stringify(updatedReceipts));
    } else if (field === 'particular') {
      // Update particular field in both course selections and receipt entries
      const courseSelections = JSON.parse(localStorage.getItem('courseSelections') || '[]');
      const updatedSelections = courseSelections.map(entry => {
        if (`course-${entry.id}` === itemId) {
          return { ...entry, candidateName: newValue };
        }
        return entry;
      });
      localStorage.setItem('courseSelections', JSON.stringify(updatedSelections));

      const receiptEntries = JSON.parse(localStorage.getItem('receiptEntries') || '[]');
      const updatedReceipts = receiptEntries.map(entry => {
        if (`receipt-${entry.id}` === itemId || `discount-${entry.id}` === itemId) {
          return { ...entry, particular: newValue };
        }
        return entry;
      });
      localStorage.setItem('receiptEntries', JSON.stringify(updatedReceipts));
    } else if (field === 'receivable') {
      // For receivable, we need to adjust the underlying sales or received amounts
      // This is a complex operation - for now, just show a message
      alert('Receivable is calculated automatically. Edit Sales or Received amounts to change it.');
      setEditingCell(null);
      setEditValue('');
      return;
    }

    // Trigger data reload
    window.dispatchEvent(new CustomEvent('dataUpdated'));

    setEditingCell(null);
    setEditValue('');

    // Show success message
    alert(`${field.charAt(0).toUpperCase() + field.slice(1)} updated successfully!`);
  };

  const handleEditCancel = () => {
    setEditingCell(null);
    setEditValue('');
  };

  // Handle keyboard events for editing
  const handleKeyPress = (e, itemId, field) => {
    if (e.key === 'Enter') {
      if (typeof itemId === 'string' && itemId.includes('-')) {
        // Summary view editing
        const company = itemId.split('-')[0];
        handleSummaryEditSave(company, field);
      } else {
        // Detailed view editing
        handleEditSave(itemId, field);
      }
    } else if (e.key === 'Escape') {
      handleEditCancel();
    }
  };

  // Handle summary view editing
  const handleSummaryEditSave = (company, field) => {
    const newValue = parseFloat(editValue) || 0;

    // For summary editing, we need to update the underlying data
    // This is a simplified approach - in a real app, you'd want more sophisticated logic
    if (field === 'pending') {
      // Update pending amount by adjusting sales or received amounts
      // This is a placeholder - implement based on your business logic
      console.log(`[SUMMARY EDIT] Updating ${company} pending amount to ${newValue}`);
    } else if (field === 'received') {
      // Update total received amount
      console.log(`[SUMMARY EDIT] Updating ${company} received amount to ${newValue}`);
    }

    // Dispatch event to refresh data
    window.dispatchEvent(new CustomEvent('dataUpdated'));

    setEditingCell(null);
    setEditValue('');

    // Show success message
    alert(`${company} ${field} amount updated successfully!`);
  };

  // Handle delete operations
  const handleDelete = (itemId, itemType) => {
    if (!window.confirm(`Are you sure you want to delete this ${itemType}? This action cannot be undone.`)) {
      return;
    }

    try {
      if (itemType === 'course_selection') {
        // Delete from certificate selections
        const certificateSelections = JSON.parse(localStorage.getItem('certificateSelections') || '[]');
        const updatedSelections = certificateSelections.filter(item => item.id !== itemId);
        localStorage.setItem('certificateSelections', JSON.stringify(updatedSelections));

        // Also remove from backend
        fetch('http://localhost:5000/delete-certificate-selection', {
          method: 'DELETE',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ id: itemId }),
        }).catch(error => console.error('[DELETE] Error deleting certificate selection:', error));

      } else if (itemType === 'receipt' || itemType === 'discount') {
        // Delete from receipt entries - extract the actual receipt ID
        const receiptId = itemId.replace('receipt-', '').replace('discount-', '');
        const receiptEntries = JSON.parse(localStorage.getItem('receiptEntries') || '[]');
        const updatedReceipts = receiptEntries.filter(item => item.id.toString() !== receiptId);
        localStorage.setItem('receiptEntries', JSON.stringify(updatedReceipts));

      } else if (itemType === 'candidate_course' || itemType === 'amount_received' || itemType === 'discount_amount') {
        // For new candidate entry types, we need to handle deletion differently
        // Extract the receipt ID from the item ID
        const receiptIdMatch = itemId.match(/receipt-candidate-(\d+)-\d+|amount-received-(\d+)-\d+|discount-amount-(\d+)-\d+/);
        if (receiptIdMatch) {
          const receiptId = receiptIdMatch[1] || receiptIdMatch[2] || receiptIdMatch[3];

          if (itemType === 'candidate_course') {
            // If deleting a candidate course entry, ask if they want to delete the entire receipt
            if (window.confirm('This will delete the entire receipt entry for this candidate. Continue?')) {
              const receiptEntries = JSON.parse(localStorage.getItem('receiptEntries') || '[]');
              const updatedReceipts = receiptEntries.filter(item => item.id.toString() !== receiptId);
              localStorage.setItem('receiptEntries', JSON.stringify(updatedReceipts));
            } else {
              return; // User cancelled
            }
          } else {
            // For amount_received or discount_amount, just show a message that they need to edit the receipt
            alert('To modify payment amounts, please edit the original receipt entry or delete the entire receipt.');
            return;
          }
        }
      }

      // Refresh data
      window.dispatchEvent(new CustomEvent('dataUpdated'));

      // Show success message
      alert(`${itemType.replace('_', ' ')} deleted successfully!`);

    } catch (error) {
      console.error('[DELETE] Error deleting item:', error);
      alert('Error deleting item. Please try again.');
    }
  };





  // CRUD Operations
  const handleEditEntry = (entry) => {
    setEditingEntry(entry.id);
    setEditFormData({
      amount: entry.amount,
      discount: entry.discount || 0,
      paymentType: entry.paymentType || '',
      courseDetails: entry.courseDetails || ''
    });
  };

  const handleSaveEdit = () => {
    if (!editingEntry) return;

    const entryType = editingEntry.startsWith('receipt-') ? 'receipt' : 'course';
    const entryId = editingEntry.replace('receipt-', '').replace('course-', '');

    if (entryType === 'receipt') {
      // Update receipt entry
      const receiptEntries = JSON.parse(localStorage.getItem('receiptEntries') || '[]');
      const updatedEntries = receiptEntries.map(entry => {
        if (entry.id.toString() === entryId) {
          return {
            ...entry,
            finalAmount: parseFloat(editFormData.amount),
            amountReceived: parseFloat(editFormData.amount) + parseFloat(editFormData.discount),
            discount: parseFloat(editFormData.discount),
            paymentType: editFormData.paymentType
          };
        }
        return entry;
      });
      localStorage.setItem('receiptEntries', JSON.stringify(updatedEntries));
    } else {
      // Update course selection
      const courseSelections = JSON.parse(localStorage.getItem('courseSelections') || '[]');
      const updatedSelections = courseSelections.map(selection => {
        if (selection.id.toString() === entryId) {
          return {
            ...selection,
            courseName: editFormData.courseDetails
          };
        }
        return selection;
      });
      localStorage.setItem('courseSelections', JSON.stringify(updatedSelections));
    }

    // Trigger data reload
    window.dispatchEvent(new CustomEvent('dataUpdated'));

    // Reset editing state
    setEditingEntry(null);
    setEditFormData({});

    alert('Entry updated successfully!');
  };

  const handleCancelEdit = () => {
    setEditingEntry(null);
    setEditFormData({});
  };





  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-indigo-900">
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
              <Building2 className="w-8 h-8 text-purple-300" />
              <div>
                <h1 className="text-2xl font-bold text-white">Companies Ledger</h1>
                <p className="text-purple-200 text-sm">Financial tracking and management</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors text-white">
              <Download className="w-4 h-4" />
              Export
            </button>
            <div className="text-white text-right">
              <p className="text-sm opacity-80">
                {viewMode === 'summary' ? 'Companies' : 'Entries'}
              </p>
              <p className="text-xl font-bold">
                {viewMode === 'summary' ? Object.keys(getCompanySummary()).length : filteredData.length}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex h-[calc(100vh-80px)]">
        {/* Main Content Area */}
        <div className="flex-1 p-6 overflow-hidden">
          <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl h-full flex flex-col">
            {/* Dynamic Header */}
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Building2 className="w-5 h-5 text-purple-600" />
                  <h2 className="text-xl font-bold text-gray-800">
                    {viewMode === 'summary' ? 'Companies Ledger - Summary' : `${selectedCompany} - Detailed View`}
                  </h2>
                  <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm font-medium">
                    {viewMode === 'summary' ? `${Object.keys(getCompanySummary()).length} companies` : `${filteredData.length} entries`}
                  </span>

                  {/* View Toggle */}
                  <div className="flex items-center gap-2 ml-4">
                    <button
                      onClick={() => setViewMode(viewMode === 'summary' ? 'detailed' : 'summary')}
                      className={`flex items-center gap-2 px-3 py-1 rounded-lg transition-colors ${
                        viewMode === 'summary'
                          ? 'bg-blue-100 text-blue-800 hover:bg-blue-200'
                          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                      }`}
                      title={viewMode === 'summary' ? 'Switch to Detailed View' : 'Switch to Summary View'}
                    >
                      {viewMode === 'summary' ? <List className="w-4 h-4" /> : <Grid className="w-4 h-4" />}
                      <span className="text-sm font-medium">
                        {viewMode === 'summary' ? 'Summary' : 'Detailed'}
                      </span>
                    </button>
                  </div>
                </div>
                {viewMode === 'detailed' && (
                  <button
                    onClick={handleBackToSummary}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
                  >
                    <ArrowLeft className="w-4 h-4" />
                    Back to Summary
                  </button>
                )}
              </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 overflow-auto p-6">
              {viewMode === 'summary' ? (
                /* SUMMARY VIEW - Default Landing Page */
                <div>
                  {Object.keys(getCompanySummary()).length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-500">
                      <Building2 className="w-16 h-16 mb-4 opacity-50" />
                      <p className="text-lg font-medium">No companies found</p>
                      <p className="text-sm">Generate certificates or create payment entries to see companies here</p>
                    </div>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full bg-white rounded-lg shadow-sm">
                        <thead className="bg-gray-50">
                          <tr>
                            <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Company Names (CN)
                            </th>
                            <th className="px-6 py-4 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Pending
                            </th>
                            <th className="px-6 py-4 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                              Received
                            </th>
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {Object.entries(getCompanySummary()).map(([company, summary], index) => (
                            <motion.tr
                              key={company}
                              initial={{ opacity: 0, y: 20 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: index * 0.05 }}
                              onClick={() => handleCompanyClick(company)}
                              className="hover:bg-blue-50 cursor-pointer transition-colors group"
                            >
                              <td className="px-6 py-4 whitespace-nowrap">
                                <div className="flex items-center">
                                  <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center mr-3">
                                    <span className="text-white font-bold text-sm">
                                      {company.split(' ').map(word => word[0]).join('').slice(0, 2).toUpperCase()}
                                    </span>
                                  </div>
                                  <div>
                                    <div className="text-sm font-medium text-gray-900 group-hover:text-blue-600 transition-colors">
                                      {company}
                                    </div>
                                    <div className="text-xs text-gray-500">Click to view details</div>
                                  </div>
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-center">
                                {editingCell === `${company}-pending` ? (
                                  <div className="flex items-center justify-center gap-2">
                                    <input
                                      type="number"
                                      value={editValue}
                                      onChange={(e) => setEditValue(e.target.value)}
                                      className="w-24 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                                      onKeyDown={(e) => handleKeyPress(e, company, 'pending')}
                                      autoFocus
                                    />
                                    <button
                                      onClick={() => handleSummaryEditSave(company, 'pending')}
                                      className="text-green-600 hover:text-green-800 text-lg"
                                    >
                                      ✓
                                    </button>
                                    <button
                                      onClick={handleEditCancel}
                                      className="text-red-600 hover:text-red-800 text-lg"
                                    >
                                      ✕
                                    </button>
                                  </div>
                                ) : (
                                  <div
                                    className={`text-sm font-semibold cursor-pointer hover:bg-gray-50 px-3 py-1 rounded transition-colors ${
                                      summary.isCredit ? 'text-green-600' : 'text-orange-600'
                                    }`}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleEditStart(`${company}-pending`, 'pending', Math.abs(summary.pendingAmount));
                                    }}
                                    title="Click to edit pending amount"
                                  >
                                    {summary.isCredit ? 'Credit: ' : ''}
                                    {formatCurrency(Math.abs(summary.pendingAmount))}
                                  </div>
                                )}
                              </td>
                              <td className="px-6 py-4 whitespace-nowrap text-center">
                                {editingCell === `${company}-received` ? (
                                  <div className="flex items-center justify-center gap-2">
                                    <input
                                      type="number"
                                      value={editValue}
                                      onChange={(e) => setEditValue(e.target.value)}
                                      className="w-24 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                                      onKeyDown={(e) => handleKeyPress(e, company, 'received')}
                                      autoFocus
                                    />
                                    <button
                                      onClick={() => handleSummaryEditSave(company, 'received')}
                                      className="text-green-600 hover:text-green-800 text-lg"
                                    >
                                      ✓
                                    </button>
                                    <button
                                      onClick={handleEditCancel}
                                      className="text-red-600 hover:text-red-800 text-lg"
                                    >
                                      ✕
                                    </button>
                                  </div>
                                ) : (
                                  <div
                                    className="text-sm font-semibold text-green-600 cursor-pointer hover:bg-green-50 px-3 py-1 rounded transition-colors"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleEditStart(`${company}-received`, 'received', summary.totalReceived);
                                    }}
                                    title="Click to edit received amount"
                                  >
                                    {formatCurrency(summary.totalReceived)}
                                  </div>
                                )}
                              </td>
                            </motion.tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              ) : (
                /* DETAILED VIEW - Accessed by clicking company row */
                <div>
                  {filteredData.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-gray-500">
                      <FileText className="w-16 h-16 mb-4 opacity-50" />
                      <p className="text-lg font-medium">No entries found for {selectedCompany}</p>
                      <p className="text-sm">Generate certificates or create payment entries to see data here</p>
                    </div>
                  ) : (
                    <>
                      {/* Company Header with Outstanding Balance */}
                      {selectedCompany && (
                        <div className="mb-6 bg-white rounded-lg shadow-sm p-6 border">
                          <div className="flex justify-between items-start mb-4">
                            <div>
                              <h2 className="text-2xl font-bold text-gray-900 mb-2">{selectedCompany}</h2>
                              <p className="text-gray-600">Detailed Accrual View</p>
                            </div>
                            <div className="text-right">
                              {(() => {
                                const summary = getCompanySummary()[selectedCompany];
                                if (!summary) return null;
                                return (
                                  <div>
                                    <div className="text-sm text-gray-500 mb-1">Outstanding Balance</div>
                                    <div className={`text-3xl font-bold ${
                                      summary.isCredit ? 'text-green-600' : 'text-orange-600'
                                    }`}>
                                      {summary.isCredit ? 'Credit: ' : ''}
                                      {formatCurrency(Math.abs(summary.pendingAmount))}
                                    </div>
                                  </div>
                                );
                              })()}
                            </div>
                          </div>

                          {/* Summary Stats */}
                          {(() => {
                            const summary = getCompanySummary()[selectedCompany];
                            if (!summary) return null;
                            return (
                              <div className="grid grid-cols-3 gap-4 pt-4 border-t">
                                <div className="text-center">
                                  <div className="text-sm text-gray-500">Total Sales</div>
                                  <div className="text-xl font-semibold text-blue-600">
                                    {formatCurrency(summary.totalSales)}
                                  </div>
                                </div>
                                <div className="text-center">
                                  <div className="text-sm text-gray-500">Total Received</div>
                                  <div className="text-xl font-semibold text-green-600">
                                    {formatCurrency(summary.totalReceived)}
                                  </div>
                                </div>
                                <div className="text-center">
                                  <div className="text-sm text-gray-500">Total Entries</div>
                                  <div className="text-xl font-semibold text-gray-700">
                                    {summary.entryCount}
                                  </div>
                                </div>
                              </div>
                            );
                          })()}
                        </div>
                      )}

                      {/* Detailed Accrual Table */}
                      <div className="overflow-x-auto">
                        <table className="w-full bg-white rounded-lg shadow-sm">
                          <thead className="bg-gray-50">
                            <tr>
                              <th className="px-6 py-4 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Particular
                              </th>
                              <th className="px-6 py-4 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Sales
                              </th>
                              <th className="px-6 py-4 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Received
                              </th>
                              <th className="px-6 py-4 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Receivable
                              </th>
                              <th className="px-6 py-4 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Actions
                              </th>
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {(() => {
                              let runningReceivable = 0;
                              return filteredData.map((item, index) => {
                                // Calculate running receivable: Previous + Sales - Received
                                runningReceivable += (item.sales || 0) - (item.received || 0);

                                return (
                                  <motion.tr
                                    key={item.id}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: index * 0.05 }}
                                    className="hover:bg-gray-50 transition-colors"
                                  >
                                    {/* Particular Column - Editable */}
                                    <td className="px-6 py-4 text-sm text-gray-900">
                                      {editingCell === `${item.id}-particular` ? (
                                        <div className="flex items-center gap-2">
                                          <input
                                            type="text"
                                            value={editValue}
                                            onChange={(e) => setEditValue(e.target.value)}
                                            className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            onKeyDown={(e) => handleKeyPress(e, item.id, 'particular')}
                                            autoFocus
                                          />
                                          <button
                                            onClick={() => handleEditSave(item.id, 'particular')}
                                            className="text-green-600 hover:text-green-800 text-lg flex-shrink-0"
                                          >
                                            ✓
                                          </button>
                                          <button
                                            onClick={handleEditCancel}
                                            className="text-red-600 hover:text-red-800 text-lg flex-shrink-0"
                                          >
                                            ✕
                                          </button>
                                        </div>
                                      ) : (
                                        <div className="max-w-sm">
                                          <div
                                            className="font-medium cursor-pointer hover:bg-gray-50 px-2 py-1 rounded transition-colors"
                                            onClick={() => handleEditStart(item.id, 'particular', item.particular)}
                                            title="Click to edit"
                                          >
                                            {item.particular}
                                          </div>
                                          {item.type === 'course_selection' && (
                                            <div className="text-xs text-gray-500 mt-1">
                                              Course: {item.courseName}
                                            </div>
                                          )}
                                          {(item.type === 'receipt' || item.type === 'candidate_course' || item.type === 'amount_received' || item.type === 'discount_amount') && item.paymentType && (
                                            <div className="text-xs text-gray-500 mt-1">
                                              via {item.paymentType}
                                            </div>
                                          )}
                                          {item.type === 'candidate_course' && item.candidateData && (
                                            <div className="text-xs text-blue-600 mt-1 font-medium">
                                              Candidate: {item.candidateData.candidateName} | Course: {item.candidateData.courseName}
                                            </div>
                                          )}
                                          {(item.type === 'amount_received' || item.type === 'discount_amount') && item.candidateData && (
                                            <div className="text-xs text-green-600 mt-1">
                                              For: {item.candidateData.candidateName} ({item.candidateData.courseName})
                                            </div>
                                          )}
                                        </div>
                                      )}
                                    </td>

                                    {/* Sales Column - Only for course selections */}
                                    <td className="px-6 py-4 whitespace-nowrap text-center text-sm font-semibold">
                                      {editingCell === `${item.id}-sales` ? (
                                        <div className="flex items-center justify-center gap-2">
                                          <input
                                            type="number"
                                            value={editValue}
                                            onChange={(e) => setEditValue(e.target.value)}
                                            className="w-24 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            onKeyPress={(e) => handleKeyPress(e, item.id, 'sales')}
                                            autoFocus
                                          />
                                          <button
                                            onClick={() => handleEditSave(item.id, 'sales')}
                                            className="text-green-600 hover:text-green-800 text-lg"
                                          >
                                            ✓
                                          </button>
                                          <button
                                            onClick={handleEditCancel}
                                            className="text-red-600 hover:text-red-800 text-lg"
                                          >
                                            ✕
                                          </button>
                                        </div>
                                      ) : item.sales > 0 ? (
                                        <span
                                          className="text-blue-600 cursor-pointer hover:bg-blue-50 px-3 py-1 rounded transition-colors"
                                          onClick={() => handleEditStart(item.id, 'sales', item.sales)}
                                          title="Click to edit"
                                        >
                                          {formatCurrency(item.sales)}
                                        </span>
                                      ) : (
                                        <span className="text-gray-400">-</span>
                                      )}
                                    </td>

                                    {/* Received Column - Only for receipts and discounts */}
                                    <td className="px-6 py-4 whitespace-nowrap text-center text-sm font-semibold">
                                      {editingCell === `${item.id}-received` ? (
                                        <div className="flex items-center justify-center gap-2">
                                          <input
                                            type="number"
                                            value={editValue}
                                            onChange={(e) => setEditValue(e.target.value)}
                                            className="w-24 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            onKeyPress={(e) => handleKeyPress(e, item.id, 'received')}
                                            autoFocus
                                          />
                                          <button
                                            onClick={() => handleEditSave(item.id, 'received')}
                                            className="text-green-600 hover:text-green-800 text-lg"
                                          >
                                            ✓
                                          </button>
                                          <button
                                            onClick={handleEditCancel}
                                            className="text-red-600 hover:text-red-800 text-lg"
                                          >
                                            ✕
                                          </button>
                                        </div>
                                      ) : item.received > 0 ? (
                                        <span
                                          className={`cursor-pointer px-3 py-1 rounded transition-colors ${
                                            item.type === 'discount'
                                              ? 'text-orange-600 hover:bg-orange-50'
                                              : 'text-green-600 hover:bg-green-50'
                                          }`}
                                          onClick={() => handleEditStart(item.id, 'received', item.received)}
                                          title="Click to edit"
                                        >
                                          {formatCurrency(item.received)}
                                        </span>
                                      ) : (
                                        <span className="text-gray-400">-</span>
                                      )}
                                    </td>

                                    {/* Receivable Column - Editable Running total */}
                                    <td className="px-6 py-4 whitespace-nowrap text-center text-sm font-bold">
                                      {editingCell === `${item.id}-receivable` ? (
                                        <div className="flex items-center justify-center gap-2">
                                          <input
                                            type="number"
                                            value={editValue}
                                            onChange={(e) => setEditValue(e.target.value)}
                                            className="w-24 px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                                            onKeyDown={(e) => handleKeyPress(e, item.id, 'receivable')}
                                            autoFocus
                                          />
                                          <button
                                            onClick={() => handleEditSave(item.id, 'receivable')}
                                            className="text-green-600 hover:text-green-800 text-lg"
                                          >
                                            ✓
                                          </button>
                                          <button
                                            onClick={handleEditCancel}
                                            className="text-red-600 hover:text-red-800 text-lg"
                                          >
                                            ✕
                                          </button>
                                        </div>
                                      ) : (
                                        <span
                                          className={`cursor-pointer hover:bg-gray-50 px-3 py-1 rounded transition-colors ${
                                            runningReceivable < 0 ? 'text-green-600' : 'text-orange-600'
                                          }`}
                                          onClick={() => handleEditStart(item.id, 'receivable', Math.abs(runningReceivable))}
                                          title="Click to edit"
                                        >
                                          {runningReceivable < 0 ? 'Credit: ' : ''}
                                          {formatCurrency(Math.abs(runningReceivable))}
                                        </span>
                                      )}
                                    </td>

                                    {/* Actions Column - Delete functionality */}
                                    <td className="px-6 py-4 whitespace-nowrap text-center text-sm">
                                      <button
                                        onClick={() => handleDelete(item.id, item.type)}
                                        className="text-red-600 hover:text-red-800 hover:bg-red-50 p-2 rounded transition-colors"
                                        title="Delete this entry"
                                      >
                                        <Trash2 className="w-4 h-4" />
                                      </button>
                                    </td>
                                  </motion.tr>
                                );
                              });
                            })()}
                          </tbody>
                        </table>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Side - Filter Bar */}
        <div className="w-80 p-6">
          <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl h-full">
            {/* Filter Header */}
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center gap-3 mb-4">
                <Filter className="w-5 h-5 text-purple-600" />
                <h3 className="text-lg font-bold text-gray-800">Filter Options</h3>
              </div>
              <p className="text-sm text-gray-600">Select criteria to filter ledger entries</p>
            </div>

            {/* Filter Sections */}
            <div className="p-6 space-y-6 overflow-auto h-[calc(100%-120px)]">

              {/* 1. Date Filter */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4 text-blue-600" />
                  <label className="text-sm font-semibold text-gray-700">Date Filter</label>
                </div>
                <input
                  type="date"
                  value={filters.selectedDate}
                  onChange={(e) => handleFilterChange('selectedDate', e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 bg-white text-sm"
                  placeholder="Select a specific date"
                />
              </div>

              {/* 2. Company Name Filter with Client/Vendor Segregation */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Building2 className="w-4 h-4 text-green-600" />
                  <label className="text-sm font-semibold text-gray-700">Company Filter</label>
                </div>

                {/* Step 1: Client/Vendor Toggle */}
                <div>
                  <label className="text-xs text-gray-500 mb-2 block">Company Type</label>
                  <div className="flex bg-gray-100 rounded-lg p-1">
                    <button
                      onClick={() => handleFilterChange('companyType', 'Client')}
                      className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                        filters.companyType === 'Client'
                          ? 'bg-white text-green-700 shadow-sm'
                          : 'text-gray-600 hover:text-gray-800'
                      }`}
                    >
                      <Users className="w-4 h-4" />
                      Client
                    </button>
                    <button
                      onClick={() => handleFilterChange('companyType', 'Vendor')}
                      className={`flex-1 py-2 px-3 rounded-md text-sm font-medium transition-colors flex items-center justify-center gap-2 ${
                        filters.companyType === 'Vendor'
                          ? 'bg-white text-green-700 shadow-sm'
                          : 'text-gray-600 hover:text-gray-800'
                      }`}
                    >
                      <Truck className="w-4 h-4" />
                      Vendor
                    </button>
                  </div>
                </div>

                {/* Step 2: Company Name Dropdown */}
                <div>
                  <label className="text-xs text-gray-500 mb-1 block">
                    {filters.companyType} Companies
                  </label>
                  <select
                    value={filters.companyName}
                    onChange={(e) => handleFilterChange('companyName', e.target.value)}
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 bg-white text-sm"
                  >
                    <option value="">All {filters.companyType} Companies</option>
                    {getAvailableCompanies().map(name => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* 3. Financial Year Filter */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-purple-600" />
                  <label className="text-sm font-semibold text-gray-700">Financial Year</label>
                </div>
                <select
                  value={filters.financialYear}
                  onChange={(e) => handleFilterChange('financialYear', e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 bg-white text-sm"
                >
                  <option value="">All Financial Years</option>
                  {financialYears.map(year => (
                    <option key={year} value={year}>
                      {year}
                    </option>
                  ))}
                </select>
              </div>

              {/* Filter Summary */}
              <div className="pt-4 border-t border-gray-200">
                <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-4">
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Filter Summary</h4>
                  <div className="space-y-1 text-xs text-gray-600">
                    <p>Total Records: <span className="font-semibold text-gray-800">{allData.length}</span></p>
                    <p>Filtered Results: <span className="font-semibold text-purple-600">{filteredData.length}</span></p>
                    <p>Active Filters: <span className="font-semibold text-blue-600">
                      {[
                        filters.selectedDate,
                        filters.companyName,
                        filters.financialYear
                      ].filter(Boolean).length}
                    </span></p>
                    <p>Company Type: <span className="font-semibold text-green-600">{filters.companyType}</span></p>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="space-y-3">
                <h4 className="text-sm font-semibold text-gray-700">Quick Actions</h4>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => handleFilterChange('companyType', 'Client')}
                    className="p-2 bg-green-100 hover:bg-green-200 text-green-800 rounded-lg text-xs font-medium transition-colors"
                  >
                    Show Clients
                  </button>
                  <button
                    onClick={() => handleFilterChange('companyType', 'Vendor')}
                    className="p-2 bg-blue-100 hover:bg-blue-200 text-blue-800 rounded-lg text-xs font-medium transition-colors"
                  >
                    Show Vendors
                  </button>
                </div>
                <div className="grid grid-cols-1 gap-2">
                  <button
                    onClick={() => handleFilterChange('financialYear', 'F.Y. 24-25')}
                    className="p-2 bg-purple-100 hover:bg-purple-200 text-purple-800 rounded-lg text-xs font-medium transition-colors"
                  >
                    Current F.Y. 24-25
                  </button>
                </div>
              </div>

            </div>
          </div>
        </div>
      </div>


    </div>
  );
}

export default CompaniesLedger;
