import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { ArrowLeft, Printer, Save, Trash2, X } from 'lucide-react';

function SelectedCourses() {
  const navigate = useNavigate();
  const location = useLocation();
  const [selectedCourses, setSelectedCourses] = useState([]);
  const [paymentData, setPaymentData] = useState({});

  useEffect(() => {
    // Get selected courses and payment data from navigation state
    if (location.state) {
      setSelectedCourses(location.state.selectedCourses || []);
      setPaymentData(location.state.paymentData || {});
    }
  }, [location.state]);

  const calculateSubtotal = () => {
    return selectedCourses.reduce((total, course) => total + course.amount, 0);
  };

  const calculateCGST = () => {
    return calculateSubtotal() * 0.09; // 9% CGST
  };

  const calculateSGST = () => {
    return calculateSubtotal() * 0.09; // 9% SGST
  };

  const calculateTotal = () => {
    return calculateSubtotal() + calculateCGST() + calculateSGST();
  };

  const getCurrentDate = () => {
    return new Date().toLocaleDateString('en-GB');
  };

  const getDocumentNumber = () => {
    return `DOC-${Date.now().toString().slice(-6)}`;
  };

  return (
    <div className="min-h-screen bg-gray-50 p-4">
      {/* Back Button */}
      <div className="mb-4">
        <button
          onClick={() => navigate('/bookkeeping/payment-receipt')}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded transition-colors flex items-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Payment Receipt
        </button>
      </div>

      {/* Voucher Document */}
      <div className="max-w-4xl mx-auto bg-white shadow-lg">
        {/* Header Section */}
        <div className="border-b-2 border-blue-600">
          <div className="flex justify-between items-start p-4">
            <div>
              <div className="bg-blue-600 text-white px-3 py-1 text-sm font-medium inline-block">
                Accounting Voucher Alteration (Secondary)
              </div>
              <div className="mt-2">
                <span className="bg-blue-100 text-blue-800 px-2 py-1 text-xs font-medium">
                  No
                </span>
                <span className="ml-2 bg-blue-100 text-blue-800 px-2 py-1 text-xs font-medium">
                  PERFORMA INVOICE
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm font-medium text-blue-600">
                ANGEL SEAFARER DOCUMENTATION PVT LTD New
              </div>
              <div className="text-xs text-gray-600 mt-1">
                {getCurrentDate()}
              </div>
              <div className="text-xs text-gray-600">
                Wednesday
              </div>
            </div>
          </div>
        </div>

        {/* Party Information Section */}
        <div className="border-b border-gray-300">
          <div className="p-4">
            <div className="grid grid-cols-2 gap-8">
              <div>
                <div className="mb-2">
                  <span className="text-sm font-medium text-gray-700">Party A/c name</span>
                </div>
                <div className="text-lg font-semibold text-gray-900 bg-gray-50 p-2 border">
                  {paymentData.partyName || 'BIN OFFSHORE SERVICES PRIVATE LIMITED'}
                </div>
                <div className="text-xs text-gray-600 mt-1">
                  Contact Reference: {paymentData.dateOfEntry || '7.15.644.389.57'}
                </div>
              </div>
              <div>
                <div className="mb-2">
                  <span className="text-sm font-medium text-gray-700">Particulars</span>
                </div>
                <div className="text-right">
                  <div className="text-sm font-medium">Rate per</div>
                  <div className="text-sm font-medium">Amount</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Service Charges Section */}
        <div className="border-b border-gray-300">
          <div className="p-4">
            <div className="mb-4">
              <h3 className="text-sm font-bold text-gray-800 uppercase">SERVICE CHARGES</h3>
            </div>

            {selectedCourses.map((course, index) => (
              <div key={index} className="grid grid-cols-2 gap-8 mb-2">
                <div>
                  <div className="text-sm text-gray-900">{course.name}</div>
                  <div className="text-xs text-gray-600">{course.description}</div>
                </div>
                <div className="text-right">
                  <div className="text-sm">${course.amount.toFixed(2)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Tax Section */}
        <div className="border-b border-gray-300">
          <div className="p-4">
            <div className="grid grid-cols-2 gap-8">
              <div>
                <div className="mb-2">
                  <div className="text-sm font-medium">CGST 9%</div>
                  <div className="text-sm font-medium">SGST 9%</div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm">9 %</div>
                <div className="text-sm">9 %</div>
              </div>
              <div></div>
              <div className="text-right">
                <div className="text-sm">${calculateCGST().toFixed(2)}</div>
                <div className="text-sm">${calculateSGST().toFixed(2)}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Narration/Total Section */}
        <div className="border-b border-gray-300">
          <div className="p-4">
            <div className="grid grid-cols-2 gap-8">
              <div>
                <span className="text-sm font-medium text-gray-700">Narration</span>
              </div>
              <div className="text-right">
                <div className="text-lg font-bold">${calculateTotal().toFixed(2)}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons Section */}
        <div className="p-4 bg-gray-50">
          <div className="flex justify-between items-center">
            <div className="flex space-x-2">
              <button className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 text-sm rounded flex items-center gap-1">
                <Printer className="w-4 h-4" />
                Quit
              </button>
              <button className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 text-sm rounded flex items-center gap-1">
                <Save className="w-4 h-4" />
                Accept
              </button>
              <button className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 text-sm rounded flex items-center gap-1">
                <Trash2 className="w-4 h-4" />
                Delete
              </button>
              <button className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 text-sm rounded flex items-center gap-1">
                <X className="w-4 h-4" />
                Cancel Vch
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SelectedCourses;
