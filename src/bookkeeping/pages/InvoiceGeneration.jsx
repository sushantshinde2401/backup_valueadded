import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  FileSpreadsheet,
  Plus,
  Search,
  Download,
  Edit3,
  Copy,
  Eye
} from 'lucide-react';

function InvoiceGeneration() {
  const navigate = useNavigate();

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
              Back to Bookkeeping
            </button>
            <div className="flex items-center gap-3">
              <FileSpreadsheet className="w-8 h-8 text-green-300" />
              <div>
                <h1 className="text-2xl font-bold text-white">Invoice Generation</h1>
                <p className="text-green-200 text-sm">Create and manage invoices</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg transition-colors text-white font-semibold">
              <Plus className="w-5 h-5" />
              New Invoice
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto p-6">
        <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl">
          {/* Dashboard Header */}
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-3">
                <FileSpreadsheet className="w-6 h-6 text-green-600" />
                <h2 className="text-2xl font-bold text-gray-800">Invoice Dashboard</h2>
              </div>
              <div className="flex items-center gap-4">
                {/* Search */}
                <div className="relative">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search invoices..."
                    className="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
                {/* Status Filter */}
                <select className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500">
                  <option value="all">All Status</option>
                  <option value="draft">Draft</option>
                  <option value="final">Final</option>
                </select>
              </div>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <FileSpreadsheet className="w-8 h-8 text-green-600" />
                  <div>
                    <p className="text-sm text-green-600 font-medium">Total Invoices</p>
                    <p className="text-2xl font-bold text-green-800">0</p>
                  </div>
                </div>
              </div>
              <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <Edit3 className="w-8 h-8 text-blue-600" />
                  <div>
                    <p className="text-sm text-blue-600 font-medium">Draft Invoices</p>
                    <p className="text-2xl font-bold text-blue-800">0</p>
                  </div>
                </div>
              </div>
              <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <Copy className="w-8 h-8 text-purple-600" />
                  <div>
                    <p className="text-sm text-purple-600 font-medium">Final Invoices</p>
                    <p className="text-2xl font-bold text-purple-800">0</p>
                  </div>
                </div>
              </div>
              <div className="bg-gradient-to-r from-orange-50 to-orange-100 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <Download className="w-8 h-8 text-orange-600" />
                  <div>
                    <p className="text-sm text-orange-600 font-medium">Total Value</p>
                    <p className="text-2xl font-bold text-orange-800">₹0.00</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Invoice Table Placeholder */}
          <div className="p-6">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Invoice #</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Date Created</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Client Name</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Total Amount</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Status</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {/* Empty state */}
                </tbody>
              </table>
            </div>

            {/* Empty State */}
            <div className="text-center py-12">
              <FileSpreadsheet className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No invoices found</h3>
              <p className="text-gray-600 mb-4">
                Create your first invoice to get started
              </p>
              <button className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors">
                <Plus className="w-4 h-4" />
                Create New Invoice
              </button>
            </div>
          </div>
        </div>

        {/* Placeholder Notice */}
        <div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <Eye className="w-5 h-5 text-blue-600" />
            <div>
              <h4 className="text-sm font-medium text-blue-800">Placeholder Page</h4>
              <p className="text-sm text-blue-600">
                This is a basic invoice generation page. Invoice creation and management functionality will be implemented in future updates.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default InvoiceGeneration;
