import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CreditCard,
  FileText,
  Building2,
  Calendar,
  BarChart3,
  DollarSign,
  ArrowLeft
} from 'lucide-react';
import { motion } from 'framer-motion';

function BookkeepingDashboard() {
  const navigate = useNavigate();

  const containerVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        when: "beforeChildren",
        staggerChildren: 0.15,
      },
    },
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 30, scale: 0.9 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        duration: 0.5,
        ease: "easeOut"
      }
    },
  };

  const headerVariants = {
    hidden: { opacity: 0, y: -50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        ease: "easeOut"
      }
    },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 relative overflow-hidden">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse animation-delay-2000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-indigo-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-pulse animation-delay-4000"></div>
      </div>

      {/* Grid Pattern Overlay */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>

      {/* Main Content */}
      <motion.div
        className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 py-8"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Header Section */}
        <motion.div
          className="text-center mb-16"
          variants={headerVariants}
        >
          <motion.div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-6 shadow-2xl">
            <BarChart3 className="w-10 h-10 text-white" />
          </motion.div>
          <motion.h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-white via-blue-100 to-purple-100 bg-clip-text text-transparent mb-4">
            BOOKKEEPING
          </motion.h1>
          <motion.h2 className="text-2xl md:text-3xl font-semibold text-blue-200 mb-2">
            MANAGEMENT SYSTEM
          </motion.h2>
          <motion.p className="text-lg text-blue-100 opacity-80 max-w-2xl mx-auto">
            Comprehensive financial management and reporting tools for maritime training institutes
          </motion.p>
        </motion.div>

        {/* Enhanced 6 Buttons Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl w-full">

          {/* Button 1: Payment/Receipt Entries */}
          <motion.div
            variants={cardVariants}
            className="group relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
            <motion.button
              onClick={() => navigate('/bookkeeping/payment-receipt')}
              whileHover={{ scale: 1.02, y: -5 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 text-center hover:shadow-2xl transition-all duration-300 border border-white/20"
            >
              <div className="flex flex-col items-center justify-center gap-4 h-full">
                <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center shadow-lg">
                  <CreditCard className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 leading-tight">
                  PAYMENT/RECEIPT
                </h3>
                <h4 className="text-lg font-semibold text-blue-600">
                  ENTRIES
                </h4>
                <p className="text-sm text-gray-600 opacity-80">
                  Manage payments and receipts
                </p>
              </div>
            </motion.button>
          </motion.div>

          {/* Button 2: Invoice Generation */}
          <motion.div
            variants={cardVariants}
            className="group relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
            <motion.button
              onClick={() => navigate('/bookkeeping/invoice-generation')}
              whileHover={{ scale: 1.02, y: -5 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 text-center hover:shadow-2xl transition-all duration-300 border border-white/20"
            >
              <div className="flex flex-col items-center justify-center gap-4 h-full">
                <div className="w-16 h-16 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full flex items-center justify-center shadow-lg">
                  <FileText className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 leading-tight">
                  INVOICE
                </h3>
                <h4 className="text-lg font-semibold text-emerald-600">
                  GENERATION
                </h4>
                <p className="text-sm text-gray-600 opacity-80">
                  Create and manage invoices
                </p>
              </div>
            </motion.button>
          </motion.div>

          {/* Button 3: Companies Ledger */}
          <motion.div
            variants={cardVariants}
            className="group relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
            <motion.button
              onClick={() => navigate('/bookkeeping/companies-ledger')}
              whileHover={{ scale: 1.02, y: -5 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 text-center hover:shadow-2xl transition-all duration-300 border border-white/20"
            >
              <div className="flex flex-col items-center justify-center gap-4 h-full">
                <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center shadow-lg">
                  <Building2 className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 leading-tight">
                  COMPANIES
                </h3>
                <h4 className="text-lg font-semibold text-purple-600">
                  LEDGER
                </h4>
                <p className="text-sm text-gray-600 opacity-80">
                  Track company accounts
                </p>
              </div>
            </motion.button>
          </motion.div>

          {/* Button 4: Daily/Monthly/Yearly Ledger */}
          <motion.div
            variants={cardVariants}
            className="group relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
            <motion.button
              onClick={() => navigate('/bookkeeping/ledger')}
              whileHover={{ scale: 1.02, y: -5 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 text-center hover:shadow-2xl transition-all duration-300 border border-white/20"
            >
              <div className="flex flex-col items-center justify-center gap-4 h-full">
                <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center shadow-lg">
                  <Calendar className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 leading-tight">
                  PERIODIC
                </h3>
                <h4 className="text-lg font-semibold text-orange-600">
                  LEDGER
                </h4>
                <p className="text-sm text-gray-600 opacity-80">
                  Daily, monthly & yearly reports
                </p>
              </div>
            </motion.button>
          </motion.div>

          {/* Button 5: Summary Report */}
          <motion.div
            variants={cardVariants}
            className="group relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-rose-500 to-pink-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
            <motion.button
              onClick={() => navigate('/bookkeeping/summary-report')}
              whileHover={{ scale: 1.02, y: -5 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 text-center hover:shadow-2xl transition-all duration-300 border border-white/20"
            >
              <div className="flex flex-col items-center justify-center gap-4 h-full">
                <div className="w-16 h-16 bg-gradient-to-r from-rose-500 to-pink-500 rounded-full flex items-center justify-center shadow-lg">
                  <BarChart3 className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 leading-tight">
                  SUMMARY
                </h3>
                <h4 className="text-lg font-semibold text-rose-600">
                  REPORT
                </h4>
                <p className="text-sm text-gray-600 opacity-80">
                  Comprehensive analytics
                </p>
              </div>
            </motion.button>
          </motion.div>

          {/* Button 6: RateList Entries */}
          <motion.div
            variants={cardVariants}
            className="group relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-blue-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
            <motion.button
              onClick={() => navigate('/bookkeeping/ratelist-entries')}
              whileHover={{ scale: 1.02, y: -5 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 text-center hover:shadow-2xl transition-all duration-300 border border-white/20"
            >
              <div className="flex flex-col items-center justify-center gap-4 h-full">
                <div className="w-16 h-16 bg-gradient-to-r from-indigo-500 to-blue-500 rounded-full flex items-center justify-center shadow-lg">
                  <DollarSign className="w-8 h-8 text-white" />
                </div>
                <h3 className="text-xl font-bold text-gray-800 leading-tight">
                  RATELIST
                </h3>
                <h4 className="text-lg font-semibold text-indigo-600">
                  ENTRIES
                </h4>
                <p className="text-sm text-gray-600 opacity-80">
                  Manage pricing & rates
                </p>
              </div>
            </motion.button>
          </motion.div>

        </div>

        {/* Enhanced Back to Home Button */}
        <motion.div
          className="mt-16"
          variants={cardVariants}
        >
          <motion.button
            onClick={() => navigate('/')}
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className="group relative inline-flex items-center gap-3 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white px-8 py-4 rounded-full shadow-2xl font-semibold text-lg transition-all duration-300"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-red-400 to-pink-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
            <ArrowLeft className="w-5 h-5 relative z-10 group-hover:-translate-x-1 transition-transform duration-200" />
            <span className="relative z-10">Back to Home</span>
          </motion.button>
        </motion.div>

        {/* Footer Info */}
        <motion.div
          className="mt-8 text-center"
          variants={cardVariants}
        >
          <p className="text-blue-200 opacity-60 text-sm">
            Maritime Training Institute • Financial Management System
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}

export default BookkeepingDashboard;
