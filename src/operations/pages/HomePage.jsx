import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Briefcase, BookOpen, Database, Ship, Award } from 'lucide-react';
import { motion } from 'framer-motion';

function HomePage() {
  const navigate = useNavigate();

  const containerVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        when: "beforeChildren",
        staggerChildren: 0.2,
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
        duration: 0.6,
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
          <motion.div className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-8 shadow-2xl">
            <Ship className="w-12 h-12 text-white" />
          </motion.div>
          <motion.h1 className="text-6xl md:text-7xl font-bold bg-gradient-to-r from-white via-blue-100 to-purple-100 bg-clip-text text-transparent mb-4">
            MARITIME
          </motion.h1>
          <motion.h2 className="text-3xl md:text-4xl font-semibold text-blue-200 mb-4">
            TRAINING INSTITUTE
          </motion.h2>
          <motion.p className="text-xl text-blue-100 opacity-80 max-w-3xl mx-auto mb-2">
            Comprehensive Certificate Management System
          </motion.p>
          <motion.div className="flex items-center justify-center gap-2 text-blue-200 opacity-60">
            <Award className="w-5 h-5" />
            <span className="text-lg font-medium">VALUE ADDED SERVICES</span>
            <Award className="w-5 h-5" />
          </motion.div>
        </motion.div>

        {/* Enhanced Main Sections Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-6xl w-full">

          {/* Operations Section */}
          <motion.div
            variants={cardVariants}
            className="group relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
            <motion.button
              onClick={() => navigate('/upload-docx')}
              whileHover={{ scale: 1.02, y: -8 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-10 text-center hover:shadow-2xl transition-all duration-300 border border-white/20"
            >
              <div className="flex flex-col items-center justify-center gap-6 h-full">
                <div className="w-20 h-20 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center shadow-lg">
                  <Briefcase className="w-10 h-10 text-white" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-2">
                    OPERATIONS
                  </h3>
                  <p className="text-blue-600 font-semibold text-lg mb-3">
                    MANAGEMENT
                  </p>
                  <p className="text-sm text-gray-600 opacity-80 leading-relaxed">
                    Document processing, candidate management, and certificate generation
                  </p>
                </div>
              </div>
            </motion.button>
          </motion.div>

          {/* Bookkeeping Section */}
          <motion.div
            variants={cardVariants}
            className="group relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
            <motion.button
              onClick={() => navigate('/bookkeeping')}
              whileHover={{ scale: 1.02, y: -8 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-10 text-center hover:shadow-2xl transition-all duration-300 border border-white/20"
            >
              <div className="flex flex-col items-center justify-center gap-6 h-full">
                <div className="w-20 h-20 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full flex items-center justify-center shadow-lg">
                  <BookOpen className="w-10 h-10 text-white" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-2">
                    BOOKKEEPING
                  </h3>
                  <p className="text-emerald-600 font-semibold text-lg mb-3">
                    MANAGEMENT
                  </p>
                  <p className="text-sm text-gray-600 opacity-80 leading-relaxed">
                    Financial tracking, invoicing, and comprehensive reporting
                  </p>
                </div>
              </div>
            </motion.button>
          </motion.div>

          {/* Database Section */}
          <motion.div
            variants={cardVariants}
            className="group relative"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
            <motion.button
              onClick={() => navigate('/database')}
              whileHover={{ scale: 1.02, y: -8 }}
              whileTap={{ scale: 0.98 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
              className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-10 text-center hover:shadow-2xl transition-all duration-300 border border-white/20"
            >
              <div className="flex flex-col items-center justify-center gap-6 h-full">
                <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center shadow-lg">
                  <Database className="w-10 h-10 text-white" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-800 mb-2">
                    DATABASE
                  </h3>
                  <p className="text-purple-600 font-semibold text-lg mb-3">
                    MANAGEMENT
                  </p>
                  <p className="text-sm text-gray-600 opacity-80 leading-relaxed">
                    Data analytics, search capabilities, and system administration
                  </p>
                </div>
              </div>
            </motion.button>
          </motion.div>
        </div>

        {/* Footer Info */}
        <motion.div
          className="mt-16 text-center"
          variants={cardVariants}
        >
          <p className="text-blue-200 opacity-60 text-sm">
            Professional Maritime Training • Certificate Management • Quality Assurance
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}

export default HomePage;
