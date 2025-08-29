import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Database,
  Search,
  BarChart3,
  Download,
  Upload,
  CheckCircle,
  ArrowLeft,
  Users,
  Settings,
  Shield
} from 'lucide-react';
import { motion } from 'framer-motion';

function DatabaseDashboard() {
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

  const features = [
    {
      icon: Users,
      title: "Candidate Records",
      description: "Comprehensive candidate data management and tracking",
      color: "from-blue-500 to-cyan-500",
      textColor: "text-blue-600"
    },
    {
      icon: Search,
      title: "Advanced Search",
      description: "Powerful search capabilities with filters and sorting",
      color: "from-emerald-500 to-teal-500",
      textColor: "text-emerald-600"
    },
    {
      icon: BarChart3,
      title: "Reports & Analytics",
      description: "Detailed insights and performance analytics",
      color: "from-purple-500 to-pink-500",
      textColor: "text-purple-600"
    },
    {
      icon: Download,
      title: "Data Export",
      description: "Export data in multiple formats (CSV, PDF, Excel)",
      color: "from-orange-500 to-red-500",
      textColor: "text-orange-600"
    },
    {
      icon: Upload,
      title: "Data Import",
      description: "Bulk import capabilities with validation",
      color: "from-rose-500 to-pink-500",
      textColor: "text-rose-600"
    },
    {
      icon: CheckCircle,
      title: "Data Validation",
      description: "Automated quality checks and integrity verification",
      color: "from-indigo-500 to-blue-500",
      textColor: "text-indigo-600"
    }
  ];

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
          <motion.div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-600 rounded-full mb-6 shadow-2xl">
            <Database className="w-10 h-10 text-white" />
          </motion.div>
          <motion.h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-white via-purple-100 to-pink-100 bg-clip-text text-transparent mb-4">
            DATABASE
          </motion.h1>
          <motion.h2 className="text-2xl md:text-3xl font-semibold text-purple-200 mb-4">
            MANAGEMENT SYSTEM
          </motion.h2>
          <motion.p className="text-lg text-purple-100 opacity-80 max-w-2xl mx-auto mb-6">
            Advanced data management, analytics, and reporting capabilities for maritime training institutes
          </motion.p>

          {/* Development Status Badge */}
          <motion.div className="inline-flex items-center gap-2 bg-yellow-500/20 border border-yellow-500/30 rounded-full px-4 py-2 text-yellow-200">
            <Settings className="w-4 h-4" />
            <span className="text-sm font-medium">Currently Under Development</span>
          </motion.div>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl w-full mb-16">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              variants={cardVariants}
              className="group relative"
            >
              <div className={`absolute inset-0 bg-gradient-to-r ${feature.color} rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300`}></div>
              <div className="relative w-full bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 text-center hover:shadow-2xl transition-all duration-300 border border-white/20">
                <div className="flex flex-col items-center justify-center gap-4 h-full">
                  <div className={`w-16 h-16 bg-gradient-to-r ${feature.color} rounded-full flex items-center justify-center shadow-lg`}>
                    <feature.icon className="w-8 h-8 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-800 leading-tight">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-gray-600 opacity-80 leading-relaxed">
                    {feature.description}
                  </p>
                  <div className="mt-2">
                    <span className={`inline-flex items-center gap-1 ${feature.textColor} text-sm font-medium`}>
                      <Shield className="w-4 h-4" />
                      Coming Soon
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Enhanced Back to Home Button */}
        <motion.div
          className="mt-8"
          variants={cardVariants}
        >
          <motion.button
            onClick={() => navigate('/')}
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className="group relative inline-flex items-center gap-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-8 py-4 rounded-full shadow-2xl font-semibold text-lg transition-all duration-300"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
            <ArrowLeft className="w-5 h-5 relative z-10 group-hover:-translate-x-1 transition-transform duration-200" />
            <span className="relative z-10">Back to Home</span>
          </motion.button>
        </motion.div>

        {/* Footer Info */}
        <motion.div
          className="mt-8 text-center"
          variants={cardVariants}
        >
          <p className="text-purple-200 opacity-60 text-sm">
            Maritime Training Institute • Database Management • Advanced Analytics
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}

export default DatabaseDashboard;
