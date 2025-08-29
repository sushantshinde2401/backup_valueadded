import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import certificateMap from "../../shared/utils/certificateMap";
import { ArrowLeft, Eye, BookOpen, CheckCircle, XCircle, Cloud, LogOut, GraduationCap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

function CoursePreview() {
  const [courses, setCourses] = useState([]);
  const [openMenus, setOpenMenus] = useState({});
  const [statusFlags, setStatusFlags] = useState({});
  const navigate = useNavigate();

  // Load courses and status
  useEffect(() => {
    const savedCourses = localStorage.getItem("courses");
    if (savedCourses) {
      const parsedCourses = JSON.parse(savedCourses);
      setCourses(parsedCourses);

      const flags = {};
      parsedCourses.forEach((course) => {
        const key = `status_${course}`;
        let visited = localStorage.getItem(key);
        if (visited === null) {
          localStorage.setItem(key, "false"); // first time = not visited
          visited = "false";
        }
        flags[course] = visited === "true";
      });
      setStatusFlags(flags);
    }
  }, []);

  // Navigate and set status visited
  const handleCourseClick = (course) => {
    localStorage.setItem(`status_${course}`, "true");
    // Store selected course information for certificate generation
    localStorage.setItem("selectedCourse", course);
    localStorage.setItem("selectedCourseTimestamp", new Date().toISOString());
    console.log(`[COURSE] Selected course: ${course} for certificate generation`);
    navigate(certificateMap[course]);
  };

  // Toggle menu per card
  const toggleMenu = (course) => {
    setOpenMenus((prev) => ({
      ...prev,
      [course]: !prev[course],
    }));
  };

  // Reset all status flags
  const handleEditSelection = () => {
    courses.forEach((course) => {
      localStorage.removeItem(`status_${course}`);
    });
    navigate("/course-selection");
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.6,
        when: "beforeChildren",
        staggerChildren: 0.2,
      },
    },
  };

  const cardVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0 },
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

      {/* Enhanced Floating Logout Button */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="fixed bottom-6 right-6 z-50"
      >
        <motion.button
          onClick={() => navigate("/")}
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          className="group relative bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white p-3 rounded-full shadow-2xl transition-all duration-300"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-red-400 to-pink-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
          <LogOut size={20} className="relative z-10" />
        </motion.button>
      </motion.div>

      {/* Enhanced Page Content */}
      <motion.div
        className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 py-10"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        {/* Enhanced Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-6 shadow-2xl">
            <GraduationCap className="w-10 h-10 text-white" />
          </div>
          <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-white via-blue-100 to-purple-100 bg-clip-text text-transparent mb-4">
            Course Preview
          </h1>
          <p className="text-blue-200 text-xl max-w-2xl mx-auto opacity-90">
            Review your selected courses and generate certificates
          </p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl w-full items-start">
          {courses.map((course, index) => (
            <motion.div
              key={index}
              variants={cardVariants}
              className="group relative"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl blur opacity-25 group-hover:opacity-40 transition-opacity duration-300"></div>
              <div className="relative bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-8 hover:shadow-2xl transition-all duration-300 border border-white/20">
                {/* Enhanced Header */}
                <div className="flex justify-between items-start mb-6">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-4">
                      <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center shadow-lg">
                        <BookOpen className="w-6 h-6 text-white" />
                      </div>
                      <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                        {index + 1}
                      </div>
                    </div>
                    <motion.button
                      onClick={() => handleCourseClick(course)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      transition={{ type: "spring", stiffness: 300 }}
                      className="text-gray-800 text-lg font-bold text-left hover:text-blue-600 transition-colors duration-200 leading-tight"
                    >
                      {course}
                    </motion.button>
                  </div>

                  <motion.button
                    onClick={() => toggleMenu(course)}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    className="text-gray-500 hover:text-blue-600 transition-colors duration-200 p-2 rounded-full hover:bg-blue-50"
                  >
                    <Eye className="w-5 h-5" />
                  </motion.button>
                </div>

                {/* Enhanced Status Section */}
                <AnimatePresence>
                  {openMenus[course] && (
                    <motion.div
                      initial={{ opacity: 0, height: 0, y: -10 }}
                      animate={{ opacity: 1, height: "auto", y: 0 }}
                      exit={{ opacity: 0, height: 0, y: -10 }}
                      transition={{ duration: 0.4, ease: "easeOut" }}
                      className="mt-6 space-y-4 border-t border-gray-200 pt-6"
                    >
                      <h4 className="text-sm font-bold text-gray-700 uppercase tracking-wide mb-4">Course Status</h4>

                      <div className="space-y-3">
                        <div className="flex justify-between items-center p-3 bg-gray-50 rounded-xl">
                          <div className="flex items-center gap-3">
                            <CheckCircle className="w-4 h-4 text-gray-600" />
                            <span className="text-sm font-semibold text-gray-700">Certificate Status</span>
                          </div>
                          <div className="flex items-center gap-2">
                            {statusFlags[course] ? (
                              <>
                                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                                <span className="text-xs font-medium text-green-600">Visited</span>
                              </>
                            ) : (
                              <>
                                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                                <span className="text-xs font-medium text-red-600">Pending</span>
                              </>
                            )}
                          </div>
                        </div>

                        <div className="flex justify-between items-center p-3 bg-gray-50 rounded-xl">
                          <div className="flex items-center gap-3">
                            <XCircle className="w-4 h-4 text-gray-600" />
                            <span className="text-sm font-semibold text-gray-700">Verification</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                            <span className="text-xs font-medium text-red-600">Pending</span>
                          </div>
                        </div>

                        <div className="flex justify-between items-center p-3 bg-gray-50 rounded-xl">
                          <div className="flex items-center gap-3">
                            <Cloud className="w-4 h-4 text-gray-600" />
                            <span className="text-sm font-semibold text-gray-700">Cloud Storage</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                            <span className="text-xs font-medium text-red-600">Not Synced</span>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Enhanced Edit Button */}
        <motion.div
          className="mt-16 flex justify-center"
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.6 }}
        >
          <motion.button
            onClick={handleEditSelection}
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 20 }}
            className="group relative flex items-center gap-3 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white font-bold py-4 px-8 rounded-full shadow-2xl transition-all duration-300"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-gray-500 to-gray-600 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
            <ArrowLeft className="w-5 h-5 relative z-10" />
            <span className="relative z-10">Edit Course Selection</span>
          </motion.button>
        </motion.div>
      </motion.div>
    </div>
  );
}

export default CoursePreview;
