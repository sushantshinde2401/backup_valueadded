import React, { useState, useEffect } from "react";
import "../../App.css";
import { useNavigate } from "react-router-dom";
import {
  Info,
  ArrowLeftCircle,
  ArrowRight,
  BookOpen,
  Plus,
  Trash2,
  ChevronDown,
  GraduationCap,
  LogOut
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

function CourseSelection() {
  const [courses, setCourses] = useState(() => {
    try {
      const saved = localStorage.getItem("courses");
      return saved ? JSON.parse(saved) : [];
    } catch (error) {
      console.error("Error loading courses from localStorage:", error);
      return [];
    }
  });
  const [validationError, setValidationError] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedCourseForAdd, setSelectedCourseForAdd] = useState("");
  const navigate = useNavigate();

  // Available maritime courses with descriptions
  const availableCourses = [
    { code: "STCW", name: "STCW - Basic Safety Training Certificate", hasPage: true },
    { code: "STSDSD", name: "STSDSD - Verification Certificate", hasPage: true },
    { code: "BOSIET", name: "BOSIET - Safety Training Certificate", hasPage: true },
    { code: "H2S", name: "H2S - Safety Training Certificate", hasPage: true },
    { code: "HUET", name: "HUET - Helicopter Underwater Escape Training", hasPage: false },
    { code: "FOET", name: "FOET - Further Offshore Emergency Training", hasPage: false },
    { code: "MIST", name: "MIST - Minimum Industry Safety Training", hasPage: false },
    { code: "OPITO", name: "OPITO - Offshore Petroleum Industry Training", hasPage: false },
    { code: "TBOSIET", name: "TBOSIET - Tropical Basic Offshore Safety Induction", hasPage: false },
    { code: "CA-EBS", name: "CA-EBS - Compressed Air Emergency Breathing System", hasPage: false }
  ];

  useEffect(() => {
    try {
      localStorage.setItem("courses", JSON.stringify(courses));
      console.log("CourseSelection: Courses updated in localStorage:", courses);
    } catch (error) {
      console.error("Error saving courses to localStorage:", error);
    }
  }, [courses]);

  // Debug component mounting
  useEffect(() => {
    console.log("CourseSelection component mounted");
    console.log("Available courses:", availableCourses);
    console.log("Current courses:", courses);
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (showDropdown && !event.target.closest('.dropdown-container')) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showDropdown]);



  const handleSave = () => {
    if (courses.length === 0) {
      setValidationError("Please select at least one course before saving.");
      return;
    }

    setValidationError("");
    localStorage.setItem("courses", JSON.stringify(courses));
    console.log(`[COURSE] Saved ${courses.length} courses:`, courses);
    navigate("/course-preview");
  };



  // Add course from dropdown
  const handleAddCourseFromDropdown = () => {
    if (selectedCourseForAdd && !courses.includes(selectedCourseForAdd)) {
      const updatedCourses = [...courses, selectedCourseForAdd];
      setCourses(updatedCourses);

      // Reset selection
      setSelectedCourseForAdd("");
      setShowDropdown(false);

      console.log(`[COURSE] Added course from dropdown: ${selectedCourseForAdd}`);
    }
  };

  // Remove course from list
  const handleRemoveCourseFromList = (courseToRemove) => {
    const updatedCourses = courses.filter(course => course !== courseToRemove);
    setCourses(updatedCourses);
    console.log(`[COURSE] Removed course: ${courseToRemove}`);
  };

  // Error boundary for debugging
  try {
    console.log("CourseSelection: Rendering component");
    console.log("CourseSelection: availableCourses length:", availableCourses.length);
    console.log("CourseSelection: courses state:", courses);

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

      {/* Enhanced Floating Action Buttons */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="fixed bottom-6 right-6 flex gap-3 z-50"
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

        <motion.button
          onClick={() => navigate("/candidate-details")}
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          className="group relative bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white p-3 rounded-full shadow-2xl transition-all duration-300"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-gray-500 to-gray-600 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
          <ArrowLeftCircle size={20} className="relative z-10" />
        </motion.button>
      </motion.div>


      <div className="relative z-10 flex flex-col items-center justify-start min-h-screen p-6 py-12">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="bg-white/95 backdrop-blur-sm shadow-2xl rounded-3xl w-full max-w-5xl p-10 mb-6 border border-white/20"
          style={{ position: 'relative', zIndex: 'auto' }}
        >
          {/* Enhanced Header */}
          <motion.div
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="text-center mb-10"
          >
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-6 shadow-lg">
              <GraduationCap className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
              Course Selection
            </h1>
            <p className="text-gray-600 text-lg max-w-2xl mx-auto">
              Choose from 10 maritime courses using the dropdown menu below, then proceed to course preview
            </p>
          </motion.div>

          {/* Step 1: Maritime Course Selection */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.6 }}
            className="mb-12"
          >
            <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-white/20 w-full" style={{ overflow: 'visible', display: 'block', position: 'relative', zIndex: 1 }}>
              <div className="flex items-center gap-3 mb-6">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
                  <Plus className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold text-gray-800">Step 1: Add Maritime Courses</h3>
                  <p className="text-gray-600 text-sm mt-1">Select courses from the dropdown menu below</p>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row gap-3">
                <div className="relative flex-1 dropdown-container" style={{ overflow: 'visible', zIndex: 1000 }}>
                  <button
                    type="button"
                    onClick={() => {
                      console.log('Dropdown button clicked, current state:', showDropdown);
                      setShowDropdown(!showDropdown);
                    }}
                    className="dropdown-button w-full flex items-center justify-between px-4 py-3 bg-gray-50 border-2 border-gray-200 rounded-xl hover:bg-gray-100 hover:border-blue-300 focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-200 transition-all duration-200 text-left cursor-pointer"
                  >
                    <span className="text-gray-700">
                      {selectedCourseForAdd
                        ? availableCourses.find(c => c.code === selectedCourseForAdd)?.name
                        : "Select a maritime course..."}
                    </span>
                    <ChevronDown className={`w-5 h-5 text-gray-500 transition-transform duration-200 ${showDropdown ? 'rotate-180' : ''}`} />
                  </button>

                  <AnimatePresence>
                    {showDropdown && (
                      <>
                        {/* Dropdown backdrop for better visibility */}
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="fixed inset-0 bg-black/10"
                          style={{ zIndex: 999998 }}
                          onClick={() => setShowDropdown(false)}
                        />

                        {/* Dropdown menu */}
                        <motion.div
                          initial={{ opacity: 0, y: -10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          transition={{ duration: 0.2 }}
                          className="dropdown-menu absolute top-full left-0 right-0 mt-2 bg-white border-2 border-gray-300 rounded-xl shadow-2xl max-h-64 overflow-y-auto"
                          style={{
                            position: 'absolute',
                            zIndex: 999999,
                            backgroundColor: 'white',
                            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
                            minWidth: '100%'
                          }}
                        >
                        {(() => {
                          const availableCoursesFiltered = availableCourses.filter(course => !courses.includes(course.code));
                          console.log('Available courses for dropdown:', availableCoursesFiltered);
                          console.log('Current selected courses:', courses);

                          if (availableCoursesFiltered.length === 0) {
                            return (
                              <div className="px-4 py-3 text-gray-500 text-center">
                                All courses have been added
                              </div>
                            );
                          }

                          return availableCoursesFiltered.map((course) => (
                            <button
                              key={course.code}
                              type="button"
                              onClick={() => {
                                console.log('Course selected:', course.code);
                                setSelectedCourseForAdd(course.code);
                                setShowDropdown(false);
                              }}
                              className="w-full px-4 py-3 text-left hover:bg-blue-50 focus:bg-blue-50 focus:outline-none transition-colors duration-200 border-b border-gray-100 last:border-b-0 group cursor-pointer"
                            >
                              <div className="flex items-center justify-between">
                                <div>
                                  <div className="font-medium text-gray-800 group-hover:text-blue-600">
                                    {course.code}
                                  </div>
                                  <div className="text-sm text-gray-500 mt-1">
                                    {course.name.split(' - ')[1]}
                                  </div>
                                </div>
                                {course.hasPage && (
                                  <div className="text-xs bg-green-100 text-green-600 px-2 py-1 rounded-full">
                                    Certificate Available
                                  </div>
                                )}
                              </div>
                            </button>
                          ));
                        })()}
                        </motion.div>
                      </>
                    )}
                  </AnimatePresence>
                </div>

                <motion.button
                  type="button"
                  onClick={handleAddCourseFromDropdown}
                  disabled={!selectedCourseForAdd}
                  whileHover={{ scale: selectedCourseForAdd ? 1.05 : 1 }}
                  whileTap={{ scale: selectedCourseForAdd ? 0.95 : 1 }}
                  className={`px-6 py-3 rounded-xl font-medium transition-all duration-200 w-full sm:w-auto ${
                    selectedCourseForAdd
                      ? 'bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white shadow-lg'
                      : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  }`}
                >
                  Add Course
                </motion.button>
              </div>
            </div>
          </motion.div>

          {/* Visual Separator */}
          {courses.length > 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.6 }}
              className="flex items-center justify-center mb-8"
            >
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent"></div>
              <div className="mx-4 text-gray-500 text-sm font-medium">Selected Courses</div>
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent"></div>
            </motion.div>
          )}

          {/* Step 2: Selected Courses Review */}
          {courses.length > 0 && (
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.5, duration: 0.6 }}
              className="mb-12"
            >
              <div className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-white/20">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                    <BookOpen className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold text-gray-800">Step 2: Review Selected Courses</h3>
                    <p className="text-gray-600 text-sm mt-1">Your chosen maritime training courses ({courses.length} selected)</p>
                  </div>
                </div>
                <div className="space-y-3">
                  {courses.map((course, index) => {
                    const courseInfo = availableCourses.find(c => c.code === course);
                    return (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
                        <div className="flex items-center gap-3">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                            courseInfo?.hasPage
                              ? 'bg-green-100 text-green-600'
                              : 'bg-orange-100 text-orange-600'
                          }`}>
                            <BookOpen className="w-4 h-4" />
                          </div>
                          <div>
                            <div className="font-medium text-gray-800">{course}</div>
                            <div className="text-sm text-gray-500">
                              {courseInfo?.name.split(' - ')[1] || 'Maritime Safety Training'}
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          {courseInfo?.hasPage && (
                            <div className="text-xs bg-green-100 text-green-600 px-2 py-1 rounded-full">
                              Certificate Available
                            </div>
                          )}
                          <button
                            type="button"
                            onClick={() => handleRemoveCourseFromList(course)}
                            className="text-gray-400 hover:text-red-500 transition-colors duration-200 p-1 rounded-full hover:bg-red-50"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}

          {validationError && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-gradient-to-r from-red-50 to-pink-50 border border-red-200 rounded-2xl p-4 mb-6 shadow-lg"
            >
              <p className="text-red-700 font-medium flex items-center gap-3">
                <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center">
                  <Info className="w-4 h-4 text-white" />
                </div>
                {validationError}
              </p>
            </motion.div>
          )}


          {/* Step 3: Save & Continue */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.6 }}
            className="text-center mt-16"
          >
            <div className="mb-6">
              <h3 className="text-xl font-bold text-white mb-2">Step 3: Proceed to Course Preview</h3>
              <p className="text-blue-200">Save your selection and continue to the course preview page</p>
            </div>
            <motion.button
              onClick={handleSave}
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              className="group relative bg-gradient-to-r from-green-500 to-blue-500 hover:from-green-600 hover:to-blue-600 text-white font-bold py-4 px-8 rounded-full transition-all duration-300 flex items-center gap-3 shadow-2xl mx-auto"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-green-400 to-blue-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
              <ArrowRight className="w-5 h-5 relative z-10" />
              <span className="relative z-10">Save & Continue to Course Preview</span>
            </motion.button>
          </motion.div>
        </motion.div>



      </div>
    </div>
  );
  } catch (error) {
    console.error("CourseSelection rendering error:", error);
    return (
      <div className="min-h-screen bg-red-100 flex items-center justify-center">
        <div className="bg-white p-8 rounded-lg shadow-lg max-w-md">
          <h2 className="text-xl font-bold text-red-600 mb-4">Component Error</h2>
          <p className="text-gray-700 mb-4">There was an error loading the Course Selection page.</p>
          <p className="text-sm text-gray-500">Check the browser console for details.</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Reload Page
          </button>
        </div>
      </div>
    );
  }
}

export default CourseSelection;
