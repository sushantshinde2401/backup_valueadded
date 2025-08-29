import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  UploadCloud,
  ImageUp,
  PenLine,
  BadgeHelp,
  FileImage,
  IdCard,
  FileCheck,
  LogOut,
  ArrowRightCircle,
  CheckCircle,
  AlertCircle,
  Loader2,
} from "lucide-react";
import { motion } from "framer-motion";

function UploadDocx() {
  const navigate = useNavigate();

  // State management
  const [files, setFiles] = useState({
    photo: null,
    signature: null,
    passport_front_img: null,
    passport_back_img: null,
    cdc_img: null,
    marksheet: null,
  });

  const [previews, setPreviews] = useState({});
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [errors, setErrors] = useState({});

  // Required fields
  const requiredFields = ['photo', 'signature', 'passport_front_img', 'passport_back_img'];

  // Check if upload button should be enabled
  const isUploadEnabled = () => {
    return requiredFields.every(field => files[field] !== null);
  };

  // Handle file selection
  const handleFileChange = (fieldName, event) => {
    const file = event.target.files[0];
    if (file) {
      // Validate file type
      if (!file.type.startsWith('image/')) {
        setErrors(prev => ({
          ...prev,
          [fieldName]: 'Please select an image file'
        }));
        return;
      }

      // Validate file size (50MB limit)
      if (file.size > 50 * 1024 * 1024) {
        setErrors(prev => ({
          ...prev,
          [fieldName]: 'File size must be less than 50MB'
        }));
        return;
      }

      // Clear any previous errors
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[fieldName];
        return newErrors;
      });

      // Update files state
      setFiles(prev => ({
        ...prev,
        [fieldName]: file
      }));

      // Create preview
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreviews(prev => ({
          ...prev,
          [fieldName]: e.target.result
        }));
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();

    // Validate required fields
    const missingFields = requiredFields.filter(field => !files[field]);
    if (missingFields.length > 0) {
      setErrors({
        general: `Please upload all required files: ${missingFields.join(', ')}`
      });
      return;
    }

    setIsUploading(true);
    setUploadProgress("Preparing files for upload...");
    setErrors({});

    try {
      // Create FormData
      const formData = new FormData();

      // Add all files to FormData
      Object.entries(files).forEach(([key, file]) => {
        if (file) {
          formData.append(key, file);
        }
      });

      setUploadProgress("Uploading files and processing OCR...");

      // Send to backend
      const response = await fetch('http://localhost:5000/upload-images', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const result = await response.json();

      if (result.status === 'success') {
        setUploadProgress("Upload successful! Redirecting...");

        // Navigate to candidate details with OCR data and session ID
        setTimeout(() => {
          navigate('/candidate-details', {
            state: {
              ocrData: result.data,
              jsonFile: result.json_file,
              sessionId: result.session_id
            }
          });
        }, 1500);
      } else {
        throw new Error(result.message || 'Upload failed');
      }

    } catch (error) {
      console.error('Upload error:', error);
      setErrors({
        general: `Upload failed: ${error.message}`
      });
      setIsUploading(false);
      setUploadProgress("");
    }
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

      {/* Enhanced Floating Buttons */}
      <div className="fixed bottom-6 right-6 z-50 flex gap-3">
        <motion.button
          onClick={() => navigate("/")}
          disabled={isUploading}
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          className="group relative bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white p-3 rounded-full shadow-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-red-400 to-pink-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
          <LogOut size={20} className="relative z-10" />
        </motion.button>

        <motion.button
          onClick={() => navigate("/candidate-details")}
          disabled={isUploading}
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          className="group relative bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white p-3 rounded-full shadow-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
          <ArrowRightCircle size={20} className="relative z-10" />
        </motion.button>
      </div>

      {/* Enhanced Upload Progress Overlay */}
      {isUploading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center"
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white/95 backdrop-blur-sm rounded-2xl p-8 max-w-md w-full mx-4 shadow-2xl border border-white/20"
          >
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg">
                <Loader2 className="animate-spin h-8 w-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-3">Processing Upload</h3>
              <p className="text-gray-600 text-sm leading-relaxed">{uploadProgress}</p>
              <div className="mt-4 w-full bg-gray-200 rounded-full h-2">
                <div className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* Enhanced Error Display */}
      {errors.general && (
        <motion.div
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          className="fixed top-6 right-6 z-50 bg-red-50/95 backdrop-blur-sm border border-red-200 text-red-700 px-6 py-4 rounded-2xl max-w-md shadow-2xl"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-red-500 rounded-full flex items-center justify-center">
              <AlertCircle className="h-4 w-4 text-white" />
            </div>
            <span className="font-medium">{errors.general}</span>
          </div>
        </motion.div>
      )}

      {/* Enhanced Page Content */}
      <motion.div
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.8 }}
        className="relative z-10 flex justify-center items-center py-10 px-4 pt-20"
      >
        <div className="bg-white/95 backdrop-blur-sm rounded-3xl shadow-2xl p-12 w-full max-w-6xl border border-white/20">
          {/* Header Section */}
          <motion.div
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="text-center mb-12"
          >
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-6 shadow-lg">
              <UploadCloud className="w-8 h-8 text-white" />
            </div>
            <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
              Document Upload Center
            </h2>
            <p className="text-gray-600 text-lg max-w-2xl mx-auto">
              Upload your documents for automated OCR processing and candidate data extraction
            </p>
          </motion.div>

          <motion.form
            onSubmit={handleSubmit}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.7, duration: 0.6 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8"
          >
            {/* Enhanced Column 1 */}
            <motion.div
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 0.8, duration: 0.6 }}
              className="space-y-8"
            >
              {/* Photo Upload */}
              <div className="group">
                <label className="block font-semibold mb-3 flex items-center gap-3 text-gray-700">
                  <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center">
                    <ImageUp className="h-4 w-4 text-white" />
                  </div>
                  Upload Photo
                  <span className="text-red-500 ml-1">*</span>
                  {files.photo && <CheckCircle className="h-5 w-5 text-green-500" />}
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileChange('photo', e)}
                    className="block w-full text-sm text-gray-700 bg-white border-2 border-gray-200 rounded-xl cursor-pointer focus:outline-none focus:border-blue-500 transition-colors duration-200 p-3 hover:border-blue-300"
                    disabled={isUploading}
                  />
                </div>
                {errors.photo && (
                  <p className="text-red-500 text-sm mt-2 flex items-center gap-1">
                    <AlertCircle className="h-4 w-4" />
                    {errors.photo}
                  </p>
                )}
                {previews.photo && (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="mt-4"
                  >
                    <img src={previews.photo} alt="Photo preview" className="h-24 w-24 object-cover rounded-xl shadow-lg border-2 border-white" />
                  </motion.div>
                )}
              </div>

              {/* Signature Upload */}
              <div className="group">
                <label className="block font-semibold mb-3 flex items-center gap-3 text-gray-700">
                  <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                    <PenLine className="h-4 w-4 text-white" />
                  </div>
                  Upload Signature
                  <span className="text-red-500 ml-1">*</span>
                  {files.signature && <CheckCircle className="h-5 w-5 text-green-500" />}
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileChange('signature', e)}
                    className="block w-full text-sm text-gray-700 bg-white border-2 border-gray-200 rounded-xl cursor-pointer focus:outline-none focus:border-purple-500 transition-colors duration-200 p-3 hover:border-purple-300"
                    disabled={isUploading}
                  />
                </div>
                {errors.signature && (
                  <p className="text-red-500 text-sm mt-2 flex items-center gap-1">
                    <AlertCircle className="h-4 w-4" />
                    {errors.signature}
                  </p>
                )}
                {previews.signature && (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="mt-4"
                  >
                    <img src={previews.signature} alt="Signature preview" className="h-24 w-24 object-cover rounded-xl shadow-lg border-2 border-white" />
                  </motion.div>
                )}
              </div>
            </motion.div>

            {/* Enhanced Column 2 */}
            <motion.div
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.9, duration: 0.6 }}
              className="space-y-8"
            >
              {/* Passport Front Upload */}
              <div className="group">
                <label className="block font-semibold mb-3 flex items-center gap-3 text-gray-700">
                  <div className="w-8 h-8 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full flex items-center justify-center">
                    <IdCard className="h-4 w-4 text-white" />
                  </div>
                  Passport Front Image
                  <span className="text-red-500 ml-1">*</span>
                  {files.passport_front_img && <CheckCircle className="h-5 w-5 text-green-500" />}
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileChange('passport_front_img', e)}
                    className="block w-full text-sm text-gray-700 bg-white border-2 border-gray-200 rounded-xl cursor-pointer focus:outline-none focus:border-emerald-500 transition-colors duration-200 p-3 hover:border-emerald-300"
                    disabled={isUploading}
                  />
                </div>
                {errors.passport_front_img && (
                  <p className="text-red-500 text-sm mt-2 flex items-center gap-1">
                    <AlertCircle className="h-4 w-4" />
                    {errors.passport_front_img}
                  </p>
                )}
                {previews.passport_front_img && (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="mt-4"
                  >
                    <img src={previews.passport_front_img} alt="Passport front preview" className="h-24 w-36 object-cover rounded-xl shadow-lg border-2 border-white" />
                  </motion.div>
                )}
              </div>

              {/* Passport Back Upload */}
              <div className="group">
                <label className="block font-semibold mb-3 flex items-center gap-3 text-gray-700">
                  <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center">
                    <BadgeHelp className="h-4 w-4 text-white" />
                  </div>
                  Passport Back Image
                  <span className="text-red-500 ml-1">*</span>
                  {files.passport_back_img && <CheckCircle className="h-5 w-5 text-green-500" />}
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileChange('passport_back_img', e)}
                    className="block w-full text-sm text-gray-700 bg-white border-2 border-gray-200 rounded-xl cursor-pointer focus:outline-none focus:border-orange-500 transition-colors duration-200 p-3 hover:border-orange-300"
                    disabled={isUploading}
                  />
                </div>
                {errors.passport_back_img && (
                  <p className="text-red-500 text-sm mt-2 flex items-center gap-1">
                    <AlertCircle className="h-4 w-4" />
                    {errors.passport_back_img}
                  </p>
                )}
                {previews.passport_back_img && (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="mt-4"
                  >
                    <img src={previews.passport_back_img} alt="Passport back preview" className="h-24 w-36 object-cover rounded-xl shadow-lg border-2 border-white" />
                  </motion.div>
                )}
              </div>
            </motion.div>

            {/* Enhanced Column 3 */}
            <motion.div
              initial={{ x: 20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: 1.0, duration: 0.6 }}
              className="space-y-8"
            >
              {/* CDC Image Upload */}
              <div className="group">
                <label className="block font-semibold mb-3 flex items-center gap-3 text-gray-700">
                  <div className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-blue-500 rounded-full flex items-center justify-center">
                    <FileImage className="h-4 w-4 text-white" />
                  </div>
                  CDC Image
                  <span className="text-gray-400 text-sm">(Optional)</span>
                  {files.cdc_img && <CheckCircle className="h-5 w-5 text-green-500" />}
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileChange('cdc_img', e)}
                    className="block w-full text-sm text-gray-700 bg-white border-2 border-gray-200 rounded-xl cursor-pointer focus:outline-none focus:border-indigo-500 transition-colors duration-200 p-3 hover:border-indigo-300"
                    disabled={isUploading}
                  />
                </div>
                {errors.cdc_img && (
                  <p className="text-red-500 text-sm mt-2 flex items-center gap-1">
                    <AlertCircle className="h-4 w-4" />
                    {errors.cdc_img}
                  </p>
                )}
                {previews.cdc_img && (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="mt-4"
                  >
                    <img src={previews.cdc_img} alt="CDC preview" className="h-24 w-36 object-cover rounded-xl shadow-lg border-2 border-white" />
                  </motion.div>
                )}
              </div>

              {/* Marksheet Upload */}
              <div className="group">
                <label className="block font-semibold mb-3 flex items-center gap-3 text-gray-700">
                  <div className="w-8 h-8 bg-gradient-to-r from-rose-500 to-pink-500 rounded-full flex items-center justify-center">
                    <FileCheck className="h-4 w-4 text-white" />
                  </div>
                  Upload Marksheet
                  <span className="text-gray-400 text-sm">(Optional)</span>
                  {files.marksheet && <CheckCircle className="h-5 w-5 text-green-500" />}
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileChange('marksheet', e)}
                    className="block w-full text-sm text-gray-700 bg-white border-2 border-gray-200 rounded-xl cursor-pointer focus:outline-none focus:border-rose-500 transition-colors duration-200 p-3 hover:border-rose-300"
                    disabled={isUploading}
                  />
                </div>
                {errors.marksheet && (
                  <p className="text-red-500 text-sm mt-2 flex items-center gap-1">
                    <AlertCircle className="h-4 w-4" />
                    {errors.marksheet}
                  </p>
                )}
                {previews.marksheet && (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="mt-4"
                  >
                    <img src={previews.marksheet} alt="Marksheet preview" className="h-24 w-36 object-cover rounded-xl shadow-lg border-2 border-white" />
                  </motion.div>
                )}
              </div>
            </motion.div>

            {/* Enhanced Submit Button */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 1.1, duration: 0.6 }}
              className="col-span-1 md:col-span-3 pt-12 flex justify-center"
            >
              <motion.button
                type="submit"
                disabled={!isUploadEnabled() || isUploading}
                whileHover={isUploadEnabled() && !isUploading ? { scale: 1.05, y: -2 } : {}}
                whileTap={isUploadEnabled() && !isUploading ? { scale: 0.95 } : {}}
                className={`group relative font-bold px-12 py-4 rounded-full shadow-2xl transition-all duration-300 flex items-center gap-3 text-lg ${
                  isUploadEnabled() && !isUploading
                    ? 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white cursor-pointer'
                    : 'bg-gray-400 text-gray-200 cursor-not-allowed'
                }`}
              >
                {isUploadEnabled() && !isUploading && (
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
                )}
                {isUploading ? (
                  <>
                    <Loader2 className="animate-spin relative z-10" size={20} />
                    <span className="relative z-10">Processing...</span>
                  </>
                ) : (
                  <>
                    <UploadCloud size={20} className="relative z-10" />
                    <span className="relative z-10">Upload Documents</span>
                  </>
                )}
              </motion.button>
            </motion.div>

            {/* Enhanced Upload Status */}
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 1.2, duration: 0.6 }}
              className="col-span-1 md:col-span-3 text-center pt-6"
            >
              <div className="bg-blue-50 rounded-2xl p-6 border border-blue-100">
                <p className="text-sm text-gray-700 font-medium mb-2">
                  📋 Required fields: Photo, Signature, Passport Front & Back
                </p>
                {!isUploadEnabled() && (
                  <p className="text-sm text-red-600 font-medium flex items-center justify-center gap-2">
                    <AlertCircle className="h-4 w-4" />
                    Please upload all required files to enable upload
                  </p>
                )}
                {isUploadEnabled() && !isUploading && (
                  <p className="text-sm text-green-600 font-medium flex items-center justify-center gap-2">
                    <CheckCircle className="h-4 w-4" />
                    All required files uploaded - Ready to process!
                  </p>
                )}
              </div>
            </motion.div>
          </motion.form>
        </div>
      </motion.div>
    </div>
  );
}

export default UploadDocx;
