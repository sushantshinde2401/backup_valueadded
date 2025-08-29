import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import "../../App.css";
import {
  User,
  Calendar,
  Home,
  Fingerprint,
  ShieldCheck,
  BadgeCheck,
  Mail,
  Phone,
  Building2,
  CreditCard,
  Wallet,
  LogOut,
  ArrowLeftCircle,
  ArrowRightCircle,
  FileText,
  CheckCircle,
  AlertCircle,
  Flag
} from "lucide-react";

function CandidateDetails() {
  const navigate = useNavigate();
  const location = useLocation();

  // Get OCR data from navigation state
  const ocrData = location.state?.ocrData;
  const jsonFile = location.state?.jsonFile;
  const sessionId = location.state?.sessionId;

  // Company names list - same as used in Payment Receipt and Rate List
  const companyNames = [
    'ABC Maritime Training',
    'XYZ Shipping Company',
    'Marine Safety Institute',
    'Offshore Training Center',
    'Coastal Academy',
    'Deep Sea Training'
  ];

  const [formData, setFormData] = useState({
    lastName: "",
    firstName: "",
    dob: "",
    nationality: "",
    address: "",
    passport: "",
    cdcNo: "",
    indosNo: "",
    email: "",
    phone: "",
    companyName: "",
    partyName: "", // New field for company selection
    vendorName: "",
    paymentStatus: "",
    rollNo: "",
    paymentProof: null,
    session_id: sessionId || "",
  });

  const [isAutoFilled, setIsAutoFilled] = useState(false);
  const [fillStatus, setFillStatus] = useState("");

  // Save initial form data to localStorage on component mount
  useEffect(() => {
    localStorage.setItem('candidateData', JSON.stringify(formData));
    window.dispatchEvent(new CustomEvent('candidateDataUpdated', { detail: formData }));
  }, []); // Empty dependency array means this runs once on mount

  const handleFill = async () => {
    try {
      setFillStatus("Filling form with OCR data...");

      let dataToUse = ocrData;

      // If no OCR data in state, try to fetch from backend using single file
      if (!dataToUse) {
        console.log("Fetching OCR data from structured_passport_data.json");
        try {
          const res = await fetch(`http://localhost:5000/get-candidate-data/structured_passport_data.json`);
          if (res.ok) {
            const result = await res.json();
            dataToUse = result.data;
            console.log("Fetched data from backend:", dataToUse);
          } else {
            console.error("Failed to fetch from backend:", res.status, res.statusText);
          }
        } catch (error) {
          console.error("Error fetching OCR data:", error);
        }
      }

      if (!dataToUse) {
        setFillStatus("No OCR data available");
        console.warn("No OCR data found in state or backend");
        return;
      }

      // Debug: Log the OCR data structure
      console.log("=== OCR DATA ANALYSIS ===");
      console.log("Full OCR Data received:", dataToUse);

      const front = dataToUse.passport_front || {};
      const back = dataToUse.passport_back || {};
      const cdc = dataToUse.cdc || {};

      console.log("Passport Front data:", front);
      console.log("Passport Back data:", back);
      console.log("CDC data:", cdc);

      // Show raw text if available for debugging
      if (front.raw_text) {
        console.log("Raw passport front text:", front.raw_text.substring(0, 200) + "...");
      }
      if (back.raw_text) {
        console.log("Raw passport back text:", back.raw_text.substring(0, 200) + "...");
      }
      if (cdc.raw_text) {
        console.log("Raw CDC text:", cdc.raw_text.substring(0, 200) + "...");
      }

      // Simplified field mapping for essential fields only
      const extractedData = {
        lastName: front["Surname"] || front.surname || front["SURNAME"] || "",
        firstName: front["Given Name(s)"] || front["Given Names"] || front.given_names ||
                  front["GIVEN_NAMES"] || "",
        dob: front["Date of Birth"] || front.date_of_birth || front["DATE_OF_BIRTH"] || "",
        nationality: front["Nationality"] || front.nationality || front["NATIONALITY"] ||
                    front["Country Code"] || front.country_code || front["COUNTRY_CODE"] || "",
        address: back["Address"] || back.address || back["ADDRESS"] || "",
        passport: front["Passport No."] || front["Passport No"] || front.passport_no ||
                 front["PASSPORT_NO"] || "",
        cdcNo: cdc.cdc_no || cdc["cdc_no"] || cdc["CDC_NO"] || "",
        indosNo: cdc.indos_no || cdc["indos_no"] || cdc["INDOS_NO"] || "",
      };

      console.log("=== EXTRACTED DATA FOR FORM ===");
      console.log("Extracted data:", extractedData);

      // Count how many fields were successfully extracted
      const filledFields = Object.entries(extractedData).filter(([key, value]) => value && value.trim() !== "").length;
      console.log(`Successfully extracted ${filledFields} out of ${Object.keys(extractedData).length} fields`);

      const updatedFormData = {
        ...formData,
        ...extractedData,
        // Keep existing values for fields not in OCR
        email: formData.email,
        phone: formData.phone,
        companyName: formData.companyName,
        vendorName: formData.vendorName,
        paymentStatus: formData.paymentStatus,
        rollNo: formData.rollNo,
        paymentProof: formData.paymentProof,
      };

      setFormData(updatedFormData);

      // Save to localStorage for real-time access by certificate pages
      localStorage.setItem('candidateData', JSON.stringify(updatedFormData));

      // Dispatch custom event to notify certificate pages
      window.dispatchEvent(new CustomEvent('candidateDataUpdated', { detail: updatedFormData }));

      setIsAutoFilled(true);
      setFillStatus(`Form filled! Extracted ${filledFields} fields from OCR data.`);

      // Clear status after 5 seconds
      setTimeout(() => setFillStatus(""), 5000);

    } catch (err) {
      console.error("Fill error:", err);
      setFillStatus("Failed to fill form with OCR data");
      setTimeout(() => setFillStatus(""), 3000);
    }
  };

  const handleChange = (e) => {
    const { id, value } = e.target;
    const isDropdown = id === "vendorName" || id === "paymentStatus" || id === "partyName";

    const updatedFormData = {
      ...formData,
      [id]: isDropdown ? value : value.toUpperCase(),
    };

    setFormData(updatedFormData);

    // Save to localStorage for real-time access by certificate pages
    localStorage.setItem('candidateData', JSON.stringify(updatedFormData));

    // Dispatch custom event to notify certificate pages
    window.dispatchEvent(new CustomEvent('candidateDataUpdated', { detail: updatedFormData }));

    // If party name (company) is selected, also update certificate selections
    if (id === "partyName" && value) {
      updateCertificateSelectionsWithCompany(value);
    }
  };

  // Function to update certificate selections with selected company
  const updateCertificateSelectionsWithCompany = async (selectedCompany) => {
    try {
      // Get current candidate data
      const candidateData = {
        firstName: formData.firstName,
        lastName: formData.lastName
      };

      // Only update if we have candidate name data
      if (candidateData.firstName && candidateData.lastName) {
        console.log(`[CANDIDATE] Updating certificates for ${candidateData.firstName} ${candidateData.lastName} with company: ${selectedCompany}`);

        // This will be used when certificates are generated
        // The certificate generation process will check for this company selection
        localStorage.setItem('selectedCompanyForCandidate', JSON.stringify({
          candidateName: `${candidateData.firstName} ${candidateData.lastName}`,
          companyName: selectedCompany,
          timestamp: new Date().toISOString()
        }));

        // Dispatch event for real-time updates
        window.dispatchEvent(new CustomEvent('companySelectedForCandidate', {
          detail: {
            candidateName: `${candidateData.firstName} ${candidateData.lastName}`,
            companyName: selectedCompany
          }
        }));
      }
    } catch (error) {
      console.error('[CANDIDATE] Error updating certificate selections with company:', error);
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file || !sessionId) return;

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append('paymentScreenshot', file);
      formData.append('sessionId', sessionId);

      // Upload to backend
      const response = await fetch('http://localhost:5000/upload-payment-screenshot', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.status === 'success') {
        // Update form data with uploaded filename
        setFormData(prev => ({
          ...prev,
          paymentProof: result.filename // Store the server filename
        }));

        alert('Payment screenshot uploaded successfully!');
      } else {
        throw new Error(result.error || 'Upload failed');
      }

    } catch (error) {
      console.error('Upload error:', error);
      alert(`Upload failed: ${error.message}`);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      // Prepare data for submission
      const submissionData = {
        ...formData,
        // Add metadata
        submission_timestamp: new Date().toISOString(),
        ocr_source: jsonFile || "manual_entry",
        auto_filled: isAutoFilled
      };

      const res = await fetch("http://localhost:5000/save-candidate-data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(submissionData),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const result = await res.json();

      if (result.status === 'success') {
        alert("Candidate data saved successfully!");
        console.log("Saved data:", result);

        // Navigate to course selection
        navigate("/course-selection", {
          state: {
            candidateData: submissionData,
            savedFile: result.filename
          }
        });
      } else {
        throw new Error(result.message || "Save failed");
      }

    } catch (err) {
      console.error("Submission error:", err);
      alert(`Submission failed: ${err.message}`);
    }
  };

  const iconMap = {
    lastName: <User size={16} />,
    firstName: <User size={16} />,
    dob: <Calendar size={16} />,
    nationality: <Flag size={16} />,
    address: <Home size={16} />,
    passport: <Fingerprint size={16} />,
    cdcNo: <ShieldCheck size={16} />,
    indosNo: <BadgeCheck size={16} />,
    email: <Mail size={16} />,
    phone: <Phone size={16} />,
    companyName: <Building2 size={16} />,
    partyName: <Building2 size={16} />,
    vendorName: <Building2 size={16} />,
    paymentStatus: <CreditCard size={16} />,
    rollNo: <Wallet size={16} />,
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

      {/* Enhanced Floating Navigation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="fixed bottom-6 right-6 z-50 flex gap-3"
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
          onClick={() => navigate("/upload-docx")}
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          className="group relative bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white p-3 rounded-full shadow-2xl transition-all duration-300"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-gray-500 to-gray-600 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
          <ArrowLeftCircle size={20} className="relative z-10" />
        </motion.button>

        <motion.button
          onClick={() => navigate("/course-selection")}
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          className="group relative bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white p-3 rounded-full shadow-2xl transition-all duration-300"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-cyan-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
          <ArrowRightCircle size={20} className="relative z-10" />
        </motion.button>
      </motion.div>


      {/* Enhanced Form Container */}
      <motion.div
        initial={{ y: 50, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.3, duration: 0.8 }}
        className="relative z-10 flex justify-center items-center py-8 px-4 pt-20"
      >
        <div className="max-w-5xl w-full">
          <motion.form
            onSubmit={handleSubmit}
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5, duration: 0.6 }}
            className="bg-white/95 backdrop-blur-sm p-10 shadow-2xl rounded-3xl border border-white/20"
          >
            {/* Enhanced Header */}
            <motion.div
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.7, duration: 0.6 }}
              className="text-center mb-10"
            >
              <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-6 shadow-lg">
                <User className="w-8 h-8 text-white" />
              </div>
              <h2 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-4">
                Candidate Information
              </h2>
              <p className="text-gray-600 text-lg max-w-2xl mx-auto">
                Enter candidate details or use OCR data from uploaded documents
              </p>
            </motion.div>

            <div className="flex justify-between items-center mb-8">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-600" />
                <span className="text-lg font-semibold text-gray-700">Form Details</span>
              </div>

              {/* Enhanced OCR Controls */}
              <div className="flex items-center gap-3">
                {(ocrData || jsonFile) && (
                  <>
                    <motion.button
                      type="button"
                      onClick={handleFill}
                      whileHover={{ scale: 1.05, y: -1 }}
                      whileTap={{ scale: 0.95 }}
                      className="group relative bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white px-6 py-3 rounded-full flex items-center gap-2 transition-all duration-300 shadow-lg"
                    >
                      <div className="absolute inset-0 bg-gradient-to-r from-green-400 to-emerald-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
                      <FileText size={16} className="relative z-10" />
                      <span className="relative z-10 font-medium">Fill from OCR</span>
                    </motion.button>

                    <motion.button
                      type="button"
                      onClick={() => {
                        console.log("Current OCR Data:", ocrData);
                        console.log("JSON File:", jsonFile);
                        alert(`OCR Data available: ${ocrData ? 'Yes' : 'No'}\nJSON File: ${jsonFile || 'None'}`);
                      }}
                      whileHover={{ scale: 1.05, y: -1 }}
                      whileTap={{ scale: 0.95 }}
                      className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-3 rounded-full text-sm transition-all duration-300 shadow-lg"
                    >
                      Debug OCR
                    </motion.button>


                  </>
                )}

                {fillStatus && (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium shadow-lg ${
                      fillStatus.includes('success')
                        ? 'bg-green-100 text-green-700 border border-green-200'
                        : fillStatus.includes('Failed')
                        ? 'bg-red-100 text-red-700 border border-red-200'
                        : 'bg-blue-100 text-blue-700 border border-blue-200'
                    }`}>
                    {fillStatus.includes('success') ? (
                      <CheckCircle size={16} />
                    ) : fillStatus.includes('Failed') ? (
                      <AlertCircle size={16} />
                    ) : (
                      <FileText size={16} />
                    )}
                    {fillStatus}
                  </motion.div>
                )}
              </div>
            </div>

            {/* Enhanced OCR Status Indicators */}
            {(ocrData || jsonFile) && !isAutoFilled && (
              <motion.div
                initial={{ y: -10, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.8, duration: 0.6 }}
                className="mb-8 p-6 bg-gradient-to-r from-blue-50 to-cyan-50 border border-blue-200 rounded-2xl shadow-lg"
              >
                <div className="flex items-center gap-3 text-blue-700 mb-2">
                  <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                    <FileText size={16} className="text-white" />
                  </div>
                  <span className="font-bold text-lg">OCR Data Available!</span>
                </div>
                <p className="text-blue-600 leading-relaxed">
                  Click "Fill from OCR" to automatically populate the form with extracted document data.
                </p>
              </motion.div>
            )}

            {isAutoFilled && (
              <motion.div
                initial={{ y: -10, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.8, duration: 0.6 }}
                className="mb-8 p-6 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-2xl shadow-lg"
              >
                <div className="flex items-center gap-3 text-green-700 mb-2">
                  <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                    <CheckCircle size={16} className="text-white" />
                  </div>
                  <span className="font-bold text-lg">Form Auto-filled with OCR Data</span>
                </div>
                <p className="text-green-600 leading-relaxed">
                  Please review and modify the information as needed before submitting.
                </p>
              </motion.div>
            )}

            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.9, duration: 0.6 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-8"
            >
              {[
                // OCR-extracted fields (autocomplete enabled)
                ["lastName", "Last Name", "text", "family-name"],
                ["firstName", "First Name", "text", "given-name"],
                ["dob", "Date of Birth", "date", "bday"],
                ["nationality", "Nationality", "text", "country-name"],
                ["address", "Permanent Address", "text", "street-address"],
                ["passport", "Passport No.", "text", "off"],
                ["cdcNo", "CDC No.", "text", "off"],
                ["indosNo", "INDOS No.", "text", "off"],
                // Manual entry fields (autocomplete disabled)
                ["email", "Email ID", "email", "off"],
                ["phone", "Phone No.", "tel", "off"],
                ["companyName", "Company Joining Name", "text", "off"],
                ["partyName", "Party Name (Company)", "select", "off"],
                ["rollNo", "ROLL NO", "text", "off"],
              ].map(([id, label, type = "text", autocomplete = "off"], index) => (
                <motion.div
                  key={id}
                  initial={{ x: index % 2 === 0 ? -20 : 20, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  transition={{ delay: 1.0 + (index * 0.1), duration: 0.5 }}
                  className="group"
                >
                  <label
                    htmlFor={id}
                    className="block font-semibold mb-3 flex items-center gap-3 text-gray-700"
                  >
                    <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
                      {iconMap[id] && React.cloneElement(iconMap[id], {
                        size: 16,
                        className: "text-white"
                      })}
                    </div>
                    {label}:
                  </label>
                  <div className="relative">
                    {type === "select" && id === "partyName" ? (
                      <select
                        id={id}
                        name={id}
                        value={formData[id]}
                        onChange={handleChange}
                        required
                        className="w-full text-sm bg-white border-2 border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors duration-200 hover:border-blue-300 shadow-sm"
                      >
                        <option value="">Select Company</option>
                        {companyNames.map(company => (
                          <option key={company} value={company}>
                            {company}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={type}
                        id={id}
                        name={id}
                        value={formData[id]}
                        onChange={handleChange}
                        autoComplete={autocomplete}
                        required
                        className="w-full text-sm bg-white border-2 border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:border-blue-500 transition-colors duration-200 hover:border-blue-300 shadow-sm"
                        placeholder={`Enter ${label.toLowerCase()}`}
                      />
                    )}
                  </div>
                </motion.div>
              ))}

              {/* Enhanced Vendor Name dropdown */}
              <motion.div
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 2.1, duration: 0.5 }}
                className="group"
              >
                <label
                  htmlFor="vendorName"
                  className="block font-semibold mb-3 flex items-center gap-3 text-gray-700"
                >
                  <div className="w-8 h-8 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full flex items-center justify-center">
                    <Building2 size={16} className="text-white" />
                  </div>
                  Vendor Name:
                </label>
                <div className="relative">
                  <select
                    id="vendorName"
                    name="vendorName"
                    value={formData.vendorName}
                    onChange={handleChange}
                    autoComplete="organization"
                    required
                    className="w-full text-sm bg-white border-2 border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:border-emerald-500 transition-colors duration-200 hover:border-emerald-300 shadow-sm appearance-none"
                  >
                    <option value="">Select Vendor</option>
                    <option value="Vendor A">Vendor A</option>
                    <option value="Vendor B">Vendor B</option>
                    <option value="Vendor C">Vendor C</option>
                  </select>
                </div>
              </motion.div>

              {/* Enhanced Payment Status dropdown */}
              <motion.div
                initial={{ x: 20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 2.2, duration: 0.5 }}
                className="group"
              >
                <label
                  htmlFor="paymentStatus"
                  className="block font-semibold mb-3 flex items-center gap-3 text-gray-700"
                >
                  <div className="w-8 h-8 bg-gradient-to-r from-orange-500 to-red-500 rounded-full flex items-center justify-center">
                    <CreditCard size={16} className="text-white" />
                  </div>
                  Payment Status:
                </label>
                <div className="relative">
                  <select
                    id="paymentStatus"
                    name="paymentStatus"
                    value={formData.paymentStatus}
                    onChange={handleChange}
                    autoComplete="off"
                    required
                    className="w-full text-sm bg-white border-2 border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:border-orange-500 transition-colors duration-200 hover:border-orange-300 shadow-sm appearance-none"
                  >
                    <option value="">Select Status</option>
                    <option value="PAID">PAID</option>
                    <option value="CREDIT">CREDIT</option>
                  </select>
                </div>
              </motion.div>

              {/* Enhanced Payment Proof Upload */}
              {formData.paymentStatus === "PAID" && (
                <motion.div
                  initial={{ y: 20, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ duration: 0.5 }}
                  className="col-span-1 md:col-span-2 group"
                >
                  <label className="block font-semibold mb-3 flex items-center gap-3 text-gray-700">
                    <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                      <FileText size={16} className="text-white" />
                    </div>
                    Attach Payment Screenshot:
                  </label>
                  <div className="relative">
                    <input
                      type="file"
                      name="paymentProof"
                      accept="image/*"
                      onChange={handleFileChange}
                      autoComplete="off"
                      className="w-full text-sm bg-white border-2 border-gray-200 rounded-xl px-4 py-3 focus:outline-none focus:border-purple-500 transition-colors duration-200 hover:border-purple-300 shadow-sm file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
                    />
                    {formData.paymentProof && (
                      <div className="mt-2 text-sm text-green-600 flex items-center gap-2">
                        <CheckCircle size={16} />
                        Screenshot uploaded: {formData.paymentProof}
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </motion.div>

            {/* Enhanced Submit Buttons */}
            <motion.div
              initial={{ y: 30, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 2.3, duration: 0.6 }}
              className="flex justify-center gap-6 pt-12"
            >
              <motion.button
                type="submit"
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="group relative bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white px-8 py-4 rounded-full font-bold text-lg shadow-2xl transition-all duration-300"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-green-400 to-emerald-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
                <span className="relative z-10">Save & Continue</span>
              </motion.button>

              <motion.button
                type="button"
                onClick={handleFill}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="group relative bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white px-8 py-4 rounded-full font-bold text-lg shadow-2xl transition-all duration-300"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
                <span className="relative z-10">Auto Fill</span>
              </motion.button>
            </motion.div>
          </motion.form>
        </div>
      </motion.div>
    </div>
  );
}

export default CandidateDetails;
