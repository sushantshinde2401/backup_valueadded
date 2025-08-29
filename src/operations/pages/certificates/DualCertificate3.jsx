import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ChevronRight,
  ChevronLeft,
  Download,
  LogOut,
  ArrowLeft,
  QrCode,
  User,
} from "lucide-react";
import { motion } from "framer-motion";
import { useCertificateManager } from "../../../shared/hooks/useCertificateManager";

function DualCertificate3() {
  const navigate = useNavigate();
  const [selectedDate, setSelectedDate] = useState("");
  const canvasLeftRef = useRef(null);
  const canvasRightRef = useRef(null);
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [candidateData, setCandidateData] = useState(null);

  // Use the custom certificate manager hook
  const {
    showRight,
    qrVisible,
    qrUrl,
    isUploading,
    driveLink,
    qrPosition,
    qrSize,
    isDragging,
    isResizing,
    handleDownloadLeft,
    handleDownloadRight,
    handleShowSecond,
    handleGenerateQR,
    handleMouseDown,
    handleResizeStart
  } = useCertificateManager("H2S", "H2S_VERIFICATION.pdf");

  // Button state management for Fetch Name functionality
  const [isFetchNameEnabled, setIsFetchNameEnabled] = useState(false);

  // Check candidate data validity
  const checkCandidateData = () => {
    const candidateDataStr = localStorage.getItem('candidateData');
    if (!candidateDataStr) return false;

    try {
      const candidateData = JSON.parse(candidateDataStr);
      const { firstName, lastName, companyName } = candidateData;

      return !!(firstName && firstName.trim() !== '' &&
                lastName && lastName.trim() !== '' &&
                companyName && companyName.trim() !== '');
    } catch (error) {
      return false;
    }
  };

  // Update button state based on candidate data
  const updateButtonState = () => {
    setIsFetchNameEnabled(checkCandidateData());
  };

  // Listen for candidate data changes
  useEffect(() => {
    // Initial check
    updateButtonState();

    // Listen for storage changes
    const handleStorageChange = (e) => {
      if (e.key === 'candidateData') {
        updateButtonState();
      }
    };

    // Listen for custom events (when data is updated from same tab)
    const handleCandidateDataUpdate = () => {
      updateButtonState();
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('candidateDataUpdated', handleCandidateDataUpdate);

    // Periodic check for real-time updates (fallback)
    const interval = setInterval(updateButtonState, 1000);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('candidateDataUpdated', handleCandidateDataUpdate);
      clearInterval(interval);
    };
  }, []);

  // STEP 3: Fetch Name functionality - Extract and save certificate data for financial tracking
  const handleFetchName = async () => {
    // Double-check validation (should not be needed due to button state)
    if (!isFetchNameEnabled) {
      alert('Please complete candidate details form before generating certificate records');
      return;
    }

    // Use candidateData from state (auto-populated from current_candidate_for_certificate.json)
    if (!candidateData) {
      alert('No candidate data found. Please complete the candidate details form first.');
      return;
    }

    // Validate required fields
    if (!candidateData.firstName || !candidateData.lastName) {
      alert('Candidate first name and last name are required. Please complete the candidate details form.');
      return;
    }

    const certificateName = "H2S Training";

    try {
      // STEP 3 & 4: Save certificate data for financial tracking
      const certificateData = {
        firstName: candidateData.firstName,
        lastName: candidateData.lastName,
        certificateName: certificateName,
        // Use company from candidate details (partyName field)
        companyName: candidateData.partyName || "",
        // Get rate data from localStorage for amount calculation
        rateData: JSON.parse(localStorage.getItem('courseRates') || '{}')
      };

      const response = await fetch('http://localhost:5000/save-certificate-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(certificateData),
      });

      if (response.ok) {
        const result = await response.json();
        console.log('[CERTIFICATE] Certificate data saved for receipt processing:', result);

        if (result.duplicate) {
          alert(`⚠️ Certificate Already Exists!\n\nCandidate: ${candidateData.firstName} ${candidateData.lastName}\nCertificate: ${certificateName}\n\nThis certificate has already been added to the receipt processing list.`);
        } else {
          alert(`✅ Certificate Added Successfully!\n\nCandidate: ${candidateData.firstName} ${candidateData.lastName}\nCertificate: ${certificateName}\n\nData saved for receipt processing.\nTotal certificates available: ${result.total_certificates}`);

          // Dispatch events for real-time synchronization
          window.dispatchEvent(new CustomEvent('certificateDataUpdated', {
            detail: {
              type: 'certificate_added',
              data: result.data,
              certificateName: certificateName,
              candidateName: `${candidateData.firstName} ${candidateData.lastName}`
            }
          }));
          window.dispatchEvent(new CustomEvent('dataUpdated', {
            detail: {
              type: 'certificate',
              action: 'added',
              data: result.data
            }
          }));
        }
      } else {
        const error = await response.json();
        console.error('[CERTIFICATE] Failed to save certificate data:', error);
        alert(`❌ Failed to save certificate data: ${error.error}`);
      }
    } catch (error) {
      console.error('[CERTIFICATE] Error saving certificate data:', error);
      alert(`Error saving certificate data: ${error.message}`);
    }
  };

  // Check if H2S course was selected and fetch candidate data
  useEffect(() => {
    const h2sVisited = localStorage.getItem("status_H2S");
    const selectedCourse = localStorage.getItem("selectedCourse");

    if (h2sVisited === "true") {
      setSelectedCourse("H2S");
      console.log(`[COURSE] H2S certificate page loaded. Selected course: ${selectedCourse}`);
      fetchCandidateData();
    }
  }, []);

  const fetchCandidateData = async () => {
    try {
      console.log("[CANDIDATE] Fetching current candidate data for H2S course...");
      const response = await fetch('http://localhost:5000/get-current-candidate-for-certificate');
      if (response.ok) {
        const result = await response.json();
        if (result.status === 'success' && result.data) {
          setCandidateData(result.data);
          console.log("[CANDIDATE] Current candidate data fetched successfully:", result.data);
        } else {
          console.warn("[CANDIDATE] No current candidate data available");
          setCandidateData(null);
        }
      } else if (response.status === 404) {
        console.warn("[CANDIDATE] No current candidate data found for certificate generation");
        setCandidateData(null);
      } else {
        console.error("[CANDIDATE] Failed to fetch current candidate data:", response.status);
        setCandidateData(null);
      }
    } catch (error) {
      console.error('[CANDIDATE] Error fetching current candidate data:', error);
      setCandidateData(null);
    }
  };

  // Text wrapping function - 19 characters per line
  const wrapText = (ctx, text, x, y, maxCharsPerLine = 19, lineHeight = 20) => {
    const lines = [];
    for (let i = 0; i < text.length; i += maxCharsPerLine) {
      lines.push(text.substring(i, i + maxCharsPerLine));
    }

    lines.forEach((line, index) => {
      ctx.fillText(line, x, y + (index * lineHeight));
    });

    return lines.length;
  };

  const drawCandidateTextLeft = (ctx, data) => {
    if (!data) return;

    console.log("[CANVAS] Drawing candidate text on H2S verification LEFT canvas with text wrapping");

    // Configure text style
    ctx.fillStyle = '#000000';
    ctx.font = '16px Arial';
    ctx.textAlign = 'left';

    const fullName = `${data.firstName} ${data.lastName}`;

    // H2S Certificate - Separate coordinates for name and passport
    // Name field coordinates: (310, 155)
    const nameLines = wrapText(ctx, fullName, 310, 155, 19, 20);

    // Passport field coordinates: (310, 210) - separate position, no overlap
    wrapText(ctx, data.passport, 310, 210, 19, 20);

    console.log("[CANVAS] LEFT H2S - Name at (310, 155), Passport at (310, 210)");
    console.log("[CANVAS] LEFT - Text drawn with wrapping - Name:", fullName, "Lines:", nameLines, "Passport:", data.passport);
  };

  const drawCandidateTextRight = (ctx, data) => {
    if (!data) return;

    console.log("[CANVAS] Drawing candidate text on H2S verification RIGHT canvas WITHOUT text wrapping");

    // Configure text style
    ctx.fillStyle = '#000000';
    ctx.font = '16px Arial';
    ctx.textAlign = 'left';

    const fullName = `${data.firstName} ${data.lastName}`;

    // Draw data on right canvas WITHOUT 19-character wrapping
    ctx.fillText(fullName, 180, 260);
    ctx.fillText(data.passport, 340, 300);
    ctx.fillText(data.nationality, 120, 280);
    ctx.fillText(data.dob, 340, 280);
    ctx.fillText(data.cdcNo, 80, 320);

    console.log("[CANVAS] RIGHT H2S - Text drawn without wrapping - Name:", fullName, "Nationality:", data.nationality, "DOB:", data.dob);
  };

  useEffect(() => {
    const leftImage = new Image();
    const rightImage = new Image();

    leftImage.src = '/static/H2S VERIFICATION.jpg';
    rightImage.src = '/static/angelmaritime3.jpg';

    leftImage.onload = () => {
      const ctx = canvasLeftRef.current.getContext('2d');

      // Draw background image
      ctx.drawImage(leftImage, 0, 0, 595, 842);

      // Only draw candidate text if H2S course was selected
      if (selectedCourse && candidateData) {
        drawCandidateTextLeft(ctx, candidateData);
      }
    };

    rightImage.onload = () => {
      const ctx = canvasRightRef.current.getContext('2d');

      // Draw background image
      ctx.drawImage(rightImage, 0, 0, 595, 842);

      // Draw candidate text on right canvas if H2S course was selected
      if (selectedCourse && candidateData) {
        drawCandidateTextRight(ctx, candidateData);
      }
    };
  }, [selectedCourse, candidateData]);



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

      {/* Enhanced Floating Navigation Buttons */}
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
          onClick={() => navigate("/course-preview")}
          whileHover={{ scale: 1.05, y: -2 }}
          whileTap={{ scale: 0.95 }}
          className="group relative bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white p-3 rounded-full shadow-2xl transition-all duration-300"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-gray-500 to-gray-600 rounded-full blur opacity-30 group-hover:opacity-50 transition-opacity duration-300"></div>
          <ArrowLeft size={20} className="relative z-10" />
        </motion.button>


      </motion.div>

      {/* Candidate Data Status */}
      <div className="relative z-10 flex justify-center pt-4 px-4">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="flex flex-col gap-2 items-center"
        >
          {/* Course Information */}
          <div className="px-4 py-2 rounded-lg text-sm font-medium bg-blue-100 text-blue-800 border border-blue-200">
            <span className="flex items-center gap-2">
              📚 Course: H2S Safety Training Certificate
            </span>
          </div>

          {/* Candidate Data Status */}
          <div className={`px-4 py-2 rounded-lg text-sm font-medium ${
            candidateData
              ? 'bg-green-100 text-green-800 border border-green-200'
              : 'bg-yellow-100 text-yellow-800 border border-yellow-200'
          }`}>
            {candidateData ? (
              <span className="flex items-center gap-2">
                ✅ Certificate data loaded: {candidateData.firstName} {candidateData.lastName} ({candidateData.passport})
              </span>
            ) : (
              <span className="flex items-center gap-2">
                ⚠️ No current candidate data - Please submit candidate information first
              </span>
            )}
          </div>
        </motion.div>
      </div>

      {/* Enhanced Control Panel */}
      <div className="relative z-10 flex justify-center pt-4 px-4">

        <motion.div
          className="bg-white/95 backdrop-blur-sm rounded-2xl shadow-2xl p-6 border border-white/20"
          animate={{ x: showRight ? 0 : 0 }}
          transition={{ type: "spring", stiffness: 200, damping: 20 }}
        >
          <div className="flex items-center gap-4 flex-wrap justify-center">
            <div className="relative">
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="px-4 py-3 rounded-xl text-sm bg-white border-2 border-gray-200 hover:border-blue-300 focus:border-blue-500 focus:outline-none transition-colors duration-200 shadow-sm"
              />
            </div>

            <motion.button
              onClick={handleFetchName}
              disabled={!isFetchNameEnabled}
              whileHover={isFetchNameEnabled ? { scale: 1.05 } : {}}
              whileTap={isFetchNameEnabled ? { scale: 0.95 } : {}}
              className={`px-6 py-3 rounded-xl flex items-center gap-2 text-sm font-medium transition-all duration-200 shadow-lg ${
                isFetchNameEnabled
                  ? 'bg-gradient-to-r from-purple-500 to-indigo-500 hover:from-purple-600 hover:to-indigo-600 text-white cursor-pointer'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
              title={isFetchNameEnabled ? 'Generate certificate record' : 'Please complete candidate details first'}
            >
              <User size={16} /> Fetch Name
            </motion.button>

            <motion.button
              onClick={() => handleDownloadLeft(canvasLeftRef)}
              disabled={isUploading}
              whileHover={!isUploading ? { scale: 1.05 } : {}}
              whileTap={!isUploading ? { scale: 0.95 } : {}}
              className={`px-6 py-3 rounded-xl flex items-center gap-2 text-sm font-medium transition-all duration-200 shadow-lg ${
                isUploading
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white"
              }`}
            >
              <Download size={16} /> {isUploading ? "Uploading..." : "Left PDF"}
            </motion.button>

            <motion.button
              onClick={handleShowSecond}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 rounded-xl flex items-center gap-2 text-sm font-medium bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white transition-all duration-200 shadow-lg"
            >
              {showRight ? (
                <>
                  <ChevronLeft size={16} /> Hide
                </>
              ) : (
                <>
                  <ChevronRight size={16} /> Show Second
                </>
              )}
            </motion.button>

            {showRight && (
              <>
                <motion.button
                  onClick={handleGenerateQR}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="px-6 py-3 rounded-xl flex items-center gap-2 text-sm font-medium bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 text-white transition-all duration-200 shadow-lg"
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  <QrCode size={16} /> Generate QR
                </motion.button>

                <motion.button
                  onClick={() => handleDownloadRight(canvasRightRef)}
                  disabled={isUploading}
                  whileHover={!isUploading ? { scale: 1.05 } : {}}
                  whileTap={!isUploading ? { scale: 0.95 } : {}}
                  className={`px-6 py-3 rounded-xl flex items-center gap-2 text-sm font-medium transition-all duration-200 shadow-lg ${
                    isUploading
                      ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                      : "bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white"
                  }`}
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ duration: 0.3, delay: 0.1 }}
                >
                  <Download size={16} /> {isUploading ? "Uploading..." : "Right PDF"}
                </motion.button>
              </>
            )}
          </div>
        </motion.div>
      </div>

      {/* Enhanced Canvas Section */}
      <div className="relative z-10 flex flex-col items-center gap-8 py-12 px-4">
        <motion.div
          className="flex items-center transition-all duration-700"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="relative">
            <canvas
              ref={canvasLeftRef}
              width={595}
              height={842}
              className="border-2 border-white/20 shadow-2xl rounded-lg bg-white"
            ></canvas>
            <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg blur opacity-25"></div>
          </div>

          {/* Enhanced Right canvas + QR */}
          <div
            className={`relative transition-all duration-700 overflow-hidden ${
              showRight
                ? "w-[595px] ml-8 opacity-100 translate-x-0"
                : "w-0 opacity-0 translate-x-10"
            }`}
          >
            <div className="relative">
              <canvas
                ref={canvasRightRef}
                width={595}
                height={842}
                className={`border-2 border-white/20 shadow-2xl rounded-lg bg-white ${showRight ? "block" : "hidden"}`}
              ></canvas>
              <div className="absolute -inset-1 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg blur opacity-25"></div>
            </div>

            {qrVisible && qrUrl && showRight && (
              <motion.div
                className="absolute z-10"
                style={{
                  left: qrPosition.x,
                  top: qrPosition.y,
                  width: qrSize.width,
                  height: qrSize.height,
                }}
                onMouseDown={handleMouseDown}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.3 }}
              >
                <div className="relative w-full h-full">
                  {/* Enhanced Drag handle */}
                  <div className="qr-drag-handle absolute inset-0 cursor-move flex items-center justify-center">
                    <div className="relative">
                      <img
                        src={qrUrl}
                        alt="QR Code"
                        className="w-full h-full object-contain pointer-events-none rounded-lg shadow-lg"
                        draggable={false}
                      />
                      <div className="absolute -inset-1 bg-gradient-to-r from-emerald-400 to-teal-400 rounded-lg blur opacity-30"></div>
                    </div>
                  </div>

                  {/* Enhanced Resize handle */}
                  <div
                    className="resize-handle absolute bottom-0 right-0 w-4 h-4 bg-gradient-to-r from-blue-500 to-cyan-500 opacity-70 hover:opacity-100 transition-opacity cursor-se-resize rounded-tl-lg shadow-lg"
                    onMouseDown={handleResizeStart}
                  />
                </div>
              </motion.div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default DualCertificate3;
