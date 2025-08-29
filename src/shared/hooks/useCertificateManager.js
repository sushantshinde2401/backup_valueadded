import { useState, useEffect } from 'react';
import jsPDF from 'jspdf';

export const useCertificateManager = (courseName, pdfFileName) => {
  // State management
  const [showRight, setShowRight] = useState(false);
  const [qrVisible, setQrVisible] = useState(false);
  const [qrUrl, setQrUrl] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [driveLink, setDriveLink] = useState("");
  const [qrPosition, setQrPosition] = useState({ x: 50, y: 50 });
  const [qrSize, setQrSize] = useState({ width: 120, height: 120 });
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [resizeStart, setResizeStart] = useState({
    x: 0,
    y: 0,
    width: 0,
    height: 0,
  });

  // PDF Generation and Upload Function
  const handleDownloadLeft = async (canvasRef) => {
    try {
      setIsUploading(true);
      const canvas = canvasRef.current;

      // Convert canvas to image data
      const imgData = canvas.toDataURL("image/jpeg", 0.95);

      // Create PDF from canvas
      const pdf = new jsPDF({ unit: "pt", format: "a4" });
      pdf.addImage(imgData, "JPEG", 0, 0, 595.28, 841.89);

      // Download PDF locally first - COMMENTED OUT TO SAVE STORAGE
      // pdf.save(pdfFileName);

      // Convert to blob for upload
      const pdfBlob = pdf.output("blob");

      // Upload to Google Drive
      const formData = new FormData();
      formData.append("pdf", pdfBlob, pdfFileName);

      const response = await fetch("http://localhost:5000/upload", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setDriveLink(data.drive_link);
        setQrUrl(`data:image/png;base64,${data.qr_image}`);

        // Log the actual Drive link for reference
        console.log("Google Drive Link:", data.drive_link);

        if (data.storage_type === "google_drive") {
          alert(
            `PDF downloaded and uploaded to Google Drive!\nDrive Link: ${data.drive_link}\n\nQR code will work on any device.`
          );
        } else {
          alert(
            `PDF downloaded and saved locally.\nWarning: QR code may only work on this network.\n\n${
              data.warning || ""
            }`
          );
        }
      } else {
        alert(`Upload failed: ${data.error || "Unknown error"}`);
      }
    } catch (error) {
      console.error("Error in handleDownloadLeft:", error);
      console.error("Full error details:", error);
      alert(
        `Error: ${error.message}\n\nCheck browser console for more details.`
      );
    } finally {
      setIsUploading(false);
    }
  };

  // Show/Hide second canvas
  const handleShowSecond = () => {
    setShowRight(!showRight);
  };

  // Generate QR code
  const handleGenerateQR = () => {
    if (qrUrl && showRight) {
      setQrVisible(true);
    } else if (!qrUrl) {
      alert(
        "QR not available. Please click 'Left PDF' first to generate QR code."
      );
    } else if (!showRight) {
      alert("Please click 'Show Second' to display the second image first.");
    }
  };



  // Right Canvas PDF Generation with QR Code and Backend Save
  const handleDownloadRight = async (canvasRef) => {
    if (!showRight) {
      alert("Please click 'Show Second' to display the right canvas first.");
      return;
    }

    try {
      setIsUploading(true);
      const canvas = canvasRef.current;

      // Create a temporary canvas to combine the certificate and QR code
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = 595;
      tempCanvas.height = 842;
      const tempCtx = tempCanvas.getContext("2d");

      // FIRST: Draw the complete right canvas content (includes background + text already drawn)
      tempCtx.drawImage(canvas, 0, 0, 595, 842);
      console.log("Complete right canvas content copied (background + existing text)");

      // Note: No need to redraw candidate text as it's already on the canvas

      console.log("Canvas drawn, QR visible:", qrVisible, "QR URL exists:", !!qrUrl);

      // Debug: Check what's on the canvas before QR
      const debugImg = tempCanvas.toDataURL("image/jpeg", 0.95);
      console.log("Canvas content before QR (first 100 chars):", debugImg.substring(0, 100));

      // Function to generate and save PDF
      const generateAndSavePDF = async (finalCanvas) => {
        console.log("Generating PDF from canvas...");
        console.log("Final canvas dimensions:", finalCanvas.width, "x", finalCanvas.height);

        // Convert canvas to image data
        const imgData = finalCanvas.toDataURL("image/jpeg", 0.95);
        console.log("Canvas converted to image data, length:", imgData.length);
        console.log("Image data preview:", imgData.substring(0, 100));

        // Create PDF from canvas
        const pdf = new jsPDF({ unit: "pt", format: "a4" });
        pdf.addImage(imgData, "JPEG", 0, 0, 595.28, 841.89);
        console.log("PDF created with image");

        // Convert to blob for backend save
        const pdfBlob = pdf.output("blob");
        console.log("PDF blob size:", pdfBlob.size);

        // Create filename for right PDF (Angel Maritime)
        const rightPdfName = pdfFileName.replace("_VERIFICATION.pdf", "_ANGELMARITIME.pdf");
        console.log("Saving as:", rightPdfName);

        // Save to backend only (no Google Drive, no QR generation)
        const formData = new FormData();
        formData.append("pdf", pdfBlob, rightPdfName);
        formData.append("filename", rightPdfName);

        const response = await fetch("http://localhost:5000/save-right-pdf", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (data.success) {
          console.log("Right PDF saved to backend:", data.local_link);
          alert(`Right PDF saved to backend successfully!\nFile: ${rightPdfName}\nIncludes: Certificate + Candidate Text + QR Code`);
        } else {
          console.error("Save failed:", data);
          alert(`Right PDF save failed: ${data.error || "Unknown error"}`);
        }
      };

      // THIRD: If QR is visible, draw it ON TOP of the certificate + text
      if (qrVisible && qrUrl) {
        const qrImg = new Image();
        qrImg.onload = async () => {
          console.log("QR Image loaded successfully");
          console.log("Drawing QR at position:", qrPosition, "size:", qrSize);
          console.log("Canvas dimensions:", tempCanvas.width, "x", tempCanvas.height);

          // Ensure QR position is within canvas bounds
          const safeX = Math.max(0, Math.min(qrPosition.x, tempCanvas.width - qrSize.width));
          const safeY = Math.max(0, Math.min(qrPosition.y, tempCanvas.height - qrSize.height));

          console.log("Safe QR position:", safeX, safeY);

          // Draw QR code ON TOP of the certificate + text at user-positioned location
          tempCtx.drawImage(
            qrImg,
            safeX,
            safeY,
            qrSize.width,
            qrSize.height
          );

          console.log("QR code drawn successfully");

          // Generate and save PDF with certificate + text + QR
          await generateAndSavePDF(tempCanvas);
        };

        qrImg.onerror = () => {
          console.error("Failed to load QR image");
          // Save without QR if image fails to load
          generateAndSavePDF(tempCanvas);
        };

        qrImg.src = qrUrl;
      } else {
        // No QR code visible, just save the certificate + text
        console.log("No QR code, saving certificate + text only");
        await generateAndSavePDF(tempCanvas);
      }
    } catch (error) {
      console.error("Error in handleDownloadRight:", error);
      alert(
        `Error: ${error.message}\n\nCheck browser console for more details.`
      );
    } finally {
      setIsUploading(false);
    }
  };

  // Custom drag handlers
  const handleMouseDown = (e) => {
    // Check if we're clicking on the resize handle
    if (e.target.classList.contains("resize-handle")) {
      return; // Let resize handle take precedence
    }

    // Start dragging for any click on the QR container
    setIsDragging(true);
    setDragStart({
      x: e.clientX - qrPosition.x,
      y: e.clientY - qrPosition.y,
    });
    e.preventDefault();
    e.stopPropagation();
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setQrPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
    if (isResizing) {
      const newWidth = Math.max(
        60,
        Math.min(250, resizeStart.width + (e.clientX - resizeStart.x))
      );
      const newHeight = Math.max(
        60,
        Math.min(250, resizeStart.height + (e.clientY - resizeStart.y))
      );
      setQrSize({ width: newWidth, height: newWidth }); // Keep it square
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsResizing(false);
  };

  const handleResizeStart = (e) => {
    setIsResizing(true);
    setResizeStart({
      x: e.clientX,
      y: e.clientY,
      width: qrSize.width,
      height: qrSize.height,
    });
    e.preventDefault();
    e.stopPropagation();
  };

  // Add global mouse event listeners
  useEffect(() => {
    if (isDragging || isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, isResizing, dragStart, resizeStart]);

  return {
    // State
    showRight,
    qrVisible,
    qrUrl,
    isUploading,
    driveLink,
    qrPosition,
    qrSize,
    isDragging,
    isResizing,
    
    // Functions
    handleDownloadLeft,
    handleDownloadRight,
    handleShowSecond,
    handleGenerateQR,
    handleMouseDown,
    handleResizeStart,
    
    // Setters (if needed for external control)
    setShowRight,
    setQrVisible,
    setQrUrl
  };
};
