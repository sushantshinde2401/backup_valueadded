# Maritime Certificate Generation System - Technical Documentation

## Table of Contents

1. [Application Overview](#1-application-overview)
2. [Technical Implementation Details](#2-technical-implementation-details)
3. [Feature Documentation](#3-feature-documentation)
4. [API Endpoints](#4-api-endpoints)
5. [Data Flow and Integration](#5-data-flow-and-integration)
6. [Certificate Generation](#6-certificate-generation)
7. [File Structure and Organization](#7-file-structure-and-organization)
8. [Configuration and Setup](#8-configuration-and-setup)
9. [Testing and Verification](#9-testing-and-verification)
10. [Recent Updates and Changes](#10-recent-updates-and-changes)

---

## 1. Application Overview

### 1.1 Purpose and Functionality

The Maritime Certificate Generation System is a comprehensive web application designed to streamline the process of creating, managing, and distributing maritime safety training certificates. The system automates document processing, candidate data management, and certificate generation for various maritime training courses.

**Core Functionalities:**
- Document upload and OCR processing for passport data extraction
- Automated candidate data management with file organization
- Payment verification through screenshot uploads
- Multi-course certificate generation (STCW, H2S, STSDSD, BOSIET)
- PDF generation with Google Drive integration
- QR code generation for certificate verification

### 1.2 Target Users and Use Cases

**Primary Users:**
- **Maritime Training Centers**: Course administrators managing candidate certifications
- **Training Coordinators**: Staff processing candidate applications and documents
- **Candidates**: Maritime professionals seeking safety training certificates
- **Verification Personnel**: Authorities validating certificate authenticity

**Use Cases:**
- Bulk processing of candidate applications with document uploads
- Automated extraction of passport information using OCR technology
- Generation of standardized certificates for multiple maritime courses
- Digital distribution and verification of certificates
- Compliance tracking for maritime safety training requirements

### 1.3 System Architecture Overview

The system follows a modern web application architecture with clear separation of concerns:

```
Frontend (React.js) ↔ Backend (Flask/Python) ↔ File Storage System
                                ↓
                        Google Drive Integration
                                ↓
                        Certificate Distribution
```

**Architecture Components:**
- **Frontend**: React.js single-page application with component-based architecture
- **Backend**: Flask REST API server with Python-based processing
- **Storage**: File-based system with organized directory structure
- **Processing**: OCR integration for document analysis
- **Integration**: Google Drive API for certificate distribution

---

## 2. Technical Implementation Details

### 2.1 Frontend Technology Stack

**Core Technologies:**
- **React.js 18+**: Component-based UI framework
- **React Router**: Client-side routing and navigation
- **Framer Motion**: Animation and transition library
- **Tailwind CSS**: Utility-first CSS framework
- **HTML5 Canvas**: Certificate rendering and generation

**Key Components:**
```javascript
src/
├── operations/
│   ├── pages/
│   │   ├── certificates/          // Certificate generation pages
│   │   ├── CoursePreview.js       // Course selection interface
│   │   └── UploadDocx.js          // Document upload interface
│   └── components/                // Reusable UI components
```

**State Management:**
- React Hooks (useState, useEffect) for local component state
- LocalStorage for session persistence and course selection tracking
- Context API for global application state where needed

### 2.2 Backend Technology Stack

**Core Technologies:**
- **Flask**: Lightweight Python web framework
- **Python 3.8+**: Core programming language
- **OpenCV**: Image processing for document analysis
- **Pytesseract**: OCR engine for text extraction
- **Pillow**: Image manipulation and processing
- **Requests**: HTTP client for external API integration

**Key Modules:**
```python
backend/
├── app.py                    // Main Flask application
├── shared/
│   └── utils.py             // Utility functions
└── uploads/                 // File storage system
    ├── temp/               // Temporary session storage
    ├── images/             // Organized candidate folders
    ├── json/               // Data storage and OCR results
    └── pdfs/               // Generated certificates
```

### 2.3 File Organization and Storage System

**Storage Architecture:**
The system uses a file-based storage approach with organized directory structures for different types of content:

```
backend/uploads/
├── temp/                           // Temporary session storage
│   └── {sessionId}/               // Individual session folders
├── images/                         // Permanent candidate storage
│   └── {firstName}_{lastName}_{passport}/  // Candidate-specific folders
├── json/                          // Data and configuration files
│   ├── current_candidate_for_certificate.json  // Active candidate data
│   └── structured_passport_data_{sessionId}.json  // OCR results
└── pdfs/                          // Generated certificate files
```

### 2.4 Database Structure (Simplified JSON System)

**Current Candidate Data Structure:**
```json
{
  "firstName": "John",
  "lastName": "Smith",
  "passport": "JS123456",
  "nationality": "US",
  "dob": "1985-03-15",
  "cdcNo": "CDC789012",
  "address": "123 Maritime Street",
  "email": "john.smith@maritime.com",
  "phone": "1234567890",
  "companyName": "Ocean Shipping Ltd",
  "vendorName": "Maritime Training Center",
  "paymentStatus": "PAID",
  "rollNo": "MS001",
  "timestamp": "20250814_164102",
  "last_updated": "2025-08-14T16:41:02.293556",
  "candidate_folder": "John_Smith_JS123456",
  "moved_files": ["photo.jpg", "signature.png", "passport_front.jpg"],
  "session_id": "session-12345"
}
```

---

## 3. Feature Documentation

### 3.1 Document Upload Workflow

**Process Flow:**
1. **Session Creation**: Unique session ID generated for each upload session
2. **Temporary Storage**: Files uploaded to `temp/{sessionId}/` directory
3. **File Validation**: File type and size validation on upload
4. **OCR Processing**: Automatic passport data extraction using Tesseract
5. **Data Structuring**: Extracted data formatted into structured JSON
6. **Form Pre-filling**: OCR results used to populate candidate form

**Supported File Types:**
- Images: JPG, PNG, GIF, BMP
- Documents: PDF
- Maximum file size: 10MB per file

**OCR Processing:**
```python
# OCR extraction process
def extract_passport_data(image_path):
    image = cv2.imread(image_path)
    processed_image = preprocess_image(image)
    text = pytesseract.image_to_string(processed_image)
    structured_data = parse_passport_text(text)
    return structured_data
```

### 3.2 Payment Screenshot Handling

**Payment Verification Process:**
- Automatic detection of payment status selection
- Conditional screenshot upload requirement for "PAID" status
- File validation and storage in candidate session folder
- Integration with candidate data for verification tracking

**Implementation:**
```javascript
// Payment status handling
const handlePaymentStatusChange = (status) => {
  setPaymentStatus(status);
  if (status === 'PAID') {
    setShowPaymentUpload(true);
  } else {
    setShowPaymentUpload(false);
  }
};
```

### 3.3 Course Selection and Certificate Generation

**Course Types Supported:**
1. **STCW** - Basic Safety Training Certificate
2. **H2S** - Hydrogen Sulfide Safety Training
3. **STSDSD** - Ship-to-Ship Transfer Safety
4. **BOSIET** - Basic Offshore Safety Induction

**Selection Workflow:**
```javascript
const handleCourseClick = (course) => {
  localStorage.setItem(`status_${course}`, "true");
  localStorage.setItem("selectedCourse", course);
  localStorage.setItem("selectedCourseTimestamp", new Date().toISOString());
  navigate(certificateMap[course]);
};
```

### 3.4 File Organization with Conflict Resolution

**Folder Naming Convention:**
- Primary: `{firstName}_{lastName}_{passportNo}`
- Conflicts: `{firstName}_{lastName}_{passportNo}_1`, `_2`, etc.

**Conflict Resolution Algorithm:**
```python
def create_unique_candidate_folder(base_path, folder_name):
    original_name = folder_name
    counter = 1
    
    while os.path.exists(os.path.join(base_path, folder_name)):
        folder_name = f"{original_name}_{counter}"
        counter += 1
    
    folder_path = os.path.join(base_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path, folder_name
```

---

## 4. API Endpoints

### 4.1 Core Endpoints

#### Document Upload Endpoints

**POST /upload-images**
- **Purpose**: Upload multiple images to temporary session
- **Request**: Multipart form data with files and session ID
- **Response**: 
```json
{
  "status": "success",
  "session_id": "session-12345",
  "uploaded_files": ["photo.jpg", "passport.jpg"],
  "files_count": 2
}
```

**POST /upload-payment-screenshot**
- **Purpose**: Upload payment verification screenshot
- **Request**: Multipart form data with payment screenshot
- **Response**: File upload confirmation with storage location

#### Data Management Endpoints

**POST /save-candidate-data**
- **Purpose**: Save candidate form data and organize files
- **Request**: JSON payload with candidate information
- **Response**:
```json
{
  "status": "success",
  "message": "Candidate data saved and files organized successfully",
  "candidate_folder": "John_Smith_JS123456",
  "moved_files": ["photo.jpg", "signature.png"],
  "files_count": 7
}
```

**GET /get-current-candidate-for-certificate**
- **Purpose**: Retrieve current candidate data for certificate generation
- **Response**:
```json
{
  "status": "success",
  "data": {
    "firstName": "John",
    "lastName": "Smith",
    "passport": "JS123456",
    "nationality": "US",
    "dob": "1985-03-15",
    "cdcNo": "CDC789012"
  },
  "last_updated": "2025-08-14T16:41:02.293556"
}
```

#### Certificate Generation Endpoints

**POST /save-pdf**
- **Purpose**: Generate PDF and upload to Google Drive
- **Request**: Base64 encoded PDF data
- **Response**: Google Drive upload confirmation with shareable link

**POST /save-right-pdf**
- **Purpose**: Save PDF locally without Drive upload
- **Request**: Base64 encoded PDF data
- **Response**: Local file save confirmation

### 4.2 Utility Endpoints

**GET /**
- **Purpose**: Health check and system status
- **Response**: Server status and available endpoints list

**POST /cleanup-expired-sessions**
- **Purpose**: Clean up old temporary session folders
- **Response**: Cleanup operation results

**GET /list-files**
- **Purpose**: List all files in the system
- **Response**: Comprehensive file listing by category

### 4.3 Error Handling and Status Codes

**Standard HTTP Status Codes:**
- `200 OK`: Successful operation
- `400 Bad Request`: Invalid request data or missing parameters
- `404 Not Found`: Resource not found (e.g., no current candidate data)
- `500 Internal Server Error`: Server-side processing error

**Error Response Format:**
```json
{
  "error": "Descriptive error message",
  "status": "error",
  "details": "Additional error context"
}
```

---

## 5. Data Flow and Integration

### 5.1 Complete Workflow Overview

```
1. Document Upload → 2. OCR Processing → 3. Form Completion → 
4. Data Submission → 5. Course Selection → 6. Certificate Generation → 
7. PDF Creation → 8. Drive Upload → 9. QR Code Generation
```

### 5.2 Step-by-Step Process

**Step 1: Document Upload**
```javascript
// Frontend: File upload with session management
const uploadFiles = async (files, sessionId) => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  formData.append('session_id', sessionId);
  
  const response = await fetch('/upload-images', {
    method: 'POST',
    body: formData
  });
  return response.json();
};
```

**Step 2: OCR Processing**
```python
# Backend: Automatic passport data extraction
@app.route('/upload-images', methods=['POST'])
def upload_images():
    # File upload processing
    # OCR extraction
    # Data structuring
    return jsonify({"status": "success", "ocr_data": extracted_data})
```

**Step 3: Form Completion and Submission**
```javascript
// Frontend: Form submission with candidate data
const submitCandidateData = async (formData) => {
  const response = await fetch('/save-candidate-data', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(formData)
  });
  return response.json();
};
```

**Step 4: Course Selection Integration**
```javascript
// Course selection triggers certificate page navigation
const selectCourse = (courseType) => {
  localStorage.setItem('selectedCourse', courseType);
  navigate(`/certificate-${courseType.toLowerCase()}`);
};
```

**Step 5: Certificate Generation**
```javascript
// Certificate page automatically fetches current candidate data
useEffect(() => {
  fetchCurrentCandidateData();
}, []);

const fetchCurrentCandidateData = async () => {
  const response = await fetch('/get-current-candidate-for-certificate');
  const data = await response.json();
  setCandidateData(data.data);
};
```

### 5.3 Current Candidate Data System

**Single Source of Truth:**
- File: `current_candidate_for_certificate.json`
- Purpose: Contains most recently submitted candidate data
- Behavior: Overwrites with each new submission
- Access: Via `/get-current-candidate-for-certificate` endpoint

**Data Synchronization:**
```python
# Backend: Update current candidate data
def save_candidate_data():
    # Process candidate data
    # Save to current_candidate_for_certificate.json
    # Organize files into candidate folder
    return success_response
```

---

## 6. Certificate Generation

### 6.1 Canvas-Based Certificate Creation

**Technology Stack:**
- HTML5 Canvas for certificate rendering
- JavaScript for dynamic content generation
- Base64 encoding for PDF conversion

**Certificate Generation Process:**
```javascript
const generateCertificate = (candidateData) => {
  const canvas = document.getElementById('certificateCanvas');
  const ctx = canvas.getContext('2d');
  
  // Load certificate template
  const templateImage = new Image();
  templateImage.onload = () => {
    ctx.drawImage(templateImage, 0, 0);
    drawCandidateData(ctx, candidateData);
  };
  templateImage.src = certificateTemplate;
};
```

### 6.2 Field Mapping and Positioning

**Certificate Fields (5 fields displayed):**
1. **Full Name**: `firstName + lastName` at position (180, 260)
2. **Passport**: `passport` at position (340, 300)
3. **Nationality**: `nationality` at position (120, 280)
4. **Date of Birth**: `dob` at position (340, 280)
5. **CDC No.**: `cdcNo` at position (80, 320)

**Canvas Drawing Implementation:**
```javascript
const drawCandidateData = (ctx, data) => {
  const fullName = `${data.firstName} ${data.lastName}`;
  
  ctx.font = '16px Arial';
  ctx.fillStyle = '#000000';
  
  ctx.fillText(fullName, 180, 260);
  ctx.fillText(data.passport, 340, 300);
  ctx.fillText(data.nationality, 120, 280);
  ctx.fillText(data.dob, 340, 280);
  ctx.fillText(data.cdcNo, 80, 320);
};
```

### 6.3 Certificate Types and Templates

**STCW Basic Safety Training Certificate**
- File: `DualCertificate.jsx`
- Template: STCW-specific design
- Fields: Standard 5-field layout

**H2S Safety Training Certificate**
- File: `DualCertificate3.jsx`
- Template: H2S-specific design
- Fields: Standard 5-field layout

**STSDSD Verification Certificate**
- File: `DualCertificate2.jsx`
- Template: STSDSD-specific design
- Fields: Standard 5-field layout

**BOSIET Safety Training Certificate**
- File: `DualCertificate4.jsx`
- Template: BOSIET-specific design
- Fields: Standard 5-field layout

### 6.4 PDF Generation and Google Drive Integration

**PDF Creation Process:**
```javascript
const generatePDF = async (canvasElement) => {
  const imageData = canvasElement.toDataURL('image/png');
  const pdfData = await convertToPDF(imageData);
  return pdfData;
};
```

**Google Drive Upload:**
```python
# Backend: Google Drive integration
@app.route('/save-pdf', methods=['POST'])
def save_pdf():
    # Decode base64 PDF data
    # Upload to Google Drive
    # Generate shareable link
    # Return drive file information
    return jsonify({"drive_link": shareable_url})
```

---

## 7. File Structure and Organization

### 7.1 Project Directory Structure

```
VALUE_ADDED_2.0/
├── src/                           # Frontend React application
│   ├── operations/
│   │   ├── pages/
│   │   │   ├── certificates/      # Certificate generation pages
│   │   │   │   ├── DualCertificate.jsx    # STCW certificates
│   │   │   │   ├── DualCertificate2.jsx   # STSDSD certificates
│   │   │   │   ├── DualCertificate3.jsx   # H2S certificates
│   │   │   │   └── DualCertificate4.jsx   # BOSIET certificates
│   │   │   ├── CoursePreview.js   # Course selection interface
│   │   │   └── UploadDocx.js      # Document upload interface
│   │   └── components/            # Reusable UI components
│   ├── App.js                     # Main application component
│   └── index.js                   # Application entry point
├── backend/                       # Flask backend application
│   ├── app.py                     # Main Flask server
│   ├── shared/
│   │   └── utils.py              # Utility functions
│   ├── uploads/                   # File storage system
│   │   ├── temp/                 # Temporary session storage
│   │   ├── images/               # Organized candidate folders
│   │   ├── json/                 # Data storage and OCR results
│   │   └── pdfs/                 # Generated certificates
│   └── test_*.py                 # Test scripts
├── public/                        # Static assets
└── package.json                   # Node.js dependencies
```

### 7.2 File Naming Conventions

**Candidate Folders:**
- Format: `{firstName}_{lastName}_{passportNo}`
- Example: `John_Smith_JS123456`
- Conflicts: `John_Smith_JS123456_1`, `John_Smith_JS123456_2`

**Session Folders:**
- Format: `{sessionId}`
- Example: `session-1692123456789`
- Location: `backend/uploads/temp/`

**Data Files:**
- Current candidate: `current_candidate_for_certificate.json`
- OCR results: `structured_passport_data_{sessionId}.json`
- Generated PDFs: `certificate_{timestamp}.pdf`

### 7.3 Storage Locations

**Temporary Storage:**
```
backend/uploads/temp/
└── {sessionId}/
    ├── photo.jpg
    ├── signature.png
    ├── passport_front.jpg
    ├── passport_back.jpg
    ├── cdc.jpg
    ├── marksheet.pdf
    └── payment_screenshot.jpg
```

**Permanent Storage:**
```
backend/uploads/images/
└── {candidateName}_{passport}/
    ├── photo.jpg
    ├── signature.png
    ├── passport_front.jpg
    ├── passport_back.jpg
    ├── cdc.jpg
    ├── marksheet.pdf
    └── payment_screenshot.jpg
```

**Data Storage:**
```
backend/uploads/json/
├── current_candidate_for_certificate.json
└── structured_passport_data_{sessionId}.json
```

---

## 8. Configuration and Setup

### 8.1 Installation Requirements

**System Requirements:**
- Node.js 16+ for frontend development
- Python 3.8+ for backend services
- Tesseract OCR engine for document processing
- OpenCV for image processing

**Frontend Dependencies:**
```json
{
  "dependencies": {
    "react": "^18.0.0",
    "react-router-dom": "^6.0.0",
    "framer-motion": "^6.0.0",
    "tailwindcss": "^3.0.0"
  }
}
```

**Backend Dependencies:**
```python
# requirements.txt
Flask==2.3.0
opencv-python==4.8.0
pytesseract==0.3.10
Pillow==10.0.0
requests==2.31.0
numpy==1.24.0
```

### 8.2 Environment Setup Instructions

**Frontend Setup:**
```bash
# Navigate to project root
cd VALUE_ADDED_2.0

# Install dependencies
npm install

# Start development server
npm start
```

**Backend Setup:**
```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Install Tesseract OCR
# Windows: Download from GitHub releases
# macOS: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr

# Start Flask server
python app.py
```

### 8.3 Configuration Parameters

**Flask Configuration:**
```python
# app.py configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'pdf'}
```

**OCR Configuration:**
```python
# Tesseract configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Linux/macOS
```

**Google Drive API Configuration:**
```python
# Google Drive integration setup
GOOGLE_DRIVE_CREDENTIALS = 'path/to/credentials.json'
DRIVE_FOLDER_ID = 'your-drive-folder-id'
```

---

## 9. Testing and Verification

### 9.1 Test Scripts and Purposes

**Integration Tests:**
- `test_course_certificate_integration.py`: Complete workflow testing
- `test_simplified_system.py`: Simplified data system verification
- `test_certificate_fields.py`: Certificate field validation

**Test Coverage:**
```python
# Example test execution
def test_complete_workflow():
    # 1. Submit candidate data
    # 2. Verify data storage
    # 3. Test course selection
    # 4. Validate certificate generation
    # 5. Check file organization
    pass
```

### 9.2 Integration Testing Procedures

**Automated Test Execution:**
```bash
# Run all integration tests
cd backend
python test_course_certificate_integration.py
python test_simplified_system.py
python test_certificate_fields.py
```

**Manual Testing Checklist:**
1. Document upload functionality
2. OCR data extraction accuracy
3. Form pre-filling from OCR results
4. Payment screenshot upload for paid candidates
5. Course selection and navigation
6. Certificate generation with correct data
7. PDF creation and download
8. Google Drive upload functionality

### 9.3 Field Validation and Error Handling

**Data Validation Rules:**
```javascript
const validateCandidateData = (data) => {
  const required = ['firstName', 'lastName', 'passport', 'nationality', 'dob', 'cdcNo'];
  const missing = required.filter(field => !data[field]);
  
  if (missing.length > 0) {
    throw new Error(`Missing required fields: ${missing.join(', ')}`);
  }
  
  return true;
};
```

**Error Handling Strategies:**
- Graceful degradation for OCR failures
- User-friendly error messages for validation failures
- Automatic retry mechanisms for network issues
- Comprehensive logging for debugging

---

## 10. Recent Updates and Changes

### 10.1 Simplification from Dual Storage to Single JSON System

**Previous System (Dual Storage):**
- Centralized database (`candidates_database.json`)
- Individual candidate files (`candidate_data.json`, `candidate_data.txt`)
- Complex atomic operations with rollback mechanisms
- File locking for concurrent access

**Current System (Simplified):**
- Single dynamic file (`current_candidate_for_certificate.json`)
- Overwrite behavior for new submissions
- Simplified API endpoints
- Reduced complexity and maintenance overhead

**Migration Benefits:**
- Reduced system complexity
- Faster data access
- Simplified error handling
- Easier maintenance and debugging

### 10.2 Certificate Field Updates

**Previous Fields (6 fields):**
1. Full Name
2. Passport
3. CDC No.
4. INDOS No. *(removed)*
5. Company *(removed)*
6. Roll No. *(removed)*

**Current Fields (5 fields):**
1. Full Name (`firstName + lastName`)
2. Passport (`passport`)
3. Nationality (`nationality`) *(new)*
4. Date of Birth (`dob`) *(new)*
5. CDC No. (`cdcNo`)

**Updated Canvas Positioning:**
```javascript
// New positioning for all certificate types
ctx.fillText(fullName, 180, 260);        // Full Name
ctx.fillText(data.passport, 340, 300);   // Passport
ctx.fillText(data.nationality, 120, 280); // Nationality
ctx.fillText(data.dob, 340, 280);        // Date of Birth
ctx.fillText(data.cdcNo, 80, 320);       // CDC No.
```

### 10.3 Course Selection Integration Improvements

**Enhanced Features:**
- Automatic course information storage in localStorage
- Real-time candidate data status indicators
- Improved error handling for missing candidate data
- Visual feedback for data availability

**Integration Workflow:**
```javascript
// Enhanced course selection with data integration
const handleCourseClick = (course) => {
  localStorage.setItem(`status_${course}`, "true");
  localStorage.setItem("selectedCourse", course);
  localStorage.setItem("selectedCourseTimestamp", new Date().toISOString());
  navigate(certificateMap[course]);
};
```

**Status Indicators:**
- Green indicator: Candidate data loaded successfully
- Yellow warning: No current candidate data available
- Course information display on certificate pages

---

## Conclusion

This Maritime Certificate Generation System provides a comprehensive solution for maritime training certificate management. The system's modular architecture, simplified data management, and robust error handling make it suitable for production use in maritime training environments.

The recent updates have significantly improved system maintainability while preserving all core functionality. The simplified JSON-based data system and updated certificate field structure provide a clean, efficient foundation for certificate generation across multiple maritime training courses.

For technical support or additional information, refer to the test scripts and integration documentation provided with the system.

---

## Appendices

### Appendix A: Complete API Reference

#### A.1 Request/Response Examples

**Upload Images Example:**
```bash
curl -X POST http://localhost:5000/upload-images \
  -F "files=@photo.jpg" \
  -F "files=@passport.jpg" \
  -F "session_id=session-12345"
```

**Save Candidate Data Example:**
```bash
curl -X POST http://localhost:5000/save-candidate-data \
  -H "Content-Type: application/json" \
  -d '{
    "firstName": "John",
    "lastName": "Smith",
    "passport": "JS123456",
    "nationality": "US",
    "dob": "1985-03-15",
    "cdcNo": "CDC789012",
    "paymentStatus": "PAID",
    "session_id": "session-12345"
  }'
```

#### A.2 Error Response Examples

**Missing Required Fields:**
```json
{
  "error": "firstName, lastName, and passport are required for file organization",
  "status": "error"
}
```

**No Current Candidate Data:**
```json
{
  "error": "No current candidate data found for certificate generation",
  "status": "error"
}
```

### Appendix B: Troubleshooting Guide

#### B.1 Common Issues and Solutions

**Issue: OCR Not Working**
- **Cause**: Tesseract not installed or incorrect path
- **Solution**: Install Tesseract and update path in configuration
- **Verification**: Run `tesseract --version` in terminal

**Issue: File Upload Fails**
- **Cause**: File size exceeds limit or unsupported format
- **Solution**: Check file size (max 16MB) and format (JPG, PNG, PDF)
- **Verification**: Check browser console for error messages

**Issue: Certificate Generation Fails**
- **Cause**: No current candidate data available
- **Solution**: Submit candidate data before generating certificates
- **Verification**: Check `/get-current-candidate-for-certificate` endpoint

#### B.2 Debug Mode Configuration

**Enable Debug Logging:**
```python
# app.py
app.config['DEBUG'] = True
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Frontend Debug Mode:**
```javascript
// Enable React development mode
const DEBUG = process.env.NODE_ENV === 'development';
if (DEBUG) {
  console.log('Debug mode enabled');
}
```

### Appendix C: Performance Optimization

#### C.1 File Processing Optimization

**Image Compression:**
```python
def optimize_image(image_path, max_size=(1920, 1080)):
    with Image.open(image_path) as img:
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        img.save(image_path, optimize=True, quality=85)
```

**Batch Processing:**
```python
def process_multiple_files(file_list):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, file) for file in file_list]
        results = [future.result() for future in futures]
    return results
```

#### C.2 Frontend Performance

**Component Optimization:**
```javascript
// Use React.memo for expensive components
const CertificateCanvas = React.memo(({ candidateData }) => {
  // Canvas rendering logic
});

// Lazy loading for certificate pages
const DualCertificate = lazy(() => import('./certificates/DualCertificate'));
```

### Appendix D: Security Considerations

#### D.1 File Upload Security

**File Validation:**
```python
def validate_file(file):
    # Check file extension
    if not allowed_file(file.filename):
        raise ValueError("File type not allowed")

    # Check file size
    if len(file.read()) > MAX_FILE_SIZE:
        raise ValueError("File too large")

    # Reset file pointer
    file.seek(0)
    return True
```

**Path Sanitization:**
```python
def sanitize_folder_name(name):
    # Remove dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Limit length
    return sanitized[:50]
```

#### D.2 Data Protection

**Sensitive Data Handling:**
- Passport numbers and personal information encrypted at rest
- Session-based temporary storage with automatic cleanup
- No persistent storage of payment information

### Appendix E: Deployment Guide

#### E.1 Production Deployment

**Docker Configuration:**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install Tesseract
RUN apt-get update && apt-get install -y tesseract-ocr

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

**Environment Variables:**
```bash
# .env file
FLASK_ENV=production
GOOGLE_DRIVE_CREDENTIALS_PATH=/path/to/credentials.json
UPLOAD_FOLDER=/app/uploads
MAX_FILE_SIZE=16777216
```

#### E.2 Monitoring and Logging

**Application Monitoring:**
```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000000, backupCount=3)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
```

**Health Check Endpoint:**
```python
@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0",
        "services": {
            "ocr": check_tesseract_availability(),
            "storage": check_storage_availability(),
            "drive": check_drive_connectivity()
        }
    })
```

---

*Document Version: 1.0*
*Last Updated: August 14, 2025*
*System Version: VALUE_ADDED_2.0*
*Total Pages: 45*
