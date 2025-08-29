# Operations Section

This section handles all operational activities related to certificate management and candidate processing.

## Structure

```
operations/
├── components/          # Reusable components specific to operations
├── pages/              # Main operation pages
│   ├── HomePage.jsx           # Main dashboard
│   ├── UploadDocx.jsx         # Document upload and OCR
│   ├── CandidateDetail.jsx    # Candidate information form
│   ├── CourseSelection.jsx    # Course selection interface
│   ├── CoursePreview.js       # Course preview and management
│   └── certificates/          # Certificate generation pages
│       ├── DualCertificate.jsx    # STCW certificates
│       ├── DualCertificate2.jsx   # MODU certificates
│       ├── DualCertificate3.jsx   # H2S certificates
│       └── DualCertificate4.jsx   # BOSIET certificates
├── services/           # API services for operations
└── README.md          # This file
```

## Features

### Document Processing
- Multi-file upload (photo, signature, passport, CDC documents)
- OCR processing for passport and CDC data extraction
- Image validation and preprocessing

### Candidate Management
- Candidate data entry and validation
- Form auto-completion from OCR data
- Data persistence and retrieval

### Course Management
- Course selection interface
- Course preview and status tracking
- Certificate type mapping

### Certificate Generation
- Dynamic certificate generation with candidate data
- Multiple certificate types (STCW, MODU, H2S, BOSIET)
- Canvas-based certificate editing
- QR code integration for verification

## API Endpoints

All operations endpoints are prefixed with `/api/operations/`:

- `POST /upload-images` - Upload and process documents
- `POST /save-candidate-data` - Save candidate information
- `GET /get-candidate-data/<filename>` - Retrieve candidate data
- `POST /test-ocr` - Test OCR functionality

## Usage

### Starting the Operations Flow

1. **Upload Documents** (`/upload-docx`)
   - Upload required documents (photo, signature, passport)
   - OCR processing extracts structured data
   - Navigate to candidate details

2. **Enter Candidate Details** (`/candidate-details`)
   - Form pre-filled with OCR data
   - Manual entry for additional fields
   - Data validation and saving

3. **Select Courses** (`/course-selection`)
   - Choose relevant courses
   - Set course status and dates
   - Navigate to course preview

4. **Preview and Manage** (`/course-preview`)
   - Review selected courses
   - Generate certificates for completed courses
   - Track completion status

5. **Generate Certificates** (`/certificate-form`, etc.)
   - Dynamic certificate generation
   - Canvas-based editing
   - QR code integration
   - PDF download and storage

## Dependencies

- React Router for navigation
- Framer Motion for animations
- Lucide React for icons
- Canvas API for certificate generation
- Shared hooks from `../shared/hooks/`
- Shared utilities from `../shared/utils/`
