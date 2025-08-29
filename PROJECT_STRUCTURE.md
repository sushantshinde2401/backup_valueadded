# Certificate Management System - Project Structure

This document outlines the organized structure of the Certificate Management System, which has been separated into three main sections: Operations, Bookkeeping, and Database Management.

## 🏗️ Overall Architecture

The system follows a modular architecture with clear separation of concerns:

```
my-app/
├── src/                    # Frontend (React)
│   ├── shared/            # Shared components and utilities
│   ├── operations/        # Operations section
│   ├── bookkeeping/       # Bookkeeping section
│   └── database/          # Database management section
├── backend/               # Backend (Flask)
│   ├── shared/           # Shared backend utilities
│   ├── operations/       # Operations API routes
│   ├── bookkeeping/      # Bookkeeping API routes
│   └── database/         # Database API routes
└── PROJECT_STRUCTURE.md  # This file
```

## 📁 Frontend Structure (src/)

### Shared Components (`src/shared/`)
```
shared/
├── components/
│   └── ThemeContext.jsx      # Theme management
├── hooks/
│   └── useCertificateManager.js  # Certificate management hook
├── utils/
│   └── certificateMap.js     # Certificate type mappings
└── styles/
    └── Navbar.css            # Shared navigation styles
```

### Operations Section (`src/operations/`)
**Purpose**: Document processing, candidate management, certificate generation

```
operations/
├── components/               # Operation-specific components
├── pages/
│   ├── HomePage.jsx         # Main dashboard
│   ├── UploadDocx.jsx       # Document upload & OCR
│   ├── CandidateDetail.jsx  # Candidate information form
│   ├── CourseSelection.jsx  # Course selection interface
│   ├── CoursePreview.js     # Course preview & management
│   └── certificates/        # Certificate generation
│       ├── DualCertificate.jsx   # STCW certificates
│       ├── DualCertificate2.jsx  # MODU certificates
│       ├── DualCertificate3.jsx  # H2S certificates
│       └── DualCertificate4.jsx  # BOSIET certificates
├── services/                # API service calls
└── README.md               # Operations documentation
```

### Bookkeeping Section (`src/bookkeeping/`)
**Purpose**: File management, storage, Google Drive integration

```
bookkeeping/
├── components/              # Bookkeeping-specific components
├── pages/
│   └── BookkeepingDashboard.jsx  # File management dashboard
├── services/                # API service calls
└── README.md               # Bookkeeping documentation
```

### Database Section (`src/database/`)
**Purpose**: Data search, reporting, analytics

```
database/
├── components/              # Database-specific components
├── pages/
│   └── DatabaseDashboard.jsx    # Data management dashboard
├── services/                # API service calls
└── README.md               # Database documentation
```

## 🔧 Backend Structure (backend/)

### Shared Backend (`backend/shared/`)
```
shared/
├── config.py               # Application configuration
├── utils.py                # Common utilities
└── __init__.py
```

### Operations Backend (`backend/operations/`)
**Purpose**: File upload, OCR processing, candidate data management

```
operations/
├── routes/
│   ├── upload.py           # File upload & OCR routes
│   ├── candidates.py       # Candidate data routes
│   └── __init__.py
├── services/               # Business logic services
└── __init__.py
```

### Bookkeeping Backend (`backend/bookkeeping/`)
**Purpose**: File management, Google Drive, PDF storage

```
bookkeeping/
├── routes/
│   ├── files.py            # File management routes
│   ├── drive.py            # Google Drive integration
│   └── __init__.py
├── services/               # Business logic services
└── __init__.py
```

### Database Backend (`backend/database/`)
**Purpose**: Data search, export, validation, reporting

```
database/
├── routes/
│   ├── data.py             # Data management routes
│   └── __init__.py
├── services/               # Business logic services
└── __init__.py
```

## 🚀 API Endpoints

### Operations API (`/api/operations/`)
- `POST /upload-images` - Upload and process documents
- `POST /save-candidate-data` - Save candidate information
- `GET /get-candidate-data/<filename>` - Retrieve candidate data
- `POST /test-ocr` - Test OCR functionality

### Bookkeeping API (`/api/bookkeeping/`)
- `GET /list-files` - List all uploaded files
- `POST /save-pdf` - Save PDF and upload to Drive
- `GET /download-pdf/<filename>` - Download PDF files
- `GET /storage-stats` - Get storage statistics
- `GET /drive-status` - Check Google Drive status

### Database API (`/api/database/`)
- `GET /search-candidates` - Search candidates
- `GET /data-summary` - Get data summary
- `GET /export-data` - Export all data
- `GET /data-validation` - Validate data integrity

## 🔄 Navigation Flow

```
HomePage (Operations Dashboard)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│   OPERATIONS    │   BOOKKEEPING   │    DATABASE     │
│                 │                 │                 │
│ UploadDocx      │ File Management │ Search & Query  │
│      ↓          │ PDF Storage     │ Data Export     │
│ CandidateDetail │ Drive Sync      │ Reports         │
│      ↓          │ QR Management   │ Analytics       │
│ CourseSelection │                 │                 │
│      ↓          │                 │                 │
│ CoursePreview   │                 │                 │
│      ↓          │                 │                 │
│ Certificates    │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘
```

## 🛠️ Development Workflow

### Starting the Application

1. **Backend** (Choose one):
   ```bash
   # New modular structure
   cd backend
   python app_new.py
   
   # Legacy structure (for compatibility)
   python app.py
   ```

2. **Frontend**:
   ```bash
   npm start
   ```

### Adding New Features

1. **Identify the Section**: Determine if the feature belongs to Operations, Bookkeeping, or Database
2. **Frontend**: Add components/pages to the appropriate section folder
3. **Backend**: Add routes to the corresponding backend section
4. **Shared**: Place reusable code in shared folders

### File Organization Rules

- **Operations**: Anything related to document processing, candidate management, or certificate generation
- **Bookkeeping**: File storage, Google Drive, PDF management, QR codes
- **Database**: Data search, reporting, analytics, export/import
- **Shared**: Components, utilities, or configurations used across multiple sections

## 📋 Migration Status

### ✅ Completed
- [x] Created organized folder structure
- [x] Moved existing components to appropriate sections
- [x] Updated import paths
- [x] Created new modular backend structure
- [x] Added section dashboards
- [x] Updated main navigation

### 🔄 In Progress
- [ ] Move actual OCR logic to operations services
- [ ] Move Google Drive logic to bookkeeping services
- [ ] Create comprehensive API documentation
- [ ] Add error handling and logging
- [ ] Implement authentication and authorization

### 📅 Planned
- [ ] Add unit tests for each section
- [ ] Create deployment configurations
- [ ] Add monitoring and analytics
- [ ] Implement caching strategies
- [ ] Add API rate limiting

## 🔧 Configuration

### Environment Variables
```bash
# Flask Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True

# Google Drive
GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE=service-account.json

# File Storage
MAX_FILE_SIZE=50000000
ALLOWED_EXTENSIONS=pdf,jpg,jpeg,png,doc,docx

# Features
ENABLE_CHATGPT_FILTERING=true
OPENAI_API_KEY=your_key_here
```

## 📚 Documentation

Each section has its own README.md with detailed information:
- [Operations README](src/operations/README.md)
- [Bookkeeping README](src/bookkeeping/README.md)
- [Database README](src/database/README.md)

## 🤝 Contributing

When contributing to this project:
1. Follow the modular structure
2. Place code in the appropriate section
3. Use shared utilities when possible
4. Update documentation
5. Add tests for new features
