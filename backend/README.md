# Document Processing System Backend

This is a comprehensive backend service for document processing, OCR extraction, and certificate generation with Google Drive integration.

## 🚀 Features

- **Multi-file Upload**: Handle multiple document uploads simultaneously
- **OCR Processing**: Extract text from passport images and CDC documents
- **Data Extraction**: Structured extraction of passport and CDC information
- **Google Drive Integration**: Automatic upload and shareable link generation
- **QR Code Generation**: Create QR codes for document links
- **PDF Processing**: Handle PDF generation and storage
- **RESTful API**: Complete API for frontend integration

## 📋 Quick Start

### 1. Automatic Setup (Recommended)

```bash
# Install dependencies and setup environment
python setup.py

# Test the setup
python test_setup.py

# Start the server
python start_server.py
```

### 2. Manual Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p uploads/images uploads/json uploads/pdfs

# Start the server
python app.py
```

## 🔧 Installation Requirements

### Python Dependencies
All Python packages are listed in `requirements.txt`:
- Flask & Flask-CORS
- Google Drive API libraries
- OCR libraries (pytesseract, opencv-python)
- Image processing (Pillow, numpy)
- QR code generation

### System Dependencies

**Tesseract OCR** (Required for OCR functionality):

- **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
- **CentOS/RHEL**: `sudo yum install tesseract`

## 📁 Project Structure

```
backend/
├── app.py                 # Main Flask application
├── setup.py              # Automated setup script
├── test_setup.py         # Setup verification script
├── start_server.py       # Server startup script
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
├── service-account.json  # Google Drive credentials (you need to add this)
├── uploads/              # File storage
│   ├── images/          # Uploaded images
│   ├── json/            # Extracted OCR data
│   └── pdfs/            # Generated PDFs
└── README.md            # This file
```

## 🌐 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload-images` | Upload multiple images + OCR processing |
| POST | `/save-candidate-data` | Save candidate form data |
| GET | `/get-candidate-data/<filename>` | Retrieve candidate data |
| POST | `/save-pdf` | Save PDF + Google Drive upload |
| GET | `/download-pdf/<filename>` | Download PDF file |
| POST | `/upload` | Legacy PDF upload (backward compatibility) |
| GET | `/list-files` | List all uploaded files |
| POST | `/test-ocr` | Test OCR with single image |

## 🚀 Running the Server

```bash
# Simple start
python app.py

# With checks
python start_server.py

# With setup
python setup.py && python app.py
```

Server will be available at: `http://localhost:5000`

## 📝 Workflow

1. **Upload Documents** → `/upload-images`
2. **OCR Processing** → Extract structured data
3. **Save Candidate Data** → `/save-candidate-data`
4. **Generate Certificates** → Frontend canvas
5. **Save PDFs** → `/save-pdf` + Google Drive
6. **Generate QR Codes** → Automatic with links

## 🔧 Troubleshooting

**OCR Issues:**
- Install Tesseract OCR
- Check PATH environment variable
- Verify image quality

**Google Drive Issues:**
- Check service-account.json
- Verify API is enabled
- Check permissions

**File Upload Issues:**
- Check file size limits
- Verify file extensions
- Check directory permissions