# Simplified File Organization System

## Overview

The Flask backend implements a streamlined file organization system that:

1. **Temporarily stores** uploaded files in session-based folders
2. **Organizes files** into candidate-specific folders upon form submission
3. **Handles payment screenshots** for paid candidates
4. **Prevents conflicts** with automatic folder naming
5. **Preserves original filenames** throughout the process
6. **Single Dynamic JSON** - Stores current candidate data for certificate generation
7. **OCR Processing** - Maintains structured passport data extraction

## Directory Structure

```
backend/uploads/
├── temp/                           # Temporary session storage
│   ├── {sessionId1}/              # Files from upload-docx page
│   │   ├── photo.jpg
│   │   ├── signature.png
│   │   ├── passport_front.jpg
│   │   ├── passport_back.jpg
│   │   ├── cdc.jpg
│   │   ├── marksheet.pdf
│   │   └── payment_screenshot.jpg  # If payment status is PAID
│   └── {sessionId2}/
│       └── ...
├── images/                         # Organized candidate folders
│   ├── John_Doe_A1234567/         # {firstName}_{lastName}_{passportNo}
│   │   ├── photo.jpg              # Uploaded files only
│   │   ├── signature.png
│   │   ├── passport_front.jpg
│   │   ├── passport_back.jpg
│   │   ├── cdc.jpg
│   │   ├── marksheet.pdf
│   │   └── payment_screenshot.jpg
│   ├── Jane_Smith_B7654321/
│   │   └── [uploaded files only...]
│   └── John_Doe_A1234567_1/       # Conflict resolution with _1, _2, etc.
│       └── ...
├── json/                          # Certificate data and OCR results
│   ├── current_candidate_for_certificate.json  # CURRENT CANDIDATE DATA
│   ├── structured_passport_data_{sessionId}.json  # OCR extracted data
│   └── ...
└── pdfs/                          # Generated certificates
    └── ...
```

## Process Flow

### 1. Document Upload (UploadDocx Page)
- **Endpoint**: `POST /upload-images`
- **Action**: Files saved to `backend/uploads/temp/{sessionId}/`
- **Response**: Includes `session_id` for tracking
- **Files**: photo, signature, passport_front_img, passport_back_img, cdc_img, marksheet

### 2. OCR Processing
- **Action**: Extract data from uploaded documents
- **Storage**: OCR results saved as `structured_passport_data_{sessionId}.json`
- **Session**: Session ID included in OCR data

### 3. Candidate Details Form
- **Session**: Session ID passed from upload page
- **Payment Screenshot**: If payment status is "PAID", upload to same temp session folder
- **Endpoint**: `POST /upload-payment-screenshot`

### 4. Form Submission (Simplified)
- **Endpoint**: `POST /save-candidate-data`
- **Actions**:
  1. Create sanitized folder name: `{firstName}_{lastName}_{passportNo}`
  2. Handle conflicts by appending `_1`, `_2`, etc.
  3. Move all files from temp session to candidate folder
  4. **Save current candidate data** to `current_candidate_for_certificate.json`
  5. Clean up empty temp session folder

### 5. Certificate Data Retrieval
- **Endpoint**: `GET /get-current-candidate-for-certificate`
- **Purpose**: Retrieve current candidate data for certificate generation
- **Returns**: Complete candidate information including file paths and metadata

## API Endpoints

### New Endpoints

#### Upload Payment Screenshot
```http
POST /upload-payment-screenshot
Content-Type: multipart/form-data

paymentScreenshot: <file>
sessionId: <string>
```

#### Cleanup Expired Sessions
```http
POST /cleanup-expired-sessions
Content-Type: application/json

{
  "hours_old": 24  // Default: 24 hours
}
```

### Modified Endpoints

#### Upload Images
```http
POST /upload-images
```
**Changes**:
- Files saved to temp session folder instead of permanent storage
- Returns `session_id` and `temp_folder` in response
- OCR data includes session information

#### Save Candidate Data
```http
POST /save-candidate-data
```
**Changes**:
- Requires `firstName`, `lastName`, `passport` for folder creation
- Accepts `session_id` to locate temp files
- Moves files from temp to organized candidate folder
- Returns file organization details

## Frontend Integration

### UploadDocx.jsx Changes
```javascript
// Store and pass session ID
if (result.status === 'success') {
  navigate('/candidate-details', {
    state: {
      ocrData: result.data,
      jsonFile: result.json_file,
      sessionId: result.session_id  // NEW
    }
  });
}
```

### CandidateDetail.jsx Changes
```javascript
// Get session ID from navigation
const sessionId = location.state?.sessionId;

// Include session ID in form data
const [formData, setFormData] = useState({
  // ... other fields
  session_id: sessionId || "",
});

// Upload payment screenshot immediately when selected
const handleFileChange = async (e) => {
  const file = e.target.files[0];
  if (!file || !sessionId) return;

  const formData = new FormData();
  formData.append('paymentScreenshot', file);
  formData.append('sessionId', sessionId);

  const response = await fetch('/upload-payment-screenshot', {
    method: 'POST',
    body: formData
  });
  // Handle response...
};
```

## Utility Functions

### Backend (shared/utils.py)
- `generate_session_id()`: Create unique session identifier
- `sanitize_folder_name()`: Remove special characters from folder names
- `create_unique_candidate_folder()`: Handle folder name conflicts
- `move_files_to_candidate_folder()`: Move files and cleanup temp folder

### Configuration (shared/config.py)
- Added `TEMP_FOLDER` configuration
- Updated directory creation to include temp folder

## Benefits

1. **Data Integrity**: Files are organized by candidate, preventing mix-ups
2. **Conflict Resolution**: Automatic handling of duplicate folder names
3. **Clean Organization**: Logical folder structure for easy file management
4. **Session Management**: Temporary storage prevents orphaned files
5. **Original Filenames**: Preserves user's original file names
6. **Payment Integration**: Seamless payment screenshot handling
7. **Cleanup**: Automatic removal of expired temporary sessions

## Testing

Run the test script to verify functionality:
```bash
cd backend
python test_file_organization.py
```

The test script will:
1. Upload test files to a temp session
2. Upload a payment screenshot
3. Save candidate data and organize files
4. Verify the file organization structure

## Maintenance

### Cleanup Expired Sessions
Regularly clean up old temporary sessions:
```bash
curl -X POST http://localhost:5000/cleanup-expired-sessions \
  -H "Content-Type: application/json" \
  -d '{"hours_old": 24}'
```

### Monitoring
- Check temp folder size regularly
- Monitor for orphaned session folders
- Verify candidate folder organization

## Error Handling

- **Missing Session**: Returns error if session ID not found
- **File Conflicts**: Automatically resolves with numbered suffixes
- **Move Errors**: Logs errors but continues processing
- **Validation**: Ensures required candidate fields are present
- **Cleanup**: Removes empty temp folders after successful moves
