# Bookkeeping Section

This section handles file management, storage, and financial tracking for the certificate management system.

## Structure

```
bookkeeping/
├── components/          # Reusable components for bookkeeping
├── pages/              # Bookkeeping management pages
│   └── BookkeepingDashboard.jsx  # Main bookkeeping dashboard
├── services/           # API services for bookkeeping
└── README.md          # This file
```

## Features

### File Management
- Comprehensive file listing and organization
- File type categorization (images, PDFs, JSON)
- Storage statistics and usage tracking
- File download and access management

### PDF Storage & Management
- Local PDF storage with organized structure
- Google Drive integration for cloud backup
- Automatic file naming with timestamps
- File size and metadata tracking

### Google Drive Integration
- Automatic upload to Google Drive
- Shareable link generation
- Fallback to local storage if Drive fails
- Drive status monitoring and configuration

### QR Code Management
- QR code generation for document links
- QR code positioning and customization
- Integration with certificate generation
- QR code storage and retrieval

## API Endpoints

All bookkeeping endpoints are prefixed with `/api/bookkeeping/`:

- `GET /list-files` - List all uploaded files
- `GET /download-pdf/<filename>` - Download PDF files
- `GET /download-image/<filename>` - Download image files
- `POST /save-pdf` - Save PDF and upload to Drive
- `GET /storage-stats` - Get storage statistics
- `GET /drive-status` - Check Google Drive integration status
- `GET /file-info/<type>/<filename>` - Get detailed file information

## Planned Features

### Financial Tracking
- Certificate generation costs
- Storage usage billing
- Revenue tracking per certificate
- Monthly/yearly financial reports

### Audit Trail
- File access logging
- User activity tracking
- Change history for important files
- Compliance reporting

### Backup Management
- Automated backup scheduling
- Multiple backup destinations
- Backup verification and integrity checks
- Disaster recovery procedures

### Storage Optimization
- Automatic file compression
- Duplicate file detection
- Archive old files to cold storage
- Storage quota management

## Usage

### File Management Workflow

1. **Upload Processing**
   - Files uploaded through operations section
   - Automatic categorization and storage
   - Metadata extraction and indexing

2. **Storage Management**
   - Monitor storage usage and limits
   - Organize files by date, type, or project
   - Clean up temporary and duplicate files

3. **Google Drive Sync**
   - Automatic upload of important files
   - Generate shareable links for certificates
   - Monitor sync status and handle failures

4. **QR Code Generation**
   - Create QR codes for certificate verification
   - Manage QR code positioning on certificates
   - Track QR code usage and scanning

## Configuration

### Google Drive Setup
1. Create Google Cloud Project
2. Enable Google Drive API
3. Create service account credentials
4. Download `service-account.json`
5. Place in backend directory

### Storage Configuration
- Set maximum file sizes
- Configure allowed file types
- Set up automatic cleanup policies
- Configure backup schedules

## Dependencies

- Google Drive API for cloud storage
- QR code generation libraries
- File system utilities
- Storage monitoring tools
