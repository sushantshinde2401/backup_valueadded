"""
Main Flask application with organized route structure
"""
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime
import sys
import os

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

# Import configuration and utilities
from shared.config import Config, create_directories, validate_config
from shared.utils import create_success_response, create_error_response

# Import route blueprints
from operations.routes.upload import upload_bp
from operations.routes.candidates import candidates_bp
from bookkeeping.routes.files import files_bp
from bookkeeping.routes.drive import drive_bp
from database.routes.data import data_bp

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    
    # Configure Flask app
    app.config.from_object(Config)
    
    # Enable CORS
    CORS(app)
    
    # Create necessary directories
    create_directories()
    
    # Register blueprints with URL prefixes
    
    # Operations routes
    app.register_blueprint(upload_bp, url_prefix='/api/operations')
    app.register_blueprint(candidates_bp, url_prefix='/api/operations')
    
    # Bookkeeping routes
    app.register_blueprint(files_bp, url_prefix='/api/bookkeeping')
    app.register_blueprint(drive_bp, url_prefix='/api/bookkeeping')
    
    # Database routes
    app.register_blueprint(data_bp, url_prefix='/api/database')
    
    # Legacy routes (for backward compatibility)
    app.register_blueprint(upload_bp, url_prefix='/')
    app.register_blueprint(candidates_bp, url_prefix='/')
    app.register_blueprint(files_bp, url_prefix='/')
    app.register_blueprint(drive_bp, url_prefix='/')
    
    return app

# Create the Flask app
app = create_app()

# ============================================================================
# MAIN ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return create_success_response(
        "Document Processing Server is running",
        server_info={
            "version": "2.0.0",
            "architecture": "modular",
            "sections": ["operations", "bookkeeping", "database"],
            "timestamp": datetime.now().isoformat()
        }
    )

@app.route('/api/status', methods=['GET'])
def api_status():
    """API status and configuration check"""
    try:
        # Validate configuration
        warnings = validate_config()
        
        status_info = {
            "api_version": "2.0.0",
            "sections": {
                "operations": {
                    "description": "File upload, OCR processing, candidate management",
                    "endpoints": [
                        "/api/operations/upload-images",
                        "/api/operations/save-candidate-data",
                        "/api/operations/get-candidate-data/<filename>",
                        "/api/operations/test-ocr"
                    ]
                },
                "bookkeeping": {
                    "description": "File management, Google Drive integration, PDF storage",
                    "endpoints": [
                        "/api/bookkeeping/list-files",
                        "/api/bookkeeping/save-pdf",
                        "/api/bookkeeping/download-pdf/<filename>",
                        "/api/bookkeeping/drive-status"
                    ]
                },
                "database": {
                    "description": "Data search, export, validation, reporting",
                    "endpoints": [
                        "/api/database/search-candidates",
                        "/api/database/data-summary",
                        "/api/database/export-data",
                        "/api/database/data-validation"
                    ]
                }
            },
            "configuration": {
                "upload_folder": Config.UPLOAD_FOLDER,
                "max_file_size_mb": Config.MAX_CONTENT_LENGTH / (1024 * 1024),
                "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
                "google_drive_enabled": os.path.exists(Config.GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE),
                "chatgpt_enabled": Config.ENABLE_CHATGPT_FILTERING
            },
            "warnings": warnings
        }
        
        return create_success_response(
            "API status retrieved successfully",
            data=status_info
        )
        
    except Exception as e:
        return create_error_response(f"Failed to get API status: {str(e)}", 500)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return create_error_response(
        f"File too large. Maximum size: {Config.MAX_CONTENT_LENGTH / (1024 * 1024):.1f}MB",
        413
    )

@app.errorhandler(404)
def not_found(e):
    """Handle not found error"""
    return create_error_response("Endpoint not found", 404)

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server error"""
    return create_error_response("Internal server error", 500)

# ============================================================================
# STARTUP
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 DOCUMENT PROCESSING SERVER v2.0 - MODULAR ARCHITECTURE")
    print("=" * 80)
    print(f"📁 Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"🖼️  Images folder: {Config.IMAGES_FOLDER}")
    print(f"📄 JSON folder: {Config.JSON_FOLDER}")
    print(f"📋 PDFs folder: {Config.PDFS_FOLDER}")
    print("=" * 80)
    print("📡 Available API sections:")
    print("   🔧 OPERATIONS    - /api/operations/*")
    print("      • File upload and OCR processing")
    print("      • Candidate data management")
    print("      • Certificate generation support")
    print()
    print("   📚 BOOKKEEPING   - /api/bookkeeping/*")
    print("      • File management and storage")
    print("      • Google Drive integration")
    print("      • PDF and QR code handling")
    print()
    print("   🗄️  DATABASE      - /api/database/*")
    print("      • Data search and retrieval")
    print("      • Export and validation")
    print("      • Reporting and analytics")
    print("=" * 80)
    print("🌐 Server starting on: http://localhost:5000")
    print("📖 API Status: http://localhost:5000/api/status")
    print("=" * 80)
    
    # Validate configuration and show warnings
    warnings = validate_config()
    if warnings:
        print("⚠️  Configuration warnings:")
        for warning in warnings:
            print(f"   • {warning}")
        print("=" * 80)
    
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
