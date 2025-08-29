"""
Operations routes for file upload and OCR processing
"""
from flask import Blueprint, request, jsonify
import os
from datetime import datetime

from ...shared.config import Config
from ...shared.utils import (
    create_success_response, 
    create_error_response,
    validate_required_files,
    secure_filename_with_timestamp,
    allowed_file,
    save_json_data,
    cleanup_temp_files
)

# Import OCR functions from the main app (we'll move these to services later)
# For now, we'll create placeholder functions

upload_bp = Blueprint('upload', __name__)

def perform_ocr_processing(file_paths):
    """
    Placeholder for OCR processing functionality
    This should be moved to operations/services/ocr_service.py
    """
    # This is a simplified version - the actual OCR logic from app.py should be moved here
    return {
        "passport_data": {
            "name": "Sample Name",
            "passport_number": "A12345678",
            "date_of_birth": "1990-01-01",
            "nationality": "Sample Country"
        },
        "cdc_data": {
            "cdc_number": "CDC123456",
            "issue_date": "2023-01-01",
            "expiry_date": "2028-01-01"
        },
        "processing_status": "success"
    }

@upload_bp.route('/upload-images', methods=['POST', 'OPTIONS'])
def upload_images():
    """
    Handle multiple file uploads and perform OCR processing
    Expected files: photo, signature, passport_front_img, passport_back_img, cdc_img (optional), marksheet (optional)
    """
    try:
        # Check if required files are present
        required_files = ['photo', 'signature', 'passport_front_img', 'passport_back_img']
        missing_files = validate_required_files(request, required_files)
        
        if missing_files:
            return create_error_response(
                f"Missing required files: {', '.join(missing_files)}", 
                400
            )

        # Validate file types and save files
        saved_files = {}
        for file_key in ['photo', 'signature', 'passport_front_img', 'passport_back_img', 'cdc_img', 'marksheet']:
            if file_key in request.files:
                file = request.files[file_key]
                if file.filename != '':
                    # Validate file type
                    if not allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
                        return create_error_response(
                            f"Invalid file type for {file_key}. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}", 
                            400
                        )
                    
                    # Save file
                    filename = secure_filename_with_timestamp(file.filename)
                    file_path = os.path.join(Config.IMAGES_FOLDER, filename)
                    file.save(file_path)
                    saved_files[file_key] = file_path
                    
                    print(f"[UPLOAD] Saved {file_key}: {filename}")

        # Perform OCR processing
        print("[OCR] Starting OCR processing...")
        ocr_data = perform_ocr_processing(saved_files)
        
        # Save OCR data to JSON file
        json_filename = "structured_passport_data.json"
        json_path = os.path.join(Config.JSON_FOLDER, json_filename)
        success, error = save_json_data(ocr_data, json_path)
        
        if not success:
            return create_error_response(f"Failed to save OCR data: {error}", 500)
        
        print(f"[JSON] Updated {json_filename} with new OCR data")

        # Clean up temporary files to save storage
        cleanup_temp_files(saved_files.values())

        return create_success_response(
            "Files uploaded and processed successfully",
            data=ocr_data,
            json_file=json_filename,
            files_processed=len(saved_files)
        )

    except Exception as e:
        print(f"[ERROR] Upload processing failed: {e}")
        return create_error_response(f"Upload processing failed: {str(e)}", 500)

@upload_bp.route('/test-ocr', methods=['POST'])
def test_ocr():
    """Test OCR with single image"""
    try:
        if 'image' not in request.files:
            return create_error_response("No image file provided", 400)
        
        file = request.files['image']
        if file.filename == '':
            return create_error_response("No file selected", 400)
        
        if not allowed_file(file.filename, Config.ALLOWED_EXTENSIONS):
            return create_error_response(
                f"Invalid file type. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}", 
                400
            )
        
        # Save temporary file
        filename = secure_filename_with_timestamp(file.filename)
        file_path = os.path.join(Config.IMAGES_FOLDER, filename)
        file.save(file_path)
        
        # Perform OCR (placeholder)
        ocr_result = perform_ocr_processing({file.filename: file_path})
        
        # Clean up
        cleanup_temp_files([file_path])
        
        return create_success_response(
            "OCR test completed",
            data=ocr_result,
            filename=filename
        )
        
    except Exception as e:
        return create_error_response(f"OCR test failed: {str(e)}", 500)
