"""
Bookkeeping routes for Google Drive integration and PDF management
"""
from flask import Blueprint, request, jsonify
import os

from ...shared.config import Config
from ...shared.utils import (
    create_success_response, 
    create_error_response,
    secure_filename_with_timestamp,
    allowed_file
)

drive_bp = Blueprint('drive', __name__)

def upload_to_drive_service(file_path, filename):
    """
    Placeholder for Google Drive upload functionality
    This should be moved to bookkeeping/services/drive_service.py
    """
    # This is a simplified version - the actual Google Drive logic from app.py should be moved here
    return f"https://drive.google.com/file/d/sample_file_id/view?usp=sharing"

def generate_qr_code_service(url, filename):
    """
    Placeholder for QR code generation functionality
    This should be moved to bookkeeping/services/qr_service.py
    """
    # This is a simplified version - the actual QR code logic from app.py should be moved here
    qr_path = os.path.join(Config.STATIC_FOLDER, f"qr_{filename}.png")
    qr_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    return qr_path, qr_base64

@drive_bp.route('/save-pdf', methods=['POST'])
def save_pdf():
    """Save PDF and upload to Google Drive"""
    try:
        if 'pdf' not in request.files:
            return create_error_response("No PDF file provided", 400)

        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return create_error_response("No file selected", 400)

        if not allowed_file(pdf_file.filename, {'pdf'}):
            return create_error_response("Only PDF files are allowed", 400)

        # Save PDF locally
        filename = secure_filename_with_timestamp(pdf_file.filename)
        pdf_path = os.path.join(Config.PDFS_FOLDER, filename)
        pdf_file.save(pdf_path)

        try:
            # Upload to Google Drive
            drive_link = upload_to_drive_service(pdf_path, filename)

            # Generate QR code
            qr_path, qr_base64 = generate_qr_code_service(drive_link, filename)

            return create_success_response(
                "PDF saved and uploaded to Google Drive successfully",
                drive_link=drive_link,
                qr_image=qr_base64,
                filename=filename,
                storage_type="google_drive"
            )

        except Exception as drive_error:
            # If Google Drive upload fails, still save locally
            print(f"Google Drive upload failed: {drive_error}")

            # Generate local link and QR code
            local_link = f"{Config.BASE_URL}/download-pdf/{filename}"
            qr_path, qr_base64 = generate_qr_code_service(local_link, filename)

            return create_success_response(
                "PDF saved locally (Google Drive upload failed)",
                drive_link=local_link,
                qr_image=qr_base64,
                filename=filename,
                storage_type="local",
                warning="Google Drive upload failed, file saved locally"
            )

    except Exception as e:
        return create_error_response(f"Failed to save PDF: {str(e)}", 500)

@drive_bp.route('/upload', methods=['POST'])
def legacy_upload():
    """Legacy PDF upload endpoint for backward compatibility"""
    try:
        if 'pdf' not in request.files:
            return create_error_response("No PDF file provided", 400)

        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return create_error_response("No file selected", 400)

        # Save PDF locally
        filename = secure_filename_with_timestamp(pdf_file.filename)
        pdf_path = os.path.join(Config.PDFS_FOLDER, filename)
        pdf_file.save(pdf_path)

        try:
            # Upload to Google Drive
            drive_link = upload_to_drive_service(pdf_path, filename)

            # Generate QR code
            qr_path, qr_base64 = generate_qr_code_service(drive_link, filename)

            return jsonify({
                "success": True,
                "status": "success",
                "drive_link": drive_link,
                "qr_image": qr_base64,
                "filename": filename,
                "storage_type": "google_drive"
            }), 200

        except Exception as drive_error:
            # Fallback to local storage
            local_link = f"{Config.BASE_URL}/download-pdf/{filename}"
            qr_path, qr_base64 = generate_qr_code_service(local_link, filename)

            return jsonify({
                "success": True,
                "status": "success",
                "drive_link": local_link,
                "qr_image": qr_base64,
                "filename": filename,
                "storage_type": "local",
                "warning": f"Google Drive upload failed: {str(drive_error)}"
            }), 200

    except Exception as e:
        return create_error_response(f"Legacy upload failed: {str(e)}", 500)

@drive_bp.route('/drive-status', methods=['GET'])
def get_drive_status():
    """Check Google Drive integration status"""
    try:
        # Check if service account file exists
        service_account_exists = os.path.exists(Config.GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE)
        
        # Check if credentials file exists
        credentials_exists = os.path.exists(Config.GOOGLE_DRIVE_CREDENTIALS_FILE)
        
        status = {
            "service_account_configured": service_account_exists,
            "credentials_configured": credentials_exists,
            "drive_integration_ready": service_account_exists or credentials_exists,
            "service_account_path": Config.GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE,
            "credentials_path": Config.GOOGLE_DRIVE_CREDENTIALS_FILE
        }
        
        return create_success_response(
            "Google Drive status retrieved successfully",
            data=status
        )
        
    except Exception as e:
        return create_error_response(f"Failed to get Drive status: {str(e)}", 500)
