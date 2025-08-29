"""
Bookkeeping routes for file management
"""
from flask import Blueprint, request, jsonify, send_file
import os

from ...shared.config import Config
from ...shared.utils import (
    create_success_response, 
    create_error_response,
    format_file_info
)

files_bp = Blueprint('files', __name__)

@files_bp.route('/list-files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    try:
        files_data = {
            "images": [],
            "json": [],
            "pdfs": []
        }
        
        # List images
        if os.path.exists(Config.IMAGES_FOLDER):
            for filename in os.listdir(Config.IMAGES_FOLDER):
                file_path = os.path.join(Config.IMAGES_FOLDER, filename)
                file_info = format_file_info(file_path)
                if file_info:
                    files_data["images"].append(file_info)
        
        # List JSON files
        if os.path.exists(Config.JSON_FOLDER):
            for filename in os.listdir(Config.JSON_FOLDER):
                file_path = os.path.join(Config.JSON_FOLDER, filename)
                file_info = format_file_info(file_path)
                if file_info:
                    files_data["json"].append(file_info)
        
        # List PDFs
        if os.path.exists(Config.PDFS_FOLDER):
            for filename in os.listdir(Config.PDFS_FOLDER):
                file_path = os.path.join(Config.PDFS_FOLDER, filename)
                file_info = format_file_info(file_path)
                if file_info:
                    files_data["pdfs"].append(file_info)
        
        # Calculate totals
        total_files = len(files_data["images"]) + len(files_data["json"]) + len(files_data["pdfs"])
        
        return create_success_response(
            "Files listed successfully",
            data=files_data,
            total_files=total_files,
            breakdown={
                "images": len(files_data["images"]),
                "json": len(files_data["json"]),
                "pdfs": len(files_data["pdfs"])
            }
        )
        
    except Exception as e:
        return create_error_response(f"Failed to list files: {str(e)}", 500)

@files_bp.route('/download-pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    """Download PDF file"""
    try:
        file_path = os.path.join(Config.PDFS_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return create_error_response("File not found", 404)
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return create_error_response(f"Failed to download file: {str(e)}", 500)

@files_bp.route('/download-image/<filename>', methods=['GET'])
def download_image(filename):
    """Download image file"""
    try:
        file_path = os.path.join(Config.IMAGES_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return create_error_response("File not found", 404)
        
        # Determine mimetype based on extension
        ext = filename.lower().split('.')[-1]
        mimetype_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp'
        }
        mimetype = mimetype_map.get(ext, 'application/octet-stream')
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype
        )
        
    except Exception as e:
        return create_error_response(f"Failed to download image: {str(e)}", 500)

@files_bp.route('/file-info/<file_type>/<filename>', methods=['GET'])
def get_file_info(file_type, filename):
    """Get detailed information about a specific file"""
    try:
        # Determine folder based on file type
        folder_map = {
            'image': Config.IMAGES_FOLDER,
            'json': Config.JSON_FOLDER,
            'pdf': Config.PDFS_FOLDER
        }
        
        if file_type not in folder_map:
            return create_error_response("Invalid file type", 400)
        
        file_path = os.path.join(folder_map[file_type], filename)
        file_info = format_file_info(file_path)
        
        if not file_info:
            return create_error_response("File not found", 404)
        
        return create_success_response(
            "File information retrieved successfully",
            data=file_info
        )
        
    except Exception as e:
        return create_error_response(f"Failed to get file info: {str(e)}", 500)

@files_bp.route('/storage-stats', methods=['GET'])
def get_storage_stats():
    """Get storage statistics"""
    try:
        stats = {
            "folders": {},
            "total_size_mb": 0,
            "total_files": 0
        }
        
        folders = {
            "images": Config.IMAGES_FOLDER,
            "json": Config.JSON_FOLDER,
            "pdfs": Config.PDFS_FOLDER
        }
        
        for folder_name, folder_path in folders.items():
            folder_stats = {
                "file_count": 0,
                "size_mb": 0,
                "files": []
            }
            
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        file_info = format_file_info(file_path)
                        if file_info:
                            folder_stats["files"].append(file_info)
                            folder_stats["file_count"] += 1
                            folder_stats["size_mb"] += file_info["size_mb"]
            
            stats["folders"][folder_name] = folder_stats
            stats["total_files"] += folder_stats["file_count"]
            stats["total_size_mb"] += folder_stats["size_mb"]
        
        # Round total size
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        
        return create_success_response(
            "Storage statistics retrieved successfully",
            data=stats
        )
        
    except Exception as e:
        return create_error_response(f"Failed to get storage stats: {str(e)}", 500)
