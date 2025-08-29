"""
Operations routes for candidate data management
"""
from flask import Blueprint, request, jsonify
import os

from ...shared.config import Config
from ...shared.utils import (
    create_success_response, 
    create_error_response,
    save_json_data,
    load_json_data
)

candidates_bp = Blueprint('candidates', __name__)

@candidates_bp.route('/save-candidate-data', methods=['POST'])
def save_candidate_data():
    """Save candidate form data"""
    try:
        data = request.get_json()

        if not data:
            return create_error_response("No data provided", 400)

        # Save candidate data to current_candidate_for_certificate.json
        json_filename = "current_candidate_for_certificate.json"
        json_path = os.path.join(Config.JSON_FOLDER, json_filename)

        success, error = save_json_data(data, json_path)
        
        if not success:
            return create_error_response(f"Failed to save candidate data: {error}", 500)

        print(f"[JSON] Updated {json_filename} with new candidate data")

        return create_success_response(
            "Candidate data saved successfully",
            filename=json_filename
        )

    except Exception as e:
        return create_error_response(f"Failed to save candidate data: {str(e)}", 500)

@candidates_bp.route('/get-candidate-data/<filename>', methods=['GET'])
def get_candidate_data(filename):
    """Retrieve candidate data by filename"""
    try:
        json_path = os.path.join(Config.JSON_FOLDER, filename)
        
        data, error = load_json_data(json_path)
        
        if error:
            if "File not found" in error:
                return create_error_response("File not found", 404)
            else:
                return create_error_response(f"Failed to load data: {error}", 500)

        return create_success_response(
            "Candidate data retrieved successfully",
            data=data
        )

    except Exception as e:
        return create_error_response(f"Failed to retrieve candidate data: {str(e)}", 500)

@candidates_bp.route('/list-candidate-data', methods=['GET'])
def list_candidate_data():
    """List all candidate data files"""
    try:
        json_files = []
        
        if os.path.exists(Config.JSON_FOLDER):
            for filename in os.listdir(Config.JSON_FOLDER):
                if filename.endswith('.json'):
                    file_path = os.path.join(Config.JSON_FOLDER, filename)
                    file_stat = os.stat(file_path)
                    
                    json_files.append({
                        "filename": filename,
                        "size_bytes": file_stat.st_size,
                        "modified": file_stat.st_mtime,
                        "created": file_stat.st_ctime
                    })
        
        return create_success_response(
            "Candidate data files listed successfully",
            data=json_files,
            count=len(json_files)
        )
        
    except Exception as e:
        return create_error_response(f"Failed to list candidate data: {str(e)}", 500)
