"""
Database management routes for data operations
"""
from flask import Blueprint, request, jsonify
import os
import json
from datetime import datetime

from ...shared.config import Config
from ...shared.utils import (
    create_success_response, 
    create_error_response,
    load_json_data
)

data_bp = Blueprint('data', __name__)

@data_bp.route('/search-candidates', methods=['GET'])
def search_candidates():
    """Search candidates by various criteria"""
    try:
        # Get search parameters
        search_term = request.args.get('q', '').lower()
        search_field = request.args.get('field', 'all')  # name, passport, cdc, email, etc.
        limit = int(request.args.get('limit', 50))
        
        results = []
        
        # Search in current_candidate_for_certificate.json
        candidate_file = os.path.join(Config.JSON_FOLDER, "current_candidate_for_certificate.json")
        candidate_data, error = load_json_data(candidate_file)
        
        if candidate_data and not error:
            # Check if search term matches
            match_found = False
            
            if search_field == 'all':
                # Search in all fields
                candidate_str = json.dumps(candidate_data).lower()
                if search_term in candidate_str:
                    match_found = True
            else:
                # Search in specific field
                if search_field in candidate_data:
                    field_value = str(candidate_data[search_field]).lower()
                    if search_term in field_value:
                        match_found = True
            
            if match_found:
                results.append({
                    "type": "candidate",
                    "source": "current_candidate_for_certificate.json",
                    "data": candidate_data,
                    "match_score": 1.0
                })
        
        # Search in structured_passport_data.json
        passport_file = os.path.join(Config.JSON_FOLDER, "structured_passport_data.json")
        passport_data, error = load_json_data(passport_file)
        
        if passport_data and not error:
            match_found = False
            
            if search_field == 'all':
                passport_str = json.dumps(passport_data).lower()
                if search_term in passport_str:
                    match_found = True
            else:
                # Search in nested passport data
                for key, value in passport_data.items():
                    if isinstance(value, dict):
                        if search_field in value:
                            field_value = str(value[search_field]).lower()
                            if search_term in field_value:
                                match_found = True
                                break
            
            if match_found:
                results.append({
                    "type": "passport_ocr",
                    "source": "structured_passport_data.json",
                    "data": passport_data,
                    "match_score": 0.8
                })
        
        # Limit results
        results = results[:limit]
        
        return create_success_response(
            f"Search completed. Found {len(results)} results",
            data=results,
            search_params={
                "query": search_term,
                "field": search_field,
                "limit": limit
            }
        )
        
    except Exception as e:
        return create_error_response(f"Search failed: {str(e)}", 500)

@data_bp.route('/data-summary', methods=['GET'])
def get_data_summary():
    """Get summary of all data in the system"""
    try:
        summary = {
            "candidates": 0,
            "passport_records": 0,
            "total_files": 0,
            "last_updated": None,
            "data_sources": []
        }
        
        # Check candidate data
        candidate_file = os.path.join(Config.JSON_FOLDER, "current_candidate_for_certificate.json")
        if os.path.exists(candidate_file):
            candidate_data, error = load_json_data(candidate_file)
            if candidate_data and not error:
                summary["candidates"] = 1
                summary["data_sources"].append({
                    "name": "current_candidate_for_certificate.json",
                    "type": "candidate_records",
                    "last_updated": candidate_data.get("last_updated"),
                    "record_count": 1
                })
                
                if candidate_data.get("last_updated"):
                    summary["last_updated"] = candidate_data["last_updated"]
        
        # Check passport OCR data
        passport_file = os.path.join(Config.JSON_FOLDER, "structured_passport_data.json")
        if os.path.exists(passport_file):
            passport_data, error = load_json_data(passport_file)
            if passport_data and not error:
                summary["passport_records"] = 1
                summary["data_sources"].append({
                    "name": "structured_passport_data.json",
                    "type": "ocr_records",
                    "last_updated": passport_data.get("last_updated"),
                    "record_count": 1
                })
                
                # Update last_updated if this is more recent
                if passport_data.get("last_updated"):
                    if not summary["last_updated"] or passport_data["last_updated"] > summary["last_updated"]:
                        summary["last_updated"] = passport_data["last_updated"]
        
        # Count total files
        for folder in [Config.IMAGES_FOLDER, Config.JSON_FOLDER, Config.PDFS_FOLDER]:
            if os.path.exists(folder):
                summary["total_files"] += len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
        
        return create_success_response(
            "Data summary retrieved successfully",
            data=summary
        )
        
    except Exception as e:
        return create_error_response(f"Failed to get data summary: {str(e)}", 500)

@data_bp.route('/export-data', methods=['GET'])
def export_data():
    """Export all data as a single JSON file"""
    try:
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0",
            "data": {}
        }
        
        # Export candidate data
        candidate_file = os.path.join(Config.JSON_FOLDER, "current_candidate_for_certificate.json")
        candidate_data, error = load_json_data(candidate_file)
        if candidate_data and not error:
            export_data["data"]["candidate_data"] = candidate_data
        
        # Export passport OCR data
        passport_file = os.path.join(Config.JSON_FOLDER, "structured_passport_data.json")
        passport_data, error = load_json_data(passport_file)
        if passport_data and not error:
            export_data["data"]["passport_ocr_data"] = passport_data
        
        # Add file listings
        export_data["data"]["file_inventory"] = {
            "images": [],
            "pdfs": [],
            "json": []
        }
        
        # List files in each folder
        folders = {
            "images": Config.IMAGES_FOLDER,
            "pdfs": Config.PDFS_FOLDER,
            "json": Config.JSON_FOLDER
        }
        
        for folder_name, folder_path in folders.items():
            if os.path.exists(folder_path):
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        file_stat = os.stat(file_path)
                        export_data["data"]["file_inventory"][folder_name].append({
                            "filename": filename,
                            "size_bytes": file_stat.st_size,
                            "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                            "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                        })
        
        return create_success_response(
            "Data exported successfully",
            data=export_data
        )
        
    except Exception as e:
        return create_error_response(f"Data export failed: {str(e)}", 500)

@data_bp.route('/data-validation', methods=['GET'])
def validate_data():
    """Validate data integrity and consistency"""
    try:
        validation_results = {
            "status": "valid",
            "issues": [],
            "warnings": [],
            "summary": {}
        }
        
        # Validate candidate data
        candidate_file = os.path.join(Config.JSON_FOLDER, "current_candidate_for_certificate.json")
        if os.path.exists(candidate_file):
            candidate_data, error = load_json_data(candidate_file)
            if error:
                validation_results["issues"].append(f"Cannot read current_candidate_for_certificate.json: {error}")
                validation_results["status"] = "invalid"
            else:
                # Check required fields
                required_fields = ["firstName", "lastName", "passport", "email"]
                missing_fields = [field for field in required_fields if not candidate_data.get(field)]
                if missing_fields:
                    validation_results["warnings"].append(f"Missing candidate fields: {', '.join(missing_fields)}")
        else:
            validation_results["warnings"].append("No candidate data file found")
        
        # Validate passport OCR data
        passport_file = os.path.join(Config.JSON_FOLDER, "structured_passport_data.json")
        if os.path.exists(passport_file):
            passport_data, error = load_json_data(passport_file)
            if error:
                validation_results["issues"].append(f"Cannot read structured_passport_data.json: {error}")
                validation_results["status"] = "invalid"
        else:
            validation_results["warnings"].append("No passport OCR data file found")
        
        # Check file consistency
        for folder_name, folder_path in [("images", Config.IMAGES_FOLDER), ("json", Config.JSON_FOLDER), ("pdfs", Config.PDFS_FOLDER)]:
            if not os.path.exists(folder_path):
                validation_results["issues"].append(f"Missing {folder_name} folder: {folder_path}")
                validation_results["status"] = "invalid"
        
        validation_results["summary"] = {
            "total_issues": len(validation_results["issues"]),
            "total_warnings": len(validation_results["warnings"]),
            "overall_status": validation_results["status"]
        }
        
        return create_success_response(
            "Data validation completed",
            data=validation_results
        )
        
    except Exception as e:
        return create_error_response(f"Data validation failed: {str(e)}", 500)
