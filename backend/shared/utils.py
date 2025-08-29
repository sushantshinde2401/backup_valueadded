"""
Shared utilities for the backend application
"""
import os
import json
import uuid
import re
import shutil
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from flask import jsonify

def generate_timestamp():
    """Generate a timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_unique_id():
    """Generate a unique ID"""
    return str(uuid.uuid4())[:8]

def generate_session_id():
    """Generate a unique session ID for temporary file storage"""
    return str(uuid.uuid4())

def secure_filename_with_timestamp(filename):
    """Generate a secure filename with timestamp"""
    timestamp = generate_timestamp()
    unique_id = generate_unique_id()
    name, ext = os.path.splitext(secure_filename(filename))
    return f"{timestamp}_{unique_id}_{name}{ext}"

def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def save_json_data(data, filepath):
    """Save data to JSON file"""
    try:
        # Add metadata
        data['timestamp'] = generate_timestamp()
        data['last_updated'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as json_file:
            json.dump(data, json_file, indent=2)
        
        return True, None
    except Exception as e:
        return False, str(e)

def load_json_data(filepath):
    """Load data from JSON file"""
    try:
        if not os.path.exists(filepath):
            return None, "File not found"
        
        with open(filepath, 'r') as json_file:
            data = json.load(json_file)
        
        return data, None
    except Exception as e:
        return None, str(e)

def create_success_response(message, data=None, **kwargs):
    """Create a standardized success response"""
    response = {
        "status": "success",
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if data is not None:
        response["data"] = data
    
    # Add any additional fields
    response.update(kwargs)
    
    return jsonify(response)

def create_error_response(message, error_code=500, **kwargs):
    """Create a standardized error response"""
    response = {
        "status": "error",
        "error": message,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add any additional fields
    response.update(kwargs)
    
    return jsonify(response), error_code

def validate_required_files(request, required_files):
    """Validate that all required files are present in the request"""
    missing_files = []
    
    for file_key in required_files:
        if file_key not in request.files:
            missing_files.append(file_key)
        else:
            file = request.files[file_key]
            if file.filename == '':
                missing_files.append(file_key)
    
    return missing_files

def cleanup_temp_files(file_paths):
    """Clean up temporary files"""
    cleaned_count = 0
    errors = []
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleaned_count += 1
                print(f"[CLEANUP] Removed temporary file: {file_path}")
        except Exception as e:
            error_msg = f"Failed to remove {file_path}: {e}"
            errors.append(error_msg)
            print(f"[CLEANUP] {error_msg}")
    
    return cleaned_count, errors

def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    return 0

def sanitize_folder_name(name):
    """Sanitize folder name by removing special characters"""
    # Remove special characters, keep only alphanumeric, underscore, hyphen
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', str(name))
    # Replace multiple underscores/hyphens with single ones
    sanitized = re.sub(r'[-_]+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_-')
    return sanitized or 'unnamed'

def create_unique_candidate_folder(base_path, folder_name):
    """Create a unique candidate folder, appending _1, _2, etc. if needed"""
    original_folder = folder_name
    counter = 1

    while os.path.exists(os.path.join(base_path, folder_name)):
        folder_name = f"{original_folder}_{counter}"
        counter += 1

    full_path = os.path.join(base_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path, folder_name

def move_files_to_candidate_folder(temp_folder, candidate_folder):
    """Move all files from temp folder to candidate folder, preserving original names"""
    moved_files = []
    errors = []

    if not os.path.exists(temp_folder):
        return moved_files, ["Temporary folder not found"]

    try:
        for filename in os.listdir(temp_folder):
            src_path = os.path.join(temp_folder, filename)
            dst_path = os.path.join(candidate_folder, filename)

            if os.path.isfile(src_path):
                shutil.move(src_path, dst_path)
                moved_files.append(filename)
                print(f"[MOVE] Moved {filename} to candidate folder")

        # Remove empty temp folder
        if os.path.exists(temp_folder) and not os.listdir(temp_folder):
            os.rmdir(temp_folder)
            print(f"[CLEANUP] Removed empty temp folder: {temp_folder}")

    except Exception as e:
        errors.append(f"Error moving files: {str(e)}")

    return moved_files, errors



def format_file_info(file_path):
    """Get formatted file information"""
    if not os.path.exists(file_path):
        return None
    
    stat = os.stat(file_path)
    return {
        "filename": os.path.basename(file_path),
        "size_mb": round(get_file_size_mb(file_path), 2),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
    }
