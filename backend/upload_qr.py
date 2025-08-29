import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from dotenv import load_dotenv
import qrcode

# Define the base directory for the backend
backend_dir = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file in the backend directory
dotenv_path = os.path.join(backend_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Flask config
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Google Drive config
SERVICE_ACCOUNT_FILE = os.path.join(backend_dir, os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE"))
SCOPES = [os.getenv("GOOGLE_DRIVE_SCOPES", "https://www.googleapis.com/auth/drive.file")]

def get_drive_service():
    """Authenticate with Google Drive using a service account"""
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE,
            scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=creds)
        print("[OK] Drive service (Service Account) created successfully")
        return service
    except Exception as e:
        print(f"[ERROR] Failed to create Drive service: {e}")
        raise

def upload_to_drive(file_path, filename):
    if not os.path.exists(SERVICE_ACCOUNT_FILE):
        raise Exception("Service account file not found")

    service = get_drive_service()

    # IMPORTANT: PASTE YOUR SHARED DRIVE ID HERE
    # This is the ID from the URL of your new Shared Drive.
    shared_drive_id = "PASTE_YOUR_SHARED_DRIVE_ID_HERE"



    file_metadata = {
        'name': filename,
        'parents': [shared_drive_id]
    }

    media = MediaFileUpload(file_path, resumable=True)
    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id',
        supportsAllDrives=True
    ).execute()

    file_id = uploaded_file.get('id')

    # Make file shareable
    service.permissions().create(
        fileId=file_id,
        body={'type': 'anyone', 'role': 'reader'},
    ).execute()

    link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    return link

def generate_qr_code(link, filename):
    img = qrcode.make(link)
    qr_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_qr.png")
    img.save(qr_path)
    return qr_path

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['pdf']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        drive_link = upload_to_drive(file_path, filename)
        qr_path = generate_qr_code(drive_link, filename)

        return jsonify({
            "status": "success",
            "drive_link": drive_link,
            "qr_code_path": qr_path
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
