"""
Shared configuration for the backend application
"""
import os
from dotenv import load_dotenv

# Get the backend directory path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file in the backend directory
dotenv_path = os.path.join(backend_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Flask configuration
class Config:
    # Basic Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    HOST = os.getenv('FLASK_HOST', '0.0.0.0')
    PORT = int(os.getenv('FLASK_PORT', 5000))
    
    # File upload settings
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_FILE_SIZE', 50000000))  # 50MB default
    ALLOWED_EXTENSIONS = set(os.getenv('ALLOWED_EXTENSIONS', 'pdf,jpg,jpeg,png,doc,docx').split(','))
    
    # Directory paths
    UPLOAD_FOLDER = os.path.join(backend_dir, "uploads")
    IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
    JSON_FOLDER = os.path.join(UPLOAD_FOLDER, "json")
    PDFS_FOLDER = os.path.join(UPLOAD_FOLDER, "pdfs")
    TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, "temp")
    STATIC_FOLDER = os.path.join(backend_dir, "static")
    
    # Google Drive settings
    GOOGLE_DRIVE_CREDENTIALS_FILE = os.path.join(backend_dir, os.getenv('GOOGLE_DRIVE_CREDENTIALS_FILE', 'credentials.json'))
    GOOGLE_DRIVE_TOKEN_FILE = os.path.join(backend_dir, os.getenv('GOOGLE_DRIVE_TOKEN_FILE', 'token.json'))
    GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE = os.path.join(backend_dir, os.getenv('GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE', 'service-account.json'))
    GOOGLE_DRIVE_SCOPES = [os.getenv('GOOGLE_DRIVE_SCOPES', 'https://www.googleapis.com/auth/drive.file')]
    
    # QR Code settings
    QR_VERSION = int(os.getenv('QR_VERSION', 1))
    QR_BOX_SIZE = int(os.getenv('QR_BOX_SIZE', 10))
    QR_BORDER = int(os.getenv('QR_BORDER', 4))
    
    # OCR settings
    TESSERACT_CMD = os.getenv('TESSERACT_CMD', None)  # Will use system default if None
    
    # ChatGPT settings
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ENABLE_CHATGPT_FILTERING = os.getenv('ENABLE_CHATGPT_FILTERING', 'true').lower() == 'true'
    
    # Application settings
    APP_NAME = os.getenv('APP_NAME', 'Certificate Management System')
    BASE_URL = os.getenv('BASE_URL', f'http://localhost:{PORT}')

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        Config.UPLOAD_FOLDER,
        Config.IMAGES_FOLDER,
        Config.JSON_FOLDER,
        Config.PDFS_FOLDER,
        Config.TEMP_FOLDER,
        Config.STATIC_FOLDER
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directory ensured: {directory}")

def validate_config():
    """Validate configuration and return any warnings"""
    warnings = []
    
    # Check Google Drive credentials
    if not os.path.exists(Config.GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE):
        warnings.append(f"Google Drive service account file not found: {Config.GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE}")
    
    # Check OpenAI API key
    if Config.ENABLE_CHATGPT_FILTERING and not Config.OPENAI_API_KEY:
        warnings.append("ChatGPT filtering is enabled but OPENAI_API_KEY is not set")
    
    # Check Tesseract
    if Config.TESSERACT_CMD and not os.path.exists(Config.TESSERACT_CMD):
        warnings.append(f"Tesseract executable not found: {Config.TESSERACT_CMD}")
    
    return warnings
