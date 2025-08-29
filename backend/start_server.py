#!/usr/bin/env python3
"""
Startup script for Document Processing Backend
This script performs basic checks and starts the Flask server
"""

import os
import sys
import subprocess
from app import app

def check_dependencies():
    """Check if critical dependencies are available"""
    print("🔍 Checking dependencies...")
    
    try:
        import pytesseract
        print("  ✅ pytesseract")
    except ImportError:
        print("  ❌ pytesseract not found. Run: pip install pytesseract")
        return False
    
    try:
        import cv2
        print("  ✅ opencv-python")
    except ImportError:
        print("  ❌ opencv-python not found. Run: pip install opencv-python")
        return False
    
    try:
        import numpy
        print("  ✅ numpy")
    except ImportError:
        print("  ❌ numpy not found. Run: pip install numpy")
        return False
    
    # Check Tesseract executable
    try:
        subprocess.run(["tesseract", "--version"], check=True, capture_output=True)
        print("  ✅ Tesseract OCR executable")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ⚠️  Tesseract executable not found in PATH")
        print("     OCR functionality may not work properly")
    
    return True

def check_directories():
    """Ensure required directories exist"""
    print("\n📁 Checking directories...")
    
    directories = [
        "uploads",
        "uploads/images",
        "uploads/json", 
        "uploads/pdfs"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"  ✅ Created {directory}")
        else:
            print(f"  ✅ {directory} exists")

def check_config():
    """Check configuration files"""
    print("\n⚙️  Checking configuration...")
    
    if os.path.exists('.env'):
        print("  ✅ .env file found")
    else:
        print("  ⚠️  .env file not found (using defaults)")
    
    if os.path.exists('service-account.json'):
        print("  ✅ Google Drive service account configured")
    else:
        print("  ⚠️  Google Drive not configured (will use local storage)")

def main():
    """Main startup function"""
    print("=" * 60)
    print("🚀 STARTING DOCUMENT PROCESSING SERVER")
    print("=" * 60)
    
    # Run checks
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        print("💡 Run: python setup.py")
        sys.exit(1)
    
    check_directories()
    check_config()
    
    print("\n" + "=" * 60)
    print("✅ All checks passed! Starting server...")
    print("=" * 60)
    
    # Start the Flask application
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
