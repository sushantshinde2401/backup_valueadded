#!/usr/bin/env python3
"""
Setup script for Document Processing Backend
This script installs all required dependencies and sets up the environment
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_tesseract():
    """Install Tesseract OCR based on the operating system"""
    system = platform.system().lower()
    
    if system == "windows":
        print("\n📋 For Windows, please install Tesseract manually:")
        print("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("2. Install the executable")
        print("3. Add Tesseract to your PATH environment variable")
        print("4. Typical installation path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        
        # Check if tesseract is already available
        try:
            subprocess.run(["tesseract", "--version"], check=True, capture_output=True)
            print("✅ Tesseract is already installed and available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Tesseract not found in PATH. Please install it manually.")
            return False
            
    elif system == "darwin":  # macOS
        print("\n🍎 Installing Tesseract on macOS...")
        if run_command("brew install tesseract", "Installing Tesseract via Homebrew"):
            return True
        else:
            print("❌ Failed to install Tesseract. Please install Homebrew first:")
            print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False
            
    elif system == "linux":
        print("\n🐧 Installing Tesseract on Linux...")
        # Try different package managers
        commands = [
            ("apt-get update && apt-get install -y tesseract-ocr", "Installing via apt-get"),
            ("yum install -y tesseract", "Installing via yum"),
            ("dnf install -y tesseract", "Installing via dnf"),
            ("pacman -S tesseract", "Installing via pacman")
        ]
        
        for cmd, desc in commands:
            if run_command(cmd, desc):
                return True
        
        print("❌ Could not install Tesseract automatically. Please install manually:")
        print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("   CentOS/RHEL: sudo yum install tesseract")
        print("   Fedora: sudo dnf install tesseract")
        print("   Arch: sudo pacman -S tesseract")
        return False
    
    return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("🚀 DOCUMENT PROCESSING BACKEND SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install Python dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Install Tesseract OCR
    print("\n📖 Setting up Tesseract OCR...")
    tesseract_installed = install_tesseract()
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = ["uploads", "uploads/images", "uploads/json", "uploads/pdfs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Check if Google Drive credentials exist
    print("\n🔐 Checking Google Drive setup...")
    if os.path.exists("service-account.json"):
        print("✅ Google Drive service account file found")
    else:
        print("⚠️  Google Drive service account file not found")
        print("   Please add your service-account.json file to enable Google Drive integration")
    
    # Final status
    print("\n" + "=" * 60)
    print("📋 SETUP SUMMARY")
    print("=" * 60)
    print("✅ Python dependencies installed")
    print(f"{'✅' if tesseract_installed else '⚠️ '} Tesseract OCR {'installed' if tesseract_installed else 'needs manual installation'}")
    print("✅ Directories created")
    print(f"{'✅' if os.path.exists('service-account.json') else '⚠️ '} Google Drive {'configured' if os.path.exists('service-account.json') else 'needs configuration'}")
    
    if tesseract_installed and os.path.exists("service-account.json"):
        print("\n🎉 Setup completed successfully!")
        print("🚀 You can now run: python app.py")
    else:
        print("\n⚠️  Setup completed with warnings. Please address the issues above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
