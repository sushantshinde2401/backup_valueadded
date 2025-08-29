#!/usr/bin/env python3
"""
Test script to verify the Document Processing Backend setup
"""

import os
import sys
import subprocess
import requests
import time
from PIL import Image
import io

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing Python imports...")
    
    modules = [
        'flask', 'flask_cors', 'werkzeug', 'google.oauth2.service_account',
        'googleapiclient.discovery', 'qrcode', 'PIL', 'cv2', 'numpy', 'pytesseract'
    ]
    
    failed_imports = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    return len(failed_imports) == 0

def test_tesseract():
    """Test if Tesseract OCR is working"""
    print("\n🔍 Testing Tesseract OCR...")
    
    try:
        import pytesseract
        from PIL import Image
        
        # Create a simple test image with text
        img = Image.new('RGB', (200, 50), color='white')
        
        # Try to extract text (should return empty or minimal text)
        text = pytesseract.image_to_string(img)
        print("  ✅ Tesseract is working")
        return True
        
    except Exception as e:
        print(f"  ❌ Tesseract test failed: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    print("\n📁 Testing directories...")
    
    required_dirs = [
        'uploads',
        'uploads/images', 
        'uploads/json',
        'uploads/pdfs'
    ]
    
    all_exist = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"  ✅ {directory}")
        else:
            print(f"  ❌ {directory} (missing)")
            all_exist = False
    
    return all_exist

def test_google_drive_config():
    """Test Google Drive configuration"""
    print("\n🔐 Testing Google Drive configuration...")
    
    if os.path.exists('service-account.json'):
        print("  ✅ service-account.json found")
        
        try:
            import json
            with open('service-account.json', 'r') as f:
                config = json.load(f)
            
            required_keys = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email']
            missing_keys = [key for key in required_keys if key not in config]
            
            if missing_keys:
                print(f"  ⚠️  Missing keys in service-account.json: {missing_keys}")
                return False
            else:
                print("  ✅ service-account.json appears valid")
                return True
                
        except Exception as e:
            print(f"  ❌ Error reading service-account.json: {e}")
            return False
    else:
        print("  ⚠️  service-account.json not found (Google Drive features will be limited)")
        return False

def test_server_startup():
    """Test if the Flask server can start"""
    print("\n🌐 Testing Flask server startup...")
    
    try:
        # Start the server in a subprocess
        import subprocess
        import time
        
        print("  🔄 Starting server...")
        process = subprocess.Popen([sys.executable, 'app.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Test if server is responding
        try:
            response = requests.get('http://localhost:5000/', timeout=5)
            if response.status_code == 200:
                print("  ✅ Server started successfully")
                print("  ✅ Health check endpoint working")
                success = True
            else:
                print(f"  ❌ Server responded with status {response.status_code}")
                success = False
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Could not connect to server: {e}")
            success = False
        
        # Stop the server
        process.terminate()
        process.wait(timeout=5)
        
        return success
        
    except Exception as e:
        print(f"  ❌ Server startup test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 DOCUMENT PROCESSING BACKEND TESTS")
    print("=" * 60)
    
    tests = [
        ("Python Imports", test_imports),
        ("Tesseract OCR", test_tesseract),
        ("Directories", test_directories),
        ("Google Drive Config", test_google_drive_config),
        ("Server Startup", test_server_startup)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("🚀 Run 'python app.py' to start the server")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        print("💡 Run 'python setup.py' to fix common issues")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
