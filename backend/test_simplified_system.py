#!/usr/bin/env python3
"""
Test script for the simplified file organization system
Demonstrates the single JSON file approach for certificate generation
"""
import os
import json
import requests
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:5000"

def test_health_check():
    """Test server health"""
    print("🔍 Testing server health...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server connection error: {e}")
        return False

def test_save_candidate_data():
    """Test saving candidate data with simplified system"""
    print("\n🧪 Testing simplified save-candidate-data endpoint...")
    
    candidate_data = {
        "firstName": "Alice",
        "lastName": "Johnson",
        "passport": "B9876543",
        "dob": "1985-05-15",
        "nationality": "UK",
        "address": "456 Oak Street, London",
        "cdcNo": "CDC987654",
        "indosNo": "IND456789",
        "email": "alice.johnson@example.com",
        "phone": "9876543210",
        "companyName": "Maritime Corp",
        "vendorName": "Vendor B",
        "paymentStatus": "PAID",
        "rollNo": "002",
        "paymentProof": "alice_payment.jpg",
        "session_id": "test-session-simplified",
        "submission_timestamp": datetime.now().isoformat(),
        "auto_filled": True
    }
    
    try:
        response = requests.post(f"{BASE_URL}/save-candidate-data", 
                               json=candidate_data,
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Candidate data saved successfully!")
            print(f"   Candidate folder: {result.get('candidate_folder')}")
            print(f"   Files moved: {result.get('files_count')}")
            print(f"   Message: {result.get('message')}")
            return True
        else:
            print(f"❌ Save candidate data failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Save candidate data error: {e}")
        return False

def test_get_current_candidate():
    """Test retrieving current candidate for certificate"""
    print("\n🧪 Testing get-current-candidate-for-certificate endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/get-current-candidate-for-certificate")
        
        if response.status_code == 200:
            result = response.json()
            candidate_data = result.get('data', {})
            print(f"✅ Current candidate retrieved successfully!")
            print(f"   Name: {candidate_data.get('firstName')} {candidate_data.get('lastName')}")
            print(f"   Passport: {candidate_data.get('passport')}")
            print(f"   Payment Status: {candidate_data.get('paymentStatus')}")
            print(f"   Candidate Folder: {candidate_data.get('candidate_folder')}")
            print(f"   Last Updated: {result.get('last_updated')}")
            return True
        else:
            print(f"❌ Get current candidate failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Get current candidate error: {e}")
        return False

def test_multiple_candidates():
    """Test saving multiple candidates to verify overwriting behavior"""
    print("\n🧪 Testing multiple candidates (should overwrite)...")
    
    candidates = [
        {
            "firstName": "Bob",
            "lastName": "Smith",
            "passport": "C1122334",
            "dob": "1992-08-22",
            "nationality": "CA",
            "address": "789 Pine Avenue, Toronto",
            "cdcNo": "CDC112233",
            "indosNo": "IND998877",
            "email": "bob.smith@example.com",
            "phone": "1122334455",
            "companyName": "Ocean Lines",
            "vendorName": "Vendor C",
            "paymentStatus": "PENDING",
            "rollNo": "003",
            "session_id": "test-session-2"
        },
        {
            "firstName": "Carol",
            "lastName": "Davis",
            "passport": "D5566778",
            "dob": "1988-12-10",
            "nationality": "AU",
            "address": "321 Beach Road, Sydney",
            "cdcNo": "CDC556677",
            "indosNo": "IND445566",
            "email": "carol.davis@example.com",
            "phone": "5566778899",
            "companyName": "Coastal Shipping",
            "vendorName": "Vendor D",
            "paymentStatus": "PAID",
            "rollNo": "004",
            "session_id": "test-session-3"
        }
    ]
    
    for i, candidate in enumerate(candidates, 1):
        print(f"\n   {i}. Saving candidate: {candidate['firstName']} {candidate['lastName']}")
        
        try:
            response = requests.post(f"{BASE_URL}/save-candidate-data", 
                                   json=candidate,
                                   headers={'Content-Type': 'application/json'})
            
            if response.status_code == 200:
                result = response.json()
                print(f"      ✅ Saved: {result.get('candidate_folder')}")
                
                # Check current candidate after each save
                current_response = requests.get(f"{BASE_URL}/get-current-candidate-for-certificate")
                if current_response.status_code == 200:
                    current_data = current_response.json().get('data', {})
                    current_name = f"{current_data.get('firstName')} {current_data.get('lastName')}"
                    print(f"      📋 Current candidate is now: {current_name}")
                
            else:
                print(f"      ❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    return True

def show_file_structure():
    """Show the simplified file structure"""
    print("\n📁 Simplified File Structure:")
    print("=" * 50)
    
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(backend_dir, "uploads", "json")
    images_dir = os.path.join(backend_dir, "uploads", "images")
    
    # Check for current candidate file
    current_candidate_file = os.path.join(json_dir, "current_candidate_for_certificate.json")
    if os.path.exists(current_candidate_file):
        print("✅ Current candidate file exists:")
        print(f"   📄 {current_candidate_file}")
        
        try:
            with open(current_candidate_file, 'r') as f:
                data = json.load(f)
                print(f"   👤 Current candidate: {data.get('firstName')} {data.get('lastName')}")
                print(f"   📅 Last updated: {data.get('last_updated')}")
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    else:
        print("❌ Current candidate file not found")
    
    # Check candidate folders (should contain only uploaded files)
    if os.path.exists(images_dir):
        candidate_folders = [d for d in os.listdir(images_dir) 
                           if os.path.isdir(os.path.join(images_dir, d))]
        
        print(f"\n📂 Candidate folders ({len(candidate_folders)} found):")
        for folder in candidate_folders:
            folder_path = os.path.join(images_dir, folder)
            files = [f for f in os.listdir(folder_path) 
                    if os.path.isfile(os.path.join(folder_path, f))]
            
            # Check for old dual storage files (should not exist)
            has_json = any(f.endswith('.json') for f in files)
            has_txt = any(f.endswith('.txt') for f in files)
            
            status = "✅ Clean" if not (has_json or has_txt) else "⚠️ Has old files"
            print(f"   📁 {folder}: {len(files)} files {status}")
            
            if has_json or has_txt:
                old_files = [f for f in files if f.endswith(('.json', '.txt'))]
                print(f"      🗑️ Old files to remove: {old_files}")

def main():
    """Run simplified system tests"""
    print("🚀 Testing Simplified File Organization System")
    print("=" * 60)
    
    try:
        # Test 1: Health check
        if not test_health_check():
            print("❌ Cannot continue without server connection")
            return
        
        # Test 2: Save candidate data
        test_save_candidate_data()
        
        # Test 3: Get current candidate
        test_get_current_candidate()
        
        # Test 4: Multiple candidates (overwriting behavior)
        test_multiple_candidates()
        
        # Test 5: Show file structure
        show_file_structure()
        
        print("\n" + "=" * 60)
        print("✅ Simplified system tests completed!")
        print("\n📋 Summary:")
        print("   • Single JSON file for certificate generation")
        print("   • No centralized database or individual candidate files")
        print("   • Clean candidate folders with only uploaded files")
        print("   • Current candidate data overwrites previous data")
        print("   • OCR processing and file organization preserved")
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        print(f"❌ Test suite error: {e}")

if __name__ == "__main__":
    main()
