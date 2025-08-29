#!/usr/bin/env python3
"""
Test script for ChatGPT OCR integration
This script demonstrates how to test the ChatGPT filtering functionality
"""

import requests
import json

def test_chatgpt_ocr_filtering():
    """Test the ChatGPT OCR filtering endpoint"""
    
    # Test data - simulated OCR text with errors
    test_cases = [
        {
            "type": "passport_front",
            "text": """
            REPUBLIC 0F INDIA
            PASSF0RT
            Type/Type P
            C0untry C0de/C0de du pays IND
            Passpart N0./N0 du passepart AB1234567
            Surname/N0m de famille SM1TH
            Given Name(s)/Prén0m(s) J0HN DAV1D
            Nati0nality/Nati0nalité INDIAN
            Sex/Sexe M
            Date 0f Birth/Date de naissance 15/01/1990
            Place 0f Birth/Lieu de naissance MUMBAI
            """
        },
        {
            "type": "passport_back", 
            "text": """
            Address/Adresse:
            123 MA1N STREET
            MUMBAI MAHARASHTRA
            400001 IND1A
            
            Father's Name: RAJ SM1TH
            Mother's Name: PRIYA SM1TH
            """
        },
        {
            "type": "cdc",
            "text": """
            CERTIFICATE 0F C0MPETENCY
            CDC N0: CDC123456789
            IND0S N0: IND0S987654321
            Name: J0HN SM1TH
            Rank: AB SEAMAN
            """
        }
    ]
    
    print("🧪 Testing ChatGPT OCR Integration")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['type'].upper()}")
        print("-" * 30)
        
        try:
            response = requests.post(
                "http://localhost:5000/test-chatgpt-ocr",
                json={
                    "text": test_case["text"],
                    "type": test_case["type"]
                },
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Status:", result["status"])
                print("🤖 ChatGPT Enabled:", result["chatgpt_enabled"])
                print("🔑 API Key Configured:", result["api_key_configured"])
                print("📄 Document Type:", result["document_type"])
                print("📝 Raw Text Preview:", result["raw_text_preview"][:100] + "...")
                
                if result["extracted_data"]:
                    print("🎯 Extracted Data:")
                    for key, value in result["extracted_data"].items():
                        print(f"   {key}: {value}")
                else:
                    print("⚠️  No data extracted (API key may not be configured)")
                    
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Error: Could not connect to server. Make sure the backend is running on http://localhost:5000")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print("1. If 'API Key Configured: False' - Add your OpenAI API key to backend/.env")
    print("2. If 'ChatGPT Enabled: False' - Set ENABLE_CHATGPT_FILTERING=true in backend/.env")
    print("3. If extraction works - ChatGPT will clean OCR errors and extract structured data")
    print("4. If extraction fails - System will fallback to regex patterns")

if __name__ == "__main__":
    test_chatgpt_ocr_filtering()
