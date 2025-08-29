import os
import json
import base64
import io
import cv2
import numpy as np
import pytesseract
import uuid
import re
import shutil
from datetime import datetime, timedelta

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    # Try common installation paths
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"[OK] Tesseract found at: {path}")
            break
    else:
        print("[WARNING] Tesseract not found in common paths. Please install Tesseract OCR.")
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from dotenv import load_dotenv
import qrcode
from PIL import Image
import openai
from openai import OpenAI

# Define the base directory for the backend
backend_dir = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file in the backend directory
dotenv_path = os.path.join(backend_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Flask config
app = Flask(__name__)
CORS(app)

# Configure upload folders
UPLOAD_FOLDER = os.path.join(backend_dir, "uploads")
IMAGES_FOLDER = os.path.join(UPLOAD_FOLDER, "images")
JSON_FOLDER = os.path.join(UPLOAD_FOLDER, "json")
PDFS_FOLDER = os.path.join(UPLOAD_FOLDER, "pdfs")
TEMP_FOLDER = os.path.join(UPLOAD_FOLDER, "temp")

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, IMAGES_FOLDER, JSON_FOLDER, PDFS_FOLDER, TEMP_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER
app.config['JSON_FOLDER'] = JSON_FOLDER
app.config['PDFS_FOLDER'] = PDFS_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# File upload settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Google Drive config
SERVICE_ACCOUNT_FILE = os.path.join(backend_dir, os.getenv("GOOGLE_DRIVE_SERVICE_ACCOUNT_FILE", "service-account.json"))
SCOPES = [os.getenv("GOOGLE_DRIVE_SCOPES", "https://www.googleapis.com/auth/drive.file")]

# Initialize OpenAI client
try:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and openai_api_key != "your_openai_api_key_here":
        # Initialize OpenAI client with proper parameters
        client = OpenAI(
            api_key=openai_api_key,
            timeout=30.0,
            max_retries=2
        )
        print("[OK] OpenAI client initialized successfully")

        # Test the client with a simple request
        try:
            # Make a test call to verify the API key works
            test_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            print("[OK] OpenAI API key verified and working")
        except Exception as test_error:
            print(f"[WARNING] OpenAI API key test failed: {test_error}")
            client = None
    else:
        client = None
        print("[WARNING] OpenAI API key not configured")
except Exception as e:
    print(f"[WARNING] OpenAI client initialization failed: {e}")
    print("[INFO] Falling back to regex-based text extraction")
    client = None

def enhanced_regex_filtering(raw_text, document_type="passport_front"):
    """Enhanced regex-based text filtering as fallback"""
    import re

    print(f"[REGEX] Processing {document_type} with enhanced regex patterns...")

    # Clean common OCR errors
    text = raw_text.upper()

    # Fix common OCR mistakes
    replacements = {
        'PASSF0RT': 'PASSPORT',
        'PASSPCRT': 'PASSPORT',
        'PASSP0RT': 'PASSPORT',
        'N0:': 'NO:',
        'N0.': 'NO.',
        'SURNAM3': 'SURNAME',
        'NAM3': 'NAME',
        'D0B': 'DOB',
        'BIRTH': 'BIRTH',
        '0': '0',  # Ensure zeros are consistent
        'O': '0',  # Convert O to 0 in numbers
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    result = {}

    if document_type == "passport_front":
        # Enhanced passport number patterns - more flexible
        passport_patterns = [
            r'PASSPORT\s*NO[.:\s]*([A-Z0-9]{6,12})',
            r'PASS\w*\s*NO[.:\s]*([A-Z0-9]{6,12})',
            r'NO[.:\s]*([A-Z0-9]{6,12})',
            r'([A-Z]{1,2}[0-9]{6,10})',  # Common passport format
            r'([0-9]{7,9})',  # Pure numeric passport numbers
            r'P\s*([A-Z0-9]{6,11})',  # P prefix format
        ]

        for pattern in passport_patterns:
            match = re.search(pattern, text)
            if match:
                passport_num = match.group(1).strip()
                if len(passport_num) >= 6:  # Minimum length check
                    result["Passport No."] = passport_num
                    print(f"[REGEX] Found passport: {result['Passport No.']}")
                    break

        # Enhanced name patterns - more flexible
        surname_patterns = [
            r'SURNAME[:\s]*([A-Z][A-Z\s]{1,30})(?:\s*\n|\s*GIVEN|\s*NAME|\s*$)',
            r'LAST\s*NAME[:\s]*([A-Z][A-Z\s]{1,30})(?:\s*\n|\s*GIVEN|\s*FIRST|\s*$)',
            r'([A-Z]{2,})\s*,\s*([A-Z\s]+)',  # "SURNAME, GIVEN NAMES" format
        ]

        for pattern in surname_patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) > 1:  # Comma-separated format
                    result["Surname"] = match.group(1).strip()
                    result["Given Name(s)"] = match.group(2).strip()
                    print(f"[REGEX] Found surname: {result['Surname']}")
                    print(f"[REGEX] Found given name: {result['Given Name(s)']}")
                else:
                    surname = match.group(1).strip()
                    if len(surname) >= 2:  # Minimum length check
                        result["Surname"] = surname
                        print(f"[REGEX] Found surname: {result['Surname']}")
                break

        # Given names patterns - only if not found above
        if not result.get("Given Name(s)"):
            given_patterns = [
                r'GIVEN\s*NAME[S]?[:\s]*([A-Z][A-Z\s]{1,30})(?:\s*\n|\s*NATIONALITY|\s*SEX|\s*$)',
                r'FIRST\s*NAME[:\s]*([A-Z][A-Z\s]{1,30})(?:\s*\n|\s*LAST|\s*SURNAME|\s*$)',
            ]

            for pattern in given_patterns:
                match = re.search(pattern, text)
                if match:
                    given_name = match.group(1).strip()
                    if len(given_name) >= 2:  # Minimum length check
                        result["Given Name(s)"] = given_name
                        print(f"[REGEX] Found given name: {result['Given Name(s)']}")
                        break

        # Additional patterns for nationality, sex, and date of birth
        nationality_patterns = [
            r'NATIONALITY[:\s]*([A-Z]{3})',
            r'NAT[:\s]*([A-Z]{3})',
        ]

        for pattern in nationality_patterns:
            match = re.search(pattern, text)
            if match:
                result["Nationality"] = match.group(1).strip()
                print(f"[REGEX] Found nationality: {result['Nationality']}")
                break

        sex_patterns = [
            r'SEX[:\s]*([MF])',
            r'GENDER[:\s]*([MF])',
        ]

        for pattern in sex_patterns:
            match = re.search(pattern, text)
            if match:
                result["Sex"] = match.group(1).strip()
                print(f"[REGEX] Found sex: {result['Sex']}")
                break

        # Date of birth patterns
        dob_patterns = [
            r'DATE\s*OF\s*BIRTH[:\s]*(\d{2}[/.-]\d{2}[/.-]\d{4})',
            r'DOB[:\s]*(\d{2}[/.-]\d{2}[/.-]\d{4})',
            r'BIRTH[:\s]*(\d{2}[/.-]\d{2}[/.-]\d{4})',
        ]

        for pattern in dob_patterns:
            match = re.search(pattern, text)
            if match:
                result["Date of Birth"] = match.group(1).strip()
                print(f"[REGEX] Found date of birth: {result['Date of Birth']}")
                break

    return result if result else None

def filter_text_with_chatgpt(raw_text, document_type="passport_front"):
    """Use ChatGPT to filter and extract structured data from OCR text"""
    if not client or not os.getenv("OPENAI_API_KEY"):
        print("[CHATGPT] API key not configured, using enhanced regex filtering")
        return enhanced_regex_filtering(raw_text, document_type)

    try:
        print(f"[CHATGPT] Processing {document_type} with AI...")

        if document_type == "passport_front":
            prompt = f"""Extract passport information from this OCR text. The text may contain errors and noise.
Please extract only the following fields and return as JSON:
- "Passport No.": passport number
- "Surname": last name/family name
- "Given Name(s)": first/given names
- "Nationality": nationality code (3 letters)
- "Sex": M or F
- "Date of Birth": format DD/MM/YYYY

OCR Text:
{raw_text}

Return only valid JSON with the extracted fields. If a field cannot be found, use empty string."""

        elif document_type == "passport_back":
            prompt = f"""Extract address information from this passport back page OCR text.
Please extract only the Address field and return as JSON:
- "Address": complete address

OCR Text:
{raw_text}

Return only valid JSON. If address cannot be found, use empty string."""

        elif document_type == "cdc":
            prompt = f"""Extract CDC certificate information from this OCR text.
Please extract only the following fields and return as JSON:
- "cdc_no": CDC number/certificate number
- "indos_no": INDOS number

OCR Text:
{raw_text}

Return only valid JSON. If a field cannot be found, use empty string."""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured data from passport and certificate documents. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
        )

        # Parse the JSON response
        content = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:-3]
        elif content.startswith("```"):
            content = content[3:-3]

        extracted_data = json.loads(content)
        print(f"[CHATGPT] Successfully extracted data: {extracted_data}")
        return extracted_data

    except Exception as e:
        print(f"[CHATGPT] Error filtering text: {e}")
        print("[CHATGPT] Falling back to enhanced regex filtering")
        return enhanced_regex_filtering(raw_text, document_type)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_session_id():
    """Generate a unique session ID for temporary file storage"""
    return str(uuid.uuid4())

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



def preprocess_image(image_path):
    """Enhanced image preprocessing for better OCR results - processes in memory only"""
    try:
        print(f"[PREPROCESS] Processing image in memory: {image_path}")

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return image_path, None

        print(f"[PREPROCESS] Original image size: {img.shape}")

        # Resize image if too small (OCR works better on larger images)
        height, width = img.shape[:2]
        if width < 1000:
            scale_factor = 1000 / width
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            print(f"[PREPROCESS] Resized to: {img.shape}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply multiple preprocessing techniques

        # 1. Noise reduction
        denoised = cv2.medianBlur(gray, 3)

        # 2. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)

        # 3. Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        # 4. Adaptive thresholding for better text separation
        adaptive_thresh = cv2.adaptiveThreshold(
            morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # 5. Also create OTSU threshold version
        _, otsu_thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Process in memory only - no file saving
        print(f"[PREPROCESS] Image processed in memory (no files saved)")

        # Return the original path and processed image array for OCR
        return image_path, otsu_thresh

    except Exception as e:
        print(f"[ERROR] Error preprocessing image: {e}")
        return image_path, None

def extract_passport_front_data(image_path):
    """Enhanced passport front data extraction with better accuracy"""
    try:
        print(f"[OCR] Processing passport front image: {image_path}")

        # Preprocess image in memory
        original_path, processed_image = preprocess_image(image_path)

        # Try multiple OCR approaches for better accuracy
        print("[OCR] Running enhanced OCR extraction...")

        # OCR configurations optimized for passport text - removed character whitelist for better accuracy
        configs = [
            '--psm 6',  # Uniform block of text
            '--psm 4',  # Single column of text
            '--psm 8',  # Single word
            '--psm 13', # Raw line
            '--psm 3',  # Full page
        ]

        all_texts = []
        for i, config in enumerate(configs):
            try:
                # Use processed image if available, otherwise use original
                if processed_image is not None:
                    # Convert numpy array to PIL Image for OCR
                    pil_image = Image.fromarray(processed_image)
                    text = pytesseract.image_to_string(pil_image, config=config)
                else:
                    text = pytesseract.image_to_string(Image.open(original_path), config=config)
                all_texts.append(text)
                print(f"[OCR] Config {i+1} extracted {len(text)} characters")
            except Exception as e:
                print(f"[OCR] Config {i+1} failed: {e}")
                continue

        # Select best text based on quality indicators, not just length
        best_text = ""
        best_score = 0

        for i, text in enumerate(all_texts):
            if not text.strip():
                continue

            # Score based on passport-specific patterns
            score = 0
            text_upper = text.upper()

            # Higher score for passport-related keywords
            if 'PASSPORT' in text_upper: score += 10
            if 'SURNAME' in text_upper: score += 8
            if 'GIVEN' in text_upper: score += 8
            if 'NATIONALITY' in text_upper: score += 5
            if 'SEX' in text_upper: score += 3
            if 'BIRTH' in text_upper: score += 3

            # Bonus for reasonable length
            if 100 < len(text) < 2000: score += 5

            # Penalty for too much noise (excessive special characters)
            noise_chars = sum(1 for c in text if c in '!@#$%^&*()[]{}|\\')
            score -= min(noise_chars, 10)

            print(f"[OCR] Config {i+1} score: {score}, length: {len(text)}")

            if score > best_score:
                best_score = score
                best_text = text

        # Fallback to longest if no good score found
        if not best_text and all_texts:
            best_text = max(all_texts, key=len)
            print("[OCR] Using longest text as fallback")

        print(f"[OCR] Selected text length: {len(best_text)} characters, score: {best_score}")
        print(f"[OCR] Raw text preview:\n{best_text[:300]}...")

        # Initialize data structure - only essential fields
        passport_data = {
            "Passport No.": "",
            "Surname": "",
            "Given Name(s)": "",
            "Nationality": "",
            "Sex": "",
            "Date of Birth": "",
            "raw_text": best_text
        }

        # Try ChatGPT filtering first
        if os.getenv("ENABLE_CHATGPT_FILTERING", "true").lower() == "true":
            print("[CHATGPT] Attempting AI-powered extraction...")
            chatgpt_data = filter_text_with_chatgpt(best_text, "passport_front")

            if chatgpt_data:
                print("[CHATGPT] AI extraction successful")
                # Update passport_data with ChatGPT results
                for key, value in chatgpt_data.items():
                    if value and value.strip():  # Only update if ChatGPT found a value
                        passport_data[key] = value.strip()
                        print(f"[CHATGPT] Extracted {key}: {value.strip()}")
            else:
                print("[CHATGPT] AI extraction failed, using fallback methods")

        # Process all extracted texts for better pattern matching
        combined_text = "\n".join(all_texts)
        text_upper = combined_text.upper()

        # Extract MRZ lines (Machine Readable Zone)
        print("[OCR] Extracting MRZ lines...")
        lines = combined_text.split('\n')
        mrz_lines = []
        for line in lines:
            clean_line = line.strip()
            # MRZ lines: 44 characters, contain < and alphanumeric
            if len(clean_line) >= 40 and '<' in clean_line and any(c.isalnum() for c in clean_line):
                mrz_lines.append(clean_line)

        if len(mrz_lines) >= 2:
            print(f"[OCR] Found MRZ lines: {mrz_lines}")

            # Extract only essential data from MRZ
            try:
                mrz1 = mrz_lines[0]
                mrz2 = mrz_lines[1]

                # MRZ Line 1: P<COUNTRY<SURNAME<<GIVEN_NAMES<<<<<<<<<<<<<<<
                if mrz1.startswith('P<'):
                    # Extract surname and given names from MRZ
                    name_part = mrz1[5:].split('<<')
                    if len(name_part) >= 2:
                        passport_data["Surname"] = name_part[0].replace('<', ' ').strip()
                        passport_data["Given Name(s)"] = name_part[1].replace('<', ' ').strip()

                # MRZ Line 2: PASSPORT_NO<COUNTRY<DOB<SEX<EXPIRY<PERSONAL_NO<CHECK
                if len(mrz2) >= 28:
                    passport_data["Passport No."] = mrz2[:9].replace('<', '').strip()
                    passport_data["Nationality"] = mrz2[10:13]

                    # Date of birth (YYMMDD)
                    dob = mrz2[13:19]
                    if dob.isdigit() and len(dob) == 6:
                        year = "19" + dob[:2] if int(dob[:2]) > 30 else "20" + dob[:2]
                        passport_data["Date of Birth"] = f"{dob[4:6]}/{dob[2:4]}/{year}"

                    # Sex
                    passport_data["Sex"] = mrz2[20]

            except Exception as e:
                print(f"[OCR] MRZ parsing error: {e}")

        # Fallback: Extract from regular text if MRZ parsing failed
        import re

        if not passport_data["Passport No."]:
            print("[OCR] Extracting passport number from text...")
            passport_patterns = [
                r'PASSPORT\s*NO[.:\s]*([A-Z0-9]{6,9})',
                r'DOCUMENT\s*NO[.:\s]*([A-Z0-9]{6,9})',
                r'NO[.:\s]*([A-Z0-9]{6,9})',
                r'\b([A-Z]{1,2}[0-9]{6,8})\b',
            ]

            for pattern in passport_patterns:
                match = re.search(pattern, text_upper)
                if match:
                    passport_data["Passport No."] = match.group(1)
                    print(f"[OCR] Found passport number: {passport_data['Passport No.']}")
                    break

        if not passport_data["Surname"]:
            print("[OCR] Extracting surname from text...")
            surname_patterns = [
                r'SURNAME[:\s]*([A-Z\s]+?)(?:\n|GIVEN|FIRST)',
                r'FAMILY\s*NAME[:\s]*([A-Z\s]+?)(?:\n|GIVEN|FIRST)',
                r'LAST\s*NAME[:\s]*([A-Z\s]+?)(?:\n|GIVEN|FIRST)',
            ]

            for pattern in surname_patterns:
                match = re.search(pattern, text_upper)
                if match:
                    passport_data["Surname"] = match.group(1).strip()
                    print(f"[OCR] Found surname: {passport_data['Surname']}")
                    break

        if not passport_data["Given Name(s)"]:
            print("[OCR] Extracting given names from text...")
            given_patterns = [
                r'GIVEN\s*NAMES?[:\s]*([A-Z\s]+?)(?:\n|NATIONALITY|SEX)',
                r'FIRST\s*NAME[:\s]*([A-Z\s]+?)(?:\n|NATIONALITY|SEX)',
            ]

            for pattern in given_patterns:
                match = re.search(pattern, text_upper)
                if match:
                    passport_data["Given Name(s)"] = match.group(1).strip()
                    print(f"[OCR] Found given names: {passport_data['Given Name(s)']}")
                    break

        print(f"[OCR] Final extracted passport data:")
        for key, value in passport_data.items():
            if key != "raw_text" and value:
                print(f"  {key}: {value}")

        # No temporary files to clean up (processing done in memory)
        return passport_data

    except Exception as e:
        print(f"[ERROR] Error extracting passport front data: {e}")
        import traceback
        traceback.print_exc()
        return {
            "Passport No.": "",
            "Surname": "",
            "Given Name(s)": "",
            "Nationality": "",
            "Sex": "",
            "Date of Birth": "",
            "raw_text": "",
            "error": str(e)
        }

def extract_passport_back_data(image_path):
    """Extract only address from passport back page using OCR"""
    try:
        print(f"[OCR] Processing passport back image: {image_path}")

        # Preprocess image in memory
        original_path, processed_image = preprocess_image(image_path)

        # Perform OCR with multiple configurations
        configs = [
            '--psm 6',  # Uniform block of text
            '--psm 4',  # Single column of text
            '--psm 3',  # Fully automatic page segmentation
        ]

        best_text = ""
        for config in configs:
            try:
                # Use processed image if available, otherwise use original
                if processed_image is not None:
                    # Convert numpy array to PIL Image for OCR
                    pil_image = Image.fromarray(processed_image)
                    text = pytesseract.image_to_string(pil_image, config=config)
                else:
                    text = pytesseract.image_to_string(Image.open(original_path), config=config)
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
            except:
                continue

        text = best_text
        print(f"[OCR] Passport back text length: {len(text)} characters")
        print(f"[OCR] Passport back text preview: {text[:200]}...")

        # Initialize data structure - only address needed
        passport_back_data = {
            "Address": "",
            "raw_text": text
        }

        # Try ChatGPT filtering first
        if os.getenv("ENABLE_CHATGPT_FILTERING", "true").lower() == "true":
            print("[CHATGPT] Attempting AI-powered address extraction...")
            chatgpt_data = filter_text_with_chatgpt(text, "passport_back")

            if chatgpt_data and chatgpt_data.get("Address"):
                print("[CHATGPT] AI address extraction successful")
                passport_back_data["Address"] = chatgpt_data["Address"].strip()
                print(f"[CHATGPT] Extracted Address: {passport_back_data['Address']}")
            else:
                print("[CHATGPT] AI address extraction failed, using fallback methods")

        # Extract address information using fallback methods if ChatGPT didn't find it
        if not passport_back_data["Address"]:
            lines = text.split('\n')
            text_upper = text.upper()

        # Look for address patterns
        import re
        address_patterns = [
            r'ADDRESS[:\s]*([A-Z0-9\s,.-]+?)(?:\n\n|\nFATHER|\nMOTHER|\nSPOUSE|$)',
            r'PERMANENT\s*ADDRESS[:\s]*([A-Z0-9\s,.-]+?)(?:\n\n|\nFATHER|\nMOTHER|\nSPOUSE|$)',
            r'RESIDENTIAL\s*ADDRESS[:\s]*([A-Z0-9\s,.-]+?)(?:\n\n|\nFATHER|\nMOTHER|\nSPOUSE|$)',
        ]

        for pattern in address_patterns:
            match = re.search(pattern, text_upper, re.DOTALL)
            if match:
                address = match.group(1).strip()
                # Clean up the address
                address = re.sub(r'\s+', ' ', address)  # Replace multiple spaces with single space
                passport_back_data["Address"] = address
                print(f"[OCR] Found address: {address}")
                break

        # If no pattern match, try simple line-by-line search
        if not passport_back_data["Address"]:
            for i, line in enumerate(lines):
                line_upper = line.upper()
                if 'ADDRESS' in line_upper:
                    # Take the next few lines as address
                    address_lines = []
                    for j in range(i+1, min(i+4, len(lines))):
                        if lines[j].strip() and not any(word in lines[j].upper() for word in ['FATHER', 'MOTHER', 'SPOUSE']):
                            address_lines.append(lines[j].strip())
                    if address_lines:
                        passport_back_data["Address"] = ' '.join(address_lines)
                        print(f"[OCR] Found address (line search): {passport_back_data['Address']}")
                    break

        print(f"[OCR] Final passport back data: {passport_back_data}")

        # No temporary files to clean up (processing done in memory)
        return passport_back_data

    except Exception as e:
        print(f"[ERROR] Error extracting passport back data: {e}")
        return {
            "Address": "",
            "raw_text": "",
            "error": str(e)
        }

def extract_cdc_data(image_path):
    """Extract CDC number and INDOS number from CDC image using OCR"""
    try:
        print(f"[OCR] Processing CDC image: {image_path}")

        # Preprocess image in memory
        original_path, processed_image = preprocess_image(image_path)

        # Perform OCR with multiple configurations
        configs = [
            '--psm 6',  # Uniform block of text
            '--psm 4',  # Single column of text
            '--psm 3',  # Fully automatic page segmentation
        ]

        best_text = ""
        for config in configs:
            try:
                # Use processed image if available, otherwise use original
                if processed_image is not None:
                    # Convert numpy array to PIL Image for OCR
                    pil_image = Image.fromarray(processed_image)
                    text = pytesseract.image_to_string(pil_image, config=config)
                else:
                    text = pytesseract.image_to_string(Image.open(original_path), config=config)
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
            except:
                continue

        text = best_text
        print(f"[OCR] CDC text length: {len(text)} characters")
        print(f"[OCR] CDC text preview: {text[:200]}...")

        # Initialize data structure
        cdc_data = {
            "cdc_no": "",
            "indos_no": "",
            "raw_text": text  # Include raw text for debugging
        }

        # Try ChatGPT filtering first
        if os.getenv("ENABLE_CHATGPT_FILTERING", "true").lower() == "true":
            print("[CHATGPT] Attempting AI-powered CDC extraction...")
            chatgpt_data = filter_text_with_chatgpt(text, "cdc")

            if chatgpt_data:
                print("[CHATGPT] AI CDC extraction successful")
                if chatgpt_data.get("cdc_no"):
                    cdc_data["cdc_no"] = chatgpt_data["cdc_no"].strip()
                    print(f"[CHATGPT] Extracted CDC No: {cdc_data['cdc_no']}")
                if chatgpt_data.get("indos_no"):
                    cdc_data["indos_no"] = chatgpt_data["indos_no"].strip()
                    print(f"[CHATGPT] Extracted INDOS No: {cdc_data['indos_no']}")
            else:
                print("[CHATGPT] AI CDC extraction failed, using fallback methods")

        # Extract CDC and INDOS numbers with fallback patterns if ChatGPT didn't find them
        if not cdc_data["cdc_no"] or not cdc_data["indos_no"]:
            text_upper = text.upper()
            import re

        # CDC number patterns
        cdc_patterns = [
            r'CDC\s*NO[.:]?\s*([A-Z0-9]{6,})',
            r'CDC[:\s]*([A-Z0-9]{6,})',
            r'CERTIFICATE\s*NO[.:]?\s*([A-Z0-9]{6,})',
        ]

        for pattern in cdc_patterns:
            match = re.search(pattern, text_upper)
            if match:
                cdc_data["cdc_no"] = match.group(1)
                print(f"[OCR] Found CDC number: {cdc_data['cdc_no']}")
                break

        # INDOS number patterns
        indos_patterns = [
            r'INDOS\s*NO[.:]?\s*([A-Z0-9]{6,})',
            r'INDOS[:\s]*([A-Z0-9]{6,})',
            r'IND[:\s]*([A-Z0-9]{6,})',
        ]

        for pattern in indos_patterns:
            match = re.search(pattern, text_upper)
            if match:
                cdc_data["indos_no"] = match.group(1)
                print(f"[OCR] Found INDOS number: {cdc_data['indos_no']}")
                break

        print(f"[OCR] Extracted CDC data: {cdc_data}")

        # No temporary files to clean up (processing done in memory)
        return cdc_data

    except Exception as e:
        print(f"[ERROR] Error extracting CDC data: {e}")
        return {
            "cdc_no": "",
            "indos_no": "",
            "raw_text": "",
            "error": str(e)
        }

# Google Drive functions (from existing code)
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
    """Upload file to Google Drive and return shareable link"""
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
    """Generate QR code for the given link"""
    img = qrcode.make(link)
    qr_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{filename}_qr.png")
    img.save(qr_path)

    # Convert to base64 for frontend
    with open(qr_path, "rb") as img_file:
        qr_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    return qr_path, qr_base64

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "success",
        "message": "Document Processing Server is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/upload-images', methods=['POST', 'OPTIONS'])
def upload_images():
    """
    Handle multiple file uploads and perform OCR processing
    Expected files: photo, signature, passport_front_img, passport_back_img, cdc_img (optional), marksheet (optional)
    """
    try:
        # Check if required files are present
        required_files = ['photo', 'signature', 'passport_front_img', 'passport_back_img']
        for file_key in required_files:
            if file_key not in request.files:
                return jsonify({"error": f"Missing required file: {file_key}"}), 400

            file = request.files[file_key]
            if file.filename == '':
                return jsonify({"error": f"No file selected for: {file_key}"}), 400

        # Generate session ID for this upload session
        session_id = generate_session_id()

        # Create temporary session folder
        temp_session_folder = os.path.join(app.config['TEMP_FOLDER'], session_id)
        os.makedirs(temp_session_folder, exist_ok=True)

        # Save uploaded files to temporary session folder
        saved_files = {}

        for file_key in ['photo', 'signature', 'passport_front_img', 'passport_back_img', 'cdc_img', 'marksheet']:
            if file_key in request.files:
                file = request.files[file_key]
                if file and file.filename != '' and allowed_file(file.filename):
                    # Keep original filename for temp storage
                    filename = secure_filename(file.filename)

                    # Save to temp session folder
                    file_path = os.path.join(temp_session_folder, filename)
                    file.save(file_path)
                    saved_files[file_key] = file_path

                    print(f"[TEMP UPLOAD] Saved {file_key}: {filename} to session {session_id}")

        # Perform OCR on passport and CDC images
        ocr_data = {}

        # Extract passport front data
        if 'passport_front_img' in saved_files:
            ocr_data['passport_front'] = extract_passport_front_data(saved_files['passport_front_img'])

        # Extract passport back data
        if 'passport_back_img' in saved_files:
            ocr_data['passport_back'] = extract_passport_back_data(saved_files['passport_back_img'])

        # Extract CDC data if provided
        if 'cdc_img' in saved_files:
            ocr_data['cdc'] = extract_cdc_data(saved_files['cdc_img'])
        else:
            ocr_data['cdc'] = {"cdc_no": "", "indos_no": ""}

        # Add session information to OCR data
        ocr_data['session_id'] = session_id
        ocr_data['temp_folder'] = temp_session_folder
        ocr_data['uploaded_files'] = {
            key: os.path.basename(path) for key, path in saved_files.items()
        }
        ocr_data['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        ocr_data['last_updated'] = datetime.now().isoformat()

        # Save OCR data to JSON file with session ID
        json_filename = f"structured_passport_data_{session_id}.json"
        json_path = os.path.join(app.config['JSON_FOLDER'], json_filename)
        with open(json_path, 'w') as json_file:
            json.dump(ocr_data, json_file, indent=2)

        print(f"[JSON] Saved OCR data: {json_filename}")
        print(f"[TEMP STORAGE] Files saved to session folder: {session_id}")
        for file_key, file_path in saved_files.items():
            print(f"[TEMP STORAGE] {file_key}: {os.path.basename(file_path)}")

        return jsonify({
            "status": "success",
            "message": "Files uploaded and processed successfully",
            "data": ocr_data,
            "session_id": session_id,
            "json_file": json_filename,
            "files_processed": len(saved_files),
            "temp_folder": temp_session_folder
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload-payment-screenshot', methods=['POST', 'OPTIONS'])
def upload_payment_screenshot():
    """Upload payment screenshot to session temp folder"""
    try:
        if 'paymentScreenshot' not in request.files:
            return jsonify({"error": "No payment screenshot provided"}), 400

        if 'sessionId' not in request.form:
            return jsonify({"error": "Session ID required"}), 400

        file = request.files['paymentScreenshot']
        session_id = request.form['sessionId']

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Validate file type (images only)
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Only images allowed"}), 400

        # Save to session temp folder
        temp_session_folder = os.path.join(app.config['TEMP_FOLDER'], session_id)
        if not os.path.exists(temp_session_folder):
            return jsonify({"error": "Invalid session ID or session expired"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(temp_session_folder, filename)
        file.save(file_path)

        print(f"[PAYMENT UPLOAD] Saved payment screenshot: {filename} to session {session_id}")

        return jsonify({
            "status": "success",
            "message": "Payment screenshot uploaded successfully",
            "filename": filename,
            "session_id": session_id
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save-candidate-data', methods=['POST'])
def save_candidate_data():
    """Save candidate form data and organize files into candidate folder"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract candidate information for folder naming
        first_name = sanitize_folder_name(data.get('firstName', ''))
        last_name = sanitize_folder_name(data.get('lastName', ''))
        passport_no = sanitize_folder_name(data.get('passport', ''))
        session_id = data.get('session_id', '')

        if not all([first_name, last_name, passport_no]):
            return jsonify({"error": "firstName, lastName, and passport are required for file organization"}), 400

        # Create candidate folder name
        candidate_folder_name = f"{first_name}_{last_name}_{passport_no}"

        # Create unique candidate folder in images directory
        candidate_folder_path, final_folder_name = create_unique_candidate_folder(
            app.config['IMAGES_FOLDER'],
            candidate_folder_name
        )

        # Move files from temp session folder to candidate folder
        moved_files = []
        if session_id:
            temp_session_folder = os.path.join(app.config['TEMP_FOLDER'], session_id)
            moved_files, move_errors = move_files_to_candidate_folder(
                temp_session_folder,
                candidate_folder_path
            )

            if move_errors:
                print(f"[WARNING] File move errors: {move_errors}")

        # Add metadata to candidate data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data.update({
            'timestamp': timestamp,
            'last_updated': datetime.now().isoformat(),
            'candidate_folder': final_folder_name,
            'candidate_folder_path': candidate_folder_path,
            'moved_files': moved_files,
            'session_id': session_id
        })

        # Save current candidate data for certificate generation
        current_candidate_filename = "current_candidate_for_certificate.json"
        current_candidate_path = os.path.join(app.config['JSON_FOLDER'], current_candidate_filename)

        try:
            with open(current_candidate_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"[JSON] Updated {current_candidate_filename} for certificate generation")

        except Exception as save_error:
            print(f"[ERROR] Failed to save current candidate for certificate: {save_error}")
            # Don't fail the entire request if this fails, just log the error

        print(f"[SUCCESS] Candidate data saved successfully")
        print(f"[SUCCESS] Files organized in folder: {final_folder_name}")
        print(f"[SUCCESS] Moved {len(moved_files)} files to candidate folder")

        return jsonify({
            "status": "success",
            "message": "Candidate data saved and files organized successfully",
            "filename": current_candidate_filename,
            "candidate_folder": final_folder_name,
            "moved_files": moved_files,
            "files_count": len(moved_files)
        }), 200

    except Exception as e:
        print(f"[ERROR] Save candidate data failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-candidate-data/<filename>', methods=['GET'])
def get_candidate_data(filename):
    """Retrieve candidate data by filename"""
    try:
        json_path = os.path.join(app.config['JSON_FOLDER'], filename)

        if not os.path.exists(json_path):
            return jsonify({"error": "File not found"}), 404

        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        return jsonify({
            "status": "success",
            "data": data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get-current-candidate-for-certificate', methods=['GET'])
def get_current_candidate_for_certificate():
    """STEP 2: Get current candidate data for certificate generation"""
    try:
        json_path = os.path.join(app.config['JSON_FOLDER'], 'current_candidate_for_certificate.json')

        if not os.path.exists(json_path):
            return jsonify({"error": "No current candidate data found"}), 404

        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

        print(f"[JSON] Retrieved current candidate data for certificate generation")
        return jsonify({
            "status": "success",
            "data": data
        }), 200

    except Exception as e:
        print(f"[ERROR] Failed to get current candidate for certificate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save-certificate-data', methods=['POST'])
def save_certificate_data():
    """Save certificate data to certificate_selections_for_receipt.json for receipt processing"""
    try:
        data = request.get_json()

        # Extract required fields
        firstName = data.get('firstName', '')
        lastName = data.get('lastName', '')
        certificateName = data.get('certificateName', '')
        companyName = data.get('companyName', '')
        rateData = data.get('rateData', {})

        if not firstName or not lastName or not certificateName:
            return jsonify({"error": "firstName, lastName, and certificateName are required"}), 400

        # Path to certificate selections file
        certificate_selections_path = os.path.join(app.config['JSON_FOLDER'], 'certificate_selections_for_receipt.json')

        # Load existing certificate selections or create empty array
        certificate_selections = []
        if os.path.exists(certificate_selections_path):
            try:
                with open(certificate_selections_path, 'r') as json_file:
                    certificate_selections = json.load(json_file)
                    if not isinstance(certificate_selections, list):
                        certificate_selections = []
            except (json.JSONDecodeError, Exception) as e:
                print(f"[WARNING] Error reading existing certificate selections, starting fresh: {e}")
                certificate_selections = []

        # Check for duplicates (same firstName + lastName + certificateName)
        duplicate_found = False
        for existing_cert in certificate_selections:
            if (existing_cert.get('firstName') == firstName and
                existing_cert.get('lastName') == lastName and
                existing_cert.get('certificateName') == certificateName):
                duplicate_found = True
                break

        if duplicate_found:
            return jsonify({
                "status": "warning",
                "message": f"Certificate already exists for {firstName} {lastName} - {certificateName}",
                "duplicate": True
            }), 200

        # Generate course-specific ID based on certificate name
        def generate_course_id(certificate_name, existing_certificates):
            # Map certificate names to prefixes
            course_prefixes = {
                'Basic Safety Training (STCW)': 'stcw',
                'H2S Training': 'h2s',
                'BOSIET Training': 'bosiet',
                'MODU Survival Training': 'modu',
                'Advanced Fire Fighting': 'aff',
                'Medical First Aid': 'mfa',
                'Personal Survival Techniques': 'pst',
                'Personal Safety and Social Responsibilities': 'pssr'
            }

            # Get prefix for the course
            prefix = course_prefixes.get(certificate_name, 'cert')

            # Count existing certificates with same prefix
            count = 1
            for cert in existing_certificates:
                if cert.get('id', '').startswith(f"{prefix}_"):
                    try:
                        existing_num = int(cert.get('id', '').split('_')[1])
                        if existing_num >= count:
                            count = existing_num + 1
                    except (ValueError, IndexError):
                        continue

            return f"{prefix}_{count:03d}"

        # Calculate amount from rate data if company is provided
        amount = 0
        if companyName and rateData and companyName in rateData:
            company_rates = rateData.get(companyName, {})
            amount = company_rates.get(certificateName, 0)

        # Check if this is a new candidate (different from existing certificates)
        current_candidate = f"{firstName.upper()} {lastName.upper()}"
        existing_candidates = set()

        for existing_cert in certificate_selections:
            existing_candidate = f"{existing_cert.get('firstName', '').upper()} {existing_cert.get('lastName', '').upper()}"
            existing_candidates.add(existing_candidate)

        # If this is a new candidate, clear all existing certificates
        is_new_candidate = len(existing_candidates) > 0 and current_candidate not in existing_candidates
        if is_new_candidate:
            print(f"[JSON] New candidate detected: {current_candidate}")
            print(f"[JSON] Clearing existing certificates for previous candidates: {existing_candidates}")
            certificate_selections = []  # Clear all existing certificates

        # Check for duplicates within current candidate's certificates
        duplicate_found = False
        for existing_cert in certificate_selections:
            if (existing_cert.get('firstName', '').upper() == firstName.upper() and
                existing_cert.get('lastName', '').upper() == lastName.upper() and
                existing_cert.get('certificateName', '') == certificateName):
                duplicate_found = True
                break

        if duplicate_found:
            print(f"[JSON] Duplicate certificate found for current candidate: {firstName} {lastName} - {certificateName}")
            return jsonify({
                "status": "success",
                "message": "Certificate already exists for current candidate",
                "duplicate": True,
                "total_certificates": len(certificate_selections)
            }), 200

        # Generate unique course-based ID (recalculate after potential clearing)
        unique_id = generate_course_id(certificateName, certificate_selections)

        # Create new certificate entry with enhanced structure
        certificate_entry = {
            'id': unique_id,  # Course-specific unique ID
            'firstName': firstName,
            'lastName': lastName,
            'certificateName': certificateName,
            'companyName': companyName,  # Use provided company name
            'amount': amount,            # Use calculated amount from rate list
            'timestamp': datetime.now().isoformat()
        }

        # Append to existing array
        certificate_selections.append(certificate_entry)

        # Save updated array back to file
        with open(certificate_selections_path, 'w') as json_file:
            json.dump(certificate_selections, json_file, indent=2)

        print(f"[JSON] Added certificate selection: {firstName} {lastName} - {certificateName}")

        return jsonify({
            "status": "success",
            "message": "Certificate data saved for receipt processing",
            "filename": "certificate_selections_for_receipt.json",
            "data": certificate_entry,
            "total_certificates": len(certificate_selections)
        }), 200

    except Exception as e:
        print(f"[ERROR] Failed to save certificate data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get-certificate-selections-for-receipt', methods=['GET'])
def get_certificate_selections_for_receipt():
    """Get certificate selections for receipt processing"""
    try:
        certificate_selections_path = os.path.join(app.config['JSON_FOLDER'], 'certificate_selections_for_receipt.json')

        # Return empty array if file doesn't exist
        if not os.path.exists(certificate_selections_path):
            print("[JSON] Certificate selections file not found, returning empty array")
            return jsonify({
                "status": "success",
                "data": [],
                "message": "No certificate selections found"
            }), 200

        # Load and return certificate selections
        with open(certificate_selections_path, 'r') as json_file:
            certificate_selections = json.load(json_file)

            # Ensure it's a list
            if not isinstance(certificate_selections, list):
                certificate_selections = []

        print(f"[JSON] Retrieved {len(certificate_selections)} certificate selections for receipt processing")

        return jsonify({
            "status": "success",
            "data": certificate_selections,
            "total_certificates": len(certificate_selections)
        }), 200

    except Exception as e:
        print(f"[ERROR] Failed to get certificate selections: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/update-certificate-company-data', methods=['POST'])
def update_certificate_company_data():
    """Update certificate selections with company name and amount data"""
    try:
        data = request.get_json()

        # Extract required fields
        certificate_ids = data.get('certificateIds', [])
        company_name = data.get('companyName', '')
        rate_data = data.get('rateData', {})

        if not certificate_ids or not company_name:
            return jsonify({"error": "certificateIds and companyName are required"}), 400

        # Path to certificate selections file
        certificate_selections_path = os.path.join(app.config['JSON_FOLDER'], 'certificate_selections_for_receipt.json')

        if not os.path.exists(certificate_selections_path):
            return jsonify({"error": "Certificate selections file not found"}), 404

        # Load certificate selections
        with open(certificate_selections_path, 'r') as json_file:
            certificate_selections = json.load(json_file)

        if not isinstance(certificate_selections, list):
            return jsonify({"error": "Invalid certificate selections format"}), 400

        # Update certificates with company and amount data
        updated_count = 0
        for cert in certificate_selections:
            if cert.get('id') in certificate_ids:
                cert['companyName'] = company_name

                # Get amount from rate data
                certificate_name = cert.get('certificateName', '')
                company_rates = rate_data.get(company_name, {})
                cert['amount'] = company_rates.get(certificate_name, 0)

                updated_count += 1

        # Save updated data
        with open(certificate_selections_path, 'w') as json_file:
            json.dump(certificate_selections, json_file, indent=2)

        print(f"[JSON] Updated {updated_count} certificates with company: {company_name}")

        return jsonify({
            "status": "success",
            "message": f"Updated {updated_count} certificates with company data",
            "updated_count": updated_count,
            "company_name": company_name
        }), 200

    except Exception as e:
        print(f"[ERROR] Failed to update certificate company data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete-certificate-selection', methods=['DELETE'])
def delete_certificate_selection():
    """Delete a certificate selection by ID"""
    try:
        data = request.get_json()
        certificate_id = data.get('id', '')

        if not certificate_id:
            return jsonify({"error": "Certificate ID is required"}), 400

        # Path to certificate selections file
        certificate_selections_path = os.path.join(app.config['JSON_FOLDER'], 'certificate_selections_for_receipt.json')

        if not os.path.exists(certificate_selections_path):
            return jsonify({"error": "Certificate selections file not found"}), 404

        # Load certificate selections
        with open(certificate_selections_path, 'r') as json_file:
            certificate_selections = json.load(json_file)

        if not isinstance(certificate_selections, list):
            return jsonify({"error": "Invalid certificate selections format"}), 400

        # Find and remove the certificate
        original_count = len(certificate_selections)
        certificate_selections = [cert for cert in certificate_selections if cert.get('id') != certificate_id]

        if len(certificate_selections) == original_count:
            return jsonify({"error": "Certificate not found"}), 404

        # Save updated data
        with open(certificate_selections_path, 'w') as json_file:
            json.dump(certificate_selections, json_file, indent=2)

        print(f"[JSON] Deleted certificate selection: {certificate_id}")

        return jsonify({
            "status": "success",
            "message": "Certificate selection deleted successfully",
            "deleted_id": certificate_id,
            "remaining_count": len(certificate_selections)
        }), 200

    except Exception as e:
        print(f"[ERROR] Failed to delete certificate selection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/update-certificate-selection', methods=['PUT'])
def update_certificate_selection():
    """Update a specific field in a certificate selection"""
    try:
        data = request.get_json()
        certificate_id = data.get('id', '')
        field = data.get('field', '')
        value = data.get('value', '')

        if not certificate_id or not field:
            return jsonify({"error": "Certificate ID and field are required"}), 400

        # Path to certificate selections file
        certificate_selections_path = os.path.join(app.config['JSON_FOLDER'], 'certificate_selections_for_receipt.json')

        if not os.path.exists(certificate_selections_path):
            return jsonify({"error": "Certificate selections file not found"}), 404

        # Load certificate selections
        with open(certificate_selections_path, 'r') as json_file:
            certificate_selections = json.load(json_file)

        if not isinstance(certificate_selections, list):
            return jsonify({"error": "Invalid certificate selections format"}), 400

        # Find and update the certificate
        updated = False
        for cert in certificate_selections:
            if cert.get('id') == certificate_id:
                # Handle different field mappings
                if field == 'candidateName':
                    # Split candidate name into firstName and lastName
                    name_parts = value.split(' ', 1)
                    cert['firstName'] = name_parts[0] if len(name_parts) > 0 else ''
                    cert['lastName'] = name_parts[1] if len(name_parts) > 1 else ''
                elif field == 'sales':
                    cert['amount'] = value
                else:
                    cert[field] = value
                updated = True
                break

        if not updated:
            return jsonify({"error": "Certificate not found"}), 404

        # Save updated data
        with open(certificate_selections_path, 'w') as json_file:
            json.dump(certificate_selections, json_file, indent=2)

        print(f"[JSON] Updated certificate selection {certificate_id}: {field} = {value}")

        return jsonify({
            "status": "success",
            "message": "Certificate selection updated successfully",
            "updated_id": certificate_id,
            "field": field,
            "value": value
        }), 200

    except Exception as e:
        print(f"[ERROR] Failed to update certificate selection: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup-expired-sessions', methods=['POST'])
def cleanup_expired_sessions():
    """Clean up temporary session folders older than specified hours"""
    try:
        hours_old = request.json.get('hours_old', 24) if request.json else 24  # Default 24 hours
        cutoff_time = datetime.now() - timedelta(hours=hours_old)

        cleaned_folders = []
        errors = []

        if os.path.exists(app.config['TEMP_FOLDER']):
            for session_folder in os.listdir(app.config['TEMP_FOLDER']):
                session_path = os.path.join(app.config['TEMP_FOLDER'], session_folder)

                if os.path.isdir(session_path):
                    folder_mtime = datetime.fromtimestamp(os.path.getmtime(session_path))

                    if folder_mtime < cutoff_time:
                        try:
                            shutil.rmtree(session_path)
                            cleaned_folders.append(session_folder)
                            print(f"[CLEANUP] Removed expired session: {session_folder}")
                        except Exception as e:
                            errors.append(f"Failed to remove {session_folder}: {str(e)}")

        return jsonify({
            "status": "success",
            "message": f"Cleanup completed. Removed {len(cleaned_folders)} expired sessions",
            "cleaned_folders": cleaned_folders,
            "errors": errors
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save-pdf', methods=['POST'])
def save_pdf():
    """Save generated PDF and upload to Google Drive"""
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400

        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save PDF locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(pdf_file.filename)
        name, ext = os.path.splitext(filename)
        filename = f"{timestamp}_{name}{ext}"

        pdf_path = os.path.join(app.config['PDFS_FOLDER'], filename)
        pdf_file.save(pdf_path)

        try:
            # Upload to Google Drive
            drive_link = upload_to_drive(pdf_path, filename)

            # Generate QR code
            qr_path, qr_base64 = generate_qr_code(drive_link, filename)

            return jsonify({
                "success": True,
                "status": "success",
                "drive_link": drive_link,
                "qr_image": qr_base64,
                "filename": filename,
                "storage_type": "google_drive"
            }), 200

        except Exception as drive_error:
            # If Google Drive upload fails, still save locally
            print(f"Google Drive upload failed: {drive_error}")

            # Generate local link and QR code
            local_link = f"http://localhost:5000/download-pdf/{filename}"
            qr_path, qr_base64 = generate_qr_code(local_link, filename)

            return jsonify({
                "success": True,
                "status": "success",
                "drive_link": local_link,
                "qr_image": qr_base64,
                "filename": filename,
                "storage_type": "local",
                "warning": "Google Drive upload failed, file saved locally"
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save-right-pdf', methods=['POST'])
def save_right_pdf():
    """Save right PDF locally only (no Google Drive, no QR generation)"""
    try:
        if 'pdf' not in request.files:
            return jsonify({"error": "No PDF file provided"}), 400

        pdf_file = request.files['pdf']
        if pdf_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save PDF locally only
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = secure_filename(pdf_file.filename)
        name, ext = os.path.splitext(filename)
        filename = f"{timestamp}_{name}{ext}"

        pdf_path = os.path.join(app.config['PDFS_FOLDER'], filename)
        pdf_file.save(pdf_path)

        print(f"[SUCCESS] Right PDF saved locally: {pdf_path}")

        return jsonify({
            "success": True,
            "status": "success",
            "local_link": f"http://localhost:5000/download-pdf/{filename}",
            "filename": filename,
            "message": "Right PDF saved to backend successfully"
        }), 200

    except Exception as e:
        print(f"[ERROR] Error saving right PDF: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download-pdf/<filename>', methods=['GET'])
def download_pdf(filename):
    """Download PDF file"""
    try:
        return send_from_directory(app.config['PDFS_FOLDER'], filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Legacy upload endpoint for backward compatibility"""
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
        qr_path, qr_base64 = generate_qr_code(drive_link, filename)

        return jsonify({
            "success": True,
            "status": "success",
            "drive_link": drive_link,
            "qr_image": qr_base64,
            "filename": filename,
            "storage_type": "google_drive"
        }), 200

    except Exception as e:
        # Fallback to local storage
        local_link = f"http://localhost:5000/download-pdf/{filename}"
        qr_path, qr_base64 = generate_qr_code(local_link, filename)

        return jsonify({
            "success": True,
            "status": "success",
            "drive_link": local_link,
            "qr_image": qr_base64,
            "filename": filename,
            "storage_type": "local",
            "warning": f"Google Drive upload failed: {str(e)}"
        }), 200

@app.route('/list-files', methods=['GET'])
def list_files():
    """List all uploaded files"""
    try:
        files = {
            "images": os.listdir(app.config['IMAGES_FOLDER']),
            "json": os.listdir(app.config['JSON_FOLDER']),
            "pdfs": os.listdir(app.config['PDFS_FOLDER'])
        }
        return jsonify({
            "status": "success",
            "files": files
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test-ocr', methods=['POST'])
def test_ocr():
    """Test OCR functionality with a single image"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{filename}")
        file.save(temp_path)

        # Perform OCR
        text = pytesseract.image_to_string(Image.open(temp_path))

        # Clean up
        os.remove(temp_path)

        return jsonify({
            "status": "success",
            "extracted_text": text
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/test-chatgpt-ocr', methods=['POST'])
def test_chatgpt_ocr():
    """Test ChatGPT OCR filtering with sample text"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        raw_text = data.get('text', '')
        doc_type = data.get('type', 'passport_front')

        if not raw_text:
            return jsonify({"error": "No text provided"}), 400

        if doc_type not in ['passport_front', 'passport_back', 'cdc']:
            return jsonify({"error": "Invalid document type. Use: passport_front, passport_back, or cdc"}), 400

        result = filter_text_with_chatgpt(raw_text, doc_type)

        return jsonify({
            "status": "success",
            "document_type": doc_type,
            "extracted_data": result,
            "raw_text_preview": raw_text[:200] + "..." if len(raw_text) > 200 else raw_text,
            "chatgpt_enabled": os.getenv("ENABLE_CHATGPT_FILTERING", "true").lower() == "true",
            "api_key_configured": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here")
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print(" DOCUMENT PROCESSING SERVER STARTING")
    print("=" * 60)
    print(f"[FOLDER] Upload folder: {UPLOAD_FOLDER}")
    print(f"[IMAGES] Images folder: {IMAGES_FOLDER}")
    print(f"[JSON] JSON folder: {JSON_FOLDER}")
    print(f"[PDFS] PDFs folder: {PDFS_FOLDER}")
    print(f"[TEMP] Temp folder: {TEMP_FOLDER}")
    print("=" * 60)
    print("[API] Available endpoints:")
    print("   GET  /                     - Health check")
    print("   POST /upload-images        - Upload multiple images to temp session")
    print("   POST /upload-payment-screenshot - Upload payment screenshot to session")
    print("   POST /save-candidate-data  - Save candidate data & organize files")
    print("   GET  /get-candidate-data/<filename> - Get candidate data (legacy)")
    print("   GET  /get-current-candidate-for-certificate - Get current candidate for certificate")
    print("   POST /cleanup-expired-sessions - Clean up old temp sessions")
    print("   POST /save-pdf             - Save PDF + upload to Drive")
    print("   POST /save-right-pdf       - Save right PDF locally only")
    print("   GET  /download-pdf/<filename> - Download PDF")
    print("   POST /upload               - Legacy PDF upload")
    print("   GET  /list-files           - List all files")
    print("   POST /test-ocr             - Test OCR with single image")
    print("   POST /test-chatgpt-ocr     - Test ChatGPT OCR filtering")
    print("=" * 60)
    print("[SERVER] Server will start on: http://localhost:5000")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)
