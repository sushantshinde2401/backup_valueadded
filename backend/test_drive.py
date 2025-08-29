import os
import io
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Test Google Drive connection
def test_drive_connection():
    try:
        print("Testing Google Drive connection...")
        
        # Check if credentials file exists
        if not os.path.exists('credentials.json'):
            print("ERROR: credentials.json not found")
            return False
        
        print("[OK] credentials.json found")
        
        # Check if token file exists
        if os.path.exists('token.json'):
            print("[OK] token.json found")
        else:
            print("[INFO] token.json not found (will need to authenticate)")
        
        # Try to load credentials
        creds = None
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/drive.file'])
            print("[OK] Loaded existing token")
        
        # Test connection
        if creds and creds.valid:
            service = build('drive', 'v3', credentials=creds)
            # Try a simple API call
            results = service.files().list(pageSize=1, fields="files(id, name)").execute()
            items = results.get('files', [])
            print(f"[OK] Google Drive connection successful. Found {len(items)} files.")
            return True
        else:
            print("[INFO] Credentials not valid or missing. Would need to re-authenticate.")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == '__main__':
    test_drive_connection()
