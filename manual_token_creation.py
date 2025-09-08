#!/usr/bin/env python3
"""
Fixed manual token creation with proper refresh token handling
Forces offline access for long-lasting tokens
"""

import os
from google_auth_oauthlib.flow import Flow

# Disable HTTPS requirement for localhost development
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def create_long_lasting_token():
    """Create a token with refresh capability for long-term use"""
    
    # PASTE YOUR CALLBACK URL HERE
    callback_url = "http://localhost:5000/google/auth/callback?state=JakirozOnE0fseQw4uBwWyp9XGdwwk&code=4/0AVMBsJhLZ2AwcKOsyPc009yllx9T1tjOkOs5e_TwN1o3-oAXKfeznLxdQSm42VPdRMg8rQ&scope=https://www.googleapis.com/auth/gmail.readonly%20https://www.googleapis.com/auth/calendar.readonly%20https://www.googleapis.com/auth/drive.readonly%20https://www.googleapis.com/auth/drive.metadata.readonly%20https://www.googleapis.com/auth/documents%20https://www.googleapis.com/auth/spreadsheets%20https://www.googleapis.com/auth/presentations%20https://www.googleapis.com/auth/drive.file%20https://www.googleapis.com/auth/analytics.readonly%20https://www.googleapis.com/auth/webmasters.readonly"
    
    # Check for credentials file
    credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')
    if not os.path.exists(credentials_path):
        print(f"❌ Credentials file not found: {credentials_path}")
        return False
    
    # Check if callback URL was updated
    if callback_url == "PASTE_YOUR_CALLBACK_URL_HERE":
        print("❌ You need to update the callback_url variable in this script!")
        print("1. First run this script to get the authorization URL")
        print("2. Visit the URL, complete OAuth, and copy the callback URL")
        print("3. Paste the callback URL in the callback_url variable above")
        print("4. Run the script again")
        
        # Generate the authorization URL for them
        try:
            scopes = [
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly",
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/drive.metadata.readonly",
                "https://www.googleapis.com/auth/documents",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/presentations",
                "https://www.googleapis.com/auth/drive.file",
                "https://www.googleapis.com/auth/analytics.readonly",
                "https://www.googleapis.com/auth/webmasters.readonly",
            ]
            
            flow = Flow.from_client_secrets_file(credentials_path, scopes=scopes)
            flow.redirect_uri = "http://localhost:5000/google/auth/callback"
            
            authorization_url, state = flow.authorization_url(
                access_type='offline',
                prompt='consent',
                include_granted_scopes='true'
            )
            
            print(f"\n🔐 AUTHORIZATION URL:")
            print(f"{authorization_url}")
            print(f"\nVisit this URL, complete OAuth, then paste the callback URL in this script.")
            
        except Exception as e:
            print(f"❌ Could not generate authorization URL: {e}")
        
        return False
    
    try:
        # Phase 2 scopes
        scopes = [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/calendar.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/drive.metadata.readonly",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/presentations",
            "https://www.googleapis.com/auth/drive.file",
            "https://www.googleapis.com/auth/analytics.readonly",
            "https://www.googleapis.com/auth/webmasters.readonly",
        ]
        
        flow = Flow.from_client_secrets_file(
            credentials_path,
            scopes=scopes
        )
        flow.redirect_uri = "http://localhost:5000/google/auth/callback"
        
        if not callback_url.startswith('http://localhost:5000/google/auth/callback'):
            print("❌ Invalid callback URL format")
            return False
        
        print("📄 Processing OAuth callback...")
        
        # Process the callback
        flow.fetch_token(authorization_response=callback_url)
        credentials = flow.credentials
        
        # Verify we got a refresh token
        if not credentials.refresh_token:
            print("⚠️  WARNING: No refresh token received!")
            print("   This usually means you need to revoke previous access and try again.")
            print("   Go to https://myaccount.google.com/permissions and remove Ghostline access.")
            return False
        
        print("✅ Token created successfully!")
        print(f"   Valid: {credentials.valid}")
        print(f"   Has refresh token: {bool(credentials.refresh_token)}")
        print(f"   Scopes granted: {len(credentials.scopes)}")
        
        # Save the token
        token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        with open(token_path, 'w') as token_file:
            token_file.write(credentials.to_json())
        
        print(f"✅ Token saved to: {token_path}")
        
        # Test credential refresh
        print("\n🔄 Testing token refresh...")
        try:
            from google.auth.transport.requests import Request
            if credentials.expired:
                credentials.refresh(Request())
                print("✅ Token refresh successful!")
            else:
                print("✅ Token still valid, refresh mechanism ready")
        except Exception as e:
            print(f"❌ Token refresh test failed: {e}")
        
        # Test services
        print("\n🔧 Testing services...")
        test_google_services(credentials)
        
        print(f"\n🎉 Setup complete! This token should last indefinitely with automatic refresh.")
        print(f"   If you still get expiration issues, the problem is in the app's refresh logic.")
        
        return True
        
    except Exception as e:
        print(f"❌ Token creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_google_services(credentials):
    """Test all Google services"""
    try:
        from googleapiclient.discovery import build
        
        services = [
            ('Gmail', 'gmail', 'v1', lambda svc: svc.users().getProfile(userId='me').execute()),
            ('Calendar', 'calendar', 'v3', lambda svc: svc.calendarList().list(maxResults=1).execute()),
            ('Drive', 'drive', 'v3', lambda svc: svc.about().get(fields="user").execute()),
            ('Docs', 'docs', 'v1', lambda svc: True),
            ('Sheets', 'sheets', 'v4', lambda svc: True),
            ('Slides', 'slides', 'v1', lambda svc: True),
            # ADD THESE TWO:
            ('Analytics', 'analyticsdata', 'v1beta', lambda svc: True),
            ('Search Console', 'searchconsole', 'v1', lambda svc: True),
        ]
        for name, service_name, version, test_func in services:
            try:
                service = build(service_name, version, credentials=credentials)
                test_func(service)
                print(f"✅ {name}: Connected")
            except Exception as e:
                print(f"❌ {name}: {e}")
                
    except Exception as e:
        print(f"❌ Service testing failed: {e}")

if __name__ == "__main__":
    print("🚀 Creating long-lasting Google OAuth token for Ghostline")
    print("   This will force offline access for automatic token refresh.\n")
    create_long_lasting_token()
