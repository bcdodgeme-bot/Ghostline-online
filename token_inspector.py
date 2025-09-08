#!/usr/bin/env python3
"""
Inspect token.json to check refresh token status
"""

import json
import os
from datetime import datetime

def inspect_token():
    """Check the token structure and refresh capability"""
    
    token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
    
    if not os.path.exists(token_path):
        print("❌ No token.json found")
        return
    
    try:
        with open(token_path, 'r') as f:
            token_data = json.load(f)
        
        print("🔍 TOKEN ANALYSIS:")
        print("=" * 50)
        
        # Check expiry
        expiry = token_data.get('expiry')
        if expiry:
            expiry_dt = datetime.fromisoformat(expiry.replace('Z', '+00:00'))
            now = datetime.now(expiry_dt.tzinfo)
            time_left = expiry_dt - now
            
            print(f"Access Token Expiry: {expiry}")
            print(f"Time until expiry: {time_left}")
            print(f"Token expired: {'YES' if time_left.total_seconds() <= 0 else 'NO'}")
        else:
            print("Access Token Expiry: Not found")
        
        # Check refresh token
        refresh_token = token_data.get('refresh_token')
        if refresh_token:
            print(f"Refresh Token: PRESENT (length: {len(refresh_token)} chars)")
            print("✅ GOOD: Refresh token available for automatic renewal")
        else:
            print("Refresh Token: ❌ MISSING")
            print("🚨 PROBLEM: No refresh token - will need manual re-auth")
        
        # Check other fields
        print(f"\nToken Type: {token_data.get('token', 'Unknown')}")
        print(f"Client ID: {token_data.get('client_id', 'Not found')[:20]}...")
        print(f"Client Secret: {'Present' if token_data.get('client_secret') else 'Missing'}")
        
        # Check scopes
        scopes = token_data.get('scopes', [])
        if scopes:
            print(f"\nGranted Scopes ({len(scopes)}):")
            for scope in scopes:
                scope_name = scope.split('/')[-1]
                print(f"  - {scope_name}")
        
        print("\n" + "=" * 50)
        
        if refresh_token:
            print("✅ DIAGNOSIS: Token is properly configured")
            print("   - Access token expires in 1 hour (normal)")
            print("   - Refresh token will automatically get new access tokens")
            print("   - Should work indefinitely without manual intervention")
            print("\n💡 If you're still getting expiry errors in Ghostline:")
            print("   1. Check that the app is calling credentials.refresh()")
            print("   2. Verify the refreshed token is being saved back to file")
            print("   3. Make sure GOOGLE_TOKEN_PATH environment variable is correct")
        else:
            print("❌ DIAGNOSIS: Token missing refresh capability")
            print("   - You'll need to re-authenticate every hour")
            print("   - This happens when prompt='consent' wasn't used")
            print("   - Or when you've already granted permissions recently")
            print("\n🔧 SOLUTIONS:")
            print("   1. Revoke existing permissions at https://myaccount.google.com/permissions")
            print("   2. Run the token creation script again")
            print("   3. Make sure to use prompt='consent' in OAuth flow")
        
        return refresh_token is not None
        
    except Exception as e:
        print(f"❌ Failed to inspect token: {e}")
        return False

def check_refresh_mechanism():
    """Test if the refresh mechanism actually works"""
    print("\n🔄 TESTING REFRESH MECHANISM:")
    print("=" * 50)
    
    try:
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        
        token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        credentials = Credentials.from_authorized_user_file(token_path)
        
        print(f"Before refresh - Valid: {credentials.valid}")
        print(f"Before refresh - Expired: {credentials.expired}")
        
        if credentials.refresh_token:
            print("Attempting refresh...")
            try:
                credentials.refresh(Request())
                print("✅ Refresh successful!")
                print(f"After refresh - Valid: {credentials.valid}")
                print(f"New expiry: {credentials.expiry}")
                
                # Save the refreshed token
                with open(token_path, 'w') as f:
                    f.write(credentials.to_json())
                print("✅ Refreshed token saved to file")
                
                return True
                
            except Exception as refresh_error:
                print(f"❌ Refresh failed: {refresh_error}")
                return False
        else:
            print("❌ No refresh token available")
            return False
            
    except Exception as e:
        print(f"❌ Refresh test failed: {e}")
        return False

if __name__ == "__main__":
    has_refresh = inspect_token()
    
    if has_refresh:
        check_refresh_mechanism()
    else:
        print("\n🚨 IMMEDIATE ACTION REQUIRED:")
        print("Your token will expire soon and cannot auto-refresh.")
        print("Run the complete token reset script to fix this.")