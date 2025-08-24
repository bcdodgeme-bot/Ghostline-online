import os
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly"
]

def generate_token():
    # Make sure secrets directory exists
    os.makedirs("secrets", exist_ok=True)
    
    # Run OAuth flow
    flow = InstalledAppFlow.from_client_secrets_file(
        'secrets/gmail_credentials.json', SCOPES)
    
    # This will open your browser for authorization
    creds = flow.run_local_server(port=0)
    
    # Save the token
    with open('secrets/gmail_token.json', 'w') as token:
        token.write(creds.to_json())
    
    print("Token saved to secrets/gmail_token.json")
    print("Now commit this file to your repository")

if __name__ == '__main__':
    generate_token()