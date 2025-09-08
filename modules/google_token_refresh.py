# modules/google_token_refresh.py
import os
import json
import datetime
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import logging

logger = logging.getLogger(__name__)

class GoogleTokenManager:
    """Handles automatic Google OAuth token refresh"""
    
    def __init__(self):
        self.token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        self.credentials = None
        self._last_refresh = None
    
    def load_credentials(self):
        """Load credentials from token file"""
        try:
            if os.path.exists(self.token_path):
                self.credentials = Credentials.from_authorized_user_file(
                    self.token_path, 
                    scopes=None  # Use scopes from saved token
                )
                logger.info(f"Loaded credentials from {self.token_path}")
                return True
            else:
                logger.warning(f"Token file not found: {self.token_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to load credentials: {e}")
            return False
    
    def get_valid_credentials(self):
        """Get valid credentials, refreshing if necessary"""
        if not self.credentials:
            if not self.load_credentials():
                return None
        
        # Check if token needs refresh
        if self.credentials.expired and self.credentials.refresh_token:
            try:
                logger.info("Token expired, attempting refresh...")
                self.credentials.refresh(Request())
                
                # Save refreshed token
                self.save_credentials()
                self._last_refresh = datetime.datetime.now()
                
                logger.info("Token refreshed successfully")
                return self.credentials
                
            except Exception as e:
                logger.error(f"Token refresh failed: {e}")
                # If refresh fails, the token file may be invalid
                self._handle_refresh_failure()
                return None
        
        elif self.credentials.expired and not self.credentials.refresh_token:
            logger.error("Token expired and no refresh token available - need re-authentication")
            return None
        
        # Token is still valid
        return self.credentials
    
    def save_credentials(self):
        """Save current credentials to file"""
        try:
            with open(self.token_path, 'w') as token_file:
                token_file.write(self.credentials.to_json())
            logger.info(f"Credentials saved to {self.token_path}")
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    def _handle_refresh_failure(self):
        """Handle cases where token refresh fails"""
        logger.warning("Token refresh failed - may need re-authentication")
        
        # Optionally, you could:
        # 1. Send an alert
        # 2. Invalidate the current token
        # 3. Set a flag for manual intervention
        
        # For now, just log the issue
        self.credentials = None
    
    def get_token_status(self):
        """Get detailed token status information"""
        if not self.credentials:
            self.load_credentials()
        
        if not self.credentials:
            return {
                'status': 'missing',
                'message': 'No token file found',
                'needs_auth': True
            }
        
        status = {
            'status': 'unknown',
            'valid': self.credentials.valid,
            'expired': self.credentials.expired,
            'has_refresh_token': bool(self.credentials.refresh_token),
            'scopes': list(self.credentials.scopes) if self.credentials.scopes else [],
            'last_refresh': self._last_refresh.isoformat() if self._last_refresh else None,
            'needs_auth': False
        }
        
        if self.credentials.valid:
            status['status'] = 'valid'
            status['message'] = 'Token is valid and ready to use'
        elif self.credentials.expired and self.credentials.refresh_token:
            status['status'] = 'expired_refreshable'
            status['message'] = 'Token expired but can be refreshed automatically'
        elif self.credentials.expired and not self.credentials.refresh_token:
            status['status'] = 'expired_no_refresh'
            status['message'] = 'Token expired and cannot be refreshed - need re-authentication'
            status['needs_auth'] = True
        else:
            status['status'] = 'unknown'
            status['message'] = 'Token status unclear'
        
        return status

# Global token manager instance
token_manager = GoogleTokenManager()

def get_google_credentials():
    """Get valid Google credentials with automatic refresh"""
    return token_manager.get_valid_credentials()

def force_token_refresh():
    """Force a token refresh (for testing/debugging)"""
    if token_manager.credentials and token_manager.credentials.refresh_token:
        try:
            token_manager.credentials.refresh(Request())
            token_manager.save_credentials()
            return True
        except Exception as e:
            logger.error(f"Forced refresh failed: {e}")
            return False
    return False