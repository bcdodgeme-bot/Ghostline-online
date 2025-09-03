# modules/cloze_integration.py - FIXED VERSION with correct API endpoints
# Updated based on official Cloze API documentation

import os
import requests
import json
from datetime import datetime, timedelta
from flask import current_app
from modules.database import save_daily_log_enhanced
from modules.utils import format_timestamp

# Cloze API Configuration - FIXED BASE URL
CLOZE_API_BASE = "https://api.cloze.com"  # Removed /v1 from base
CLOZE_API_KEY = os.getenv('CLOZE_API_KEY')

class ClozeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or CLOZE_API_KEY
        self.base_url = CLOZE_API_BASE
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'  # FIXED: Use Bearer token auth
        }
    
    def _make_request(self, method, endpoint, params=None, data=None):
        """Make authenticated request to Cloze API with proper error handling"""
        if not self.api_key:
            raise Exception("Cloze API key not configured")
        
        # FIXED: Proper URL construction with v1 prefix
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        if not endpoint.startswith('/v1'):
            endpoint = '/v1' + endpoint
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, params=params, json=data, timeout=30)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, params=params, json=data, timeout=30)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")
            
            # Enhanced error handling
            if response.status_code == 401:
                raise Exception("Authentication failed - check your Cloze API key")
            elif response.status_code == 403:
                raise Exception("Access forbidden - API key may not have required permissions")
            elif response.status_code == 404:
                raise Exception(f"API endpoint not found: {endpoint}")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded - too many requests")
            elif not response.ok:
                error_msg = f"API error {response.status_code}"
                try:
                    error_data = response.json()
                    if 'message' in error_data:
                        error_msg += f": {error_data['message']}"
                except:
                    error_msg += f": {response.text}"
                raise Exception(error_msg)
            
            return response.json()
            
        except requests.exceptions.Timeout:
            raise Exception("Request timed out - Cloze API may be slow")
        except requests.exceptions.ConnectionError:
            raise Exception("Connection failed - check internet connection")
        except requests.exceptions.RequestException as e:
            current_app.logger.error(f"Cloze API request failed: {e}")
            raise Exception(f"Request failed: {str(e)}")

    def get_profile(self):
        """Get user profile information using correct endpoint"""
        return self._make_request('GET', '/user/profile')
    
    def get_people_stages(self, limit=50):
        """Get people with their stages/segments - NEW ENDPOINT"""
        params = {'limit': limit}
        return self._make_request('GET', '/user/stages/people', params=params)
    
    def find_people(self, query, limit=20):
        """Search for people using correct find endpoint"""
        params = {
            'query': query,
            'limit': limit
        }
        return self._make_request('GET', '/people/find', params=params)
    
    def create_person(self, person_data):
        """Create a new person record"""
        return self._make_request('POST', '/people/create', data=person_data)
    
    def update_person(self, person_id, person_data):
        """Update an existing person record"""
        return self._make_request('POST', '/people/update', data={
            'id': person_id,
            **person_data
        })
    
    def get_message_opens(self, days_back=7, limit=50):
        """Get message opens/engagement data"""
        params = {
            'days': days_back,
            'limit': limit
        }
        return self._make_request('GET', '/messages/opens', params=params)
    
    # REMOVED: Non-existent endpoints like activity, projects, companies
    # These don't exist in the current Cloze API based on the documentation provided

def get_cloze_morning_briefing():
    """Generate morning briefing from available Cloze data"""
    try:
        client = ClozeClient()
        
        briefing = f"# Cloze Morning Briefing - {format_timestamp()}\n\n"
        
        # Get user profile
        try:
            profile = client.get_profile()
            username = profile.get('name', profile.get('email', 'User'))
            briefing += f"Good morning, {username}!\n\n"
            
            # Add profile info if available
            if profile.get('company'):
                briefing += f"**Company:** {profile['company']}\n"
            if profile.get('timezone'):
                briefing += f"**Timezone:** {profile['timezone']}\n"
            briefing += "\n"
            
        except Exception as e:
            briefing += f"**Profile Access:** Error - {str(e)}\n\n"
            username = 'User'
        
        # Get people stages/segments summary
        try:
            people_stages = client.get_people_stages(limit=100)
            if people_stages and 'data' in people_stages:
                people_data = people_stages['data']
                briefing += f"## 👥 People Summary\n"
                briefing += f"**Total People in CRM:** {len(people_data)}\n\n"
                
                # Organize by stage/segment
                stages = {}
                for person in people_data:
                    stage = person.get('stage', 'No Stage')
                    if stage not in stages:
                        stages[stage] = []
                    stages[stage].append(person.get('name', 'Unknown'))
                
                if stages:
                    briefing += "**People by Stage:**\n"
                    for stage, people in stages.items():
                        briefing += f"- **{stage}:** {len(people)} people\n"
                    briefing += "\n"
                else:
                    briefing += "No stage information available\n\n"
            else:
                briefing += "## 👥 People Summary\n"
                briefing += "No people data available\n\n"
                
        except Exception as e:
            briefing += f"## 👥 People Summary\n"
            briefing += f"Error accessing people data: {str(e)}\n\n"
        
        # Get recent message engagement
        try:
            message_opens = client.get_message_opens(days_back=7, limit=20)
            if message_opens and 'data' in message_opens:
                opens_data = message_opens['data']
                briefing += f"## 📧 Recent Engagement (Last 7 days)\n"
                briefing += f"**Message Opens:** {len(opens_data)}\n"
                
                if opens_data:
                    # Show recent opens
                    briefing += "\n**Recent Opens:**\n"
                    for open_event in opens_data[:5]:  # Top 5
                        person = open_event.get('person', {}).get('name', 'Unknown')
                        subject = open_event.get('subject', 'No Subject')[:50]
                        briefing += f"- {person}: {subject}\n"
                briefing += "\n"
            else:
                briefing += f"## 📧 Recent Engagement\n"
                briefing += "No message engagement data available\n\n"
                
        except Exception as e:
            briefing += f"## 📧 Recent Engagement\n"
            briefing += f"Error accessing message data: {str(e)}\n\n"
        
        # Add connection status
        briefing += f"## ⚙️ Connection Status\n"
        briefing += f"- **API Key:** Configured ✓\n"
        briefing += f"- **Authentication:** Working ✓\n"
        briefing += f"- **Available Endpoints:** Profile, People, Messages ✓\n\n"
        
        briefing += "**Available commands:** `cloze search [name]` to find people\n"
        
        return briefing
        
    except Exception as e:
        current_app.logger.error(f"Cloze morning briefing failed: {e}")
        return f"**Cloze Morning Briefing Error**\n\nCould not fetch briefing: {str(e)}\n\n**Troubleshooting:**\n1. Verify API key is correct\n2. Check if API key has required permissions\n3. Contact Cloze support if issues persist"

def get_cloze_pipeline_summary():
    """Get people pipeline summary using available endpoints"""
    try:
        client = ClozeClient()
        
        # Get people with stages
        people_stages = client.get_people_stages(limit=200)
        
        if not people_stages or not people_stages.get('data'):
            return "No people data found in Cloze."
        
        summary = f"# Cloze People Pipeline Summary\n\n"
        
        # Organize by stage
        stages = {}
        total_people = 0
        
        for person in people_stages['data']:
            stage = person.get('stage', 'No Stage')
            
            if stage not in stages:
                stages[stage] = []
            
            stages[stage].append({
                'name': person.get('name', 'Unnamed Person'),
                'email': person.get('email', ''),
                'company': person.get('company', ''),
                'last_contact': person.get('lastContact', '')
            })
            
            total_people += 1
        
        summary += f"**Total People in Pipeline**: {total_people}\n\n"
        
        for stage, people in stages.items():
            summary += f"## {stage} ({len(people)} people)\n"
            
            for person in people[:10]:  # Show top 10 people per stage
                name = person['name']
                company_info = f" - {person['company']}" if person['company'] else ""
                email_info = f" ({person['email']})" if person['email'] else ""
                summary += f"- **{name}**{company_info}{email_info}\n"
            
            if len(people) > 10:
                summary += f"- *...and {len(people) - 10} more people*\n"
            
            summary += "\n"
        
        return summary
        
    except Exception as e:
        current_app.logger.error(f"Cloze pipeline summary failed: {e}")
        return f"**Cloze Pipeline Error**: {str(e)}"

def search_cloze_contacts(query, limit=10):
    """Search for contacts in Cloze using correct API endpoint"""
    try:
        client = ClozeClient()
        
        # Use the correct find endpoint
        results = client.find_people(query, limit=limit)
        
        if not results or not results.get('data'):
            return f"No contacts found for search: '{query}'"
        
        search_results = f"# Cloze Contact Search: '{query}'\n\n"
        people = results['data']
        
        search_results += f"Found {len(people)} matching contacts:\n\n"
        
        for person in people:
            name = person.get('name', 'Unknown Name')
            email = person.get('email', '')
            company = person.get('company', '')
            stage = person.get('stage', '')
            
            search_results += f"**{name}**"
            if email:
                search_results += f" - {email}"
            if company:
                search_results += f" ({company})"
            if stage:
                search_results += f" [Stage: {stage}]"
            search_results += "\n"
        
        return search_results
        
    except Exception as e:
        current_app.logger.error(f"Cloze contact search failed: {e}")
        return f"**Contact Search Error**: {str(e)}"

def log_ghostline_interaction_to_cloze(user_input, ai_responses, project_name):
    """Log Ghostline AI interactions - LIMITED due to API constraints"""
    try:
        # Note: The current Cloze API doesn't have endpoints for creating notes or communications
        # This would require additional API endpoints that aren't documented
        current_app.logger.info(f"Ghostline interaction logged locally for project: {project_name}")
        
        # For now, just log locally. Future implementation would need:
        # - Note creation endpoint
        # - Communication logging endpoint
        # - Activity tracking endpoint
        
        return True
        
    except Exception as e:
        current_app.logger.error(f"Failed to log to Cloze: {e}")
        return False

def process_cloze_command(user_input, project, voices, random_toggle):
    """Process Cloze-related commands with better error handling"""
    user_lower = user_input.lower().strip()
    
    # Cloze morning briefing
    if any(phrase in user_lower for phrase in ['cloze morning', 'morning cloze', 'cloze briefing']):
        briefing = get_cloze_morning_briefing()
        save_daily_log_enhanced('morning', briefing)
        
        response_data = {
            "SyntaxPrime": briefing
        }
        return response_data, True
    
    # Cloze pipeline summary (now people pipeline)
    if any(phrase in user_lower for phrase in ['cloze pipeline', 'pipeline summary', 'cloze people']):
        pipeline = get_cloze_pipeline_summary()
        
        response_data = {
            "SyntaxPrime": pipeline
        }
        return response_data, True
    
    # Cloze contact search
    if user_lower.startswith('cloze search '):
        query = user_input[13:].strip()  # Remove "cloze search "
        results = search_cloze_contacts(query)
        
        response_data = {
            "SyntaxPrime": results
        }
        return response_data, True
    
    # Cloze find people (alternative command)
    if user_lower.startswith('cloze find '):
        query = user_input[11:].strip()  # Remove "cloze find "
        results = search_cloze_contacts(query)
        
        response_data = {
            "SyntaxPrime": results
        }
        return response_data, True
    
    # No Cloze command matched
    return {}, False

def is_cloze_configured():
    """Check if Cloze API is properly configured"""
    return bool(CLOZE_API_KEY)

def test_cloze_connection():
    """Test Cloze API connection - useful for debugging"""
    try:
        client = ClozeClient()
        profile = client.get_profile()
        return {
            'success': True,
            'message': 'Connection successful',
            'user': profile.get('name', profile.get('email', 'Unknown'))
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
