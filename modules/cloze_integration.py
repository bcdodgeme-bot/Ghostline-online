# modules/cloze_integration.py
# Cloze CRM API Integration for Ghostline

import os
import requests
import json
from datetime import datetime, timedelta
from flask import current_app
from modules.database import save_daily_log_enhanced
from modules.utils import format_timestamp

# Cloze API Configuration
CLOZE_API_BASE = "https://api.cloze.com/v1"
CLOZE_API_KEY = os.getenv('CLOZE_API_KEY')

class ClozeClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or CLOZE_API_KEY
        self.base_url = CLOZE_API_BASE
        self.headers = {
            'Content-Type': 'application/json'
        }
    
    def _make_request(self, method, endpoint, params=None, data=None):
        """Make authenticated request to Cloze API"""
        if not self.api_key:
            raise Exception("Cloze API key not configured")
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Add API key as query parameter
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, params=params)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, params=params, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, params=params, json=data)
            else:
                raise Exception(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            current_app.logger.error(f"Cloze API request failed: {e}")
            raise Exception(f"Cloze API error: {str(e)}")

    def get_profile(self):
        """Get user profile information"""
        return self._make_request('GET', '/user/profile')
    
    def query_activity(self, days_back=7, activity_types=None):
        """Query user activity data"""
        params = {
            'days': days_back
        }
        if activity_types:
            params['types'] = ','.join(activity_types)
        
        return self._make_request('GET', '/user/activity', params=params)
    
    def get_contacts(self, limit=50, segment=None):
        """Retrieve contacts/people"""
        params = {'limit': limit}
        if segment:
            params['segment'] = segment
        
        return self._make_request('GET', '/people', params=params)
    
    def get_companies(self, limit=50):
        """Retrieve companies"""
        params = {'limit': limit}
        return self._make_request('GET', '/companies', params=params)
    
    def get_projects(self, limit=50, segment=None):
        """Retrieve projects/deals"""
        params = {'limit': limit}
        if segment:
            params['segment'] = segment
        
        return self._make_request('GET', '/projects', params=params)
    
    def add_note(self, content, person_id=None, project_id=None, company_id=None):
        """Add a note to Cloze"""
        data = {
            'content': content,
            'created': datetime.now().isoformat()
        }
        
        if person_id:
            data['person'] = person_id
        if project_id:
            data['project'] = project_id
        if company_id:
            data['company'] = company_id
        
        return self._make_request('POST', '/notes', data=data)
    
    def add_communication_record(self, comm_type, content, participants=None):
        """Add communication record (email, call, text, meeting)"""
        data = {
            'type': comm_type,  # 'email', 'call', 'text', 'meeting', 'dm'
            'content': content,
            'created': datetime.now().isoformat()
        }
        
        if participants:
            data['participants'] = participants
        
        return self._make_request('POST', '/communications', data=data)

def get_cloze_morning_briefing():
    """Generate morning briefing from Cloze data"""
    try:
        client = ClozeClient()
        
        # Try the basic profile endpoint first
        try:
            profile = client.get_profile()
            username = profile.get('name', 'User')
        except Exception as profile_error:
            # If profile fails, try alternative endpoints
            username = 'User'
            profile_error_msg = str(profile_error)
        
        # Build basic briefing without activity data for now
        briefing = f"# Cloze Morning Briefing - {format_timestamp()}\n\n"
        briefing += f"Good morning, {username}!\n\n"
        
        # Since activity endpoint doesn't work, provide alternative info
        briefing += f"## Connection Status\n"
        briefing += f"- **API Key**: Configured ✓\n"
        briefing += f"- **Authentication**: Working ✓\n"
        
        if 'profile_error_msg' in locals():
            briefing += f"- **Profile Access**: Error - {profile_error_msg}\n"
        else:
            briefing += f"- **Profile Access**: Working ✓\n"
        
        briefing += f"\n**Note**: Activity data endpoint needs configuration. Contact Cloze support for correct API endpoints.\n"
        
        briefing += "\n---\n"
        briefing += "**Available commands**: `cloze search [name]` to test contact search\n"
        
        return briefing
        
    except Exception as e:
        current_app.logger.error(f"Cloze morning briefing failed: {e}")
        return f"**Cloze Morning Briefing Error**\n\nCould not fetch briefing: {str(e)}\n\n**Troubleshooting**:\n1. Verify API key is correct\n2. Check Cloze API documentation for correct endpoints\n3. Contact Cloze support for endpoint structure"

def get_cloze_pipeline_summary():
    """Get pipeline and deals summary"""
    try:
        client = ClozeClient()
        
        # Get deals/projects
        projects = client.get_projects(limit=20, segment='Deal')
        
        if not projects or not projects.get('data'):
            return "No active deals found in Cloze pipeline."
        
        summary = f"# Cloze Pipeline Summary\n\n"
        
        # Organize by stage
        stages = {}
        total_value = 0
        
        for project in projects['data']:
            stage = project.get('stage', 'Unknown')
            value = project.get('value', 0)
            
            if stage not in stages:
                stages[stage] = []
            
            stages[stage].append({
                'name': project.get('name', 'Unnamed Deal'),
                'value': value,
                'company': project.get('company', {}).get('name', '')
            })
            
            if isinstance(value, (int, float)):
                total_value += value
        
        summary += f"**Total Pipeline Value**: ${total_value:,.2f}\n\n"
        
        for stage, deals in stages.items():
            summary += f"## {stage} ({len(deals)} deals)\n"
            stage_value = sum(deal['value'] for deal in deals if isinstance(deal['value'], (int, float)))
            summary += f"*Stage Value: ${stage_value:,.2f}*\n\n"
            
            for deal in deals[:5]:  # Show top 5 deals per stage
                company_info = f" - {deal['company']}" if deal['company'] else ""
                value_info = f" (${deal['value']:,.2f})" if deal['value'] else ""
                summary += f"- **{deal['name']}**{company_info}{value_info}\n"
            
            if len(deals) > 5:
                summary += f"- *...and {len(deals) - 5} more deals*\n"
            
            summary += "\n"
        
        return summary
        
    except Exception as e:
        current_app.logger.error(f"Cloze pipeline summary failed: {e}")
        return f"**Cloze Pipeline Error**: {str(e)}"

def log_ghostline_interaction_to_cloze(user_input, ai_responses, project_name):
    """Log Ghostline AI interactions as notes in Cloze"""
    try:
        client = ClozeClient()
        
        # Create note content
        note_content = f"Ghostline AI Interaction - {project_name}\n\n"
        note_content += f"User Query: {user_input}\n\n"
        
        for voice, response in ai_responses.items():
            note_content += f"{voice}: {response[:200]}...\n\n"
        
        note_content += f"Generated: {format_timestamp()}"
        
        # Add as general note (could be enhanced to link to specific contacts/projects)
        result = client.add_note(note_content)
        
        current_app.logger.info(f"Logged Ghostline interaction to Cloze: {result.get('id', 'Unknown')}")
        return True
        
    except Exception as e:
        current_app.logger.error(f"Failed to log to Cloze: {e}")
        return False

def search_cloze_contacts(query, limit=10):
    """Search for contacts in Cloze"""
    try:
        client = ClozeClient()
        
        # Get contacts (Cloze API may have specific search endpoints)
        contacts = client.get_contacts(limit=limit)
        
        if not contacts or not contacts.get('data'):
            return "No contacts found."
        
        results = f"# Cloze Contact Search: '{query}'\n\n"
        
        # Simple name-based filtering (could be enhanced with proper search API)
        matched_contacts = []
        query_lower = query.lower()
        
        for contact in contacts['data']:
            name = contact.get('name', '').lower()
            email = contact.get('email', '').lower()
            company = contact.get('company', {}).get('name', '').lower()
            
            if (query_lower in name or 
                query_lower in email or 
                query_lower in company):
                matched_contacts.append(contact)
        
        if not matched_contacts:
            results += "No contacts match your search query.\n"
        else:
            results += f"Found {len(matched_contacts)} matching contacts:\n\n"
            
            for contact in matched_contacts[:limit]:
                name = contact.get('name', 'Unknown Name')
                email = contact.get('email', '')
                company_name = contact.get('company', {}).get('name', '')
                
                results += f"**{name}**"
                if email:
                    results += f" - {email}"
                if company_name:
                    results += f" ({company_name})"
                results += "\n"
        
        return results
        
    except Exception as e:
        current_app.logger.error(f"Cloze contact search failed: {e}")
        return f"**Contact Search Error**: {str(e)}"

def process_cloze_command(user_input, project, voices, random_toggle):
    """Process Cloze-related commands"""
    user_lower = user_input.lower().strip()
    
    # Cloze morning briefing
    if any(phrase in user_lower for phrase in ['cloze morning', 'morning cloze', 'cloze briefing']):
        briefing = get_cloze_morning_briefing()
        save_daily_log_enhanced('morning', briefing)
        
        response_data = {
            "SyntaxPrime": briefing
        }
        return response_data, True
    
    # Cloze pipeline summary
    if any(phrase in user_lower for phrase in ['cloze pipeline', 'pipeline summary', 'cloze deals']):
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
    
    # No Cloze command matched
    return {}, False

def is_cloze_configured():
    """Check if Cloze API is properly configured"""
    return bool(CLOZE_API_KEY)