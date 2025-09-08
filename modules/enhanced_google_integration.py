# modules/enhanced_google_integration.py - Section 1: Imports and Base Classes
# Consolidated Google Ecosystem Integration with GA4 Data API

import os
import datetime
import json
import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import pytz

from utils.gmail_client import (
    list_overnight, search as gmail_search,
    list_today_events, list_tomorrow_events, search_calendar,
    get_next_meeting, format_calendar_summary
)
from utils.ghostline_engine import generate_response
from utils.rag_basic import is_ready
from modules.database import save_conversation_enhanced, save_daily_log_enhanced
from modules.brain import enhanced_retrieve

# Google API imports
try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    GOOGLE_APIS_AVAILABLE = True
except ImportError:
    GOOGLE_APIS_AVAILABLE = False
    print("Google API libraries not available")

CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

# Global context storage for conversational follow-ups
email_context_cache = {}


class ConversationalEmailContext:
    """Manages conversational context for email follow-ups"""
    
    @staticmethod
    def store_morning_briefing_emails(session_id: str, emails: list):
        """Store emails from morning briefing for follow-up questions"""
        email_context_cache[session_id] = {
            'type': 'morning_briefing',
            'timestamp': datetime.datetime.now(),
            'emails': emails,
            'senders': [email.get('sender', 'Unknown') for email in emails]
        }
    
    @staticmethod
    def get_context(session_id: str):
        """Get stored email context"""
        context = email_context_cache.get(session_id)
        if not context:
            return None
        
        # Expire context after 30 minutes
        if (datetime.datetime.now() - context['timestamp']).seconds > 1800:
            del email_context_cache[session_id]
            return None
        
        return context
    
    @staticmethod
    def find_email_by_sender(context: dict, sender_query: str) -> list:
        """Find emails matching sender query from stored context"""
        if not context or 'emails' not in context:
            return []
        
        matching_emails = []
        sender_query_lower = sender_query.lower()
        
        for email in context['emails']:
            sender = email.get('sender', '').lower()
            # Match if sender query is contained in sender name
            if sender_query_lower in sender or sender in sender_query_lower:
                matching_emails.append(email)
        
        return matching_emails


def is_likely_person_name(text: str) -> bool:
    """Simple heuristic to determine if text looks like a person's name"""
    if not text or len(text) < 2:
        return False
    
    # Split into words
    words = text.strip().split()
    
    # Too many words probably not a name
    if len(words) > 3:
        return False
    
    # Check if words look like names (capitalized, no numbers, reasonable length)
    for word in words:
        if not word.isalpha():
            return False
        if len(word) < 2 or len(word) > 20:
            return False
        # At least one word should be capitalized (like a proper name)
        if not any(w[0].isupper() for w in words):
            return False
    
    # Exclude common non-name words
    non_name_words = {
        'the', 'and', 'or', 'but', 'with', 'about', 'project', 'meeting',
        'email', 'message', 'document', 'file', 'report', 'update',
        'budget', 'schedule', 'task', 'work', 'team', 'company',
        'machine', 'learning', 'data', 'analysis', 'system'
    }
    
    if any(word.lower() in non_name_words for word in words):
        return False
    
    return True


class EnhancedConversationalContext:
    """Enhanced conversational context for all Google integrations"""
    
    def __init__(self):
        self.context_cache = {}
    
    def store_analytics_report(self, session_id: str, site_name: str, report_data: dict, user_query: str):
        """Store analytics report for follow-up questions"""
        self.context_cache[session_id] = {
            'type': 'analytics_report',
            'timestamp': datetime.datetime.now(),
            'site_name': site_name,
            'report_data': report_data,
            'original_query': user_query,
            'summary': self._generate_analytics_summary(report_data.get('data', []))
        }
    
    def store_search_console_report(self, session_id: str, site_name: str, report_data: dict, user_query: str):
        """Store search console report for follow-up questions"""
        self.context_cache[session_id] = {
            'type': 'search_console_report',
            'timestamp': datetime.datetime.now(),
            'site_name': site_name,
            'report_data': report_data,
            'original_query': user_query,
            'summary': self._generate_search_console_summary(report_data.get('data', []))
        }
    
    def store_gmail_report(self, session_id: str, emails: list, report_type: str):
        """Store Gmail report for follow-up questions"""
        self.context_cache[session_id] = {
            'type': f'gmail_{report_type}',
            'timestamp': datetime.datetime.now(),
            'emails': emails,
            'email_count': len(emails),
            'senders': [email.get('sender', 'Unknown') for email in emails],
            'summary': self._generate_gmail_summary(emails, report_type)
        }
    
    def store_calendar_report(self, session_id: str, events: list, report_type: str):
        """Store calendar report for follow-up questions"""
        self.context_cache[session_id] = {
            'type': f'calendar_{report_type}',
            'timestamp': datetime.datetime.now(),
            'events': events,
            'event_count': len(events),
            'summary': self._generate_calendar_summary(events, report_type)
        }
    
    def get_context(self, session_id: str):
        """Get stored context"""
        context = self.context_cache.get(session_id)
        if not context:
            return None
        
        # Expire context after 30 minutes
        if (datetime.datetime.now() - context['timestamp']).seconds > 1800:
            del self.context_cache[session_id]
            return None
        
        return context
    
    def detect_follow_up_question(self, user_input: str, context: dict) -> bool:
        """Detect if this is a follow-up question about recent report"""
        if not context:
            return False
        
        user_lower = user_input.lower().strip()
        
        # General follow-up indicators
        follow_up_patterns = [
            'what do you think', 'what\'s your opinion', 'your thoughts',
            'analyze this', 'analyze that', 'what does this mean',
            'is this good', 'is this bad', 'how is this',
            'explain this', 'tell me about', 'what about',
            'any insights', 'any recommendations', 'what should',
            'that report', 'this data', 'these results',
            'the numbers', 'the metrics', 'the traffic'
        ]
        
        if any(pattern in user_lower for pattern in follow_up_patterns):
            return True
        
        # Context-specific patterns
        if context['type'] == 'search_console_report':
            seo_patterns = [
                'seo', 'search', 'ranking', 'queries', 'clicks',
                'impressions', 'ctr', 'position', 'keywords'
            ]
            if any(pattern in user_lower for pattern in seo_patterns):
                return True
        
        elif context['type'] == 'analytics_report':
            analytics_patterns = [
                'traffic', 'visitors', 'sessions', 'pageviews',
                'bounce rate', 'users', 'analytics', 'website'
            ]
            if any(pattern in user_lower for pattern in analytics_patterns):
                return True
        
        elif context['type'].startswith('gmail_'):
            email_patterns = [
                'emails', 'messages', 'senders', 'inbox',
                'important', 'urgent', 'respond', 'reply'
            ]
            if any(pattern in user_lower for pattern in email_patterns):
                return True
        
        return False
     
# Section 2: Context Methods and GoogleIntegration Class Setup
# Section 2: Context Methods and GoogleIntegration Class Setup

    def generate_contextual_response(self, user_input: str, context: dict, project: str, use_voices: list, random_toggle: bool) -> Dict:
        """Generate response with full context"""
        context_prompt = self._build_context_prompt(user_input, context)
        
        # Get relevant background information
        retrieval_ctx = enhanced_retrieve(context_prompt, k=5, project=project) if is_ready() else []
        
        response_data = generate_response(
            context_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        
        return response_data
    
    def _build_context_prompt(self, user_input: str, context: dict) -> str:
        """Build comprehensive context prompt"""
        prompt = f"User just asked: '{user_input}'\n\n"
        prompt += f"This is a follow-up question about a recent report I provided.\n\n"
        
        if context['type'] == 'search_console_report':
            prompt += f"**Recent Search Console Report for {context['site_name']}:**\n"
            prompt += f"Original query: {context['original_query']}\n"
            prompt += context['summary']
            
            data = context['report_data'].get('data', [])
            if data and len(data) > 0:
                prompt += f"\n\n**Detailed Search Data:**\n"
                prompt += f"- Total Clicks: {sum(row.get('clicks', 0) for row in data):,}\n"
                prompt += f"- Total Impressions: {sum(row.get('impressions', 0) for row in data):,}\n"
                prompt += f"- Number of Queries: {len(data)}\n\n"
                
                prompt += f"**Top Search Queries:**\n"
                for i, row in enumerate(data[:10], 1):
                    query = row.get('query', 'Unknown')
                    clicks = row.get('clicks', 0)
                    impressions = row.get('impressions', 0)
                    ctr = row.get('ctr', 0.0) * 100
                    prompt += f"{i}. '{query}' - {clicks} clicks, {impressions} impressions, {ctr:.2f}% CTR\n"
        
        elif context['type'] == 'analytics_report':
            prompt += f"**Recent Analytics Report for {context['site_name']}:**\n"
            prompt += f"Original query: {context['original_query']}\n"
            prompt += context['summary']
            
            data = context['report_data'].get('data', [])
            if data:
                total_sessions = sum(row.get('sessions', 0) for row in data)
                total_users = sum(row.get('users', 0) for row in data)
                total_pageviews = sum(row.get('pageviews', 0) for row in data)
                
                prompt += f"\n\n**Detailed Analytics Data:**\n"
                prompt += f"- Total Sessions: {total_sessions:,}\n"
                prompt += f"- Total Users: {total_users:,}\n"
                prompt += f"- Total Pageviews: {total_pageviews:,}\n"
                prompt += f"- Days of Data: {len(data)}\n"
        
        elif context['type'].startswith('gmail_'):
            prompt += f"**Recent Gmail Report ({context['type'].replace('gmail_', '').title()}):**\n"
            prompt += context['summary']
            
            if context.get('emails'):
                prompt += f"\n\n**Email Details:**\n"
                for i, email in enumerate(context['emails'][:10], 1):
                    sender = email.get('sender', 'Unknown')
                    subject = email.get('subject', 'No Subject')
                    prompt += f"{i}. From: {sender} - Subject: {subject}\n"
        
        elif context['type'].startswith('calendar_'):
            prompt += f"**Recent Calendar Report ({context['type'].replace('calendar_', '').title()}):**\n"
            prompt += context['summary']
        
        prompt += f"\n\nPlease provide insights, analysis, or recommendations based on this data and the user's follow-up question."
        
        return prompt
    
    def _generate_analytics_summary(self, data: list) -> str:
        """Generate summary for analytics data"""
        if not data:
            return "No analytics data available."
        
        total_sessions = sum(row.get('sessions', 0) for row in data)
        total_users = sum(row.get('users', 0) for row in data)
        total_pageviews = sum(row.get('pageviews', 0) for row in data)
        
        return f"Analytics Summary: {total_sessions:,} sessions, {total_users:,} users, {total_pageviews:,} pageviews over {len(data)} days"
    
    def _generate_search_console_summary(self, data: list) -> str:
        """Generate summary for search console data"""
        if not data:
            return "No search console data available."
        
        total_clicks = sum(row.get('clicks', 0) for row in data)
        total_impressions = sum(row.get('impressions', 0) for row in data)
        avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        
        return f"Search Console Summary: {total_clicks:,} clicks, {total_impressions:,} impressions, {avg_ctr:.2f}% CTR across {len(data)} queries"
    
    def _generate_gmail_summary(self, emails: list, report_type: str) -> str:
        """Generate summary for Gmail data"""
        if not emails:
            return f"No emails found for {report_type}."
        
        senders = list(set(email.get('sender', 'Unknown') for email in emails))
        return f"Gmail {report_type.title()} Summary: {len(emails)} emails from {len(senders)} unique senders"
    
    def _generate_calendar_summary(self, events: list, report_type: str) -> str:
        """Generate summary for calendar data"""
        if not events:
            return f"No calendar events found for {report_type}."
        
        return f"Calendar {report_type.title()} Summary: {len(events)} scheduled events"


# Global instance for conversation context
enhanced_conversation_context = EnhancedConversationalContext()


class GoogleIntegration:
    """Complete Google Ecosystem Integration - All services in one class"""
    
    def __init__(self):
        self.timezone = self._get_user_timezone()
        self.credentials = None
        self.services = {}
        self.sites_config = self._load_sites_config()
        
        if GOOGLE_APIS_AVAILABLE:
            self._initialize_services()
    
    def _get_user_timezone(self):
        """Get user's timezone with proper fallback"""
        try:
            from flask import has_request_context, session
            if has_request_context() and session:
                user_tz = session.get('user_timezone')
                if user_tz:
                    return pytz.timezone(user_tz)
        except:
            pass
        return pytz.timezone('America/New_York')
    
    def _initialize_services(self):
        """Initialize all Google API services with automatic token refresh"""
        try:
            # Import the token manager
            from modules.google_token_refresh import get_google_credentials, token_manager
            
            # Get valid credentials with automatic refresh
            self.credentials = get_google_credentials()
            
            if not self.credentials:
                print("No valid Google credentials available - authentication required")
                print("Visit /google/auth/start to re-authenticate")
                return
            
            print("Valid Google credentials obtained")
            
            # Initialize all services with refreshed credentials
            try:
                # Core services (Phase 1)
                self.services['gmail'] = build('gmail', 'v1', credentials=self.credentials, cache_discovery=False)
                self.services['calendar'] = build('calendar', 'v3', credentials=self.credentials, cache_discovery=False)
                self.services['drive'] = build('drive', 'v3', credentials=self.credentials, cache_discovery=False)
                
                # Content creation services (Phase 2)
                self.services['docs'] = build('docs', 'v1', credentials=self.credentials, cache_discovery=False)
                self.services['sheets'] = build('sheets', 'v4', credentials=self.credentials, cache_discovery=False)
                self.services['slides'] = build('slides', 'v1', credentials=self.credentials, cache_discovery=False)
                
                # Analytics services (Phase 2) - UPDATED FOR GA4
                self.services['analyticsdata'] = build('analyticsdata', 'v1beta', credentials=self.credentials, cache_discovery=False)
                self.services['searchconsole'] = build('searchconsole', 'v1', credentials=self.credentials, cache_discovery=False)
                
                print(f"Google services initialized: {list(self.services.keys())}")
                
                # Get token status for debugging
                token_status = token_manager.get_token_status()
                print(f"Token status: {token_status['status']} - {token_status['message']}")
                
            except Exception as service_error:
                print(f"Failed to initialize some Google services: {service_error}")
                # Continue with partial services if some fail
                
        except ImportError:
            print("Google token refresh module not available - falling back to legacy token handling")
            self._initialize_services_legacy()
        except Exception as e:
            print(f"Failed to initialize Google services with token manager: {e}")
            self._initialize_services_legacy()
    
    def _initialize_services_legacy(self):
        """Legacy service initialization (fallback)"""
        try:
            token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
            if os.path.exists(token_path):
                # Load credentials
                self.credentials = Credentials.from_authorized_user_file(token_path)
                
                # CRITICAL: Always check and refresh if needed
                if self.credentials:
                    if self.credentials.expired and self.credentials.refresh_token:
                        try:
                            self.credentials.refresh(Request())
                            
                            # Save the refreshed token back to file
                            with open(token_path, 'w') as token_file:
                                token_file.write(self.credentials.to_json())
                            
                            print("Token refreshed successfully (legacy method)")
                        except Exception as refresh_error:
                            print(f"Token refresh failed: {refresh_error}")
                            print("   You may need to re-authenticate")
                            return
                    
                    elif self.credentials.expired and not self.credentials.refresh_token:
                        print("Token expired and no refresh token available")
                        print("   Run the manual token creation script with prompt='consent'")
                        return
                    
                    if self.credentials.valid:
                        # Initialize all services
                        self.services['gmail'] = build('gmail', 'v1', credentials=self.credentials)
                        self.services['calendar'] = build('calendar', 'v3', credentials=self.credentials)
                        self.services['drive'] = build('drive', 'v3', credentials=self.credentials)
                        
                        # Content creation services
                        try:
                            self.services['docs'] = build('docs', 'v1', credentials=self.credentials)
                            self.services['sheets'] = build('sheets', 'v4', credentials=self.credentials)
                            self.services['slides'] = build('slides', 'v1', credentials=self.credentials)
                        except Exception as e:
                            print(f"Content creation APIs failed: {e}")
                        
                        # Analytics services - UPDATED FOR GA4
                        try:
                            # GA4 Data API instead of old Reporting API
                            self.services['analyticsdata'] = build('analyticsdata', 'v1beta', credentials=self.credentials)
                            self.services['searchconsole'] = build('searchconsole', 'v1', credentials=self.credentials)
                            print("GA4 Analytics Data API initialized (legacy)")
                        except Exception as e:
                            print(f"Analytics APIs failed: {e}")
                        
                        print(f"Google services initialized (legacy): {list(self.services.keys())}")
                    else:
                        print("Google credentials invalid after refresh attempt")
                else:
                    print("Could not load Google credentials")
            else:
                print(f"No Google token file found at: {token_path}")
        except Exception as e:
            print(f"Failed to initialize Google services (legacy): {e}")
    
    def refresh_credentials_if_needed(self):
        """Public method to refresh credentials when needed - UPDATED"""
        try:
            from modules.google_token_refresh import get_google_credentials, force_token_refresh
            
            # Try to get fresh credentials
            fresh_credentials = get_google_credentials()
            
            if fresh_credentials:
                # Update stored credentials
                self.credentials = fresh_credentials
                
                # Clear service cache to force recreation with new credentials
                self.services.clear()
                
                # Re-initialize services
                self._initialize_services()
                
                print("Credentials refreshed and services reinitialized")
                return True
            else:
                print("Could not obtain valid credentials")
                return False
                
        except ImportError:
            # Fall back to legacy refresh
            if not self.credentials:
                return False
                
            if self.credentials.expired and self.credentials.refresh_token:
                try:
                    self.credentials.refresh(Request())
                    
                    # Save refreshed token
                    token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
                    with open(token_path, 'w') as token_file:
                        token_file.write(self.credentials.to_json())
                    
                    print("Credentials refreshed successfully (legacy)")
                    return True
                except Exception as e:
                    print(f"Credential refresh failed: {e}")
                    return False
            
            return self.credentials.valid if self.credentials else False
        except Exception as e:
            print(f"Credential refresh error: {e}")
            return False
    
    def ensure_valid_credentials(self):
        """Ensure we have valid credentials before making API calls"""
        if not self.credentials or self.credentials.expired:
            success = self.refresh_credentials_if_needed()
            if not success:
                raise Exception("Google authentication required - visit /google/auth/start to re-authenticate")
        return True
        
# Section 3: Site Configuration and Date Parsing Methods

    def _load_sites_config(self) -> Dict[str, Dict]:
        """Load multi-site configuration from environment variables"""
        sites = {}
        
        # Method 1: JSON configuration (recommended)
        sites_json = os.getenv('GOOGLE_SITES_CONFIG')
        if sites_json:
            try:
                return json.loads(sites_json)
            except json.JSONDecodeError as e:
                print(f"Invalid GOOGLE_SITES_CONFIG JSON: {e}")
        
        # Method 2: Individual environment variables (fallback)
        site_index = 1
        while True:
            site_name = os.getenv(f'SITE_{site_index}_NAME')
            if not site_name:
                break
            
            analytics_view_id = os.getenv(f'SITE_{site_index}_ANALYTICS_VIEW_ID')
            search_console_url = os.getenv(f'SITE_{site_index}_SEARCH_CONSOLE_URL')
            aliases_str = os.getenv(f'SITE_{site_index}_ALIASES', '')
            aliases = [alias.strip() for alias in aliases_str.split(',') if alias.strip()]
            
            if analytics_view_id or search_console_url:
                sites[site_name.lower().replace(' ', '_')] = {
                    'name': site_name,
                    'analytics_view_id': analytics_view_id,
                    'search_console_url': search_console_url,
                    'aliases': aliases
                }
            
            site_index += 1
        
        # Method 3: Legacy single site support
        if not sites:
            legacy_view_id = os.getenv('GOOGLE_ANALYTICS_VIEW_ID')
            legacy_search_url = os.getenv('SEARCH_CONSOLE_SITE_URL')
            
            if legacy_view_id or legacy_search_url:
                sites['default'] = {
                    'name': 'Default Site',
                    'analytics_view_id': legacy_view_id,
                    'search_console_url': legacy_search_url,
                    'aliases': []
                }
        
        return sites

    def get_available_sites(self) -> List[Dict]:
        """Get list of all configured sites"""
        return [
            {
                'key': key,
                'name': config['name'],
                'has_analytics': bool(config.get('analytics_view_id')),
                'has_search_console': bool(config.get('search_console_url')),
                'aliases': config.get('aliases', [])
            }
            for key, config in self.sites_config.items()
        ]
    
    def find_site_by_name(self, site_query: str) -> Optional[str]:
        """Find site key by name or alias"""
        site_query_lower = site_query.lower().strip()
        
        # Direct key match
        if site_query_lower in self.sites_config:
            return site_query_lower
        
        # Name match
        for key, config in self.sites_config.items():
            if config['name'].lower() == site_query_lower:
                return key
        
        # Alias match
        for key, config in self.sites_config.items():
            aliases = [alias.strip().lower() for alias in config.get('aliases', [])]
            if site_query_lower in aliases:
                return key
        
        # Partial name match
        for key, config in self.sites_config.items():
            if site_query_lower in config['name'].lower():
                return key
        
        return None

    def parse_date_range_from_input(self, user_input: str) -> Tuple[str, str]:
        """Parse flexible date ranges from user input for GA4"""
        user_lower = user_input.lower().strip()
        
        # Default to last 7 days
        start_date = "7daysAgo"
        end_date = "today"
        
        # Flexible date range patterns
        if 'last 6 months' in user_lower or 'past 6 months' in user_lower:
            start_date, end_date = "180daysAgo", "today"
        elif 'last year' in user_lower or 'past year' in user_lower:
            start_date, end_date = "365daysAgo", "today"
        elif 'last 3 months' in user_lower or 'past 3 months' in user_lower:
            start_date, end_date = "90daysAgo", "today"
        elif 'last 2 months' in user_lower or 'past 2 months' in user_lower:
            start_date, end_date = "60daysAgo", "today"
        elif 'last 30 days' in user_lower or 'past 30 days' in user_lower:
            start_date, end_date = "30daysAgo", "today"
        elif 'last 90 days' in user_lower or 'past 90 days' in user_lower:
            start_date, end_date = "90daysAgo", "today"
        elif 'last week' in user_lower:
            start_date, end_date = "7daysAgo", "today"
        elif 'last month' in user_lower:
            start_date, end_date = "30daysAgo", "today"
        elif 'yesterday' in user_lower:
            start_date, end_date = "yesterday", "yesterday"
        elif 'this month' in user_lower:
            # First day of current month to today
            today = datetime.date.today()
            first_day = today.replace(day=1)
            start_date = first_day.strftime('%Y-%m-%d')
            end_date = "today"
        elif 'this year' in user_lower:
            # First day of current year to today
            today = datetime.date.today()
            first_day = today.replace(month=1, day=1)
            start_date = first_day.strftime('%Y-%m-%d')
            end_date = "today"
        
        # Look for specific number patterns like "last 45 days"
        days_match = re.search(r'last (\d+) days?', user_lower)
        if days_match:
            days = int(days_match.group(1))
            start_date = f"{days}daysAgo"
            end_date = "today"
        
        # Look for specific months
        months_match = re.search(r'last (\d+) months?', user_lower)
        if months_match:
            months = int(months_match.group(1))
            days = months * 30  # Approximate
            start_date = f"{days}daysAgo"
            end_date = "today"
        
        return start_date, end_date
        
# Section 4: Gmail Integration - FIXED SEARCH COMMAND CONFLICTS

    def handle_gmail_commands(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Handle Gmail commands within the consolidated Google integration - FIXED: Search conflicts"""
        user_lower = user_input.lower().strip()
        
        # Gmail overnight/inbox commands
        if user_lower in ["overnight", "mail", "emails", "inbox", "check mail"]:
            try:
                print("Fetching overnight emails...")
                msgs = list_overnight(include_unread=True, include_primary=False)
                
                if not msgs or not isinstance(msgs, list):
                    response_data = {"SyntaxPrime": "No overnight emails found or Gmail service unavailable. Check your authentication."}
                    save_conversation_enhanced(project, user_input, response_data)
                    return response_data, True
                
                if len(msgs) == 0:
                    summary_prompt = "No overnight emails found. Your inbox appears to be up to date."
                else:
                    # Process email data
                    email_summaries = []
                    for msg in msgs[:10]:  # Limit to 10 emails
                        sender = self._extract_email_sender(msg)
                        subject = self._extract_email_subject(msg)
                        
                        if sender and subject:
                            if len(subject) > 80:
                                subject = subject[:80] + "..."
                            email_summaries.append(f"• {sender}: {subject}")
                    
                    if email_summaries:
                        summary_prompt = f"Found {len(email_summaries)} overnight emails:\n\n" + "\n".join(email_summaries)
                        
                        # Store context for follow-up questions
                        ConversationalEmailContext.store_morning_briefing_emails(
                            session_id=project,  # Using project as session ID
                            emails=[{
                                'sender': self._extract_email_sender(msg),
                                'subject': self._extract_email_subject(msg),
                                'original_msg': msg
                            } for msg in msgs[:10]]
                        )
                        
                        # Store in enhanced context
                        enhanced_conversation_context.store_gmail_report(
                            session_id=project,
                            emails=[{
                                'sender': self._extract_email_sender(msg),
                                'subject': self._extract_email_subject(msg)
                            } for msg in msgs[:10]],
                            report_type='overnight'
                        )
                    else:
                        summary_prompt = "Found emails but could not parse sender/subject information. Check Gmail API permissions."
                
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
                
            except Exception as e:
                print(f"Gmail overnight check failed: {e}")
                response_data = {"SyntaxPrime": f"Gmail integration error: {str(e)}. Please check your Google OAuth setup."}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
        
        # FIXED: Gmail search commands with better conflict resolution
        gmail_search_triggers = ['search email', 'find email', 'email about']
        
        # Exclusion words that indicate other services
        search_exclusions = ['console', 'seo', 'analytics', 'site', 'website', 'domain']
        
        # Check for Gmail search - be more specific about what constitutes Gmail search
        is_gmail_search = False
        
        if any(trigger in user_lower for trigger in gmail_search_triggers):
            # Explicit Gmail search triggers always work
            is_gmail_search = True
        elif user_lower.startswith('search '):
            # Only treat as Gmail search if:
            # 1. Doesn't contain exclusion words
            # 2. Contains email-related context words OR looks like a person name
            has_exclusions = any(word in user_lower for word in search_exclusions)
            
            email_context_words = ['message', 'from', 'to', 'subject', 'inbox', 'sent', 'received']
            has_email_context = any(word in user_lower for word in email_context_words)
            
            # The search query after "search "
            search_query = user_input[7:].strip()  # Remove "search "
            
            # If the query looks like a person's name or email topic, likely Gmail
            looks_like_person = is_likely_person_name(search_query)
            
            # Gmail search if: no exclusions AND (has email context OR looks like person name)
            is_gmail_search = not has_exclusions and (has_email_context or looks_like_person)
        
        if is_gmail_search:
            try:
                # Extract query
                query_text = ""
                for prefix in ["search email ", "find email ", "email about ", "search "]:
                    if user_input.lower().startswith(prefix):
                        query_text = user_input[len(prefix):].strip()
                        break
                
                if not query_text:
                    response_data = {"SyntaxPrime": "Please provide a search term after the command (e.g., 'search project updates')"}
                    save_conversation_enhanced(project, user_input, response_data)
                    return response_data, True
                
                print(f"Searching Gmail for: {query_text}")
                msgs = gmail_search(query_text)
                
                if not msgs or not isinstance(msgs, list):
                    response_data = {"SyntaxPrime": f"Gmail search failed or returned no results for '{query_text}'. Check your authentication."}
                    save_conversation_enhanced(project, user_input, response_data)
                    return response_data, True
                
                if len(msgs) == 0:
                    summary_prompt = f"No emails found matching '{query_text}'. Try different search terms."
                else:
                    # Process search results
                    search_results = []
                    for msg in msgs[:15]:  # Limit to 15 results
                        sender = self._extract_email_sender(msg)
                        subject = self._extract_email_subject(msg)
                        
                        if sender and subject:
                            if len(subject) > 60:
                                subject = subject[:60] + "..."
                            search_results.append(f"• {sender}: {subject}")
                    
                    if search_results:
                        summary_prompt = f"Found {len(search_results)} emails matching '{query_text}':\n\n" + "\n".join(search_results)
                        
                        # Store context
                        enhanced_conversation_context.store_gmail_report(
                            session_id=project,
                            emails=[{
                                'sender': self._extract_email_sender(msg),
                                'subject': self._extract_email_subject(msg)
                            } for msg in msgs[:15]],
                            report_type='search'
                        )
                    else:
                        summary_prompt = f"Search completed for '{query_text}' but no readable results found."
                
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
                
            except Exception as e:
                print(f"Gmail search failed: {e}")
                response_data = {"SyntaxPrime": f"Gmail search error: {str(e)}. Please check your Google OAuth setup."}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
        
        # Calendar commands
        if user_lower in ["calendar", "today", "meetings", "schedule"]:
            try:
                print("Fetching today's calendar events...")
                events = list_today_events(max_results=20)
                
                if not events or not isinstance(events, list):
                    response_data = {"SyntaxPrime": "Calendar service unavailable. Check your Google OAuth setup."}
                    save_conversation_enhanced(project, user_input, response_data)
                    return response_data, True
                
                if len(events) == 0:
                    summary_prompt = "No events found for today. Your calendar is clear."
                else:
                    calendar_summary = format_calendar_summary(events, "Today's Calendar")
                    summary_prompt = f"Here's your calendar for today:\n\n{calendar_summary}"
                    
                    # Store context
                    enhanced_conversation_context.store_calendar_report(
                        session_id=project,
                        events=events,
                        report_type='today'
                    )
                
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
                
            except Exception as e:
                print(f"Calendar check failed: {e}")
                response_data = {"SyntaxPrime": f"Calendar integration error: {str(e)}. Please check your Google OAuth setup."}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
        
        # Next meeting command
        if user_lower in ["next meeting", "next", "upcoming"]:
            try:
                print("Getting next meeting...")
                next_meeting = get_next_meeting()
                
                if not next_meeting or isinstance(next_meeting, dict) and "error" in next_meeting:
                    summary_prompt = "No upcoming meetings found in your calendar."
                elif next_meeting and next_meeting.get('summary'):
                    summary_prompt = f"Next meeting: {next_meeting['summary']} at {next_meeting.get('start_formatted', 'Unknown time')}"
                else:
                    summary_prompt = "Next meeting lookup completed but no readable meeting data found."
                
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
                
            except Exception as e:
                print(f"Next meeting check failed: {e}")
                response_data = {"SyntaxPrime": f"Next meeting error: {str(e)}. Please check your Google OAuth setup."}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
        
        # Good morning briefing
        if user_lower in ["good morning", "morning", "gm"]:
            try:
                print("Good Morning command triggered")
                
                # Build comprehensive morning briefing
                morning_briefing = "Good morning! Here's your daily briefing:\n\n"
                
                # Email section
                try:
                    msgs = list_overnight(include_unread=True, include_primary=False)
                    morning_briefing += "**OVERNIGHT EMAILS**\n"
                    if msgs and isinstance(msgs, list) and len(msgs) > 0:
                        morning_briefing += f"Found {len(msgs)} overnight emails\n"
                        
                        # Store context for follow-ups
                        ConversationalEmailContext.store_morning_briefing_emails(
                            session_id=project,
                            emails=[{
                                'sender': self._extract_email_sender(msg),
                                'subject': self._extract_email_subject(msg),
                                'original_msg': msg
                            } for msg in msgs[:10]]
                        )
                    else:
                        morning_briefing += "No overnight emails found\n"
                except Exception as e:
                    morning_briefing += f"Email check failed: {str(e)}\n"
                
                # Calendar section
                try:
                    events = list_today_events(max_results=20)
                    morning_briefing += "\n**TODAY'S CALENDAR**\n"
                    if events and isinstance(events, list) and len(events) > 0:
                        calendar_summary = format_calendar_summary(events, "")
                        morning_briefing += calendar_summary + "\n"
                    else:
                        morning_briefing += "No events scheduled for today\n"
                except Exception as e:
                    morning_briefing += f"Calendar check failed: {str(e)}\n"
                
                # Next meeting section
                try:
                    next_meeting = get_next_meeting()
                    morning_briefing += "\n**NEXT MEETING**\n"
                    if next_meeting and next_meeting.get('summary'):
                        morning_briefing += f"{next_meeting.get('summary', 'Unknown')} at {next_meeting.get('start_formatted', 'Unknown time')}\n"
                    else:
                        morning_briefing += "No upcoming meetings found\n"
                except Exception as e:
                    morning_briefing += f"Next meeting check failed: {str(e)}\n"
                
                # Save daily log
                try:
                    save_daily_log_enhanced("morning", morning_briefing)
                except Exception as e:
                    print(f"Failed to save daily log: {e}")
                
                retrieval_ctx = enhanced_retrieve(morning_briefing, k=5) if is_ready() else []
                response_data = generate_response(
                    f"Summarize this morning briefing and suggest 3 key priorities:\n\n{morning_briefing}",
                    use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
                
            except Exception as e:
                print(f"Morning briefing failed: {e}")
                response_data = {"SyntaxPrime": f"Morning briefing failed: {str(e)}. Please check your Google OAuth setup."}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
        
        return {}, False

    def _extract_email_sender(self, msg):
        """Extract sender from email message"""
        if not isinstance(msg, dict):
            return None
        
        # Try different possible field names for sender
        for sender_field in ['sender', 'from', 'From', 'fromEmail', 'senderEmail', 'author']:
            if sender_field in msg:
                sender = msg[sender_field]
                if sender:
                    # Clean up sender (remove email brackets if present)
                    if '<' in sender and '>' in sender:
                        sender = sender.split('<')[0].strip()
                        if not sender:  # If no name, use email
                            sender = sender.split('<')[1].split('>')[0].strip()
                    return sender
        
        # Try nested structures
        if 'headers' in msg and isinstance(msg['headers'], list):
            for header in msg['headers']:
                if isinstance(header, dict) and header.get('name', '').lower() == 'from':
                    return header.get('value')
        
        return "Unknown Sender"

    def _extract_email_subject(self, msg):
        """Extract subject from email message"""
        if not isinstance(msg, dict):
            return None
        
        # Try different possible field names for subject
        for subject_field in ['subject', 'Subject', 'title', 'summary', 'snippet']:
            if subject_field in msg:
                subject = msg[subject_field]
                if subject:
                    return subject
        
        # Try nested structures
        if 'headers' in msg and isinstance(msg['headers'], list):
            for header in msg['headers']:
                if isinstance(header, dict) and header.get('name', '').lower() == 'subject':
                    return header.get('value')
        
        return "No Subject"
        
# Section 5: Google Docs and Sheets Integration

    # =============================================================================
    # GOOGLE DOCS INTEGRATION
    # =============================================================================
    
    def create_document(self, title: str, content: str = "") -> Dict:
        """Create a new Google Doc"""
        if 'docs' not in self.services:
            return {'success': False, 'error': 'Google Docs API not available'}
        
        try:
            # Create the document
            document = self.services['docs'].documents().create(body={'title': title}).execute()
            document_id = document.get('documentId')
            
            # Add content if provided
            if content:
                requests = [
                    {
                        'insertText': {
                            'location': {'index': 1},
                            'text': content
                        }
                    }
                ]
                
                self.services['docs'].documents().batchUpdate(
                    documentId=document_id,
                    body={'requests': requests}
                ).execute()
            
            document_url = f"https://docs.google.com/document/d/{document_id}"
            
            return {
                'success': True,
                'document_id': document_id,
                'document_url': document_url,
                'title': title
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def append_to_document(self, document_id: str, content: str) -> Dict:
        """Append content to an existing Google Doc"""
        if 'docs' not in self.services:
            return {'success': False, 'error': 'Google Docs API not available'}
        
        try:
            # Get document to find end index
            document = self.services['docs'].documents().get(documentId=document_id).execute()
            end_index = document.get('body', {}).get('content', [{}])[-1].get('endIndex', 1)
            
            requests = [
                {
                    'insertText': {
                        'location': {'index': end_index - 1},
                        'text': f"\n\n{content}"
                    }
                }
            ]
            
            self.services['docs'].documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()
            
            return {
                'success': True,
                'message': f'Content appended to document',
                'document_url': f"https://docs.google.com/document/d/{document_id}"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # =============================================================================
    # GOOGLE SHEETS INTEGRATION
    # =============================================================================
    
    def create_spreadsheet(self, title: str) -> Dict:
        """Create a new Google Sheet"""
        if 'sheets' not in self.services:
            return {'success': False, 'error': 'Google Sheets API not available'}
        
        try:
            spreadsheet = {
                'properties': {
                    'title': title
                }
            }
            
            result = self.services['sheets'].spreadsheets().create(body=spreadsheet).execute()
            spreadsheet_id = result.get('spreadsheetId')
            spreadsheet_url = result.get('spreadsheetUrl')
            
            return {
                'success': True,
                'spreadsheet_id': spreadsheet_id,
                'spreadsheet_url': spreadsheet_url,
                'title': title
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def read_sheet_data(self, spreadsheet_id: str, range_name: str = "A1:Z1000") -> Dict:
        """Read data from a Google Sheet - FIXED: Preserve case in spreadsheet_id"""
        if 'sheets' not in self.services:
            return {'success': False, 'error': 'Google Sheets API not available'}
        
        try:
            result = self.services['sheets'].spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,  # Keep original case
                range=range_name
            ).execute()
            
            values = result.get('values', [])
            
            return {
                'success': True,
                'data': values,
                'rows': len(values),
                'spreadsheet_url': f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
# Section 6: GA4 Analytics and Search Console Integration

    # =============================================================================
    # GA4 ANALYTICS INTEGRATION - COMPLETELY REWRITTEN FOR GA4 DATA API
    # =============================================================================
    
    def get_ga4_analytics_report(self, property_id: str, start_date: str = "7daysAgo", end_date: str = "today") -> Dict:
        """Get GA4 Analytics report using the Data API"""
        if 'analyticsdata' not in self.services:
            return {'success': False, 'error': 'GA4 Analytics Data API not available'}
        
        try:
            # GA4 Data API request format
            request_body = {
                'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
                'metrics': [
                    {'name': 'sessions'},
                    {'name': 'totalUsers'},
                    {'name': 'screenPageViews'},
                    {'name': 'bounceRate'}
                ],
                'dimensions': [
                    {'name': 'date'}
                ],
                'orderBys': [
                    {'dimension': {'dimensionName': 'date'}}
                ]
            }
            
            # Make the GA4 API call
            response = self.services['analyticsdata'].properties().runReport(
                property=f'properties/{property_id}',
                body=request_body
            ).execute()
            
            # Process GA4 response format
            analytics_data = []
            rows = response.get('rows', [])
            
            for row in rows:
                # GA4 uses different response structure
                dimension_values = row.get('dimensionValues', [])
                metric_values = row.get('metricValues', [])
                
                if dimension_values and metric_values:
                    date_value = dimension_values[0].get('value', '')
                    
                    # Parse metrics safely
                    sessions = int(metric_values[0].get('value', '0')) if len(metric_values) > 0 else 0
                    users = int(metric_values[1].get('value', '0')) if len(metric_values) > 1 else 0
                    pageviews = int(metric_values[2].get('value', '0')) if len(metric_values) > 2 else 0
                    bounce_rate = float(metric_values[3].get('value', '0.0')) * 100 if len(metric_values) > 3 else 0.0  # GA4 returns decimal, convert to percentage
                    
                    analytics_data.append({
                        'date': date_value,
                        'sessions': sessions,
                        'users': users,
                        'pageviews': pageviews,
                        'bounce_rate': bounce_rate
                    })
            
            return {
                'success': True,
                'data': analytics_data,
                'date_range': f"{start_date} to {end_date}",
                'total_rows': len(analytics_data)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_analytics_data(self, site_key: str = None, start_date: str = "7daysAgo", end_date: str = "today") -> Dict:
        """Get analytics data for a specific site using multi-site config"""
        if not site_key:
            # Use default site or first available
            if 'default' in self.sites_config:
                site_key = 'default'
            elif self.sites_config:
                site_key = list(self.sites_config.keys())[0]
            else:
                return {'success': False, 'error': 'No sites configured'}
        
        if site_key not in self.sites_config:
            return {'success': False, 'error': f'Site "{site_key}" not found'}
        
        site_config = self.sites_config[site_key]
        property_id = site_config.get('analytics_view_id')  # This is actually a GA4 Property ID now
        
        if not property_id:
            return {'success': False, 'error': f'GA4 Property ID not configured for site "{site_config["name"]}"'}
        
        result = self.get_ga4_analytics_report(property_id, start_date, end_date)
        if result['success']:
            result['site_name'] = site_config['name']
            result['site_key'] = site_key
        
        return result
    
    def get_all_sites_analytics(self, start_date: str = "7daysAgo", end_date: str = "today") -> Dict:
        """Get analytics for all configured sites"""
        results = {}
        
        for site_key, site_config in self.sites_config.items():
            if site_config.get('analytics_view_id'):
                result = self.get_analytics_data(site_key, start_date, end_date)
                results[site_key] = result
        
        return {
            'success': True,
            'sites': results,
            'total_sites': len(results)
        }

    # =============================================================================
    # SEARCH CONSOLE INTEGRATION (UNCHANGED)
    # =============================================================================
    
    def get_search_console_data(self, site_url: str, start_date: str = None, end_date: str = None) -> Dict:
        """Get Search Console performance data"""
        if 'searchconsole' not in self.services:
            return {'success': False, 'error': 'Search Console API not available'}
        
        try:
            if not start_date:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
            request_body = {
                'startDate': start_date,
                'endDate': end_date,
                'dimensions': ['query', 'page'],
                'rowLimit': 100
            }
            
            response = self.services['searchconsole'].searchanalytics().query(
                siteUrl=site_url,
                body=request_body
            ).execute()
            
            rows = response.get('rows', [])
            search_data = []
            
            for row in rows:
                search_data.append({
                    'query': row['keys'][0],
                    'page': row['keys'][1] if len(row['keys']) > 1 else '',
                    'clicks': row.get('clicks', 0),
                    'impressions': row.get('impressions', 0),
                    'ctr': row.get('ctr', 0.0),
                    'position': row.get('position', 0.0)
                })
            
            return {
                'success': True,
                'data': search_data,
                'date_range': f"{start_date} to {end_date}",
                'total_queries': len(search_data)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_search_console_data_for_site(self, site_key: str = None, start_date: str = None, end_date: str = None) -> Dict:
        """Get search console data for a specific site using multi-site config"""
        if not site_key:
            # Use default site or first available
            if 'default' in self.sites_config:
                site_key = 'default'
            elif self.sites_config:
                site_key = list(self.sites_config.keys())[0]
            else:
                return {'success': False, 'error': 'No sites configured'}
        
        if site_key not in self.sites_config:
            return {'success': False, 'error': f'Site "{site_key}" not found'}
        
        site_config = self.sites_config[site_key]
        site_url = site_config.get('search_console_url')
        
        if not site_url:
            return {'success': False, 'error': f'Search Console not configured for site "{site_config["name"]}"'}
        
        result = self.get_search_console_data(site_url, start_date, end_date)
        if result['success']:
            result['site_name'] = site_config['name']
            result['site_key'] = site_key
        
        return result
        
# Section 7: Command Handlers (COMPLETELY REWRITTEN - FIXED Regex Error)

    # =============================================================================
    # COMMAND HANDLERS (UPDATED WITH CONTEXT STORAGE)
    # =============================================================================
    
    def handle_docs_command(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Handle Google Docs commands with improved parsing"""
        user_lower = user_input.lower().strip()
        
        # Create document command with improved content parsing
        if 'create document' in user_lower or 'create doc' in user_lower:
            # Better parsing to separate title from content
            content_match = re.search(r'create (?:document|doc) ["\']?([^"\']+?)["\']?\s+with content:\s*(.+)', user_input, re.IGNORECASE)
            if content_match:
                title = content_match.group(1).strip()
                content = content_match.group(2).strip()
            else:
                # Simple title extraction without content
                title_match = re.search(r'create (?:document|doc) (?:called |titled |named )?["\']?([^"\']+)["\']?', user_lower)
                if title_match:
                    title = title_match.group(1).strip()
                    content = ""
                else:
                    return {"SyntaxPrime": "Please specify a document title (e.g., 'create document Meeting Notes')"}, True
            
            result = self.create_document(title, content)
            
            if result['success']:
                response_text = f"**Document Created Successfully!**\n\n"
                response_text += f"**Title:** {result['title']}\n"
                response_text += f"**Document ID:** {result['document_id']}\n"
                response_text += f"**URL:** {result['document_url']}\n\n"
                response_text += f"You can now edit this document in Google Docs or add more content using commands like:\n"
                response_text += f"- 'add to document {result['document_id']}: your content here'"
            else:
                response_text = f"Failed to create document: {result['error']}"
            
            response_data = {"SyntaxPrime": response_text}
            save_conversation_enhanced(project, user_input, response_data)
            return response_data, True
        
        # Add to document command - Preserve case in document ID
        if 'add to document' in user_lower or 'append to document' in user_lower:
            doc_match = re.search(r'(?:add to|append to) document ([a-zA-Z0-9_-]+)[:\s]+(.+)', user_input, re.IGNORECASE)
            if doc_match:
                document_id = doc_match.group(1).strip()  # Keep original case
                content = doc_match.group(2).strip()
                
                result = self.append_to_document(document_id, content)
                
                if result['success']:
                    response_text = f"**Content Added Successfully!**\n\n"
                    response_text += f"Content has been appended to the document.\n"
                    response_text += f"**View Document:** {result['document_url']}"
                else:
                    response_text = f"Failed to add content: {result['error']}"
                
                response_data = {"SyntaxPrime": response_text}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
        
        return {}, False
    
    def handle_sheets_command(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Handle Google Sheets commands with enhanced URL/name parsing"""
        user_lower = user_input.lower().strip()
        
        # Create spreadsheet command
        if 'create spreadsheet' in user_lower or 'create sheet' in user_lower:
            title_match = re.search(r'create (?:spreadsheet|sheet) (?:called |titled |named )?["\']?([^"\']+)["\']?', user_lower)
            if title_match:
                title = title_match.group(1).strip()
                
                result = self.create_spreadsheet(title)
                
                if result['success']:
                    response_text = f"**Spreadsheet Created Successfully!**\n\n"
                    response_text += f"**Title:** {result['title']}\n"
                    response_text += f"**Spreadsheet ID:** {result['spreadsheet_id']}\n"
                    response_text += f"**URL:** {result['spreadsheet_url']}\n\n"
                    response_text += f"You can now add data or read it using commands like:\n"
                    response_text += f"- 'read sheet {result['title']}' (by name)\n"
                    response_text += f"- 'read sheet {result['spreadsheet_url']}' (by URL)"
                else:
                    response_text = f"Failed to create spreadsheet: {result['error']}"
                
                response_data = {"SyntaxPrime": response_text}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
        
        # Enhanced read spreadsheet command - supports URL, ID, and name
        if 'read sheet' in user_lower or 'get data from sheet' in user_lower:
            spreadsheet_id = None
            
            # Try to parse Google Sheets URL
            url_match = re.search(r'(?:read sheet|get data from sheet)\s+https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9_-]+)', user_input)
            if url_match:
                spreadsheet_id = url_match.group(1)
            else:
                # Try direct ID or name format
                sheet_match = re.search(r'(?:read sheet|get data from sheet)\s+(.+)', user_input)
                if sheet_match:
                    sheet_identifier = sheet_match.group(1).strip()
                    
                    # CRITICAL FIX: Properly close the regex string literal
                    # This was the source of the syntax error on line 1378
                    if re.match(r'^[a-zA-Z0-9_-]+$', sheet_identifier) and len(sheet_identifier) > 20:
                        spreadsheet_id = sheet_identifier
                    else:
                        # Try to find by name using Drive API
                        if 'drive' in self.services:
                            try:
                                query_string = f"name='{sheet_identifier}' and mimeType='application/vnd.google-apps.spreadsheet'"
                                results = self.services['drive'].files().list(q=query_string, fields="files(id,name)").execute()
                                files = results.get('files', [])
                                
                                if files:
                                    spreadsheet_id = files[0]['id']
                                else:
                                    response_text = f"No spreadsheet found with name '{sheet_identifier}'. Try using the full URL or spreadsheet ID."
                                    response_data = {"SyntaxPrime": response_text}
                                    save_conversation_enhanced(project, user_input, response_data)
                                    return response_data, True
                            except Exception as e:
                                response_text = f"Error searching for spreadsheet by name: {str(e)}"
                                response_data = {"SyntaxPrime": response_text}
                                save_conversation_enhanced(project, user_input, response_data)
                                return response_data, True
                        else:
                            response_text = "Drive API not available for name-based lookup. Please use the spreadsheet URL or ID."
                            response_data = {"SyntaxPrime": response_text}
                            save_conversation_enhanced(project, user_input, response_data)
                            return response_data, True
            
            if not spreadsheet_id:
                response_text = "Please provide either:\n- The full Google Sheets URL\n- The spreadsheet ID\n- The exact spreadsheet name"
                response_data = {"SyntaxPrime": response_text}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
            
            result = self.read_sheet_data(spreadsheet_id)
            
            if result['success']:
                response_text = f"**Sheet Data Retrieved:**\n\n"
                response_text += f"**Rows found:** {result['rows']}\n"
                response_text += f"**Spreadsheet URL:** {result['spreadsheet_url']}\n\n"
                
                if result['data']:
                    response_text += f"**Sample data (first 5 rows):**\n"
                    for i, row in enumerate(result['data'][:5]):
                        response_text += f"Row {i+1}: {row}\n"
                else:
                    response_text += "No data found in the spreadsheet."
            else:
                response_text = f"Failed to read spreadsheet: {result['error']}"
            
            response_data = {"SyntaxPrime": response_text}
            save_conversation_enhanced(project, user_input, response_data)
            return response_data, True
        
        return {}, False
        
# Section 8: Multi-Site Analytics and Search Console Command Handlers

    def handle_multi_site_analytics_command(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Handle analytics commands with multi-site support and flexible dates"""
        user_lower = user_input.lower().strip()
        
        # Parse site specification from command
        site_key = None
        
        # Look for site specification patterns
        site_patterns = [
            r'analytics for (.+?) (?:last|this|from|past)',
            r'analytics for (.+?)$',
            r'(.+?) analytics',
            r'(.+?) traffic',
            r'(.+?) website analytics'
        ]
        
        for pattern in site_patterns:
            match = re.search(pattern, user_lower)
            if match:
                potential_site = match.group(1).strip()
                found_site = self.find_site_by_name(potential_site)
                if found_site:
                    site_key = found_site
                    break
        
        # Handle "all sites" commands
        if 'all sites' in user_lower or 'all websites' in user_lower:
            start_date, end_date = self.parse_date_range_from_input(user_input)
            result = self.get_all_sites_analytics(start_date, end_date)
            
            if result['success']:
                response_text = f"**GA4 Analytics Report for All Sites ({start_date} to {end_date})**\n\n"
                
                for site_key, site_result in result['sites'].items():
                    if site_result['success']:
                        data = site_result['data']
                        site_name = site_result['site_name']
                        
                        if data:
                            total_sessions = sum(row['sessions'] for row in data)
                            total_users = sum(row['users'] for row in data)
                            total_pageviews = sum(row['pageviews'] for row in data)
                            
                            response_text += f"**{site_name}:**\n"
                            response_text += f"- Sessions: {total_sessions:,}\n"
                            response_text += f"- Users: {total_users:,}\n"
                            response_text += f"- Pageviews: {total_pageviews:,}\n\n"
                        else:
                            response_text += f"**{site_name}:** No data available\n\n"
                    else:
                        response_text += f"**{site_result['site_name']}:** {site_result['error']}\n\n"
            else:
                response_text = f"Failed to get analytics for all sites: {result['error']}"
            
            response_data = {"SyntaxPrime": response_text}
            save_conversation_enhanced(project, user_input, response_data)
            return response_data, True
        
        # Handle site list command
        elif 'list sites' in user_lower or 'available sites' in user_lower:
            sites = self.get_available_sites()
            
            if sites:
                response_text = f"**Available Sites for Analytics:**\n\n"
                for site in sites:
                    status_icons = []
                    if site['has_analytics']:
                        status_icons.append("GA4 Analytics")
                    if site['has_search_console']:
                        status_icons.append("Search Console")
                    
                    status = " | ".join(status_icons) if status_icons else "Not configured"
                    response_text += f"**{site['name']}** ({site['key']})\n"
                    response_text += f"- Status: {status}\n"
                    
                    if site['aliases']:
                        response_text += f"- Aliases: {', '.join(site['aliases'])}\n"
                    response_text += "\n"
                
                response_text += f"**Usage Examples:**\n"
                response_text += f"- \"analytics for {sites[0]['name']}\"\n"
                response_text += f"- \"analytics for {sites[0]['name']} last 6 months\"\n"
                response_text += f"- \"all sites analytics last year\"\n"
                response_text += f"- \"search console for {sites[0]['name']} last 3 months\"\n"
            else:
                response_text = "No sites configured. Set up GOOGLE_SITES_CONFIG environment variable."
            
            response_data = {"SyntaxPrime": response_text}
            save_conversation_enhanced(project, user_input, response_data)
            return response_data, True
        
        # Handle single site analytics with flexible date ranges
        else:
            start_date, end_date = self.parse_date_range_from_input(user_input)
            result = self.get_analytics_data(site_key, start_date, end_date)
            
            if result['success']:
                data = result['data']
                site_name = result.get('site_name', 'Unknown Site')
                
                # Store context for follow-up questions
                enhanced_conversation_context.store_analytics_report(
                    session_id=project,
                    site_name=site_name,
                    report_data=result,
                    user_query=user_input
                )
                
                response_text = f"**GA4 Analytics Report for {site_name} ({start_date} to {end_date})**\n\n"
                
                if data:
                    total_sessions = sum(row['sessions'] for row in data)
                    total_users = sum(row['users'] for row in data)
                    total_pageviews = sum(row['pageviews'] for row in data)
                    avg_bounce_rate = sum(row['bounce_rate'] for row in data) / len(data) if data else 0
                    
                    response_text += f"**Summary:**\n"
                    response_text += f"- Total Sessions: {total_sessions:,}\n"
                    response_text += f"- Total Users: {total_users:,}\n"
                    response_text += f"- Total Pageviews: {total_pageviews:,}\n"
                    response_text += f"- Average Bounce Rate: {avg_bounce_rate:.1f}%\n\n"
                    
                    # Show recent breakdown or full period depending on range
                    if len(data) <= 31:  # Show all days if 31 or fewer
                        response_text += f"**Daily Breakdown:**\n"
                        for row in data:
                            response_text += f"- {row['date']}: {row['sessions']} sessions, {row['users']} users\n"
                    else:  # Show summary for longer periods
                        response_text += f"**Recent Activity (Last 7 days):**\n"
                        for row in data[-7:]:
                            response_text += f"- {row['date']}: {row['sessions']} sessions, {row['users']} users\n"
                else:
                    response_text += "No analytics data found for the specified period."
            else:
                response_text = f"Failed to retrieve analytics: {result['error']}"
                
                # Show available sites if site not found
                if "not found" in result['error'].lower():
                    sites = self.get_available_sites()
                    if sites:
                        response_text += f"\n\n**Available sites:**\n"
                        for site in sites:
                            response_text += f"- {site['name']} ({site['key']})\n"
            
            response_data = {"SyntaxPrime": response_text}
            save_conversation_enhanced(project, user_input, response_data)
            return response_data, True
    
    def handle_multi_site_search_console_command(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Handle search console commands with multi-site support and flexible dates"""
        user_lower = user_input.lower().strip()
        
        # Parse site specification from command
        site_key = None
        
        # Look for site specification patterns
        site_patterns = [
            r'search console for (.+?)(?:\s+last|\s+this|\s+from|\s+past|$)',
            r'seo for (.+?)(?:\s+last|\s+this|\s+from|\s+past|$)',
            r'(.+?) search console',
            r'(.+?) seo'
        ]
        
        for pattern in site_patterns:
            match = re.search(pattern, user_lower)
            if match:
                potential_site = match.group(1).strip()
                found_site = self.find_site_by_name(potential_site)
                if found_site:
                    site_key = found_site
                    break
        
        # Parse date range
        start_date, end_date = self.parse_date_range_from_input(user_input)
        
        # Convert Google Analytics format to Search Console format (YYYY-MM-DD)
        if start_date.endswith('daysAgo'):
            days_ago = int(start_date.replace('daysAgo', ''))
            start_date_obj = datetime.date.today() - datetime.timedelta(days=days_ago)
            start_date = start_date_obj.strftime('%Y-%m-%d')
        elif start_date == 'yesterday':
            start_date_obj = datetime.date.today() - datetime.timedelta(days=1)
            start_date = start_date_obj.strftime('%Y-%m-%d')
        elif start_date == 'today':
            start_date = datetime.date.today().strftime('%Y-%m-%d')
        
        if end_date == 'today':
            end_date = datetime.date.today().strftime('%Y-%m-%d')
        
        result = self.get_search_console_data_for_site(site_key, start_date, end_date)
        
        if result['success']:
            data = result['data']
            site_name = result.get('site_name', 'Unknown Site')
            
            # Store context for follow-up questions
            enhanced_conversation_context.store_search_console_report(
                session_id=project,
                site_name=site_name,
                report_data=result,
                user_query=user_input
            )
            
            response_text = f"**Search Console Report for {site_name} ({result['date_range']})**\n\n"
            
            if data:
                total_clicks = sum(row['clicks'] for row in data)
                total_impressions = sum(row['impressions'] for row in data)
                avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
                
                response_text += f"**Summary:**\n"
                response_text += f"- Total Clicks: {total_clicks:,}\n"
                response_text += f"- Total Impressions: {total_impressions:,}\n"
                response_text += f"- Average CTR: {avg_ctr:.2f}%\n"
                response_text += f"- Total Queries: {result['total_queries']}\n\n"
                
                response_text += f"**Top Performing Queries:**\n"
                sorted_data = sorted(data, key=lambda x: x['clicks'], reverse=True)
                for i, row in enumerate(sorted_data[:10], 1):
                    response_text += f"{i}. {row['query']} - {row['clicks']} clicks, {row['impressions']} impressions\n"
            else:
                response_text += "No search console data found for the specified period."
        else:
            response_text = f"Failed to retrieve search console data: {result['error']}"
            
            # Show available sites if site not found
            if "not found" in result['error'].lower():
                sites = self.get_available_sites()
                if sites:
                    response_text += f"\n\n**Available sites:**\n"
                    for site in sites:
                        if site['has_search_console']:
                            response_text += f"- {site['name']} ({site['key']})\n"
        
        response_data = {"SyntaxPrime": response_text}
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True
        
# Section 9: Main Command Processor and Entry Point

    def process_google_commands(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Main command processor for all Google services with enhanced context handling"""
        user_lower = user_input.lower().strip()
        
        # PRIORITY 0: Check for follow-up questions first
        context = enhanced_conversation_context.get_context(project)
        if context and enhanced_conversation_context.detect_follow_up_question(user_input, context):
            response_data = enhanced_conversation_context.generate_contextual_response(
                user_input, context, project, use_voices, random_toggle
            )
            save_conversation_enhanced(project, user_input, response_data)
            return response_data, True
        
        # PRIORITY 1: Multi-site search console commands (MOVED UP - more specific than Gmail search)
        multi_search_triggers = ['search console for', 'seo for', 'search console', 'seo data']
        if any(trigger in user_lower for trigger in multi_search_triggers):
            return self.handle_multi_site_search_console_command(user_input, project, use_voices, random_toggle)
        
        # PRIORITY 2: Multi-site analytics commands
        multi_analytics_triggers = [
            'analytics for', 'all sites analytics', 'list sites', 'available sites',
            'analytics report', 'website analytics', 'site traffic'
        ]
        if any(trigger in user_lower for trigger in multi_analytics_triggers):
            return self.handle_multi_site_analytics_command(user_input, project, use_voices, random_toggle)
        
        # PRIORITY 3: Gmail/Calendar commands (moved down, with more specific search patterns)
        gmail_triggers = [
            'overnight', 'mail', 'emails', 'inbox', 'check mail',
            'calendar', 'today', 'meetings', 'schedule',
            'next meeting', 'next', 'upcoming',
            'good morning', 'morning', 'gm'
        ]
        
        # Gmail search commands (more specific patterns to avoid conflicts)
        gmail_search_triggers = ['search email', 'find email', 'email about']
        
        if any(trigger in user_lower for trigger in gmail_triggers):
            response_data, handled = self.handle_gmail_commands(user_input, project, use_voices, random_toggle)
            if handled:
                return response_data, True
        elif any(trigger in user_lower for trigger in gmail_search_triggers) or (user_lower.startswith('search ') and 'console' not in user_lower):
            # Only treat as Gmail search if it starts with 'search ' and doesn't contain 'console'
            response_data, handled = self.handle_gmail_commands(user_input, project, use_voices, random_toggle)
            if handled:
                return response_data, True
        
        # PRIORITY 4: Document creation commands
        docs_triggers = ['create document', 'create doc', 'add to document', 'append to document']
        if any(trigger in user_lower for trigger in docs_triggers):
            return self.handle_docs_command(user_input, project, use_voices, random_toggle)
        
        # PRIORITY 5: Spreadsheet commands
        sheets_triggers = ['create spreadsheet', 'create sheet', 'read sheet', 'get data from sheet']
        if any(trigger in user_lower for trigger in sheets_triggers):
            return self.handle_sheets_command(user_input, project, use_voices, random_toggle)
        
        # PRIORITY 6: Slides commands (placeholder for future)
        slides_triggers = ['create presentation', 'create slides', 'add slide']
        if any(trigger in user_lower for trigger in slides_triggers):
            response_data = {"SyntaxPrime": "Google Slides integration coming soon! Currently supports Google Docs and Sheets."}
            save_conversation_enhanced(project, user_input, response_data)
            return response_data, True
        
        return {}, False

# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def process_google_ecosystem_commands(user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
    """Main entry point for all Google ecosystem commands"""
    if not GOOGLE_APIS_AVAILABLE:
        error_response = {"SyntaxPrime": "Google API libraries not available. Please install google-api-python-client and google-auth packages."}
        return error_response, True
    
    # Initialize the Google integration
    google_integration = GoogleIntegration()
    
    # Process the command
    return google_integration.process_google_commands(user_input, project, use_voices, random_toggle)

