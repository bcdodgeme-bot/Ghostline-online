# =============================================================================
# SECTION 1: IMPORTS AND BASE CLASSES
# =============================================================================

# modules/enhanced_google_integration.py - Consolidated Google Ecosystem Integration with GA4 Data API

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
    get_next_meeting, format_calendar_summary,
    list_today_events_all_calendars, format_calendar_summary_enhanced
)
from utils.ghostline_engine import generate_response
from utils.rag_basic import is_ready
from modules.database import save_conversation_enhanced, save_daily_log_enhanced
from modules.brain import enhanced_retrieve
# Add the analytics validation import
from modules.analytics_validation_enhanced import validate_analytics_data_comprehensive, should_block_ai_suggestions_enhanced

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

# =============================================================================
# SECTION 2: ENHANCED CONVERSATIONAL CONTEXT CLASS
# =============================================================================
# =============================================================================
# SECTION 2: ENHANCED CONVERSATIONAL CONTEXT CLASS - FIXED VERSION 9/11/25
# =============================================================================

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
    
    def generate_contextual_response(self, user_input: str, context: dict, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """FIXED: Generate contextual response based on stored context"""
        try:
            context_type = context.get('type', 'unknown')
            
            # Build contextual prompt based on context type
            if context_type == 'search_console_report':
                site_name = context.get('site_name', 'Unknown Site')
                summary = context.get('summary', 'No summary available')
                
                contextual_prompt = f"""The user is asking a follow-up question about a recent Search Console report for {site_name}.
                
Original query: {context.get('original_query', 'Unknown')}
Report summary: {summary}

User's follow-up question: {user_input}

Please provide specific insights and recommendations based on the Search Console data mentioned above."""
                
            elif context_type == 'analytics_report':
                site_name = context.get('site_name', 'Unknown Site')
                summary = context.get('summary', 'No summary available')
                
                contextual_prompt = f"""The user is asking a follow-up question about a recent Google Analytics report for {site_name}.
                
Original query: {context.get('original_query', 'Unknown')}
Report summary: {summary}

User's follow-up question: {user_input}

Please provide specific insights and recommendations based on the analytics data mentioned above."""
                
            elif context_type.startswith('gmail_'):
                email_count = context.get('email_count', 0)
                
                contextual_prompt = f"""The user is asking a follow-up question about a recent Gmail report.
                
Email report contained {email_count} emails.
User's follow-up question: {user_input}

Please provide helpful insights about email management and priorities."""
                
            else:
                # Generic contextual response
                contextual_prompt = f"""The user is asking a follow-up question about a recent report or interaction.
                
Context type: {context_type}
User's follow-up question: {user_input}

Please provide helpful insights based on the context."""
            
            # Get retrieval context for enhanced responses
            retrieval_ctx = enhanced_retrieve(contextual_prompt, k=5, project=project) if is_ready() else []
            
            # Generate the response
            response_data = generate_response(
                contextual_prompt, use_voices, random_toggle,
                project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
            )
            
            return response_data, True
            
        except Exception as e:
            print(f"Contextual response generation failed: {e}")
            # Fallback to normal response
            return {}, False
    
    def _generate_analytics_summary(self, data: list) -> str:
        """Generate summary for analytics data"""
        if not data:
            return "No analytics data available"
        
        total_sessions = sum(row.get('sessions', 0) for row in data)
        total_users = sum(row.get('users', 0) for row in data)
        return f"{total_sessions} sessions, {total_users} users over {len(data)} days"
    
    def _generate_search_console_summary(self, data: list) -> str:
        """Generate summary for search console data"""
        if not data:
            return "No search console data available"
        
        total_clicks = sum(row.get('clicks', 0) for row in data)
        total_impressions = sum(row.get('impressions', 0) for row in data)
        return f"{total_clicks} clicks, {total_impressions} impressions from {len(data)} queries"
    
    def _generate_gmail_summary(self, emails: list, report_type: str) -> str:
        """Generate summary for Gmail data"""
        return f"{len(emails)} emails in {report_type} report"
    
    def _generate_calendar_summary(self, events: list, report_type: str) -> str:
        """Generate summary for calendar data"""
        return f"{len(events)} events in {report_type} calendar report"

# Initialize global context manager
enhanced_conversation_context = EnhancedConversationalContext()

# =============================================================================
# SECTION 3: GOOGLE INTEGRATION MAIN CLASS
# =============================================================================

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
            # Import the token manager if available
            try:
                from modules.google_token_refresh import get_google_credentials, token_manager
                # Get valid credentials with automatic refresh
                self.credentials = get_google_credentials()
            except ImportError:
                # Fallback to direct credential loading
                from utils.gmail_client import _build_creds
                self.credentials = _build_creds()
            
            if not self.credentials:
                print("No valid Google credentials available - authentication required")
                return
            
            print("Valid Google credentials obtained")
            
            # Initialize all services with refreshed credentials
            try:
                # Core services
                self.services['gmail'] = build('gmail', 'v1', credentials=self.credentials, cache_discovery=False)
                self.services['calendar'] = build('calendar', 'v3', credentials=self.credentials, cache_discovery=False)
                self.services['drive'] = build('drive', 'v3', credentials=self.credentials, cache_discovery=False)
                self.services['docs'] = build('docs', 'v1', credentials=self.credentials, cache_discovery=False)
                self.services['sheets'] = build('sheets', 'v4', credentials=self.credentials, cache_discovery=False)
                
                # Analytics services
                try:
                    self.services['analyticsdata'] = build('analyticsdata', 'v1beta', credentials=self.credentials, cache_discovery=False)
                    self.services['searchconsole'] = build('searchconsole', 'v1', credentials=self.credentials, cache_discovery=False)
                    print("Analytics services initialized: ['analyticsdata', 'searchconsole']")
                except Exception as e:
                    print(f"Analytics services failed to initialize: {e}")
                
                print("Google services initialized: ['gmail', 'calendar', 'drive', 'docs', 'sheets']")
                
            except Exception as e:
                print(f"Error initializing Google services: {e}")
                
        except Exception as e:
            print(f"Failed to initialize Google integration: {e}")
    
    def _extract_email_sender(self, msg):
        """Extract sender from email message"""
        if not isinstance(msg, dict):
            return "Unknown Sender"
        
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
        
        return "Unknown Sender"

    def _extract_email_subject(self, msg):
        """Extract subject from email message"""
        if not isinstance(msg, dict):
            return "No Subject"
        
        # Try different possible field names for subject
        for subject_field in ['subject', 'Subject', 'title', 'summary', 'snippet']:
            if subject_field in msg:
                subject = msg[subject_field]
                if subject:
                    return subject
        
        return "No Subject"
        
# =============================================================================
# SECTION 4: SITE CONFIGURATION AND DATE PARSING METHODS
# =============================================================================

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

# =============================================================================
# SECTION 5: GMAIL AND CALENDAR COMMAND HANDLERS
# =============================================================================
# =============================================================================
# SECTION 5: GMAIL AND CALENDAR COMMAND HANDLERS 9/12/25
# =============================================================================

    def handle_gmail_commands(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Handle Gmail and Calendar commands with enhanced multi-calendar support and FIXED calendar data formatting"""
        user_lower = user_input.lower().strip()
        
        # SUPER MORNING COMMANDS - More specific patterns to avoid false triggers
        super_morning_commands = [
            "daily briefing", "morning briefing", "super morning",
            "full briefing", "start my day", "what's up today",
            "complete briefing", "daily", "briefing", "everything"
        ]
        
        if user_lower in super_morning_commands:
            response_data = self.handle_super_morning_command(project, use_voices, random_toggle)
            save_conversation_enhanced(project, user_input, response_data)
            return response_data, True
        
        # Gmail overnight - more specific patterns
        overnight_patterns = ["overnight", "overnight emails", "check mail", "inbox", "new emails"]
        if any(pattern == user_lower for pattern in overnight_patterns):
            try:
                print("Fetching overnight emails...")
                msgs = list_overnight(include_unread=True, include_primary=False)
                
                if not msgs or not isinstance(msgs, list):
                    response_data = {"SyntaxPrime": "Gmail service unavailable. Check your Google OAuth setup."}
                    save_conversation_enhanced(project, user_input, response_data)
                    return response_data, True
                
                if len(msgs) == 0:
                    summary_prompt = "No new emails since midnight. Your inbox is caught up."
                else:
                    # Store context for follow-ups
                    ConversationalEmailContext.store_morning_briefing_emails(
                        session_id=project,
                        emails=[{
                            'sender': self._extract_email_sender(msg),
                            'subject': self._extract_email_subject(msg),
                            'original_msg': msg
                        } for msg in msgs[:10]]
                    )
                    
                    summary_prompt = f"Found {len(msgs)} overnight emails:\n\n"
                    for i, msg in enumerate(msgs[:5], 1):
                        sender = self._extract_email_sender(msg)
                        subject = self._extract_email_subject(msg)
                        summary_prompt += f"{i}. {sender}: {subject}\n"
                    
                    if len(msgs) > 5:
                        summary_prompt += f"\n... and {len(msgs) - 5} more emails"
                
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
                
            except Exception as e:
                print(f"Overnight emails check failed: {e}")
                response_data = {"SyntaxPrime": f"Gmail integration error: {str(e)}. Please check your Google OAuth setup."}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True

        # Gmail search - more specific patterns
        if user_lower.startswith(("search ", "find ", "email about ")):
            try:
                # Extract search term
                search_term = user_input[user_input.find(' ') + 1:]  # Get everything after first space
                print(f"Gmail search for: {search_term}")
                
                msgs = gmail_search(search_term)
                
                if not msgs or not isinstance(msgs, list):
                    response_data = {"SyntaxPrime": f"Gmail search unavailable. Check your Google OAuth setup."}
                    save_conversation_enhanced(project, user_input, response_data)
                    return response_data, True
                
                if len(msgs) == 0:
                    summary_prompt = f"No emails found matching '{search_term}'"
                else:
                    summary_prompt = f"Found {len(msgs)} emails matching '{search_term}':\n\n"
                    for i, msg in enumerate(msgs[:5], 1):
                        sender = self._extract_email_sender(msg)
                        subject = self._extract_email_subject(msg)
                        summary_prompt += f"{i}. {sender}: {subject}\n"
                
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
        
        # Calendar commands - FIXED: exact matches only with proper data formatting
        calendar_exact_matches = ["calendar", "today", "meetings", "schedule"]
        if user_lower in calendar_exact_matches:
            try:
                print("Fetching today's calendar events from all calendars...")
                # Use the new all-calendars approach
                events = list_today_events_all_calendars(max_results=20)
                
                if not events or not isinstance(events, list):
                    response_data = {"SyntaxPrime": "Calendar service unavailable. Check your Google OAuth setup."}
                    save_conversation_enhanced(project, user_input, response_data)
                    return response_data, True
                
                if len(events) == 0:
                    summary_prompt = "REAL CALENDAR DATA: No events found for today across all your calendars. Your schedule is completely clear.\n\nCRITICAL INSTRUCTION: Do not fabricate or invent any meetings, appointments, or events. The calendar is genuinely empty for today."
                else:
                    # FIXED: Format real calendar data properly for AI
                    summary_prompt = f"REAL CALENDAR DATA for today ({len(events)} actual events):\n\n"
                    summary_prompt += "CRITICAL INSTRUCTION: Use ONLY the following real calendar events. Do not add, invent, or fabricate any additional meetings.\n\n"
                    summary_prompt += "TODAY'S ACTUAL EVENTS:\n"
                    
                    for i, event in enumerate(events, 1):
                        # Extract real event details
                        if isinstance(event, dict):
                            title = event.get('title', event.get('summary', 'Untitled Event'))
                            start_time = event.get('start_time_formatted', event.get('start_formatted', 'Time not specified'))
                            calendar_name = event.get('calendar_name', 'Unknown Calendar')
                            location = event.get('location', '')
                            
                            summary_prompt += f"{i}. **{title}**\n"
                            summary_prompt += f"   - Time: {start_time}\n"
                            summary_prompt += f"   - Calendar: {calendar_name}\n"
                            if location:
                                summary_prompt += f"   - Location: {location}\n"
                            summary_prompt += "\n"
                        else:
                            # Fallback for string format
                            summary_prompt += f"{i}. {event}\n"
                    
                    summary_prompt += f"\nEND OF REAL CALENDAR DATA - These are the {len(events)} actual events in your calendar today. Do not mention any events not listed above."
                    
                    # Store context for follow-ups
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
        
        # Next meeting command - FIXED: exact matches only with proper data formatting
        next_meeting_matches = ["next meeting", "next", "upcoming"]
        if user_lower in next_meeting_matches:
            try:
                print("Getting next meeting...")
                next_meeting = get_next_meeting()
                
                if not next_meeting or isinstance(next_meeting, dict) and "error" in next_meeting:
                    summary_prompt = "REAL CALENDAR DATA: No upcoming meetings found in your calendar.\n\nCRITICAL INSTRUCTION: Do not fabricate or invent any meetings. There genuinely are no upcoming meetings scheduled."
                elif next_meeting and next_meeting.get('summary'):
                    # FIXED: Format real meeting data properly
                    title = next_meeting.get('summary', 'Untitled Meeting')
                    start_time = next_meeting.get('start_formatted', 'Time not specified')
                    calendar_name = next_meeting.get('calendar_name', '')
                    location = next_meeting.get('location', '')
                    
                    summary_prompt = f"REAL CALENDAR DATA - Your next meeting:\n\n"
                    summary_prompt += f"CRITICAL INSTRUCTION: This is the actual next meeting from your calendar. Do not invent or add any other meetings.\n\n"
                    summary_prompt += f"NEXT MEETING:\n"
                    summary_prompt += f"**{title}**\n"
                    summary_prompt += f"- Time: {start_time}\n"
                    if calendar_name:
                        summary_prompt += f"- Calendar: {calendar_name}\n"
                    if location:
                        summary_prompt += f"- Location: {location}\n"
                    summary_prompt += f"\nThis is your only upcoming meeting. Do not mention any other meetings."
                else:
                    summary_prompt = "REAL CALENDAR DATA: Next meeting lookup completed but no readable meeting data found.\n\nCRITICAL INSTRUCTION: Do not fabricate meeting details."
                
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
        
        return {}, False

    def handle_super_morning_command(self, project: str, use_voices: list, random_toggle: bool) -> Dict:
        """FIXED: Handle super morning briefing with proper calendar data formatting"""
        try:
            print("🌅 Starting comprehensive morning briefing...")
            
            # Gather all data
            overnight_emails = list_overnight(include_unread=True, include_primary=False)
            today_events = list_today_events_all_calendars(max_results=20)
            next_meeting = get_next_meeting()
            
            # Build comprehensive anti-fabrication prompt with REAL data
            summary_prompt = """COMPREHENSIVE MORNING BRIEFING - REAL DATA ONLY

CRITICAL INSTRUCTION: This briefing contains only REAL data from your accounts.
Do not fabricate, invent, or imagine any additional emails, meetings, or information.
All data below is verified and extracted from actual sources.

"""
            
            # Add email section with real data
            if overnight_emails and isinstance(overnight_emails, list) and len(overnight_emails) > 0:
                summary_prompt += f"📧 EMAILS: {len(overnight_emails)} new overnight emails\n"
                for i, msg in enumerate(overnight_emails[:5], 1):
                    sender = self._extract_email_sender(msg)
                    subject = self._extract_email_subject(msg)
                    summary_prompt += f"  {i}. {sender or 'Unknown'}: {subject or 'No Subject'}\n"
                
                if len(overnight_emails) > 5:
                    summary_prompt += f"  ... and {len(overnight_emails) - 5} more emails\n"
                summary_prompt += "\n"
            else:
                summary_prompt += "📧 EMAILS: No new overnight emails found. Your inbox is up to date.\n\n"
            
            # Add calendar section with REAL data - FIXED formatting
            if today_events and isinstance(today_events, list) and len(today_events) > 0:
                summary_prompt += f"📅 CALENDAR: {len(today_events)} actual events today\n"
                summary_prompt += "REAL EVENTS FROM YOUR CALENDAR:\n"
                
                for i, event in enumerate(today_events, 1):
                    if isinstance(event, dict):
                        title = event.get('title', event.get('summary', 'Untitled Event'))
                        start_time = event.get('start_time_formatted', event.get('start_formatted', 'Time not specified'))
                        calendar_name = event.get('calendar_name', '')
                        location = event.get('location', '')
                        
                        summary_prompt += f"  {i}. **{title}** at {start_time}"
                        if calendar_name:
                            summary_prompt += f" ({calendar_name})"
                        if location:
                            summary_prompt += f" - {location}"
                        summary_prompt += "\n"
                    else:
                        # Handle string format
                        summary_prompt += f"  {i}. {event}\n"
                
                summary_prompt += "\n"
            else:
                summary_prompt += "📅 CALENDAR: No events scheduled for today. You have a clear calendar.\n\n"
            
            # Add next meeting info with REAL data
            if next_meeting and isinstance(next_meeting, dict) and next_meeting.get('summary'):
                title = next_meeting.get('summary', 'Untitled Meeting')
                start_time = next_meeting.get('start_formatted', 'Time not specified')
                calendar_name = next_meeting.get('calendar_name', '')
                
                summary_prompt += f"⏰ NEXT MEETING: **{title}** at {start_time}"
                if calendar_name:
                    summary_prompt += f" ({calendar_name})"
                summary_prompt += "\n\n"
            else:
                summary_prompt += "⏰ NEXT MEETING: No upcoming meetings found.\n\n"
            
            summary_prompt += "END OF REAL DATA - Do not add any additional information not listed above."
            
            # Generate response with context
            retrieval_ctx = enhanced_retrieve(summary_prompt, k=5, project=project) if is_ready() else []
            
            response_data = generate_response(
                summary_prompt, use_voices, random_toggle,
                project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
            )
            
            print(f"🏁 Morning briefing completed with real data")
            
            return response_data
            
        except Exception as e:
            error_msg = f"Morning briefing failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {"SyntaxPrime": f"Morning briefing error: {error_msg}. Please check your Google OAuth setup."}
# =============================================================================
# SECTION 6: MAIN COMMAND PROCESSOR (WITH YOUR UPDATED CODE)
# =============================================================================
# =============================================================================
# SECTION 6: MAIN COMMAND PROCESSOR AND MISSING METHODS 9/11/25
# =============================================================================
# =============================================================================
# SECTION 6: MAIN COMMAND PROCESSOR AND MISSING METHODS - FIXED SITE NAME PARSING 9/12/25
# =============================================================================
# =============================================================================
# SECTION 6: MAIN COMMAND PROCESSOR AND MISSING METHODS - ANTI-HALLUCINATION VERSION 9/12/25
# =============================================================================
# =============================================================================
# SECTION 6: MAIN COMMAND PROCESSOR AND MISSING METHODS - ANTI-HALLUCINATION VERSION WITH REVERSE PATTERN FIX 9/12/25
# =============================================================================

    def find_site_by_name(self, site_query: str) -> Optional[str]:
        """Find site key by name or alias - ENHANCED with better multi-word matching"""
        if not site_query:
            return None
            
        site_query_clean = site_query.lower().strip()
        
        print(f"🔍 Site search: Looking for '{site_query_clean}' in {list(self.sites_config.keys())}")
        
        # Method 1: Direct key match (exact)
        if site_query_clean in self.sites_config:
            print(f"✅ Direct key match: {site_query_clean}")
            return site_query_clean
        
        # Method 2: Convert multi-word site name to expected key format
        # "Damn It Carl" -> "damn_it_carl"
        # "TV Signals" -> "tv_signals"
        # "Rose and Angel" -> "rose_and_angel"
        # "Meals N Feelz" -> "meals_n_feelz"
        key_format = site_query_clean.replace(' ', '_').replace('&', 'and').replace('n ', 'n_')
        if key_format in self.sites_config:
            print(f"✅ Key format match: '{site_query_clean}' -> '{key_format}'")
            return key_format
        
        # Method 3: Exact name match (case insensitive)
        for key, config in self.sites_config.items():
            if config['name'].lower() == site_query_clean:
                print(f"✅ Name match: '{site_query_clean}' -> '{key}' ({config['name']})")
                return key
        
        # Method 4: Alias match (exact)
        for key, config in self.sites_config.items():
            aliases = [alias.strip().lower() for alias in config.get('aliases', [])]
            if site_query_clean in aliases:
                print(f"✅ Alias match: '{site_query_clean}' -> '{key}' (alias: {site_query_clean})")
                return key
        
        # Method 5: Partial name match (contains)
        for key, config in self.sites_config.items():
            config_name_lower = config['name'].lower()
            if site_query_clean in config_name_lower or config_name_lower in site_query_clean:
                print(f"✅ Partial name match: '{site_query_clean}' -> '{key}' ({config['name']})")
                return key
        
        # Method 6: Partial alias match
        for key, config in self.sites_config.items():
            aliases = [alias.strip().lower() for alias in config.get('aliases', [])]
            for alias in aliases:
                if site_query_clean in alias or alias in site_query_clean:
                    print(f"✅ Partial alias match: '{site_query_clean}' -> '{key}' (partial alias: {alias})")
                    return key
        
        print(f"❌ No site match found for: '{site_query_clean}'")
        print(f"Available sites: {[(key, config['name']) for key, config in self.sites_config.items()]}")
        return None

    def format_search_console_response(self, result: Dict, user_input: str) -> str:
        """ANTI-HALLUCINATION: Template-based Search Console response using ONLY real API data"""
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            response = f"**Search Console Error:**\n{error_msg}\n\n"
            
            # Show available sites if site not found
            if "not found" in error_msg.lower():
                sites = self.get_available_sites()
                if sites:
                    response += f"**Available sites for Search Console:**\n"
                    for site in sites:
                        if site.get('has_search_console'):
                            response += f"• **{site['name']}** (key: `{site['key']}`)\n"
                            if site['aliases']:
                                response += f"  Aliases: {', '.join(site['aliases'])}\n"
                    
                    response += f"\n**Usage examples:**\n"
                    response += f"• `search console for {sites[0]['name']}`\n"
                    response += f"• `search console {sites[0]['name']} last 30 days`\n"
            
            return response
        
        # SUCCESS: Format real API data ONLY
        data = result.get('data', [])
        site_name = result.get('site_name', 'Unknown Site')
        date_range = result.get('date_range', 'Unknown range')
        
        response = f"**Search Console Report for {site_name} ({date_range})**\n\n"
        response += f"**Summary:**\n"
        response += f"- Total Clicks: {result.get('total_clicks', 0):,}\n"
        response += f"- Total Impressions: {result.get('total_impressions', 0):,}\n"
        response += f"- Average CTR: {result.get('average_ctr', 0):.2f}%\n"
        response += f"- Total Queries: {result.get('total_queries', 0)}\n\n"
        
        if data and isinstance(data, list) and len(data) > 0:
            response += f"**Top Performing Queries:**\n"
            sorted_data = sorted([row for row in data if isinstance(row, dict)],
                               key=lambda x: x.get('clicks', 0), reverse=True)
            for i, row in enumerate(sorted_data[:10], 1):
                query = row.get('query', 'Unknown query')
                clicks = row.get('clicks', 0)
                impressions = row.get('impressions', 0)
                response += f"{i}. {query} - {clicks} clicks, {impressions} impressions\n"
        else:
            response += "No search console data found for the specified period."
        
        return response

    def format_analytics_response(self, result: Dict, start_date: str, end_date: str) -> str:
        """ANTI-HALLUCINATION: Template-based Analytics response using ONLY real API data"""
        if not result.get('success'):
            error_msg = result.get('error', 'Unknown error')
            response = f"**Analytics Error:**\n{error_msg}\n\n"
            
            # Show available sites if site not found
            if "not found" in error_msg.lower():
                sites = self.get_available_sites()
                if sites:
                    response += f"**Available sites for Analytics:**\n"
                    for site in sites:
                        if site.get('has_analytics'):
                            response += f"• **{site['name']}** (key: `{site['key']}`)\n"
                            if site['aliases']:
                                response += f"  Aliases: {', '.join(site['aliases'])}\n"
                    
                    response += f"\n**Usage examples:**\n"
                    response += f"• `analytics for {sites[0]['name']}`\n"
                    response += f"• `analytics for {sites[0]['name']} last 30 days`\n"
            
            return response
        
        # SUCCESS: Format real API data ONLY
        data = result.get('data', [])
        site_name = result.get('site_name', 'Unknown Site')
        
        response = f"**GA4 Analytics Report for {site_name} ({start_date} to {end_date})**\n\n"
        response += f"**Summary:**\n"
        response += f"- Total Sessions: {result.get('total_sessions', 0):,}\n"
        response += f"- Total Users: {result.get('total_users', 0):,}\n"
        response += f"- Total Pageviews: {result.get('total_pageviews', 0):,}\n"
        response += f"- Average Bounce Rate: {result.get('avg_bounce_rate', 0):.1f}%\n\n"
        
        if data and isinstance(data, list) and len(data) > 0:
            # Show recent breakdown or full period depending on range
            if len(data) <= 31:  # Show all days if 31 or fewer
                response += f"**Daily Breakdown:**\n"
                for row in data:
                    if isinstance(row, dict):
                        date = row.get('date', 'Unknown')
                        sessions = row.get('sessions', 0)
                        users = row.get('users', 0)
                        response += f"- {date}: {sessions} sessions, {users} users\n"
            else:  # Show summary for longer periods
                response += f"**Recent Activity (Last 7 days):**\n"
                for row in data[-7:]:
                    if isinstance(row, dict):
                        date = row.get('date', 'Unknown')
                        sessions = row.get('sessions', 0)
                        users = row.get('users', 0)
                        response += f"- {date}: {sessions} sessions, {users} users\n"
        else:
            response += "No analytics data found for the specified period."
        
        return response

    def format_all_sites_analytics_response(self, result: Dict, start_date: str, end_date: str) -> str:
        """ANTI-HALLUCINATION: Template-based response for all sites analytics using ONLY real API data"""
        if not result.get('success'):
            return f"**Error:** {result.get('error', 'Failed to get analytics for all sites')}"
        
        response = f"**GA4 Analytics Report for All Sites ({start_date} to {end_date})**\n\n"
        
        sites_data = result.get('sites', {})
        for site_key, site_result in sites_data.items():
            if site_result.get('success'):
                data = site_result.get('data', [])
                site_name = site_result.get('site_name', site_key)
                
                if data and isinstance(data, list):
                    total_sessions = sum(row.get('sessions', 0) for row in data if isinstance(row, dict))
                    total_users = sum(row.get('users', 0) for row in data if isinstance(row, dict))
                    total_pageviews = sum(row.get('pageviews', 0) for row in data if isinstance(row, dict))
                    
                    response += f"**{site_name}:**\n"
                    response += f"- Sessions: {total_sessions:,}\n"
                    response += f"- Users: {total_users:,}\n"
                    response += f"- Pageviews: {total_pageviews:,}\n\n"
                else:
                    response += f"**{site_name}:** No data available\n\n"
            else:
                site_name = site_result.get('site_name', site_key)
                error = site_result.get('error', 'Unknown error')
                response += f"**{site_name}:** Error - {error}\n\n"
        
        return response

    def format_sites_list_response(self) -> str:
        """ANTI-HALLUCINATION: Template-based sites list using ONLY configuration data"""
        sites = self.get_available_sites()
        
        if not sites:
            return "No sites configured. Set up GOOGLE_SITES_CONFIG environment variable."
        
        response = f"**Available Sites for Analytics:**\n\n"
        for site in sites:
            status_icons = []
            if site['has_analytics']:
                status_icons.append("GA4 Analytics")
            if site['has_search_console']:
                status_icons.append("Search Console")
            
            status = " | ".join(status_icons) if status_icons else "Not configured"
            response += f"**{site['name']}** (key: `{site['key']}`)\n"
            response += f"- Status: {status}\n"
            
            if site['aliases']:
                response += f"- Aliases: {', '.join(site['aliases'])}\n"
            response += "\n"
        
        response += f"**Usage Examples:**\n"
        response += f"- `analytics for {sites[0]['name']}`\n"
        response += f"- `analytics for {sites[0]['name']} last 6 months`\n"
        response += f"- `all sites analytics last year`\n"
        response += f"- `search console for {sites[0]['name']} last 3 months`\n"
        
        return response

    def process_google_commands(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Main command processor for all Google services with ANTI-HALLUCINATION response generation - FIXED REVERSE PATTERNS"""
        user_lower = user_input.lower().strip()
        
        print(f"🔍 Google command processing: '{user_input}'")
        
        # PRIORITY 0: Check for follow-up questions first
        context = enhanced_conversation_context.get_context(project)
        if context and enhanced_conversation_context.detect_follow_up_question(user_input, context):
            response_data, handled = enhanced_conversation_context.generate_contextual_response(
                user_input, context, project, use_voices, random_toggle
            )
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
            return response_data, handled
        
        # Blog suggestions commands - WITH VALIDATION BLOCKING
        blog_patterns = [
            'blog suggestions', 'blog ideas', 'content ideas', 'what to write',
            'blog suggestions for', 'content suggestions', 'post ideas'
        ]
        
        if any(pattern in user_lower for pattern in blog_patterns):
            print("✏️ Detected blog suggestions command - will validate data first")
            response_data, handled = self.handle_blog_suggestions_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
            return response_data, handled
        
        # PRIORITY 1: Multi-site search console commands - TEMPLATE-BASED RESPONSES
        search_console_patterns = [
            'search console for', 'seo for', 'search console', 'seo data'
        ]
        
        if any(trigger in user_lower for trigger in search_console_patterns):
            print("🔍 Detected Search Console command")
            
            # Extract site name using enhanced parsing
            site_key = None
            site_name_extracted = None
            
            # Strategy 1: "search console for SITE_NAME [time_period]"
            for_pattern = re.search(r'search console for\s+(.+?)(?:\s+(?:last|this|from|past|in)\s|$)', user_lower)
            if for_pattern:
                site_name_extracted = for_pattern.group(1).strip()
            
            # Strategy 2: "SITE_NAME search console"
            if not site_name_extracted:
                site_console_pattern = re.search(r'^(.+?)\s+search console', user_lower)
                if site_console_pattern:
                    site_name_extracted = site_console_pattern.group(1).strip()
            
            # Strategy 3: "search console SITE_NAME"
            if not site_name_extracted:
                console_site_pattern = re.search(r'search console\s+(.+?)(?:\s+(?:last|this|from|past|in)\s|$)', user_lower)
                if console_site_pattern:
                    site_name_extracted = console_site_pattern.group(1).strip()
            
            if site_name_extracted:
                site_key = self.find_site_by_name(site_name_extracted)
            
            # Parse date range
            start_date, end_date = self.parse_date_range_from_input(user_input)
            
            # Convert to Search Console format
            if start_date.endswith('daysAgo'):
                days_ago = int(start_date.replace('daysAgo', ''))
                start_date_obj = datetime.date.today() - datetime.timedelta(days=days_ago)
                start_date = start_date_obj.strftime('%Y-%m-%d')
            elif start_date == 'today':
                start_date = datetime.date.today().strftime('%Y-%m-%d')
            
            if end_date == 'today':
                end_date = datetime.date.today().strftime('%Y-%m-%d')
            
            # Get data and format with template (NO AI GENERATION)
            result = self.get_search_console_data_for_site(site_key, start_date, end_date)
            response_text = self.format_search_console_response(result, user_input)
            
            response_data = {"SyntaxPrime": response_text}
            save_conversation_enhanced(project, user_input, response_data)
            return response_data, True
        
        # PRIORITY 2: Multi-site analytics commands - TEMPLATE-BASED RESPONSES WITH REVERSE PATTERN FIX
        analytics_triggers = [
            'analytics for', 'all sites analytics', 'list sites', 'available sites',
            'analytics report', 'website analytics', 'site traffic'
        ]
        
        # Check for direct analytics triggers
        is_analytics_command = any(trigger in user_lower for trigger in analytics_triggers)
        
        # CRITICAL FIX: Also check for "[SITE] analytics" and "[SITE] traffic" patterns
        if not is_analytics_command:
            # Pattern: "Rose and Angel analytics", "TV Signals analytics", etc.
            reverse_analytics_pattern = re.search(r'^(.+?)\s+(analytics|traffic)(?:\s|$)', user_lower)
            if reverse_analytics_pattern:
                potential_site = reverse_analytics_pattern.group(1).strip()
                # Verify this is actually a configured site
                if self.find_site_by_name(potential_site):
                    is_analytics_command = True
                    print(f"📊 Detected reverse analytics pattern: '{potential_site} {reverse_analytics_pattern.group(2)}'")
        
        if is_analytics_command:
            print("📊 Detected Analytics command")
            
            # Handle "all sites" commands
            if 'all sites' in user_lower or 'all websites' in user_lower:
                start_date, end_date = self.parse_date_range_from_input(user_input)
                result = self.get_all_sites_analytics(start_date, end_date)
                response_text = self.format_all_sites_analytics_response(result, start_date, end_date)
                
                response_data = {"SyntaxPrime": response_text}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
            
            # Handle site list command
            elif 'list sites' in user_lower or 'available sites' in user_lower:
                response_text = self.format_sites_list_response()
                
                response_data = {"SyntaxPrime": response_text}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
            
            # Handle single site analytics
            else:
                # Extract site name using enhanced parsing - FIXED TO HANDLE BOTH PATTERNS
                site_key = None
                site_name_extracted = None
                
                # Strategy 1: "analytics for SITE_NAME [time_period]"
                for_pattern = re.search(r'analytics for\s+(.+?)(?:\s+(?:last|this|from|past|in)\s|$)', user_lower)
                if for_pattern:
                    site_name_extracted = for_pattern.group(1).strip()
                
                # Strategy 2: "SITE_NAME analytics" - THIS WAS MISSING!
                if not site_name_extracted:
                    reverse_pattern = re.search(r'^(.+?)\s+analytics', user_lower)
                    if reverse_pattern:
                        site_name_extracted = reverse_pattern.group(1).strip()
                
                # Strategy 3: "SITE_NAME traffic"
                if not site_name_extracted:
                    traffic_pattern = re.search(r'^(.+?)\s+traffic', user_lower)
                    if traffic_pattern:
                        site_name_extracted = traffic_pattern.group(1).strip()
                
                if site_name_extracted:
                    site_key = self.find_site_by_name(site_name_extracted)
                
                start_date, end_date = self.parse_date_range_from_input(user_input)
                result = self.get_analytics_data(site_key, start_date, end_date)
                response_text = self.format_analytics_response(result, start_date, end_date)
                
                response_data = {"SyntaxPrime": response_text}
                save_conversation_enhanced(project, user_input, response_data)
                return response_data, True
        
        # PRIORITY 3: Gmail/Calendar commands (keep AI generation for these)
        gmail_triggers = [
            'overnight', 'mail', 'emails', 'inbox', 'check mail',
            'calendar', 'today', 'meetings', 'schedule',
            'next meeting', 'next', 'upcoming',
            'good morning', 'morning', 'gm'
        ]
        
        if any(trigger in user_lower for trigger in gmail_triggers):
            response_data, handled = self.handle_gmail_commands(user_input, project, use_voices, random_toggle)
            if handled:
                return response_data, True
        
        # PRIORITY 4: Document creation commands (keep AI generation)
        docs_triggers = ['create document', 'create doc', 'add to document', 'append to document']
        if any(trigger in user_lower for trigger in docs_triggers):
            return self.handle_docs_command(user_input, project, use_voices, random_toggle)
        
        # PRIORITY 5: Spreadsheet commands (keep AI generation)
        sheets_triggers = ['create spreadsheet', 'create sheet', 'read sheet', 'get data from sheet']
        if any(trigger in user_lower for trigger in sheets_triggers):
            return self.handle_sheets_command(user_input, project, use_voices, random_toggle)
        
        return {}, False

    def get_search_console_data_for_site(self, site_key: str, start_date: str = None, end_date: str = None) -> Dict:
        """Get search console data for a specific site with ENHANCED VALIDATION - FIXED VERSION"""
        
        # CRITICAL FIX: Validate site_key parameter
        if not site_key or site_key not in self.sites_config:
            available_sites = [(key, config['name']) for key, config in self.sites_config.items()] if self.sites_config else []
            return {
                'success': False,
                'error': f'Site "{site_key}" not found in configuration.\n\nAvailable sites: {available_sites}\n\nTip: Try using the exact site name like "Damn It Carl" or an alias like "carl".'
            }
        
        site_config = self.sites_config[site_key]
        site_url = site_config.get('search_console_url')
        
        if not site_url:
            return {
                'success': False,
                'error': f'No Search Console URL configured for site "{site_config["name"]}". Add search_console_url to your GOOGLE_SITES_CONFIG.'
            }
        
        # Get the raw search console data
        try:
            result = self.get_search_console_data(site_url, start_date, end_date)
            
            # Validate the result is a dictionary
            if not isinstance(result, dict):
                return {
                    'success': False,
                    'error': f'get_search_console_data returned {type(result)} instead of dict'
                }
            
            # Add site information to successful results
            if result.get('success'):
                result['site_name'] = site_config['name']
                result['site_key'] = site_key
            
            return result
            
        except Exception as e:
            print(f"get_search_console_data_for_site exception: {e}")
            return {
                'success': False,
                'error': f'Exception in search console data retrieval: {str(e)}'
            }
# =============================================================================
# SECTION 7: GOOGLE DOCS AND SHEETS INTEGRATION
# =============================================================================

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
                    
                    # Check if it looks like a spreadsheet ID
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
        
# =============================================================================
# SECTION 8: GA4 ANALYTICS AND SEARCH CONSOLE INTEGRATION WITH DATA VALIDATION
# =============================================================================
# =============================================================================
# SECTION 8: GA4 ANALYTICS AND SEARCH CONSOLE INTEGRATION WITH DATA VALIDATION - COMPLETE METHODS 9/12/25
# =============================================================================

    def _validate_analytics_response(self, response_data, operation, property_id):
        """Strict validation to prevent fabricated analytics data"""
        if not response_data:
            return False, f"AUTHENTICATION FAILED: {operation} - No data returned from Google Analytics for property {property_id}. Check your GA4 property ID and permissions."
        
        if 'error' in response_data:
            return False, f"API ERROR: {operation} failed - {response_data['error']}"
        
        # Validate GA4 response structure
        if not isinstance(response_data, dict):
            return False, f"INVALID DATA: {operation} returned malformed response (not a dictionary)"
        
        # Check for GA4 specific structure
        if 'rows' not in response_data and 'rowCount' not in response_data:
            return False, f"NO REAL DATA: GA4 {operation} returned empty or invalid response structure"
        
        # If we have rowCount, check it's a real number
        if 'rowCount' in response_data:
            try:
                row_count = int(response_data.get('rowCount', 0))
                if row_count == 0:
                    return True, f"VALID EMPTY: No data found for property {property_id} in specified date range"
            except (ValueError, TypeError):
                return False, f"INVALID DATA: rowCount is not a valid number"
        
        return True, None
    
    def get_ga4_analytics_report(self, property_id: str, start_date: str = "7daysAgo", end_date: str = "today") -> Dict:
        """Get GA4 Analytics report using the Data API with strict validation"""
        if 'analyticsdata' not in self.services:
            return {
                'success': False,
                'error': 'GA4 Analytics Data API not available. Check your Google Cloud Console - Analytics Data API v1 must be enabled.'
            }
        
        try:
            print(f"📊 GA4 Analytics API: Fetching data for property {property_id}")
            
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
            
            print(f"📊 GA4 Raw Response: {response}")
            
            # CRITICAL: Validate response before processing
            is_valid, validation_error = self._validate_analytics_response(response, "GA4 Analytics", property_id)
            if not is_valid:
                return {'success': False, 'error': validation_error}
            
            # Process GA4 response format - ONLY if validation passed
            analytics_data = []
            rows = response.get('rows', [])
            
            # If no rows, this is legitimate (no traffic)
            if len(rows) == 0:
                return {
                    'success': True,
                    'data': [],
                    'total_sessions': 0,
                    'total_users': 0,
                    'total_pageviews': 0,
                    'avg_bounce_rate': 0,
                    'date_range': f"{start_date} to {end_date}",
                    'property_id': property_id,
                    'message': f"No analytics data found for property {property_id} in the specified date range. This could be normal for new sites or quiet periods."
                }
            
            total_sessions = 0
            total_users = 0
            total_pageviews = 0
            bounce_rates = []
            
            for row in rows:
                # GA4 uses different response structure
                dimension_values = row.get('dimensionValues', [])
                metric_values = row.get('metricValues', [])
                
                if dimension_values and metric_values:
                    date = dimension_values[0].get('value', 'Unknown')
                    
                    # Extract metrics safely
                    sessions = int(metric_values[0].get('value', '0') or '0')
                    users = int(metric_values[1].get('value', '0') or '0')
                    pageviews = int(metric_values[2].get('value', '0') or '0')
                    bounce_rate = float(metric_values[3].get('value', '0') or '0')
                    
                    analytics_data.append({
                        'date': date,
                        'sessions': sessions,
                        'users': users,
                        'pageviews': pageviews,
                        'bounce_rate': bounce_rate
                    })
                    
                    # Accumulate totals
                    total_sessions += sessions
                    total_users += users
                    total_pageviews += pageviews
                    if bounce_rate > 0:
                        bounce_rates.append(bounce_rate)
            
            avg_bounce_rate = sum(bounce_rates) / len(bounce_rates) if bounce_rates else 0
            
            print(f"✅ GA4 Analytics: Processed {len(analytics_data)} days of REAL data")
            print(f"   📈 Totals: {total_sessions} sessions, {total_users} users, {total_pageviews} pageviews")
            
            return {
                'success': True,
                'data': analytics_data,
                'total_sessions': total_sessions,
                'total_users': total_users,
                'total_pageviews': total_pageviews,
                'avg_bounce_rate': round(avg_bounce_rate, 2),
                'date_range': f"{start_date} to {end_date}",
                'property_id': property_id,
                'raw_response_rows': len(rows)  # For debugging
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ GA4 Analytics API Error: {error_msg}")
            
            # Provide specific guidance based on common errors
            if "403" in error_msg:
                detailed_error = f"Permission denied for GA4 property {property_id}. Check: 1) Property ID is correct, 2) You have Analytics Admin/Viewer access, 3) Analytics Data API v1 is enabled in Google Cloud Console."
            elif "404" in error_msg:
                detailed_error = f"GA4 property {property_id} not found. Verify the property ID in your GA4 dashboard (Admin > Property Settings)."
            elif "invalid_grant" in error_msg:
                detailed_error = "OAuth token expired or invalid. Please re-authenticate via /google/auth/start"
            else:
                detailed_error = f"GA4 API error: {error_msg}"
            
            return {'success': False, 'error': detailed_error}
    
    def get_analytics_data(self, site_key: str, start_date: str = "7daysAgo", end_date: str = "today") -> Dict:
        """Get analytics for a specific site with ENHANCED VALIDATION to prevent fabricated data"""
        if site_key not in self.sites_config:
            return {'success': False, 'error': f'Site "{site_key}" not found in configuration'}
        
        site_config = self.sites_config[site_key]
        property_id = site_config.get('analytics_view_id')
        
        if not property_id:
            return {
                'success': False,
                'error': f'No GA4 property ID configured for site "{site_config["name"]}". Add analytics_view_id to your GOOGLE_SITES_CONFIG.'
            }
        
        # Get the raw analytics data
        result = self.get_ga4_analytics_report(property_id, start_date, end_date)
        
        # ENHANCED: Add comprehensive validation to prevent "Christianity B2B" disasters
        if result.get('success'):
            try:
                # Validate the data before returning
                validation_results = validate_analytics_data_comprehensive(
                    site_config,
                    analytics_result=result
                )
                
                analytics_validation = validation_results.get('analytics')
                if analytics_validation:
                    # Add validation metadata to the result
                    result['validation'] = {
                        'is_valid': analytics_validation.is_valid,
                        'confidence_score': analytics_validation.confidence_score,
                        'warnings': analytics_validation.warnings,
                        'recommendation': analytics_validation.recommendation
                    }
                    
                    # Log validation warnings
                    if analytics_validation.warnings:
                        print(f"⚠️  Analytics validation warnings for {site_config['name']}:")
                        for warning in analytics_validation.warnings:
                            print(f"   - {warning}")
                    
                    # Add validation info to result for user display
                    if analytics_validation.confidence_score < 0.8:
                        result['quality_warning'] = f"Data quality score: {analytics_validation.confidence_score:.1f}/1.0"
                        
            except Exception as e:
                print(f"Analytics validation failed: {e}")
                # Don't fail the entire request if validation fails
        
        # Add site information to successful results
        if result['success']:
            result['site_name'] = site_config['name']
            result['site_key'] = site_key
        
        return result

    def _validate_search_console_response(self, response_data, operation, site_url):
        """Strict validation to prevent fabricated search console data"""
        if not response_data:
            return False, f"AUTHENTICATION FAILED: {operation} - No data returned from Search Console for {site_url}. Check site verification and permissions."
        
        if 'error' in response_data:
            return False, f"API ERROR: {operation} failed - {response_data['error']}"
        
        # Validate Search Console response structure
        if not isinstance(response_data, dict):
            return False, f"INVALID DATA: {operation} returned malformed response"
        
        # Search Console can legitimately return no rows for new sites or quiet periods
        rows = response_data.get('rows', [])
        if len(rows) == 0:
            return True, f"VALID EMPTY: No search data found for {site_url}. This is normal for new sites or sites with no search traffic."
        
        # Validate row structure if we have data
        for i, row in enumerate(rows[:3]):  # Check first 3 rows
            if not isinstance(row, dict) or 'keys' not in row:
                return False, f"INVALID DATA: Row {i} missing required 'keys' field"
        
        return True, None

    def get_search_console_data(self, site_url: str, start_date: str = None, end_date: str = None) -> Dict:
        """Get Search Console performance data with strict validation"""
        if 'searchconsole' not in self.services:
            return {
                'success': False,
                'error': 'Search Console API not available. Check your Google Cloud Console - Search Console API must be enabled.'
            }
        
        try:
            if not start_date:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
            print(f"🔍 Search Console API: Fetching data for {site_url}")
            
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
            
            print(f"📊 Search Console Raw Response: {response}")
            
            # CRITICAL: Validate response before processing
            is_valid, validation_error = self._validate_search_console_response(response, "Search Console", site_url)
            if not is_valid:
                return {'success': False, 'error': validation_error}
            
            # Process response - ONLY if validation passed
            rows = response.get('rows', [])
            search_data = []
            
            # If no rows, this is legitimate (no search traffic yet)
            if len(rows) == 0:
                return {
                    'success': True,
                    'data': [],
                    'total_clicks': 0,
                    'total_impressions': 0,
                    'total_queries': 0,
                    'average_ctr': 0,
                    'average_position': 0,
                    'date_range': f"{start_date} to {end_date}",
                    'site_url': site_url,
                    'message': f"No search console data found for {site_url}. This is normal for new sites or sites with no search visibility yet."
                }
            
            total_clicks = 0
            total_impressions = 0
            total_queries = len(rows)
            ctr_values = []
            position_values = []
            
            for row in rows:
                query = row['keys'][0]
                page = row['keys'][1] if len(row['keys']) > 1 else ''
                clicks = row.get('clicks', 0)
                impressions = row.get('impressions', 0)
                ctr = row.get('ctr', 0.0)
                position = row.get('position', 0.0)
                
                search_data.append({
                    'query': query,
                    'page': page,
                    'clicks': clicks,
                    'impressions': impressions,
                    'ctr': round(ctr * 100, 2),  # Convert to percentage
                    'position': round(position, 1)
                })
                
                # Accumulate totals
                total_clicks += clicks
                total_impressions += impressions
                if ctr > 0:
                    ctr_values.append(ctr)
                if position > 0:
                    position_values.append(position)
            
            avg_ctr = (sum(ctr_values) / len(ctr_values) * 100) if ctr_values else 0
            avg_position = sum(position_values) / len(position_values) if position_values else 0
            
            print(f"✅ Search Console: Processed {len(search_data)} queries of REAL data")
            print(f"   📈 Totals: {total_clicks} clicks, {total_impressions} impressions")
            
            return {
                'success': True,
                'data': search_data,
                'total_clicks': total_clicks,
                'total_impressions': total_impressions,
                'total_queries': total_queries,
                'average_ctr': round(avg_ctr, 2),
                'average_position': round(avg_position, 1),
                'date_range': f"{start_date} to {end_date}",
                'site_url': site_url,
                'raw_response_rows': len(rows)  # For debugging
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Search Console API Error: {error_msg}")
            
            # Provide specific guidance based on common errors
            if "403" in error_msg:
                detailed_error = f"Permission denied for {site_url}. Check: 1) Site is verified in Search Console, 2) You have Owner/Full User access, 3) Search Console API is enabled in Google Cloud Console."
            elif "404" in error_msg:
                detailed_error = f"Site {site_url} not found in Search Console. Add and verify the site first."
            elif "invalid_grant" in error_msg:
                detailed_error = "OAuth token expired or invalid. Please re-authenticate via /google/auth/start"
            else:
                detailed_error = f"Search Console API error: {error_msg}"
            
            return {'success': False, 'error': detailed_error}

    def get_all_sites_analytics(self, start_date: str = "7daysAgo", end_date: str = "today") -> Dict:
        """Get analytics for all configured sites with data validation"""
        results = {}
        successful_sites = 0
        failed_sites = 0
        
        for site_key, site_config in self.sites_config.items():
            if site_config.get('analytics_view_id'):
                print(f"📊 Fetching analytics for {site_config['name']}...")
                result = self.get_analytics_data(site_key, start_date, end_date)
                results[site_key] = result
                
                if result['success']:
                    successful_sites += 1
                    print(f"   ✅ Success: {result.get('total_sessions', 0)} sessions")
                else:
                    failed_sites += 1
                    print(f"   ❌ Failed: {result['error']}")
            else:
                results[site_key] = {
                    'success': False,
                    'error': f'No analytics property ID configured for {site_config["name"]}'
                }
                failed_sites += 1
        
        return {
            'success': True,
            'sites': results,
            'total_sites': len(results),
            'successful_sites': successful_sites,
            'failed_sites': failed_sites,
            'summary': f"Analytics retrieved for {successful_sites}/{len(results)} configured sites"
        }

    def handle_blog_suggestions_command(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Generate blog suggestions with CRITICAL validation to prevent inappropriate content"""
        user_lower = user_input.lower().strip()
        
        # Site identification
        site_key = None
        if 'for ' in user_lower:
            site_name = user_lower.split('for ')[-1].strip()
            site_key = self.find_site_by_name(site_name)
        
        if not site_key and self.sites_config:
            site_key = list(self.sites_config.keys())[0]
        
        if not site_key:
            return {"SyntaxPrime": "No sites configured for blog suggestions."}, True
        
        site_config = self.sites_config[site_key]
        
        # Gather data from multiple sources WITH VALIDATION
        analytics_result = None
        sc_result = None
        
        # Get Search Console data with proper date format
        if site_config.get('search_console_url'):
            # Calculate proper date format for Search Console
            end_date = datetime.date.today().strftime('%Y-%m-%d')
            start_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            sc_result = self.get_search_console_data_for_site(site_key, start_date, end_date)
        
        # Get Analytics data
        if site_config.get('analytics_view_id'):
            analytics_result = self.get_analytics_data(site_key, '30daysAgo', 'today')
        
        # CRITICAL: Validate all data before using for AI suggestions
        validation_results = validate_analytics_data_comprehensive(site_config, analytics_result, sc_result)
        
        # Check if we should block AI suggestions due to data quality issues
        should_block, block_reason, overall_confidence = should_block_ai_suggestions_enhanced(validation_results)
        
        if should_block:
            error_response = f"🚨 **DATA VALIDATION FAILED - BLOG SUGGESTIONS BLOCKED** 🚨\n\n"
            error_response += f"**Cannot generate blog suggestions due to data quality issues:**\n"
            error_response += f"• {block_reason}\n\n"
            
            error_response += f"**Data Quality Report for {site_config['name']}:**\n"
            for data_type, result in validation_results.items():
                status = '✅ Valid' if result.is_valid else '❌ Invalid'
                error_response += f"• {data_type.title()}: {status} "
                error_response += f"(Confidence: {result.confidence_score:.1f}/1.0, Relevance: {result.site_relevance_score:.1f}/1.0)\n"
                
                if result.validation_errors:
                    for error in result.validation_errors[:2]:  # Limit to top 2 errors
                        error_response += f"  - ❌ {error}\n"
                        
                if result.warnings:
                    for warning in result.warnings[:2]:  # Limit to top 2 warnings
                        error_response += f"  - ⚠️ {warning}\n"
            
            error_response += f"\n**This prevents disasters like suggesting religious content for non-religious sites!**\n\n"
            error_response += f"**Manual Verification Required:**\n"
            error_response += f"1. Check {site_config['name']} in Google Analytics dashboard\n"
            error_response += f"2. Verify search queries in Search Console match expected content\n"
            error_response += f"3. Ensure Analytics View ID and Search Console URL are correct\n"
            error_response += f"4. Re-run command once data issues are resolved"
            
            return {"SyntaxPrime": error_response}, True
        
        # Data is validated - proceed with generating suggestions
        data_summary = f"Blog suggestions for {site_config['name']} (DATA VERIFIED ✅):\n\n"
        data_summary += f"**Data Quality Score: {overall_confidence:.1f}/1.0** ✅\n\n"
        
        # Process Search Console data (already validated)
        if sc_result and sc_result['success'] and sc_result['data']:
            # Find queries with high impressions but low CTR (opportunity keywords)
            opportunities = []
            for row in sc_result['data']:
                if row['impressions'] > 100 and row['ctr'] < 5:  # High impressions, low CTR
                    opportunities.append(row)
            
            opportunities.sort(key=lambda x: x['impressions'], reverse=True)
            
            if opportunities:
                data_summary += "**SEO Opportunities (High Impressions, Low CTR):**\n"
                for row in opportunities[:5]:
                    data_summary += f"• \"{row['query']}\" - {row['impressions']} impressions, {row['ctr']:.1f}% CTR\n"
                data_summary += "\n"
        
        # Generate AI suggestions using the VALIDATED data
        from utils.ghostline_engine import generate_response, CHAT_MODEL
        from utils.rag_basic import enhanced_retrieve, is_ready
        
        prompt = f"""Based on the following VALIDATED website performance data, suggest 5-7 specific blog post ideas for {site_config['name']}.

Website Focus: {site_config.get('expected_keywords', [])}
{data_summary}

IMPORTANT CONTEXT:
- This data has been validated for accuracy (confidence: {overall_confidence:.1f}/1.0)
- Only suggest content relevant to this specific website's focus
- DO NOT suggest content from other topics/niches

Focus on:
1. Content that addresses high-impression, low-CTR search queries
2. Topics that could capture more traffic for trending searches  
3. Content that builds on successful existing pages
4. Seasonal or timely content opportunities

Provide specific, actionable blog post titles with brief explanations."""
        
        # Generate AI response
        retrieval_ctx = enhanced_retrieve(prompt, k=3, project=project) if is_ready() else []
        
        response_data = generate_response(
            prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        
        return response_data, True

# =============================================================================
# SECTION 9: MULTI-SITE ANALYTICS COMMAND HANDLERS
# =============================================================================
# =============================================================================
# SECTION 9: MULTI-SITE ANALYTICS COMMAND HANDLERS 9/11/25
# =============================================================================

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

    def get_all_sites_analytics(self, start_date: str = "7daysAgo", end_date: str = "today") -> Dict:
        """Get analytics for all configured sites with data validation"""
        results = {}
        successful_sites = 0
        failed_sites = 0
        
        for site_key, site_config in self.sites_config.items():
            if site_config.get('analytics_view_id'):
                print(f"📊 Fetching analytics for {site_config['name']}...")
                result = self.get_analytics_data(site_key, start_date, end_date)
                results[site_key] = result
                
                if result['success']:
                    successful_sites += 1
                    print(f"   ✅ Success: {result.get('total_sessions', 0)} sessions")
                else:
                    failed_sites += 1
                    print(f"   ❌ Failed: {result['error']}")
            else:
                results[site_key] = {
                    'success': False,
                    'error': f'No analytics property ID configured for {site_config["name"]}'
                }
                failed_sites += 1
        
        return {
            'success': True,
            'sites': results,
            'total_sites': len(results),
            'successful_sites': successful_sites,
            'failed_sites': failed_sites,
            'summary': f"Analytics retrieved for {successful_sites}/{len(results)} configured sites"
        }

    def handle_blog_suggestions_command(self, user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
        """Generate blog suggestions with CRITICAL validation to prevent inappropriate content"""
        user_lower = user_input.lower().strip()
        
        # Site identification
        site_key = None
        if 'for ' in user_lower:
            site_name = user_lower.split('for ')[-1].strip()
            site_key = self.find_site_by_name(site_name)
        
        if not site_key and self.sites_config:
            site_key = list(self.sites_config.keys())[0]
        
        if not site_key:
            return {"SyntaxPrime": "No sites configured for blog suggestions."}, True
        
        site_config = self.sites_config[site_key]
        
        # Gather data from multiple sources WITH VALIDATION
        analytics_result = None
        sc_result = None
        
        # Get Search Console data with proper date format
        if site_config.get('search_console_url'):
            # Calculate proper date format for Search Console
            end_date = datetime.date.today().strftime('%Y-%m-%d')
            start_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
            sc_result = self.get_search_console_data_for_site(site_key, start_date, end_date)
        
        # Get Analytics data
        if site_config.get('analytics_view_id'):
            analytics_result = self.get_analytics_data(site_key, '30daysAgo', 'today')
        
        # CRITICAL: Validate all data before using for AI suggestions
        validation_results = validate_analytics_data_comprehensive(site_config, analytics_result, sc_result)
        
        # Check if we should block AI suggestions due to data quality issues
        should_block, block_reason, overall_confidence = should_block_ai_suggestions_enhanced(validation_results)
        
        if should_block:
            error_response = f"🚨 **DATA VALIDATION FAILED - BLOG SUGGESTIONS BLOCKED** 🚨\n\n"
            error_response += f"**Cannot generate blog suggestions due to data quality issues:**\n"
            error_response += f"• {block_reason}\n\n"
            
            error_response += f"**Data Quality Report for {site_config['name']}:**\n"
            for data_type, result in validation_results.items():
                status = '✅ Valid' if result.is_valid else '❌ Invalid'
                error_response += f"• {data_type.title()}: {status} "
                error_response += f"(Confidence: {result.confidence_score:.1f}/1.0, Relevance: {result.site_relevance_score:.1f}/1.0)\n"
                
                if result.validation_errors:
                    for error in result.validation_errors[:2]:  # Limit to top 2 errors
                        error_response += f"  - ❌ {error}\n"
                        
                if result.warnings:
                    for warning in result.warnings[:2]:  # Limit to top 2 warnings
                        error_response += f"  - ⚠️ {warning}\n"
            
            error_response += f"\n**This prevents disasters like suggesting religious content for non-religious sites!**\n\n"
            error_response += f"**Manual Verification Required:**\n"
            error_response += f"1. Check {site_config['name']} in Google Analytics dashboard\n"
            error_response += f"2. Verify search queries in Search Console match expected content\n"
            error_response += f"3. Ensure Analytics View ID and Search Console URL are correct\n"
            error_response += f"4. Re-run command once data issues are resolved"
            
            return {"SyntaxPrime": error_response}, True
        
        # Data is validated - proceed with generating suggestions
        data_summary = f"Blog suggestions for {site_config['name']} (DATA VERIFIED ✅):\n\n"
        data_summary += f"**Data Quality Score: {overall_confidence:.1f}/1.0** ✅\n\n"
        
        # Process Search Console data (already validated)
        if sc_result and sc_result['success'] and sc_result['data']:
            # Find queries with high impressions but low CTR (opportunity keywords)
            opportunities = []
            for row in sc_result['data']:
                if row['impressions'] > 100 and row['ctr'] < 5:  # High impressions, low CTR
                    opportunities.append(row)
            
            opportunities.sort(key=lambda x: x['impressions'], reverse=True)
            
            if opportunities:
                data_summary += "**SEO Opportunities (High Impressions, Low CTR):**\n"
                for row in opportunities[:5]:
                    data_summary += f"• \"{row['query']}\" - {row['impressions']} impressions, {row['ctr']:.1f}% CTR\n"
                data_summary += "\n"
        
        # Generate AI suggestions using the VALIDATED data
        from utils.ghostline_engine import generate_response, CHAT_MODEL
        from utils.rag_basic import enhanced_retrieve, is_ready
        
        prompt = f"""Based on the following VALIDATED website performance data, suggest 5-7 specific blog post ideas for {site_config['name']}.

Website Focus: {site_config.get('expected_keywords', [])}
{data_summary}

IMPORTANT CONTEXT:
- This data has been validated for accuracy (confidence: {overall_confidence:.1f}/1.0)
- Only suggest content relevant to this specific website's focus
- DO NOT suggest content from other topics/niches

Focus on:
1. Content that addresses high-impression, low-CTR search queries
2. Topics that could capture more traffic for trending searches  
3. Content that builds on successful existing pages
4. Seasonal or timely content opportunities

Provide specific, actionable blog post titles with brief explanations."""
        
        # Generate AI response
        retrieval_ctx = enhanced_retrieve(prompt, k=3, project=project) if is_ready() else []
        
        response_data = generate_response(
            prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        
        return response_data, True
    
        
# =============================================================================
# SECTION 10: FINAL ENTRY POINT AND INTEGRATION
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
