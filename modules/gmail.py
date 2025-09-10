# modules/gmail.py - Complete version with FIXED data integrity and fabrication prevention
# This module handles all Gmail and Calendar integration commands with strict validation

import os
import datetime
import json
import re
from utils.gmail_client import (
    list_overnight, search as gmail_search,
    list_today_events, list_tomorrow_events, search_calendar,
    get_next_meeting, format_calendar_summary
)
from utils.ghostline_engine import generate_response
from utils.rag_basic import is_ready
from modules.database import save_conversation_enhanced, save_daily_log_enhanced
from modules.brain import enhanced_retrieve

CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

# =============================================================================
# CORE DATA VALIDATION FUNCTIONS - PREVENT FABRICATION
# =============================================================================

def _handle_integration_error(error_msg, operation):
    """Standard error handler that prevents data fabrication"""
    return {
        "SyntaxPrime": f"🔧 Gmail/Calendar Integration Error\n\n"
                      f"**Operation:** {operation}\n"
                      f"**Issue:** {error_msg}\n\n"
                      f"**Next Steps:**\n"
                      f"• Check your Google OAuth setup at `/integrations`\n"
                      f"• Try re-authenticating via `/google/auth/start`\n"
                      f"• Ensure Gmail API is enabled in Google Cloud Console\n\n"
                      f"**Note:** No fake data will be generated - you'll only see real email/calendar content."
    }

def _debug_email_structure(msgs, operation):
    """DEBUG: Log the actual structure of email objects to understand format"""
    print(f"\n=== DEBUG EMAIL STRUCTURE for {operation} ===")
    print(f"Total emails received: {len(msgs) if msgs else 0}")
    
    if msgs and len(msgs) > 0:
        print(f"Type of first email: {type(msgs[0])}")
        print(f"First email keys: {list(msgs[0].keys()) if isinstance(msgs[0], dict) else 'Not a dict'}")
        
        # Log first email structure completely
        if isinstance(msgs[0], dict):
            print("First email content:")
            for key, value in msgs[0].items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}... (truncated)")
                else:
                    print(f"  {key}: {value}")
        
        # Check what fields are actually available across all emails
        all_keys = set()
        for msg in msgs[:5]:  # Check first 5 emails
            if isinstance(msg, dict):
                all_keys.update(msg.keys())
        
        print(f"All available keys across emails: {sorted(all_keys)}")
    
    print("=== END DEBUG EMAIL STRUCTURE ===\n")

def _extract_email_info(msg):
    """Extract sender and subject from email object, handling different possible formats"""
    if not isinstance(msg, dict):
        return None, None
    
    # Try different possible field names for sender
    sender = None
    for sender_field in ['sender', 'from', 'From', 'fromEmail', 'senderEmail', 'author']:
        if sender_field in msg:
            sender = msg[sender_field]
            break
    
    # If sender is still None, try nested structures (Gmail API format)
    if not sender and 'headers' in msg:
        headers = msg['headers']
        if isinstance(headers, list):
            for header in headers:
                if isinstance(header, dict) and header.get('name', '').lower() == 'from':
                    sender = header.get('value')
                    break
    
    # Try different possible field names for subject
    subject = None
    for subject_field in ['subject', 'Subject', 'title', 'summary', 'snippet']:
        if subject_field in msg:
            subject = msg[subject_field]
            break
    
    # If subject is still None, try nested structures (Gmail API format)
    if not subject and 'headers' in msg:
        headers = msg['headers']
        if isinstance(headers, list):
            for header in headers:
                if isinstance(header, dict) and header.get('name', '').lower() == 'subject':
                    subject = header.get('value')
                    break
    
    # Clean up sender (remove email brackets if present)
    if sender and '<' in sender and '>' in sender:
        # Extract just the name part before the email
        sender_name = sender.split('<')[0].strip()
        if sender_name:  # If we have a name, use it
            sender = sender_name
        else:  # If no name, use the email address
            email_match = re.search(r'<([^>]+)>', sender)
            if email_match:
                sender = email_match.group(1)
    
    return sender, subject

def _validate_gmail_response(msgs, operation):
    """Strict validation to prevent fabricated Gmail content"""
    if msgs is None:
        return False, f"AUTHENTICATION FAILED: {operation} returned no data. Check your Gmail API authentication and permissions."
    
    if isinstance(msgs, dict) and "error" in msgs:
        return False, f"API ERROR: {operation} failed - {msgs['error']}"
    
    if not isinstance(msgs, list):
        return False, f"INVALID DATA FORMAT: {operation} returned data that isn't a list of emails (got {type(msgs)})"
    
    # Additional validation: check if we have at least valid dict structures
    if len(msgs) > 0:
        for i, msg in enumerate(msgs[:3]):  # Check first 3 messages
            if not isinstance(msg, dict):
                return False, f"INVALID EMAIL FORMAT: Message {i+1} is not a dictionary (got {type(msg)})"
            # Gmail API messages should have at least an 'id' field
            if 'id' not in msg and 'headers' not in msg and 'sender' not in msg:
                return False, f"SUSPICIOUS EMAIL FORMAT: Message {i+1} missing expected fields (no id, headers, or sender)"
    
    return True, None

def _validate_calendar_response(events, operation):
    """Strict validation to prevent fabricated Calendar content"""
    if events is None:
        return False, f"AUTHENTICATION FAILED: {operation} returned no data. Check your Calendar API authentication and permissions."
    
    if isinstance(events, dict) and "error" in events:
        return False, f"API ERROR: {operation} failed - {events['error']}"
    
    if not isinstance(events, list):
        return False, f"INVALID DATA FORMAT: {operation} returned data that isn't a list of events (got {type(events)})"
    
    # Additional validation: check if we have valid event structures
    if len(events) > 0:
        for i, event in enumerate(events[:3]):  # Check first 3 events
            if not isinstance(event, dict):
                return False, f"INVALID EVENT FORMAT: Event {i+1} is not a dictionary (got {type(event)})"
    
    return True, None

def _count_real_emails(msgs):
    """Count how many emails have extractable information"""
    if not msgs:
        return 0
    
    real_count = 0
    for msg in msgs:
        sender, subject = _extract_email_info(msg)
        if sender or subject:  # Count as real if we can extract either field
            real_count += 1
    
    return real_count

def _create_anti_fabrication_prompt(operation, real_count, total_count):
    """Create prompts that explicitly prevent AI fabrication"""
    if real_count == 0:
        return f"""SYSTEM STATUS: No {operation} found with readable data. 

CRITICAL INSTRUCTION: Do not fabricate, invent, or imagine any emails, people, companies, or content. 
Respond only with factual information about the empty state. 
Do not mention any specific people, companies, or email content as none were actually found.

Provide general advice about email management or productivity instead."""
    
    return f"""SYSTEM STATUS: Found {real_count} readable {operation} out of {total_count} total.

CRITICAL INSTRUCTION: Only reference the actual email data provided below. 
Do not add, invent, or fabricate any additional emails, people, companies, or content.
If specific emails are mentioned, they must be from the actual data provided."""

# =============================================================================
# MAIN COMMAND HANDLERS
# =============================================================================

def handle_overnight_command(project, use_voices, random_toggle):
    """Handle overnight email check commands with STRICT data validation"""
    try:
        print("🌙 Starting overnight email check with enhanced validation...")
        msgs = list_overnight(include_unread=True, include_primary=False)
        
        # STRICT VALIDATION: Prevent any fabricated content
        is_valid, error_msg = _validate_gmail_response(msgs, "overnight email check")
        if not is_valid:
            print(f"❌ Overnight email validation failed: {error_msg}")
            return _handle_integration_error(error_msg, "overnight email check")
        
        # DEBUG: Log the actual email structure for debugging
        _debug_email_structure(msgs, "overnight emails")
        
        # Count how many emails we can actually extract data from
        real_email_count = _count_real_emails(msgs)
        total_emails = len(msgs) if msgs else 0
        
        print(f"📊 Email Analysis: {real_email_count} readable emails out of {total_emails} total")
        
        if real_email_count == 0:
            # ANTI-FABRICATION: Explicit empty state handling
            summary_prompt = _create_anti_fabrication_prompt("overnight emails", 0, total_emails)
        else:
            # REAL DATA PROCESSING: Only process emails we can extract data from
            real_emails = []
            for msg in msgs:
                sender, subject = _extract_email_info(msg)
                if sender or subject:  # Only include emails with extractable data
                    real_emails.append({
                        'sender': sender or 'Unknown Sender',
                        'subject': subject or 'No Subject',
                        'timestamp': msg.get('date', 'Unknown Time')
                    })
            
            print(f"✅ Extracted data from {len(real_emails)} emails")
            
            # Create prompt with REAL email data
            summary_prompt = _create_anti_fabrication_prompt("overnight emails", real_email_count, total_emails)
            summary_prompt += f"\n\nACTUAL EMAIL DATA:\n"
            
            for i, email in enumerate(real_emails[:10], 1):  # Limit to 10 emails for prompt size
                summary_prompt += f"{i}. From: {email['sender']} | Subject: {email['subject']}\n"
            
            if len(real_emails) > 10:
                summary_prompt += f"... and {len(real_emails) - 10} more emails\n"
        
        # Generate response with context
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5, project=project) if is_ready() else []
        
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        
        # Log the operation for debugging
        print(f"📝 Overnight command completed: {real_email_count} real emails processed")
        
        return response_data
        
    except Exception as e:
        error_msg = f"Overnight email processing failed: {str(e)}"
        print(f"❌ {error_msg}")
        return _handle_integration_error(error_msg, "overnight email check")

def handle_gmail_search_command(search_query, project, use_voices, random_toggle):
    """Handle Gmail search commands with STRICT data validation"""
    try:
        print(f"🔍 Starting Gmail search for: '{search_query}'")
        msgs = gmail_search(search_query, max_results=25)
        
        # STRICT VALIDATION
        is_valid, error_msg = _validate_gmail_response(msgs, f"Gmail search '{search_query}'")
        if not is_valid:
            print(f"❌ Gmail search validation failed: {error_msg}")
            return _handle_integration_error(error_msg, f"Gmail search '{search_query}'")
        
        # DEBUG: Log search results structure
        _debug_email_structure(msgs, f"Gmail search '{search_query}'")
        
        # Count readable emails
        real_email_count = _count_real_emails(msgs)
        total_emails = len(msgs) if msgs else 0
        
        print(f"📊 Search Results: {real_email_count} readable emails out of {total_emails} total")
        
        if real_email_count == 0:
            summary_prompt = f"""SYSTEM STATUS: No emails found for search query '{search_query}'.

CRITICAL INSTRUCTION: Do not fabricate or invent any email content. 
The search returned no readable results. Provide suggestions for:
1. Alternative search terms
2. Email organization tips
3. Search syntax help

Do not mention any specific people, companies, or email content."""
        else:
            # Process real search results
            search_results = []
            for msg in msgs:
                sender, subject = _extract_email_info(msg)
                if sender or subject:
                    search_results.append({
                        'sender': sender or 'Unknown Sender',
                        'subject': subject or 'No Subject',
                        'snippet': msg.get('snippet', 'No preview available')[:100]
                    })
            
            summary_prompt = f"""GMAIL SEARCH RESULTS for '{search_query}':

Found {real_email_count} readable emails out of {total_emails} total.

CRITICAL INSTRUCTION: Only reference the actual search results provided below.
Do not add, invent, or fabricate any additional emails.

ACTUAL SEARCH RESULTS:
"""
            
            for i, result in enumerate(search_results[:15], 1):  # Limit to 15 for prompt size
                summary_prompt += f"{i}. From: {result['sender']} | Subject: {result['subject']}\n"
                if result['snippet']:
                    summary_prompt += f"   Preview: {result['snippet']}...\n"
                summary_prompt += "\n"
            
            if len(search_results) > 15:
                summary_prompt += f"... and {len(search_results) - 15} more results\n"
        
        # Generate response with context
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5, project=project) if is_ready() else []
        
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        
        print(f"📝 Gmail search completed: {real_email_count} real results processed")
        
        return response_data
        
    except Exception as e:
        error_msg = f"Gmail search failed: {str(e)}"
        print(f"❌ {error_msg}")
        return _handle_integration_error(error_msg, f"Gmail search '{search_query}'")

def handle_calendar_command(command_type, project, use_voices, random_toggle):
    """Handle calendar commands with STRICT data validation"""
    try:
        print(f"📅 Starting calendar command: {command_type}")
        
        # Determine which calendar function to call
        if command_type in ['today', 'today events', 'today\'s events']:
            events = list_today_events()
            operation = "today's calendar events"
        elif command_type in ['tomorrow', 'tomorrow events', 'tomorrow\'s events']:
            events = list_tomorrow_events()
            operation = "tomorrow's calendar events"
        elif command_type in ['next meeting', 'next', 'upcoming']:
            next_meeting = get_next_meeting()
            if next_meeting:
                events = [next_meeting]  # Convert single meeting to list format
                operation = "next meeting"
            else:
                events = []
                operation = "next meeting"
        else:
            # Default to today's events
            events = list_today_events()
            operation = "calendar events"
        
        # STRICT VALIDATION
        is_valid, error_msg = _validate_calendar_response(events, operation)
        if not is_valid:
            print(f"❌ Calendar validation failed: {error_msg}")
            return _handle_integration_error(error_msg, operation)
        
        print(f"📊 Calendar Results: {len(events)} events found")
        
        if len(events) == 0:
            summary_prompt = f"""SYSTEM STATUS: No {operation} found.

CRITICAL INSTRUCTION: Do not fabricate or invent any calendar events, meetings, or appointments.
The calendar is genuinely empty for this time period.

Provide general productivity advice or suggest calendar management tips instead.
Do not mention any specific meetings, people, or events as none were found."""
        else:
            # Process real calendar events
            summary_prompt = f"""CALENDAR: {operation.upper()}

Found {len(events)} real events.

CRITICAL INSTRUCTION: Only reference the actual calendar events provided below.
Do not add, invent, or fabricate any additional meetings or events.

ACTUAL EVENTS:
"""
            
            for i, event in enumerate(events, 1):
                title = event.get('summary', 'No Title')
                start_time = event.get('start', {}).get('dateTime', 'Time not specified')
                location = event.get('location', 'No location')
                
                summary_prompt += f"{i}. {title}\n"
                summary_prompt += f"   Time: {start_time}\n"
                if location and location != 'No location':
                    summary_prompt += f"   Location: {location}\n"
                summary_prompt += "\n"
        
        # Generate response with context
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5, project=project) if is_ready() else []
        
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        
        print(f"📝 Calendar command completed: {len(events)} real events processed")
        
        return response_data
        
    except Exception as e:
        error_msg = f"Calendar processing failed: {str(e)}"
        print(f"❌ {error_msg}")
        return _handle_integration_error(error_msg, f"calendar {command_type}")

def handle_morning_briefing_command(project, use_voices, random_toggle):
    """Handle comprehensive morning briefing with STRICT data validation"""
    try:
        print("🌅 Starting comprehensive morning briefing...")
        
        # Gather all data with validation
        overnight_emails = list_overnight(include_unread=True, include_primary=False)
        today_events = list_today_events()
        next_meeting = get_next_meeting()
        
        # Validate each data source
        email_valid, email_error = _validate_gmail_response(overnight_emails, "overnight emails")
        calendar_valid, calendar_error = _validate_calendar_response(today_events, "today's events")
        
        # Count real data
        real_email_count = _count_real_emails(overnight_emails) if email_valid else 0
        real_event_count = len(today_events) if calendar_valid else 0
        
        print(f"📊 Morning Briefing Data: {real_email_count} emails, {real_event_count} events")
        
        # Create comprehensive anti-fabrication prompt
        summary_prompt = """COMPREHENSIVE MORNING BRIEFING

CRITICAL INSTRUCTION: This briefing contains only REAL data from your accounts.
Do not fabricate, invent, or imagine any additional emails, meetings, or information.
All data below is verified and extracted from actual sources.

"""
        
        # Add email section
        if not email_valid:
            summary_prompt += f"📧 EMAILS: API Error - {email_error}\n\n"
        elif real_email_count == 0:
            summary_prompt += "📧 EMAILS: No new overnight emails found. Your inbox is up to date.\n\n"
        else:
            summary_prompt += f"📧 EMAILS: {real_email_count} new overnight emails\n"
            real_emails = []
            for msg in overnight_emails:
                sender, subject = _extract_email_info(msg)
                if sender or subject:
                    real_emails.append({'sender': sender or 'Unknown', 'subject': subject or 'No Subject'})
            
            for i, email in enumerate(real_emails[:5], 1):  # Show first 5
                summary_prompt += f"  {i}. {email['sender']}: {email['subject']}\n"
            
            if len(real_emails) > 5:
                summary_prompt += f"  ... and {len(real_emails) - 5} more emails\n"
            summary_prompt += "\n"
        
        # Add calendar section
        if not calendar_valid:
            summary_prompt += f"📅 CALENDAR: API Error - {calendar_error}\n\n"
        elif real_event_count == 0:
            summary_prompt += "📅 CALENDAR: No events scheduled for today. You have a clear calendar.\n\n"
        else:
            summary_prompt += f"📅 CALENDAR: {real_event_count} events today\n"
            for i, event in enumerate(today_events[:5], 1):  # Show first 5
                title = event.get('summary', 'No Title')
                start_time = event.get('start', {}).get('dateTime', 'Time not specified')
                summary_prompt += f"  {i}. {title} at {start_time}\n"
            
            if len(today_events) > 5:
                summary_prompt += f"  ... and {len(today_events) - 5} more events\n"
            summary_prompt += "\n"
        
        # Add next meeting info
        if next_meeting:
            title = next_meeting.get('summary', 'No Title')
            start_time = next_meeting.get('start', {}).get('dateTime', 'Time not specified')
            summary_prompt += f"⏰ NEXT MEETING: {title} at {start_time}\n\n"
        else:
            summary_prompt += "⏰ NEXT MEETING: No upcoming meetings found.\n\n"
        
        summary_prompt += "END OF REAL DATA - Do not add any additional information not listed above."
        
        # Generate response with context
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5, project=project) if is_ready() else []
        
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        
        print(f"📝 Morning briefing completed: {real_email_count} emails, {real_event_count} events")
        
        return response_data
        
    except Exception as e:
        error_msg = f"Morning briefing failed: {str(e)}"
        print(f"❌ {error_msg}")
        return _handle_integration_error(error_msg, "morning briefing")

# =============================================================================
# MAIN COMMAND PROCESSOR
# =============================================================================

def process_gmail_command(user_input, project, use_voices, random_toggle):
    """
    Main processor for Gmail/Calendar commands with enhanced pattern matching
    Returns (response_data, handled) tuple
    """
    user_lower = user_input.lower().strip()
    
    print(f"🔍 Processing Gmail/Calendar command: '{user_input}'")
    
    # Morning briefing patterns (highest priority)
    morning_patterns = [
        'good morning', 'morning briefing', 'daily briefing', 'start my day',
        'morning', 'gm', 'briefing', 'what\'s up today', 'today\'s agenda'
    ]
    
    if any(pattern in user_lower for pattern in morning_patterns):
        print("🌅 Detected morning briefing command")
        response_data = handle_morning_briefing_command(project, use_voices, random_toggle)
        return response_data, True
    
    # Overnight email patterns
    overnight_patterns = [
        'overnight', 'overnight emails', 'emails overnight', 'new emails',
        'check mail', 'check emails', 'any new emails', 'email update'
    ]
    
    if any(pattern in user_lower for pattern in overnight_patterns):
        print("🌙 Detected overnight email command")
        response_data = handle_overnight_command(project, use_voices, random_toggle)
        return response_data, True
    
    # Calendar patterns
    calendar_patterns = {
        'today': ['today', 'today\'s events', 'what\'s today', 'schedule today'],
        'tomorrow': ['tomorrow', 'tomorrow\'s events', 'what\'s tomorrow', 'schedule tomorrow'],
        'next meeting': ['next meeting', 'next', 'upcoming', 'what\'s next']
    }
    
    for command_type, patterns in calendar_patterns.items():
        if any(pattern in user_lower for pattern in patterns):
            print(f"📅 Detected calendar command: {command_type}")
            response_data = handle_calendar_command(command_type, project, use_voices, random_toggle)
            return response_data, True
    
    # Gmail search patterns
    search_patterns = [
        'search email', 'find email', 'email about', 'emails from',
        'search gmail', 'find in gmail', 'gmail search'
    ]
    
    if any(pattern in user_lower for pattern in search_patterns):
        # Extract search query
        search_query = user_input
        for pattern in ['search email for', 'find email about', 'email about', 'emails from', 'search gmail for']:
            if pattern in user_lower:
                search_query = user_input.lower().replace(pattern, '').strip()
                break
        
        print(f"🔍 Detected Gmail search command: '{search_query}'")
        response_data = handle_gmail_search_command(search_query, project, use_voices, random_toggle)
        return response_data, True
    
    # No Gmail/Calendar command detected
    print("❌ No Gmail/Calendar pattern matched")
    return {}, False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_gmail_configured():
    """Check if Gmail integration is properly configured"""
    try:
        # Try to import the Gmail client functions
        from utils.gmail_client import list_overnight
        
        # Check if credentials file exists
        import os
        if os.path.exists('token.json'):
            return True
        elif os.path.exists(os.getenv('GOOGLE_TOKEN_PATH', 'token.json')):
            return True
        else:
            return False
    except ImportError:
        return False

def get_gmail_status():
    """Get detailed Gmail integration status for diagnostics"""
    status = {
        'configured': is_gmail_configured(),
        'credentials_file': os.path.exists('token.json'),
        'environment_vars': {
            'GOOGLE_TOKEN_PATH': os.getenv('GOOGLE_TOKEN_PATH'),
            'GOOGLE_CREDENTIALS_PATH': os.getenv('GOOGLE_CREDENTIALS_PATH')
        },
        'last_test': None,
        'test_results': {}
    }
    
    if status['configured']:
        try:
            # Try a simple API call to test connectivity
            msgs = list_overnight(include_unread=True, include_primary=False)
            status['last_test'] = datetime.datetime.now().isoformat()
            status['test_results'] = {
                'success': True,
                'message_count': len(msgs) if msgs else 0,
                'error': None
            }
        except Exception as e:
            status['last_test'] = datetime.datetime.now().isoformat()
            status['test_results'] = {
                'success': False,
                'message_count': 0,
                'error': str(e)
            }
    
    return status
