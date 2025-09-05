# modules/gmail.py - Complete version with all required functions

import os
import datetime
import json
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

def _handle_integration_error(error_msg, operation):
    """Standard error handler that prevents data fabrication"""
    return {
        "SyntaxPrime": f"Gmail/Calendar integration error: {operation} failed. {error_msg}\n\n"
                      "Please check your Google OAuth setup at /integrations or try re-authenticating."
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
    
    # If sender is still None, try nested structures
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
    
    # If subject is still None, try nested structures
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
        sender = sender.split('<')[0].strip()
        if not sender:  # If no name, use email
            sender = sender.split('<')[1].split('>')[0].strip()
    
    return sender, subject

def _validate_gmail_response(msgs, operation):
    """Validate Gmail response and prevent fabricated content"""
    if msgs is None:
        return False, f"{operation} returned no data - check authentication"
    
    if isinstance(msgs, dict) and "error" in msgs:
        return False, f"{operation} failed: {msgs['error']}"
    
    if not isinstance(msgs, list):
        return False, f"{operation} returned invalid data format"
    
    return True, None

def _validate_calendar_response(events, operation):
    """Validate Calendar response and prevent fabricated content"""
    if events is None:
        return False, f"{operation} returned no data - check authentication"
    
    if isinstance(events, dict) and "error" in events:
        return False, f"{operation} failed: {events['error']}"
    
    if not isinstance(events, list):
        return False, f"{operation} returned invalid data format"
    
    return True, None

def handle_overnight_command(project, use_voices, random_toggle):
    """Handle overnight email check commands with FIXED data extraction"""
    try:
        print("Starting overnight email check...")
        msgs = list_overnight(include_unread=True, include_primary=False)
        
        # Validate response
        is_valid, error_msg = _validate_gmail_response(msgs, "overnight email check")
        if not is_valid:
            print(f"Overnight email validation failed: {error_msg}")
            return _handle_integration_error(error_msg, "overnight email check")
        
        # DEBUG: Log the actual email structure
        _debug_email_structure(msgs, "overnight emails")
        
        # Check if we actually got real data
        if len(msgs) == 0:
            summary_prompt = "No overnight emails found. Your inbox appears to be up to date."
        else:
            # FIXED: Use proper email info extraction
            real_emails = []
            for msg in msgs:
                sender, subject = _extract_email_info(msg)
                if sender or subject:  # Accept if we have either sender OR subject
                    real_emails.append({
                        'sender': sender or 'Unknown Sender',
                        'subject': subject or 'No Subject',
                        'original_msg': msg  # Keep original for debugging
                    })
            
            print(f"DEBUG: Extracted {len(real_emails)} readable emails from {len(msgs)} total")
            
            if len(real_emails) == 0:
                # CRITICAL FIX: Don't let AI make up content, show the actual issue
                summary_prompt = f"""Found {len(msgs)} overnight emails but could not extract readable sender/subject information. 

This indicates a data structure issue with the Gmail API response. 
Email parsing needs to be updated to handle the actual format returned by Gmail.

Raw debug info: Check server logs for email structure details."""
            else:
                # Build actual email list from real data
                lines = []
                for email in real_emails[:10]:  # Limit to 10 emails
                    sender = email['sender']
                    subject = email['subject']
                    
                    # Truncate long subjects
                    if len(subject) > 80:
                        subject = subject[:80] + "..."
                    
                    lines.append(f"• {sender}: {subject}")
                
                summary_prompt = f"""Found {len(real_emails)} overnight emails:

{chr(10).join(lines)}

Focus on the most important or time-sensitive items."""
        
        # Generate response with actual data
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        return response_data
        
    except Exception as e:
        print(f"Gmail overnight check failed: {e}")
        return _handle_integration_error(str(e), "overnight email check")

def handle_gmail_search_command(user_input, project, use_voices, random_toggle):
    """Handle Gmail search commands with FIXED data extraction"""
    # Extract query after the command
    query_text = ""
    for prefix in ["search ", "find ", "email about "]:
        if user_input.lower().startswith(prefix):
            query_text = user_input[len(prefix):].strip()
            break
    
    if not query_text:
        return {"SyntaxPrime": "Please provide a search term after the command (e.g., 'search project updates')"}
    
    try:
        print(f"Searching Gmail for: {query_text}")
        msgs = gmail_search(query_text)
        
        # Validate response
        is_valid, error_msg = _validate_gmail_response(msgs, f"Gmail search for '{query_text}'")
        if not is_valid:
            return _handle_integration_error(error_msg, "Gmail search")
        
        # DEBUG: Log search results structure
        _debug_email_structure(msgs, f"Gmail search for '{query_text}'")
        
        if len(msgs) == 0:
            summary_prompt = f"No emails found matching '{query_text}'. Try different search terms or check if the emails exist."
        else:
            # FIXED: Use proper email info extraction for search results
            readable_results = []
            for msg in msgs:
                sender, subject = _extract_email_info(msg)
                if sender or subject:
                    readable_results.append({
                        'sender': sender or 'Unknown Sender',
                        'subject': subject or 'No Subject'
                    })
            
            if len(readable_results) == 0:
                summary_prompt = f"Search completed for '{query_text}' but no readable results found. Check email parsing logic."
            else:
                lines = []
                for result in readable_results[:15]:  # Limit to 15 results
                    subject = result['subject']
                    if len(subject) > 60:
                        subject = subject[:60] + "..."
                    lines.append(f"• {result['sender']}: {subject}")
                
                summary_prompt = f"""Found {len(readable_results)} emails matching '{query_text}':

{chr(10).join(lines)}

What would you like to know about these search results?"""
        
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        return response_data
        
    except Exception as e:
        print(f"Gmail search failed: {e}")
        return _handle_integration_error(str(e), f"Gmail search for '{query_text}'")

def handle_calendar_today_command(project, use_voices, random_toggle):
    """Handle today's calendar commands"""
    try:
        print("Fetching today's calendar events...")
        events = list_today_events(max_results=20)
        
        # Validate response
        is_valid, error_msg = _validate_calendar_response(events, "today's calendar check")
        if not is_valid:
            return _handle_integration_error(error_msg, "today's calendar check")
        
        if len(events) == 0:
            calendar_summary = "No events found for today."
            summary_prompt = "Carl's calendar is clear today. No meetings or events scheduled."
        else:
            calendar_summary = format_calendar_summary(events, "Today's Calendar")
            summary_prompt = (
                f"Here's Carl's calendar for today. Summarize the key meetings and suggest priorities:\n\n"
                f"{calendar_summary}"
            )
        
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        return response_data
        
    except Exception as e:
        print(f"Calendar check failed: {e}")
        return _handle_integration_error(str(e), "today's calendar check")

def handle_next_meeting_command(project, use_voices, random_toggle):
    """Handle next meeting commands"""
    try:
        print("Getting next meeting...")
        next_meeting = get_next_meeting()
        
        if next_meeting is None:
            summary_prompt = "No upcoming meetings found in your calendar."
        elif isinstance(next_meeting, dict) and "error" in next_meeting:
            return _handle_integration_error(next_meeting["error"], "next meeting lookup")
        elif next_meeting and next_meeting.get('summary'):
            summary_prompt = (
                f"Carl's next meeting: {next_meeting['summary']} at {next_meeting.get('start_formatted', 'Unknown time')}. "
                f"Give a brief overview and any prep suggestions."
            )
        else:
            summary_prompt = "Next meeting lookup completed but no readable meeting data found. Check calendar permissions."
        
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        return response_data
        
    except Exception as e:
        print(f"Next meeting check failed: {e}")
        return _handle_integration_error(str(e), "next meeting lookup")

def handle_good_morning_command(project, use_voices, random_toggle):
    """Handle good morning briefing commands - ADDED MISSING FUNCTION"""
    print("Good Morning command triggered")
    
    # Track which operations succeeded/failed
    operations = {
        'emails': {'success': False, 'data': None, 'error': None},
        'calendar': {'success': False, 'data': None, 'error': None},
        'next_meeting': {'success': False, 'data': None, 'error': None}
    }
    
    # Try to get overnight emails
    try:
        print("Fetching overnight emails...")
        msgs = list_overnight(include_unread=True, include_primary=False)
        is_valid, error_msg = _validate_gmail_response(msgs, "overnight emails")
        
        if is_valid and len(msgs) > 0:
            operations['emails']['success'] = True
            operations['emails']['data'] = msgs
            print(f"Successfully got {len(msgs)} emails")
        else:
            operations['emails']['error'] = error_msg or "No emails found"
            print(f"Email fetch issue: {operations['emails']['error']}")
            
    except Exception as e:
        operations['emails']['error'] = str(e)
        print(f"Email fetch failed: {e}")
    
    # Try to get today's events
    try:
        print("Fetching today's events...")
        events = list_today_events(max_results=20)
        is_valid, error_msg = _validate_calendar_response(events, "today's events")
        
        if is_valid:
            operations['calendar']['success'] = True
            operations['calendar']['data'] = events
            print(f"Successfully got {len(events)} events")
        else:
            operations['calendar']['error'] = error_msg
            print(f"Calendar fetch issue: {operations['calendar']['error']}")
            
    except Exception as e:
        operations['calendar']['error'] = str(e)
        print(f"Calendar fetch failed: {e}")
    
    # Try to get next meeting
    try:
        print("Getting next meeting...")
        next_meeting = get_next_meeting()
        
        if next_meeting and next_meeting.get('summary'):
            operations['next_meeting']['success'] = True
            operations['next_meeting']['data'] = next_meeting
            print(f"Got next meeting: {next_meeting.get('summary')}")
        else:
            operations['next_meeting']['error'] = "No upcoming meeting found"
            print("No next meeting found")
            
    except Exception as e:
        operations['next_meeting']['error'] = str(e)
        print(f"Next meeting fetch failed: {e}")
    
    # Build morning briefing based on what worked
    morning_briefing = "Good morning! Here's your daily briefing:\n\n"
    
    # Email section
    morning_briefing += "**OVERNIGHT EMAILS**\n"
    if operations['emails']['success']:
        email_count = len(operations['emails']['data'])
        morning_briefing += f"Found {email_count} overnight emails\n"
    else:
        morning_briefing += f"Email check failed: {operations['emails']['error']}\n"
    
    morning_briefing += "\n**TODAY'S CALENDAR**\n"
    if operations['calendar']['success']:
        events = operations['calendar']['data']
        if len(events) == 0:
            morning_briefing += "No events scheduled for today\n"
        else:
            calendar_summary = format_calendar_summary(events, "")
            morning_briefing += calendar_summary + "\n"
    else:
        morning_briefing += f"Calendar check failed: {operations['calendar']['error']}\n"
    
    morning_briefing += "\n**NEXT MEETING**\n"
    if operations['next_meeting']['success']:
        meeting = operations['next_meeting']['data']
        morning_briefing += f"{meeting.get('summary', 'Unknown')} at {meeting.get('start_formatted', 'Unknown time')}\n"
    else:
        morning_briefing += f"Next meeting lookup failed: {operations['next_meeting']['error']}\n"
    
    # Add status summary
    successful_ops = sum(1 for op in operations.values() if op['success'])
    if successful_ops == 0:
        morning_briefing += "\n**INTEGRATION STATUS**\n"
        morning_briefing += "⚠️  All Google integrations failed. Please check authentication at /integrations\n"
    elif successful_ops < 3:
        morning_briefing += "\n**INTEGRATION STATUS**\n"
        morning_briefing += f"⚠️  {3-successful_ops} integration(s) failed. Some data may be incomplete.\n"
    
    morning_briefing += "\n**PRIORITIES FOR TODAY**\n"
    morning_briefing += "• Review and respond to urgent emails\n"
    morning_briefing += "• Prepare for upcoming meetings\n"
    morning_briefing += "• Check for any calendar conflicts\n"
    
    try:
        print("Saving daily log...")
        save_daily_log_enhanced("morning", morning_briefing)
        print("Daily log saved successfully")
    except Exception as e:
        print(f"Failed to save daily log: {e}")
    
    try:
        print("Generating AI response...")
        retrieval_ctx = enhanced_retrieve(morning_briefing, k=5) if is_ready() else []
        
        response_data = generate_response(
            f"Summarize this morning briefing and suggest 3 key priorities:\n\n{morning_briefing}",
            use_voices, random_toggle, project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        print("Morning briefing response generated successfully")
        return response_data
        
    except Exception as e:
        print(f"Response generation failed: {e}")
        return {"SyntaxPrime": f"Morning briefing compiled but response generation failed: {str(e)}\n\n{morning_briefing}"}

def process_gmail_command(user_input, project, use_voices, random_toggle):
    """Process Gmail/calendar commands and return response data"""
    user_lower = user_input.lower().strip()
    
    # Gmail overnight (multiple aliases)
    if user_lower in ["overnight", "mail", "emails", "inbox", "check mail"]:
        response_data = handle_overnight_command(project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Gmail search (multiple aliases)
    if user_lower.startswith(("search ", "find ", "email about ")):
        response_data = handle_gmail_search_command(user_input, project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Today's calendar
    if user_lower in ["calendar", "today", "meetings", "schedule"]:
        response_data = handle_calendar_today_command(project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Next meeting
    if user_lower in ["next meeting", "next", "upcoming"]:
        response_data = handle_next_meeting_command(project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Good Morning
    if user_lower in ["good morning", "morning", "gm"]:
        response_data = handle_good_morning_command(project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Command not recognized
    return None, False
