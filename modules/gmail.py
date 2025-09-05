# modules/gmail.py - DEBUGGED VERSION with proper data structure handling

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

# Keep all other existing functions unchanged for now...
# (handle_calendar_today_command, handle_good_morning_command, etc.)

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

    # Command not recognized (return just the functions we fixed for now)
    return None, False
