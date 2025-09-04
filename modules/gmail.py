# modules/gmail.py - Gmail Integration Module (FIXED VERSION)

import os
import datetime
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
    """Handle overnight email check commands with proper error handling"""
    try:
        print("Starting overnight email check...")
        msgs = list_overnight(include_unread=True, include_primary=False)
        
        # Validate response
        is_valid, error_msg = _validate_gmail_response(msgs, "overnight email check")
        if not is_valid:
            print(f"Overnight email validation failed: {error_msg}")
            return _handle_integration_error(error_msg, "overnight email check")
        
        # Check if we actually got real data
        if len(msgs) == 0:
            summary_prompt = "No overnight emails found. Your inbox appears to be up to date."
        else:
            # Verify the emails have real data
            real_emails = [msg for msg in msgs if msg.get('sender') and msg.get('subject')]
            if len(real_emails) == 0:
                summary_prompt = "Overnight email check completed but no readable emails found. This might indicate an authentication or permission issue."
            else:
                lines = [f"- {msg.get('sender', 'Unknown')}: {msg.get('subject', 'No Subject')}"
                        for msg in real_emails[:25]]
                summary_prompt = (
                    f"Found {len(real_emails)} overnight emails. Here's the summary:\n\n"
                    + "\n".join(lines)
                )
        
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
    """Handle Gmail search commands with proper validation"""
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
        
        if len(msgs) == 0:
            summary_prompt = f"No emails found matching '{query_text}'. Try different search terms or check if the emails exist."
        else:
            # Validate that we have real message data
            real_msgs = [msg for msg in msgs if msg.get('id')]
            if len(real_msgs) == 0:
                summary_prompt = f"Search completed for '{query_text}' but no readable results found. This might indicate a permission issue."
            else:
                lines = [f"- Message ID: {msg.get('id', 'Unknown')}" for msg in real_msgs[:25]]
                summary_prompt = (
                    f"Found {len(real_msgs)} messages for search query: '{query_text}'\n\n"
                    + "\n".join(lines)
                )
        
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
    """Handle today's calendar commands with comprehensive event retrieval"""
    try:
        print("Fetching today's calendar events...")
        events = list_today_events(max_results=20)
        
        # Validate response
        is_valid, error_msg = _validate_calendar_response(events, "today's calendar check")
        if not is_valid:
            return _handle_integration_error(error_msg, "today's calendar check")
        
        # Filter and categorize events properly
        now = datetime.datetime.now()
        work_events = []
        personal_events = []
        
        for event in events:
            # Check if event has required fields
            if not event.get('summary'):
                continue
                
            # Categorize events (this is a simple heuristic - you may need to adjust)
            summary_lower = event.get('summary', '').lower()
            if any(work_term in summary_lower for work_term in ['meeting', 'standup', 'review', 'sync', 'call', 'interview', 'demo']):
                work_events.append(event)
            else:
                personal_events.append(event)
        
        total_events = len(work_events) + len(personal_events)
        
        if total_events == 0:
            calendar_summary = "No events found for today."
            summary_prompt = "Carl's calendar is clear today. No meetings or events scheduled."
        else:
            calendar_summary = format_calendar_summary(events, "Today's Calendar")
            
            work_count = len(work_events)
            personal_count = len(personal_events)
            
            summary_prompt = (
                f"Here's Carl's calendar for today ({work_count} work, {personal_count} personal events). "
                f"Summarize the key meetings and suggest priorities:\n\n"
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

def handle_calendar_tomorrow_command(project, use_voices, random_toggle):
    """Handle tomorrow's calendar commands with proper validation"""
    try:
        print("Fetching tomorrow's calendar events...")
        events = list_tomorrow_events(max_results=20)
        
        # Validate response
        is_valid, error_msg = _validate_calendar_response(events, "tomorrow's calendar check")
        if not is_valid:
            return _handle_integration_error(error_msg, "tomorrow's calendar check")
        
        if len(events) == 0:
            calendar_summary = "No events scheduled for tomorrow."
            summary_prompt = "Carl's calendar is clear tomorrow. No meetings or events scheduled."
        else:
            calendar_summary = format_calendar_summary(events, "Tomorrow's Calendar")
            summary_prompt = (
                f"Here's Carl's calendar for tomorrow. Highlight important meetings and prep needed:\n\n"
                f"{calendar_summary}"
            )
        
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        return response_data
        
    except Exception as e:
        print(f"Tomorrow's calendar failed: {e}")
        return _handle_integration_error(str(e), "tomorrow's calendar check")

def handle_next_meeting_command(project, use_voices, random_toggle):
    """Handle next meeting commands with proper validation"""
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

def handle_calendar_search_command(user_input, project, use_voices, random_toggle):
    """Handle calendar search commands with validation"""
    # Extract query after the command
    query_text = ""
    for prefix in ["meeting about ", "calendar search "]:
        if user_input.lower().startswith(prefix):
            query_text = user_input[len(prefix):].strip()
            break
    
    if not query_text:
        return {"SyntaxPrime": "Please provide a search term after the command (e.g., 'meeting about project review')"}
    
    try:
        print(f"Searching calendar for: {query_text}")
        events = search_calendar(query_text, days_ahead=30, max_results=10)
        
        # Validate response
        is_valid, error_msg = _validate_calendar_response(events, f"calendar search for '{query_text}'")
        if not is_valid:
            return _handle_integration_error(error_msg, "calendar search")
        
        if len(events) == 0:
            calendar_summary = f"No meetings found matching '{query_text}'"
            summary_prompt = f"Carl searched his calendar for '{query_text}' but no matching meetings were found. Try different search terms."
        else:
            calendar_summary = format_calendar_summary(events, f"Calendar search: '{query_text}'")
            summary_prompt = (
                f"Carl searched his calendar for '{query_text}'. Here are the relevant meetings:\n\n"
                f"{calendar_summary}\n\n"
                f"Summarize the key meetings and any patterns or next steps."
            )
        
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        return response_data
        
    except Exception as e:
        print(f"Calendar search failed: {e}")
        return _handle_integration_error(str(e), f"calendar search for '{query_text}'")

def handle_good_morning_command(project, use_voices, random_toggle):
    """Handle good morning briefing commands with comprehensive error handling"""
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

def handle_good_evening_command(project, use_voices, random_toggle):
    """Handle good evening briefing commands with proper validation"""
    try:
        # Get today's events with validation
        today_events = list_today_events(max_results=20)
        is_valid_today, error_msg_today = _validate_calendar_response(today_events, "today's events")
        
        tomorrow_events = list_tomorrow_events(max_results=15)
        is_valid_tomorrow, error_msg_tomorrow = _validate_calendar_response(tomorrow_events, "tomorrow's events")
        
        # Build evening summary with error handling
        evening_summary = "Good evening! Here's your day wrap-up:\n\n"
        
        if is_valid_today and len(today_events) > 0:
            # Filter completed events
            now = datetime.datetime.now()
            completed_events = []
            upcoming_events = []
            
            for event in today_events:
                if not event.get('start') or not event.get('summary'):
                    continue
                    
                # Simple time comparison - events that started before now are "completed"
                if 'T' in event['start']:
                    try:
                        event_time = datetime.datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                        if event_time < now:
                            completed_events.append(event)
                        else:
                            upcoming_events.append(event)
                    except:
                        # If time parsing fails, assume it's completed
                        completed_events.append(event)
            
            evening_summary += f"**TODAY'S COMPLETED MEETINGS ({len(completed_events)})**\n"
            if completed_events:
                meeting_lines = [f"• {e.get('start_formatted', 'Unknown time')} — {e.get('summary', 'Unknown')}"
                               for e in completed_events[:5]]
                evening_summary += "\n".join(meeting_lines) + "\n"
            else:
                evening_summary += "No completed meetings\n"
            
            evening_summary += f"\n**STILL UPCOMING TODAY**\n"
            if upcoming_events:
                upcoming_lines = [f"• {e.get('start_formatted', 'Unknown time')} — {e.get('summary', 'Unknown')}"
                                for e in upcoming_events]
                evening_summary += "\n".join(upcoming_lines) + "\n"
            else:
                evening_summary += "No more meetings today\n"
        else:
            evening_summary += "**TODAY'S MEETINGS**\n"
            if error_msg_today:
                evening_summary += f"Calendar check failed: {error_msg_today}\n"
            else:
                evening_summary += "No meetings found for today\n"
        
        evening_summary += "\n**TOMORROW'S PREP NEEDED**\n"
        if is_valid_tomorrow and len(tomorrow_events) > 0:
            tomorrow_summary = format_calendar_summary(tomorrow_events[:5], "")
            evening_summary += tomorrow_summary + "\n"
        else:
            if error_msg_tomorrow:
                evening_summary += f"Tomorrow's calendar check failed: {error_msg_tomorrow}\n"
            else:
                evening_summary += "No events scheduled for tomorrow\n"
        
        evening_summary += "\n**END OF DAY CHECKLIST**\n"
        evening_summary += "• Review and respond to urgent emails\n"
        evening_summary += "• Prepare materials for tomorrow's meetings\n"
        evening_summary += "• Set priorities for tomorrow\n"
        evening_summary += "• Clear desk and close open tasks\n"
        
        # Save to daily log
        try:
            save_daily_log_enhanced("evening", evening_summary)
        except Exception as e:
            print(f"Failed to save evening log: {e}")
        
        # Generate AI response
        retrieval_ctx = enhanced_retrieve(evening_summary, k=5) if is_ready() else []
        response_data = generate_response(
            f"Summarize this evening wrap-up and suggest 3 things to prepare for tomorrow:\n\n{evening_summary}",
            use_voices, random_toggle, project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        return response_data
        
    except Exception as e:
        print(f"Evening summary failed: {e}")
        return _handle_integration_error(str(e), "evening summary")

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

    # Tomorrow's calendar
    if user_lower in ["tomorrow", "tomorrow's schedule", "next day"]:
        response_data = handle_calendar_tomorrow_command(project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Next meeting
    if user_lower in ["next meeting", "next", "upcoming"]:
        response_data = handle_next_meeting_command(project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Search calendar
    if user_lower.startswith(("meeting about ", "calendar search ")):
        response_data = handle_calendar_search_command(user_input, project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Good Morning
    if user_lower in ["good morning", "morning", "gm"]:
        response_data = handle_good_morning_command(project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Good Evening
    if user_lower in ["good evening", "evening", "ge", "wrap up", "day summary"]:
        response_data = handle_good_evening_command(project, use_voices, random_toggle)
        save_conversation_enhanced(project, user_input, response_data)
        return response_data, True

    # Command not recognized
    return None, False
