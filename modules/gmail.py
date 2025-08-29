# modules/gmail.py - Gmail Integration Module

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

def handle_overnight_command(project, use_voices, random_toggle):
    """Handle overnight email check commands"""
    try:
        msgs = list_overnight(include_unread=True, include_primary=False)
        lines = [f"- {msg.get('sender', 'Unknown')}: {msg.get('subject', 'No Subject')}" for msg in msgs[:25]]
        summary_prompt = (
            f"Found {len(msgs)} overnight emails. Here's the summary:\n\n"
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
        return {"SyntaxPrime": f"Gmail check failed: {e}"}

def handle_gmail_search_command(user_input, project, use_voices, random_toggle):
    """Handle Gmail search commands"""
    # Extract query after the command
    for prefix in ["search ", "find ", "email about "]:
        if user_input.lower().startswith(prefix):
            query_text = user_input[len(prefix):].strip()
            break
    
    try:
        msgs = gmail_search(query_text)
        lines = [f"- Message ID: {msg.get('id', 'Unknown')}" for msg in msgs[:25]]
        summary_prompt = (
            f"Found {len(msgs)} messages for search query: '{query_text}'\n\n"
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
        return {"SyntaxPrime": f"Gmail search failed: {e}"}

def handle_calendar_today_command(project, use_voices, random_toggle):
    """Handle today's calendar commands"""
    try:
        events = list_today_events(max_results=20)
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
        return {"SyntaxPrime": f"Calendar check failed: {e}"}

def handle_calendar_tomorrow_command(project, use_voices, random_toggle):
    """Handle tomorrow's calendar commands"""
    try:
        events = list_tomorrow_events(max_results=20)
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
        return {"SyntaxPrime": f"Tomorrow's calendar failed: {e}"}

def handle_next_meeting_command(project, use_voices, random_toggle):
    """Handle next meeting commands"""
    try:
        next_meeting = get_next_meeting()
        if next_meeting and next_meeting.get('summary'):
            summary_prompt = (
                f"Carl's next meeting: {next_meeting['summary']} at {next_meeting.get('start_formatted', 'Unknown time')}. "
                f"Give a brief overview and any prep suggestions."
            )
        else:
            summary_prompt = "No upcoming meetings found."
        
        retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
        response_data = generate_response(
            summary_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        return response_data
    except Exception as e:
        print(f"Next meeting check failed: {e}")
        return {"SyntaxPrime": f"Next meeting check failed: {e}"}

def handle_calendar_search_command(user_input, project, use_voices, random_toggle):
    """Handle calendar search commands"""
    # Extract query after the command
    for prefix in ["meeting about ", "calendar search "]:
        if user_input.lower().startswith(prefix):
            query_text = user_input[len(prefix):].strip()
            break
    
    try:
        events = search_calendar(query_text, days_ahead=30, max_results=10)
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
        return {"SyntaxPrime": f"Calendar search failed: {e}"}

def handle_good_morning_command(project, use_voices, random_toggle):
    """Handle good morning briefing commands"""
    print("Good Morning command triggered")
    try:
        print("About to call list_overnight")
        msgs = list_overnight(include_unread=True, include_primary=False)
        print(f"Got {len(msgs)} messages")
        
        print("About to call list_today_events")
        events = list_today_events(max_results=20)
        print(f"Got {len(events)} events")
        
        print("About to call get_next_meeting")
        next_meeting = get_next_meeting()
        print(f"Got next meeting: {next_meeting}")
        
        # Format briefing
        email_summary = f"Found {len(msgs)} overnight emails"
        calendar_summary = format_calendar_summary(events, "Today's Schedule")
        
        morning_briefing = f"""Good morning! Here's your daily briefing:

**OVERNIGHT EMAILS**
{email_summary}

**TODAY'S CALENDAR**
{calendar_summary}

**NEXT MEETING**
{f"{next_meeting.get('summary', 'Unknown')} at {next_meeting.get('start_formatted', 'Unknown time')}" if next_meeting else "No meetings scheduled"}

**PRIORITIES FOR TODAY**
• Review urgent emails
• Prepare for upcoming meetings
• Check calendar for conflicts"""

        print("About to save daily log")
        save_daily_log_enhanced("morning", morning_briefing)
        print("Daily log saved")
        
        print("About to call retrieve")
        retrieval_ctx = enhanced_retrieve(morning_briefing, k=5) if is_ready() else []
        print("Retrieve completed")
        
        print("About to call generate_response")
        response_data = generate_response(
            f"Summarize this morning briefing and suggest 3 key priorities:\n\n{morning_briefing}",
            use_voices, random_toggle, project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        print("generate_response completed")
        return response_data
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error trace: {error_details}")
        return {"SyntaxPrime": f"Morning briefing failed: {str(e)} | Type: {type(e).__name__} | Details: {error_details[:200]}"}

def handle_good_evening_command(project, use_voices, random_toggle):
    """Handle good evening briefing commands"""
    try:
        # Get today's sent emails and completed meetings
        today_events = list_today_events(max_results=20)
        tomorrow_events = list_tomorrow_events(max_results=15)
        
        # Filter completed events
        now = datetime.datetime.now()
        completed_events = []
        upcoming_events = []
        
        for event in today_events:
            # Simple time comparison - events that started before now are "completed"
            if 'T' in event['start']:
                event_time = datetime.datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                if event_time < now:
                    completed_events.append(event)
                else:
                    upcoming_events.append(event)
        
        evening_summary = f"""Good evening! Here's your day wrap-up:

**TODAY'S COMPLETED MEETINGS ({len(completed_events)})**
{chr(10).join([f"• {e['start_formatted']} — {e['summary']}" for e in completed_events[:5]]) if completed_events else "No meetings completed"}

**STILL UPCOMING TODAY**
{chr(10).join([f"• {e['start_formatted']} — {e['summary']}" for e in upcoming_events]) if upcoming_events else "No more meetings today"}

**TOMORROW'S PREP NEEDED**
{format_calendar_summary(tomorrow_events[:5], "")}

**END OF DAY CHECKLIST**
• Review and respond to urgent emails
• Prepare materials for tomorrow's meetings  
• Set priorities for tomorrow
• Clear desk and close open tasks"""

        # Save to daily log
        save_daily_log_enhanced("evening", evening_summary)
        
        # Generate AI response
        retrieval_ctx = enhanced_retrieve(evening_summary, k=5) if is_ready() else []
        response_data = generate_response(
            f"Summarize this evening wrap-up and suggest 3 things to prepare for tomorrow:\n\n{evening_summary}",
            use_voices, random_toggle, project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        return response_data
        
    except Exception as e:
        print(f"Evening summary failed: {e}")
        return {"SyntaxPrime": f"Evening summary failed: {e}"}

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