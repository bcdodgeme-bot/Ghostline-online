# utils/gmail_client.py
from __future__ import annotations
import os, datetime, base64, email
from typing import List, Dict, Any, Optional
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Updated scopes to include both Gmail and Calendar
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly"
]

SECRETS_DIR = "secrets"
CREDS_PATH = os.path.join(SECRETS_DIR, "gmail_credentials.json")
TOKEN_PATH = os.path.join(SECRETS_DIR, "gmail_token.json")

def _ensure_secrets_dir():
    os.makedirs(SECRETS_DIR, exist_ok=True)

def _is_server_environment():
    """Always treat Railway as server environment to prevent browser launches"""
    return True

def get_gmail_service():
    """Get Gmail API service"""
    _ensure_secrets_dir()
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Always treat as server environment on Railway
            raise RuntimeError("Gmail token missing. Please generate gmail_token.json locally and commit it to your repository.")
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def get_calendar_service():
    """Get Calendar API service"""
    _ensure_secrets_dir()
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Always treat as server environment on Railway
            raise RuntimeError("Gmail token missing. Please generate gmail_token.json locally and commit it to your repository.")
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)

# Keep the original get_service() for backward compatibility
def get_service():
    """Legacy function - returns Gmail service"""
    return get_gmail_service()

def _headers_to_dict(payload_headers):
    d = {}
    for h in payload_headers or []:
        d[h.get("name", "").lower()] = h.get("value", "")
    return d

# =============================================================================
# GMAIL FUNCTIONS (existing functionality)
# =============================================================================

def list_messages(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    svc = get_gmail_service()
    resp = svc.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
    ids = [m["id"] for m in resp.get("messages", [])]
    out = []
    for mid in ids:
        msg = svc.users().messages().get(userId="me", id=mid, format="metadata",
                                         metadataHeaders=["From","Subject","Date"]).execute()
        hdrs = _headers_to_dict(msg.get("payload", {}).get("headers", []))
        out.append({
            "id": mid,
            "from": hdrs.get("from",""),
            "subject": hdrs.get("subject","(no subject)"),
            "date": hdrs.get("date",""),
            "threadId": msg.get("threadId","")
        })
    return out

def midnight_query(unread_only: bool = True) -> str:
    # Since local midnight (your machine's timezone)
    now = datetime.datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    ts = int(midnight.timestamp())
    q = f"after:{ts} label:inbox category:primary"
    if unread_only:
        q += " is:unread"
    return q

def list_overnight(max_results: int = 25, unread_only: bool = True):
    return list_messages(midnight_query(unread_only), max_results=max_results)

def search(query_text: str, max_results: int = 25):
    # allow natural text + gmail operators
    q = f"{query_text} label:inbox"
    return list_messages(q, max_results=max_results)

# =============================================================================
# CALENDAR FUNCTIONS (new functionality)
# =============================================================================

def _format_datetime(dt_str: str) -> str:
    """Format datetime string for display"""
    try:
        if 'T' in dt_str:
            # Parse ISO format datetime
            if dt_str.endswith('Z'):
                dt = datetime.datetime.fromisoformat(dt_str[:-1] + '+00:00')
            else:
                dt = datetime.datetime.fromisoformat(dt_str)
            return dt.strftime("%I:%M %p")
        else:
            # All-day event
            return "All day"
    except:
        return dt_str

def _parse_calendar_datetime(dt_obj: Dict) -> Optional[datetime.datetime]:
    """Parse calendar datetime object"""
    try:
        if 'dateTime' in dt_obj:
            dt_str = dt_obj['dateTime']
            if dt_str.endswith('Z'):
                return datetime.datetime.fromisoformat(dt_str[:-1] + '+00:00')
            else:
                return datetime.datetime.fromisoformat(dt_str)
        elif 'date' in dt_obj:
            # All-day event
            date_str = dt_obj['date']
            return datetime.datetime.fromisoformat(date_str + 'T00:00:00')
    except:
        pass
    return None

def list_calendar_events(
    start_date: datetime.datetime, 
    end_date: datetime.datetime, 
    max_results: int = 20
) -> List[Dict[str, Any]]:
    """List calendar events between start_date and end_date"""
    try:
        service = get_calendar_service()
        
        # Convert to RFC3339 format
        time_min = start_date.isoformat() + 'Z'
        time_max = end_date.isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            maxResults=max_results,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        formatted_events = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            
            formatted_events.append({
                'id': event.get('id', ''),
                'summary': event.get('summary', 'No title'),
                'start': start,
                'end': end,
                'start_formatted': _format_datetime(start),
                'end_formatted': _format_datetime(end),
                'location': event.get('location', ''),
                'description': event.get('description', ''),
                'attendees': [a.get('email', '') for a in event.get('attendees', [])],
                'organizer': event.get('organizer', {}).get('email', ''),
                'status': event.get('status', ''),
                'htmlLink': event.get('htmlLink', '')
            })
        
        return formatted_events
        
    except Exception as e:
        print(f"Calendar API error: {e}")
        return []

def list_today_events(max_results: int = 20) -> List[Dict[str, Any]]:
    """Get today's calendar events"""
    now = datetime.datetime.now()
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    return list_calendar_events(start_of_day, end_of_day, max_results)

def list_tomorrow_events(max_results: int = 20) -> List[Dict[str, Any]]:
    """Get tomorrow's calendar events"""
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    start_of_day = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = tomorrow.replace(hour=23, minute=59, second=59, microsecond=999999)
    
    return list_calendar_events(start_of_day, end_of_day, max_results)

def list_this_week_events(max_results: int = 50) -> List[Dict[str, Any]]:
    """Get this week's calendar events"""
    now = datetime.datetime.now()
    # Start of week (Monday)
    start_of_week = now - datetime.timedelta(days=now.weekday())
    start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    # End of week (Sunday)
    end_of_week = start_of_week + datetime.timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    return list_calendar_events(start_of_week, end_of_week, max_results)

def search_calendar(query: str, days_ahead: int = 30, max_results: int = 20) -> List[Dict[str, Any]]:
    """Search calendar events by query string"""
    now = datetime.datetime.now()
    start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = start_date + datetime.timedelta(days=days_ahead)
    
    # Get all events in the time range
    all_events = list_calendar_events(start_date, end_date, max_results * 2)
    
    # Filter events that match the query
    query_lower = query.lower()
    matching_events = []
    
    for event in all_events:
        # Search in title, description, location
        searchable_text = ' '.join([
            event.get('summary', ''),
            event.get('description', ''),
            event.get('location', ''),
            event.get('organizer', '')
        ]).lower()
        
        if query_lower in searchable_text:
            matching_events.append(event)
            
        if len(matching_events) >= max_results:
            break
    
    return matching_events

def get_next_meeting(hours_ahead: int = 24) -> Optional[Dict[str, Any]]:
    """Get the next upcoming meeting"""
    now = datetime.datetime.now()
    end_time = now + datetime.timedelta(hours=hours_ahead)
    
    events = list_calendar_events(now, end_time, max_results=5)
    
    # Find the next event that hasn't started yet
    for event in events:
        start_dt = _parse_calendar_datetime({'dateTime': event['start']})
        if start_dt and start_dt > now:
            return event
    
    return None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_calendar_summary(events: List[Dict[str, Any]], title: str = "Calendar Events") -> str:
    """Format calendar events for display"""
    if not events:
        return f"{title}: No events found."
    
    lines = [f"{title}:"]
    
    for event in events:
        time_str = event['start_formatted']
        if event['start_formatted'] != event['end_formatted'] and event['end_formatted'] != "All day":
            time_str += f" - {event['end_formatted']}"
        
        line = f"• {time_str} — {event['summary']}"
        if event.get('location'):
            line += f" ({event['location']})"
        
        lines.append(line)
    
    return '\n'.join(lines)

def test_calendar_connection() -> bool:
    """Test if calendar API is working"""
    try:
        service = get_calendar_service()
        # Try to get calendar list
        service.calendarList().list().execute()
        return True
    except Exception as e:
        print(f"Calendar connection test failed: {e}")
        return False

def test_gmail_connection() -> bool:
    """Test if Gmail API is working"""
    try:
        service = get_gmail_service()
        # Try to get profile
        service.users().getProfile(userId='me').execute()
        return True
    except Exception as e:
        print(f"Gmail connection test failed: {e}")
        return False