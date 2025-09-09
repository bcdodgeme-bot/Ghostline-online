# utils/gmail_client.py - WITH MULTI-CALENDAR FIX
# This version includes detailed error logging to identify the exact Google API issue

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# --------------------------- Config / Scopes ---------------------------

DEFAULT_TZ = ZoneInfo(os.getenv("APP_TIMEZONE", "America/New_York"))
TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")

# Combined scopes for both Gmail and Calendar
ALL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly"
]


# --------------------------- Auth helpers ---------------------------

def _build_creds() -> Credentials:
    """Build credentials with all required scopes"""
    creds: Optional[Credentials] = None
    
    print(f"DEBUG: Looking for token at: {TOKEN_PATH}")
    print(f"DEBUG: Token file exists: {os.path.exists(TOKEN_PATH)}")
    
    # Try to load existing credentials
    if os.path.exists(TOKEN_PATH):
        try:
            print("DEBUG: Loading existing token...")
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, ALL_SCOPES)
            print("DEBUG: Token loaded successfully")
            print(f"DEBUG: Token valid: {creds.valid}")
            print(f"DEBUG: Token expired: {creds.expired}")
            print(f"DEBUG: Has refresh token: {bool(creds.refresh_token)}")
            print(f"DEBUG: Token scopes: {creds.scopes}")
        except Exception as e:
            print(f"DEBUG: Failed to load existing token: {e}")
            creds = None

    # If token invalid/expired, try to refresh
    if creds and not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                print("DEBUG: Refreshing expired token...")
                creds.refresh(Request())
                print("DEBUG: Token refreshed successfully")
                
                # Save the refreshed token
                with open(TOKEN_PATH, "w") as token_file:
                    token_file.write(creds.to_json())
                print("DEBUG: Refreshed token saved")
                
            except Exception as e:
                print(f"DEBUG: Token refresh failed: {e}")
                creds = None
        else:
            print("DEBUG: Token invalid and cannot refresh")
            creds = None

    # If no valid credentials, start OAuth flow
    if not creds:
        print("DEBUG: No valid credentials - OAuth flow needed")
        if not os.path.exists(CREDENTIALS_PATH):
            raise FileNotFoundError(f"Credentials file not found: {CREDENTIALS_PATH}")
        
        print("DEBUG: Starting OAuth flow...")
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, ALL_SCOPES)
        creds = flow.run_local_server(port=0)
        
        # Save the credentials for next time
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())
        print("DEBUG: New credentials saved")

    print(f"DEBUG: Final credentials check - valid: {creds.valid}")
    return creds


def _gmail_service():
    """Build Gmail API service"""
    print("DEBUG: Creating Gmail service...")
    creds = _build_creds()
    return build("gmail", "v1", credentials=creds)


def _calendar_service():
    """Build Calendar API service"""
    print("DEBUG: Creating Calendar service...")
    creds = _build_creds()
    return build("calendar", "v3", credentials=creds)


# --------------------------- Time helpers ---------------------------

def _iso_bounds_today_local() -> tuple[str, str, str]:
    """ISO time bounds for 'local today': 00:00 to 23:59:59, with timezone name."""
    now = datetime.now(DEFAULT_TZ)
    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start_of_day.isoformat(), end_of_day.isoformat(), str(DEFAULT_TZ)


def _overnight_query(include_unread: bool = False, include_primary: bool = False, query_extra: Optional[str] = None) -> str:
    """Build Gmail query for messages since local midnight."""
    now_local = datetime.now(DEFAULT_TZ)
    midnight_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    after_str = midnight_local.strftime("%Y/%m/%d")
    
    parts = [f"after:{after_str}"]
    if include_unread:
        parts.append("is:unread")
    if include_primary:
        parts.append("category:primary")
    if query_extra:
        parts.append(query_extra)
    
    return " ".join(parts)


def _to_local(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string to local timezone."""
    if not dt_str:
        return None
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(DEFAULT_TZ)
    except Exception:
        return None


def _format_time_local(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.strftime("%-I:%M %p") if os.name != "nt" else dt.strftime("%#I:%M %p")


def _event_item(e: Dict) -> Dict:
    """Normalize a Calendar event to what app.py expects"""
    summary = e.get("summary") or "(No title)"
    start = e.get("start", {})
    start_iso = start.get("dateTime") or start.get("date")
    start_dt_local = _to_local(start_iso) if start_iso else None
    start_formatted = _format_time_local(start_dt_local) if start_dt_local else ("All day" if start.get("date") else "")
    return {
        "id": e.get("id"),
        "summary": summary,
        "start": start_iso,
        "start_formatted": start_formatted,
        "date_formatted": start_dt_local.strftime("%A, %B %d, %Y") if start_dt_local else "",
        "time_formatted": start_formatted
    }


def _get_message_details(service, message_id: str, user_id: str = "me") -> Dict:
    """Fetch sender and subject for a specific message"""
    try:
        message = service.users().messages().get(
            userId=user_id,
            id=message_id,
            format='metadata',  # Only get headers, not full content
            metadataHeaders=['From', 'Subject', 'Date']
        ).execute()
        
        payload = message.get('payload', {})
        headers = payload.get('headers', [])
        
        # Extract sender and subject from headers
        sender = "Unknown Sender"
        subject = "No Subject"
        date = ""
        
        for header in headers:
            name = header.get('name', '').lower()
            value = header.get('value', '')
            
            if name == 'from':
                # Extract just the name/email, remove extra formatting
                if '<' in value and '>' in value:
                    # Format: "Name <email@domain.com>" -> "Name"
                    sender = value.split('<')[0].strip().strip('"')
                    if not sender:
                        # If no name, use email
                        sender = value.split('<')[1].split('>')[0]
                else:
                    sender = value
                    
            elif name == 'subject':
                subject = value
                
            elif name == 'date':
                date = value
        
        return {
            'id': message_id,
            'sender': sender,
            'subject': subject,
            'date': date
        }
        
    except Exception as e:
        print(f"DEBUG: Failed to get details for message {message_id}: {e}")
        return {
            'id': message_id,
            'sender': "Unknown",
            'subject': "Error fetching details",
            'date': ""
        }


def _gmail_list_with_details(query: str, user_id: str = "me", max_results: int = 25) -> List[Dict]:
    """Get Gmail messages with sender/subject details"""
    try:
        svc = _gmail_service()
        print(f"DEBUG: Gmail search query: '{query}'")
        
        # First, get the list of message IDs
        resp = svc.users().messages().list(
            userId=user_id, q=query, maxResults=max_results
        ).execute()
        
        message_batch = resp.get("messages", [])
        print(f"DEBUG: Found {len(message_batch)} messages")
        
        # Then fetch details for each message
        detailed_messages = []
        for msg in message_batch:
            details = _get_message_details(svc, msg['id'], user_id)
            detailed_messages.append(details)
        
        print(f"DEBUG: Fetched details for {len(detailed_messages)} messages")
        return detailed_messages
        
    except Exception as e:
        print(f"DEBUG: Gmail list with details error: {e}")
        raise


# --- Backward-compatible: used by app.py ---

def list_overnight(include_unread: bool = False, include_primary: bool = False, query_extra: Optional[str] = None) -> List[Dict]:
    """Return list of detailed message info since local midnight"""
    try:
        q = _overnight_query(include_unread, include_primary, query_extra)
        detailed_msgs = _gmail_list_with_details(q, max_results=25)
        print(f"DEBUG: list_overnight returning {len(detailed_msgs)} detailed messages")
        return detailed_msgs
    except Exception as e:
        print(f"DEBUG: list_overnight error: {e}")
        raise


def search(query: str) -> List[Dict]:
    """Generic Gmail search with sender/subject details"""
    try:
        detailed_msgs = _gmail_list_with_details(query, max_results=25)
        print(f"DEBUG: Gmail search returning {len(detailed_msgs)} detailed messages")
        return detailed_msgs
    except Exception as e:
        print(f"DEBUG: Gmail search error: {e}")
        raise


# =============================================================================
# CALENDAR FUNCTIONS - WITH MULTI-CALENDAR FIX
# =============================================================================

def list_today_events_all_calendars(max_results: int = 50) -> List[Dict]:
    """
    FIXED: Calendar events from ALL calendars for today (local timezone)
    This fixes the issue where events in secondary calendars weren't showing
    """
    try:
        print("DEBUG: === Starting list_today_events_all_calendars ===")
        
        svc = _calendar_service()
        timeMin, timeMax, tzname = _iso_bounds_today_local()
        
        print(f"DEBUG: Time bounds - Min: {timeMin}, Max: {timeMax}")
        print(f"DEBUG: Timezone: {tzname}")
        
        # Get all calendars first
        print("DEBUG: Fetching all accessible calendars...")
        calendar_list = svc.calendarList().list().execute()
        calendars = calendar_list.get('items', [])
        
        print(f"DEBUG: Found {len(calendars)} calendars")
        for cal in calendars:
            cal_name = cal.get('summary', 'Unnamed')
            cal_id = cal.get('id', 'No ID')
            is_primary = cal.get('primary', False)
            print(f"DEBUG: - {'[PRIMARY]' if is_primary else '[SECONDARY]'} {cal_name}")
        
        # Collect events from all calendars
        all_events = []
        
        for calendar in calendars:
            cal_id = calendar.get('id')
            cal_name = calendar.get('summary', 'Unnamed Calendar')
            
            # Skip certain calendar types that are usually not relevant
            if 'holiday' in cal_name.lower() or 'birthday' in cal_name.lower():
                print(f"DEBUG: Skipping calendar: {cal_name}")
                continue
            
            try:
                print(f"DEBUG: Fetching events from calendar: {cal_name}")
                
                resp = svc.events().list(
                    calendarId=cal_id,
                    timeMin=timeMin,
                    timeMax=timeMax,
                    singleEvents=True,
                    orderBy="startTime",
                    timeZone=tzname,
                    maxResults=max_results,
                ).execute()
                
                items = resp.get("items", [])
                print(f"DEBUG: Found {len(items)} events in {cal_name}")
                
                # Process each event and add calendar context
                for item in items:
                    event = _event_item(item)
                    event['calendar_name'] = cal_name
                    event['calendar_id'] = cal_id
                    all_events.append(event)
                    
            except Exception as e:
                print(f"DEBUG: Error fetching from calendar {cal_name}: {e}")
                # Continue with other calendars even if one fails
                continue
        
        # Sort all events by start time
        all_events.sort(key=lambda x: x.get('start', ''))
        
        print(f"DEBUG: Successfully processed {len(all_events)} total events from all calendars")
        
        # Log event details for debugging
        for event in all_events:
            print(f"DEBUG: Event - {event.get('start_formatted', 'No time')}: {event.get('summary', 'No title')} (from {event.get('calendar_name', 'Unknown calendar')})")
        
        print("DEBUG: === Completed list_today_events_all_calendars ===")
        return all_events
        
    except Exception as e:
        print(f"DEBUG: list_today_events_all_calendars FAILED - Type: {type(e).__name__}")
        print(f"DEBUG: Error details: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        raise


def list_today_events(max_results: int = 10, calendar_id: str = "primary") -> List[Dict]:
    """
    UPDATED: Calendar events for today - now checks all calendars by default
    Maintains backward compatibility but uses the improved multi-calendar approach
    """
    if calendar_id == "primary":
        # Use the new all-calendars approach for better results
        return list_today_events_all_calendars(max_results)
    else:
        # Keep original single-calendar logic for specific calendar requests
        try:
            print("DEBUG: === Starting list_today_events (single calendar) ===")
            print(f"DEBUG: Requesting {max_results} events for calendar '{calendar_id}'")
            
            svc = _calendar_service()
            timeMin, timeMax, tzname = _iso_bounds_today_local()
            
            print(f"DEBUG: Time bounds - Min: {timeMin}, Max: {timeMax}")
            print(f"DEBUG: Timezone: {tzname}")
            
            resp = svc.events().list(
                calendarId=calendar_id,
                timeMin=timeMin,
                timeMax=timeMax,
                singleEvents=True,
                orderBy="startTime",
                timeZone=tzname,
                maxResults=max_results,
            ).execute()
            
            items = resp.get("items", [])
            result = [_event_item(e) for e in items]
            print(f"DEBUG: Successfully processed {len(result)} events for {calendar_id}")
            return result
            
        except Exception as e:
            print(f"DEBUG: Single calendar list_today_events FAILED: {e}")
            raise


def list_tomorrow_events(max_results: int = 10, calendar_id: str = "primary") -> List[Dict]:
    """Calendar events for tomorrow, local day bounds."""
    try:
        svc = _calendar_service()
        now = datetime.now(DEFAULT_TZ)
        tomorrow = now + timedelta(days=1)
        start_of_day = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = tomorrow.replace(hour=23, minute=59, second=59, microsecond=999999)
        timeMin = start_of_day.isoformat()
        timeMax = end_of_day.isoformat()
        tzname = str(DEFAULT_TZ)

        resp = svc.events().list(
            calendarId=calendar_id,
            timeMin=timeMin,
            timeMax=timeMax,
            singleEvents=True,
            orderBy="startTime",
            timeZone=tzname,
            maxResults=max_results,
        ).execute()
        items = resp.get("items", [])
        return [_event_item(e) for e in items]
    except Exception as e:
        print(f"DEBUG: list_tomorrow_events error: {e}")
        raise


def search_calendar(query: str) -> List[Dict]:
    """Search calendar events by text."""
    try:
        svc = _calendar_service()
        resp = svc.events().list(
            calendarId="primary",
            q=query,
            singleEvents=True,
            orderBy="startTime",
            maxResults=20,
        ).execute()
        items = resp.get("items", [])
        return [_event_item(e) for e in items]
    except Exception as e:
        print(f"DEBUG: search_calendar error: {e}")
        raise


def get_next_meeting() -> Optional[Dict]:
    """Get the next upcoming calendar event."""
    try:
        svc = _calendar_service()
        now_iso = datetime.now(DEFAULT_TZ).isoformat()
        
        # Check all calendars for next meeting
        calendar_list = svc.calendarList().list().execute()
        calendars = calendar_list.get('items', [])
        
        all_upcoming = []
        
        for calendar in calendars:
            cal_id = calendar.get('id')
            cal_name = calendar.get('summary', 'Unnamed Calendar')
            
            # Skip holidays/birthdays
            if 'holiday' in cal_name.lower() or 'birthday' in cal_name.lower():
                continue
            
            try:
                resp = svc.events().list(
                    calendarId=cal_id,
                    timeMin=now_iso,
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=5,  # Get first few from each calendar
                ).execute()
                
                items = resp.get("items", [])
                for item in items:
                    event = _event_item(item)
                    event['calendar_name'] = cal_name
                    all_upcoming.append(event)
                    
            except Exception as e:
                print(f"DEBUG: Error getting next meeting from {cal_name}: {e}")
                continue
        
        if all_upcoming:
            # Sort by start time and return the earliest
            all_upcoming.sort(key=lambda x: x.get('start', ''))
            return all_upcoming[0]
        
        return None
        
    except Exception as e:
        print(f"DEBUG: get_next_meeting error: {e}")
        return None


def format_calendar_summary(events: List[Dict], title: str = "") -> str:
    """Format calendar events for display."""
    if not events:
        return f"{title}\n\nNo events scheduled.\n"
    
    lines = [title] if title else []
    lines.append("")
    
    current_date = None
    for event in events:
        event_date = event.get('date_formatted', '')
        event_time = event.get('time_formatted', event.get('start_formatted', ''))
        event_title = event.get('summary', 'Untitled Event')
        
        # Add date header if date changed
        if current_date != event_date and event_date:
            if current_date is not None:  # Add spacing between dates
                lines.append("")
            lines.append(f"📅 {event_date}")
            current_date = event_date
        
        lines.append(f"   {event_time} - {event_title}")
    
    return "\n".join(lines) + "\n"


def format_calendar_summary_enhanced(events: List[Dict], title: str = "") -> str:
    """
    Enhanced calendar summary that shows which calendar each event is from
    """
    if not events:
        return f"{title}\n\nNo events scheduled.\n"
    
    lines = [title] if title else []
    lines.append("")
    
    current_date = None
    for event in events:
        event_date = event.get('date_formatted', '')
        event_time = event.get('time_formatted', event.get('start_formatted', ''))
        event_title = event.get('summary', 'Untitled Event')
        calendar_name = event.get('calendar_name', '')
        
        # Add date header if date changed
        if current_date != event_date and event_date:
            if current_date is not None:  # Add spacing between dates
                lines.append("")
            lines.append(f"📅 {event_date}")
            current_date = event_date
        
        # Format the event line with calendar context
        if calendar_name and calendar_name != 'Primary':
            event_line = f"   {event_time} - {event_title} ({calendar_name})"
        else:
            event_line = f"   {event_time} - {event_title}"
        
        lines.append(event_line)
    
    return "\n".join(lines) + "\n"
