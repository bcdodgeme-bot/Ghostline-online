# utils/gmail_client.py
# Gmail + Google Calendar helpers
# - Backward-compatible exports for app.py:
#   list_overnight, search, list_today_events, list_tomorrow_events,
#   search_calendar, get_next_meeting, format_calendar_summary
# - Timezone-aware using America/New_York by default (override with APP_TIMEZONE)
# - Secrets loaded from env paths (keep token/credentials out of git)

from __future__ import annotations

import os
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
    
    # Try to load existing credentials
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, ALL_SCOPES)
        except Exception as e:
            print(f"DEBUG: Failed to load existing token: {e}")
            creds = None

    # Check if credentials are valid
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print("DEBUG: Refreshing expired credentials...")
                creds.refresh(Request())
                print("DEBUG: Credentials refreshed successfully")
            except Exception as e:
                print(f"DEBUG: Failed to refresh credentials: {e}")
                creds = None
        
        # If we still don't have valid credentials, create new ones
        if not creds:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Missing Google OAuth credentials at '{CREDENTIALS_PATH}'. "
                    "Set GOOGLE_CREDENTIALS_PATH or place credentials.json."
                )
            print("DEBUG: Creating new credentials...")
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, ALL_SCOPES)
            creds = flow.run_local_server(port=0)
            print("DEBUG: New credentials created successfully")

        # Save the credentials
        try:
            with open(TOKEN_PATH, "w") as f:
                f.write(creds.to_json())
            print(f"DEBUG: Credentials saved to {TOKEN_PATH}")
        except Exception as e:
            print(f"DEBUG: Failed to save credentials: {e}")

    return creds


def _gmail_service():
    """Build Gmail service with shared credentials"""
    try:
        creds = _build_creds()
        service = build("gmail", "v1", credentials=creds)
        print("DEBUG: Gmail service created successfully")
        return service
    except Exception as e:
        print(f"DEBUG: Failed to create Gmail service: {e}")
        raise


def _calendar_service():
    """Build Calendar service with shared credentials"""
    try:
        creds = _build_creds()
        service = build("calendar", "v3", credentials=creds)
        print("DEBUG: Calendar service created successfully")
        return service
    except Exception as e:
        print(f"DEBUG: Failed to create Calendar service: {e}")
        raise


# --------------------------- Gmail helpers ---------------------------

@dataclass
class GmailMessage:
    id: str
    thread_id: str

def _gmail_list(query: str, user_id: str = "me", max_pages: int = 10) -> List[GmailMessage]:
    try:
        svc = _gmail_service()
        messages: List[GmailMessage] = []
        page_token: Optional[str] = None
        pages = 0
        
        print(f"DEBUG: Gmail search query: '{query}'")
        
        while True:
            resp = svc.users().messages().list(
                userId=user_id, q=query, pageToken=page_token, maxResults=100
            ).execute()
            
            message_batch = resp.get("messages", [])
            print(f"DEBUG: Found {len(message_batch)} messages in this batch")
            
            for m in message_batch:
                messages.append(GmailMessage(id=m["id"], thread_id=m["threadId"]))
            
            page_token = resp.get("nextPageToken")
            pages += 1
            if not page_token or pages >= max_pages:
                break
        
        print(f"DEBUG: Total Gmail messages found: {len(messages)}")
        return messages
        
    except Exception as e:
        print(f"DEBUG: Gmail list error: {e}")
        raise


def _overnight_query(include_unread: bool, include_primary: bool, query_extra: Optional[str]) -> str:
    # Local midnight today in DEFAULT_TZ; Gmail date format YYYY/MM/DD
    today_start = datetime.now(DEFAULT_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    after_str = today_start.strftime("%Y/%m/%d")
    parts = [f"after:{after_str}", "in:inbox"]
    if include_unread:
        parts.append("is:unread")
    if include_primary:
        parts.append("category:primary")
    if query_extra:
        parts.append(query_extra.strip())
    query = " ".join(parts)
    print(f"DEBUG: Overnight query: '{query}'")
    return query


# --- Backward-compatible: used by app.py ---

def list_overnight(include_unread: bool = False, include_primary: bool = False, query_extra: Optional[str] = None) -> List[Dict]:
    """Return list of message dicts since local midnight. (App uses len() only.)"""
    try:
        q = _overnight_query(include_unread, include_primary, query_extra)
        msgs = _gmail_list(q)
        result = [{"id": m.id, "threadId": m.thread_id} for m in msgs]
        print(f"DEBUG: list_overnight returning {len(result)} messages")
        return result
    except Exception as e:
        print(f"DEBUG: list_overnight error: {e}")
        raise


def search(query: str) -> List[Dict]:
    """Generic Gmail search; returns list of {id, threadId} dicts."""
    try:
        msgs = _gmail_list(query)
        result = [{"id": m.id, "threadId": m.thread_id} for m in msgs]
        print(f"DEBUG: Gmail search returning {len(result)} messages")
        return result
    except Exception as e:
        print(f"DEBUG: Gmail search error: {e}")
        raise


# --------------------------- Calendar helpers ---------------------------

def _iso_bounds_today_local():
    start = datetime.now(DEFAULT_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    # RFC3339 with offset (ZoneInfo ensures offset present)
    return start.isoformat(), end.isoformat(), str(DEFAULT_TZ)


def _iso_bounds_tomorrow_local():
    start = (datetime.now(DEFAULT_TZ).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat(), str(DEFAULT_TZ)


def _to_local(dt_str: str) -> Optional[datetime]:
    """Parse RFC3339 string to tz-aware datetime in DEFAULT_TZ."""
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
    # e.g., 1:30 PM
    return dt.strftime("%-I:%M %p") if os.name != "nt" else dt.strftime("%#I:%M %p")


def _event_item(e: Dict) -> Dict:
    """
    Normalize a Calendar event to what app.py expects:
    - 'summary': event title (fallback to '(No title)')
    - 'start': ISO string (start.dateTime or start.date)
    - 'start_formatted': local time string (empty for all-day)
    """
    summary = e.get("summary") or "(No title)"
    start = e.get("start", {})
    start_iso = start.get("dateTime") or start.get("date")  # all-day is date only
    start_dt_local = _to_local(start_iso) if start_iso else None
    start_formatted = _format_time_local(start_dt_local) if start_dt_local else ("All day" if start.get("date") else "")
    return {
        "id": e.get("id"),
        "summary": summary,
        "start": start_iso,
        "start_formatted": start_formatted,
    }


def list_today_events(max_results: int = 10, calendar_id: str = "primary") -> List[Dict]:
    """Calendar events from local today 00:00 to tomorrow 00:00, tz-aware."""
    try:
        print("DEBUG: Getting today's calendar events...")
        svc = _calendar_service()
        timeMin, timeMax, tzname = _iso_bounds_today_local()
        
        print(f"DEBUG: Calendar query - timeMin: {timeMin}, timeMax: {timeMax}, tz: {tzname}")
        
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
        print(f"DEBUG: Found {len(result)} events for today")
        return result
        
    except Exception as e:
        print(f"DEBUG: list_today_events error: {e}")
        raise


def list_tomorrow_events(max_results: int = 10, calendar_id: str = "primary") -> List[Dict]:
    """Calendar events for tomorrow, local day bounds."""
    try:
        print("DEBUG: Getting tomorrow's calendar events...")
        svc = _calendar_service()
        timeMin, timeMax, tzname = _iso_bounds_tomorrow_local()
        
        print(f"DEBUG: Tomorrow calendar query - timeMin: {timeMin}, timeMax: {timeMax}")
        
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
        print(f"DEBUG: Found {len(result)} events for tomorrow")
        return result
        
    except Exception as e:
        print(f"DEBUG: list_tomorrow_events error: {e}")
        raise


def search_calendar(query: str, days_ahead: int = 90, max_results: int = 50, calendar_id: str = "primary") -> List[Dict]:
    """Free-text search in upcoming window."""
    try:
        print(f"DEBUG: Searching calendar for: '{query}'")
        svc = _calendar_service()
        start = datetime.now(DEFAULT_TZ)
        end = start + timedelta(days=days_ahead)
        
        resp = svc.events().list(
            calendarId=calendar_id,
            q=query,
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            timeZone=str(DEFAULT_TZ),
            maxResults=max_results,
        ).execute()
        
        items = resp.get("items", [])
        result = [_event_item(e) for e in items]
        print(f"DEBUG: Calendar search found {len(result)} events")
        return result
        
    except Exception as e:
        print(f"DEBUG: search_calendar error: {e}")
        raise


def get_next_meeting(calendar_id: str = "primary") -> Dict:
    """Return the next upcoming meeting after now."""
    try:
        print("DEBUG: Getting next meeting...")
        svc = _calendar_service()
        now = datetime.now(DEFAULT_TZ)
        
        resp = svc.events().list(
            calendarId=calendar_id,
            timeMin=now.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            timeZone=str(DEFAULT_TZ),
            maxResults=1,
        ).execute()
        
        items = resp.get("items", [])
        result = _event_item(items[0]) if items else {}
        print(f"DEBUG: Next meeting: {result.get('summary', 'None')}")
        return result
        
    except Exception as e:
        print(f"DEBUG: get_next_meeting error: {e}")
        raise


def format_calendar_summary(events: List[Dict], header: str = "") -> str:
    """Format bullet list summary expected by app.py."""
    if not events:
        return "(No events)"
    lines = []
    if header:
        lines.append(header.strip())
    for e in events:
        lines.append(f"• {e.get('start_formatted','')} — {e.get('summary','')}".strip())
    return "\n".join(lines)
