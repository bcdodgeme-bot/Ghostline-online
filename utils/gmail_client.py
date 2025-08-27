# utils/gmail_client.py - Debug-enhanced version
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

    # Check if credentials are valid
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print("DEBUG: Refreshing expired credentials...")
                creds.refresh(Request())
                print("DEBUG: Credentials refreshed successfully")
                
                # Save refreshed credentials
                with open(TOKEN_PATH, "w") as f:
                    f.write(creds.to_json())
                print("DEBUG: Refreshed token saved")
                
            except Exception as e:
                print(f"DEBUG: Failed to refresh credentials: {e}")
                creds = None
        
        # If we still don't have valid credentials, show error
        if not creds:
            print("DEBUG: No valid credentials available")
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Missing Google OAuth credentials at '{CREDENTIALS_PATH}'. "
                    "Set GOOGLE_CREDENTIALS_PATH or place credentials.json."
                )
            
            # Don't try to create new creds in server environment
            raise RuntimeError(
                "No valid token available. Please regenerate token.json locally and upload it."
            )

    print("DEBUG: Final credentials check - valid:", creds.valid)
    return creds


def _gmail_service():
    """Build Gmail service with shared credentials"""
    try:
        print("DEBUG: Creating Gmail service...")
        creds = _build_creds()
        service = build("gmail", "v1", credentials=creds)
        print("DEBUG: Gmail service created successfully")
        return service
    except Exception as e:
        print(f"DEBUG: Failed to create Gmail service: {type(e).__name__}: {e}")
        raise


def _calendar_service():
    """Build Calendar service with shared credentials"""
    try:
        print("DEBUG: Creating Calendar service...")
        creds = _build_creds()
        service = build("calendar", "v3", credentials=creds)
        print("DEBUG: Calendar service created successfully")
        return service
    except Exception as e:
        print(f"DEBUG: Failed to create Calendar service: {type(e).__name__}: {e}")
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
            try:
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
                    
            except HttpError as e:
                print(f"DEBUG: Gmail API HttpError: Status {e.resp.status}, Reason: {e.resp.reason}")
                print(f"DEBUG: Gmail API Error content: {e.content}")
                raise
        
        print(f"DEBUG: Total Gmail messages found: {len(messages)}")
        return messages
        
    except Exception as e:
        print(f"DEBUG: Gmail list error - Type: {type(e).__name__}, Details: {e}")
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
        print(f"DEBUG: list_overnight error - Type: {type(e).__name__}, Details: {e}")
        raise


def search(query: str) -> List[Dict]:
    """Generic Gmail search; returns list of {id, threadId} dicts."""
    try:
        msgs = _gmail_list(query)
        result = [{"id": m.id, "threadId": m.thread_id} for m in msgs]
        print(f"DEBUG: Gmail search returning {len(result)} messages")
        return result
    except Exception as e:
        print(f"DEBUG: Gmail search error - Type: {type(e).__name__}, Details: {e}")
        raise


# --------------------------- Calendar helpers ---------------------------

def _iso_bounds_today_local():
    start = datetime.now(DEFAULT_TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
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
    }


def list_today_events(max_results: int = 10, calendar_id: str = "primary") -> List[Dict]:
    """Calendar events from local today 00:00 to tomorrow 00:00, tz-aware."""
    try:
        print("DEBUG: === Starting list_today_events ===")
        print(f"DEBUG: Requesting {max_results} events for calendar '{calendar_id}'")
        
        svc = _calendar_service()
        timeMin, timeMax, tzname = _iso_bounds_today_local()
        
        print(f"DEBUG: Time bounds - Min: {timeMin}, Max: {timeMax}")
        print(f"DEBUG: Timezone: {tzname}")
        
        # Make the actual API call with detailed error handling
        try:
            print("DEBUG: Making Calendar API call...")
            resp = svc.events().list(
                calendarId=calendar_id,
                timeMin=timeMin,
                timeMax=timeMax,
                singleEvents=True,
                orderBy="startTime",
                timeZone=tzname,
                maxResults=max_results,
            ).execute()
            print("DEBUG: Calendar API call successful")
            
        except HttpError as he:
            print(f"DEBUG: Calendar API HttpError Details:")
            print(f"DEBUG: - Status Code: {he.resp.status}")
            print(f"DEBUG: - Reason: {he.resp.reason}")
            print(f"DEBUG: - Error Content: {he.content}")
            
            # Try to parse the error content for more details
            try:
                error_details = json.loads(he.content.decode('utf-8'))
                print(f"DEBUG: - Parsed Error: {error_details}")
            except:
                pass
                
            raise
        
        items = resp.get("items", [])
        result = [_event_item(e) for e in items]
        print(f"DEBUG: Successfully processed {len(result)} events for today")
        print("DEBUG: === Completed list_today_events ===")
        return result
        
    except Exception as e:
        print(f"DEBUG: list_today_events FAILED - Type: {type(e).__name__}")
        print(f"DEBUG: Error details: {str(e)}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        raise


def list_tomorrow_events(max_results: int = 10, calendar_id: str = "primary") -> List[Dict]:
    """Calendar events for tomorrow, local day bounds."""
    try:
        print("DEBUG: Getting tomorrow's calendar events...")
        svc = _calendar_service()
        timeMin, timeMax, tzname = _iso_bounds_tomorrow_local()
        
        print(f"DEBUG: Tomorrow bounds - Min: {timeMin}, Max: {timeMax}")
        
        try:
            resp = svc.events().list(
                calendarId=calendar_id,
                timeMin=timeMin,
                timeMax=timeMax,
                singleEvents=True,
                orderBy="startTime",
                timeZone=tzname,
                maxResults=max_results,
            ).execute()
            
        except HttpError as he:
            print(f"DEBUG: Tomorrow events HttpError: Status {he.resp.status}, Content: {he.content}")
            raise
        
        items = resp.get("items", [])
        result = [_event_item(e) for e in items]
        print(f"DEBUG: Found {len(result)} events for tomorrow")
        return result
        
    except Exception as e:
        print(f"DEBUG: list_tomorrow_events error - Type: {type(e).__name__}, Details: {e}")
        raise


def search_calendar(query: str, days_ahead: int = 90, max_results: int = 50, calendar_id: str = "primary") -> List[Dict]:
    """Free-text search in upcoming window."""
    try:
        print(f"DEBUG: Searching calendar for: '{query}'")
        svc = _calendar_service()
        start = datetime.now(DEFAULT_TZ)
        end = start + timedelta(days=days_ahead)
        
        try:
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
            
        except HttpError as he:
            print(f"DEBUG: Calendar search HttpError: Status {he.resp.status}, Content: {he.content}")
            raise
        
        items = resp.get("items", [])
        result = [_event_item(e) for e in items]
        print(f"DEBUG: Calendar search found {len(result)} events")
        return result
        
    except Exception as e:
        print(f"DEBUG: search_calendar error - Type: {type(e).__name__}, Details: {e}")
        raise


def get_next_meeting(calendar_id: str = "primary") -> Dict:
    """Return the next upcoming meeting after now."""
    try:
        print("DEBUG: Getting next meeting...")
        svc = _calendar_service()
        now = datetime.now(DEFAULT_TZ)
        
        try:
            resp = svc.events().list(
                calendarId=calendar_id,
                timeMin=now.isoformat(),
                singleEvents=True,
                orderBy="startTime",
                timeZone=str(DEFAULT_TZ),
                maxResults=1,
            ).execute()
            
        except HttpError as he:
            print(f"DEBUG: Next meeting HttpError: Status {he.resp.status}, Content: {he.content}")
            raise
        
        items = resp.get("items", [])
        result = _event_item(items[0]) if items else {}
        print(f"DEBUG: Next meeting: {result.get('summary', 'None')}")
        return result
        
    except Exception as e:
        print(f"DEBUG: get_next_meeting error - Type: {type(e).__name__}, Details: {e}")
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
