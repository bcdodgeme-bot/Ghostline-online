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

# Read-only scopes
SCOPES_GMAIL = ["https://www.googleapis.com/auth/gmail.readonly"]
SCOPES_CAL = ["https://www.googleapis.com/auth/calendar.readonly"]


# --------------------------- Auth helpers ---------------------------

def _build_creds(scopes: List[str]) -> Credentials:
    creds: Optional[Credentials] = None
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, scopes)
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Missing Google OAuth credentials at '{CREDENTIALS_PATH}'. "
                    "Set GOOGLE_CREDENTIALS_PATH or place credentials.json."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, scopes)
            creds = flow.run_local_server(port=0)

        # Save/refresh token (keep out of git)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return creds


def _gmail_service():
    creds = _build_creds(SCOPES_GMAIL)
    return build("gmail", "v1", credentials=creds)


def _calendar_service():
    creds = _build_creds(SCOPES_CAL)
    return build("calendar", "v3", credentials=creds)


# --------------------------- Gmail helpers ---------------------------

@dataclass
class GmailMessage:
    id: str
    thread_id: str

def _gmail_list(query: str, user_id: str = "me", max_pages: int = 10) -> List[GmailMessage]:
    svc = _gmail_service()
    messages: List[GmailMessage] = []
    page_token: Optional[str] = None
    pages = 0
    while True:
        resp = svc.users().messages().list(
            userId=user_id, q=query, pageToken=page_token, maxResults=100
        ).execute()
        for m in resp.get("messages", []):
            messages.append(GmailMessage(id=m["id"], thread_id=m["threadId"]))
        page_token = resp.get("nextPageToken")
        pages += 1
        if not page_token or pages >= max_pages:
            break
    return messages


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
    return " ".join(parts)


# --- Backward-compatible: used by app.py ---

def list_overnight(include_unread: bool = False, include_primary: bool = False, query_extra: Optional[str] = None) -> List[Dict]:
    """Return list of message dicts since local midnight. (App uses len() only.)"""
    q = _overnight_query(include_unread, include_primary, query_extra)
    msgs = _gmail_list(q)
    # Return simple dicts (id only is fine; app counts them)
    return [{"id": m.id, "threadId": m.thread_id} for m in msgs]


def search(query: str) -> List[Dict]:
    """Generic Gmail search; returns list of {id, threadId} dicts."""
    msgs = _gmail_list(query)
    return [{"id": m.id, "threadId": m.thread_id} for m in msgs]


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
    svc = _calendar_service()
    timeMin, timeMax, tzname = _iso_bounds_today_local()
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


def list_tomorrow_events(max_results: int = 10, calendar_id: str = "primary") -> List[Dict]:
    """Calendar events for tomorrow, local day bounds."""
    svc = _calendar_service()
    timeMin, timeMax, tzname = _iso_bounds_tomorrow_local()
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


def search_calendar(query: str, days_ahead: int = 90, max_results: int = 50, calendar_id: str = "primary") -> List[Dict]:
    """Free-text search in upcoming window."""
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
    return [_event_item(e) for e in items]


def get_next_meeting(calendar_id: str = "primary") -> Dict:
    """Return the next upcoming meeting after now."""
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
    return _event_item(items[0]) if items else {}


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

