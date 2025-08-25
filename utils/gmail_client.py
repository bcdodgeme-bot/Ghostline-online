# gmail_client.py
# Read-only Gmail helper with timezone-aware "overnight" query.
# Safe to drop in. Does NOT touch any brain modules.

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo  # stdlib (no pytz dependency)

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---- Scopes ----
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# ---- Config (env var–friendly) ----
DEFAULT_TZ = os.getenv("APP_TIMEZONE", "America/New_York")  # change if you want
TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")


@dataclass
class GmailMessage:
    id: str
    thread_id: str


class GmailClient:
    """
    Minimal Gmail helper:
      - manages OAuth token
      - lists messages matching a query
      - provides 'overnight' convenience using local midnight in your timezone
    """

    def __init__(
        self,
        credentials_path: str = CREDENTIALS_PATH,
        token_path: str = TOKEN_PATH,
        scopes: List[str] = SCOPES,
        timezone: str = DEFAULT_TZ,
    ):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = scopes
        self.tz = ZoneInfo(timezone)
        self.service = self._ensure_service()

    # ---------- Public API ----------

    def list_messages(self, query: str, user_id: str = "me", max_pages: int = 10) -> List[GmailMessage]:
        """
        List messages for an arbitrary Gmail search query.
        Handles pagination.
        """
        try:
            messages: List[GmailMessage] = []
            page_token: Optional[str] = None
            pages = 0

            while True:
                call = (
                    self.service.users()
                    .messages()
                    .list(userId=user_id, q=query, pageToken=page_token, maxResults=100)
                )
                resp = call.execute()
                for m in resp.get("messages", []):
                    messages.append(GmailMessage(id=m["id"], thread_id=m["threadId"]))

                page_token = resp.get("nextPageToken")
                pages += 1
                if not page_token or pages >= max_pages:
                    break

            return messages

        except HttpError as e:
            # Surface the query in the error for easier debugging
            raise RuntimeError(f"Gmail list failed for query='{query}': {e}") from e

    def list_overnight_messages(
        self,
        include_unread: bool = False,
        include_primary: bool = False,
        user_id: str = "me",
        query_extra: Optional[str] = None,
    ) -> List[GmailMessage]:
        """
        Return messages that arrived after **local midnight** today,
        using a date-based Gmail query. Example: 'after:2025/08/25 in:inbox'

        Args:
          include_unread: if True, append 'is:unread'
          include_primary: if True, append 'category:primary'
          query_extra: optional extra terms to append (e.g., '-from:noreply@…')
        """
        query = self._build_overnight_query(
            include_unread=include_unread,
            include_primary=include_primary,
            query_extra=query_extra,
        )
        return self.list_messages(query=query, user_id=user_id)

    # ---------- Internals ----------

    def _build_overnight_query(
        self,
        include_unread: bool,
        include_primary: bool,
        query_extra: Optional[str],
    ) -> str:
        """
        Builds a Gmail-compatible date query using local midnight.

        Gmail understands 'after:YYYY/MM/DD' as midnight at the start of that date.
        We combine with 'in:inbox' by default (broad but practical).
        """
        now = datetime.now(self.tz)
        local_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # Gmail date format must be YYYY/MM/DD
        after_str = local_midnight.strftime("%Y/%m/%d")

        parts = [f"after:{after_str}", "in:inbox"]
        if include_unread:
            parts.append("is:unread")
        if include_primary:
            parts.append("category:primary")
        if query_extra:
            parts.append(query_extra.strip())

        # Final query string
        return " ".join(parts)

    def _ensure_service(self):
        """
        Loads/refreshes OAuth token and returns a Gmail API service client.
        Honors env var paths and keeps tokens out of your repo.
        """
        creds = None
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_path):
                    raise FileNotFoundError(
                        f"Missing Google OAuth credentials file at '{self.credentials_path}'. "
                        "Set GOOGLE_CREDENTIALS_PATH or place credentials.json beside the app."
                    )
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.scopes)
                creds = flow.run_local_server(port=0)

            # Save the token for next runs (keep token.json out of git)
            with open(self.token_path, "w") as token:
                token.write(creds.to_json())

        return build("gmail", "v1", credentials=creds)


# ---------- Convenience entry points ----------

def get_overnight_messages(
    include_unread: bool = False,
    include_primary: bool = False,
    query_extra: Optional[str] = None,
    timezone: str = DEFAULT_TZ,
) -> List[GmailMessage]:
    """
    One-shot helper if you don’t want to manage the class yourself.
    """
    client = GmailClient(timezone=timezone)
    return client.list_overnight_messages(
        include_unread=include_unread,
        include_primary=include_primary,
        query_extra=query_extra,
    )


if __name__ == "__main__":
    # Quick manual test (prints count + first few IDs)
    msgs = get_overnight_messages(include_unread=False, include_primary=False)
    print(f"Overnight messages: {len(msgs)}")
    for m in msgs[:10]:
        print("-", m.id)
