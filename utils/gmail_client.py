# utils/gmail_client.py
from __future__ import annotations
import os, datetime, base64, email
from typing import List, Dict, Any
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
SECRETS_DIR = "secrets"
CREDS_PATH = os.path.join(SECRETS_DIR, "gmail_credentials.json")
TOKEN_PATH = os.path.join(SECRETS_DIR, "gmail_token.json")

def _ensure_secrets_dir():
    os.makedirs(SECRETS_DIR, exist_ok=True)

def get_service():
    _ensure_secrets_dir()
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Local flow launches a browser ON YOUR MACHINE only
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def _headers_to_dict(payload_headers):
    d = {}
    for h in payload_headers or []:
        d[h.get("name", "").lower()] = h.get("value", "")
    return d

def list_messages(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    svc = get_service()
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
    # Since local midnight (your machine’s timezone)
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