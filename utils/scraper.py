# utils/scraper.py
from __future__ import annotations

import requests
from typing import Dict, Any

# Primary text extractor
try:
    import trafilatura  # already in your requirements
except Exception:  # keep the module import non-fatal
    trafilatura = None

# Optional fallback (readability-lxml)
try:
    from readability import Document
except Exception:
    Document = None


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_url(url: str, timeout: float = 15.0) -> str:
    """
    Fetch raw HTML (no extra kwargs like url_mime_type).
    Raises requests.HTTPError on bad status.
    """
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    # Heuristic: only accept text-like responses
    ctype = resp.headers.get("Content-Type", "").lower()
    if "text" not in ctype and "html" not in ctype:
        # Still return text if server lies about content-type but it "looks" like HTML
        text = resp.text or ""
        if "<html" not in text.lower():
            raise ValueError(f"Unsupported content type: {ctype or 'unknown'}")
    return resp.text or ""


def _extract_with_trafilatura(html: str, url: str | None = None) -> str | None:
    if not trafilatura:
        return None
    try:
        # Trafilatura works best when given the URL for context
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_recall=True,  # a bit more liberal in extraction
        )
        return (text or "").strip() or None
    except Exception:
        return None


def _extract_with_readability(html: str) -> str | None:
    if not Document:
        return None
    try:
        doc = Document(html)
        content_html = doc.summary(html_partial=True) or ""
        # quick & small HTML->text ; keeps it dependency-light
        # (you already ship BeautifulSoup via trafilatura stack, but avoid importing here)
        # fall back to a crude strip if BS4 isn't available
        try:
            from bs4 import BeautifulSoup  # type: ignore
            text = BeautifulSoup(content_html, "lxml").get_text("\n")
        except Exception:
            # super simple tag strip
            import re
            text = re.sub(r"<[^>]+>", "", content_html)
        return (text or "").strip() or None
    except Exception:
        return None


def scrape_url(url: str) -> Dict[str, Any]:
    """
    Fetches a URL and extracts readable text.
    Returns: {"ok": True, "text": "...", "url": url} on success,
             {"ok": False, "error": "..."} on failure.
    """
    try:
        html = fetch_url(url)
    except Exception as e:
        return {"ok": False, "error": f"fetch failed: {e}"}

    # Try trafilatura first, then readability as a fallback
    text = _extract_with_trafilatura(html, url=url)
    if not text:
        text = _extract_with_readability(html)

    if text:
        # Trim very long pages to something sane for a prompt
        text = text.strip()
        if len(text) > 60_000:
            text = text[:60_000] + "\n\n[...truncated...]"
        return {"ok": True, "text": text, "url": url}

    return {"ok": False, "error": "could not extract readable text"}