# utils/scraper.py
import trafilatura

def scrape_url(url: str) -> dict:
    """
    Fetch & extract main article text from a URL.
    Returns {ok, title, text, error}
    """
    try:
        downloaded = trafilatura.fetch_url(url, no_ssl=True, url_mime_type=None)
        if not downloaded:
            return {"ok": False, "title": "", "text": "", "error": "Fetch failed"}
        extracted = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_recall=True
        )
        if not extracted:
            return {"ok": False, "title": "", "text": "", "error": "Could not extract main content"}
        return {"ok": True, "title": "", "text": extracted, "error": None}
    except Exception as e:
        return {"ok": False, "title": "", "text": "", "error": str(e)}
