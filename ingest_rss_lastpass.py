#!/usr/bin/env python3
# ingest_rss_lastpass.py
# One-shot RSS grab for: Cointelegraph, GaryV, Social Media Examiner, Mark Manson

import re, time, hashlib
from pathlib import Path
from urllib.parse import urlparse
import requests

try:
    import feedparser
except Exception:
    feedparser = None

HEADERS = {"User-Agent": "GhostlineRSS/0.3 (+contact: you@example.com)"}
TIMEOUT = 25
PAUSE = 0.4
MAX_PER_FEED = 500  # change if you want more/less

# Output
RAW_ROOT = Path("data/raw_lastpass")
LOG_ROOT = Path("logs/lastpass")
RAW_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)

FEEDS = {
    "cointelegraph.com": "https://cointelegraph.com/rss",
    "garyvaynerchuk.com": "https://garyvaynerchuk.com/feed/",
    "www.socialmediaexaminer.com": "https://www.socialmediaexaminer.com/feed/",
    "markmanson.net": "https://markmanson.net/feed",
}

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def save_html(dest_dir: Path, url: str, html: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    tail = re.sub(r"[^a-zA-Z0-9]+", "_", url[-90:]).strip("_")
    path = dest_dir / f"{h}_{tail}.html"
    path.write_text(html, encoding="utf-8")
    return path

def get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r

def fetch_and_save(dest_dir: Path, url: str, logf):
    try:
        r = get(url)
        p = save_html(dest_dir, url, r.text)
        logf.write(f"SAVED {url} -> {p.name}\n")
        return 1
    except Exception as e:
        logf.write(f"ERR   {url} :: {e}\n")
        return 0

def process_feed(domain: str, feed_url: str) -> int:
    dest = RAW_ROOT / slugify(domain)
    logp = LOG_ROOT / f"{slugify(domain)}.log"
    saved = 0

    with logp.open("w", encoding="utf-8") as logf:
        logf.write(f"== {domain} ==\n")
        if feedparser is None:
            logf.write("ERR   feedparser not installed. Run: pip install feedparser\n")
            return 0

        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            logf.write(f"ERR   parse {feed_url} :: {e}\n")
            return 0

        entries = getattr(feed, "entries", [])
        logf.write(f"INFO  feed {feed_url} entries: {len(entries)}\n")

        for i, entry in enumerate(entries[:MAX_PER_FEED], 1):
            link = getattr(entry, "link", "") or ""
            if not link.startswith("http"):
                continue
            saved += fetch_and_save(dest, link, logf)
            time.sleep(PAUSE)

        logf.write(f"INFO  saved total: {saved}\n")
    return saved

def main():
    total = 0
    for domain, feed in FEEDS.items():
        total += process_feed(domain, feed)
    print(f"Done. Saved {total} pages. Check data/raw_lastpass and logs/lastpass")

if __name__ == "__main__":
    main()
