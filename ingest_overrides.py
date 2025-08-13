#!/usr/bin/env python3
# ingest_overrides.py
import os, re, time, hashlib, json
from pathlib import Path
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

try:
    import feedparser  # RSS/Atom
except ImportError:
    feedparser = None

# Optional for YouTube transcripts (only used if you supply video URLs file)
try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
except Exception:
    YouTubeTranscriptApi = None

HEADERS = {"User-Agent": "GhostlineBot/0.1 (+contact: you@example.com)"}
TIMEOUT = 25
SLEEP = 0.4

RAW_ROOT = Path("data/raw_overrides")
LOGS_ROOT = Path("logs")
RAW_ROOT.mkdir(parents=True, exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

# --------- YOUR OVERRIDES (as requested) ----------
OVERRIDES = {
    "moz.com": {
        "rss": "https://moz.com/blog/feed",
    },
    "garyvaynerchuk.com": {
        "use": "youtube_transcripts"  # supply URLs via: sources_youtube/garyvaynerchuk.txt
    },
    "markmanson.net": {
        "use": "persona_only"  # skip scraping; tone from allowed sources later
    },
    "neilpatel.com": {
        "rss": "https://neilpatel.com/blog/feed",
    },
    "www.socialmediaexaminer.com": {
        "rss": "https://www.socialmediaexaminer.com/feed/",
    },
    "creativecommons.org": {
        "extra_urls": [
            "https://creativecommons.org/licenses/"
        ]
    },
    "www.copyright.gov": {
        "alt_sources": [
            "https://fairuse.stanford.edu/overview/fair-use/"
        ]
    },
    "www.sec.gov": {
        "use": "edgar_api"  # placeholder; we’ll wire a filings loader later
    },
    "cointelegraph.com": {
        "rss": "https://cointelegraph.com/rss"
    },
    "www.fool.com": {
        "rss": "https://www.fool.com/feeds/index.aspx"
    },
    "www.nerdfitness.com": {
        "rss": "https://www.nerdfitness.com/blog/feed/"
    },
    "www.health.harvard.edu": {
        "rss": "https://www.health.harvard.edu/blog/feed"
    },
    "www.nasm.org": {
        "extra_urls": [
            "https://www.nasm.org/certified-personal-trainer/terms-and-definitions"
        ]
    },
}

# If you want to feed YouTube video URLs, put files here, eg:
# sources_youtube/garyvaynerchuk.txt  (one https://www.youtube.com/watch?v=... per line)
YOUTUBE_URL_LIST_DIR = Path("sources_youtube")

# --------------------------------------------------

def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def site_from_url(url: str) -> str:
    return urlparse(url).netloc

def save_text(dest_dir: Path, url: str, text: str, ext: str = ".html"):
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    tail = re.sub(r"[^a-zA-Z0-9]+", "_", url[-80:]).strip("_")
    fname = f"{h}_{tail}{ext}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / fname).write_text(text, encoding="utf-8")
    return dest_dir / fname

def get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r

def fetch_and_save_html(dest_dir: Path, url: str, logf):
    try:
        r = get(url)
        # Save raw HTML as-is (no prettify to keep speed)
        p = save_text(dest_dir, url, r.text, ext=".html")
        logf.write(f"SAVED {url} -> {p.name}\n")
        return True
    except Exception as e:
        logf.write(f"ERR   {url} :: {e}\n")
        return False

def handle_rss(domain: str, rss_url: str, dest_dir: Path, logf):
    if feedparser is None:
        logf.write(f"ERR   RSS {rss_url} :: feedparser not installed\n")
        return 0
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        logf.write(f"ERR   RSS {rss_url} :: {e}\n")
        return 0

    saved = 0
    for entry in feed.entries:
        link = getattr(entry, "link", "") or ""
        if not link.startswith("http"):
            continue
        if site_from_url(link) not in (domain, "www." + domain, domain.replace("www.","")):
            # keep it simple: only same-site links
            pass
        ok = fetch_and_save_html(dest_dir, link, logf)
        if ok:
            saved += 1
        time.sleep(SLEEP)
    return saved

def extract_youtube_id(url: str) -> str | None:
    # Handles typical formats: https://www.youtube.com/watch?v=ID  or youtu.be/ID
    m = re.search(r"v=([A-Za-z0-9_\-]{6,})", url)
    if m:
        return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_\-]{6,})", url)
    if m:
        return m.group(1)
    return None

def handle_youtube_transcripts(label: str, dest_dir: Path, logf):
    """
    Looks for sources_youtube/<label>.txt with one video URL per line.
    Saves transcripts as .txt if available (language auto-detected).
    """
    if YouTubeTranscriptApi is None:
        logf.write("ERR   YouTube transcripts unavailable (youtube_transcript_api not installed)\n")
        return 0

    YOUTUBE_URL_LIST_DIR.mkdir(exist_ok=True, parents=True)
    candidates = [
        YOUTUBE_URL_LIST_DIR / f"{label}.txt",
        YOUTUBE_URL_LIST_DIR / f"{label}-videos.txt",
    ]
    urls_file = None
    for c in candidates:
        if c.exists():
            urls_file = c
            break

    if not urls_file:
        logf.write(f"INFO  YouTube list not found for {label}. Create {candidates[0].as_posix()} with video URLs.\n")
        return 0

    saved = 0
    urls = [ln.strip() for ln in urls_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for u in urls:
        vid = extract_youtube_id(u)
        if not vid:
            logf.write(f"SKIP  Not a video URL: {u}\n")
            continue
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(vid)
            # Prefer English if present, else first available
            try:
                t = transcripts.find_transcript(['en'])
            except Exception:
                t = next(iter(transcripts))
            parts = t.fetch()
            text = "\n".join([p['text'] for p in parts if p.get('text')])
            # Save transcript
            fname = f"yt_{vid}.txt"
            (dest_dir / fname).write_text(text, encoding="utf-8")
            logf.write(f"SAVED YT {u} -> {fname}\n")
            saved += 1
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            logf.write(f"MISS  YT {u} :: {e}\n")
        except Exception as e:
            logf.write(f"ERR   YT {u} :: {e}\n")
        time.sleep(SLEEP)
    return saved

def main():
    total_saved = 0
    for domain, rule in OVERRIDES.items():
        slug = slugify(domain)
        dest_dir = RAW_ROOT / slug
        log_path = LOGS_ROOT / f"{slug}_overrides.log"
        dest_dir.mkdir(parents=True, exist_ok=True)

        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"== {domain} ==\n")

            # 1) RSS
            rss_url = rule.get("rss")
            if rss_url:
                logf.write(f"INFO  RSS -> {rss_url}\n")
                saved = handle_rss(domain, rss_url, dest_dir, logf)
                logf.write(f"INFO  RSS saved: {saved}\n")
                total_saved += saved

            # 2) extra_urls
            for u in rule.get("extra_urls", []):
                logf.write(f"INFO  EXTRA -> {u}\n")
                ok = fetch_and_save_html(dest_dir, u, logf)
                if ok: total_saved += 1

            # 3) alt_sources (treat like extra for now)
            for u in rule.get("alt_sources", []):
                logf.write(f"INFO  ALT  -> {u}\n")
                ok = fetch_and_save_html(dest_dir, u, logf)
                if ok: total_saved += 1

            # 4) special use handlers
            use = rule.get("use")
            if use == "youtube_transcripts":
                # label derived from domain first label, e.g., garyvaynerchuk
                label = domain.split(".")[0]
                logf.write(f"INFO  YouTube transcripts for label: {label}\n")
                saved = handle_youtube_transcripts(label, dest_dir, logf)
                logf.write(f"INFO  YT transcripts saved: {saved}\n")
                total_saved += saved
            elif use == "persona_only":
                logf.write("INFO  Persona-only: skipping scrape (tone will be captured elsewhere)\n")
            elif use == "edgar_api":
                logf.write("TODO  SEC EDGAR loader not wired in this script. (We’ll add a filings ingester later.)\n")

    print(f"Done. Saved total: {total_saved}. Output in {RAW_ROOT}")

if __name__ == "__main__":
    main()
