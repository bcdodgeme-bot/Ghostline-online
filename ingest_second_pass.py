#!/usr/bin/env python3
# ingest_second_pass.py
"""
Second-pass intake for stubborn sources.
- Uses RSS/podcasts/transcripts/PDFs/APIs (no robots-busting)
- Saves into data/raw_second_pass/<slug>/ (keeps first scrape pristine)
"""

import os, re, time, json, hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# Optional deps
try:
    import feedparser
except Exception:
    feedparser = None

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        NoTranscriptFound,
        TranscriptsDisabled,
    )
except Exception:
    YouTubeTranscriptApi = None

# ---------- Paths ----------
RAW_ROOT = Path("data/raw_second_pass")
LOG_ROOT = Path("logs/second_pass")
YOUTUBE_LIST_DIR = Path("sources_youtube")
SEC_SOURCES = Path("sources_sec/companies.txt")

RAW_ROOT.mkdir(parents=True, exist_ok=True)
LOG_ROOT.mkdir(parents=True, exist_ok=True)
YOUTUBE_LIST_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Network ----------
HEADERS = {"User-Agent": "GhostlineSecondPass/0.2 (contact: you@example.com)"}
TIMEOUT = 25
PAUSE = 0.5

# ---------- Helpers ----------
def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")

def site_from_url(url: str) -> str:
    return urlparse(url).netloc

def save_text(dest_dir: Path, url: str, content: str, ext: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    tail = re.sub(r"[^a-zA-Z0-9]+", "_", url[-90:]).strip("_")
    path = dest_dir / f"{h}_{tail}{ext}"
    path.write_text(content, encoding="utf-8")
    return path

def save_bytes(dest_dir: Path, url: str, content: bytes, ext: str):
    dest_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    tail = re.sub(r"[^a-zA-Z0-9]+", "_", url[-90:]).strip("_")
    path = dest_dir / f"{h}_{tail}{ext}"
    path.write_bytes(content)
    return path

def http_get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r

def fetch_and_save_html(dest_dir: Path, url: str, logf) -> bool:
    try:
        r = http_get(url)
        p = save_text(dest_dir, url, r.text, ".html")
        logf.write(f"SAVED {url} -> {p.name}\n")
        return True
    except Exception as e:
        logf.write(f"ERR   {url} :: {e}\n")
        return False

def extract_youtube_id(url: str) -> str | None:
    m = re.search(r"v=([A-Za-z0-9_\-]{6,})", url)
    if m: return m.group(1)
    m = re.search(r"youtu\.be/([A-Za-z0-9_\-]{6,})", url)
    if m: return m.group(1)
    return None

def rss_fetch_articles(domain_slug: str, domain_host: str, feed_url: str, logf) -> int:
    if feedparser is None:
        logf.write(f"ERR   feedparser not installed for {feed_url}\n")
        return 0
    try:
        feed = feedparser.parse(feed_url)
    except Exception as e:
        logf.write(f"ERR   RSS parse {feed_url} :: {e}\n")
        return 0

    dest = RAW_ROOT / domain_slug
    saved = 0
    for entry in feed.entries:
        link = getattr(entry, "link", "") or ""
        if not link.startswith("http"):
            continue
        # Keep it simple: fetch whatever link points to (full article)
        ok = fetch_and_save_html(dest, link, logf)
        if ok: saved += 1
        time.sleep(PAUSE)
    logf.write(f"INFO  RSS saved: {saved} from {feed_url}\n")
    return saved

def podcast_transcript_like_fetch(domain_slug: str, feed_url: str, logf) -> int:
    """Fetch podcast episode pages (show notes)."""
    return rss_fetch_articles(domain_slug, site_from_url(feed_url), feed_url, logf)

def youtube_transcripts_from_list(label: str, domain_slug: str, logf) -> int:
    if YouTubeTranscriptApi is None:
        logf.write("ERR   youtube_transcript_api not installed\n")
        return 0
    # Find list file
    candidates = [
        YOUTUBE_LIST_DIR / f"{label}.txt",
        YOUTUBE_LIST_DIR / f"{label}-videos.txt",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if not path:
        logf.write(f"INFO  No YouTube list found for {label}. Create {candidates[0].as_posix()} (one URL per line).\n")
        return 0

    urls = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dest = RAW_ROOT / domain_slug
    saved = 0

    for u in urls:
        vid = extract_youtube_id(u)
        if not vid:
            logf.write(f"SKIP  Not a YouTube watch URL: {u}\n")
            continue
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(vid)
            try:
                t = transcripts.find_transcript(['en'])
            except Exception:
                t = next(iter(transcripts))
            parts = t.fetch()
            text = "\n".join(p['text'] for p in parts if p.get('text'))
            file_url = f"https://www.youtube.com/watch?v={vid}"
            out = save_text(dest, file_url, text, ".txt")
            logf.write(f"SAVED YT {file_url} -> {out.name}\n")
            saved += 1
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            logf.write(f"MISS  YT {u} :: {e}\n")
        except Exception as e:
            logf.write(f"ERR   YT {u} :: {e}\n")
        time.sleep(PAUSE)
    return saved

def save_pdf(url: str, dest_dir: Path, logf) -> bool:
    try:
        r = http_get(url)
        p = save_bytes(dest_dir, url, r.content, ".pdf")
        logf.write(f"SAVED PDF {url} -> {p.name}\n")
        return True
    except Exception as e:
        logf.write(f"ERR   PDF {url} :: {e}\n")
        return False

def sec_fetch_submissions(companies_file: Path, domain_slug: str, logf) -> int:
    """
    Pull JSON submissions for each CIK or ticker in companies.txt (one per line).
    Saves raw JSON; parsing/normalizing happens later in processing.
    """
    if not companies_file.exists():
        logf.write(f"INFO  No SEC companies list at {companies_file.as_posix()} (put CIK or ticker per line)\n")
        return 0

    # SEC guidance: include informative UA; rate limit ~10 requests/sec max (we're way below).
    saved = 0
    dest = RAW_ROOT / domain_slug
    tickers = [ln.strip().upper() for ln in companies_file.read_text().splitlines() if ln.strip()]

    # Simple ticker -> CIK lookup via SEC API
    def ticker_to_cik(tkr: str) -> str | None:
        try:
            r = http_get(f"https://www.sec.gov/files/company_tickers.json")
            mapping = r.json()
            # mapping is { "0": {"cik_str":..., "ticker":"AAPL", ...}, ... }
            for _, v in mapping.items():
                if v.get("ticker", "").upper() == tkr:
                    return str(v.get("cik_str", "")).zfill(10)
        except Exception:
            return None
        return None

    for ident in tickers:
        if ident.isdigit():
            cik = ident.zfill(10)
        else:
            cik = ticker_to_cik(ident)
            if not cik:
                logf.write(f"MISS  SEC could not resolve ticker {ident}\n")
                continue

        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            r = http_get(url)
            out = save_text(dest, url, json.dumps(r.json(), indent=2), ".json")
            logf.write(f"SAVED SEC submissions {cik} -> {out.name}\n")
            saved += 1
        except Exception as e:
            logf.write(f"ERR   SEC {cik} :: {e}\n")
        time.sleep(1.0)  # be gentle
    return saved

# ---------- Targets ----------
TARGETS = [
    # Cointelegraph (RSS -> article pages)
    {
        "slug": "cointelegraph-com",
        "action": lambda logf: rss_fetch_articles(
            "cointelegraph-com",
            "cointelegraph.com",
            "https://cointelegraph.com/rss",
            logf,
        )
    },
    # Social Media Examiner (site + podcast)
    {
        "slug": "socialmediaexaminer-com",
        "action": lambda logf: (
            rss_fetch_articles(
                "socialmediaexaminer-com",
                "www.socialmediaexaminer.com",
                "https://www.socialmediaexaminer.com/feed/",
                logf,
            )
            + podcast_transcript_like_fetch(
                "socialmediaexaminer-com",
                "https://www.socialmediaexaminer.com/category/podcast/feed/",
                logf,
            )
        )
    },
    # Gary Vaynerchuk (YouTube transcripts + podcast show notes)
    {
        "slug": "garyvaynerchuk-com",
        "action": lambda logf: (
            youtube_transcripts_from_list("garyvaynerchuk", "garyvaynerchuk-com", logf)
            + podcast_transcript_like_fetch(
                "garyvaynerchuk-com",
                "https://feeds.simplecast.com/54nAGcIl",
                logf,
            )
        )
    },
    # Neil Patel (YouTube transcripts + podcast "Marketing School")
    {
        "slug": "neilpatel-com",
        "action": lambda logf: (
            youtube_transcripts_from_list("neilpatel", "neilpatel-com", logf)
            + podcast_transcript_like_fetch(
                "neilpatel-com",
                "https://feeds.megaphone.fm/ADV7489070471",
                logf,
            )
        )
    },
    # Mark Manson (persona only – skip)
    {
        "slug": "markmanson-net",
        "action": lambda logf: (logf.write("INFO  persona_only (skip scraping)\n") or 0)
    },
    # Harvard Health replacements (NHS, Mayo, MedlinePlus)
    {
        "slug": "health-harvard-replacements",
        "action": lambda logf: (
            rss_fetch_articles("health-harvard-replacements", "www.nhs.uk", "https://www.nhs.uk/news/rss", logf)
            + rss_fetch_articles("health-harvard-replacements", "www.mayoclinic.org", "https://www.mayoclinic.org/rss/all-health-information", logf)
            + rss_fetch_articles("health-harvard-replacements", "medlineplus.gov", "https://medlineplus.gov/feeds/news_en.xml", logf)
        )
    },
    # NASM PDFs (public resources)
    {
        "slug": "nasm-org",
        "action": lambda logf: sum([
            1 if save_pdf(url, RAW_ROOT / "nasm-org", logf) else 0
            for url in [
                # Add/adjust as needed – these are examples of commonly public docs
                "https://www.nasm.org/docs/default-source/pt-resources/nasm-cpt-glossary.pdf",
                "https://www.nasm.org/docs/default-source/pt-resources/opt-model-overview.pdf",
            ]
        ])
    },
    # SEC EDGAR (submissions JSON for each CIK/ticker in sources_sec/companies.txt)
    {
        "slug": "sec-gov",
        "action": lambda logf: sec_fetch_submissions(SEC_SOURCES, "sec-gov", logf)
    },
]

def main():
    total = 0
    for t in TARGETS:
        slug = t["slug"]
        log_path = LOG_ROOT / f"{slug}.log"
        (RAW_ROOT / slug).mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as logf:
            logf.write(f"== {slug} ==\n")
            try:
                count = t["action"](logf)
                logf.write(f"INFO  saved {count} items\n")
                total += count
            except Exception as e:
                logf.write(f"ERR   {slug} top-level :: {e}\n")
    print(f"Done. Total saved: {total}. See data/raw_second_pass and logs/second_pass.")

if __name__ == "__main__":
    main()
