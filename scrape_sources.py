#!/usr/bin/env python3
import os, re, time, json, hashlib
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

try:
    import feedparser
except ImportError:
    feedparser = None

# ----------------------------
# Settings
# ----------------------------
MAX_URLS_PER_SITE = 1000
SLEEP_SECONDS = 0.5
TIMEOUT = 25
HEADERS = {"User-Agent": "GhostlineBot/0.1 (+contact: you@example.com)"}

RAW_ROOT = Path("data/raw")
LOGS_ROOT = Path("logs")
RAW_ROOT.mkdir(parents=True, exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

SITEMAP_GUESSES = [
    "sitemap_index.xml",  # WordPress style
    "sitemap.xml",
    "sitemap-index.xml",
    "sitemap1.xml",
]

RSS_GUESSES = [
    "feed/",
    "rss.xml",
    "feed.xml",
    "atom.xml",
    "index.xml",
    "blog/feed/",
    "news/feed/",
]

# If you want to restrict to “article-like” URLs, add patterns here per site later.
# By default we include most HTML pages and skip obvious non-articles (media, tag pages, etc.)
DEFAULT_EXCLUDE_PATTERNS = [
    r"/tag/", r"/category/", r"/author/", r"/about", r"/contact", r"/privacy", r"/terms",
    r"\.(jpg|jpeg|png|gif|webp|pdf|zip|mp3|mp4|mov)(\?|$)"
]
EXCLUDE_RE = re.compile("|".join(DEFAULT_EXCLUDE_PATTERNS), re.IGNORECASE)


# ----------------------------
# Helpers
# ----------------------------
def slugify(url_or_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", url_or_name.lower()).strip("-")

def read_sources(path="sources.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def base_of(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/"

def robots_for(base_url: str):
    p = urlparse(base_url)
    robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
    except Exception:
        return None
    return rp

def allowed(rp, url: str) -> bool:
    if not rp:
        return True
    try:
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return True

def get(url: str) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r

def save_raw(dest_dir: Path, url: str, html: str):
    # Short hash + sanitized tail to keep filenames manageable
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    tail = re.sub(r"[^a-zA-Z0-9]+", "_", url[-80:]).strip("_")
    fname = f"{h}_{tail}.html"
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / fname).write_text(html, encoding="utf-8")
    return dest_dir / fname

def find_sitemaps(base_url: str) -> list[str]:
    """Try a few common sitemap locations; also sniff <link rel> if present."""
    urls = []
    b = base_of(base_url)
    # explicit guesses
    for guess in SITEMAP_GUESSES:
        try:
            u = urljoin(b, guess)
            r = get(u)
            if r.status_code == 200 and ("xml" in r.headers.get("Content-Type", "") or r.text.strip().startswith("<?xml")):
                urls.append(u)
        except Exception:
            continue

    # sniff <link> tags on homepage
    try:
        r = get(b)
        soup = BeautifulSoup(r.text, "lxml")
        for ln in soup.find_all("link"):
            href = ln.get("href")
            if not href:
                continue
            if "sitemap" in href and href.endswith((".xml", ".gz")):
                if href.startswith("/"):
                    href = urljoin(b, href)
                if href.startswith("http"):
                    urls.append(href)
    except Exception:
        pass

    # de-dup keep order
    seen = set(); out = []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

def urls_from_sitemap(sitemap_url: str) -> list[str]:
    """Return all <loc> URLs from a sitemap or sitemap index."""
    out = []
    try:
        r = get(sitemap_url)
    except Exception:
        return out
    soup = BeautifulSoup(r.text, "xml")

    # If it's a sitemap index, follow child <sitemap><loc>
    children = [loc.text.strip() for loc in soup.select("sitemap > loc")]
    if children:
        for child in children:
            out.extend(urls_from_sitemap(child))
        return out

    # Otherwise, collect <url><loc>
    out.extend([loc.text.strip() for loc in soup.select("url > loc")])
    return out

def find_rss(base_url: str) -> str | None:
    b = base_of(base_url)
    # try common guesses
    for guess in RSS_GUESSES:
        try:
            u = urljoin(b, guess)
            r = get(u)
            if r.status_code == 200 and ("xml" in r.headers.get("Content-Type", "") or r.text.strip().startswith("<?xml")):
                return u
        except Exception:
            continue
    # sniff <link rel="alternate" type="application/rss+xml">
    try:
        r = get(b)
        soup = BeautifulSoup(r.text, "lxml")
        for ln in soup.find_all("link", rel=lambda x: x and "alternate" in x):
            t = (ln.get("type") or "").lower()
            if "xml" in t:
                href = ln.get("href")
                if href:
                    if href.startswith("/"):
                        href = urljoin(b, href)
                    if href.startswith("http"):
                        return href
    except Exception:
        pass
    return None

def urls_from_rss(feed_url: str, limit: int = 1000) -> list[str]:
    urls = []
    if not feedparser:
        return urls
    try:
        feed = feedparser.parse(feed_url)
        for e in feed.entries[:limit]:
            if getattr(e, "link", None):
                urls.append(e.link)
    except Exception:
        pass
    return urls

def homepage_links(base_url: str, limit: int = 1000) -> list[str]:
    urls = []
    try:
        r = get(base_url)
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href", "").strip()
            if not href:
                continue
            if href.startswith("/"):
                href = urljoin(base_of(base_url), href)
            if not href.startswith("http"):
                continue
            if urlparse(href).netloc != urlparse(base_url).netloc:
                continue
            if EXCLUDE_RE.search(href):
                continue
            urls.append(href)
    except Exception:
        pass
    # de-dup keep order and cap
    seen = set(); out = []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
        if len(out) >= limit:
            break
    return out

def collect_urls_for_site(start_url: str, max_urls: int) -> list[str]:
    """Sitemap → RSS → Homepage; return up to max_urls unique URLs for this site."""
    collected = []

    # 1) Sitemaps (deep)
    for sm in find_sitemaps(start_url):
        collected.extend(urls_from_sitemap(sm))
        if len(collected) >= max_urls:
            break

    # 2) RSS (recent)
    if len(collected) < max_urls:
        rss = find_rss(start_url)
        if rss:
            collected.extend(urls_from_rss(rss, limit=max_urls - len(collected)))

    # 3) Homepage fallback
    if len(collected) < max_urls:
        collected.extend(homepage_links(start_url, limit=max_urls - len(collected)))

    # Clean + filter + de-dup
    cleaned = []
    seen = set()
    base_netloc = urlparse(start_url).netloc
    for u in collected:
        if not u or not u.startswith("http"):
            continue
        if urlparse(u).netloc != base_netloc:
            continue
        if EXCLUDE_RE.search(u):
            continue
        if u in seen:
            continue
        seen.add(u)
        cleaned.append(u)

    return cleaned[:max_urls]

# ----------------------------
# Main
# ----------------------------
def main():
    sources = read_sources("sources.txt")
    if not sources:
        print("No sources found in sources.txt")
        return

    for start in sources:
        base = base_of(start)
        slug = slugify(base)
        raw_dir = RAW_ROOT / slug
        log_path = LOGS_ROOT / f"{slug}.log"
        rp = robots_for(base)

        print(f"\n=== {base} → {slug} ===")
        urls = collect_urls_for_site(base, MAX_URLS_PER_SITE)
        print(f"Found {len(urls)} candidate URLs (cap {MAX_URLS_PER_SITE}). Fetching...")

        saved = 0
        with open(log_path, "w", encoding="utf-8") as logf:
            for i, u in enumerate(urls, 1):
                try:
                    if not allowed(rp, u):
                        logf.write(f"SKIP robots {u}\n")
                        continue
                    r = get(u)
                    save_raw(raw_dir, u, r.text)
                    logf.write(f"SAVED {u}\n")
                    saved += 1
                except Exception as e:
                    logf.write(f"ERR {u} :: {e}\n")
                if i % 25 == 0:
                    print(f"  …{i}/{len(urls)}")
                time.sleep(SLEEP_SECONDS)

        print(f"Saved {saved} pages → {raw_dir} (log: {log_path})")

if __name__ == "__main__":
    main()
