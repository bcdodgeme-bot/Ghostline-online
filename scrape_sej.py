#!/usr/bin/env python3
import argparse, json, os, re, time
from datetime import datetime, timezone
from urllib.parse import urljoin
from urllib import robotparser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import tldextract
import trafilatura

HEADERS = {"User-Agent": "GhostlineBot/0.1 (+contact: you@example.com)"}
TIMEOUT = 15
SLEEP = 1.0

DEFAULT_START_URL = "https://www.searchenginejournal.com/"
DEFAULT_MAX_PAGES = 30
OUT_PATH = "data/cleaned/search_engine_journal.jsonl"

def same_domain(url, base):
    a = tldextract.extract(url); b = tldextract.extract(base)
    return (a.domain, a.suffix) == (b.domain, b.suffix)

def load_robots(base_url: str):
    parsed = requests.utils.urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url); rp.read()
    except Exception:
        pass
    return rp

def allowed(rp, url: str) -> bool:
    try:
        return rp.can_fetch(HEADERS["User-Agent"], url)
    except Exception:
        return False

def get_html(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text

def discover_links(start_url: str, max_links: int) -> list:
    """Homepage crawl → pick likely article links."""
    html = get_html(start_url)
    soup = BeautifulSoup(html, "lxml")
    found = []
    for a in soup.select("a[href]"):
        href = a["href"].strip()
        if not href: continue
        if href.startswith("/"): href = urljoin(start_url, href)
        if not href.startswith("http"): continue
        if "searchenginejournal.com" not in href: continue
        if any(x in href for x in ["/tag/","/category/","/author/","/about","/contact"]): continue
        if href.endswith((".jpg",".png",".gif",".webp",".pdf")): continue
        found.append(href)
    # de-dupe keep order
    seen=set(); out=[]
    for u in found:
        if u not in seen:
            seen.add(u); out.append(u)
    return out[:max_links]

def extract_core(url: str):
    downloaded = trafilatura.fetch_url(url, no_ssl=True, user_agent=HEADERS["User-Agent"])
    if not downloaded: return None
    text = trafilatura.extract(downloaded, include_links=False, with_metadata=True,
                               favor_precision=True, deduplicate=True)
    if not text: return None
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len("\n".join(lines)) < 400: return None

    core = distill("\n".join(lines))
    if len(core) < 200: return None

    return {
        "source": "Search Engine Journal",
        "url": url,
        "title": guess_title(lines),
        "tags": ["seo","sem","marketing"],
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "core": core,
        "key_terms": key_terms("\n".join(lines)),
        "citations": [{"title": guess_title(lines), "url": url}]
    }

def guess_title(lines):
    for ln in lines[:10]:
        if len(ln.split()) >= 3: return ln[:200]
    return lines[0][:200] if lines else ""

def distill(text: str) -> str:
    # lightweight, paraphrase‑only bullets: keep definitions/rules/steps/metrics
    sents = split_sentences(text)
    keep = []
    pat = re.compile(r"( is | are | means | defined as | should | avoid | steps? | checklist | best practice| metric| update| algorithm| crawl| index| ranking| E-E-A-T| Core Web Vitals)", re.I)
    for s in sents:
        s = s.strip()
        if len(s) < 40: continue
        if pat.search(s):
            keep.append(f"- {paraphrase(s)}")
    if len(keep) < 8:
        for s in sents:
            if 60 <= len(s) <= 240 and s.endswith((".", "!", "?")):
                keep.append(f"- {paraphrase(s)}")
            if len(keep) >= 18: break
    # de-dupe + cap
    seen=set(); out=[]
    for k in keep:
        if k not in seen:
            seen.add(k); out.append(k)
    return "\n".join(out[:40])

def split_sentences(t: str):
    t = re.sub(r"\s+"," ",t)
    return [p.strip() for p in re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])", t) if p.strip()]

def paraphrase(s: str) -> str:
    # minimal “defluff”
    s = re.sub(r"\b(in order to)\b", "to", s, flags=re.I)
    s = re.sub(r"\b(leverag(e|ing))\b", "use", s, flags=re.I)
    s = re.sub(r"\b(utiliz(e|ing|ation))\b", "use", s, flags=re.I)
    s = re.sub(r"\b(as well as)\b", "and", s, flags=re.I)
    s = re.sub(r"\b(at scale)\b", "", s, flags=re.I)
    return re.sub(r"\s+"," ",s).strip()

def key_terms(text: str, top_k: int = 12):
    stop = set("a an the and or but if then this that these those for to in on of with by from as at it be are is was were has have had not no yes you your we our they them".split())
    words = re.findall(r"[A-Za-z][A-Za-z\-+]{2,}", text.lower())
    freq={}
    for w in words:
        if w in stop: continue
        freq[w]=freq.get(w,0)+1
    return [w for w,_ in sorted(freq.items(), key=lambda x:x[1], reverse=True)[:top_k]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-url", default=DEFAULT_START_URL)
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--save-raw-html", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.save_raw_html:
        os.makedirs("data/raw", exist_ok=True)

    rp = load_robots(args.start_url)
    links = discover_links(args.start_url, args.max_pages)
    kept = []

    for url in tqdm(links, desc="Scraping SEJ"):
        if not same_domain(url, args.start_url): continue
        if rp and not allowed(rp, url): continue

        try:
            if args.save_raw_html:
                try:
                    html = get_html(url)
                    fn = re.sub(r"[^a-zA-Z0-9]+","_",url)[:80]+".html"
                    with open(os.path.join("data/raw", fn), "w", encoding="utf-8") as f:
                        f.write(html)
                except Exception:
                    pass

            item = extract_core(url)
            if item: kept.append(item)
        except Exception:
            pass
        time.sleep(SLEEP)

    with open(args.out, "w", encoding="utf-8") as f:
        for obj in kept:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(kept)} distilled SEJ articles → {args.out}")

if __name__ == "__main__":
    main()
