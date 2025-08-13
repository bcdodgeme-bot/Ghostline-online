#!/usr/bin/env python3
import os, re, json, hashlib
from pathlib import Path
from bs4 import BeautifulSoup

RAW_FOLDERS = [
    "data/raw",
    "data/raw_overrides",
    "data/raw_personal",
    "data/raw_second_pass",
    "data/raw_lastpass",
]

OUT_PATH = Path("data/cleaned/ghostline_sources.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def guess_url_from_filename(name: str) -> str:
    # Try to reverse _ replacements and detect .html extension
    base = re.sub(r"_[a-f0-9]{12,}$", "", name)  # remove hash if present
    return re.sub(r"_", "/", base).replace(".html", "").replace(".txt", "")

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator="\n", strip=True)
    # Collapse extra blank lines
    return re.sub(r"\n{2,}", "\n", text)

total = 0
with OUT_PATH.open("w", encoding="utf-8") as outf:
    for folder in RAW_FOLDERS:
        for path in Path(folder).rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in [".html", ".txt"]:
                continue
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
                if path.suffix.lower() == ".html":
                    text = extract_text_from_html(raw)
                else:
                    text = raw
                text = text.strip()
                if not text:
                    continue
                record = {
                    "source_folder": folder,
                    "filename": path.name,
                    "url_guess": guess_url_from_filename(path.stem),
                    "sha1": sha1_text(text),
                    "content": text,
                }
                outf.write(json.dumps(record) + "\n")
                total += 1
            except Exception as e:
                print(f"ERR reading {path}: {e}")

print(f"✅ Done. Wrote {total} entries to {OUT_PATH}")
