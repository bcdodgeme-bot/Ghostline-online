# utils/rag_basic.py
from __future__ import annotations

import json
import gzip
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

# Lightweight TF‑IDF retrieval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Default location (can be overridden by load_corpus(path))
_DEFAULT_JSONL_PATH = Path("data/cleaned/ghostline_sources.jsonl")

_lock = threading.Lock()
_vectorizer: Optional[TfidfVectorizer] = None
_matrix = None
_docs: List[Dict[str, Any]] = []
_current_path: Optional[Path] = None


def _open_text(path: Path):
    """
    Open a text file that may be plain .jsonl or gzip-compressed .jsonl.gz
    Returns a text-mode file object (iterator over lines).
    """
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _normalize_record(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map heterogeneous fields to a normalized schema.
    Returns None if no usable 'text'.
    """
    text = obj.get("text") or obj.get("content") or ""
    if not text:
        return None
    title = obj.get("title") or obj.get("name") or ""
    project = obj.get("project") or obj.get("folder") or ""
    source = obj.get("source") or obj.get("url") or ""
    return {
        "title": title,
        "text": text,
        "project": project,
        "source": source,
        "_raw": obj,
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if not path.exists():
        return docs
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            norm = _normalize_record(obj)
            if norm:
                docs.append(norm)
    return docs


def load_corpus(jsonl_path: str | Path = _DEFAULT_JSONL_PATH) -> None:
    """
    Build/refresh the TF‑IDF index from a JSONL or JSONL.GZ file.
    Thread‑safe; replaces the in‑memory index atomically.
    """
    global _vectorizer, _matrix, _docs, _current_path
    with _lock:
        p = Path(jsonl_path)
        parsed = _read_jsonl(p)

        # Build new vectorizer/index
        corpus = [d["text"] for d in parsed]
        if corpus:
            vec = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=100_000,
            )
            mat = vec.fit_transform(corpus)
            # Swap in atomically
            _vectorizer = vec
            _matrix = mat
            _docs = parsed
            _current_path = p
        else:
            # Empty index
            _vectorizer = None
            _matrix = None
            _docs = []
            _current_path = p


def is_ready() -> bool:
    return _vectorizer is not None and _matrix is not None and len(_docs) > 0


def retrieve(query: str, k: int = 5, project_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return top‑k docs as [{title, text, project, source, score}].
    Optional project_filter: substring match on doc['project'] (case‑insensitive).
    """
    if not is_ready():
        return []

    q_vec = _vectorizer.transform([query])  # type: ignore[arg-type]
    scores = linear_kernel(q_vec, _matrix).ravel()  # type: ignore[arg-type]
    idxs = scores.argsort()[::-1]

    results: List[Dict[str, Any]] = []
    pf = (project_filter or "").lower().strip()

    for idx in idxs:
        doc = _docs[idx]
        if pf and pf not in (doc.get("project") or "").lower():
            continue
        results.append({
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
            "project": doc.get("project", ""),
            "source": doc.get("source", ""),
            "score": float(scores[idx]),
        })
        if len(results) >= k:
            break

    return results


# Try to load a default corpus at import time (non‑fatal if missing)
try:
    load_corpus(_DEFAULT_JSONL_PATH)
except Exception:
    pass

