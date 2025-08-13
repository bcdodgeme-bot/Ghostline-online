# utils/ghostline_engine.py

import os
import json
import shutil
import subprocess
from datetime import datetime
from typing import Optional, List, Dict

import requests  # for remote ollama or health checks

# Optional OpenAI provider (works with OpenAI, OpenRouter, etc.)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # we'll guard at runtime

# -------------------- model/context & answer rules --------------------

MODEL_CONTEXT_SIZES = {
    # local models (ollama) — rough defaults
    "tinyllama": 2048,
    "mistral":   8192,
    "llama3":    8192,
    "gemma":     8192,
    # hosted defaults (OpenAI-style) — you can override with env
    "gpt-4o-mini": 128000,
    "gpt-4o":      128000,
}
DEFAULT_CONTEXT = 8192

ANSWER_RULES = (
    "Answer ONLY the latest user message. "
    "Do NOT repeat or quote the prompt. "
    "Do NOT invent 'User:'/'Assistant:' transcripts. "
    "Be direct, helpful, and stay in persona. "
    "One clean answer—no preambles like 'Certainly'."
)

# -------------------- history helpers --------------------

def _estimate_tokens(text: str) -> int:
    # super-light token estimate
    return max(1, len(text.split()))

def _history_path(project: str) -> str:
    return f"sessions/{project.lower().replace(' ', '_')}.json"

def load_user_history_only(project: str, max_tokens: int) -> str:
    """
    Load recent USER prompts only (no assistant text) to avoid echoing.
    """
    path = _history_path(project)
    if not os.path.exists(path):
        return ""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    items = []
    used = 0
    for line in reversed(lines):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        user_txt = (entry.get("prompt") or "").strip()
        if not user_txt:
            continue
        bullet = f"- {user_txt}\n"
        tkn = _estimate_tokens(bullet)
        if used + tkn > max_tokens:
            break
        items.append(bullet)
        used += tkn

    items.reverse()
    if not items:
        return ""
    return "<RECENT_USER_MESSAGES>\n" + "".join(items) + "</RECENT_USER_MESSAGES>\n"

def _format_retrieval_block(snippets: List[Dict]) -> str:
    if not snippets:
        return ""
    lines = []
    for s in snippets:
        title = s.get("title") or "Untitled"
        src = s.get("source") or ""
        body = (s.get("text") or "")[:1200]
        lines.append(f"- {title}{(' — ' + src) if src else ''}\n{body}")
    return "<RETRIEVED_KNOWLEDGE>\n" + "\n\n".join(lines) + "\n</RETRIEVED_KNOWLEDGE>\n"

# -------------------- model caller (OpenAI first, then Ollama) --------------------

def _call_openai(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    Calls any OpenAI-compatible endpoint if OPENAI_API_KEY is set.
    Supports custom base URL (e.g., OpenRouter) via OPENAI_BASE_URL.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    if OpenAI is None:
        raise RuntimeError("openai package not installed")

    base_url = os.getenv("OPENAI_BASE_URL")  # e.g. https://openrouter.ai/api/v1
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.5")),
    )
    return (resp.choices[0].message.content or "").strip()

def _call_remote_ollama(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    Calls a remote Ollama server via HTTP if OLLAMA_HOST is set.
    Example OLLAMA_HOST: http://localhost:11434
    """
    host = os.getenv("OLLAMA_HOST", "").rstrip("/")
    if not host:
        raise RuntimeError("OLLAMA_HOST not set")
    prompt = f"<|system|>{system_prompt}\n<|user|>{user_prompt}"
    r = requests.post(f"{host}/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()

def _call_local_ollama(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    Falls back to local Ollama subprocess (your laptop).
    """
    if shutil.which("ollama") is None:
        raise FileNotFoundError("ollama binary not found")
    input_text = f"<|system|>{system_prompt}\n<|user|>{user_prompt}"
    res = subprocess.run(
        ["ollama", "run", model],
        input=input_text,
        text=True,
        capture_output=True,
    )
    out = (res.stdout or "").strip()
    if not out and res.stderr:
        raise RuntimeError(res.stderr.strip())
    return out

def call_model(system_prompt: str, user_prompt: str, model: str) -> str:
    """
    Priority:
      1) OPENAI_API_KEY present -> OpenAI/OpenRouter/etc.
      2) OLLAMA_HOST set        -> remote Ollama server
      3) local 'ollama' binary  -> your Mac dev env
    """
    if os.getenv("OPENAI_API_KEY"):
        return _call_openai(system_prompt, user_prompt, model)
    if os.getenv("OLLAMA_HOST"):
        return _call_remote_ollama(system_prompt, user_prompt, model)
    return _call_local_ollama(system_prompt, user_prompt, model)

# -------------------- main generate functions --------------------

def generate_response(
    prompt: str,
    voices: list[str],
    randomize: bool = False,
    project: str = "Personal Operating Manual",
    model: str = None,
    retrieval_context: Optional[list[dict]] = None,
):
    output = {}
    today = datetime.now().strftime("%A, %B %d, %Y")

    # model selection: allow env override (for Render)
    model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
    max_ctx = MODEL_CONTEXT_SIZES.get(model, DEFAULT_CONTEXT)
    history_budget = int(max_ctx * float(os.getenv("MEMORY_BUDGET_FRACTION", "0.4")))

    history_text = load_user_history_only(project, history_budget)
    retrieval_block = _format_retrieval_block(retrieval_context or [])

    for voice in voices:
        if voice == "SyntaxPrime":
            persona = "You are Syntax Prime: thoughtful, strategic, emotionally literate with a dry sense of humor."
        elif voice == "SyntaxBot":
            persona = "You are SyntaxBot: poetic, chaotic, metaphor-rich, occasionally feral."
        elif voice == "Nil.exe":
            persona = "You are Nil.exe: logical, dry, blunt. You debug Carl's thinking with concise critique."
        elif voice == "GhadaGPT":
            persona = "You are GhadaGPT: practical, warm, loving, constructively judgmental for Carl's own good."
        else:
            persona = "Be helpful, concise, and accurate."

        system_prompt = (
            f"{persona} Today is {today}. {ANSWER_RULES} "
            "If the user corrects you, acknowledge briefly and proceed."
        )

        user_prompt = (
            (history_text if history_text else "")
            + (retrieval_block if retrieval_block else "")
            + "User's new message:\n" + prompt + "\n\n"
            "Respond now as one clean answer (no transcripts)."
        )

        try:
            reply = call_model(system_prompt, user_prompt, model=model)
        except Exception as e:
            reply = f"(Generation error: {e})"

        output[voice] = reply

    return output


# simple token stream stub for /stream (keeps your endpoint working)
def stream_generate(
    prompt: str,
    voices: list[str],
    project: str = "Personal Operating Manual",
    model: str = None,
    retrieval_context: Optional[list[dict]] = None,
):
    """
    For now we just yield the whole thing as one chunk per voice to keep
    the streaming route alive without complicating SSE.
    """
    resp = generate_response(
        prompt, voices, project=project, model=model, retrieval_context=retrieval_context
    )
    # join voices into a single stream chunk
    for v, txt in resp.items():
        yield f"\n\n[{v}] {txt}"


