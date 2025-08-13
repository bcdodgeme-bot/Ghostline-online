# utils/ghostline_engine.py
# Backend-flexible engine: OpenAI (default if OPENAI_API_KEY present) or Ollama (local)
# Also supports streaming and per-project user-history memory.

import os
import json
import subprocess
from datetime import datetime
from typing import List, Dict, Optional, Generator

# ---- Model/context sizing (used to budget memory) ----
MODEL_CONTEXT_SIZES = {
    "tinyllama": 2048,
    "mistral":   8192,
    "llama3":    8192,
    "gemma":     8192,
    # OpenAI short-hands (approx)
    "gpt-4o-mini":  16384,
    "gpt-4o":       128000,
}
DEFAULT_CONTEXT = 4096

ANSWER_RULES = (
    "Answer ONLY the latest user message. "
    "Do NOT repeat or quote the prompt. "
    "Do NOT invent 'User:'/'Assistant:' transcripts. "
    "Be direct, helpful, and stay in persona. "
    "One clean answer—no preambles like 'Certainly' or 'Here's your response'."
)

# ---------------- Backend selection helpers ----------------

def _choose_backend_env() -> str:
    """
    If OPENAI_API_KEY is present, prefer 'openai'. Otherwise default 'ollama'.
    You can override with FORCE_BACKEND=openai|ollama.
    """
    fb = os.getenv("FORCE_BACKEND")
    if fb in ("openai", "ollama"):
        return fb
    return "openai" if os.getenv("OPENAI_API_KEY") else "ollama"

BACKEND = _choose_backend_env()

def _effective_model_name(requested: str) -> str:
    """
    If backend is openai, prefer OPENAI_MODEL or default 'gpt-4o-mini'.
    If backend is ollama, use the requested (e.g., 'llama3'/'tinyllama').
    """
    if BACKEND == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return requested

# ---------------- History (per-project, user-only) ----------------

def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

def _history_path(project: str) -> str:
    return f"sessions/{project.lower().replace(' ', '_')}.json"

def load_user_history_only(project: str, max_tokens: int) -> str:
    """
    Load recent USER prompts only (no assistant text) to reduce echoing.
    Returns:
      <RECENT_USER_MESSAGES>
      - last user message
      - previous user message
      ...
      </RECENT_USER_MESSAGES>
    """
    path = _history_path(project)
    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as f:
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
        text = (s.get("text") or "")[:1200]
        lines.append(f"- {title}{(' — ' + src) if src else ''}\n{text}")
    return "<RETRIEVED_KNOWLEDGE>\n" + "\n\n".join(lines) + "\n</RETRIEVED_KNOWLEDGE>\n"

# ---------------- Personas ----------------

def _persona_for(voice: str) -> str:
    if voice == "SyntaxPrime":
        return "You are Syntax Prime: thoughtful, strategic, emotionally literate with a dry sense of humor."
    if voice == "SyntaxBot":
        return "You are SyntaxBot: poetic, chaotic, metaphor-rich, occasionally feral."
    if voice == "Nil.exe":
        return "You are Nil.exe: logical, dry, blunt. You debug Carl's thinking with concise critique."
    if voice == "GhadaGPT":
        return "You are GhadaGPT: practical, warm, loving, constructively judgmental for Carl's own good."
    return "Be helpful, concise, and accurate."

# ---------------- Ollama backend ----------------

def call_ollama(prompt: str, system_prompt: str, model: str = "llama3") -> str:
    """
    Minimal blocking call to local Ollama.
    Requires `ollama serve` running locally. Not available on Render.
    """
    input_text = f"<|system|>{system_prompt}\n<|user|>{prompt}"
    res = subprocess.run(
        ["ollama", "run", model],
        input=input_text,
        text=True,
        capture_output=True
    )
    # If ollama binary is missing, stderr typically contains the error
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip() or "ollama run failed")
    return (res.stdout or "").strip()

def stream_ollama(prompt: str, system_prompt: str, model: str = "llama3") -> Generator[str, None, None]:
    """
    Very simple line-based stream using `ollama run --stream` (best-effort).
    Falls back to non-stream if streaming is not supported.
    """
    try:
        proc = subprocess.Popen(
            ["ollama", "run", "--stream", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    except FileNotFoundError as e:
        raise RuntimeError("ollama binary not found") from e

    input_text = f"<|system|>{system_prompt}\n<|user|>{prompt}"
    assert proc.stdin is not None
    proc.stdin.write(input_text)
    proc.stdin.flush()
    proc.stdin.close()

    assert proc.stdout is not None
    for line in proc.stdout:
        chunk = line.rstrip("\n")
        if chunk:
            yield chunk
    proc.wait()

# ---------------- OpenAI backend ----------------

_openai_client = None
def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()  # uses OPENAI_API_KEY
    return _openai_client

def call_openai(prompt: str, system_prompt: str, model: str = "gpt-4o-mini") -> str:
    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()

def stream_openai(prompt: str, system_prompt: str, model: str = "gpt-4o-mini") -> Generator[str, None, None]:
    client = _get_openai_client()
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        stream=True,
    )
    for event in stream:
        delta = getattr(getattr(event, "choices", [{}])[0], "delta", None)
        if delta and getattr(delta, "content", None):
            yield delta.content

# ---------------- Public API ----------------

def generate_response(
    prompt: str,
    voices: List[str],
    randomize: bool = False,
    project: str = "Personal Operating Manual",
    model: str = "llama3",                      # local default when using Ollama
    retrieval_context: Optional[List[Dict]] = None,
) -> Dict[str, str]:
    """
    Returns { voice_name: reply_text } for each requested voice.
    Backend is chosen by env (OpenAI if key present, otherwise Ollama), and
    can be forced via FORCE_BACKEND.
    """
    output: Dict[str, str] = {}
    today = datetime.now().strftime("%A, %B %d, %Y")

    # Memory budget: ~40% of the effective context for user-only history
    eff_model = _effective_model_name(model)
    max_ctx = MODEL_CONTEXT_SIZES.get(eff_model, DEFAULT_CONTEXT)
    history_budget = int(max_ctx * 0.4)
    user_history = load_user_history_only(project, history_budget)
    retrieved = _format_retrieval_block(retrieval_context or [])

    for voice in voices:
        persona = _persona_for(voice)
        system_prompt = (
            f"{persona} Today is {today}. {ANSWER_RULES} "
            "If the user corrects you, acknowledge briefly and proceed."
        )
        user_prompt = (
            (user_history if user_history else "") +
            (retrieved if retrieved else "") +
            "User's new message:\n" + prompt + "\n\n"
            "Respond now as one clean answer (no transcripts)."
        )

        try:
            if BACKEND == "openai":
                reply = call_openai(user_prompt, system_prompt, model=eff_model)
            else:
                reply = call_ollama(user_prompt, system_prompt, model=model)
        except Exception as e:
            reply = f"(Generation error: {e})"

        output[voice] = reply

    return output


def stream_generate(
    prompt: str,
    voices: List[str],
    project: str = "Personal Operating Manual",
    model: str = "llama3",
    retrieval_context: Optional[List[Dict]] = None,
) -> Generator[str, None, None]:
    """
    Stream a single response (use the first requested voice).
    Yields plain text chunks.
    """
    voice = voices[0] if voices else "SyntaxPrime"
    persona = _persona_for(voice)
    today = datetime.now().strftime("%A, %B %d, %Y")

    eff_model = _effective_model_name(model)
    max_ctx = MODEL_CONTEXT_SIZES.get(eff_model, DEFAULT_CONTEXT)
    history_budget = int(max_ctx * 0.4)
    user_history = load_user_history_only(project, history_budget)
    retrieved = _format_retrieval_block(retrieval_context or [])

    system_prompt = (
        f"{persona} Today is {today}. {ANSWER_RULES} "
        "If the user corrects you, acknowledge briefly and proceed."
    )
    user_prompt = (
        (user_history if user_history else "") +
        (retrieved if retrieved else "") +
        "User's new message:\n" + prompt + "\n\n"
        "Respond now as one clean answer (no transcripts)."
    )

    try:
        if BACKEND == "openai":
            for chunk in stream_openai(user_prompt, system_prompt, model=eff_model):
                if chunk:
                    yield chunk
        else:
            for chunk in stream_ollama(user_prompt, system_prompt, model=model):
                if chunk:
                    yield chunk
    except Exception as e:
        yield f"(Generation error: {e})"





