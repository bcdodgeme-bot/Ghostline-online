# utils/ghostline_engine.py

import os
import json
from datetime import datetime
from typing import Optional, Iterable, List, Dict

from openai import OpenAI

# -------- OpenRouter client setup --------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    # You will get a clear error at runtime if you call the API without this.
    pass

_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

ANSWER_RULES = (
    "Answer ONLY the latest user message. "
    "Do NOT repeat or quote the prompt. "
    "Do NOT invent 'User:'/'Assistant:' transcripts. "
    "Be direct, helpful, and stay in persona. "
    "One clean answer—no preambles like 'Certainly' or 'Here's your response'."
)

# ---- Minimal per-project user-only history helpers ----
def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

def _history_path(project: str) -> str:
    return f"sessions/{project.lower().replace(' ', '_')}.json"

def load_user_history_only(project: str, max_tokens: int) -> str:
    """
    Load recent USER prompts only (no assistant text) to avoid echoing.
    Returns lines like:
      - <last user prompt>
      - <previous user prompt>
    """
    path = _history_path(project)
    if not os.path.exists(path):
        return ""

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    items: List[str] = []
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

# ---- Retrieval block formatting ----
def _format_retrieval_block(snippets: List[Dict]) -> str:
    if not snippets:
        return ""
    lines = []
    for s in snippets:
        title = s.get("title") or "Untitled"
        src = s.get("source") or ""
        body = (s.get("text") or "")[:1200]
        lines.append(f"- {title}{' — ' + src if src else ''}\n{body}")
    return "<RETRIEVED_KNOWLEDGE>\n" + "\n\n".join(lines) + "\n</RETRIEVED_KNOWLEDGE>\n"

# ---- Personas ----
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

# ---- Core non-streaming generation ----
def generate_response(
    prompt: str,
    voices: List[str],
    randomize: bool = False,
    project: str = "Personal Operating Manual",
    model: str = "openrouter/auto",
    retrieval_context: Optional[List[Dict]] = None,
) -> Dict[str, str]:
    """
    Return a dict {voice: reply}
    """
    output: Dict[str, str] = {}
    today = datetime.now().strftime("%A, %B %d, %Y")

    # reserve a small budget for user-only history (very rough)
    history_text = load_user_history_only(project, max_tokens=400)
    retrieval_block = _format_retrieval_block(retrieval_context or [])

    for voice in voices:
        system_prompt = (
            f"{_persona_for(voice)} Today is {today}. {ANSWER_RULES} "
            "If the user corrects you, acknowledge briefly and proceed."
        )

        user_prompt = (
            (history_text if history_text else "")
            + (retrieval_block if retrieval_block else "")
            + "User's new message:\n"
            + prompt
            + "\n\nRespond now as one clean answer (no transcripts)."
        )

        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7 if randomize else 0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            text = f"(Generation error: {e})"

        output[voice] = text

    return output

# ---- Streaming generation (yields text chunks) ----
def stream_generate(
    prompt: str,
    voices: List[str],
    project: str = "Personal Operating Manual",
    model: str = "openrouter/auto",
    retrieval_context: Optional[List[Dict]] = None,
) -> Iterable[str]:
    """
    Streams the first (or only) voice to the caller as plain text.
    For UI simplicity we just pick voices[0] for the stream header.
    """
    voice = voices[0] if voices else "SyntaxPrime"
    today = datetime.now().strftime("%A, %B %d, %Y")
    history_text = load_user_history_only(project, max_tokens=400)
    retrieval_block = _format_retrieval_block(retrieval_context or [])

    system_prompt = (
        f"{_persona_for(voice)} Today is {today}. {ANSWER_RULES} "
        "If the user corrects you, acknowledge briefly and proceed."
    )
    user_prompt = (
        (history_text if history_text else "")
        + (retrieval_block if retrieval_block else "")
        + "User's new message:\n"
        + prompt
        + "\n\nRespond now as one clean answer (no transcripts)."
    )

    try:
        stream = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            stream=True,
        )
        for event in stream:
            delta = getattr(event.choices[0].delta, "content", None)
            if delta:
                yield delta
    except Exception as e:
        yield f"(Generation error: {e})"



