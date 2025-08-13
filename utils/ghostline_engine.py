# utils/ghostline_engine.py

import os
import json
import subprocess
from datetime import datetime
from typing import Optional, List, Dict

MODEL_CONTEXT_SIZES = {
    "tinyllama": 2048,
    "mistral":   8192,
    "llama3":    8192,
    "gemma":     8192,
}
DEFAULT_CONTEXT = 2048

ANSWER_RULES = (
    "Answer ONLY the latest user message. "
    "Do NOT repeat or quote the prompt. "
    "Do NOT invent 'User:'/'Assistant:' transcripts. "
    "Be direct, helpful, and stay in persona. "
    "One clean answer—no preambles like 'Certainly' or 'Here's your response'."
)

def call_ollama(prompt: str, system_prompt: str, model: str = "llama3") -> str:
    """
    Minimal Ollama call. Run `ollama serve` (or the Ollama app) in the background.
    """
    input_text = f"<|system|>{system_prompt}\n<|user|>{prompt}"
    res = subprocess.run(
        ["ollama", "run", model],
        input=input_text,
        text=True,
        capture_output=True
    )
    return (res.stdout or "").strip()

def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

def _history_path(project: str) -> str:
    return f"sessions/{project.lower().replace(' ', '_')}.json"

def load_user_history_only(project: str, max_tokens: int) -> str:
    """
    Load recent USER prompts only (no assistant text) to avoid echoing.
    Returns a compact block of bullets.
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

def _format_retrieval_block(snippets: Optional[List[Dict]]) -> str:
    """
    Format retrieved knowledge snippets (from RAG) for the prompt.
    Expects each snippet dict to have keys like: title, text, source.
    """
    if not snippets:
        return ""
    lines = []
    for s in snippets:
        title = (s.get("title") or "Untitled").strip()
        src   = (s.get("source") or "").strip()
        text  = (s.get("text") or "").strip()
        # Trim long text to keep budget sane
        if len(text) > 1400:
            text = text[:1400] + " …"
        header = f"- {title}" + (f" — {src}" if src else "")
        lines.append(f"{header}\n{text}")
    return "<RETRIEVED_KNOWLEDGE>\n" + "\n\n".join(lines) + "\n</RETRIEVED_KNOWLEDGE>\n"

def generate_response(
    prompt: str,
    voices: List[str],
    randomize: bool = False,
    project: str = "Personal Operating Manual",
    model: str = "llama3",
    retrieval_context: Optional[List[Dict]] = None,  # ⬅️ new: optional RAG snippets
):
    output = {}
    today = datetime.now().strftime("%A, %B %d, %Y")

    max_ctx = MODEL_CONTEXT_SIZES.get(model, DEFAULT_CONTEXT)
    history_budget = int(max_ctx * 0.8)  # ~40% for user-only history

    user_history = load_user_history_only(project, history_budget)
    retrieved_block = _format_retrieval_block(retrieval_context) if retrieval_context else ""

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
            (user_history if user_history else "")
            + (retrieved_block if retrieved_block else "")
            + "User's new message:\n" + prompt + "\n\n"
            "Respond now as one clean answer (no transcripts)."
        )

        try:
            reply = call_ollama(user_prompt, system_prompt, model=model)
        except Exception as e:
            reply = f"(Generation error: {e})"

        output[voice] = reply

    return output
# --- Streaming support ---

def _build_prompts_for_voice(prompt, voice, project, model, user_history, retrieved_block, today):
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
        (user_history if user_history else "")
        + (retrieved_block if retrieved_block else "")
        + "User's new message:\n" + prompt + "\n\n"
        "Respond now as one clean answer (no transcripts)."
    )
    return system_prompt, user_prompt

def stream_generate(
    prompt: str,
    voices: list[str],
    project: str,
    model: str = "llama3",
    retrieval_context: list | None = None,
):
    """
    Streams text for the FIRST selected voice. Falls back to SyntaxPrime if none selected.
    Yields chunks of text.
    """
    import subprocess
    from datetime import datetime

    voice = voices[0] if voices else "SyntaxPrime"
    today = datetime.now().strftime("%A, %B %d, %Y")

    max_ctx = MODEL_CONTEXT_SIZES.get(model, DEFAULT_CONTEXT)
    history_budget = int(max_ctx * 0.4)
    user_history = load_user_history_only(project, history_budget)
    retrieved_block = _format_retrieval_block(retrieval_context) if retrieval_context else ""

    system_prompt, user_prompt = _build_prompts_for_voice(
        prompt, voice, project, model, user_history, retrieved_block, today
    )

    input_text = f"<|system|>{system_prompt}\n<|user|>{user_prompt}"

    # Stream from Ollama
    proc = subprocess.Popen(
        ["ollama", "run", model],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    try:
        proc.stdin.write(input_text)
        proc.stdin.close()
    except Exception:
        pass

    for line in proc.stdout:
        if line:
            yield line

