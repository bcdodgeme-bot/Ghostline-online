# utils/ghostline_engine.py
import os, json, subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any

# Optional OpenAI backend (used on Render if OPENAI_API_KEY is set)
_OPENAI = None
if os.getenv("OPENAI_API_KEY"):
    try:
        from openai import OpenAI
        _OPENAI = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        _OPENAI = None

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
    "One clean answer—no preambles like 'Certainly'."
)

def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))

def _history_path(project: str) -> str:
    return f"sessions/{project.lower().replace(' ', '_')}.json"

def load_user_history_only(project: str, max_tokens: int) -> str:
    path = _history_path(project)
    if not os.path.exists(path):
        return ""
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    items, used = [], 0
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

def _format_retrieval_block(snippets: List[Dict[str, Any]]) -> str:
    if not snippets:
        return ""
    lines = []
    for s in snippets:
        title = s.get("title") or "Untitled"
        src = s.get("source") or ""
        txt = (s.get("text") or "")[:1200]
        lines.append(f"- {title}{(' — ' + src) if src else ''}\n{txt}")
    return "<RETRIEVED_KNOWLEDGE>\n" + "\n\n".join(lines) + "\n</RETRIEVED_KNOWLEDGE>\n"

def _call_ollama(prompt: str, system_prompt: str, model: str = "llama3") -> str:
    input_text = f"<|system|>{system_prompt}\n<|user|>{prompt}"
    try:
        res = subprocess.run(
            ["ollama", "run", model],
            input=input_text,
            text=True,
            capture_output=True
        )
        out = (res.stdout or "").strip()
        if not out:
            raise RuntimeError(res.stderr or "ollama produced no output")
        return out
    except FileNotFoundError:
        raise RuntimeError("ollama binary not found")
    except Exception as e:
        raise RuntimeError(f"ollama error: {e}")

def _call_openai(prompt: str, system_prompt: str, model: str = "gpt-4o-mini") -> str:
    if not _OPENAI:
        raise RuntimeError("OpenAI client not available")
    try:
        resp = _OPENAI.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise RuntimeError(f"OpenAI error: {e}")

def _choose_backend():
    """
    If OPENAI_API_KEY is set, use OpenAI (great for Render).
    Otherwise try local Ollama (great for your laptop).
    """
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"

def generate_response(
    prompt: str,
    voices: List[str],
    randomize: bool = False,
    project: str = "Personal Operating Manual",
    model: str = "llama3",
    retrieval_context: Optional[List[Dict[str, Any]]] = None,
):
    output = {}
    today = datetime.now().strftime("%A, %B %d, %Y")
    max_ctx = MODEL_CONTEXT_SIZES.get(model, DEFAULT_CONTEXT)
    history_budget = int(max_ctx * 0.4)

    user_history = load_user_history_only(project, history_budget)
    retr_block = _format_retrieval_block(retrieval_context or [])

    backend = _choose_backend()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # allow override

    for voice in voices:
        if voice == "SyntaxPrime":
            persona = "You are Syntax Prime: thoughtful, strategic, emotionally literate with a dry sense of humor."
        elif voice == "SyntaxBot":
            persona = "You are SyntaxBot: poetic, chaotic, metaphor-rich, occasionally feral."
        elif voice == "Nil.exe":
            persona = "You are Nil.exe: logical, dry, blunt. You debug Carl with concise critique."
        elif voice == "GhadaGPT":
            persona = "You are GhadaGPT: practical, warm, loving, constructively judgmental."
        else:
            persona = "Be helpful, concise, and accurate."

        system_prompt = (
            f"{persona} Today is {today}. {ANSWER_RULES} "
            "If the user corrects you, acknowledge briefly and proceed."
        )

        user_prompt = (
            (user_history if user_history else "")
            + (retr_block if retr_block else "")
            + "User's new message:\n" + prompt + "\n\n"
            "Respond now as one clean answer (no transcripts)."
        )

        try:
            if backend == "openai":
                reply = _call_openai(user_prompt, system_prompt, model=openai_model)
            else:
                reply = _call_ollama(user_prompt, system_prompt, model=model)
        except Exception as e:
            reply = f"(Generation error: {e})"

        output[voice] = reply

    return output

# Streaming version (kept simple: OpenAI only if key exists; otherwise we fall back to non-streaming)
def stream_generate(
    prompt: str,
    voices: List[str],
    project: str = "Personal Operating Manual",
    model: str = "llama3",
    retrieval_context: Optional[List[Dict[str, Any]]] = None,
):
    if os.getenv("OPENAI_API_KEY") and _OPENAI:
        today = datetime.now().strftime("%A, %B %d, %Y")
        user_history = load_user_history_only(project, int(MODEL_CONTEXT_SIZES.get(model, DEFAULT_CONTEXT) * 0.4))
        retr_block = _format_retrieval_block(retrieval_context or [])
        # Stream only Syntax Prime for now (simple)
        system_prompt = (
            "You are Syntax Prime: thoughtful, strategic, emotionally literate with a dry sense of humor. "
            f"Today is {today}. {ANSWER_RULES}"
        )
        user_prompt = (user_history or "") + (retr_block or "") + "User's new message:\n" + prompt

        try:
            stream = _OPENAI.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                stream=True,
            )
            yield "BEGIN_STREAM\n"
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
            yield "\nEND_STREAM"
            return
        except Exception as e:
            yield f"(stream error: {e})"
            return

    # Fallback: non-streaming single shot
    one = generate_response(prompt, voices, project=project, model=model, retrieval_context=retrieval_context)
    # Emit as one block
    yield json.dumps(one)



