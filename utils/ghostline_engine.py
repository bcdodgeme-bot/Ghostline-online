# utils/ghostline_engine.py - FIXED VERSION with proper timezone support and Gmail fix

import os
import json
import requests
from datetime import datetime
from typing import Optional, Iterable, List, Dict

# FIXED: Import the timezone manager
try:
    from modules.timezone_handler import timezone_manager, now_user_time
    TIMEZONE_AVAILABLE = True
except ImportError:
    TIMEZONE_AVAILABLE = False
    print("Timezone handler not available - using UTC")

# -------- OpenRouter client setup --------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    # You will get a clear error at runtime if you call the API without this.
    pass

ANSWER_RULES = (
    "Answer ONLY the latest user message. "
    "Do NOT repeat or quote the prompt. "
    "Do NOT invent 'User:'/'Assistant:' transcripts. "
    "Be direct, helpful, and stay in persona. "
    "One clean answerâ€"no preambles like 'Certainly' or 'Here's your response'."
)

# FIXED: Helper function to get properly formatted current time
def get_current_time_context():
    """Get current time in user's timezone with full context"""
    if TIMEZONE_AVAILABLE:
        try:
            user_now = now_user_time()
            tz_info = timezone_manager.get_timezone_info()
            
            # Rich time context for the AI
            return (
                f"Current time: {user_now.strftime('%A, %B %d, %Y at %I:%M %p')} "
                f"({tz_info['timezone_abbr']}, {tz_info['timezone_name']})"
            )
        except Exception as e:
            print(f"Timezone context failed: {e}")
            # Fallback to basic format
            return f"Current time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p UTC')}"
    else:
        # Fallback when timezone handler not available
        return f"Current time: {datetime.now().strftime('%A, %B %d, %Y at %H:%M UTC')}"

# ---- OpenRouter HTTP client class ----
class OpenRouterClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    def _make_request(self, endpoint: str, data: dict, stream: bool = False):
        """Make HTTP request to OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ghostline.ai",  # Optional: helps with rate limits
            "X-Title": "Ghostline AI"  # Optional: shows in OpenRouter logs
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        if stream:
            data["stream"] = True
            response = requests.post(url, headers=headers, json=data, stream=True, timeout=60)
        else:
            response = requests.post(url, headers=headers, json=data, timeout=60)
        
        response.raise_for_status()
        return response
    
    def chat_completion(self, model: str, messages: List[Dict], temperature: float = 0.7, stream: bool = False):
        """Create chat completion"""
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if stream:
            return self._stream_completion(data)
        else:
            response = self._make_request("chat/completions", data)
            return response.json()
    
    def _stream_completion(self, data: dict):
        """Handle streaming completion"""
        response = self._make_request("chat/completions", data, stream=True)
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    line = line[6:]  # Remove 'data: ' prefix
                    if line.strip() == '[DONE]':
                        break
                    try:
                        chunk = json.loads(line)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

# Create client instance
_client = OpenRouterClient(OPENROUTER_API_KEY, OPENROUTER_BASE_URL)

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
        lines.append(f"- {title}{' â€" ' + src if src else ''}\n{body}")
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
    
    # FIXED: Use timezone-aware time context instead of hardcoded UTC
    time_context = get_current_time_context()

    # reserve a small budget for user-only history (very rough)
    history_text = load_user_history_only(project, max_tokens=400)
    retrieval_block = _format_retrieval_block(retrieval_context or [])

    for voice in voices:
        # FIXED: Include rich time context in system prompt
        system_prompt = (
            f"{_persona_for(voice)} {time_context}. {ANSWER_RULES} "
            "If the user corrects you, acknowledge briefly and proceed. "
            "When discussing time, dates, or schedules, use the provided current time context."
        )

        # FIXED: Gmail command detection to prevent context contamination
        is_gmail_command = any(cmd in prompt.lower() for cmd in [
            'overnight', 'email', 'inbox', 'gmail', 'unread', 'messages',
            'no emails found', 'no readable', 'overnight emails'
        ])

        user_prompt = (
            # Only include history for non-Gmail commands to prevent context contamination
            (history_text if history_text and not is_gmail_command else "")
            + (retrieval_block if retrieval_block else "")
            + "User's new message:\n"
            + prompt
            + "\n\nRespond now as one clean answer (no transcripts)."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = _client.chat_completion(
                model=model,
                messages=messages,
                temperature=0.7 if randomize else 0.2
            )
            
            text = ""
            if 'choices' in response and len(response['choices']) > 0:
                message = response['choices'][0].get('message', {})
                text = (message.get('content') or "").strip()
            
            if not text:
                text = "(Empty response from API)"
                
        except requests.exceptions.RequestException as e:
            text = f"(API request error: {e})"
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
    
    # FIXED: Use timezone-aware time context for streaming too
    time_context = get_current_time_context()
    
    history_text = load_user_history_only(project, max_tokens=400)
    retrieval_block = _format_retrieval_block(retrieval_context or [])

    # FIXED: Include rich time context in streaming system prompt
    system_prompt = (
        f"{_persona_for(voice)} {time_context}. {ANSWER_RULES} "
        "If the user corrects you, acknowledge briefly and proceed. "
        "When discussing time, dates, or schedules, use the provided current time context."
    )
    
    # FIXED: Gmail command detection for streaming function too
    is_gmail_command = any(cmd in prompt.lower() for cmd in [
        'overnight', 'email', 'inbox', 'gmail', 'unread', 'messages',
        'no emails found', 'no readable', 'overnight emails'
    ])
    
    user_prompt = (
        # Only include history for non-Gmail commands to prevent context contamination
        (history_text if history_text and not is_gmail_command else "")
        + (retrieval_block if retrieval_block else "")
        + "User's new message:\n"
        + prompt
        + "\n\nRespond now as one clean answer (no transcripts)."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        for chunk in _client.chat_completion(
            model=model,
            messages=messages,
            temperature=0.2,
            stream=True
        ):
            yield chunk
    except requests.exceptions.RequestException as e:
        yield f"(API request error: {e})"
    except Exception as e:
        yield f"(Generation error: {e})"
