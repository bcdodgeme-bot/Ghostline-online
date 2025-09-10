# utils/ghostline_engine.py - COMPLETE REWRITE with Model Blacklist and Enhanced Features

import os
import json
import requests
from datetime import datetime
from typing import Optional, Iterable, List, Dict

# ========================================================================
# CONFIGURATION AND CONSTANTS
# ========================================================================

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model filtering configuration
BLOCKED_MODELS = [
    'gpt-5o', 'openai/gpt-5o', 'gpt-5', 'openai/gpt-5',
    'gpt-5-turbo', 'openai/gpt-5-turbo'
]

# Preferred fallback models when auto-selection is blocked
FALLBACK_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "meta-llama/llama-3.1-405b-instruct"
]

# Response quality rules
ANSWER_RULES = (
    "Answer ONLY the latest user message. "
    "Do NOT repeat or quote the prompt. "
    "Do NOT invent 'User:'/'Assistant:' transcripts. "
    "Be direct, helpful, and stay in persona. "
    "One clean answer no preambles like 'Certainly' or 'Here's your response'."
)

if not OPENROUTER_API_KEY:
    print("⚠️  WARNING: OPENROUTER_API_KEY not set - API calls will fail")

# ========================================================================
# TIMEZONE AND TIME HANDLING
# ========================================================================

try:
    from modules.timezone_handler import timezone_manager, now_user_time
    TIMEZONE_AVAILABLE = True
except ImportError:
    TIMEZONE_AVAILABLE = False
    print("Timezone handler not available - using UTC")

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

# ========================================================================
# MODEL FILTERING AND SELECTION
# ========================================================================

def filter_model_selection(model: str) -> str:
    """
    Filter and validate model selection, preventing blocked models
    
    Args:
        model: The requested model name
        
    Returns:
        Safe model name to use (original or fallback)
    """
    if not model:
        return FALLBACK_MODELS[0]
    
    # Check if the model is in our blocklist
    for blocked_model in BLOCKED_MODELS:
        if blocked_model.lower() in model.lower():
            print(f"🚫 Model '{model}' is blocked. Using fallback: {FALLBACK_MODELS[0]}")
            return FALLBACK_MODELS[0]
    
    # Special handling for openrouter/auto - let it through but log it
    if model.lower() in ['openrouter/auto', 'auto']:
        print(f"🤖 Using OpenRouter auto-selection (blocked models will be filtered)")
        return model
    
    print(f"✅ Model '{model}' approved for use")
    return model

def get_model_blacklist_status():
    """Get current model blacklist configuration for diagnostics"""
    return {
        'blocked_models': BLOCKED_MODELS,
        'fallback_models': FALLBACK_MODELS,
        'current_chat_model': os.getenv("CHAT_MODEL", "openrouter/auto"),
        'filtered_chat_model': filter_model_selection(os.getenv("CHAT_MODEL", "openrouter/auto"))
    }

# ========================================================================
# OPENROUTER CLIENT
# ========================================================================

class OpenRouterClient:
    """Enhanced OpenRouter client with model filtering and error handling"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
    
    def _make_request(self, endpoint: str, data: dict, stream: bool = False):
        """Make HTTP request to OpenRouter API with enhanced error handling"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ghostline.ai",
            "X-Title": "Ghostline AI"
        }
        
        url = f"{self.base_url}/{endpoint}"
        
        # Apply model filtering before making request
        if 'model' in data:
            data['model'] = filter_model_selection(data['model'])
        
        try:
            if stream:
                data["stream"] = True
                response = requests.post(url, headers=headers, json=data, stream=True, timeout=60)
            else:
                response = requests.post(url, headers=headers, json=data, timeout=60)
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 402:
                print("💰 OpenRouter: Insufficient credits")
                raise Exception("OpenRouter API: Insufficient credits. Please add funds to your account.")
            elif e.response.status_code == 429:
                print("⏱️  OpenRouter: Rate limit exceeded")
                raise Exception("OpenRouter API: Rate limit exceeded. Please try again in a moment.")
            elif e.response.status_code == 401:
                print("🔑 OpenRouter: Authentication failed")
                raise Exception("OpenRouter API: Authentication failed. Check your API key.")
            else:
                print(f"🚨 OpenRouter HTTP Error: {e.response.status_code}")
                raise Exception(f"OpenRouter API error: {e.response.status_code}")
        except requests.exceptions.Timeout:
            print("⏱️  OpenRouter: Request timeout")
            raise Exception("OpenRouter API: Request timeout. The model may be overloaded.")
        except requests.exceptions.ConnectionError:
            print("🌐 OpenRouter: Connection failed")
            raise Exception("OpenRouter API: Connection failed. Check your internet connection.")
        except Exception as e:
            print(f"🚨 OpenRouter: Unexpected error: {e}")
            raise
    
    def chat_completion(self, model: str, messages: List[Dict], temperature: float = 0.7, stream: bool = False):
        """Create chat completion with model filtering"""
        # Log the model selection process
        original_model = model
        filtered_model = filter_model_selection(model)
        
        if original_model != filtered_model:
            print(f"🔄 Model changed: {original_model} → {filtered_model}")
        
        data = {
            "model": filtered_model,
            "messages": messages,
            "temperature": temperature
        }
        
        if stream:
            return self._stream_completion(data)
        else:
            response = self._make_request("chat/completions", data)
            return response.json()
    
    def _stream_completion(self, data: dict):
        """Handle streaming completion with error handling"""
        try:
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
        except Exception as e:
            print(f"🚨 Streaming error: {e}")
            raise

# Create global client instance
_client = OpenRouterClient(OPENROUTER_API_KEY, OPENROUTER_BASE_URL)

# ========================================================================
# CONVERSATION HISTORY MANAGEMENT
# ========================================================================

def _estimate_tokens(text: str) -> int:
    """Estimate token count for text"""
    return max(1, len(text.split()))

def _history_path(project: str) -> str:
    """Get history file path for project"""
    return f"sessions/{project.lower().replace(' ', '_')}.json"

def load_user_history_only(project: str, max_tokens: int) -> str:
    """
    Load recent USER prompts only (no assistant text) to avoid echoing.
    Returns a summary of recent conversation context.
    """
    history_file = _history_path(project)
    
    if not os.path.exists(history_file):
        return ""
    
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        
        if not isinstance(history, list):
            return ""
        
        # Extract user messages only
        user_messages = []
        total_tokens = 0
        
        # Go through history in reverse to get most recent context
        for entry in reversed(history[-20:]):  # Last 20 entries
            if isinstance(entry, dict) and "user_input" in entry:
                user_input = entry["user_input"]
                tokens = _estimate_tokens(user_input)
                
                if total_tokens + tokens > max_tokens:
                    break
                
                user_messages.insert(0, user_input)
                total_tokens += tokens
        
        if user_messages:
            context = "Recent conversation context:\n"
            for i, msg in enumerate(user_messages[-5:], 1):  # Last 5 user inputs
                context += f"{i}. {msg}\n"
            return context
        
        return ""
        
    except Exception as e:
        print(f"History loading error: {e}")
        return ""

# ========================================================================
# ENHANCED RESPONSE GENERATION
# ========================================================================

def generate_response(
    user_input: str,
    use_voices: List[str],
    random_toggle: bool,
    project: str = "default",
    model: str = None,
    retrieval_context: List[Dict] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Enhanced response generation with model filtering and improved context handling
    
    Args:
        user_input: The user's message
        use_voices: List of voice personas to use
        random_toggle: Whether to use random selection
        project: Project context for history
        model: Model to use (will be filtered)
        retrieval_context: Context from RAG system
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with voice responses
    """
    
    # Use environment model if none specified, then filter it
    if model is None:
        model = os.getenv("CHAT_MODEL", "openrouter/auto")
    
    # Apply model filtering
    filtered_model = filter_model_selection(model)
    
    # Get conversation history
    conversation_context = load_user_history_only(project, max_tokens=500)
    
    # Get current time context
    time_context = get_current_time_context()
    
    # Build comprehensive system context
    system_context = f"""You are Ghostline AI, an advanced personal assistant.

{time_context}

{conversation_context}

{ANSWER_RULES}

Current project context: {project}
"""
    
    # Add retrieval context if available
    if retrieval_context and len(retrieval_context) > 0:
        system_context += "\n\nRelevant knowledge base context:\n"
        for i, ctx in enumerate(retrieval_context[:3], 1):  # Limit to 3 most relevant
            system_context += f"{i}. {ctx.get('content', '')[:200]}...\n"
    
    # Prepare messages for API
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_input}
    ]
    
    # Generate responses for each requested voice
    responses = {}
    
    for voice in use_voices:
        try:
            print(f"🎭 Generating {voice} response using {filtered_model}")
            
            # Customize system prompt based on voice
            voice_messages = messages.copy()
            if voice == "SyntaxPrime":
                voice_messages[0]["content"] += "\n\nVoice: You are SyntaxPrime, the primary Ghostline assistant. Be helpful, direct, and professional."
            elif voice == "SyntaxBot":
                voice_messages[0]["content"] += "\n\nVoice: You are SyntaxBot, focused on technical and programming assistance. Be precise and code-focused."
            elif voice == "NilExe":
                voice_messages[0]["content"] += "\n\nVoice: You are NilExe, a philosophical and creative assistant. Be thoughtful and contemplative."
            
            # Make API call with filtered model
            response = _client.chat_completion(
                model=filtered_model,
                messages=voice_messages,
                temperature=kwargs.get('temperature', 0.7)
            )
            
            if 'choices' in response and len(response['choices']) > 0:
                content = response['choices'][0]['message']['content']
                responses[voice] = content
                print(f"✅ {voice} response generated ({len(content)} chars)")
            else:
                error_msg = f"No response content from {filtered_model}"
                print(f"❌ {error_msg}")
                responses[voice] = f"Error: {error_msg}"
                
        except Exception as e:
            error_msg = f"Response generation failed for {voice}: {str(e)}"
            print(f"❌ {error_msg}")
            responses[voice] = f"Error: {error_msg}"
    
    # Return at least SyntaxPrime response
    if not responses:
        responses["SyntaxPrime"] = "Error: All response generation failed"
    elif "SyntaxPrime" not in responses and len(responses) > 0:
        # If SyntaxPrime failed but others succeeded, copy one
        first_voice = list(responses.keys())[0]
        responses["SyntaxPrime"] = responses[first_voice]
    
    return responses

def generate_response_with_context_check(
    user_input: str,
    use_voices: List[str],
    random_toggle: bool,
    project: str,
    model: str,
    retrieval_context: List[Dict]
) -> Dict[str, str]:
    """
    Enhanced response generation with additional context validation
    """
    
    # Check if we have sufficient context for the query
    if retrieval_context and len(retrieval_context) > 0:
        context_quality = sum(1 for ctx in retrieval_context if ctx.get('score', 0) > 0.5)
        
        if context_quality == 0:
            # Low quality context - add disclaimer
            enhanced_input = f"""{user_input}

Note: Limited relevant information found in knowledge base. Please provide the best answer possible based on your general knowledge, and mention if you need more specific information."""
        else:
            enhanced_input = user_input
    else:
        enhanced_input = user_input
    
    return generate_response(
        enhanced_input,
        use_voices,
        random_toggle,
        project=project,
        model=model,
        retrieval_context=retrieval_context
    )

# ========================================================================
# STREAMING RESPONSE GENERATION
# ========================================================================

def generate_streaming_response(
    user_input: str,
    voice: str = "SyntaxPrime",
    project: str = "default",
    model: str = None,
    retrieval_context: List[Dict] = None
):
    """
    Generate streaming response with model filtering
    
    Yields:
        str: Chunks of the response as they're generated
    """
    
    if model is None:
        model = os.getenv("CHAT_MODEL", "openrouter/auto")
    
    filtered_model = filter_model_selection(model)
    
    # Get context
    conversation_context = load_user_history_only(project, max_tokens=500)
    time_context = get_current_time_context()
    
    # Build system context
    system_context = f"""You are Ghostline AI, an advanced personal assistant.

{time_context}

{conversation_context}

{ANSWER_RULES}

Voice: {voice}
Project: {project}
"""
    
    if retrieval_context:
        system_context += "\n\nRelevant context:\n"
        for ctx in retrieval_context[:3]:
            system_context += f"- {ctx.get('content', '')[:200]}...\n"
    
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_input}
    ]
    
    try:
        print(f"🎭 Streaming {voice} response using {filtered_model}")
        
        for chunk in _client.chat_completion(
            model=filtered_model,
            messages=messages,
            temperature=0.7,
            stream=True
        ):
            yield chunk
            
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        yield f"Error: {str(e)}"

# ========================================================================
# UTILITY AND DIAGNOSTIC FUNCTIONS
# ========================================================================

def test_openrouter_connection():
    """Test OpenRouter API connection and model filtering"""
    print("🔍 Testing OpenRouter connection and model filtering...")
    
    # Test basic connection
    try:
        test_response = _client.chat_completion(
            model="openai/gpt-3.5-turbo",  # Simple, reliable model
            messages=[{"role": "user", "content": "Hello, just testing the connection."}],
            temperature=0.1
        )
        
        if 'choices' in test_response:
            print("✅ OpenRouter connection successful")
            return True
        else:
            print("❌ OpenRouter connection failed - no choices in response")
            return False
            
    except Exception as e:
        print(f"❌ OpenRouter connection failed: {e}")
        return False

def get_engine_status():
    """Get comprehensive engine status for diagnostics"""
    return {
        'openrouter_api_key_configured': bool(OPENROUTER_API_KEY),
        'timezone_available': TIMEZONE_AVAILABLE,
        'current_time': get_current_time_context(),
        'model_blacklist': get_model_blacklist_status(),
        'fallback_models': FALLBACK_MODELS,
        'connection_test': test_openrouter_connection() if OPENROUTER_API_KEY else False
    }

# ========================================================================
# PERSONALITY INTEGRATION
# ========================================================================

def apply_personality_modifications(messages: List[Dict], personality: str = "SyntaxPrime") -> List[Dict]:
    """Apply personality-specific modifications to messages"""
    
    personality_prompts = {
        "SyntaxPrime": "You are SyntaxPrime, the primary Ghostline assistant. Be helpful, direct, and professional.",
        "SyntaxBot": "You are SyntaxBot, focused on technical and programming assistance. Be precise and code-focused.",
        "NilExe": "You are NilExe, a philosophical and creative assistant. Be thoughtful, contemplative, and occasionally abstract.",
        "CargoBot": "You are CargoBot, focused on logistics, shipping, and supply chain management.",
        "DataCore": "You are DataCore, specialized in data analysis, statistics, and research."
    }
    
    if personality in personality_prompts:
        # Modify system message to include personality
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += f"\n\n{personality_prompts[personality]}"
        else:
            # Insert personality system message
            messages.insert(0, {
                "role": "system",
                "content": personality_prompts[personality]
            })
    
    return messages

# ========================================================================
# EXPORT FOR MAIN APPLICATION
# ========================================================================

# Main functions for external use
__all__ = [
    'generate_response',
    'generate_response_with_context_check',
    'generate_streaming_response',
    'filter_model_selection',
    'get_model_blacklist_status',
    'get_engine_status',
    'test_openrouter_connection',
    'BLOCKED_MODELS',
    'FALLBACK_MODELS'
]
