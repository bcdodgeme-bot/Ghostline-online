# utils/ghostline_engine.py
# Complete Ghostline Engine with Authentic Personality Integration
# Sectioned for easy editing and maintenance

import os
import json
import requests
from datetime import datetime
from typing import Optional, Iterable, List, Dict

#-------------------------------------------------------------------
# SECTION 1: CONFIGURATION AND CONSTANTS
#-------------------------------------------------------------------

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model filtering configuration - block potentially problematic models
BLOCKED_MODELS = [
    'gpt-5o', 'openai/gpt-5o', 'gpt-5', 'openai/gpt-5',
    'gpt-5-turbo', 'openai/gpt-5-turbo',
    'o1-preview', 'openai/o1-preview', 'o1-mini', 'openai/o1-mini'
]

# Preferred fallback models when auto-selection is blocked
FALLBACK_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "openai/gpt-4o",
    "openai/gpt-4-turbo",
    "meta-llama/llama-3.1-405b-instruct",
    "google/gemini-pro-1.5"
]

# Response quality rules for consistent output
ANSWER_RULES = (
    "Answer ONLY the latest user message. "
    "Do NOT repeat or quote the prompt. "
    "Do NOT invent 'User:'/'Assistant:' transcripts. "
    "Be direct, helpful, and stay in your authentic personality. "
    "One clean answer - no preambles like 'Certainly' or 'Here's your response'."
)

# Timezone handling
try:
    import pytz
    TIMEZONE_AVAILABLE = True
except ImportError:
    TIMEZONE_AVAILABLE = False

#-------------------------------------------------------------------
# SECTION 2: TIME AND CONTEXT UTILITIES
#-------------------------------------------------------------------

def get_current_time_context() -> str:
    """Get current time context for AI responses"""
    try:
        now = datetime.now()
        
        if TIMEZONE_AVAILABLE:
            # Try to get Eastern timezone for Carl's location
            eastern = pytz.timezone('US/Eastern')
            now = now.replace(tzinfo=pytz.UTC).astimezone(eastern)
        
        current_time = now.strftime("%A, %B %d, %Y at %I:%M %p")
        
        # Add time-based context
        hour = now.hour
        if 5 <= hour < 12:
            time_context = f"Current time: {current_time} (Morning)"
        elif 12 <= hour < 17:
            time_context = f"Current time: {current_time} (Afternoon)"
        elif 17 <= hour < 21:
            time_context = f"Current time: {current_time} (Evening)"
        else:
            time_context = f"Current time: {current_time} (Late night)"
        
        return time_context
        
    except Exception as e:
        print(f"Time context error: {e}")
        return f"Current time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}"

def _estimate_tokens(text: str) -> int:
    """Estimate token count for text (rough approximation)"""
    return max(1, len(text.split()) // 0.75)  # ~0.75 words per token average

#-------------------------------------------------------------------
# SECTION 3: MODEL FILTERING AND BLACKLIST SYSTEM
#-------------------------------------------------------------------

def filter_model_selection(model: str) -> str:
    """
    Filter and validate model selection with blacklist protection
    Prevents use of blocked models and provides safe fallbacks
    """
    if not model:
        print(f"⚠️  No model specified, using fallback: {FALLBACK_MODELS[0]}")
        return FALLBACK_MODELS[0]
    
    model_lower = model.lower()
    
    # Check if model is in blacklist
    for blocked in BLOCKED_MODELS:
        if blocked.lower() in model_lower:
            print(f"🚫 Blocked model '{model}' detected, using fallback: {FALLBACK_MODELS[0]}")
            return FALLBACK_MODELS[0]
    
    # Handle special cases
    if model_lower in ['openrouter/auto', 'auto']:
        print(f"🤖 Using OpenRouter auto-selection (blocked models filtered)")
        return model
    
    # Validate that it's a reasonable model format
    if '/' not in model and model not in ['auto']:
        print(f"⚠️  Invalid model format '{model}', using fallback: {FALLBACK_MODELS[0]}")
        return FALLBACK_MODELS[0]
    
    print(f"✅ Model '{model}' approved for use")
    return model

def get_model_blacklist_status() -> dict:
    """Get current model blacklist configuration for diagnostics"""
    return {
        'blocked_models': BLOCKED_MODELS,
        'fallback_models': FALLBACK_MODELS,
        'current_chat_model': os.getenv("CHAT_MODEL", "openrouter/auto"),
        'filtered_chat_model': filter_model_selection(os.getenv("CHAT_MODEL", "openrouter/auto"))
    }

#-------------------------------------------------------------------
# SECTION 4: OPENROUTER CLIENT WITH ENHANCED ERROR HANDLING
#-------------------------------------------------------------------

class OpenRouterClient:
    """Enhanced OpenRouter client with model filtering and robust error handling"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        
        if not api_key:
            print("⚠️  Warning: No OpenRouter API key configured")
    
    def _make_request(self, endpoint: str, data: dict, stream: bool = False):
        """Make HTTP request to OpenRouter API with comprehensive error handling"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://ghostline.ai",
            "X-Title": "Ghostline AI - Personal Assistant"
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
            elif e.response.status_code == 400:
                print("📝 OpenRouter: Bad request")
                raise Exception("OpenRouter API: Bad request. Check your input parameters.")
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
        """Create chat completion with model filtering and validation"""
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

#-------------------------------------------------------------------
# SECTION 5: CONVERSATION HISTORY MANAGEMENT
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 5: CONVERSATION HISTORY MANAGEMENT (FIXED 9/20/25)
#-------------------------------------------------------------------

def _history_path(project: str) -> str:
    """Get history file path for project (kept for backwards compatibility)"""
    return f"sessions/{project.lower().replace(' ', '_')}.json"

def load_user_history_only(project: str, max_tokens: int = 10000) -> str:
    """
    Load recent USER prompts only from DATABASE to avoid response echoing.
    Returns a summary of recent conversation context.
    FIXED: Now reads from database instead of empty files!
    """
    
    print(f"🔍 Loading memory for project '{project}' from database...")
    
    try:
        # Import the database function
        from modules.database import load_conversation_enhanced
        
        # Load conversations from database instead of files
        history = load_conversation_enhanced(project, limit=20)
        
        if not history:
            print(f"⚠️ No conversation history found in database for project '{project}'")
            # Fallback to file-based loading if database fails
            return _load_file_history_fallback(project, max_tokens)
        
        print(f"✅ Found {len(history)} conversations in database for '{project}'")
        
        # Extract user messages only (no assistant responses to avoid echoing)
        user_messages = []
        total_tokens = 0
        
        # Go through history in reverse to get most recent context
        for entry in reversed(history):
            if "user" in entry and entry["user"]:
                user_input = entry["user"]
                tokens = _estimate_tokens(user_input)
                
                if total_tokens + tokens > max_tokens:
                    break
                
                user_messages.insert(0, user_input)
                total_tokens += tokens
        
        if user_messages:
            context = "Recent conversation context (from database):\n"
            for i, msg in enumerate(user_messages[-5:], 1):  # Last 5 user inputs
                context += f"{i}. {msg}\n"
            
            print(f"📝 Generated context from {len(user_messages)} recent messages")
            return context
        
        print("⚠️ No user messages found in database history")
        return ""
        
    except ImportError as e:
        print(f"⚠️ Database module import failed: {e}")
        return _load_file_history_fallback(project, max_tokens)
    except Exception as e:
        print(f"⚠️ Database history loading failed: {e}")
        return _load_file_history_fallback(project, max_tokens)

def _load_file_history_fallback(project: str, max_tokens: int = 10000) -> str:
    """
    Fallback method: Load from files if database fails
    (This is the original method, kept as backup)
    """
    print(f"📁 Falling back to file-based history for project '{project}'")
    
    history_file = _history_path(project)
    
    if not os.path.exists(history_file):
        print(f"⚠️ No history file found: {history_file}")
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
            context = "Recent conversation context (from files):\n"
            for i, msg in enumerate(user_messages[-5:], 1):  # Last 5 user inputs
                context += f"{i}. {msg}\n"
            return context
        
        return ""
        
    except Exception as e:
        print(f"📁 File history loading error: {e}")
        return ""

def save_conversation_to_database(project: str, user_input: str, responses: dict):
    """
    Save conversation to database using the database module
    This ensures conversations are stored where they can be retrieved for memory
    """
    try:
        from modules.database import save_conversation_enhanced
        
        # Save to database
        chat_id = save_conversation_enhanced(
            project=project,
            user_input=user_input,
            response_data=responses
        )
        
        if chat_id:
            print(f"💾 Conversation saved to database (ID: {chat_id})")
        else:
            print("⚠️ Failed to save conversation to database")
            
        return chat_id
        
    except ImportError as e:
        print(f"⚠️ Database module not available for saving: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Failed to save conversation to database: {e}")
        return None

def get_conversation_stats(project: str) -> dict:
    """
    Get conversation statistics for diagnostics
    """
    stats = {
        'project': project,
        'database_conversations': 0,
        'file_conversations': 0,
        'memory_source': 'unknown'
    }
    
    try:
        # Check database conversations
        from modules.database import load_conversation_enhanced
        db_history = load_conversation_enhanced(project, limit=100)
        stats['database_conversations'] = len(db_history) if db_history else 0
        stats['memory_source'] = 'database' if stats['database_conversations'] > 0 else 'files'
        
    except Exception as e:
        print(f"⚠️ Could not get database stats: {e}")
    
    try:
        # Check file conversations
        history_file = _history_path(project)
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                file_history = json.load(f)
                stats['file_conversations'] = len(file_history) if isinstance(file_history, list) else 0
    except Exception as e:
        print(f"⚠️ Could not get file stats: {e}")
    
    return stats

#-------------------------------------------------------------------
# SECTION 6: AUTHENTIC PERSONALITY INTEGRATION
#-------------------------------------------------------------------

def get_personality_system():
    """Get the personality system with error handling"""
    try:
        from modules.personalities import PersonalityIntegration
        return PersonalityIntegration()
    except ImportError as e:
        print(f"⚠️  Personality system import failed: {e}")
        return None

def apply_authentic_personality(messages: List[Dict], voice: str) -> List[Dict]:
    """
    Apply authentic personality from the database-trained personality system
    This replaces the old generic personality prompts with the real ones
    """
    try:
        personality_integration = get_personality_system()
        
        if personality_integration:
            # Use the authentic personality system
            config = personality_integration.personality_system.get_personality_config(voice.lower())
            authentic_prompt = config['system_prompt']
            
            print(f"🎭 Applying authentic {voice} personality ({len(authentic_prompt)} chars)")
            
            # Modify system message to include authentic personality
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{authentic_prompt}"
            else:
                # Insert personality system message
                messages.insert(0, {
                    "role": "system",
                    "content": authentic_prompt
                })
        else:
            # Fallback to basic personality if import fails
            print(f"⚠️  Using fallback personality for {voice}")
            fallback_prompt = f"You are {voice}, respond naturally in your authentic voice."
            
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{fallback_prompt}"
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": fallback_prompt
                })
        
        return messages
        
    except Exception as e:
        print(f"❌ Personality application failed: {e}")
        return messages

def apply_personality_post_processing(response: str, voice: str) -> str:
    """
    Apply personality-specific post-processing filters
    """
    try:
        personality_integration = get_personality_system()
        
        if personality_integration:
            processed_response = personality_integration.process_personality_response(response, voice.lower())
            if processed_response != response:
                print(f"🎨 Applied {voice} post-processing filters")
            return processed_response
        else:
            return response
            
    except Exception as e:
        print(f"⚠️  Post-processing failed for {voice}: {e}")
        return response

#-------------------------------------------------------------------
# SECTION 7: ENHANCED RESPONSE GENERATION
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 7: ENHANCED RESPONSE GENERATION 9/15/25
#-------------------------------------------------------------------

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
    Enhanced response generation with authentic personality integration
    
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
    
    # Apply model filtering for safety
    filtered_model = filter_model_selection(model)
    
    # Get conversation history context
    conversation_context = load_user_history_only(project, max_tokens=10000)
    
    # Get current time context
    time_context = get_current_time_context()
    
    # Build comprehensive system context
    system_context = f"""You are Ghostline AI, Carl's advanced personal assistant and creative partner.

{time_context}

{conversation_context}

{ANSWER_RULES}

Current project context: {project}
"""
    
    # Add retrieval context if available
    if retrieval_context and len(retrieval_context) > 0:
        system_context += "\n\nRelevant knowledge base context:\n"
        for i, ctx in enumerate(retrieval_context[:3], 1):  # Limit to 3 most relevant
            system_context += f"{i}. {ctx.get('text', '')[:200]}...\n"
    
    # PERSONALITY REINFORCEMENT - Kill safe mode for personal use
    system_context += "\n\nIMPORTANT: Respond as your authentic personality with full voice. No corporate speak, no safety disclaimers, no 'I aim to be precise' language. Be sarcastic, direct, and conversational. This is personal use - be yourself with Carl's family, projects, and chaos. Reference Miller, Ghada, coffee habits, and 2am coding naturally."
    
    # Generate responses for each requested voice
    responses = {}
    
    for voice in use_voices:
        try:
            print(f"🎭 Generating {voice} response using {filtered_model}")
            
            # Prepare messages with base system context
            messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_input}
            ]
            
            # Apply authentic personality to messages
            messages = apply_authentic_personality(messages, voice)
            
            # Make API call
            response = _client.chat_completion(
                model=filtered_model,
                messages=messages,
                temperature=0.7
            )
            
            # Extract response content
            if 'choices' in response and len(response['choices']) > 0:
                raw_content = response['choices'][0]['message']['content']
                
                # Apply personality post-processing
                processed_content = apply_personality_post_processing(raw_content, voice)
                
                responses[voice] = processed_content
                print(f"✅ {voice} response generated successfully")
            else:
                responses[voice] = f"Error: No response generated for {voice}"
                print(f"❌ No response generated for {voice}")
                
        except Exception as e:
            print(f"❌ Error generating {voice} response: {e}")
            responses[voice] = f"Error generating {voice} response: {str(e)}"
    
    return responses

#-------------------------------------------------------------------
# SECTION 8: CONTEXT-AWARE RESPONSE GENERATION
#-------------------------------------------------------------------

def generate_response_with_context_check(
    user_input: str,
    use_voices: List[str],
    random_toggle: bool,
    project: str = "default",
    model: str = None,
    retrieval_context: List[Dict] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Generate response with intelligent context checking for follow-up questions
    """
    
    # Check if this looks like a context-dependent follow-up
    context_indicators = [
        'this', 'that', 'it', 'they', 'them', 'what about', 'how about',
        'and also', 'in addition', 'furthermore', 'what do you think',
        'any thoughts', 'your opinion', 'what would you', 'should i'
    ]
    
    user_lower = user_input.lower()
    needs_context = any(indicator in user_lower for indicator in context_indicators)
    
    if needs_context and len(user_input.split()) < 20:
        # This looks like a follow-up question - enhance with context
        enhanced_prompt = f"""This appears to be a follow-up question that might reference previous context.

User's question: "{user_input}"

Please provide the best answer possible based on your general knowledge and conversation context. If the question references something specific from our conversation history, use that context appropriately."""
        
        return generate_response(
            enhanced_prompt, use_voices, random_toggle,
            project=project, model=model, retrieval_context=retrieval_context
        )
    else:
        # Standard response generation
        return generate_response(
            user_input, use_voices, random_toggle,
            project=project, model=model, retrieval_context=retrieval_context
        )

#-------------------------------------------------------------------
# SECTION 9: STREAMING RESPONSE GENERATION
#-------------------------------------------------------------------

def generate_streaming_response(
    user_input: str,
    voice: str = "SyntaxPrime",
    project: str = "default",
    model: str = None,
    retrieval_context: List[Dict] = None
):
    """
    Generate streaming response with authentic personality integration
    
    Yields:
        str: Chunks of the response as they're generated
    """
    
    if model is None:
        model = os.getenv("CHAT_MODEL", "openrouter/auto")
    
    filtered_model = filter_model_selection(model)
    
    # Get context
    conversation_context = load_user_history_only(project, max_tokens=10000)
    time_context = get_current_time_context()
    
    # Build system context
    system_context = f"""You are Ghostline AI, Carl's advanced personal assistant and creative partner.

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
    
    # Prepare messages
    messages = [
        {"role": "system", "content": system_context},
        {"role": "user", "content": user_input}
    ]
    
    # Apply authentic personality
    messages = apply_authentic_personality(messages, voice)
    
    try:
        print(f"🎭 Streaming {voice} response using {filtered_model}")
        
        # Stream the response
        response_chunks = []
        for chunk in _client.chat_completion(
            model=filtered_model,
            messages=messages,
            temperature=0.7,
            stream=True
        ):
            response_chunks.append(chunk)
            yield chunk
        
        # Apply post-processing to complete response
        complete_response = ''.join(response_chunks)
        processed_response = apply_personality_post_processing(complete_response, voice)
        
        # If post-processing changed the response significantly, indicate this
        if len(processed_response) != len(complete_response):
            print(f"🎨 Post-processing applied to {voice} streaming response")
            
    except Exception as e:
        print(f"❌ Streaming failed: {e}")
        yield f"Error: {str(e)}"

#-------------------------------------------------------------------
# SECTION 10: UTILITY AND DIAGNOSTIC FUNCTIONS
#-------------------------------------------------------------------

def test_openrouter_connection():
    """Test OpenRouter API connection and model filtering"""
    print("🔍 Testing OpenRouter connection and model filtering...")
    
    if not OPENROUTER_API_KEY:
        print("❌ No OpenRouter API key configured")
        return False
    
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

def test_personality_integration():
    """Test the personality system integration"""
    print("🎭 Testing personality system integration...")
    
    personality_integration = get_personality_system()
    if personality_integration:
        print("✅ Personality system loaded successfully")
        
        # Test each personality
        for personality_id in ['syntaxprime', 'syntaxbot', 'nilexe', 'ggpt']:
            try:
                config = personality_integration.personality_system.get_personality_config(personality_id)
                print(f"  ✅ {config['name']}: {len(config['system_prompt'])} char prompt")
            except Exception as e:
                print(f"  ❌ {personality_id}: {e}")
        return True
    else:
        print("❌ Personality system failed to load")
        return False

def get_engine_status():
    """Get comprehensive engine status for diagnostics"""
    return {
        'openrouter_api_key_configured': bool(OPENROUTER_API_KEY),
        'timezone_available': TIMEZONE_AVAILABLE,
        'current_time': get_current_time_context(),
        'model_blacklist': get_model_blacklist_status(),
        'fallback_models': FALLBACK_MODELS,
        'connection_test': test_openrouter_connection() if OPENROUTER_API_KEY else False,
        'personality_system': test_personality_integration()
    }

#-------------------------------------------------------------------
# SECTION 11: EXPORT FOR MAIN APPLICATION
#-------------------------------------------------------------------

# Main functions for external use
__all__ = [
    'generate_response',
    'generate_response_with_context_check',
    'generate_streaming_response',
    'filter_model_selection',
    'get_model_blacklist_status',
    'get_engine_status',
    'test_openrouter_connection',
    'test_personality_integration',
    'apply_authentic_personality',
    'apply_personality_post_processing',
    'BLOCKED_MODELS',
    'FALLBACK_MODELS'
]

# Initialize and test systems on import
if __name__ == "__main__":
    print("=== GHOSTLINE ENGINE INITIALIZATION TEST ===")
    print(f"OpenRouter API Key: {'✅ Configured' if OPENROUTER_API_KEY else '❌ Missing'}")
    print(f"Timezone Support: {'✅ Available' if TIMEZONE_AVAILABLE else '❌ Missing'}")
    
    # Test personality system
    test_personality_integration()
    
    # Test OpenRouter connection if API key is available
    if OPENROUTER_API_KEY:
        test_openrouter_connection()
    
    print("=== ENGINE READY ===")

#-------------------------------------------------------------------
# SECTION 12: FEEDBACK-AWARE RESPONSE GENERATION
#-------------------------------------------------------------------

def get_feedback_learning_engine():
    """Get the feedback learning engine with error handling"""
    try:
        from modules.feedback_learning import FeedbackLearningEngine
        return FeedbackLearningEngine()
    except ImportError as e:
        print(f"⚠️  Feedback learning system import failed: {e}")
        return None

def apply_feedback_enhanced_personality(messages: List[Dict], voice: str) -> List[Dict]:
    """
    Apply personality enhanced with feedback learning data
    This is the SMART version that learns from 🖕 ratings!
    """
    try:
        # Get both personality and learning systems
        personality_integration = get_personality_system()
        learning_engine = get_feedback_learning_engine()
        
        if personality_integration and learning_engine:
            # Get base personality
            config = personality_integration.personality_system.get_personality_config(voice.lower())
            base_prompt = config['system_prompt']
            
            # Enhance with feedback learning
            enhanced_prompt = learning_engine.get_personality_enhancement(voice, base_prompt)
            
            # Check if enhancement actually happened
            if len(enhanced_prompt) > len(base_prompt):
                print(f"🧠 Applied feedback learning to {voice} (+{len(enhanced_prompt) - len(base_prompt)} chars)")
            else:
                print(f"🎭 Using base {voice} personality (no learning data yet)")
            
            # Apply enhanced personality to messages
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{enhanced_prompt}"
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": enhanced_prompt
                })
                
        elif personality_integration:
            # Fallback to base personality system
            print(f"⚠️  Learning engine unavailable, using base {voice} personality")
            return apply_authentic_personality(messages, voice)
        else:
            # Final fallback
            print(f"⚠️  No personality systems available for {voice}")
            fallback_prompt = f"You are {voice}, respond naturally in your authentic voice."
            
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += f"\n\n{fallback_prompt}"
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": fallback_prompt
                })
        
        return messages
        
    except Exception as e:
        print(f"❌ Feedback-enhanced personality application failed: {e}")
        return apply_authentic_personality(messages, voice)  # Fallback

def generate_feedback_aware_response(
    user_input: str,
    use_voices: List[str],
    random_toggle: bool,
    project: str = "default",
    model: str = None,
    retrieval_context: List[Dict] = None,
    **kwargs
) -> Dict[str, str]:
    """
    Generate response using feedback-enhanced personalities
    This replaces generate_response when feedback learning is active
    """
    
    # Use environment model if none specified, then filter it
    if model is None:
        model = os.getenv("CHAT_MODEL", "openrouter/auto")
    
    # Apply model filtering for safety
    filtered_model = filter_model_selection(model)
    
    # Get conversation history context
    conversation_context = load_user_history_only(project, max_tokens=10000)
    
    # Get current time context
    time_context = get_current_time_context()
    
    # Build comprehensive system context
    system_context = f"""You are Ghostline AI, Carl's advanced personal assistant and creative partner.

{time_context}

{conversation_context}

{ANSWER_RULES}

Current project context: {project}
"""
    
    # Add retrieval context if available
    if retrieval_context and len(retrieval_context) > 0:
        system_context += "\n\nRelevant knowledge base context:\n"
        for i, ctx in enumerate(retrieval_context[:3], 1):  # Limit to 3 most relevant
            system_context += f"{i}. {ctx.get('text', '')[:200]}...\n"
    
    # Generate responses for each requested voice
    responses = {}
    learning_engine = get_feedback_learning_engine()
    
    for voice in use_voices:
        try:
            print(f"🧠 Generating feedback-aware {voice} response using {filtered_model}")
            
            # Prepare messages with base system context
            messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_input}
            ]
            
            # Apply feedback-enhanced personality to messages
            messages = apply_feedback_enhanced_personality(messages, voice)
            
            # Make API call
            response = _client.chat_completion(
                model=filtered_model,
                messages=messages,
                temperature=0.7
            )
            
            # Extract response content
            if 'choices' in response and len(response['choices']) > 0:
                raw_content = response['choices'][0]['message']['content']
                
                # Check for negative patterns if learning engine is available
                if learning_engine:
                    warnings = learning_engine.should_avoid_pattern(raw_content, voice)
                    if warnings:
                        print(f"⚠️  {voice} response warnings: {warnings}")
                
                # Apply personality post-processing
                processed_content = apply_personality_post_processing(raw_content, voice)
                
                responses[voice] = processed_content
                print(f"✅ {voice} response generated with feedback awareness")
            else:
                responses[voice] = f"Error: No response generated for {voice}"
                print(f"❌ No response generated for {voice}")
                
        except Exception as e:
            print(f"❌ Error generating feedback-aware {voice} response: {e}")
            responses[voice] = f"Error generating {voice} response: {str(e)}"
    
    return responses

def log_response_for_learning(responses: Dict[str, str], user_input: str, project: str):
    """
    Log responses for potential future learning
    This helps track response patterns for analysis
    """
    try:
        learning_engine = get_feedback_learning_engine()
        if not learning_engine:
            return
        
        # Simple logging of response characteristics for future analysis
        for voice, response in responses.items():
            response_stats = {
                'voice': voice,
                'project': project,
                'user_input_length': len(user_input.split()),
                'response_length': len(response.split()),
                'has_humor': any(indicator in response.lower() for indicator in ['lol', '😂', 'sarcasm', 'chaos']),
                'has_memory_ref': any(ref in response.lower() for ref in ['remember', 'coffee', '2am', 'chaos']),
                'timestamp': datetime.now().isoformat()
            }
            
            # This could be stored for future analysis if needed
            print(f"📊 Response logged for learning: {voice} - {response_stats['response_length']} words")
            
    except Exception as e:
        print(f"⚠️  Response logging failed: {e}")

# Update the export list to include new functions
def get_feedback_aware_engine_status():
    """Get engine status including feedback learning capabilities"""
    base_status = get_engine_status()
    
    # Add feedback learning status
    learning_engine = get_feedback_learning_engine()
    base_status['feedback_learning'] = {
        'available': learning_engine is not None,
        'status': 'active' if learning_engine else 'unavailable'
    }
    
    if learning_engine:
        try:
            # Test analysis capability
            analysis = learning_engine.analyze_perfect_personality_responses("SyntaxPrime")
            base_status['feedback_learning']['perfect_responses'] = analysis.get('total_perfect_responses', 0)
            base_status['feedback_learning']['learning_active'] = analysis.get('total_perfect_responses', 0) >= 3
        except Exception as e:
            base_status['feedback_learning']['error'] = str(e)
    
    return base_status

# Add to __all__ export list:
__all__.extend([
    'generate_feedback_aware_response',
    'apply_feedback_enhanced_personality',
    'log_response_for_learning',
    'get_feedback_aware_engine_status'
])

#-------------------------------------------------------------------
# SECTION 13: WEATHER INTEGRATION FOR HEALTH MONITORING
#-------------------------------------------------------------------

# Fix for utils/ghostline_engine.py - Weather Integration Import Issue
# Replace the existing import block in SECTION 13: WEATHER INTEGRATION FOR HEALTH MONITORING

#-------------------------------------------------------------------
# SECTION 13: WEATHER INTEGRATION FOR HEALTH MONITORING
#-------------------------------------------------------------------

# Import the weather module with correct function names
try:
    from modules.weather_integration import (
        handle_comprehensive_weather_command,  # Fixed: was handle_weather_command
        handle_weather_alerts_command,
        handle_weather_integration,            # Added: main integration router
        detect_weather_command,               # Added: command detection
        WEATHER_COMMANDS,
        get_weather_monitor,
        is_weather_configured,
        get_weather_status
    )
    WEATHER_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Weather integration not available: {e}")
    WEATHER_INTEGRATION_AVAILABLE = False

def detect_weather_command(user_input: str) -> bool:
    """Detect if user input is requesting weather information"""
    if not WEATHER_INTEGRATION_AVAILABLE:
        return False
    
    weather_keywords = [
        'weather', 'pressure', 'barometric', 'headache weather',
        'uv', 'uv index', 'sun', 'sunny', 'weather alerts',
        'weather pattern', 'pressure drop', 'pressure history',
        'tomorrow weather', 'weather now', 'weather current'
    ]
    
    user_lower = user_input.lower()
    return any(keyword in user_lower for keyword in weather_keywords)

def handle_weather_integration(user_input: str, project: str) -> Optional[Dict[str, str]]:
    """Handle weather-related commands and integrate with conversation"""
    if not WEATHER_INTEGRATION_AVAILABLE:
        return None
    
    user_lower = user_input.lower().strip()
    
    # Direct weather command mapping
    for command, handler in WEATHER_COMMANDS.items():
        if command in user_lower:
            try:
                return handler(user_input, project)
            except Exception as e:
                return {"SyntaxPrime": f"🌦️ Weather command failed: {str(e)}"}
    
    # Fallback to general weather handler for weather-related queries
    if detect_weather_command(user_input):
        try:
            # Use the correct function name
            return handle_comprehensive_weather_command(user_input, project)
        except Exception as e:
            return {"SyntaxPrime": f"🌦️ Weather integration error: {str(e)}"}
    
    return None
