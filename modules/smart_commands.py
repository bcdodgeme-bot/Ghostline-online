# modules/smart_commands.py - COMPREHENSIVE FIX for command over-triggering
# This fixes ALL overly aggressive keyword triggers causing false command activation
# PATCH: Fix content creation mode loops with timeout and simpler exits

import os
import re
import datetime
from utils.ghostline_engine import generate_response
from utils.rag_basic import is_ready
from modules.database import save_conversation_enhanced, save_daily_log_enhanced

# Import existing handlers
from modules.gmail import (
    handle_good_morning_command, handle_overnight_command,
    handle_calendar_today_command, handle_next_meeting_command
)

CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

# FIXED: Ultra-precise content mode patterns with strict word boundaries
CONTENT_MODES = {
    "email": {
        "prompt": "We are writing an email. Use professional tone, clear CTAs, proper email structure.",
        "patterns": [
            r'\bwrite\s+an?\s+email\b',
            r'\bdraft\s+an?\s+email\b',
            r'\bemail\s+for\s+\w+\b',
            r'\bcompose\s+an?\s+email\b',
            r'\bemail\s+about\s+\w+\b',
            r'\bgiving\s+circle\s+email\b',
            r'\bnewsletter\s+email\b',
            r'\bsend\s+an?\s+email\b'
        ],
        "tone_questions": ["What's the target audience?", "Professional or casual tone?", "Any specific CTAs needed?"]
    },
    "blog": {
        "prompt": "We are writing a blog post. Use SEO best practices, engaging headlines, proper meta descriptions.",
        "patterns": [
            r'\bwrite\s+a\s+blog\b',
            r'\bblog\s+post\s+about\b',
            r'\bdraft\s+a\s+blog\b',
            r'\bwrite\s+an?\s+article\b',
            r'\bblog\s+article\s+about\b',
            r'\bcreate\s+a\s+blog\b'
        ],
        "tone_questions": ["What's the tone?", "Any target keywords for SEO?", "What's the main message?"]
    },
    "sms": {
        "prompt": "We are writing SMS content. Keep under 160 characters, clear action, urgent tone.",
        "patterns": [
            r'\bwrite\s+an?\s+sms\b',
            r'\btext\s+message\s+for\b',
            r'\bsms\s+for\s+\w+\b',
            r'\btext\s+about\s+\w+\b',
            r'\bsend\s+a\s+text\b'
        ],
        "tone_questions": ["What's the main action?", "How urgent is this?"]
    },
    "social": {
        "prompt": "We are writing social media content. Platform-optimized, engaging, hashtag-ready.",
        "patterns": [
            r'\bsocial\s+media\s+post\b',
            r'\bsocial\s+media\s+content\b',
            r'\bwrite\s+a\s+tweet\b',
            r'\binstagram\s+post\s+(for|about)\b',
            r'\bfacebook\s+post\s+(for|about)\b',
            r'\blinkedin\s+post\s+(for|about)\b',
            r'\bcreate\s+social\s+content\b'
        ],
        "tone_questions": ["Which platform?", "Professional or casual tone?", "Any specific hashtags needed?"]
    }
}

# FIXED: Ultra-specific analysis mode patterns
ANALYSIS_MODES = {
    "marketing_plan": {
        "prompt": "We are developing a marketing plan. Focus on strategy, target audience analysis, channel selection.",
        "patterns": [
            r'\bmarketing\s+plan\s+(for|about)\b',
            r'\bmarketing\s+strategy\s+(for|about)\b',
            r'\bcampaign\s+plan\s+(for|about)\b',
            r'\bmarketing\s+analysis\s+(of|for)\b',
            r'\bbrand\s+strategy\s+(for|about)\b',
            r'\bcreate\s+a\s+marketing\s+plan\b',
            r'\bdevelop\s+a\s+marketing\s+strategy\b'
        ],
        "questions": ["What's the product/service?", "Target audience?", "Budget range?"]
    },
    "board_report": {
        "prompt": "We are writing a board report. Use executive summary format, data-driven insights, clear recommendations.",
        "patterns": [
            r'\bboard\s+report\s+(for|about)\b',
            r'\bexecutive\s+report\s+(for|about)\b',
            r'\bquarterly\s+report\s+(for|about)\b',
            r'\bboard\s+presentation\s+(for|about)\b',
            r'\bexecutive\s+summary\s+(for|of)\b',
            r'\bwrite\s+a\s+board\s+report\b',
            r'\bcreate\s+a\s+board\s+report\b'
        ],
        "questions": ["Which quarter/period?", "Key metrics to highlight?", "Any major decisions needed?"]
    },
    "competitive_analysis": {
        "prompt": "We are conducting competitive analysis. Focus on market positioning, competitive advantages.",
        "patterns": [
            r'\bcompetitive\s+analysis\s+(of|for)\b',
            r'\bcompetitor\s+research\s+(on|for)\b',
            r'\bmarket\s+analysis\s+(of|for)\b',
            r'\bcompetition\s+review\s+(of|for)\b',
            r'\bcompetitor\s+study\s+(of|for)\b'
        ],
        "questions": ["Who are the main competitors?", "What market segment?"]
    }
}

# UPDATED: Global variables for mode tracking with timeout
_current_content_mode = None
_current_context = {}
_mode_start_time = None  # ADDED: Track when mode started

def matches_pattern(text, patterns):
    """Check if text matches any of the regex patterns (case insensitive)"""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def detect_intent(user_input):
    """ULTRA-PRECISE intent detection with bulletproof exit handling and timeout"""
    global _current_content_mode, _current_context, _mode_start_time
    lower_input = user_input.lower().strip()
    
    # ADDED: Auto-exit after 10 minutes in any mode to prevent infinite loops
    if _current_content_mode and _mode_start_time:
        time_in_mode = (datetime.datetime.now() - _mode_start_time).total_seconds()
        if time_in_mode > 600:  # 10 minutes
            print(f"Auto-exiting {_current_content_mode} mode due to timeout ({time_in_mode/60:.1f} minutes)")
            _current_content_mode = None
            _current_context.clear()
            _mode_start_time = None
            return "casual"
    
    # ADDED: Simple, foolproof exit patterns that ALWAYS work
    emergency_exits = [
        r'^stop$', r'^exit$', r'^done$', r'^finished$', r'^cancel$',
        r'^quit$', r'^end$', r'^abort$', r'^clear$', r'^reset$',
        r'^new topic$', r'^different question$', r'^something else$',
        r'^never mind$', r'^forget it$', r'^start over$'
    ]
    
    if any(re.search(pattern, lower_input) for pattern in emergency_exits):
        print(f"Emergency exit detected: '{user_input}'")
        _current_content_mode = None
        _current_context.clear()
        _mode_start_time = None
        return "casual"
    
    # IMPROVED: More reliable exit pattern detection with context awareness
    exit_patterns = [
        r'^that\'?s\s+done\.?$',
        r'^that\'?s\s+finished\.?$',
        r'^that\'?s\s+complete\.?$',
        r'^okay,?\s+that\'?s\s+(done|finished|complete)\.?$',
        r'^let\'?s\s+move\s+on\.?$',
        r'^next\.?$',
        r'^now\s+let\'?s\b',
        r'^done\s+with\s+that\.?$',
        r'^finished\s+with\s+(that|this)\.?$',
        r'^good,?\s+now\s+let\'?s\b',  # ADDED
        r'^perfect,?\s+now\s+\w+',      # ADDED
        r'^great,?\s+what\s+about\b'    # ADDED
    ]
    
    if any(re.search(pattern, lower_input) for pattern in exit_patterns):
        print(f"Exiting content mode due to completion signal: '{user_input}'")
        _current_content_mode = None
        _current_context.clear()
        _mode_start_time = None
        return "casual"
    
    # ADDED: Board report specific exit patterns (the problematic loop)
    if _current_content_mode == "board_report":
        board_completion_signals = [
            r'^that\'?s\s+the\s+report\.?$',
            r'^that\'?s\s+enough\s+for\s+the\s+report\.?$',
            r'^board\s+report\s+is\s+done\.?$',
            r'^finished\s+with\s+the\s+board\s+report\.?$',
            r'^report\s+complete\.?$',
            r'^good\s+board\s+report$',  # ADDED
            r'^that\s+works\s+for\s+the\s+board$'  # ADDED
        ]
        
        if any(re.search(pattern, lower_input) for pattern in board_completion_signals):
            print(f"Board report mode exit detected: '{user_input}'")
            _current_content_mode = None
            _current_context.clear()
            _mode_start_time = None
            return "casual"
    
    # FIXED: Ultra-precise content mode transitions (only when explicitly requested)
    transition_patterns = [
        r'^let\'?s\s+write\s+social\s+media\b',
        r'^now\s+create\s+social\s+media\b',
        r'^social\s+media\s+posts?\s+for\s+this\b'
    ]
    
    if any(re.search(pattern, lower_input) for pattern in transition_patterns):
        _current_content_mode = "social"
        _mode_start_time = datetime.datetime.now()  # ADDED: Set start time
        print(f"Transitioning to social content mode")
        return "content_creation"
    
    # If we're in content mode, check if this is still content-related
    if _current_content_mode:
        # IMPROVED: Ultra-specific casual conversation indicators that ALWAYS break content mode
        casual_conversation_indicators = [
            r'^(hello|hi|hey)(\s+\w+)?\.?$',  # Simple greetings only
            r'^(thanks?|thank\s+you)\.?$',
            r'^(good\s+morning|good\s+afternoon|good\s+evening)\.?$',
            r'^how\s+are\s+you(\s+doing)?\.?$',
            r'^what\'?s\s+up\.?$',
            r'^(awesome|great|nice|cool|perfect)\.?$',  # UPDATED: added 'perfect'
            r'^(got\s+it|understood|perfect)\.?$',
            r'^\w*\s*praying\s+fajr\b',  # Religious activities
            r'^cup\s+(one|two|three|four|five|\d+)\b',  # Coffee references
            r'^(coffee|tea)\s+time\b',
            r'^taking\s+a\s+break\b',
            r'^just\s+(checking|saying)\s+hi\b',
            r'^what\s+else\s+can\s+you\s+do\b',  # ADDED
            r'^tell\s+me\s+about\s+\w+',         # ADDED
            r'^how\s+do\s+i\s+\w+'               # ADDED
        ]
        
        if any(re.search(pattern, lower_input) for pattern in casual_conversation_indicators):
            print(f"Breaking out of {_current_content_mode} mode for casual conversation: '{lower_input}'")
            _current_content_mode = None
            _current_context.clear()
            _mode_start_time = None
            return "casual"
        
        # Continue in current mode only if input is clearly content-related
        content_continuation_patterns = [
            r'\b(write|draft|create|edit|revise|update|change|modify)\b',  # UPDATED: added 'modify'
            r'\b(add|include|mention|focus|emphasize|highlight)\b',        # UPDATED: added 'highlight'
            r'\b(tone|style|format|structure|section|paragraph)\b',
            r'\b(make\s+it|can\s+you|please|also)\b'  # ADDED
        ]
        
        if any(re.search(pattern, lower_input) for pattern in content_continuation_patterns):
            print(f"Continuing in {_current_content_mode} content mode")
            return "content_creation" if _current_content_mode in CONTENT_MODES else "analysis_mode"
        else:
            # If not clearly content-related, exit mode
            print(f"Exiting {_current_content_mode} mode - input not content-related: '{lower_input}'")
            _current_content_mode = None
            _current_context.clear()
            _mode_start_time = None
            return "casual"
    
    # FIXED: Check for new content mode entry with ultra-precise patterns
    for mode, config in CONTENT_MODES.items():
        if matches_pattern(user_input, config["patterns"]):
            _current_content_mode = mode
            _current_context = {"initial_input": user_input}
            _mode_start_time = datetime.datetime.now()  # ADDED: Set start time
            print(f"Entering {mode} content mode at {_mode_start_time}")
            return "content_creation"
    
    # FIXED: Check for analysis mode entry with ultra-precise patterns
    for mode, config in ANALYSIS_MODES.items():
        if matches_pattern(user_input, config["patterns"]):
            _current_content_mode = mode
            _current_context = {"initial_input": user_input}
            _mode_start_time = datetime.datetime.now()  # ADDED: Set start time
            print(f"Entering {mode} analysis mode at {_mode_start_time}")
            return "analysis_mode"
    
    # FIXED: Ultra-specific casual/greeting patterns that NEVER trigger commands
    casual_patterns = [
        r'^(hello|hi|hey)(\s+syntax)?(\s+prime)?\.?$',
        r'^(good\s+morning|good\s+afternoon|good\s+evening)(\s+syntax)?\.?$',
        r'^how\s+are\s+you(\s+doing)?\.?$',
        r'^what\'?s\s+up\.?$',
        r'^(thanks?|thank\s+you)(\s+syntax)?\.?$',
        r'^(ok|okay|cool|great|nice|got\s+it|understood|perfect)\.?$',
        r'^(hello|hi|hey)\s+syntax(\s+prime)?\.?$',
        r'^\w*\s*praying\s+fajr\b',
        r'^cup\s+(one|two|three|four|five|\d+)(\s+of\s+(coffee|tea))?\.?$',
        r'^(coffee|tea)\s+break\b',
        r'^just\s+(saying\s+)?hi\b'
    ]
    
    # Ultra-specific briefing keywords that NEVER match casual conversation
    briefing_keywords = [
        r'\bdaily\s+briefing\b',
        r'\bbrief\s+me\s+on\b',
        r'\bcatch\s+me\s+up\s+on\b',
        r'\bmorning\s+update\b',
        r'\bstart\s+my\s+day\b'
    ]
    
    is_casual = any(re.search(pattern, lower_input) for pattern in casual_patterns)
    is_briefing = any(re.search(pattern, lower_input) for pattern in briefing_keywords)
    
    if is_casual and not is_briefing:
        print(f"Detected casual greeting: '{lower_input}'")
        return "casual"
    
    # FIXED: Ultra-specific morning briefing patterns
    morning_patterns = [
        r'^daily\s+briefing\.?$',
        r'^brief\s+me\.?$',
        r'^catch\s+me\s+up\.?$',
        r'^morning\s+update\.?$',
        r'^daily\s+summary\.?$',
        r'^what\'?s\s+today\'?s\s+(schedule|agenda)\b',
        r'^what\s+do\s+i\s+need\s+to\s+know\s+today\b',
        r'^morning\s+sync\.?$',
        r'^daily\s+intel\.?$',
        r'^start\s+my\s+day\.?$'
    ]
    
    # Ultra-specific productivity patterns
    productivity_patterns = [
        r'^what\s+should\s+i\s+work\s+on\s+today\b',
        r'^my\s+top\s+priorities\b',
        r'^what\'?s\s+due\s+today\b',
        r'^today\'?s\s+deadlines\b',
        r'^my\s+task\s+list\b',
        r'^task\s+summary\s+for\s+today\b',
        r'^work\s+focus\s+for\s+today\b'
    ]
    
    # Ultra-specific relationship patterns
    relationship_patterns = [
        r'^who\s+should\s+i\s+follow\s+up\s+with\b',
        r'^pipeline\s+update\b',
        r'^deals\s+status\b',
        r'^crm\s+update\b',
        r'^follow\s+up\s+reminders\b'
    ]
    
    # Ultra-specific email patterns
    email_patterns = [
        r'^overnight\s+emails?\b',
        r'^check\s+emails?\b',
        r'^inbox\s+status\b',
        r'^new\s+mail\b',
        r'^email\s+summary\b'
    ]
    
    # Ultra-specific calendar patterns
    calendar_patterns = [
        r'^calendar\s+for\s+today\b',
        r'^today\'?s\s+meetings\b',
        r'^my\s+schedule\s+today\b',
        r'^next\s+meeting\b'
    ]
    
    # Ultra-specific question patterns
    specific_patterns = [
        r'^what\s+is\s+\w+\b',
        r'^what\s+does\s+\w+\b',
        r'^tell\s+me\s+about\s+\w+\b',
        r'^explain\s+\w+\b',
        r'^describe\s+\w+\b',
        r'^how\s+does\s+\w+\b'
    ]
    
    # Check patterns in priority order with ultra-precise matching
    if any(re.search(pattern, lower_input) for pattern in morning_patterns):
        return "morning_briefing"
    elif any(re.search(pattern, lower_input) for pattern in productivity_patterns):
        return "productivity_focus"
    elif any(re.search(pattern, lower_input) for pattern in relationship_patterns):
        return "relationship_focus"
    elif any(re.search(pattern, lower_input) for pattern in email_patterns):
        return "quick_check"
    elif any(re.search(pattern, lower_input) for pattern in calendar_patterns):
        return "quick_check"
    elif any(re.search(pattern, lower_input) for pattern in specific_patterns):
        return "specific_question"
    
    return "general"

def handle_content_creation(user_input, project, use_voices, random_toggle):
    """Handle content creation with improved exit detection"""
    global _current_content_mode, _current_context
    
    if not _current_content_mode:
        return {"SyntaxPrime": "Content creation mode not properly initialized."}, True
    
    # Get mode configuration
    mode_config = CONTENT_MODES.get(_current_content_mode) or ANALYSIS_MODES.get(_current_content_mode, CONTENT_MODES["email"])
    
    try:
        from modules.brain import enhanced_retrieve
        retrieval_ctx = enhanced_retrieve(f"{_current_content_mode} {user_input}", k=6, project=project)
    except ImportError:
        retrieval_ctx = []
    
    content_prompt = f"""{mode_config['prompt']}

User request: {user_input}

Current mode: {_current_content_mode}
Context: {_current_context}

Create high-quality content that matches the requested tone and style."""
    
    return generate_response(
        content_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    ), True

def handle_analysis_mode(user_input, project, use_voices, random_toggle):
    """Handle marketing plans, board reports, and data analysis with better exit handling"""
    global _current_content_mode, _current_context
    
    if not _current_content_mode:
        return {"SyntaxPrime": "Analysis mode not properly initialized."}, True
    
    mode_config = ANALYSIS_MODES[_current_content_mode]
    
    try:
        from modules.brain import enhanced_retrieve
        retrieval_ctx = enhanced_retrieve(f"{_current_content_mode} {user_input}", k=8, project=project)
    except ImportError:
        retrieval_ctx = []
    
    analysis_prompt = f"""{mode_config['prompt']}

User request: {user_input}
Analysis type: {_current_content_mode}
Context: {_current_context}

Provide thorough, actionable analysis."""
    
    return generate_response(
        analysis_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    ), True

def handle_casual_greeting(user_input, project, use_voices, random_toggle):
    """Handle casual greetings with Syntax's personality"""
    try:
        from modules.brain import enhanced_retrieve
        retrieval_ctx = enhanced_retrieve("syntax personality greeting", k=2, project=project)
    except ImportError:
        retrieval_ctx = []
    
    casual_prompt = f"""User said: {user_input}

This is a casual greeting or conversation. Respond as Syntax Prime with your characteristic personality - direct, slightly sarcastic, efficient, but helpful. Keep it brief and conversational. Don't provide briefings unless specifically asked."""
    
    return generate_response(
        casual_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

def enhanced_retrieve_with_fallbacks(query_text, k=5, project=None):
    """Enhanced retrieval with multiple fallback strategies"""
    try:
        from modules.brain import enhanced_retrieve
        return enhanced_retrieve(query_text, k, project=project)
    except ImportError:
        try:
            from utils.rag_basic import retrieve
            if is_ready():
                return retrieve(query_text, k)
        except:
            pass
    except Exception as e:
        print(f"Enhanced retrieve failed: {e}")
    return []

def handle_specific_question(user_input, project, use_voices, random_toggle):
    """Handle specific questions with enhanced context"""
    print(f"Handling specific question: {user_input}")
    
    retrieval_ctx = enhanced_retrieve_with_fallbacks(user_input, k=8, project=project)
    
    enhanced_prompt = f"""User question: {user_input}

Context information available: {len(retrieval_ctx)} relevant documents found.

Please provide a helpful and accurate answer. Use context when relevant, but also use your general knowledge when appropriate."""
    
    return generate_response(
        enhanced_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

def handle_morning_briefing(project, use_voices, random_toggle):
    """Comprehensive morning briefing from all available sources"""
    briefing_sections = []
    
    # Try Gmail/Calendar briefing
    try:
        gmail_result = handle_good_morning_command(project, use_voices, random_toggle)
        gmail_content = gmail_result.get("SyntaxPrime", "")
        if gmail_content and "failed" not in gmail_content.lower():
            briefing_sections.append(f"=== EMAIL & CALENDAR ===\n{gmail_content}")
        else:
            briefing_sections.append("=== EMAIL & CALENDAR ===\nEmail/Calendar check encountered an issue")
    except Exception as e:
        briefing_sections.append(f"=== EMAIL & CALENDAR ===\nEmail service temporarily unavailable")
    
    # Try other integrations...
    try:
        from modules.clickup_integration import get_clickup_morning_briefing, is_clickup_configured
        if is_clickup_configured():
            clickup_briefing = get_clickup_morning_briefing()
            if clickup_briefing and "error" not in clickup_briefing.lower():
                briefing_sections.append(f"=== TASKS & TIME TRACKING ===\n{clickup_briefing}")
    except ImportError:
        pass
    
    full_briefing = "\n\n".join(briefing_sections)
    save_daily_log_enhanced("comprehensive_morning", full_briefing)
    
    synthesis_prompt = f"""Here's my complete morning briefing:

{full_briefing}

Please synthesize this into a concise executive summary focusing on top priorities and time-sensitive items."""
    
    retrieval_ctx = enhanced_retrieve_with_fallbacks(synthesis_prompt, k=5, project=project)
    
    return generate_response(
        synthesis_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

# UPDATED: Helper function to check mode health with timing info
def get_current_mode_status():
    """Get current content/analysis mode for debugging with timing info"""
    global _current_content_mode, _current_context, _mode_start_time
    
    status = {
        "mode": _current_content_mode,
        "context": _current_context,
        "start_time": _mode_start_time.isoformat() if _mode_start_time else None
    }
    
    if _mode_start_time:
        time_in_mode = (datetime.datetime.now() - _mode_start_time).total_seconds()
        status["time_in_mode_seconds"] = time_in_mode
        status["time_in_mode_minutes"] = time_in_mode / 60
        status["will_timeout_at"] = (_mode_start_time + datetime.timedelta(minutes=10)).isoformat()
    
    return status

# ADDED: Manual mode reset function for debugging
def force_reset_content_mode():
    """Force reset content mode - useful for debugging"""
    global _current_content_mode, _current_context, _mode_start_time
    
    old_mode = _current_content_mode
    _current_content_mode = None
    _current_context.clear()
    _mode_start_time = None
    
    print(f"Force reset: cleared mode '{old_mode}'")
    return {"reset": True, "previous_mode": old_mode}

def process_smart_command(user_input, project, use_voices, random_toggle):
    """ULTRA-PRECISE main smart command processor with bulletproof intent detection"""
    
    intent = detect_intent(user_input)
    print(f"Smart command intent: {intent} for input: '{user_input}'")
    print(f"Current mode: {_current_content_mode}")
    
    if intent == "content_creation":
        return handle_content_creation(user_input, project, use_voices, random_toggle)
    
    elif intent == "analysis_mode":
        return handle_analysis_mode(user_input, project, use_voices, random_toggle)
    
    elif intent == "casual":
        response_data = handle_casual_greeting(user_input, project, use_voices, random_toggle)
        return response_data, True
    
    elif intent == "morning_briefing":
        response_data = handle_morning_briefing(project, use_voices, random_toggle)
        return response_data, True
    
    elif intent == "specific_question":
        response_data = handle_specific_question(user_input, project, use_voices, random_toggle)
        return response_data, True
    
    # If no smart command detected, return unhandled
    return {}, False
