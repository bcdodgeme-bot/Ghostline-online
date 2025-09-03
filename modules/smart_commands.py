# modules/smart_commands.py - FIXED VERSION with precise keyword matching
# This fixes the overly aggressive keyword triggers causing false command activation

import os
import re
from utils.ghostline_engine import generate_response
from utils.rag_basic import is_ready
from modules.database import save_conversation_enhanced, save_daily_log_enhanced

# Import existing handlers
from modules.gmail import (
    handle_good_morning_command, handle_overnight_command,
    handle_calendar_today_command, handle_next_meeting_command
)

CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

# FIXED: More precise content mode patterns - using word boundaries and specific phrases
CONTENT_MODES = {
    "email": {
        "prompt": "We are writing an email. Use professional tone, clear CTAs, proper email structure.",
        "patterns": [
            r'\bwrite\s+email\b',
            r'\bdraft\s+email\b',
            r'\bemail\s+for\b',
            r'\bcompose\s+email\b',
            r'\bemail\s+about\b',
            r'\bgiving\s+circle\s+email\b',
            r'\bnewsletter\b'
        ],
        "tone_questions": ["What's the target audience?", "Professional or casual tone?", "Any specific CTAs needed?"]
    },
    "blog": {
        "prompt": "We are writing a blog post. Use SEO best practices, engaging headlines, proper meta descriptions.",
        "patterns": [
            r'\bwrite\s+blog\b',
            r'\bblog\s+post\b',
            r'\bdraft\s+blog\b',
            r'\bwrite\s+article\b',
            r'\bblog\s+about\b'
        ],
        "tone_questions": ["What's the tone?", "Any target keywords for SEO?", "What's the main message?"]
    },
    "sms": {
        "prompt": "We are writing SMS content. Keep under 160 characters, clear action, urgent tone.",
        "patterns": [
            r'\bwrite\s+sms\b',
            r'\btext\s+message\b',
            r'\bsms\s+for\b',
            r'\btext\s+about\b'
        ],
        "tone_questions": ["What's the main action?", "How urgent is this?"]
    },
    "social": {
        "prompt": "We are writing social media content. Platform-optimized, engaging, hashtag-ready.",
        "patterns": [
            r'\bsocial\s+media\b',
            r'\bsocial\s+post\b',
            r'\btweet\b',
            r'\binstagram\s+post\b',
            r'\bfacebook\s+post\b',
            r'\blinkedin\s+post\b'
        ],
        "tone_questions": ["Which platform?", "Professional or casual tone?", "Any specific hashtags needed?"]
    }
}

# FIXED: More specific analysis mode patterns
ANALYSIS_MODES = {
    "marketing_plan": {
        "prompt": "We are developing a marketing plan. Focus on strategy, target audience analysis, channel selection.",
        "patterns": [
            r'\bmarketing\s+plan\b',
            r'\bmarketing\s+strategy\b',
            r'\bcampaign\s+plan\b',
            r'\bmarketing\s+analysis\b',
            r'\bbrand\s+strategy\b',
            r'\bcreate\s+marketing\b',
            r'\bdevelop\s+marketing\b'
        ],
        "questions": ["What's the product/service?", "Target audience?", "Budget range?"]
    },
    "board_report": {
        "prompt": "We are writing a board report. Use executive summary format, data-driven insights, clear recommendations.",
        "patterns": [
            r'\bboard\s+report\b',
            r'\bexecutive\s+report\b',
            r'\bquarterly\s+report\b',
            r'\bboard\s+presentation\b',
            r'\bexecutive\s+summary\b',
            r'\bwrite\s+board\b',
            r'\bcreate\s+board\s+report\b'
        ],
        "questions": ["Which quarter/period?", "Key metrics to highlight?", "Any major decisions needed?"]
    },
    "competitive_analysis": {
        "prompt": "We are conducting competitive analysis. Focus on market positioning, competitive advantages.",
        "patterns": [
            r'\bcompetitive\s+analysis\b',
            r'\bcompetitor\s+research\b',
            r'\bmarket\s+analysis\b',
            r'\bcompetition\s+review\b',
            r'\bcompetitor\s+study\b'
        ],
        "questions": ["Who are the main competitors?", "What market segment?"]
    }
}

# Global variables for mode tracking
_current_content_mode = None
_current_context = {}

def matches_pattern(text, patterns):
    """Check if text matches any of the regex patterns (case insensitive)"""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def detect_intent(user_input):
    """FIXED: Much more precise intent detection with better exit handling"""
    global _current_content_mode
    lower_input = user_input.lower().strip()
    
    # FIXED: Better exit pattern detection
    exit_patterns = [
        r'\bthat\'?s\s+done\b',
        r'\bthat\'?s\s+finished\b',
        r'\bthat\'?s\s+complete\b',
        r'\bokay,?\s+that\b',
        r'\blet\'?s\s+move\s+on\b',
        r'\bnext,?\s',
        r'\bnow\s+let\'?s\b',
        r'\bdone\s+with\s+that\b',
        r'\bfinished\s+with\b'
    ]
    
    if any(re.search(pattern, lower_input) for pattern in exit_patterns):
        print(f"Exiting content mode due to completion signal: '{user_input}'")
        _current_content_mode = None
        _current_context.clear()
        return "casual"  # Treat as casual conversation after exit
    
    # FIXED: More precise content mode transitions
    transition_patterns = [
        r'\blet\'?s\s+write\s+social\b',
        r'\bsocial\s+media\s+posts?\s+for\s+it\b',
        r'\bnow\s+create\s+social\b'
    ]
    
    if any(re.search(pattern, lower_input) for pattern in transition_patterns):
        _current_content_mode = "social"
        print(f"Transitioning to social content mode")
        return "content_creation"
    
    # If we're in content mode, check if this is still content-related
    if _current_content_mode:
        # FIXED: Allow natural conversation to break out of content mode
        casual_conversation_indicators = [
            r'\bhello\b', r'\bhi\b', r'\bhey\b', r'\bthanks?\b', r'\bthank\s+you\b',
            r'\bhow\s+are\s+you\b', r'\bwhat\'?s\s+up\b', r'\bgood\s+morning\b',
            r'\bhow\'?s\s+it\s+going\b', r'\bawesome\b', r'\bgreat\b',
            r'\bpraying\s+fajr\b',  # Specific fix for user's example
            r'\bcup\s+(one|two|three|\d+)\b',  # Specific fix for coffee references
            r'\bgood\s+afternoon\b', r'\bgood\s+evening\b'
        ]
        
        if any(re.search(pattern, lower_input) for pattern in casual_conversation_indicators):
            print(f"Breaking out of {_current_content_mode} mode for casual conversation")
            _current_content_mode = None
            _current_context.clear()
            return "casual"
        
        # Otherwise, continue in current mode
        print(f"Continuing in {_current_content_mode} content mode")
        return "content_creation" if _current_content_mode in CONTENT_MODES else "analysis_mode"
    
    # FIXED: Check for new content mode entry with precise patterns
    for mode, config in CONTENT_MODES.items():
        if matches_pattern(user_input, config["patterns"]):
            _current_content_mode = mode
            _current_context = {"initial_input": user_input}
            print(f"Entering {mode} content mode")
            return "content_creation"
    
    # FIXED: Check for analysis mode entry with precise patterns
    for mode, config in ANALYSIS_MODES.items():
        if matches_pattern(user_input, config["patterns"]):
            _current_content_mode = mode
            _current_context = {"initial_input": user_input}
            print(f"Entering {mode} analysis mode")
            return "analysis_mode"
    
    # FIXED: Much more specific casual/greeting patterns
    casual_patterns = [
        r'^\b(hello|hi|hey)\b',  # Must start with greeting
        r'\bgood\s+(afternoon|evening|night)\b',
        r'\bhow\s+are\s+you(\s+doing)?\b',
        r'^\bwhat\'?s\s+up\b',
        r'^\b(thanks?|thank\s+you)\b',
        r'^\b(ok|okay|cool|great|nice|got\s+it|understood|perfect)\b$',
        r'\bhello\s+syntax\b',
        r'\bhi\s+syntax\b',
        r'\bhey\s+syntax\b',
        r'\bpraying\s+fajr\b',  # User's specific example
        r'\bcup\s+(one|two|three|\d+)\b'  # Coffee references
    ]
    
    # Only match casual if it's clearly casual and not asking for briefing info
    briefing_keywords = [r'\bbriefing\b', r'\bbrief\s+me\b', r'\bcatch\s+me\s+up\b',
                        r'\bupdate\s+me\b', r'\bstart\s+my\s+day\b', r'\bdaily\b']
    
    is_casual = any(re.search(pattern, lower_input) for pattern in casual_patterns)
    is_briefing = any(re.search(pattern, lower_input) for pattern in briefing_keywords)
    
    if is_casual and not is_briefing:
        print(f"Detected casual greeting: '{lower_input}'")
        return "casual"
    
    # FIXED: More specific morning briefing patterns
    morning_patterns = [
        r'\bdaily\s+briefing\b',
        r'\bbrief\s+me\b',
        r'\bcatch\s+me\s+up\b',
        r'\bmorning\s+update\b',
        r'\bdaily\s+summary\b',
        r'\bwhat\'?s\s+today\b',
        r'\bwhat\s+do\s+i\s+need\s+to\s+know\b',
        r'\bmorning\s+sync\b',
        r'\bdaily\s+intel\b',
        r'\bstart\s+my\s+day\b'
    ]
    
    # Other intent patterns (keep existing logic but make more precise)
    productivity_patterns = [
        r'\bwhat\s+should\s+i\s+work\s+on\b',
        r'\bmy\s+priorities\b',
        r'\bwhat\'?s\s+due\b',
        r'\bdeadlines\b',
        r'\bmy\s+tasks\b',
        r'\btask\s+summary\b',
        r'\bwork\s+focus\b'
    ]
    
    relationship_patterns = [
        r'\bwho\s+should\s+i\s+follow\s+up\s+with\b',
        r'\bpipeline\b',
        r'\bdeals\b',
        r'\bcrm\b',
        r'\bfollow\s+ups?\b'
    ]
    
    email_patterns = [
        r'^\bovernight\b$',  # Must be exact word
        r'^\bemails?\b$',
        r'^\binbox\b$',
        r'^\bmail\b$',
        r'\bcheck\s+mail\b',
        r'\bcalendar\b',
        r'\bmeetings?\b',
        r'\bschedule\b'
    ]
    
    specific_patterns = [
        r'\bwhat\s+is\b',
        r'\bwhat\s+does\b',
        r'\btell\s+me\s+about\b',
        r'\bexplain\b',
        r'\bdescribe\b'
    ]
    
    # Check patterns in priority order with precise matching
    if any(re.search(pattern, lower_input) for pattern in morning_patterns):
        return "morning_briefing"
    elif any(re.search(pattern, lower_input) for pattern in productivity_patterns):
        return "productivity_focus"
    elif any(re.search(pattern, lower_input) for pattern in relationship_patterns):
        return "relationship_focus"
    elif any(re.search(pattern, lower_input) for pattern in email_patterns):
        return "quick_check"
    elif any(re.search(pattern, lower_input) for pattern in specific_patterns):
        return "specific_question"
    
    return "general"

# Keep all your existing handler functions but with better error handling

def handle_content_creation(user_input, project, use_voices, random_toggle):
    """Handle content creation with improved exit detection"""
    global _current_content_mode, _current_context
    
    if not _current_content_mode:
        return {"SyntaxPrime": "Content creation mode not properly initialized."}, True
    
    # Get mode configuration
    mode_config = CONTENT_MODES.get(_current_content_mode) or ANALYSIS_MODES.get(_current_content_mode, CONTENT_MODES["email"])
    
    # Rest of your existing content creation logic...
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
    
    # FIXED: Better exit detection for board report loops
    if "board_report" in _current_content_mode:
        board_completion_signals = [
            r'\bthat\'?s\s+the\s+report\b',
            r'\bthat\'?s\s+enough\b',
            r'\bboard\s+report\s+is\s+done\b',
            r'\bfinished\s+with\s+the\s+board\s+report\b'
        ]
        
        if any(re.search(pattern, user_input.lower()) for pattern in board_completion_signals):
            _current_content_mode = None
            _current_context.clear()
            return {"SyntaxPrime": "Board report completed! What else can I help you with?"}, True
    
    # Rest of your existing analysis mode logic...
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

This is a casual greeting. Respond as Syntax Prime with your characteristic personality - direct, slightly sarcastic, efficient, but helpful. Keep it brief and conversational. Don't provide briefings unless specifically asked."""
    
    return generate_response(
        casual_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

# Keep all your other existing handler functions unchanged...
# (handle_specific_question, handle_morning_briefing, etc.)

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

def get_current_mode_status():
    """Get current content/analysis mode for debugging"""
    global _current_content_mode, _current_context
    return {
        "mode": _current_content_mode,
        "context": _current_context
    }

def process_smart_command(user_input, project, use_voices, random_toggle):
    """FIXED: Main smart command processor with precise intent detection"""
    
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
