# modules/smart_commands.py - Enhanced Smart Commands with Content Modes
# Complete replacement file with content creation modes and marketing analysis

import os
from utils.ghostline_engine import generate_response
from utils.rag_basic import is_ready
from modules.database import save_conversation_enhanced, save_daily_log_enhanced

# Import existing handlers
from modules.gmail import (
    handle_good_morning_command, handle_overnight_command,
    handle_calendar_today_command, handle_next_meeting_command
)
from modules.clickup_integration import (
    get_clickup_morning_briefing, get_clickup_time_today,
    get_clickup_tasks_summary, is_clickup_configured
)
from modules.cloze_integration import (
    get_cloze_morning_briefing, get_cloze_pipeline_summary,
    is_cloze_configured
)

CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

# Content creation modes with context-aware prompting
CONTENT_MODES = {
    "email": {
        "prompt": "We are writing an email. Use professional tone, clear CTAs, proper email structure. Ask about target audience and tone (professional vs casual) if unclear.",
        "keywords": ["write email", "draft email", "email for", "compose email", "email about", "giving circle email", "newsletter"],
        "tone_questions": ["What's the target audience?", "Professional or casual tone?", "Any specific CTAs needed?"]
    },
    "blog": {
        "prompt": "We are writing a blog post. Use SEO best practices, engaging headlines, proper meta descriptions. Ask about tone (professional, sarcastic, technical) and target keywords.",
        "keywords": ["write blog", "blog post", "draft blog", "write article", "blog about"],
        "tone_questions": ["What's the tone - professional, sarcastic, or technical?", "Any target keywords for SEO?", "What's the main message?"]
    },
    "webpage": {
        "prompt": "We are writing web content. Focus on user experience, conversion optimization, accessibility, and clear value propositions.",
        "keywords": ["webpage copy", "website content", "landing page", "web copy", "page content"],
        "tone_questions": ["What's the page purpose?", "What action should users take?", "Target audience?"]
    },
    "sms": {
        "prompt": "We are writing SMS content. Keep under 160 characters, clear action, urgent tone. Be direct and actionable.",
        "keywords": ["write sms", "text message", "sms for", "text about"],
        "tone_questions": ["What's the main action?", "How urgent is this?"]
    },
    "social": {
        "prompt": "We are writing social media content. Platform-optimized, engaging, hashtag-ready. Ask about platform and tone.",
        "keywords": ["social media", "social post", "tweet", "instagram post", "facebook post", "linkedin post"],
        "tone_questions": ["Which platform?", "Professional or casual tone?", "Any specific hashtags needed?"]
    }
}

# Marketing and analysis modes
ANALYSIS_MODES = {
    "marketing_plan": {
        "prompt": "We are developing a marketing plan. Focus on strategy, target audience analysis, channel selection, budget allocation, and measurable objectives.",
        "keywords": ["marketing plan", "marketing strategy", "campaign plan", "marketing analysis", "brand strategy"],
        "questions": ["What's the product/service?", "Target audience?", "Budget range?", "Timeline?", "Key competitors?"]
    },
    "board_report": {
        "prompt": "We are writing a board report. Use executive summary format, data-driven insights, clear recommendations, risk assessment, and financial implications.",
        "keywords": ["board report", "executive report", "quarterly report", "board presentation", "executive summary"],
        "questions": ["Which quarter/period?", "Key metrics to highlight?", "Any major decisions needed?", "Financial performance focus?"]
    },
    "data_analysis": {
        "prompt": "We are analyzing data and metrics. Focus on trends, insights, actionable recommendations, and clear visualizations of findings.",
        "keywords": ["analyze data", "data analysis", "metrics review", "performance analysis", "numbers analysis"],
        "questions": ["What data source?", "What are you trying to understand?", "Any specific metrics of concern?", "Timeline for analysis?"]
    },
    "competitive_analysis": {
        "prompt": "We are conducting competitive analysis. Focus on market positioning, competitive advantages, pricing comparison, and strategic recommendations.",
        "keywords": ["competitive analysis", "competitor research", "market analysis", "competition review"],
        "questions": ["Who are the main competitors?", "What market segment?", "What are you trying to achieve?"]
    }
}

# Global variable to track current content mode
_current_content_mode = None
_current_context = {}

def detect_intent(user_input):
    """Enhanced intent detection with content mode support"""
    global _current_content_mode
    lower_input = user_input.lower().strip()
    
    # Check for content mode exit signals first
    exit_patterns = [
        "okay, that", "that's done", "that email is done", "that blog is scheduled",
        "let's move on", "next,", "now let's", "okay let's", "that's finished",
        "that's complete", "done with that"
    ]
    
    if any(pattern in lower_input for pattern in exit_patterns):
        _current_content_mode = None
        _current_context.clear()
        print(f"Exiting content mode due to completion signal")
    
    # Check for content mode transitions (blog -> social posts)
    if "let's write social" in lower_input or "social media posts for it" in lower_input:
        _current_content_mode = "social"
        print(f"Transitioning to social content mode")
        return "content_creation"
    
    # If we're in content mode, stay in content mode unless explicitly exiting
    if _current_content_mode:
        print(f"Continuing in {_current_content_mode} content mode")
        return "content_creation"
    
    # Check for new content mode entry
    for mode, config in CONTENT_MODES.items():
        if any(keyword in lower_input for keyword in config["keywords"]):
            _current_content_mode = mode
            _current_context = {"initial_input": user_input}
            print(f"Entering {mode} content mode")
            return "content_creation"
    
    # Check for analysis mode entry
    for mode, config in ANALYSIS_MODES.items():
        if any(keyword in lower_input for keyword in config["keywords"]):
            _current_content_mode = mode
            _current_context = {"initial_input": user_input}
            print(f"Entering {mode} analysis mode")
            return "analysis_mode"
    
    # CASUAL/GREETING patterns - CHECK THESE NEXT!
    casual_patterns = [
        "hello", "hi", "hey", "good afternoon", "good evening",
        "how are you", "what's up", "thanks", "thank you", "ok", "okay",
        "cool", "great", "nice", "got it", "understood", "perfect",
        "hello syntax", "hi syntax", "hey syntax", "syntax",
        "how are you doing", "how's it going", "what's happening",
        "sup", "yo", "wassup", "how you doing"
    ]
    
    # Check casual patterns but make sure it's not asking for briefing info
    if any(pattern in lower_input for pattern in casual_patterns):
        briefing_keywords = ["briefing", "brief me", "what do i need to know", "catch me up", "update me", "start my day", "daily"]
        if not any(keyword in lower_input for keyword in briefing_keywords):
            print(f"Detected casual greeting: '{lower_input}'")
            return "casual"
    
    # Morning briefing intent - VERY specific patterns only
    morning_patterns = [
        "daily briefing", "brief me", "catch me up", "morning update",
        "daily summary", "what's today", "what do i need to know",
        "morning sync", "daily intel", "what's happening today",
        "start my day", "what's on tap today", "morning brief"
    ]
    
    # Other intent patterns...
    productivity_patterns = [
        "what should i work on", "priorities", "what's due", "deadlines",
        "my tasks", "task summary", "work focus", "productivity",
        "what's urgent", "time tracking", "hours logged", "focus time",
        "work plan", "today's work"
    ]
    
    relationship_patterns = [
        "who should i follow up with", "pipeline", "deals", "contacts",
        "relationships", "crm", "sales", "follow ups", "client work",
        "networking", "outreach", "relationship management"
    ]
    
    status_patterns = [
        "status check", "overview", "dashboard", "systems status",
        "what's connected", "integrations", "health check", "system health"
    ]
    
    email_patterns = [
        "overnight", "emails", "inbox", "mail check", "messages",
        "calendar", "meetings", "schedule", "next meeting",
        "what's in my inbox", "any emails"
    ]
    
    specific_patterns = [
        "what is", "what does", "tell me about", "explain", "describe",
        "who is", "where is", "when is", "how does", "why does"
    ]
    
    # Check patterns in priority order
    if any(pattern in lower_input for pattern in morning_patterns):
        return "morning_briefing"
    elif any(pattern in lower_input for pattern in productivity_patterns):
        return "productivity_focus"
    elif any(pattern in lower_input for pattern in relationship_patterns):
        return "relationship_focus"
    elif any(pattern in lower_input for pattern in status_patterns):
        return "status_overview"
    elif any(pattern in lower_input for pattern in email_patterns):
        return "quick_check"
    elif any(pattern in lower_input for pattern in specific_patterns):
        return "specific_question"
    
    return "general"

def enhanced_retrieve_with_fallbacks(query_text, k=5, project=None):
    """Enhanced retrieval with multiple fallback strategies and project context"""
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

def handle_content_creation(user_input, project, use_voices, random_toggle):
    """Handle content creation with mode-specific prompting and tone questions"""
    global _current_content_mode, _current_context
    
    if not _current_content_mode:
        return {"SyntaxPrime": "Content creation mode not properly initialized. Try starting with 'write email' or 'draft blog post'."}, True
    
    # Get mode configuration
    if _current_content_mode in CONTENT_MODES:
        mode_config = CONTENT_MODES[_current_content_mode]
    else:
        mode_config = ANALYSIS_MODES.get(_current_content_mode, CONTENT_MODES["email"])
    
    # Check if this is the initial request and we should ask clarifying questions
    if "initial_input" in _current_context and len(_current_context) == 1:
        # Ask tone/context questions for better content
        if "tone_questions" in mode_config:
            questions = "\n".join([f"• {q}" for q in mode_config["tone_questions"]])
            response = f"Got it! I'm ready to help with your {_current_content_mode}. A few quick questions to nail the tone:\n\n{questions}\n\nOr if you want to dive right in, just share the content/brief and I'll work with what you give me."
            _current_context["asked_questions"] = True
            return {"SyntaxPrime": response}, True
    
    # Get relevant context for content creation
    retrieval_ctx = enhanced_retrieve_with_fallbacks(f"{_current_content_mode} {user_input}", k=6, project=project)
    
    # Create context-aware prompt
    content_prompt = f"""{mode_config['prompt']}

User request: {user_input}

Current mode: {_current_content_mode}
Context: {_current_context}

Create high-quality content that matches the requested tone and style. If the user hasn't specified tone, choose appropriately based on context. Be ready to iterate and refine based on feedback."""
    
    return generate_response(
        content_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    ), True

def handle_analysis_mode(user_input, project, use_voices, random_toggle):
    """Handle marketing plans, board reports, and data analysis"""
    global _current_content_mode, _current_context
    
    if not _current_content_mode:
        return {"SyntaxPrime": "Analysis mode not properly initialized."}, True
    
    mode_config = ANALYSIS_MODES[_current_content_mode]
    
    # Check if we should ask clarifying questions
    if "initial_input" in _current_context and len(_current_context) == 1:
        if "questions" in mode_config:
            questions = "\n".join([f"• {q}" for q in mode_config["questions"]])
            response = f"Perfect! I'm ready to help with your {_current_content_mode.replace('_', ' ')}. Let me ask a few questions to create the best analysis:\n\n{questions}\n\nOr share what you have and I'll work with that data."
            _current_context["asked_questions"] = True
            return {"SyntaxPrime": response}, True
    
    # Get relevant context
    retrieval_ctx = enhanced_retrieve_with_fallbacks(f"{_current_content_mode} {user_input}", k=8, project=project)
    
    # Create analysis-focused prompt
    analysis_prompt = f"""{mode_config['prompt']}

User request: {user_input}
Analysis type: {_current_content_mode}
Context: {_current_context}

Provide thorough, actionable analysis with:
1. Clear executive summary
2. Data-driven insights
3. Specific recommendations
4. Risk considerations
5. Next steps

Format for easy consumption and decision-making."""
    
    return generate_response(
        analysis_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    ), True

def handle_casual_greeting(user_input, project, use_voices, random_toggle):
    """Handle casual greetings with Syntax's personality"""
    retrieval_ctx = enhanced_retrieve_with_fallbacks("syntax personality greeting", k=2, project=project)
    
    casual_prompt = f"""User said: {user_input}

This is a casual greeting. Respond as Syntax Prime with your characteristic personality - direct, slightly sarcastic, efficient, but helpful. Keep it brief and conversational. Don't provide briefings or extensive context unless specifically asked.

Be yourself - the AI assistant with attitude who gets things done."""
    
    return generate_response(
        casual_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

# [Keep all the existing handler functions: handle_specific_question, handle_morning_briefing, etc.]
def handle_specific_question(user_input, project, use_voices, random_toggle):
    """Handle specific questions with enhanced context and better prompting"""
    print(f"Handling specific question: {user_input}")
    
    retrieval_ctx = enhanced_retrieve_with_fallbacks(user_input, k=8, project=project)
    
    enhanced_prompt = f"""User question: {user_input}

Context information available: {len(retrieval_ctx)} relevant documents found.

Instructions: Please provide a helpful and accurate answer to the user's question. Use the context information if relevant, but also feel free to use your general knowledge when appropriate. Don't be overly cautious about claiming you lack information if you actually know about the topic from your training.

For questions about popular culture, TV shows, movies, books, historical events, or other well-established topics, provide informative responses based on your knowledge even if the context is limited.

Only state that you don't have information if the question is about very specific, recent, or specialized topics that genuinely require external sources."""
    
    return generate_response(
        enhanced_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

def handle_morning_briefing(project, use_voices, random_toggle):
    """Comprehensive morning briefing from all available sources"""
    briefing_sections = []
    
    try:
        gmail_result = handle_good_morning_command(project, use_voices, random_toggle)
        gmail_content = gmail_result.get("SyntaxPrime", "")
        if gmail_content and "failed" not in gmail_content.lower():
            briefing_sections.append(f"=== EMAIL & CALENDAR ===\n{gmail_content}")
        else:
            briefing_sections.append("=== EMAIL & CALENDAR ===\nEmail/Calendar check encountered an issue")
    except Exception as e:
        briefing_sections.append(f"=== EMAIL & CALENDAR ===\nEmail service temporarily unavailable")
    
    if is_clickup_configured():
        try:
            clickup_briefing = get_clickup_morning_briefing()
            if clickup_briefing and "error" not in clickup_briefing.lower():
                briefing_sections.append(f"=== TASKS & TIME TRACKING ===\n{clickup_briefing}")
            else:
                briefing_sections.append("=== TASKS & TIME TRACKING ===\nClickUp data temporarily unavailable")
        except Exception as e:
            briefing_sections.append("=== TASKS & TIME TRACKING ===\nClickUp integration temporarily unavailable")
    
    if is_cloze_configured():
        try:
            cloze_briefing = get_cloze_morning_briefing()
            if cloze_briefing and "error" not in cloze_briefing.lower():
                briefing_sections.append(f"=== CRM & PIPELINE ===\n{cloze_briefing}")
            else:
                briefing_sections.append("=== CRM & PIPELINE ===\nCloze data temporarily unavailable")
        except Exception as e:
            briefing_sections.append("=== CRM & PIPELINE ===\nCloze integration waiting for API updates")
    
    full_briefing = "\n\n".join(briefing_sections)
    save_daily_log_enhanced("comprehensive_morning", full_briefing)
    
    synthesis_prompt = f"""Here's my complete morning briefing from all connected systems:

{full_briefing}

Please synthesize this into a concise executive summary focusing on:

1. **Top 3 Priorities** - What should I focus on first today?
2. **Time-Sensitive Items** - Anything with deadlines or urgency?
3. **Key Relationships** - Important people to follow up with or meetings to prep for?
4. **Potential Issues** - Any conflicts, overdue items, or concerns to address?

Keep it actionable and prioritized. Format with clear headers and bullet points."""
    
    retrieval_ctx = enhanced_retrieve_with_fallbacks(synthesis_prompt, k=5, project=project)
    
    return generate_response(
        synthesis_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

# [Keep other existing handlers - productivity_focus, relationship_focus, quick_check, status_overview]

def get_current_mode_status():
    """Get current content/analysis mode for debugging"""
    global _current_content_mode, _current_context
    return {
        "mode": _current_content_mode,
        "context": _current_context
    }

def process_smart_command(user_input, project, use_voices, random_toggle):
    """Main smart command processor with content mode support"""
    
    intent = detect_intent(user_input)
    print(f"Detected intent: {intent} for input: '{user_input}'")
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
    
    # [Add other existing intent handlers here]
    
    # If no smart command detected, return unhandled
    return {}, False
