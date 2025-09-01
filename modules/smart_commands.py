# modules/smart_commands.py - Enhanced Smart Commands with Better Context Management
# Complete replacement file with improved brain integration and proper casual handling

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

def detect_intent(user_input):
    """Detect user intent from natural language with enhanced patterns - FIXED VERSION"""
    lower_input = user_input.lower().strip()
    
    # CASUAL/GREETING patterns - CHECK THESE FIRST!
    casual_patterns = [
        "hello", "hi", "hey", "good afternoon", "good evening",
        "how are you", "what's up", "thanks", "thank you", "ok", "okay",
        "cool", "great", "nice", "got it", "understood", "perfect",
        "hello syntax", "hi syntax", "hey syntax", "syntax",
        "how are you doing", "how's it going", "what's happening",
        "sup", "yo", "wassup", "how you doing"
    ]
    
    # Check casual patterns FIRST - if it's just a greeting, don't over-analyze
    if any(pattern in lower_input for pattern in casual_patterns):
        # BUT make sure it's not actually asking for briefing info
        briefing_keywords = ["briefing", "brief me", "what do i need to know", "catch me up", "update me", "start my day", "daily"]
        if not any(keyword in lower_input for keyword in briefing_keywords):
            print(f"Detected casual greeting: '{lower_input}'")
            return "casual"
    
    # Morning briefing intent - VERY specific patterns only (removed "good morning" to avoid conflicts)
    morning_patterns = [
        "daily briefing", "brief me", "catch me up", "morning update",
        "daily summary", "what's today", "what do i need to know",
        "morning sync", "daily intel", "what's happening today",
        "start my day", "what's on tap today", "morning brief"
    ]
    
    # Task/productivity intent
    productivity_patterns = [
        "what should i work on", "priorities", "what's due", "deadlines",
        "my tasks", "task summary", "work focus", "productivity",
        "what's urgent", "time tracking", "hours logged", "focus time",
        "work plan", "today's work"
    ]
    
    # People/relationship/CRM intent
    relationship_patterns = [
        "who should i follow up with", "pipeline", "deals", "contacts",
        "relationships", "crm", "sales", "follow ups", "client work",
        "networking", "outreach", "relationship management"
    ]
    
    # Status check intent
    status_patterns = [
        "status check", "overview", "dashboard", "systems status",
        "what's connected", "integrations", "health check", "system health"
    ]
    
    # Quick email/calendar check
    email_patterns = [
        "overnight", "emails", "inbox", "mail check", "messages",
        "calendar", "meetings", "schedule", "next meeting",
        "what's in my inbox", "any emails"
    ]
    
    # Specific questions that need enhanced context
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
        # Import here to avoid circular imports
        from modules.brain import enhanced_retrieve
        return enhanced_retrieve(query_text, k, project=project)
    except ImportError:
        # Fallback if brain module isn't available
        try:
            from utils.rag_basic import retrieve
            if is_ready():
                return retrieve(query_text, k)
        except:
            pass
    except Exception as e:
        print(f"Enhanced retrieve failed: {e}")
    
    return []

def handle_casual_greeting(user_input, project, use_voices, random_toggle):
    """Handle casual greetings with Syntax's personality"""
    
    # Get minimal context for personality (not extensive briefings)
    retrieval_ctx = enhanced_retrieve_with_fallbacks("syntax personality greeting", k=2, project=project)
    
    # Create a personality-focused prompt
    casual_prompt = f"""User said: {user_input}

This is a casual greeting. Respond as Syntax Prime with your characteristic personality - direct, slightly sarcastic, efficient, but helpful. Keep it brief and conversational. Don't provide briefings or extensive context unless specifically asked.

Be yourself - the AI assistant with attitude who gets things done."""
    
    return generate_response(
        casual_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

def handle_specific_question(user_input, project, use_voices, random_toggle):
    """Handle specific questions with enhanced context and better prompting"""
    print(f"Handling specific question: {user_input}")
    
    # Get enhanced context
    retrieval_ctx = enhanced_retrieve_with_fallbacks(user_input, k=8, project=project)
    
    # Create enhanced prompt that encourages using general knowledge
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
        # Gmail/Calendar (core system - always try)
        gmail_result = handle_good_morning_command(project, use_voices, random_toggle)
        gmail_content = gmail_result.get("SyntaxPrime", "")
        if gmail_content and "failed" not in gmail_content.lower():
            briefing_sections.append(f"=== EMAIL & CALENDAR ===\n{gmail_content}")
        else:
            briefing_sections.append("=== EMAIL & CALENDAR ===\nEmail/Calendar check encountered an issue")
    except Exception as e:
        briefing_sections.append(f"=== EMAIL & CALENDAR ===\nEmail service temporarily unavailable")
    
    # ClickUp integration (if configured)
    if is_clickup_configured():
        try:
            clickup_briefing = get_clickup_morning_briefing()
            if clickup_briefing and "error" not in clickup_briefing.lower():
                briefing_sections.append(f"=== TASKS & TIME TRACKING ===\n{clickup_briefing}")
            else:
                briefing_sections.append("=== TASKS & TIME TRACKING ===\nClickUp data temporarily unavailable")
        except Exception as e:
            briefing_sections.append("=== TASKS & TIME TRACKING ===\nClickUp integration temporarily unavailable")
    
    # Cloze CRM (if configured)
    if is_cloze_configured():
        try:
            cloze_briefing = get_cloze_morning_briefing()
            if cloze_briefing and "error" not in cloze_briefing.lower():
                briefing_sections.append(f"=== CRM & PIPELINE ===\n{cloze_briefing}")
            else:
                briefing_sections.append("=== CRM & PIPELINE ===\nCloze data temporarily unavailable")
        except Exception as e:
            briefing_sections.append("=== CRM & PIPELINE ===\nCloze integration waiting for API updates")
    
    # Combine all sections
    full_briefing = "\n\n".join(briefing_sections)
    
    # Save the raw briefing to daily logs
    save_daily_log_enhanced("comprehensive_morning", full_briefing)
    
    # Use AI to synthesize everything into actionable insights
    synthesis_prompt = f"""Here's my complete morning briefing from all connected systems:

{full_briefing}

Please synthesize this into a concise executive summary focusing on:

1. **Top 3 Priorities** - What should I focus on first today?
2. **Time-Sensitive Items** - Anything with deadlines or urgency?
3. **Key Relationships** - Important people to follow up with or meetings to prep for?
4. **Potential Issues** - Any conflicts, overdue items, or concerns to address?

Keep it actionable and prioritized. Format with clear headers and bullet points."""
    
    # Get retrieval context for better responses
    retrieval_ctx = enhanced_retrieve_with_fallbacks(synthesis_prompt, k=5, project=project)
    
    return generate_response(
        synthesis_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

def handle_productivity_focus(project, use_voices, random_toggle):
    """Focus on tasks, deadlines, and productivity across systems"""
    productivity_sections = []
    
    # ClickUp tasks and time tracking
    if is_clickup_configured():
        try:
            tasks_summary = get_clickup_tasks_summary()
            time_today = get_clickup_time_today()
            productivity_sections.append(f"=== CLICKUP TASKS & TIME ===\n{tasks_summary}\n\n{time_today}")
        except Exception as e:
            productivity_sections.append("=== CLICKUP TASKS & TIME ===\nClickUp data temporarily unavailable")
    
    # Calendar for meeting prep time
    try:
        calendar_result = handle_calendar_today_command(project, use_voices, random_toggle)
        calendar_content = calendar_result.get("SyntaxPrime", "")
        if calendar_content:
            productivity_sections.append(f"=== TODAY'S MEETINGS ===\n{calendar_content}")
    except Exception:
        productivity_sections.append("=== TODAY'S MEETINGS ===\nCalendar temporarily unavailable")
    
    # Combine productivity data
    productivity_data = "\n\n".join(productivity_sections)
    
    productivity_prompt = f"""Here's my current productivity and task status:

{productivity_data}

Please help me prioritize my work today by identifying:

1. **Critical Tasks** - What absolutely must be done today?
2. **Time Blocks** - How should I structure my day around meetings?
3. **Quick Wins** - Any small tasks I can knock out between meetings?
4. **Focus Time** - When do I have uninterrupted work time?

Give me a practical work plan that maximizes productivity."""
    
    retrieval_ctx = enhanced_retrieve_with_fallbacks(productivity_prompt, k=5, project=project)
    
    return generate_response(
        productivity_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

def handle_relationship_focus(project, use_voices, random_toggle):
    """Focus on relationships, follow-ups, and CRM data"""
    relationship_sections = []
    
    # Cloze CRM data
    if is_cloze_configured():
        try:
            pipeline_summary = get_cloze_pipeline_summary()
            if pipeline_summary:
                relationship_sections.append(f"=== CRM PIPELINE ===\n{pipeline_summary}")
        except Exception:
            relationship_sections.append("=== CRM PIPELINE ===\nCloze integration waiting for API updates")
    
    # Recent emails for relationship context
    try:
        overnight_result = handle_overnight_command(project, use_voices, random_toggle)
        email_content = overnight_result.get("SyntaxPrime", "")
        if email_content:
            relationship_sections.append(f"=== RECENT COMMUNICATIONS ===\n{email_content}")
    except Exception:
        relationship_sections.append("=== RECENT COMMUNICATIONS ===\nEmail data temporarily unavailable")
    
    relationship_data = "\n\n".join(relationship_sections) if relationship_sections else "Limited relationship data available"
    
    relationship_prompt = f"""Here's my current relationship and communication status:

{relationship_data}

Please help me with relationship management by identifying:

1. **Priority Follow-ups** - Who needs my attention today?
2. **Deal Progression** - Any deals or projects that need advancement?
3. **Relationship Building** - Opportunities to strengthen key relationships?
4. **Communication Strategy** - Best approach for important conversations?

Focus on actionable relationship and business development steps."""
    
    retrieval_ctx = enhanced_retrieve_with_fallbacks(relationship_prompt, k=5, project=project)
    
    return generate_response(
        relationship_prompt, use_voices, random_toggle,
        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
    )

def handle_quick_check(project, use_voices, random_toggle):
    """Quick status check - emails and immediate calendar"""
    try:
        # Get overnight emails
        email_result = handle_overnight_command(project, use_voices, random_toggle)
        
        # Get next meeting
        next_meeting_result = handle_next_meeting_command(project, use_voices, random_toggle)
        
        # Combine for quick summary
        quick_prompt = f"""Quick status check requested. Here's what's immediate:

EMAILS: {email_result.get('SyntaxPrime', 'Email check unavailable')}

NEXT MEETING: {next_meeting_result.get('SyntaxPrime', 'No immediate meetings')}

Give me a 2-sentence summary of what needs my immediate attention."""
        
        retrieval_ctx = enhanced_retrieve_with_fallbacks(quick_prompt, k=3, project=project)
        
        return generate_response(
            quick_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        
    except Exception as e:
        return {"SyntaxPrime": f"Quick check encountered an issue: {str(e)}"}

def handle_status_overview(project, use_voices, random_toggle):
    """System status and integration overview with health check"""
    status_sections = []
    
    # Check which integrations are working
    gmail_status = "Connected"
    clickup_status = "Connected" if is_clickup_configured() else "Not configured"
    cloze_status = "Connected (waiting for API endpoints)" if is_cloze_configured() else "Not configured"
    
    # Check brain health
    try:
        from modules.database import get_brain_health_status
        brain_health = get_brain_health_status()
        brain_status = f"Status: {brain_health.get('status', 'unknown').title()}"
        if brain_health.get('total_documents'):
            brain_status += f" | Documents: {brain_health['total_documents']:,}"
    except Exception:
        brain_status = "Status check failed"
    
    status_sections.append(f"""=== SYSTEM STATUS ===
📧 Gmail/Calendar: {gmail_status}
📋 ClickUp: {clickup_status}
🏢 Cloze CRM: {cloze_status}
🧠 Brain/RAG System: {brain_status}

=== QUICK SYSTEM TEST ===""")
    
    # Test each system briefly
    if is_clickup_configured():
        try:
            time_summary = get_clickup_time_today()
            status_sections.append(f"ClickUp: Working - {time_summary.split('**')[1] if '**' in time_summary else 'Active'}")
        except Exception:
            status_sections.append("ClickUp: Connection issue")
    
    # Test brain search
    try:
        test_results = enhanced_retrieve_with_fallbacks("system test", k=1, project=project)
        status_sections.append(f"Brain Search: {len(test_results)} results for test query")
    except Exception:
        status_sections.append("Brain Search: Not responding")
    
    status_overview = "\n".join(status_sections)
    
    return {"SyntaxPrime": status_overview}

def process_smart_command(user_input, project, use_voices, random_toggle):
    """Main smart command processor with enhanced intent detection and casual handling"""
    
    intent = detect_intent(user_input)
    print(f"Detected intent: {intent} for input: '{user_input}'")
    
    if intent == "casual":
        response_data = handle_casual_greeting(user_input, project, use_voices, random_toggle)
        return response_data, True
    
    elif intent == "morning_briefing":
        response_data = handle_morning_briefing(project, use_voices, random_toggle)
        return response_data, True
    
    elif intent == "productivity_focus":
        response_data = handle_productivity_focus(project, use_voices, random_toggle)
        return response_data, True
    
    elif intent == "relationship_focus":
        response_data = handle_relationship_focus(project, use_voices, random_toggle)
        return response_data, True
    
    elif intent == "quick_check":
        response_data = handle_quick_check(project, use_voices, random_toggle)
        return response_data, True
    
    elif intent == "status_overview":
        response_data = handle_status_overview(project, use_voices, random_toggle)
        return response_data, True
    
    elif intent == "specific_question":
        response_data = handle_specific_question(user_input, project, use_voices, random_toggle)
        return response_data, True
    
    # If no smart command detected, return unhandled
    return {}, False
