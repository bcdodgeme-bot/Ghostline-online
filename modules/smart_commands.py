# modules/smart_commands.py
# Intelligent command routing that aggregates multiple data sources

import os
from utils.ghostline_engine import generate_response
from utils.rag_basic import is_ready
from modules.brain import enhanced_retrieve
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
    """Detect user intent from natural language"""
    lower_input = user_input.lower().strip()
    
    # Morning briefing intent - comprehensive patterns
    morning_patterns = [
        "good morning", "morning", "gm", "start my day", "daily briefing",
        "what's on tap", "what do i need to know", "morning update", 
        "daily summary", "what's today", "brief me", "catch me up"
    ]
    
    # Task/productivity intent
    productivity_patterns = [
        "what should i work on", "priorities", "what's due", "deadlines",
        "my tasks", "task summary", "work focus", "productivity",
        "what's urgent", "time tracking", "hours logged"
    ]
    
    # People/relationship/CRM intent
    relationship_patterns = [
        "who should i follow up with", "pipeline", "deals", "contacts",
        "relationships", "crm", "sales", "follow ups", "client work"
    ]
    
    # Status check intent
    status_patterns = [
        "how are things", "status check", "overview", "dashboard",
        "systems status", "what's connected", "integrations"
    ]
    
    # Quick email/calendar check
    email_patterns = [
        "overnight", "emails", "inbox", "mail check", "messages",
        "calendar", "meetings", "schedule", "next meeting"
    ]
    
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
    
    return "general"

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
                briefing_sections.append("=== CRM & PIPELINE ===\nCloze data temporarily unavailable (waiting for API endpoint updates)")
        except Exception as e:
            briefing_sections.append("=== CRM & PIPELINE ===\nCloze integration waiting for API endpoint updates from support")
    
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
    retrieval_ctx = enhanced_retrieve(synthesis_prompt, k=5) if is_ready() else []
    
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
    
    retrieval_ctx = enhanced_retrieve(productivity_prompt, k=5) if is_ready() else []
    
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
    
    retrieval_ctx = enhanced_retrieve(relationship_prompt, k=5) if is_ready() else []
    
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
        
        retrieval_ctx = enhanced_retrieve(quick_prompt, k=3) if is_ready() else []
        
        return generate_response(
            quick_prompt, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        
    except Exception as e:
        return {"SyntaxPrime": f"Quick check encountered an issue: {str(e)}"}

def handle_status_overview(project, use_voices, random_toggle):
    """System status and integration overview"""
    status_sections = []
    
    # Check which integrations are working
    gmail_status = "Connected" 
    clickup_status = "Connected" if is_clickup_configured() else "Not configured"
    cloze_status = "Connected (waiting for API endpoints)" if is_cloze_configured() else "Not configured"
    
    status_sections.append(f"""=== SYSTEM STATUS ===
📧 Gmail/Calendar: {gmail_status}
📋 ClickUp: {clickup_status}
🏢 Cloze CRM: {cloze_status}

=== QUICK SYSTEM TEST ===""")
    
    # Test each system briefly
    if is_clickup_configured():
        try:
            time_summary = get_clickup_time_today()
            status_sections.append(f"ClickUp: Working - {time_summary.split('**')[1] if '**' in time_summary else 'Active'}")
        except Exception:
            status_sections.append("ClickUp: Connection issue")
    
    status_overview = "\n".join(status_sections)
    
    return {"SyntaxPrime": status_overview}

def process_smart_command(user_input, project, use_voices, random_toggle):
    """Main smart command processor - returns (response_data, handled)"""
    
    intent = detect_intent(user_input)
    
    if intent == "morning_briefing":
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
    
    # If no smart command detected, return unhandled
    return {}, False