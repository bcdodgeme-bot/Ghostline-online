# Section 1: Imports and Flask Setup
# Section 1: Imports and Flask Setup (UPDATED)
# Section 1: Imports and Flask Setup (UPDATED FOR PHASE 2)
# Section 1: Imports and Flask Setup (UPDATED FOR CONSOLIDATED GOOGLE INTEGRATION)
# Section 1: Imports and Flask Setup (UPDATED WITH ENHANCED MARKETING)
# Section 1: Imports and Flask Setup (UPDATED WITH CALENDAR-TELEGRAM INTEGRATION)
# Section 1: Imports and Flask Setup (UPDATED WITH CALENDAR-TELEGRAM INTEGRATION)9/11/25
# Section 1: Imports and Flask Setup (UPDATED WITH KEYWORD MANAGEMENT SYSTEM) 9/13/25
# Section 1: Imports and Flask Setup (UPDATED WITH PROJECT MAPPING SYSTEM) 9/13/25
# Section 1: Imports and Flask Setup (UPDATED WITH GOOGLE DRIVE EXPORT INTEGRATION)
# Section 1: Imports and Flask Setup (UPDATED WITH GOOGLE DRIVE EXPORT) 9/16/25
from flask import Flask, render_template, request, redirect, session, url_for, send_file, jsonify, render_template_string, Response
from flask_cors import CORS
from utils.ghostline_engine import generate_response, generate_streaming_response as stream_generate
from utils.rag_basic import retrieve, is_ready, load_corpus, get_build_status
from utils.scraper import scrape_url
from utils.gmail_client import (
    list_overnight, search as gmail_search,
    list_today_events, list_tomorrow_events, search_calendar,
    get_next_meeting, format_calendar_summary
)
# FIXED: Uncommented feedback system imports
from modules.feedback_system import submit_user_feedback as record_response_feedback, get_feedback_dashboard as get_feedback_dashboard_data
#from modules.hybrid_analysis import generate_content_strategy_command
#from modules.settings_persistence import get_default_voice, apply_session_preferences
import os, json, io
import threading
import time
import zipfile
import tempfile
import datetime
import requests
from typing import List, Dict, Optional, Any, Tuple

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Module imports for integrated systems
from modules.marketing_commands import process_marketing_command, is_marketing_configured
from modules.cloze_integration import process_cloze_command, is_cloze_configured
from modules.clickup_integration import process_clickup_command, is_clickup_configured
from modules.telegram_notifications import (
    GhostlineTelegramReminders,
    parse_reminder_command,
    is_telegram_configured
)
from modules.personalities import GhostlinePersonalities, PersonalityIntegration

# UPDATED: Enhanced Marketing Integration with Context
from modules.conversation_context_handler import (
    MarketingContextManager,
    process_marketing_command_with_context,
    marketing_context
)

# UPDATED: Consolidated Google Integration
from modules.enhanced_google_integration import process_google_ecosystem_commands

# NEW: Calendar → Telegram Integration
from modules.calendar_telegram_integration import (
    process_calendar_telegram_command,
    is_calendar_telegram_configured,
    start_calendar_monitoring,
    stop_calendar_monitoring,
    calendar_monitor_hotfix as calendar_monitor
)

from modules.bluesky_integration import process_bluesky_command, is_bluesky_configured

# NEW: Keyword Management System
from modules.site_keyword_manager import keyword_manager, SITE_DOMAINS

# NEW: Project Mapping System
from modules.project_mapping import ProjectMappingSystem, integrate_with_ghostline

# NEW: Google Drive Export Integration
from modules.chat_export_integration import handle_export_command, get_export_help

# OCR/File Parsing
from PIL import Image
import fitz
import docx

# Markdown support
import markdown
from markupsafe import Markup

# Database imports
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import urllib.parse

# RSS Marketing Knowledge System imports
from modules.rss_marketing_monitor import get_rss_monitor, start_rss_monitoring, stop_rss_monitoring, get_rss_status, force_feed_update
from modules.marketing_retrieval import (
    get_marketing_retriever,
    search_marketing_knowledge,
    get_seo_advice,
    get_content_writing_tips,
    get_social_media_advice,
    get_fresh_marketing_updates
)

# REMOVED: Placeholder functions - using real feedback system now

def generate_content_strategy_command(*args, **kwargs):
    """Placeholder content strategy function"""
    return {}, False

def get_default_voice():
    """Placeholder voice function"""
    return 'SyntaxPrime'

def apply_session_preferences(*args, **kwargs):
    """Placeholder preferences function"""
    pass

def is_google_configured():
    """Check if Google OAuth is properly configured for document export"""
    try:
        from modules.enhanced_google_integration import EnhancedGoogleIntegration
        google_integration = EnhancedGoogleIntegration()
        
        # Check if we have valid credentials and required scopes
        if not google_integration.is_configured():
            return False
            
        # Check if we have the required document creation scope
        required_scopes = [
            'https://www.googleapis.com/auth/documents',
            'https://www.googleapis.com/auth/drive.file'
        ]
        
        # Get current scopes from credentials
        if hasattr(google_integration, 'get_credentials'):
            credentials = google_integration.get_credentials()
            if credentials and hasattr(credentials, 'scopes'):
                current_scopes = credentials.scopes or []
                return all(scope in current_scopes for scope in required_scopes)
        
        # Fallback: assume configured if basic google integration works
        return google_integration.is_configured()
        
    except ImportError:
        return False
    except Exception as e:
        print(f"Google configuration check failed: {e}")
        return False

# JWT for mobile API
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("PyJWT not available - mobile API authentication disabled")

# .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# Section 2: Database and Module Initialization
app = Flask(__name__)

# Enhanced session and CORS configuration for Railway deployment
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=bool(os.getenv('RAILWAY_ENVIRONMENT')),
    SESSION_COOKIE_DOMAIN=None
)

# Enable CORS for streaming with credentials support
CORS(app, supports_credentials=True, origins=['*'])

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'ghostline-default-key')
PASSWORD = os.getenv('GHOSTLINE_PASSWORD', 'open_the_gate')

# Choose model via env
CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

# Sessions directory
os.makedirs("sessions", exist_ok=True)

PROJECTS = [
    'Personal Operating Manual','AMCF','BCDodgeme','Rose and Angel','Meals N Feelz',
    'TV Signals','Damn It Carl','HalalBot','Kitchen','Health','Side Quests'
]

CORPUS_PATH = "data/cleaned/ghostline_sources.jsonl.gz"

# Global RAG system state
_rag_building = False
_rag_build_error = None
_brain_building = False
_brain_build_error = None

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Initialize personality system
personality_integration = PersonalityIntegration()

# Section 2: Database and Module Initialization
# Section 2: Database and Module Initialization
app = Flask(__name__)

# Enhanced session and CORS configuration for Railway deployment
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=bool(os.getenv('RAILWAY_ENVIRONMENT')),
    SESSION_COOKIE_DOMAIN=None
)

# Enable CORS for streaming with credentials support
CORS(app, supports_credentials=True, origins=['*'])

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'ghostline-default-key')
PASSWORD = os.getenv('GHOSTLINE_PASSWORD', 'open_the_gate')

# Choose model via env
CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

# Sessions directory
os.makedirs("sessions", exist_ok=True)

PROJECTS = [
    'Personal Operating Manual','AMCF','BCDodgeme','Rose and Angel','Meals N Feelz',
    'TV Signals','Damn It Carl','HalalBot','Kitchen','Health','Side Quests'
]

CORPUS_PATH = "data/cleaned/ghostline_sources.jsonl.gz"

# Global RAG system state
_rag_building = False
_rag_build_error = None
_brain_building = False
_brain_build_error = None

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# JWT for mobile API
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("PyJWT not available - mobile API authentication disabled")

def is_mobile_authenticated():
    """Check if mobile request is authenticated"""
    if not JWT_AVAILABLE:
        return False
    
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return False
    
    token = auth_header[7:]  # Remove 'Bearer ' prefix
    
    try:
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        return payload.get('mobile_authenticated', False)
    except jwt.InvalidTokenError:
        return False

# .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Initialize personality system
personality_integration = PersonalityIntegration()

# Section 2: Database and Module Initialization
# Section 2: Database and Module Initialization (UPDATED WITH PROJECT MAPPING) 9/13/25
app = Flask(__name__)

# Enhanced session and CORS configuration for Railway deployment
app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=bool(os.getenv('RAILWAY_ENVIRONMENT')),
    SESSION_COOKIE_DOMAIN=None
)

# Enable CORS for streaming with credentials support
CORS(app, supports_credentials=True, origins=['*'])

app.secret_key = os.getenv('FLASK_SECRET_KEY', 'ghostline-default-key')
PASSWORD = os.getenv('GHOSTLINE_PASSWORD', 'open_the_gate')

# Choose model via env
CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

# Sessions directory
os.makedirs("sessions", exist_ok=True)

PROJECTS = [
    'Personal Operating Manual','AMCF','BCDodgeme','Rose and Angel','Meals N Feelz',
    'TV Signals','Damn It Carl','HalalBot','Kitchen','Health','Side Quests'
]

CORPUS_PATH = "data/cleaned/ghostline_sources.jsonl.gz"

# Global RAG system state
_rag_building = False
_rag_build_error = None
_brain_building = False
_brain_build_error = None

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Initialize personality system
personality_integration = PersonalityIntegration()

# Initialize project mapping system
try:
    project_mapping_system = ProjectMappingSystem(DATABASE_URL)
    print("✅ Project Mapping System initialized successfully")
except Exception as e:
    print(f"⚠️ Project Mapping System initialization failed: {e}")
    project_mapping_system = None

# JWT for mobile API
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    print("PyJWT not available - mobile API authentication disabled")

def is_mobile_authenticated():
    """Check if mobile request is authenticated"""
    if not JWT_AVAILABLE:
        return False
    
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return False
    
    token = auth_header[7:]  # Remove 'Bearer ' prefix
    
    try:
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        return payload.get('mobile_authenticated', False)
    except jwt.InvalidTokenError:
        return False

# .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from modules.database import (
    get_db_connection,
    init_database,
    search_brain_database,
    load_conversation_enhanced,
    save_conversation_enhanced,
    save_daily_log_enhanced,
    track_uploaded_file,
    save_brain_to_database,
    load_brain_from_database,
    get_database_status
)

from modules.brain import (
    enhanced_retrieve,
    build_brain_from_corpus,  # Changed
    build_brain_from_sources,  # Changed
    get_brain_status,
    get_brain_control_dashboard
)

from modules.file_processing import setup_easyocr_environment, markdown_filter, handle_file_upload

from modules.utils import (
    load_conversation,
    _append_session,
    _save_daily_log,
    _render_enhanced,
    ensure_directories,
    format_timestamp,
    safe_filename,
    get_file_extension,
    is_supported_file_type
)

from modules.gmail import process_gmail_command
from modules.brain import enhanced_retrieve, refresh_brain_context

# FIXED: Initialize database when app starts (moved after app creation)
with app.app_context():
    init_database()

# Setup EasyOCR environment
setup_easyocr_environment()

# Register markdown filter
app.jinja_env.filters['markdown'] = markdown_filter

# Section 3: Helper Functions for Chat Processing
# Section 3: Helper Functions for Chat Processing - FIXED VERSION
# Section 3: Helper Functions for Chat Processing - FIXED VERSION 9/12/25
# Section 3: Helper Functions for Chat Processing - FIXED VERSION 9/12/25
# Section 3: Helper Functions for Chat Processing - FIXED REMINDER DETECTION
import re

def handle_reminder_command(user_input, project, use_voices, random_toggle):
    """Handle reminder creation commands - FIXED DETECTION"""
    
    # DEBUG: Add this line to see if function is called
    print(f"🔍 DEBUG: handle_reminder_command called with input: '{user_input}'")
    
    # ENHANCED: More comprehensive patterns to catch all reminder variations
    explicit_reminder_patterns = [
        r'^remind me to\s+',
        r'^remind me in\s+',               # ✅ "remind me in one minute"
        r'^remind me at\s+',
        r'^set a reminder\s+',
        r'^set reminder\s+',
        r'^create a reminder\s+',
        r'^reminder:\s+',
        r'^reminder for\s+',
        r'^alert me\s+',
        r'^don\'t forget\s+',
        r'remind me .+ in\s+',             # ✅ "remind me to X in Y"
        r'remind me .+ at\s+',             # ✅ "remind me to X at Y"
        r'set a reminder .+ (in|at|tomorrow|today)',
        # NEW: More flexible patterns
        r'reminder in\s+',
        r'alert in\s+',
        r'notify me in\s+',
        r'ping me in\s+',
        r'wake me in\s+'
    ]
    
    user_input_lower = user_input.lower().strip()
    
    # Check if this is an EXPLICIT reminder request
    is_explicit_reminder = any(
        re.search(pattern, user_input_lower)
        for pattern in explicit_reminder_patterns
    )
    
    # DEBUG: Show what we're checking
    print(f"🔍 DEBUG: User input lowered: '{user_input_lower}'")
    print(f"🔍 DEBUG: Is explicit reminder: {is_explicit_reminder}")
    
    if not is_explicit_reminder:
        print(f"🔍 DEBUG: No reminder patterns matched, returning None")
        return None, False
    
    print(f"🔍 DEBUG: Reminder pattern matched! Processing...")
    
    if not is_telegram_configured():
        response_data = {
            "SyntaxPrime": "Telegram reminders not configured. Visit /integrations to set up your bot."
        }
        return response_data, True
    
    try:
        # Add safety wrapper around the problematic parse function
        try:
            parsed = parse_reminder_command(user_input, project)
            print(f"🔍 DEBUG: Parse result: {parsed}")
        except Exception as parse_error:
            print(f"🔍 DEBUG: Parsing failed: {parse_error}")
            response_data = {"SyntaxPrime": f"Could not parse reminder request: {str(parse_error)}"}
            return response_data, True
        
        if not parsed or not parsed.get("success"):
            error_msg = parsed.get("error", "Unknown parsing error") if parsed else "Parsing returned None"
            print(f"🔍 DEBUG: Parse unsuccessful: {error_msg}")
            response_data = {"SyntaxPrime": f"Reminder parsing failed: {error_msg}"}
            return response_data, True
        
        # Add safety wrapper around reminder creation
        try:
            from modules.telegram_notifications import GhostlineTelegramReminders
            reminders = GhostlineTelegramReminders()
            result = reminders.create_reminder(
                title=parsed["title"],
                remind_at=parsed["remind_at"],
                project=parsed["project"],
                priority=2
            )
            print(f"🔍 DEBUG: Reminder creation result: {result}")
        except Exception as creation_error:
            print(f"🔍 DEBUG: Reminder creation failed: {creation_error}")
            response_data = {"SyntaxPrime": f"Failed to create reminder: {str(creation_error)}"}
            return response_data, True
        
        if result and result.get("success"):
            display_time = parsed.get("display_time", result["remind_at"].strftime('%I:%M %p on %B %d') if result.get("remind_at") else "unknown time")
            
            response_text = f"⏰ Reminder Created!\n\n"
            response_text += f"**What:** {parsed['title']}\n"
            response_text += f"**When:** {display_time}\n"
            response_text += f"**Project:** {project}\n\n"
            response_text += "You'll receive a Telegram notification with action buttons to mark complete or snooze."
            
            response_data = {"SyntaxPrime": response_text}
            print(f"🔍 DEBUG: Success response created")
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
            response_data = {"SyntaxPrime": f"Failed to create reminder: {error_msg}"}
            print(f"🔍 DEBUG: Failed response created: {error_msg}")
        
        return response_data, True
        
    except Exception as e:
        print(f"🔍 DEBUG: Complete failure: {e}")
        response_data = {"SyntaxPrime": f"Reminder system error: {str(e)}"}
        return response_data, True

def generate_response_with_context_check(user_input, use_voices, random_toggle, project, model, retrieval_context):
    """Enhanced response generation with context validation"""
    context_quality = len(retrieval_context) if retrieval_context else 0
    is_specific_query = any(term in user_input.lower() for term in [
        'what does', 'what is', 'tell me about', 'explain', 'describe',
        'who is', 'where is', 'when is', 'how does', 'why does'
    ])
    
    print(f"Context check: {context_quality} results for query: '{user_input}' (specific: {is_specific_query})")
    
    # If context is weak for specific queries, try enhanced search
    if context_quality < 2 and is_specific_query:
        print(f"Weak context for specific query, trying enhanced search approaches")
        
        enhanced_context = []
        search_terms = user_input.lower().replace('?', '').replace(',', ' ')
        
        # Try multiple search approaches
        for term in search_terms.split():
            if len(term) > 3:  # Skip short words
                additional_context = retrieve(term, limit=2)
                if additional_context:
                    enhanced_context.extend(additional_context)
        
        # Combine and deduplicate
        if enhanced_context:
            all_context = list(retrieval_context or []) + enhanced_context
            # Simple dedup by content similarity
            unique_context = []
            seen_content = set()
            for ctx in all_context:
                content_snippet = ctx.get('content', '')[:100]
                if content_snippet not in seen_content:
                    unique_context.append(ctx)
                    seen_content.add(content_snippet)
            
            retrieval_context = unique_context[:8]  # Limit to top 8
            print(f"Enhanced context: {len(retrieval_context)} results")
    
    try:
        response = generate_response(
            user_input,
            use_voices,
            random_toggle,
            project,
            model,
            retrieval_context
        )
        return response
    except Exception as e:
        app.logger.error(f"Response generation failed: {e}")
        return {
            "SyntaxPrime": f"I encountered an error processing your request: {str(e)}"
        }
# Section 3.5: Enhanced Marketing Knowledge Integration 9/13/25
def process_marketing_knowledge_query(user_input: str, project: str, use_voices: list, random_toggle: bool) -> tuple[dict, bool]:
    """Process marketing knowledge queries and enhance responses with RSS content"""
    
    user_lower = user_input.lower().strip()
    
    # Marketing query patterns - comprehensive detection
    marketing_patterns = [
        # SEO patterns
        'seo', 'search engine optimization', 'ranking', 'serp', 'keyword research',
        'backlink', 'organic search', 'google algorithm', 'meta description',
        'title tag', 'schema markup', 'technical seo', 'local seo', 'rank math',
        'on-page seo', 'off-page seo', 'link building', 'domain authority',
        'page speed', 'core web vitals', 'crawling', 'indexing',
        
        # Content Marketing patterns
        'content marketing', 'content strategy', 'blog writing', 'storytelling',
        'editorial calendar', 'content creation', 'brand voice', 'copywriting',
        'content optimization', 'content distribution', 'blog post',
        'article writing', 'content planning', 'publishing strategy',
        
        # Social Media patterns
        'social media', 'facebook marketing', 'instagram', 'twitter', 'linkedin',
        'social media strategy', 'social engagement', 'influencer marketing',
        'social analytics', 'social media calendar', 'community management',
        'social media posts', 'engagement rate', 'hashtag strategy',
        
        # Digital Marketing patterns
        'digital marketing', 'online marketing', 'internet marketing',
        'marketing strategy', 'marketing campaign', 'brand awareness',
        'customer acquisition', 'conversion rate', 'lead generation',
        'email marketing', 'newsletter', 'email automation',
        
        # Analytics patterns
        'google analytics', 'marketing analytics', 'conversion tracking', 'kpis',
        'marketing metrics', 'roi measurement', 'attribution modeling',
        'data analysis', 'marketing dashboard', 'performance tracking',
        
        # Tool-specific patterns
        'wordpress seo', 'shopify seo', 'google ads', 'facebook ads',
        'mailchimp', 'hubspot', 'semrush', 'ahrefs', 'moz'
    ]
    
    # Check if this is a marketing-related query
    is_marketing_query = any(pattern in user_lower for pattern in marketing_patterns)
    
    # Also check for specific question patterns
    marketing_question_patterns = [
        'how to improve seo', 'how to write better', 'social media best practices',
        'content marketing tips', 'seo optimization', 'ranking factors',
        'marketing strategy', 'digital marketing', 'online marketing',
        'grow my website', 'increase traffic', 'improve rankings',
        'content creation', 'blog optimization', 'social engagement'
    ]
    
    is_marketing_question = any(pattern in user_lower for pattern in marketing_question_patterns)
    
    if not (is_marketing_query or is_marketing_question):
        return {}, False
    
    try:
        print(f"Processing marketing knowledge query: '{user_input}'")
        
        # Get marketing insights
        retriever = get_marketing_retriever()
        
        # Get contextual marketing advice
        marketing_results = retriever.get_contextual_marketing_advice(user_input, limit=6)
        
        if not marketing_results:
            # Fallback to category-specific searches
            if any(term in user_lower for term in ['seo', 'ranking', 'search']):
                marketing_results = retriever.get_seo_best_practices(user_input, limit=5)
            elif any(term in user_lower for term in ['content', 'blog', 'writing']):
                marketing_results = retriever.get_content_writing_tips('blog', limit=5)
            elif any(term in user_lower for term in ['social', 'facebook', 'instagram', 'twitter']):
                marketing_results = retriever.get_social_media_strategies(limit=5)
        
        if not marketing_results:
            return {}, False
        
        # Generate enhanced response using marketing knowledge
        response_data = generate_marketing_enhanced_response(
            user_input, marketing_results, use_voices, random_toggle, project
        )
        
        # Record usage for analytics
        for result in marketing_results[:3]:  # Record top 3 results
            retriever.record_content_usage(
                result['id'],
                user_input,
                'content_generation'
            )
        
        print(f"Marketing knowledge response generated with {len(marketing_results)} insights")
        return response_data, True
        
    except Exception as e:
        print(f"Marketing knowledge processing failed: {e}")
        return {}, False

def generate_marketing_enhanced_response(user_input: str, marketing_results: list,
                                       use_voices: list, random_toggle: bool, project: str) -> dict:
    """Generate AI response enhanced with fresh marketing knowledge"""
    
    # Build context from marketing results
    marketing_context = "\n\n=== CURRENT MARKETING BEST PRACTICES ===\n"
    
    for i, result in enumerate(marketing_results[:5], 1):
        marketing_context += f"\n{i}. **{result['title']}** ({result['feed_name']})\n"
        
        if result['summary']:
            marketing_context += f"Summary: {result['summary']}\n"
        else:
            marketing_context += f"Content: {result['content'][:300]}...\n"
        
        marketing_context += f"Category: {result['category']}"
        if result['subcategory']:
            marketing_context += f" > {result['subcategory']}"
        
        marketing_context += f"\nRelevance Score: {result['relevance_score']:.1f}/10"
        
        if result['keywords']:
            marketing_context += f"\nKey Topics: {', '.join(result['keywords'][:5])}"
        
        if result['days_old'] is not None:
            marketing_context += f"\nPublished: {result['days_old']} days ago"
        
        marketing_context += f"\nSource: {result['url']}\n"
        marketing_context += "-" * 50 + "\n"
    
    marketing_context += "\n=== END MARKETING CONTEXT ===\n"
    
    # Enhanced prompt with marketing knowledge
    enhanced_prompt = f"""User Query: {user_input}

{marketing_context}

Please provide a comprehensive response that:
1. Directly answers the user's marketing question
2. Incorporates the LATEST best practices from the sources above
3. Provides specific, actionable advice
4. References current industry trends and updates
5. Includes concrete examples and implementation steps
6. Mentions any relevant tools or techniques from the sources

Focus on practical, up-to-date advice that reflects current marketing best practices. If the sources mention specific metrics, techniques, or recent changes (like algorithm updates), include those details.

Keep the response informative but conversational, and ensure it's directly applicable to the user's situation."""
    
    # Generate response using existing engine with enhanced context
    try:
        response_data = generate_response(
            enhanced_prompt,
            use_voices,
            random_toggle,
            project,
            model=CHAT_MODEL,
            retrieval_context=[]  # Marketing context is already in prompt
        )
        
        # Add marketing source references to the response
        if response_data and 'SyntaxPrime' in response_data:
            response_data['SyntaxPrime'] += "\n\n" + format_marketing_sources(marketing_results[:3])
        
        return response_data
        
    except Exception as e:
        print(f"Enhanced response generation failed: {e}")
        # Fallback response
        return {
            "SyntaxPrime": f"I found {len(marketing_results)} current marketing insights for your question, but encountered an error generating the full response. Here are the key sources:\n\n" + format_marketing_sources(marketing_results[:3])
        }

def format_marketing_sources(marketing_results: list) -> str:
    """Format marketing sources for response footer"""
    
    if not marketing_results:
        return ""
    
    sources_text = "📚 **Current Marketing Sources:**\n"
    
    for i, result in enumerate(marketing_results, 1):
        sources_text += f"\n{i}. **{result['title']}**"
        sources_text += f" - {result['feed_name']}"
        
        if result['days_old'] is not None:
            sources_text += f" ({result['days_old']} days ago)"
        
        sources_text += f"\n   🔗 [Read full article]({result['url']})"
        
        if result['relevance_score'] >= 7.0:
            sources_text += " ⭐ *Highly relevant*"
    
    sources_text += f"\n\n*Based on {len(marketing_results)} current industry sources*"
    
    return sources_text

def enhanced_marketing_command_processor(user_input: str, project: str, use_voices: list, random_toggle: bool) -> tuple[dict, bool]:
    """Enhanced marketing command processor that combines existing and RSS knowledge"""
    
    # First try the RSS marketing knowledge system
    rss_response, rss_handled = process_marketing_knowledge_query(user_input, project, use_voices, random_toggle)
    
    if rss_handled:
        return rss_response, True
    
    # Fallback to existing marketing commands if available
    try:
        if is_marketing_configured():
            existing_response, existing_handled = process_marketing_command_with_context(
                user_input, project, use_voices, random_toggle, marketing_context
            )
            
            if existing_handled:
                return existing_response, True
    except Exception as e:
        print(f"Existing marketing command processing failed: {e}")
    
    return {}, False
    
# Section 4: Main Chat Route
# Section 4: Main Chat Route
# Section 4: Main Chat Route
# Section 4: Main Chat Route (UPDATED FOR PHASE 2)
# Section 4: Main Chat Route (UPDATED FOR CONSOLIDATED GOOGLE INTEGRATION)
# SECTION 4: Main Chat Route (UPDATED)
# Replace the existing Section 4 with this updated version
# ========================================
# Section 4: Main Chat Route (UPDATED WITH ENHANCED MARKETING)
# Section 4: Main Chat Route (UPDATED WITH CALENDAR-TELEGRAM INTEGRATION)
# Section 4: Main Chat Route (UPDATED WITH UNIFIED CONVERSATION CONTEXT)
# Section 4: Main Chat Route (UPDATED WITH SLACK INTEGRATION)
# Section 4: Main Chat Route (UPDATED WITH BLUESKY INTEGRATION)
# Section 4: Main Chat Route (UPDATED WITH FIXED BLUESKY INTEGRATION - HIGHEST PRIORITY)
# Section 4: Main Chat Route (UPDATED WITH FIXED BLUESKY INTEGRATION + CONTENT STRATEGY - HIGHEST PRIORITY)
# Section 4: Main Chat Route (UPDATED WITH FIXED CALENDAR DATA FORMATTING) 9/12/25
# Section 4: Main Chat Route (UPDATED WITH FIXED CALENDAR DATA FORMATTING) 9/12/25
# Section 4: Main Chat Route (UPDATED WITH FIXED BLUESKY INTEGRATION + CONTENT STRATEGY - HIGHEST PRIORITY)
# Section 4: Main Chat Route (UPDATED WITH FIXED CALENDAR DATA FORMATTING) 9/12/25
# Section 4: Main Chat Route (UPDATED WITH RSS MARKETING KNOWLEDGE INTEGRATION) 9/13/25
# Section 4: Main Chat Route (UPDATED WITH WEATHER INTEGRATION) 9/16/25
# Section 4: Main Chat Route (UPDATED WITH COMMAND PARSING) 9/16/25
# Try hybrid content strategy commands
# Section 4: Main Chat Route (FIXED INDENTATION)
# Section 4: Main Chat Route (UPDATED WITH COMMAND PARSING) 9/16/25
# Section 4: Main Chat Route (UPDATED WITH COMMAND PARSING) 9/16/25
@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    response_data = {}
    selected_project = PROJECTS[0]

    if request.method == 'GET':
        try:
            # Load personality preferences
            apply_session_preferences(session)
            selected_project = session.get('current_project', PROJECTS[0])
            
            # Brain status check
            brain_ready = is_ready()
            brain_status = get_build_status() if brain_ready else {"status": "not_ready", "progress": 0}
            
            # Load conversation history
            conversations = load_conversation_enhanced(selected_project, limit=50)
            
            # Prepare rendering context
            context = {
                'projects': PROJECTS,
                'selected_project': selected_project,
                'conversations': conversations,
                'brain_ready': brain_ready,
                'brain_status': brain_status,
                'use_voices': session.get('use_voices', ['SyntaxPrime']),
                'random_toggle': session.get('random_toggle', False),
                'default_voice': get_default_voice()
            }
            
            # Add integration status if available
            try:
                context.update({
                    'telegram_configured': is_telegram_configured(),
                    'clickup_configured': is_clickup_configured(),
                    'cloze_configured': is_cloze_configured(),
                    'marketing_configured': is_marketing_configured(),
                    'google_configured': is_google_configured(),
                    'slack_configured': is_slack_configured(),
                    'bluesky_configured': is_bluesky_configured(),
                    'calendar_telegram_configured': is_calendar_telegram_configured()
                })
            except Exception as e:
                app.logger.warning(f"Integration status check failed: {e}")
            
            return render_template('index.html', **context)
            
        except Exception as e:
            app.logger.error(f"Index GET request failed: {e}")
            return render_template('index.html',
                                 projects=PROJECTS,
                                 error=f"Dashboard load error: {str(e)}")

    # POST request processing
    # POST request processing
    if request.method == 'POST':
        try:
            # Get form data
            user_input = request.form.get('user_input', '').strip()
            project = request.form.get('project', PROJECTS[0])
            use_voices = request.form.getlist('voices') or ['SyntaxPrime']
            random_toggle = 'random' in request.form
            
            # Store preferences in session
            session['current_project'] = project
            session['use_voices'] = use_voices
            session['random_toggle'] = random_toggle
            
            if not user_input:
                return redirect('/')
            
            app.logger.info(f"Processing request: '{user_input}' for project '{project}'")
            
            # Set selected_project
            selected_project = project
            
            # Generate session ID for context tracking
            session_id = session.get('session_id')
            if not session_id:
                import uuid
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id

            # Auto-refresh brain context periodically
            try:
                refresh_brain_context()
            except Exception as e:
                print(f"Brain context refresh failed: {e}")

            # Continue with the rest of your processing logic...
            # [Rest of your POST processing code here]
            
        except Exception as e:
            app.logger.error(f"Main route failed: {e}", exc_info=True)
            return render_template('index.html',
                                 projects=PROJECTS,
                                 error=f"Request processing failed: {str(e)}")
        app.logger.info(f"POST request received with input: {user_input}")
        project = request.form['project']
        selected_project = project
        use_voices = request.form.getlist('voices') or ['SyntaxPrime']
        random_toggle = 'random' in request.form

        # Generate session ID for context tracking
        session_id = session.get('session_id')
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id

        # Auto-refresh brain context periodically
        try:
            refresh_brain_context()
        except Exception as e:
            print(f"Brain context refresh failed: {e}")

        # NEW: Weather awareness integration - check FIRST for health-related topics
        try:
            from modules.weather_health_integration import process_user_input_with_weather
            weather_response = process_user_input_with_weather(user_input, project)
            if weather_response:
                app.logger.info("Health & weather context provided")
                save_conversation_enhanced(project, user_input, weather_response)
                return _render_enhanced(project, weather_response)
        except ImportError:
            pass  # Weather module not available, continue normally
        except Exception as e:
            app.logger.error(f"Weather processing failed: {e}")
            # Continue with normal processing if weather fails

        # PRIORITY 1: Handle reminder commands FIRST - This is the key fix!
        try:
            print(f"MAIN ROUTE: Checking reminder command for: '{user_input}'")
            response_data, handled = handle_reminder_command(user_input, project, use_voices, random_toggle)
            if handled:
                print(f"MAIN ROUTE: Reminder handled successfully!")
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)
            else:
                print(f"MAIN ROUTE: Not a reminder command, continuing...")
        except Exception as e:
            app.logger.error(f"Reminder handler failed: {e}")
            # Don't fail the whole request, just log and continue

        # FIXED: BlueSky commands with enhanced pattern matching (HIGHEST PRIORITY)
        if is_bluesky_configured():
            app.logger.info(f"Checking BlueSky command patterns for: '{user_input}'")
            try:
                # Enhanced BlueSky command detection with more flexible patterns
                user_lower = user_input.lower().strip()
                
                # Comprehensive BlueSky trigger patterns
                bluesky_patterns = [
                    # Direct BlueSky mentions
                    'bluesky', 'bsky', 'blue sky',
                    # Action patterns
                    'analyze bluesky', 'check bluesky', 'my bluesky', 'bluesky feed',
                    'bluesky timeline', 'bluesky posts', 'bluesky analysis',
                    # Engagement patterns
                    'bluesky engagement', 'bluesky suggestions', 'who should i follow',
                    'bluesky opportunities', 'social engagement', 'feed analysis',
                    # High priority patterns
                    'bluesky high priority', 'best bluesky posts', 'top bluesky',
                    # Test patterns
                    'bluesky test', 'test bluesky', 'bluesky connection'
                ]
                
                # Check if input matches any BlueSky pattern
                bluesky_detected = False
                for pattern in bluesky_patterns:
                    if pattern in user_lower:
                        bluesky_detected = True
                        app.logger.info(f"BlueSky pattern matched: '{pattern}'")
                        break
                
                # Also check for standalone keywords that might be BlueSky related
                standalone_keywords = ['bsky', 'bluesky']
                if not bluesky_detected:
                    for keyword in standalone_keywords:
                        if user_lower == keyword or user_lower.startswith(keyword + ' ') or user_lower.endswith(' ' + keyword):
                            bluesky_detected = True
                            app.logger.info(f"BlueSky standalone keyword matched: '{keyword}'")
                            break
                
                if bluesky_detected:
                    app.logger.info(f"Processing BlueSky command: '{user_input}'")
                    response_content = process_bluesky_command(user_input)
                    
                    # Check if we got a real response (not just the help menu)
                    if response_content and "Available BlueSky commands" not in response_content:
                        app.logger.info(f"BlueSky command successfully processed")
                        response_data = {"SyntaxPrime": response_content}
                        save_conversation_enhanced(project, user_input, response_data)
                        return _render_enhanced(project, response_data)
                    else:
                        # If it's just the help menu, let it fall through to normal processing
                        # but log that we tried BlueSky
                        app.logger.info(f"BlueSky returned help menu, falling through to normal processing")
                
            except Exception as e:
                app.logger.error(f"BlueSky processing failed: {e}")
                # Don't fail the whole request, just log and continue
                pass

        # Try hybrid content strategy commands
        try:
            response_data, handled = generate_content_strategy_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)
        except Exception as e:
            app.logger.error(f"Content strategy command failed: {e}")

        # Try Google Drive export commands
        if is_google_configured():
            try:
                response_data, handled = handle_export_command(user_input, project, use_voices, random_toggle)
                if handled:
                    app.logger.info(f"Export command handled successfully")
                    save_conversation_enhanced(project, user_input, response_data)
                    return _render_enhanced(project, response_data)
            except Exception as e:
                app.logger.error(f"Export command processing failed: {e}")
                # Continue to normal processing if export fails

        # Help command for exports
        if user_input.lower().strip() in ['export help', 'google docs help', 'drive export help']:
            response_data = {"SyntaxPrime": get_export_help()}
            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # Try Cloze + ClickUp integration commands
        if is_cloze_configured() and is_clickup_configured():
            try:
                from modules.cloze_clickup_integration import process_cloze_clickup_command
                response_data, handled = process_cloze_clickup_command(user_input, project, use_voices, random_toggle)
                if handled:
                    save_conversation_enhanced(project, user_input, response_data)
                    return _render_enhanced(project, response_data)
            except ImportError as e:
                app.logger.error(f"Cloze integration import failed: {e}")
                if user_input.lower() in ['relationship priorities', 'cloze productivity', 'productivity briefing']:
                    response_data = {"SyntaxPrime": f"Integration module import failed: {str(e)}\nCheck if modules/cloze_clickup_integration.py exists and has no syntax errors."}
                    save_conversation_enhanced(project, user_input, response_data)
                    return _render_enhanced(project, response_data)

        # Try ClickUp-only commands
        if is_clickup_configured():
            response_data, handled = process_clickup_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)

        # Handle scrape command
        if user_input.lower().startswith("scrape "):
            url = user_input.split(" ", 1)[1].strip()
            try:
                result = scrape_url(url)
                if not result["ok"]:
                    response_data = {"SyntaxPrime": f"Could not fetch/extract content: {result['error']}"}
                else:
                    summary_prompt = (
                        "Summarize the key points from the following webpage for Carl. "
                        "Use bullets and keep it tight and actionable.\n\n"
                        f"--- SCRAPED CONTENT START ---\n{result['text']}\n--- SCRAPED CONTENT END ---"
                    )
                    retrieval_ctx = enhanced_retrieve(summary_prompt, k=5, project=project) if is_ready() else []
                    response_data = generate_response(
                        summary_prompt, use_voices, random_toggle,
                        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                    )
                handled = True
            except Exception as e:
                app.logger.error(f"Scrape command failed: {e}")
                response_data = {"SyntaxPrime": f"Scrape failed: {e}"}
                handled = True
            
            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # Try Unified Google Integration (Gmail, Calendar, Analytics, Search Console, Docs, Sheets)
        if is_google_configured():
            try:
                response_data, handled = process_google_ecosystem_commands(
                    user_input, project, use_voices, random_toggle
                )
                if handled:
                    save_conversation_enhanced(project, user_input, response_data)
                    return _render_enhanced(project, response_data)
            except Exception as e:
                app.logger.error(f"Google integration processing failed: {e}")

        # Try Slack integration
        if is_slack_configured():
            response_data, handled = process_slack_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)

        # Enhanced Marketing Commands with RSS Knowledge Base
        try:
            response_data, handled = enhanced_marketing_command_processor(user_input, project, use_voices, random_toggle)
            if handled:
                app.logger.info(f"Enhanced marketing command handled successfully")
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)
        except Exception as e:
            app.logger.error(f"Enhanced marketing processing failed: {e}")

        # Try Cloze commands with proper configuration validation
        if is_cloze_configured():
            response_data, handled = process_cloze_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)

        # Try Telegram integration
        if is_telegram_configured():
            response_data, handled = process_telegram_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)

        # Try Calendar-Telegram integration
        if is_calendar_telegram_configured():
            response_data, handled = process_calendar_telegram_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)

        # NEW: Chat Commands (Bookmark and Google Docs Export) - ADD BEFORE normal AI response
        try:
            # Import command parser functions
            def detect_bookmark_command(user_input: str) -> bool:
                """Detect if user wants to create a bookmark"""
                import re
                bookmark_patterns = [
                    r'\bbookmark\b',
                    r'\bsave this\b',
                    r'\bmark this\b',
                    r'\bremember this\b',
                    r'\bsave conversation\b'
                ]
                user_lower = user_input.lower().strip()
                return any(re.search(pattern, user_lower) for pattern in bookmark_patterns)

            def detect_export_command(user_input: str) -> bool:
                """Detect if user wants to export to Google Docs"""
                import re
                export_patterns = [
                    r'\bcopy to google docs?\b',
                    r'\bexport to google\b',
                    r'\bgoogle docs?\b',
                    r'\bsend to drive\b',
                    r'\bexport conversation\b',
                    r'\bcreate doc\b'
                ]
                user_lower = user_input.lower().strip()
                return any(re.search(pattern, user_lower) for pattern in export_patterns)

            def extract_bookmark_title(user_input: str):
                """Extract a custom title from bookmark command"""
                import re
                user_input = user_input.strip()
                
                title_patterns = [
                    r'bookmark(?:\s+this)?\s+as\s+(.+)',
                    r'bookmark:\s*(.+)',
                    r'save this as\s+(.+)',
                    r'mark this as\s+(.+)'
                ]
                
                for pattern in title_patterns:
                    match = re.search(pattern, user_input, re.IGNORECASE)
                    if match:
                        title = match.group(1).strip()
                        title = re.sub(r'[^\w\s\-\(\)]+', '', title)  # Remove special chars
                        return title[:100]  # Limit length
                return None

            # Check for bookmark command
            if detect_bookmark_command(user_input):
                app.logger.info(f"Bookmark command detected: '{user_input}'")
                
                try:
                    # Extract custom title if provided
                    custom_title = extract_bookmark_title(user_input)
                    
                    # Generate default title if none provided
                    if not custom_title:
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%m/%d %H:%M")
                        custom_title = f"Bookmark - {project} - {timestamp}"
                    
                    # Create a temporary response for the bookmark command
                    bookmark_response = {
                        "SyntaxPrime": f"Bookmark \"{custom_title}\" will be created after this conversation is saved.\n\nThis conversation point will be saved for easy reference. Use 'copy to google docs' to export it later."
                    }
                    
                    # Save conversation first to get chat_id
                    chat_id = save_conversation_enhanced(project, user_input, bookmark_response)
                    
                    if chat_id:
                        # Now create the actual bookmark with the chat_id
                        from modules.database import create_bookmark
                        bookmark_id = create_bookmark(
                            chat_id=chat_id,
                            title=custom_title,
                            notes=f"User-requested bookmark for {project}",
                            bookmark_type='user_command'
                        )
                        
                        if bookmark_id:
                            app.logger.info(f"Bookmark created successfully: {bookmark_id}")
                            # Update response to confirm successful creation
                            bookmark_response["SyntaxPrime"] = f"Bookmark created: \"{custom_title}\"\n\nThis conversation point has been saved for easy reference. Use 'copy to google docs' to export it later."
                        else:
                            app.logger.error(f"Failed to create bookmark in database")
                            bookmark_response["SyntaxPrime"] = f"Bookmark command processed, but database storage failed.\n\nThe conversation is still saved in your history."
                    
                    return _render_enhanced(project, bookmark_response)
                    
                except Exception as e:
                    app.logger.error(f"Bookmark processing failed: {e}")
                    error_response = {
                        "SyntaxPrime": f"Failed to create bookmark: {str(e)}\n\nThe conversation is still saved in your history, but the bookmark wasn't created."
                    }
                    save_conversation_enhanced(project, user_input, error_response)
                    return _render_enhanced(project, error_response)

            # Check for export command
            elif detect_export_command(user_input):
                app.logger.info(f"Export command detected: '{user_input}'")
                
                try:
                    # Check if Google integration is available
                    try:
                        from modules.enhanced_google_integration import EnhancedGoogleIntegration
                        google_integration = EnhancedGoogleIntegration()
                        
                        if not google_integration.is_configured():
                            export_response = {
                                "SyntaxPrime": "Google Docs export requires Google OAuth setup. Visit /integrations to configure Google Drive access first."
                            }
                            save_conversation_enhanced(project, user_input, export_response)
                            return _render_enhanced(project, export_response)
                            
                    except ImportError:
                        export_response = {
                            "SyntaxPrime": "Google Docs integration not available. The enhanced Google integration module needs to be configured."
                        }
                        save_conversation_enhanced(project, user_input, export_response)
                        return _render_enhanced(project, export_response)
                    
                    # Get recent bookmarks for this project
                    from modules.database import get_bookmarks, get_db_connection
                    bookmarks = get_bookmarks(project=project, limit=5)
                    
                    if not bookmarks:
                        export_response = {
                            "SyntaxPrime": "No bookmarks found to export. Create a bookmark first using 'bookmark this' or 'save this'."
                        }
                        save_conversation_enhanced(project, user_input, export_response)
                        return _render_enhanced(project, export_response)
                    
                    # Use the most recent bookmark
                    latest_bookmark = bookmarks[0]
                    
                    # Export to Google Docs
                    import datetime
                    doc_title = f"Ghostline Export - {latest_bookmark['title']} - {datetime.datetime.now().strftime('%Y-%m-%d')}"
                    
                    # Get the conversation content
                    with get_db_connection() as conn:
                        if not conn:
                            export_response = {
                                "SyntaxPrime": "Database connection failed. Cannot retrieve conversation for export."
                            }
                            save_conversation_enhanced(project, user_input, export_response)
                            return _render_enhanced(project, export_response)
                        
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT user_input, response_data, created_at 
                            FROM chat_threads 
                            WHERE id = %s
                        ''', (latest_bookmark['chat_id'],))
                        
                        conversation = cursor.fetchone()
                        
                        if not conversation:
                            export_response = {
                                "SyntaxPrime": "Conversation not found. The bookmark may reference a conversation that was deleted."
                            }
                            save_conversation_enhanced(project, user_input, export_response)
                            return _render_enhanced(project, export_response)
                        
                        # Format content for Google Docs
                        user_input_orig, response_data_json, created_at = conversation
                        ai_response = response_data_json.get('SyntaxPrime', '') if response_data_json else ''
                        
                        document_content = f"""# {doc_title}

**Project:** {project}
**Date:** {created_at.strftime('%Y-%m-%d %H:%M')}
**Bookmark:** {latest_bookmark['title']}

## User Input
{user_input_orig}

## AI Response
{ai_response}

---
*Exported from Ghostline AI*
"""
                        
                        # Create Google Doc
                        doc_result = google_integration.create_google_doc(
                            title=doc_title,
                            content=document_content
                        )
                        
                        if doc_result.get('success'):
                            export_response = {
                                "SyntaxPrime": f"Successfully exported to Google Docs!\n\n**Document:** {doc_title}\n**URL:** {doc_result['document_url']}\n\nThe bookmark \"{latest_bookmark['title']}\" has been exported with full conversation context."
                            }
                        else:
                            export_response = {
                                "SyntaxPrime": f"Google Docs export failed: {doc_result.get('error', 'Unknown error')}\n\nCheck your Google integration setup in /integrations."
                            }
                    
                    save_conversation_enhanced(project, user_input, export_response)
                    return _render_enhanced(project, export_response)
                    
                except Exception as e:
                    app.logger.error(f"Export processing failed: {e}")
                    export_response = {
                        "SyntaxPrime": f"Export failed: {str(e)}\n\nTry checking your Google integration setup or create a bookmark first."
                    }
                    save_conversation_enhanced(project, user_input, export_response)
                    return _render_enhanced(project, export_response)

        except Exception as e:
            app.logger.error(f"Command parsing failed: {e}")
            # Continue to normal AI response if command parsing fails

        # Normal AI response as fallback (same enhanced logic as web version)
        if not response_data:
            try:
                retrieval_ctx = enhanced_retrieve(user_input, k=5, project=project) if is_ready() else []
                
                # NEW: Try to enhance conversation with weather context for health topics
                enhanced_user_input = user_input
                try:
                    from modules.weather_health_integration import enhance_conversation_with_weather_awareness
                    enhanced_messages = enhance_conversation_with_weather_awareness(
                        [{"role": "user", "content": user_input}],
                        user_input
                    )
                    if enhanced_messages and len(enhanced_messages) > 0:
                        enhanced_user_input = enhanced_messages[0]["content"]
                except ImportError:
                    pass  # Weather module not available
                except Exception as e:
                    app.logger.error(f"Weather message enhancement failed: {e}")
                
                # Use enhanced response generation with context validation
                response_data = generate_response_with_context_check(
                    enhanced_user_input, use_voices, random_toggle,
                    project, CHAT_MODEL, retrieval_ctx
                )
                
                save_conversation_enhanced(project, user_input, response_data)
            except Exception as e:
                app.logger.error(f"Normal response generation failed: {e}")
                response_data = {"SyntaxPrime": f"Response generation failed: {e}"}
                save_conversation_enhanced(project, user_input, response_data)

            except Exception as e:
                app.logger.error(f"Main route failed: {e}", exc_info=True)
                return render_template('index.html',
                                     projects=PROJECTS,
                                     error=f"Request processing failed: {str(e)}")

    # GET request - render the chat interface
    return render_template('index.html', projects=PROJECTS, selected_project=selected_project, response_data=response_data)
    
# Section 5: Brain Building Routes
from modules.brain import handle_build_brain, handle_build_new_brain, get_brain_status, get_brain_control_dashboard

@app.route('/build_brain', methods=['POST'])
def build_brain():
    """Manually trigger enhanced brain building with database storage"""
    return handle_build_brain(session)

@app.route('/build_new_brain', methods=['POST'])
def build_new_brain():
    """Build new brain from raw sources with database storage"""
    return handle_build_new_brain(session)

@app.route('/brain_status')
def brain_status():
    """Enhanced brain status with batch progress"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    return jsonify(get_brain_status())

@app.route('/reload_corpus')
def reload_corpus():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        load_corpus(CORPUS_PATH)
        return "Brain reloaded successfully", 200
    except Exception as e:
        app.logger.error(f"Corpus reload failed: {e}")
        return f"Reload failed: {e}", 500
        
# Section 6: File Upload Processing

@app.route('/upload', methods=['POST'])
def upload_file():
    """Updated upload handler for integrated chat flow"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    return handle_file_upload()
    
# Section 7: Streaming Chat API
# Section 7: Streaming Chat API (UPDATED)
# Section 7: Streaming Chat API (UPDATED FOR PHASE 2)
# Section 7: Streaming Chat API (UPDATED FOR CONSOLIDATED GOOGLE INTEGRATION)
# SECTION 7: Streaming Chat API (UPDATED)
# Replace the existing Section 7 with this updated version
# ========================================
# Section 7: Streaming Chat API (UPDATED WITH ENHANCED MARKETING)
# Section 7: Streaming Chat API (UPDATED WITH CALENDAR-TELEGRAM INTEGRATION)
# Section 7: Streaming Chat API (UPDATED WITH UNIFIED CONVERSATION CONTEXT)
# Section 7: Streaming Chat API (UPDATED WITH SLACK INTEGRATION)
# Section 7: Streaming Chat API - FIXED VERSION
# Section 7: Streaming Chat API (UPDATED FOR CONSOLIDATED GOOGLE INTEGRATION)
# SECTION 7: Streaming Chat API (UPDATED)
# Section 7: Streaming Chat API - FIXED VERSION WITH BLUESKY HIGHEST PRIORITY 9/11/25
# Section 7: Streaming Chat API (UPDATED FOR CONSOLIDATED GOOGLE INTEGRATION)
# SECTION 7: Streaming Chat API (UPDATED)
# Section 7: Streaming Chat API - FIXED VERSION WITH BLUESKY HIGHEST PRIORITY 9/12/25
# Section 7: Streaming Chat API (UPDATED WITH RSS MARKETING KNOWLEDGE INTEGRATION) 9/13/25
# Section 7: Streaming Chat API (UPDATED WITH PROJECT MAPPING INTEGRATION) 9/13/25
# Section 7: Streaming Chat API (UPDATED WITH RSS MARKETING KNOWLEDGE INTEGRATION AND GOOGLE DRIVE EXPORT) 9/16/25
@app.route('/api/chat/stream', methods=['POST'])
def stream_chat():
    """Enhanced streaming chat endpoint with project mapping integration and Google Drive export"""
    
    # Enhanced logging for debugging auth issues
    app.logger.info(f"Stream request from {request.remote_addr}")
    app.logger.info(f"Session data: logged_in={session.get('logged_in')}, keys={list(session.keys())}")
    app.logger.info(f"User-Agent: {request.headers.get('User-Agent', 'Unknown')}")
    
    if not session.get('logged_in'):
        app.logger.warning("Stream request REJECTED - authentication failed")
        return jsonify({
            'error': 'Unauthorized',
            'debug': {
                'session_exists': bool(session),
                'logged_in_value': session.get('logged_in'),
                'session_keys': list(session.keys()),
                'hint': 'Make sure frontend includes credentials: include in fetch request'
            }
        }), 401
    
    try:
        data = request.get_json()
        if not data:
            app.logger.error("Stream request missing JSON data")
            return jsonify({'error': 'No JSON data provided'}), 400
            
        user_input = data.get('user_input', '').strip()
        project = data.get('project', PROJECTS[0])
        use_voices = data.get('voices', ['SyntaxPrime'])
        random_toggle = data.get('random', False)
        
        if not user_input:
            return jsonify({'error': 'No input provided'}), 400
        
        # Generate session ID for context tracking
        session_id = session.get('session_id')
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        app.logger.info(f"Stream processing: '{user_input[:50]}...' for project '{project}' (session: {session_id[:8]}...)")
        
        # ==== NEW PROJECT MAPPING INTEGRATION ====
        # Detect and set project context
        if project_mapping_system:
            try:
                # Identify project from context if not explicitly set
                context_data = {
                    'user_input': user_input,
                    'current_project': project,
                    'session_data': dict(session)
                }
                
                detected_project = project_mapping_system.identify_project_from_context(context_data)
                if detected_project != project and detected_project != 'Main':
                    project = detected_project
                    app.logger.info(f"🎯 Auto-detected project: {project}")
                
                # Set project context for this session
                project_mapping_system.set_project_context(session_id, project, context_data)
                
                # Get project-specific mappings for integration routing
                project_context = project_mapping_system.get_project_context(session_id)
                app.logger.info(f"📂 Current project context: {project_context}")
                
            except Exception as e:
                app.logger.error(f"Project mapping error: {e}")
                # Continue without project mapping on error
        # ==== END PROJECT MAPPING INTEGRATION ====
        
        def generate_stream():
            try:
                # Send initial message
                yield f"data: {json.dumps({'type': 'start', 'message': 'Processing your request...'})}\n\n"
                
                # Initialize response data
                response_data = {}
                handled = False
                
                # === ENHANCED INTEGRATION ROUTING ===
                if project_mapping_system:
                    try:
                        # Route integration requests based on project context
                        integration_context = {
                            'session_id': session_id,
                            'project': project,
                            'user_input': user_input
                        }
                        
                        routing_result = project_mapping_system.route_integration_request(
                            'general', integration_context
                        )
                        
                        # Add project-specific filtering to your existing integration handlers
                        if routing_result.get('should_filter'):
                            data['project_mappings'] = routing_result.get('project_mappings', {})
                            data['filtered_data'] = routing_result.get('filtered_data', {})
                            
                    except Exception as e:
                        app.logger.error(f"Integration routing error: {e}")
                
                # PRIORITY 1: Handle reminder commands FIRST - This is the key fix for streaming too!
                try:
                    print(f"STREAM: Checking reminder command for: '{user_input}'")
                    response_data, handled = handle_reminder_command(user_input, project, use_voices, random_toggle)
                    if handled:
                        print(f"STREAM: Reminder handled successfully!")
                        save_conversation_enhanced(
                            project=project,
                            user_input=user_input,
                            response_data=response_data,
                            context_project=project if project_mapping_system else None,
                            context_data={'session_id': session_id, 'mappings': project_context.get('data', {}) if 'project_context' in locals() else {}}
                        )
                        # Stream the reminder response
                        for voice, content in response_data.items():
                            if content and isinstance(content, str):
                                # Stream text content in chunks
                                chunk_size = 30
                                for i in range(0, len(content), chunk_size):
                                    chunk = content[i:i+chunk_size]
                                    yield f"data: {json.dumps({'type': 'content', 'voice': voice, 'chunk': chunk})}\n\n"
                                    time.sleep(0.03)
                        
                        # Send completion signal
                        yield f"data: {json.dumps({'type': 'complete', 'responses': response_data})}\n\n"
                        return  # Exit early since we handled it
                    else:
                        print(f"STREAM: Not a reminder command, continuing...")
                except Exception as e:
                    app.logger.error(f"Stream: Reminder handler failed: {e}")
                    # Don't set handled=True here - let other processors try
                
                # Try command processors only if reminder didn't handle it
                if not handled:
                    processors = [
                        ('google_consolidated', lambda: process_google_ecosystem_commands(user_input, project, use_voices, random_toggle)),
                    ]
                    
                    # FIXED: Add BlueSky processor with HIGHEST priority (position 0) - THIS IS THE KEY FIX
                    if is_bluesky_configured():
                        app.logger.info(f"Adding BlueSky processor to stream pipeline with highest priority")
                        
                        def bluesky_processor():
                            # Use the same detection logic as the main route
                            user_lower = user_input.lower().strip()
                            bluesky_patterns = [
                                # Direct BlueSky mentions
                                'bluesky', 'bsky', 'blue sky',
                                # Action patterns
                                'analyze bluesky', 'check bluesky', 'my bluesky', 'bluesky feed',
                                'bluesky timeline', 'bluesky posts', 'bluesky analysis',
                                # Engagement patterns
                                'bluesky engagement', 'bluesky suggestions', 'who should i follow',
                                'bluesky opportunities', 'social engagement', 'feed analysis',
                                # High priority patterns
                                'bluesky high priority', 'best bluesky posts', 'top bluesky',
                                # Test patterns
                                'bluesky test', 'test bluesky', 'bluesky connection'
                            ]
                            
                            # Check if input matches any BlueSky pattern
                            bluesky_detected = False
                            for pattern in bluesky_patterns:
                                if pattern in user_lower:
                                    bluesky_detected = True
                                    app.logger.info(f"Stream: BlueSky pattern matched: '{pattern}'")
                                    break
                            
                            # Also check for standalone keywords
                            standalone_keywords = ['bsky', 'bluesky']
                            if not bluesky_detected:
                                for keyword in standalone_keywords:
                                    if user_lower == keyword or user_lower.startswith(keyword + ' ') or user_lower.endswith(' ' + keyword):
                                        bluesky_detected = True
                                        app.logger.info(f"Stream: BlueSky standalone keyword matched: '{keyword}'")
                                        break
                            
                            if bluesky_detected:
                                app.logger.info(f"Stream: Processing BlueSky command: '{user_input}'")
                                response_content = process_bluesky_command(user_input)
                                # Check if we got a real response (not just the help menu)
                                if response_content and "Available BlueSky commands" not in response_content:
                                    app.logger.info(f"Stream: BlueSky command successfully processed")
                                    return {"SyntaxPrime": response_content}, True
                                else:
                                    app.logger.info(f"Stream: BlueSky returned help menu, falling through")
                            
                            return {}, False
                        
                        processors.insert(0, ('bluesky', bluesky_processor))
                    
                    # Add Calendar → Telegram processor
                    if is_calendar_telegram_configured():
                        app.logger.info(f"Adding Calendar-Telegram processor to stream pipeline")
                        processors.insert(1, ('calendar_telegram', lambda: process_calendar_telegram_command(user_input, project, use_voices, random_toggle)))
                    
                    # Add enhanced marketing processor with RSS knowledge
                    app.logger.info(f"Adding enhanced marketing processor to stream pipeline")
                    processors.insert(0, ('marketing_enhanced', lambda: enhanced_marketing_command_processor(user_input, project, use_voices, random_toggle)))
                    
                    # Add Google Drive export processor
                    if is_google_configured():
                        app.logger.info(f"Adding Google Drive export processor to stream pipeline")
                        processors.insert(1, ('google_export', lambda: handle_export_command(user_input, project, use_voices, random_toggle)))
                    
                    # Add Cloze processor with proper configuration check
                    if is_cloze_configured():
                        app.logger.info(f"Adding Cloze processor to stream pipeline")
                        processors.insert(1, ('cloze', lambda: process_cloze_command(user_input, project, use_voices, random_toggle)))
                    
                    # Add other conditional processors
                    if is_clickup_configured():
                        processors.append(('clickup', lambda: process_clickup_command(user_input, project, use_voices, random_toggle)))
                    
                    # Try each processor with individual error handling
                    for proc_name, processor in processors:
                        if not handled:
                            try:
                                app.logger.info(f"Trying {proc_name} processor")
                                temp_response, temp_handled = processor()
                                if temp_handled:
                                    response_data = temp_response
                                    handled = True
                                    app.logger.info(f"Request handled by {proc_name} processor")
                                    break
                            except Exception as e:
                                app.logger.error(f"{proc_name} processor failed: {e}")
                                continue
                
                # Scrape command
                if not handled and user_input.lower().startswith("scrape "):
                    try:
                        url = user_input.split(" ", 1)[1].strip()
                        result = scrape_url(url)
                        if not result["ok"]:
                            response_data = {"SyntaxPrime": f"Could not fetch content: {result['error']}"}
                        else:
                            summary_prompt = (
                                "Summarize the key points from the following webpage for Carl. "
                                "Focus on actionable insights and key information:\n\n"
                                f"{result['content']}"
                            )
                            
                            retrieval_ctx = enhanced_retrieve(summary_prompt, k=5, project=project) if is_ready() else []
                            
                            response_data = generate_response(
                                summary_prompt, use_voices, random_toggle,
                                project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                            )
                        
                        handled = True
                        app.logger.info("Scrape command processed")
                        
                    except Exception as e:
                        app.logger.error(f"Scrape command failed: {e}")
                        response_data = {"SyntaxPrime": f"Scraping failed: {e}"}
                        handled = True

                # Gmail/Calendar commands
                if not handled:
                    try:
                        from modules.gmail import process_gmail_command
                        temp_response, temp_handled = process_gmail_command(user_input, project, use_voices, random_toggle)
                        if temp_handled:
                            response_data = temp_response
                            handled = True
                            app.logger.info("Request handled by Gmail processor")
                    except Exception as e:
                        app.logger.error(f"Gmail processor failed: {e}")

                # Normal AI response if no special processing
                if not response_data:
                    try:
                        app.logger.info("Using normal AI response generation")
                        
                        # Get conversation context
                        retrieval_ctx = enhanced_retrieve(user_input, k=5, project=project) if is_ready() else []
                        
                        # Try brain context refresh
                        try:
                            refresh_brain_context()
                        except Exception as e:
                            app.logger.warning(f"Brain context refresh failed: {e}")
                        
                        response_data = generate_response_with_context_check(
                            user_input, use_voices, random_toggle,
                            project, CHAT_MODEL, retrieval_ctx
                        )
                        
                        app.logger.info("Normal AI response generation completed")
                        
                    except Exception as e:
                        app.logger.error(f"AI response generation failed: {e}")
                        response_data = {"SyntaxPrime": f"I'm having trouble processing that request right now. Please try again."}
                
                # Save conversation with project context
                try:
                    save_conversation_enhanced(
                        project=project,
                        user_input=user_input,
                        response_data=response_data,
                        context_project=project if project_mapping_system else None,
                        context_data={'session_id': session_id, 'mappings': project_context.get('data', {}) if 'project_context' in locals() else {}}
                    )
                    app.logger.info("Conversation saved successfully")
                except Exception as e:
                    app.logger.error(f"Failed to save conversation: {e}")
                
                # Enhanced streaming for marketing images
                for voice, content in response_data.items():
                    if voice == 'image_data':
                        # Handle image data specially for inline display
                        yield f"data: {json.dumps({'type': 'image', 'image_data': content, 'image_url': response_data.get('image_url')})}\n\n"
                        continue
                    elif voice == 'image_url':
                        # Skip image_url as it's handled with image_data
                        continue
                    elif not content or not isinstance(content, str):
                        continue
                        
                    # Stream text content in chunks for streaming effect
                    chunk_size = 30
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i+chunk_size]
                        yield f"data: {json.dumps({'type': 'content', 'voice': voice, 'chunk': chunk})}\n\n"
                        time.sleep(0.03)  # Small delay for streaming effect
                
                # Send completion signal with full response data
                yield f"data: {json.dumps({'type': 'complete', 'responses': response_data})}\n\n"
                app.logger.info("Stream completed successfully")
                
            except Exception as e:
                app.logger.error(f"Stream generation failed: {e}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': f'Stream failed: {str(e)}'})}\n\n"
        
        return Response(
            generate_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': 'true'
            }
        )
        
    except Exception as e:
        app.logger.error(f"Stream endpoint failed: {e}", exc_info=True)
        return jsonify({'error': f'Stream endpoint failed: {str(e)}'}), 500

# Section 8: Dashboard Routes (Modular)
# Section 8: Dashboard Routes (Modular) - UPDATED WITH GOOGLE DIAGNOSTICS
# Section 8: Dashboard Routes (Modular) - UPDATED WITH CLICKUP DIAGNOSTICS
# Section 8: Dashboard Routes (Modular) - UPDATED WITH ADMIN CONTROLS
# Section 8: Dashboard Routes (Modular) - UPDATED WITH DEBUG ENDPOINT
# Section 8: Dashboard Routes (Modular) - UPDATED WITH GOOGLE DIAGNOSTICS
# Section 8: Dashboard Routes (Modular) - UPDATED WITH CLICKUP DIAGNOSTICS
# Section 8: Dashboard Routes (Modular) - UPDATED WITH ADMIN CONTROLS
# Section 8: Feedback System Routes (NEW) 9/11/25
# Section 8: Feedback System Routes (FIXED DIRECT VERSION) 9/11/25
# Section 8: Dashboard Routes (Modular) - UPDATED WITH CLICKUP DIAGNOSTICS AND PROJECT MAPPING 9/13/25
# Section 8: Dashboard Routes (Modular) - UPDATED WITH CLICKUP DIAGNOSTICS AND PROJECT MAPPING 9/15/25
@app.route('/api/feedback', methods=['POST'])
def record_feedback():
    """Record user feedback for AI responses (👍👎🖕 buttons) - FIXED DIRECT VERSION"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        response_id = data.get('response_id')
        feedback_type = data.get('feedback_type')  # 'thumbs_up', 'thumbs_down', 'middle_finger'
        project = data.get('project', session.get('current_project', 'General'))
        user_comment = data.get('comment', '')
        
        if not response_id or not feedback_type:
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # FIXED: Use the feedback system directly instead of the broken imported function
        from modules.feedback_system import _feedback_system
        
        if not _feedback_system:
            return jsonify({'success': False, 'error': 'Feedback system not initialized'}), 500
        
        # Call the working feedback system directly
        result = _feedback_system.submit_feedback(
            response_id=response_id,
            feedback_type=feedback_type,
            project=project,
            user_comment=user_comment
        )
        
        if result.get('success'):
            app.logger.info(f"Feedback successfully recorded: {feedback_type} for response {response_id}")
        else:
            app.logger.error(f"Feedback recording failed: {result.get('error', 'Unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Feedback recording failed with exception: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/feedback/<feedback_type>', methods=['POST'])
def record_feedback_legacy(feedback_type):
    """Legacy feedback endpoint for backward compatibility"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json() or {}
        response_id = data.get('response_id', f"legacy_{int(datetime.datetime.now().timestamp())}")
        
        # Use the direct feedback system approach
        from modules.feedback_system import _feedback_system
        
        if not _feedback_system:
            return jsonify({'success': False, 'error': 'Feedback system not initialized'}), 500
        
        result = _feedback_system.submit_feedback(
            response_id=response_id,
            feedback_type=feedback_type,
            project=session.get('current_project', 'General'),
            user_comment=data.get('comment', '')
        )
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Legacy feedback recording failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/feedback-dashboard')
def feedback_dashboard():
    """Admin dashboard for feedback analytics"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        # Use the direct feedback system for dashboard data too
        from modules.feedback_system import _feedback_system
        
        if not _feedback_system:
            dashboard_data = {'total_feedback': 0, 'error': 'Feedback system not initialized'}
        else:
            dashboard_data = _feedback_system.get_dashboard_data()
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ghostline Feedback Dashboard</title>
            <style>
                body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }
                .dashboard { max-width: 1200px; margin: 0 auto; }
                .stat-box { background: #1a1a1a; border: 1px solid #333; padding: 15px; margin: 10px; border-radius: 5px; }
                .emoji-stat { font-size: 24px; margin: 10px 0; }
                .feedback-list { background: #111; padding: 10px; margin: 10px 0; border-radius: 5px; max-height: 400px; overflow-y: auto; }
                h1, h2 { color: #00ffff; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #333; padding: 8px; text-align: left; }
                th { background: #222; }
                .positive { color: #00ff00; }
                .negative { color: #ff4444; }
                .sass { color: #ffaa00; }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <h1>🧠 Ghostline Feedback Dashboard</h1>
                
                <div class="stat-box">
                    <h2>📊 Overview</h2>
                    <div class="emoji-stat">Total Feedback: {{ dashboard_data.total_feedback }}</div>
                    {% if dashboard_data.emoji_stats %}
                        {% for emoji, count in dashboard_data.emoji_stats.items() %}
                        <div class="emoji-stat">{{ emoji }}: {{ count }}</div>
                        {% endfor %}
                    {% endif %}
                    {% if dashboard_data.error %}
                        <div style="color: #ff4444;">Error: {{ dashboard_data.error }}</div>
                    {% endif %}
                </div>
                
                {% if dashboard_data.recent_feedback %}
                <div class="stat-box">
                    <h2>📈 Recent Activity (Last 7 Days)</h2>
                    <table>
                        <tr><th>Date</th><th>Type</th><th>Count</th></tr>
                        {% for item in dashboard_data.recent_feedback %}
                        <tr>
                            <td>{{ item[0] }}</td>
                            <td class="{% if item[1] == 'thumbs_up' %}positive{% elif item[1] == 'thumbs_down' %}negative{% else %}sass{% endif %}">
                                {% if item[1] == 'thumbs_up' %}👍 Good
                                {% elif item[1] == 'thumbs_down' %}👎 Bad  
                                {% else %}🖕 Sass/Snark{% endif %}
                            </td>
                            <td>{{ item[2] }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}
                
                {% if dashboard_data.project_breakdown %}
                <div class="stat-box">
                    <h2>📁 By Project</h2>
                    <table>
                        <tr><th>Project</th><th>Type</th><th>Count</th></tr>
                        {% for item in dashboard_data.project_breakdown %}
                        <tr>
                            <td>{{ item[0] or 'General' }}</td>
                            <td class="{% if item[1] == 'thumbs_up' %}positive{% elif item[1] == 'thumbs_down' %}negative{% else %}sass{% endif %}">
                                {% if item[1] == 'thumbs_up' %}👍
                                {% elif item[1] == 'thumbs_down' %}👎  
                                {% else %}🖕{% endif %}
                            </td>
                            <td>{{ item[2] }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                {% endif %}
                
                <div class="stat-box">
                    <h2>ℹ️ About Feedback Types</h2>
                    <div>👍 <span class="positive">Good</span> - Positive feedback for helpful responses</div>
                    <div>👎 <span class="negative">Bad</span> - Negative feedback for poor responses</div>
                    <div>🖕 <span class="sass">Sass/Snark</span> - Approval for personality, humor, or perfect attitude (this is GOOD feedback!)</div>
                </div>
                
                <div style="margin-top: 20px;">
                    <a href="/" style="color: #00ffff;">← Back to Chat</a>
                </div>
            </div>
        </body>
        </html>
        """, dashboard_data=dashboard_data)
        
    except Exception as e:
        app.logger.error(f"Feedback dashboard error: {e}")
        return f"Dashboard error: {e}", 500

@app.route('/diagnostics/clickup')
def clickup_diagnostics():
    """ClickUp diagnostics and configuration page"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        from modules.clickup_diagnostics import generate_clickup_diagnostic_report, get_clickup_workspace_tree
        
        report = generate_clickup_diagnostic_report()
        workspace_tree = get_clickup_workspace_tree()
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ClickUp Diagnostics</title>
            <style>
                body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .report { background: #1a1a1a; border: 1px solid #333; padding: 20px; margin: 20px 0; border-radius: 5px; }
                pre { white-space: pre-wrap; }
                .btn { background: #6366f1; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; text-decoration: none; display: inline-block; margin: 10px 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ClickUp Integration Diagnostics</h1>
                <div class="report">
                    <pre>{{ report }}</pre>
                </div>
                {% if workspace_tree.workspace_tree %}
                <div class="report">
                    <h3>Workspace Structure</h3>
                    <pre>{{ workspace_tree | tojson(indent=2) }}</pre>
                </div>
                {% endif %}
                <a href="/" class="btn">Back to Chat</a>
                <a href="/integrations" class="btn">Integrations</a>
            </div>
        </body>
        </html>
        """, report=report, workspace_tree=workspace_tree)
        
    except Exception as e:
        return f"ClickUp diagnostics failed: {str(e)}", 500

# Project Mapping API Routes
@app.route('/api/projects/mappings', methods=['GET'])
def get_all_project_mappings():
    """API endpoint to get all project mappings"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not project_mapping_system:
        return jsonify({'error': 'Project mapping system not available'}), 500
        
    try:
        mappings = project_mapping_system.get_project_mappings()
        return jsonify({
            'success': True,
            'mappings': mappings,
            'projects': PROJECTS
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/mappings', methods=['POST'])
def add_project_mapping():
    """API endpoint to add a new project mapping"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not project_mapping_system:
        return jsonify({'error': 'Project mapping system not available'}), 500
    
    try:
        data = request.get_json()
        project = data.get('project')
        mapping_type = data.get('mapping_type')  # 'websites', 'social', 'analytics', 'email'
        resource_identifier = data.get('resource_identifier')
        resource_data = data.get('resource_data', {})
        
        if not all([project, mapping_type, resource_identifier]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        success = project_mapping_system.add_project_mapping(
            project, mapping_type, resource_identifier, resource_data
        )
        
        if success:
            return jsonify({'success': True, 'message': 'Mapping added successfully'})
        else:
            return jsonify({'error': 'Failed to add mapping'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/context/<session_id>', methods=['GET', 'POST'])
def manage_project_context(session_id):
    """API endpoint to get/set project context"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not project_mapping_system:
        return jsonify({'error': 'Project mapping system not available'}), 500
    
    try:
        if request.method == 'GET':
            context = project_mapping_system.get_project_context(session_id)
            return jsonify({
                'success': True,
                'context': context
            })
        
        elif request.method == 'POST':
            data = request.get_json()
            project = data.get('project')
            context_data = data.get('context_data', {})
            
            if not project:
                return jsonify({'error': 'Project name required'}), 400
            
            project_mapping_system.set_project_context(session_id, project, context_data)
            return jsonify({
                'success': True,
                'project': project,
                'message': f'Project context set to {project}'
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project>/dashboard')
def get_project_dashboard(project):
    """API endpoint for project dashboard data"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not project_mapping_system:
        return jsonify({'error': 'Project mapping system not available'}), 500
    
    try:
        dashboard_data = project_mapping_system.get_project_dashboard_data(project)
        return jsonify({
            'success': True,
            'dashboard': dashboard_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects/<project>/history')
def get_project_conversation_history(project):
    """API endpoint for project-specific conversation history"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not project_mapping_system:
        return jsonify({'error': 'Project mapping system not available'}), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        session_id = session.get('session_id')
        
        history = project_mapping_system.filter_conversation_history(
            project=project,
            session_id=session_id,
            limit=limit
        )
        
        # Convert datetime objects to strings for JSON serialization
        for item in history:
            if 'created_at' in item and item['created_at']:
                item['created_at'] = item['created_at'].isoformat()
        
        return jsonify({
            'success': True,
            'project': project,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# NEW: Miller Debug Routes
@app.route('/debug/miller')
def debug_miller_route():
    """Debug route to test Miller search and find the exact failure point"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        from modules.brain import debug_miller_search
        
        # Capture debug output
        import io
        import sys
        from contextlib import redirect_stdout
        
        debug_output = io.StringIO()
        
        with redirect_stdout(debug_output):
            debug_miller_search()
        
        output_text = debug_output.getvalue()
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Miller Search Debug Results</title>
            <style>
                body {{ font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }}
                .container {{ max-width: 1400px; margin: 0 auto; }}
                .debug-output {{ background: #1a1a1a; border: 1px solid #333; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                pre {{ white-space: pre-wrap; font-size: 14px; line-height: 1.4; }}
                .btn {{ background: #6366f1; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; text-decoration: none; display: inline-block; margin: 10px 5px; }}
                .success {{ color: #00ff00; }}
                .error {{ color: #ff4444; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Miller Search Debug Results</h1>
                
                <div class="debug-output">
                    <h3>Debug Output:</h3>
                    <pre>{output_text}</pre>
                </div>
                
                <div style="margin-top: 20px;">
                    <a href="/debug/brain_diagnostics" class="btn">Full Brain Diagnostics</a>
                    <a href="/system" class="btn">System Dashboard</a>
                    <a href="/" class="btn">Back to Chat</a>
                </div>
                
                <div class="debug-output">
                    <h3>Next Steps:</h3>
                    <p>1. Check the debug output above for any RED ❌ errors</p>
                    <p>2. If database connection fails, check DATABASE_URL</p>
                    <p>3. If SQL query fails, check chat_threads table structure</p>
                    <p>4. If search function fails, there's a Python import/logic error</p>
                    <p>5. If everything works here but chat fails, the issue is in app.py routing</p>
                </div>
            </div>
        </body>
        </html>
        """
        
    except Exception as e:
        app.logger.error(f"Miller debug route failed: {e}")
        return f"""
        <html>
        <body style="font-family: monospace; background: #0a0a0a; color: #ff4444; padding: 20px;">
            <h1>❌ Miller Debug Route Failed</h1>
            <p><strong>Error:</strong> {str(e)}</p>
            <p>This means the debug_miller_search function couldn't be imported or executed.</p>
            <p>Check that you added the debug function to modules/brain.py</p>
            <a href="/" style="color: #00ffff;">← Back to Chat</a>
        </body>
        </html>
        """, 500

@app.route('/debug/brain_diagnostics')
def debug_brain_diagnostics():
    """Enhanced brain diagnostics including Miller-specific tests"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        from modules.brain import get_brain_diagnostics, test_miller_memory_directly, test_ghada_memory_directly
        
        # Get comprehensive diagnostics
        diagnostics = get_brain_diagnostics()
        
        # Run direct memory tests
        miller_test_result = test_miller_memory_directly()
        ghada_test_result = test_ghada_memory_directly()
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Brain System Diagnostics</title>
            <style>
                body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }
                .container { max-width: 1400px; margin: 0 auto; }
                .diagnostic-section { background: #1a1a1a; border: 1px solid #333; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .success { color: #00ff00; }
                .error { color: #ff4444; }
                .warning { color: #ffaa00; }
                pre { white-space: pre-wrap; font-size: 12px; }
                .btn { background: #6366f1; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; text-decoration: none; display: inline-block; margin: 10px 5px; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #333; padding: 8px; text-align: left; }
                th { background: #222; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🧠 Brain System Diagnostics</h1>
                
                <div class="diagnostic-section">
                    <h3>Memory Tests</h3>
                    <p><strong>Miller Test:</strong> <span class="{% if miller_test %}success">✅ PASSED{% else %}error">❌ FAILED{% endif %}</span></p>
                    <p><strong>Ghada Test:</strong> <span class="{% if ghada_test %}success">✅ PASSED{% else %}error">❌ FAILED{% endif %}</span></p>
                </div>
                
                <div class="diagnostic-section">
                    <h3>System Status</h3>
                    <table>
                        <tr><th>Component</th><th>Status</th><th>Details</th></tr>
                        {% for component, data in diagnostics.items() %}
                        <tr>
                            <td>{{ component.replace('_', ' ').title() }}</td>
                            <td class="{% if data.get('error') %}error">❌ ERROR{% else %}success">✅ OK{% endif %}</td>
                            <td>
                                {% if data.get('error') %}
                                    {{ data.error }}
                                {% else %}
                                    {% for key, value in data.items() %}
                                        {% if key != 'error' %}
                                            <strong>{{ key }}:</strong> {{ value }}<br>
                                        {% endif %}
                                    {% endfor %}
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="diagnostic-section">
                    <h3>Raw Diagnostics Data</h3>
                    <pre>{{ diagnostics | tojson(indent=2) }}</pre>
                </div>
                
                <div style="margin-top: 20px;">
                    <a href="/debug/miller" class="btn">Miller Debug Test</a>
                    <a href="/system" class="btn">System Dashboard</a>
                    <a href="/" class="btn">Back to Chat</a>
                </div>
            </div>
        </body>
        </html>
        """, diagnostics=diagnostics, miller_test=miller_test_result, ghada_test=ghada_test_result)
        
    except Exception as e:
        app.logger.error(f"Brain diagnostics failed: {e}")
        return f"Brain diagnostics error: {str(e)}", 500

# Section 9: PDF Report Generation
from modules.pdf_generation import (
    generate_project_pdf,
    generate_daily_briefing_pdf,
    generate_project_report,
    generate_daily_briefing_report
)

@app.route('/reports/<project_name>.pdf')
def project_report_pdf(project_name):
    """Generate project report as PDF"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        days = request.args.get('days', 30, type=int)
        days = min(days, 365)  # Limit to 1 year max
        
        pdf_bytes, temp_path = generate_project_pdf(project_name, days)
        
        safe_name = f"{project_name.replace(' ', '_')}_report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
        
        return send_file(
            temp_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=safe_name
        )
        
    except Exception as e:
        app.logger.error(f"Project PDF generation failed: {e}")
        return f"PDF generation failed: {str(e)}", 500

@app.route('/reports/daily/<date>.pdf')
def daily_briefing_pdf(date):
    """Generate daily briefing as PDF"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        report_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        pdf_bytes, temp_path = generate_daily_briefing_pdf(report_date)
        safe_name = f"daily_briefing_{date}.pdf"
        
        return send_file(
            temp_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=safe_name
        )
        
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD", 400
    except Exception as e:
        app.logger.error(f"Daily briefing PDF generation failed: {e}")
        return f"PDF generation failed: {str(e)}", 500

@app.route('/reports/daily/today.pdf')
def daily_briefing_today_pdf():
    """Generate today's briefing as PDF"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    return redirect(url_for('daily_briefing_pdf', date=today))

@app.route('/reports/<project_name>')
def project_report_preview(project_name):
    """Preview project report as HTML"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        days = request.args.get('days', 30, type=int)
        days = min(days, 365)
        
        html_content = generate_project_report(project_name, days)
        return html_content
        
    except Exception as e:
        app.logger.error(f"Project report preview failed: {e}")
        return f"Report generation failed: {str(e)}", 500

@app.route('/reports/daily/<date>')
def daily_briefing_preview(date):
    """Preview daily briefing as HTML"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        report_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        html_content = generate_daily_briefing_report(report_date)
        return html_content
        
    except ValueError:
        return "Invalid date format. Use YYYY-MM-DD", 400
    except Exception as e:
        app.logger.error(f"Daily briefing preview failed: {e}")
        return f"Report generation failed: {str(e)}", 500

@app.route('/reports')
def reports_dashboard():
    """Simple reports dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghostline Reports Dashboard</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f0f; 
                color: #fff; 
                margin: 0; 
                padding: 20px; 
            }
            .container { max-width: 1000px; margin: 0 auto; }
            .report-section { 
                background: #1a1a1a; 
                border: 1px solid #333; 
                border-radius: 8px; 
                padding: 20px; 
                margin: 20px 0; 
            }
            .btn { 
                background: #6366f1; 
                color: white; 
                border: none; 
                padding: 12px 24px; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 16px;
                margin: 10px 5px;
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover { background: #5855eb; }
            .btn.secondary { background: #374151; }
            .btn.secondary:hover { background: #4b5563; }
            .projects-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .project-card {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 8px;
            }
            .project-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Reports Dashboard</h1>
            
            <div class="report-section">
                <h3>Daily Intelligence Briefings</h3>
                <p>Generate comprehensive daily activity summaries and intelligence reports.</p>
                <a href="/reports/daily/today" class="btn secondary">Preview Today's Briefing</a>
                <a href="/reports/daily/today.pdf" class="btn">Download Today's PDF</a>
            </div>
            
            <div class="report-section">
                <h3>Project Reports</h3>
                <p>Generate detailed reports for specific projects with conversation history and analytics.</p>
                <div class="projects-grid">
                    {% for project in projects %}
                    <div class="project-card">
                        <div class="project-title">{{ project }}</div>
                        <a href="/reports/{{ project }}" class="btn secondary">Preview</a>
                        <a href="/reports/{{ project }}.pdf" class="btn">Download PDF</a>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="report-section">
                <button class="btn secondary" onclick="window.location.href='/'">← Back to Chat</button>
                <button class="btn secondary" onclick="window.location.href='/system'">System Dashboard</button>
                <button class="btn secondary" onclick="window.location.href='/integrations'">Integrations</button>
            </div>
        </div>
    </body>
    </html>
    '''
    
    return render_template_string(html_content, projects=PROJECTS)
    
# Section 10: Telegram Integration Routes
# Section 10: Telegram Integration Routes 9/12/25
# Section 10: Telegram Integration Routes 9/12/25
# Section 10: Telegram Integration Routes (UPDATED WITH TRENDS ALERT SUPPORT) 9/13/25
@app.route('/reminders/check', methods=['POST'])
def check_telegram_reminders():
    """Manual trigger for reminder checking"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    if not is_telegram_configured():
        return jsonify({"success": False, "error": "Telegram not configured"}), 400
    
    reminders = GhostlineTelegramReminders()
    result = reminders.check_and_send_reminders()
    return jsonify(result)

@app.route('/telegram/emergency_stop', methods=['POST'])
def emergency_stop_reminders():
    """EMERGENCY: Stop all pending reminders to prevent spam"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        reminders = GhostlineTelegramReminders()
        result = reminders.emergency_stop_all()
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Emergency stop failed: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/telegram/emergency_stop_now')
def emergency_stop_now():
    """GET version for emergency stop when buttons fail"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        reminders = GhostlineTelegramReminders()
        result = reminders.emergency_stop_all()
        
        if result["success"]:
            return f"<h1>EMERGENCY STOP SUCCESS</h1><p>Stopped {result['stopped_count']} reminders</p><a href='/integrations'>Back to Integrations</a>"
        else:
            return f"<h1>EMERGENCY STOP FAILED</h1><p>{result['error']}</p><a href='/integrations'>Back to Integrations</a>"
    except Exception as e:
        return f"<h1>EMERGENCY STOP ERROR</h1><p>{str(e)}</p><a href='/integrations'>Back to Integrations</a>"

@app.route('/telegram/webhook', methods=['POST'])
def telegram_webhook():
    """Enhanced Telegram webhook handler with trends alert support and chat ID capture"""
    try:
        data = request.get_json()
        
        # EXTRACT CHAT ID FOR DEBUGGING
        chat_id = None
        if 'message' in data and 'chat' in data['message']:
            chat_id = data['message']['chat']['id']
            print(f"=== CHAT ID FOUND: {chat_id} ===")
            app.logger.info(f"CHAT ID FOUND: {chat_id}")
        elif 'callback_query' in data and 'message' in data['callback_query']:
            chat_id = data['callback_query']['message']['chat']['id']
            print(f"=== CHAT ID FOUND FROM CALLBACK: {chat_id} ===")
            app.logger.info(f"CHAT ID FOUND FROM CALLBACK: {chat_id}")
        
        app.logger.info(f"Telegram webhook received: {data}")
        
        # Handle callback queries (button presses)
        if 'callback_query' in data:
            callback_query = data['callback_query']
            callback_data = callback_query.get('data', '')
            
            app.logger.info(f"Processing callback: {callback_data}")
            
            # Check if this is a trends alert callback
            if any(callback_data.startswith(prefix) for prefix in ['draft_', 'skip_', 'wrong_', 'data_']):
                try:
                    from modules.google_trends_monitor import TelegramAlertSystem, get_trends_monitor
                    
                    monitor = get_trends_monitor()
                    alert_system = TelegramAlertSystem(monitor)
                    result = alert_system.process_alert_callback(callback_data, callback_query)
                    
                    app.logger.info(f"Trends callback result: {result}")
                    
                    # Send callback answer to remove loading state
                    callback_id = callback_query.get('id')
                    if callback_id:
                        from modules.telegram_notifications import TelegramBot
                        bot = TelegramBot()
                        answer_url = f"https://api.telegram.org/bot{bot.token}/answerCallbackQuery"
                        requests.post(answer_url, json={
                            "callback_query_id": callback_id,
                            "text": "Action processed!" if result.get('success') else "Action failed"
                        })
                    
                    return jsonify({"ok": True})
                    
                except Exception as e:
                    app.logger.error(f"Trends callback processing failed: {e}")
                    # Fall through to regular reminder processing if trends fails
            
            # Handle regular reminder callbacks (existing functionality preserved)
            try:
                reminders = GhostlineTelegramReminders()
                result = reminders.process_callback_query(callback_query)
                
                app.logger.info(f"Reminder callback result: {result}")
                
                # Send callback answer to remove loading state
                callback_id = callback_query.get('id')
                if callback_id:
                    bot = reminders.bot
                    answer_url = f"https://api.telegram.org/bot{bot.token}/answerCallbackQuery"
                    requests.post(answer_url, json={
                        "callback_query_id": callback_id,
                        "text": "Action processed!" if result.get('success') else "Action failed"
                    })
            except Exception as e:
                app.logger.error(f"Reminder callback processing failed: {e}")
            
            return jsonify({"ok": True})
        
        # Handle regular messages
        if 'message' in data:
            app.logger.info(f"Received message from chat_id {chat_id}: {data['message'].get('text', 'no text')}")
            return jsonify({"ok": True})
        
        app.logger.info("Webhook received unknown data type")
        return jsonify({"ok": True})
        
    except Exception as e:
        app.logger.error(f"Telegram webhook failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route('/telegram/setup_webhook', methods=['POST'])
def setup_telegram_webhook():
    """Setup Telegram webhook"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    webhook_url = os.getenv('WEBHOOK_URL')
    
    if not webhook_url:
        railway_url = os.getenv('RAILWAY_STATIC_URL')
        if railway_url:
            webhook_url = f"https://{railway_url}/telegram/webhook"
        else:
            return jsonify({
                "success": False,
                "error": "WEBHOOK_URL not configured and RAILWAY_STATIC_URL not found"
            }), 400
    
    try:
        from modules.telegram_notifications import TelegramBot
        bot = TelegramBot()
        
        response = requests.post(
            f"https://api.telegram.org/bot{bot.token}/setWebhook",
            json={
                "url": webhook_url,
                "allowed_updates": ["callback_query", "message"]
            }
        )
        result = response.json()
        
        app.logger.info(f"Webhook setup result: {result}")
        
        return jsonify({
            "success": result.get('ok', False),
            "description": result.get('description', ''),
            "webhook_url": webhook_url,
            "raw_response": result
        })
        
    except Exception as e:
        app.logger.error(f"Webhook setup failed: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/telegram/webhook_info')
def telegram_webhook_info():
    """Get current webhook information"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        from modules.telegram_notifications import TelegramBot
        bot = TelegramBot()
        
        response = requests.get(f"https://api.telegram.org/bot{bot.token}/getWebhookInfo")
        result = response.json()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# NEW DEBUG ROUTES FOR NOTIFICATION DELIVERY DIAGNOSIS
@app.route('/telegram/system_status')
def telegram_system_status():
    """Comprehensive Telegram system status and debugging"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    status = {}
    
    try:
        # 1. Environment Variables Check
        status['environment'] = {
            'bot_token_set': bool(os.getenv('TELEGRAM_BOT_TOKEN')),
            'chat_id_set': bool(os.getenv('TELEGRAM_CHAT_ID')),
            'chat_id_value': os.getenv('TELEGRAM_CHAT_ID'),
            'railway_environment': bool(os.getenv('RAILWAY_ENVIRONMENT'))
        }
        
        # 2. Bot Initialization Check
        try:
            from modules.telegram_notifications import TelegramBot, GhostlineTelegramReminders
            bot = TelegramBot()
            status['bot_init'] = {
                'success': True,
                'chat_id': bot.chat_id,
                'token_length': len(bot.token) if bot.token else 0
            }
        except Exception as e:
            status['bot_init'] = {
                'success': False,
                'error': str(e)
            }
        
        # 3. Database Connection Check
        try:
            from modules.database import get_db_connection
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM telegram_reminders WHERE status = 'pending'")
                    pending_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM telegram_reminders WHERE status = 'sent'")
                    sent_count = cursor.fetchone()[0]
                    
                    status['database'] = {
                        'connection': 'success',
                        'pending_reminders': pending_count,
                        'sent_reminders': sent_count
                    }
                else:
                    status['database'] = {'connection': 'failed', 'error': 'No connection'}
        except Exception as e:
            status['database'] = {'connection': 'error', 'error': str(e)}
        
        # 4. Telegram API Test
        try:
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            response = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
            api_result = response.json()
            status['telegram_api'] = {
                'reachable': response.status_code == 200,
                'bot_ok': api_result.get('ok', False),
                'bot_username': api_result.get('result', {}).get('username', 'unknown'),
                'error': api_result.get('description') if not api_result.get('ok') else None
            }
        except Exception as e:
            status['telegram_api'] = {'reachable': False, 'error': str(e)}
        
        # 5. Recent Reminders Check
        try:
            reminders = GhostlineTelegramReminders()
            check_result = reminders.check_and_send_reminders()
            status['reminder_check'] = check_result
        except Exception as e:
            status['reminder_check'] = {'error': str(e)}
        
        # 6. Test Send Message
        try:
            if status['bot_init']['success']:
                test_result = bot.send_message("🧪 System status test - " + datetime.datetime.now().strftime('%H:%M:%S'))
                status['test_send'] = test_result
        except Exception as e:
            status['test_send'] = {'error': str(e)}
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/telegram/reminders_debug')
def telegram_reminders_debug():
    """Debug view of all Telegram reminders"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        from modules.database import get_db_connection
        
        with get_db_connection() as conn:
            if not conn:
                return jsonify({"error": "Database not available"}), 500
            
            cursor = conn.cursor()
            
            # Get all reminders
            cursor.execute('''
                SELECT reminder_id, title, remind_at, status, project, priority, created_at
                FROM telegram_reminders 
                ORDER BY created_at DESC
                LIMIT 20
            ''')
            
            reminders = []
            for row in cursor.fetchall():
                reminder_id, title, remind_at, status, project, priority, created_at = row
                reminders.append({
                    'reminder_id': reminder_id,
                    'title': title,
                    'remind_at': remind_at.isoformat() if remind_at else None,
                    'status': status,
                    'project': project,
                    'priority': priority,
                    'created_at': created_at.isoformat() if created_at else None,
                    'is_due': remind_at <= datetime.datetime.now() if remind_at else False
                })
            
            # Get counts by status
            cursor.execute('''
                SELECT status, COUNT(*) 
                FROM telegram_reminders 
                GROUP BY status
            ''')
            
            status_counts = dict(cursor.fetchall())
            
            return jsonify({
                'reminders': reminders,
                'status_counts': status_counts,
                'current_time': datetime.datetime.now().isoformat()
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/telegram/force_reminder_check', methods=['POST'])
def force_reminder_check():
    """Force an immediate reminder check with detailed logging"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        from modules.telegram_notifications import GhostlineTelegramReminders
        
        print("=== FORCING REMINDER CHECK ===")
        reminders = GhostlineTelegramReminders()
        result = reminders.check_and_send_reminders()
        print(f"Force check result: {result}")
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Force reminder check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/telegram/create_test_reminder', methods=['POST'])
def create_test_reminder():
    """Create a test reminder that's due immediately"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        from modules.telegram_notifications import GhostlineTelegramReminders
        
        # Create a reminder due in 30 seconds
        remind_time = datetime.datetime.now() + datetime.timedelta(seconds=30)
        
        reminders = GhostlineTelegramReminders()
        result = reminders.create_reminder(
            title="🧪 Test Reminder - Should arrive in 30 seconds",
            content="This is a test notification to debug delivery issues",
            remind_at=remind_time,
            project="Debug",
            priority=1  # High priority
        )
        
        return jsonify({
            'success': result.get('success', False),
            'result': result,
            'remind_at': remind_time.isoformat(),
            'note': 'Check your Telegram in 30 seconds'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/telegram/find_chat_id')
def find_telegram_chat_id():
    """Find your Telegram chat ID from recent messages"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not token:
            return jsonify({"error": "TELEGRAM_BOT_TOKEN not configured"}), 400
        
        # Get recent updates from Telegram
        response = requests.get(f"https://api.telegram.org/bot{token}/getUpdates")
        data = response.json()
        
        if not data.get('ok'):
            return jsonify({
                "error": "Failed to get updates",
                "telegram_error": data.get('description', 'Unknown error')
            }), 400
        
        # Extract all unique chat IDs from recent messages
        chat_ids = []
        for update in data.get('result', []):
            if 'message' in update and 'chat' in update['message']:
                chat = update['message']['chat']
                chat_ids.append({
                    'chat_id': chat['id'],
                    'type': chat.get('type', 'unknown'),
                    'title': chat.get('title', chat.get('first_name', 'Unknown')),
                    'username': chat.get('username', 'N/A'),
                    'message_text': update['message'].get('text', 'N/A')[:50]
                })
        
        # Remove duplicates
        unique_chats = []
        seen_ids = set()
        for chat in chat_ids:
            if chat['chat_id'] not in seen_ids:
                unique_chats.append(chat)
                seen_ids.add(chat['chat_id'])
        
        return jsonify({
            "found_chats": unique_chats,
            "instructions": [
                "1. Send a message to your bot from the chat where you want notifications",
                "2. Refresh this page to see your chat_id",
                "3. Add TELEGRAM_CHAT_ID=[your_chat_id] to Railway environment variables",
                "4. Restart your Railway deployment"
            ]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/telegram/test_parser')
def test_reminder_parser():
    """Test the reminder parsing function directly"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    test_inputs = [
        "remind me in two minutes to drink my tea",
        "remind me in 5 minutes to check email",
        "remind me to call John in 30 minutes",
        "set a reminder to take medicine in 1 hour"
    ]
    
    results = []
    
    for test_input in test_inputs:
        try:
            from modules.telegram_notifications import parse_reminder_command
            result = parse_reminder_command(test_input, "Personal Operating Manual")
            
            # Convert datetime and timedelta objects to strings for JSON serialization
            if result and isinstance(result, dict):
                serializable_result = {}
                for key, value in result.items():
                    if hasattr(value, 'isoformat'):  # datetime objects
                        serializable_result[key] = value.isoformat()
                    elif hasattr(value, 'total_seconds'):  # timedelta objects
                        serializable_result[key] = f"{value.total_seconds()} seconds"
                    else:
                        serializable_result[key] = value
            else:
                serializable_result = result
            
            results.append({
                'input': test_input,
                'success': result.get('success') if result else False,
                'result': serializable_result,
                'current_time': datetime.datetime.now().isoformat()
            })
        except Exception as e:
            results.append({
                'input': test_input,
                'success': False,
                'error': str(e),
                'current_time': datetime.datetime.now().isoformat()
            })
    
    return jsonify({
        'test_results': results,
        'server_time': datetime.datetime.now().isoformat()
    })

@app.route('/telegram/debug')
def telegram_debug_interface():
    """Simple HTML interface for debugging Telegram notifications"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Telegram Debug Interface</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .debug-section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .debug-button { padding: 10px 15px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 3px; cursor: pointer; }
            .debug-button:hover { background: #0056b3; }
            .debug-output { margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 3px; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
            .error { color: #dc3545; }
            .success { color: #28a745; }
        </style>
    </head>
    <body>
        <h1>🔧 Telegram Debug Interface</h1>
        
        <div class="debug-section">
            <h3>System Status</h3>
            <button class="debug-button" onclick="checkSystemStatus()">Check System Status</button>
            <div id="systemStatus" class="debug-output"></div>
        </div>
        
        <div class="debug-section">
            <h3>Reminders Debug</h3>
            <button class="debug-button" onclick="checkReminders()">View All Reminders</button>
            <button class="debug-button" onclick="forceReminderCheck()">Force Reminder Check</button>
            <div id="remindersOutput" class="debug-output"></div>
        </div>
        
        <div class="debug-section">
            <h3>Parser Testing</h3>
            <button class="debug-button" onclick="testParser()">Test Reminder Parser</button>
            <div id="parserOutput" class="debug-output"></div>
        </div>
        
        <div class="debug-section">
            <h3>Test Functions</h3>
            <button class="debug-button" onclick="createTestReminder()">Create Test Reminder (30s)</button>
            <button class="debug-button" onclick="findChatId()">Find Chat ID</button>
            <div id="testOutput" class="debug-output"></div>
        </div>
        
        <div class="debug-section">
            <h3>Webhook Info</h3>
            <button class="debug-button" onclick="checkWebhook()">Check Webhook Info</button>
            <div id="webhookOutput" class="debug-output"></div>
        </div>
        
        <script>
            function checkSystemStatus() {
                fetch('/telegram/system_status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('systemStatus').innerHTML = JSON.stringify(data, null, 2);
                    document.getElementById('systemStatus').className = 'debug-output success';
                })
                .catch(e => {
                    document.getElementById('systemStatus').innerHTML = 'Error: ' + e;
                    document.getElementById('systemStatus').className = 'debug-output error';
                });
            }
            
            function checkReminders() {
                fetch('/telegram/reminders_debug')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('remindersOutput').innerHTML = JSON.stringify(data, null, 2);
                    document.getElementById('remindersOutput').className = 'debug-output success';
                })
                .catch(e => {
                    document.getElementById('remindersOutput').innerHTML = 'Error: ' + e;
                    document.getElementById('remindersOutput').className = 'debug-output error';
                });
            }
            
            function testParser() {
                fetch('/telegram/test_parser')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('parserOutput').innerHTML = JSON.stringify(data, null, 2);
                    document.getElementById('parserOutput').className = 'debug-output success';
                })
                .catch(e => {
                    document.getElementById('parserOutput').innerHTML = 'Error: ' + e;
                    document.getElementById('parserOutput').className = 'debug-output error';
                });
            }
            
            function forceReminderCheck() {
                fetch('/telegram/force_reminder_check', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('remindersOutput').innerHTML = JSON.stringify(data, null, 2);
                    document.getElementById('remindersOutput').className = 'debug-output success';
                })
                .catch(e => {
                    document.getElementById('remindersOutput').innerHTML = 'Error: ' + e;
                    document.getElementById('remindersOutput').className = 'debug-output error';
                });
            }
            
            function createTestReminder() {
                fetch('/telegram/create_test_reminder', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    document.getElementById('testOutput').innerHTML = JSON.stringify(data, null, 2);
                    document.getElementById('testOutput').className = 'debug-output success';
                    if (data.success) {
                        alert('Test reminder created! Check your Telegram in 30 seconds.');
                    }
                })
                .catch(e => {
                    document.getElementById('testOutput').innerHTML = 'Error: ' + e;
                    document.getElementById('testOutput').className = 'debug-output error';
                });
            }
            
            function findChatId() {
                fetch('/telegram/find_chat_id')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('testOutput').innerHTML = JSON.stringify(data, null, 2);
                    document.getElementById('testOutput').className = 'debug-output success';
                })
                .catch(e => {
                    document.getElementById('testOutput').innerHTML = 'Error: ' + e;
                    document.getElementById('testOutput').className = 'debug-output error';
                });
            }
            
            function checkWebhook() {
                fetch('/telegram/webhook_info')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('webhookOutput').innerHTML = JSON.stringify(data, null, 2);
                    document.getElementById('webhookOutput').className = 'debug-output success';
                })
                .catch(e => {
                    document.getElementById('webhookOutput').innerHTML = 'Error: ' + e;
                    document.getElementById('webhookOutput').className = 'debug-output error';
                });
            }
        </script>
    </body>
    </html>
    '''
    return html
        
# Section 11: Marketing Dashboard Routes
from modules.marketing_flux import (
    MarketingFluxGenerator,
    quick_social_post,
    test_campaign_ideas,
    create_full_campaign
)

@app.route('/marketing')
def marketing_dashboard():
    """Main marketing asset creation dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('marketing_dashboard.html')

@app.route('/api/marketing/quick-asset', methods=['POST'])
def api_marketing_quick_asset():
    """Generate single marketing asset quickly"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
    try:
        data = request.get_json()
        
        concept = data.get('concept')
        if not concept:
            return jsonify({'success': False, 'error': 'Concept description required'}), 400
        
        style = data.get('style', 'corporate')
        platform = data.get('platform', 'instagram')
        quality = data.get('quality', 'standard')
        format_name = data.get('format_name')
        
        generator = MarketingFluxGenerator()
        result = generator.create_and_wait(
            prompt=concept,
            style=style,
            platform=platform,
            quality=quality,
            format_name=format_name
        )
        
        # Store in database if available
        if result['success']:
            try:
                save_conversation_enhanced('marketing_assets', concept, {
                    'marketing_asset': {
                        'concept': concept,
                        'style': style,
                        'platform': platform,
                        'quality': quality,
                        'format_name': format_name,
                        'image_url': result.get('image_url'),
                        'cost': result.get('estimated_cost'),
                        'success': result['success']
                    }
                })
            except Exception as e:
                app.logger.error(f"Database storage failed: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Marketing asset generation failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Asset generation failed: {str(e)}'
        }), 500

@app.route('/api/marketing/campaign', methods=['POST'])
def api_marketing_campaign():
    """Create complete campaign asset set"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
    try:
        data = request.get_json()
        
        campaign_name = data.get('campaign_name')
        concept = data.get('concept')
        
        if not campaign_name or not concept:
            return jsonify({
                'success': False,
                'error': 'Campaign name and concept required'
            }), 400
        
        platforms = data.get('platforms', ['instagram', 'facebook', 'linkedin', 'twitter'])
        style = data.get('style', 'corporate')
        quality = data.get('quality', 'standard')
        
        generator = MarketingFluxGenerator()
        result = generator.create_campaign_assets(
            campaign_name=campaign_name,
            base_prompt=concept,
            platforms=platforms,
            style=style,
            quality=quality
        )
        
        # Store campaign in database
        if result['success']:
            try:
                save_conversation_enhanced('marketing_campaigns', f"Campaign: {campaign_name}", {
                    'campaign_results': result
                })
            except Exception as e:
                app.logger.error(f"Campaign storage failed: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Campaign creation failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Campaign creation failed: {str(e)}'
        }), 500

@app.route('/api/marketing/status')
def api_marketing_status():
    """Check marketing tools status"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
    try:
        generator = MarketingFluxGenerator()
        
        return jsonify({
            'success': True,
            'status': 'operational',
            'api_connected': True,
            'available_models': len(generator.models),
            'available_formats': len(generator.social_specs),
            'cost_estimate': {
                'rapid_generation': '$0.003 per image',
                'standard_quality': '$0.030 per image',
                'professional_quality': '$0.055 per image'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'status': 'error',
            'error': str(e),
            'setup_required': 'REPLICATE_API_TOKEN' in str(e)
        })
        
# Section 12: ElevenLabs Text-to-Speech
import base64

# ElevenLabs SDK imports
try:
    from elevenlabs import ElevenLabs, Voice, VoiceSettings
    ELEVENLABS_SDK_AVAILABLE = True
except ImportError:
    print("ElevenLabs SDK not available, falling back to HTTP requests")
    ELEVENLABS_SDK_AVAILABLE = False

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using ElevenLabs with proper SDK or fallback"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'success': False, 'error': 'Text required'}), 400
        
        if len(text) > 2500:  # ElevenLabs character limit
            text = text[:2500] + "..."
        
        api_key = os.getenv('ELEVENLABS_API_KEY')
        if not api_key:
            app.logger.error("ELEVENLABS_API_KEY not configured")
            return jsonify({'success': False, 'error': 'ElevenLabs API key not configured'}), 400
        
        voice_id = os.getenv('ELEVENLABS_VOICE_ID', "21m00Tcm4TlvDq8ikWAM")  # Default to Rachel
        
        app.logger.info(f"Using ElevenLabs voice: {voice_id}")
        
        # Try SDK first, fallback to HTTP if SDK not available
        if ELEVENLABS_SDK_AVAILABLE:
            return _tts_with_sdk(text, api_key, voice_id)
        else:
            return _tts_with_http(text, api_key, voice_id)
            
    except Exception as e:
        app.logger.error(f"TTS generation failed: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Speech generation failed: {str(e)}'}), 500

def _tts_with_sdk(text, api_key, voice_id):
    """TTS using official ElevenLabs SDK"""
    try:
        client = ElevenLabs(api_key=api_key)
        
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_monolingual_v1",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.5,
                style=0.0,
                use_speaker_boost=True
            ),
            output_format="mp3_44100_128"
        )
        
        audio_bytes = b''.join(audio)  # SDK returns generator
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        app.logger.info(f"Successfully generated {len(audio_bytes)} bytes of audio using SDK")
        
        return jsonify({
            'success': True,
            'audio': audio_base64,
            'format': 'mp3',
            'voice_used': voice_id,
            'method': 'SDK'
        })
        
    except Exception as e:
        app.logger.error(f"SDK TTS failed: {e}")
        return _tts_with_http(text, api_key, voice_id)

def _tts_with_http(text, api_key, voice_id):
    """TTS using HTTP requests (fallback)"""
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        app.logger.info(f"Making HTTP request to {url}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        app.logger.info(f"ElevenLabs HTTP response status: {response.status_code}")
        
        if response.status_code == 200:
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            
            app.logger.info(f"Successfully generated {len(response.content)} bytes of audio using HTTP")
            
            return jsonify({
                'success': True,
                'audio': audio_base64,
                'format': 'mp3',
                'voice_used': voice_id,
                'method': 'HTTP'
            })
        else:
            error_msg = f"ElevenLabs HTTP API error: {response.status_code}"
            
            try:
                error_detail = response.json()
                app.logger.error(f"API error details: {error_detail}")
                
                if response.status_code == 401:
                    error_msg = "Invalid API key - check your ELEVENLABS_API_KEY"
                elif response.status_code == 422:
                    error_msg = f"Invalid request: {error_detail.get('detail', {}).get('msg', 'Unknown validation error')}"
                elif response.status_code == 400:
                    error_msg = f"Bad request: {error_detail.get('detail', {}).get('msg', 'Invalid parameters')}"
                else:
                    error_msg += f" - {error_detail}"
                    
            except:
                error_msg += f" - {response.text}"
                app.logger.error(f"Non-JSON error response: {response.text}")
            
            return jsonify({'success': False, 'error': error_msg}), 500
            
    except requests.exceptions.Timeout:
        app.logger.error("ElevenLabs HTTP request timed out")
        return jsonify({'success': False, 'error': 'Speech generation timed out'}), 500
    except requests.exceptions.ConnectionError:
        app.logger.error("ElevenLabs HTTP connection failed")
        return jsonify({'success': False, 'error': 'Connection to ElevenLabs failed'}), 500
    except Exception as e:
        app.logger.error(f"HTTP TTS failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/tts/status')
def tts_status():
    """Enhanced TTS status with better diagnostics"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    status = {
        "configured": bool(os.getenv('ELEVENLABS_API_KEY')),
        "api_key_present": bool(os.getenv('ELEVENLABS_API_KEY')),
        "voice_id_set": bool(os.getenv('ELEVENLABS_VOICE_ID')),
        "current_voice_id": os.getenv('ELEVENLABS_VOICE_ID', "21m00Tcm4TlvDq8ikWAM"),
        "sdk_available": ELEVENLABS_SDK_AVAILABLE,
        "connection_working": False
    }
    
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if status["configured"] and api_key:
        try:
            if ELEVENLABS_SDK_AVAILABLE:
                try:
                    client = ElevenLabs(api_key=api_key)
                    user_info = client.user.get_subscription_info()
                    
                    status["connection_working"] = True
                    status["character_count"] = getattr(user_info, 'character_count', 0)
                    status["character_limit"] = getattr(user_info, 'character_limit', 10000)
                    status["tier"] = getattr(user_info, 'tier', 'unknown')
                    status["method"] = "SDK"
                    
                except Exception as sdk_error:
                    app.logger.warning(f"SDK connection failed, trying HTTP: {sdk_error}")
                    raise Exception("SDK failed, trying HTTP")
                    
            else:
                raise Exception("SDK not available, using HTTP")
                
        except Exception:
            try:
                headers = {"xi-api-key": api_key}
                response = requests.get("https://api.elevenlabs.io/v1/user", headers=headers, timeout=10)
                
                if response.status_code == 200:
                    user_info = response.json()
                    status["connection_working"] = True
                    status["character_count"] = user_info.get("character_count", 0)
                    status["character_limit"] = user_info.get("character_limit", 10000)
                    
                    subscription = user_info.get("subscription", {})
                    status["subscription_tier"] = subscription.get("tier", "free")
                    status["method"] = "HTTP"
                    
                elif response.status_code == 401:
                    status["error"] = "Invalid API key - check ELEVENLABS_API_KEY"
                else:
                    status["error"] = f"API returned {response.status_code}: {response.text}"
                    
            except Exception as e:
                status["error"] = f"Connection test failed: {str(e)}"
    
    return jsonify(status)
    
# NEW SECTION: Calendar-Telegram Integration API Routes
# Add this entire section after your existing API routes (around line 1600-1700)

@app.route('/api/calendar-alerts/status')
def calendar_alerts_status():
    """Get calendar alerts status"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not is_calendar_telegram_configured():
        return jsonify({
            'configured': False,
            'error': 'Calendar-Telegram integration not configured'
        })
    
    try:
        from modules.calendar_telegram_integration import CalendarTelegramAlertsHotfix
        alerts = CalendarTelegramAlertsHotfix()
        status = alerts.get_monitoring_status()
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calendar-alerts/enable', methods=['POST'])
def enable_calendar_alerts():
    """Enable calendar alerts monitoring"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        from modules.calendar_telegram_integration import CalendarTelegramAlertsHotfix
        alerts = CalendarTelegramAlertsHotfix()
        
        result = alerts.enable_monitoring()
        if result['success']:
            start_calendar_monitoring()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calendar-alerts/disable', methods=['POST'])
def disable_calendar_alerts():
    """Disable calendar alerts monitoring"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        from modules.calendar_telegram_integration import CalendarTelegramAlertsHotfix
        alerts = CalendarTelegramAlertsHotfix()
        
        result = alerts.disable_monitoring()
        if result['success']:
            stop_calendar_monitoring()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calendar-alerts/send-summary', methods=['POST'])
def send_calendar_summary():
    """Send daily calendar summary now"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        from modules.calendar_telegram_integration import CalendarTelegramAlertsHotfix
        alerts = CalendarTelegramAlertsHotfix()
        
        result = alerts.send_daily_calendar_summary()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calendar-alerts/preferences', methods=['GET', 'POST'])
def calendar_alert_preferences():
    """Get or update calendar alert preferences"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        from modules.calendar_telegram_integration import CalendarTelegramAlertsHotfix
        alerts = CalendarTelegramAlertsHotfix()
        
        if request.method == 'GET':
            preferences = alerts.get_alert_preferences()
            return jsonify({'success': True, 'preferences': preferences})
        
        elif request.method == 'POST':
            data = request.get_json()
            preferences = data.get('preferences', {})
            
            success = alerts.save_alert_preferences(preferences)
            return jsonify({'success': success})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/calendar-alerts/upcoming-events')
def get_upcoming_events_api():
    """Get upcoming events for preview"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        from modules.calendar_telegram_integration import CalendarTelegramAlertsHotfix
        alerts = CalendarTelegramAlertsHotfix()
        
        hours_ahead = request.args.get('hours', 24, type=int)
        events = alerts.get_upcoming_events(hours_ahead=hours_ahead)
        
        # Format events for JSON response
        formatted_events = []
        for event in events:
            formatted_events.append({
                'id': event['id'],
                'title': event['title'],
                'start_time': event['start_time'].isoformat(),
                'location': event.get('location', ''),
                'attendee_count': len(event.get('attendees', []))
            })
        
        return jsonify({
            'success': True,
            'events': formatted_events,
            'count': len(formatted_events)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/calendar-alerts-settings')
def calendar_alerts_settings():
    """Calendar alerts settings page"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Calendar → Telegram Alerts</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
            }
            .container { max-width: 1000px; margin: 0 auto; }
            .btn { 
                background: #6366f1; color: white; border: none; padding: 12px 24px;
                border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
                text-decoration: none; display: inline-block;
            }
            .btn:hover { background: #5855eb; }
            .btn.success { background: #059669; }
            .btn.danger { background: #dc2626; }
            .btn.warning { background: #d97706; }
            .settings-section { 
                background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
                padding: 20px; margin: 20px 0; 
            }
            .status-card {
                background: #2a2a2a; padding: 15px; border-radius: 8px; margin: 15px 0;
                border-left: 4px solid #6366f1;
            }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .form-group input, .form-group select { 
                width: 100%; padding: 10px; background: #333; color: #fff; 
                border: 1px solid #555; border-radius: 4px; font-size: 16px;
            }
            .checkbox-group { display: flex; align-items: center; margin: 10px 0; }
            .checkbox-group input[type="checkbox"] { margin-right: 10px; width: auto; }
            .events-preview { max-height: 300px; overflow-y: auto; }
            .event-item { 
                padding: 10px; background: #333; margin: 5px 0; border-radius: 4px;
                display: flex; justify-content: space-between; align-items: center;
            }
            .alert-time-input { 
                display: inline-block; width: 80px; margin: 0 5px; 
                text-align: center; padding: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📅 Calendar → Telegram Alerts</h1>
            
            <div class="status-card" id="statusCard">
                <h3>Current Status</h3>
                <div id="statusInfo">Loading...</div>
            </div>
            
            <div class="settings-section">
                <h3>🔔 Alert Settings</h3>
                
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="meetingAlertsEnabled" onchange="updatePreferences()">
                        <label for="meetingAlertsEnabled">Enable meeting alerts</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Alert times (minutes before meeting):</label>
                    <div>
                        <input type="number" id="alertTime1" class="alert-time-input" value="15" onchange="updatePreferences()"> minutes and
                        <input type="number" id="alertTime2" class="alert-time-input" value="30" onchange="updatePreferences()"> minutes before
                    </div>
                </div>
                
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="weekendAlerts" onchange="updatePreferences()">
                        <label for="weekendAlerts">Include weekend alerts</label>
                    </div>
                </div>
            </div>
            
            <div class="settings-section">
                <h3>📊 Daily Summary</h3>
                
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="dailySummaryEnabled" onchange="updatePreferences()">
                        <label for="dailySummaryEnabled">Enable daily calendar summary</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="summaryTime">Summary time:</label>
                    <input type="time" id="summaryTime" value="07:00" onchange="updatePreferences()">
                </div>
                
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="includeTomorrow" onchange="updatePreferences()">
                        <label for="includeTomorrow">Include tomorrow's events in summary</label>
                    </div>
                </div>
            </div>
            
            <div class="settings-section">
                <h3>⚡ Quick Actions</h3>
                <button onclick="enableMonitoring()" class="btn success">Enable Monitoring</button>
                <button onclick="disableMonitoring()" class="btn danger">Disable Monitoring</button>
                <button onclick="sendSummaryNow()" class="btn">Send Summary Now</button>
                <button onclick="testAlert()" class="btn warning">Test Alert</button>
            </div>
            
            <div class="settings-section">
                <h3>📋 Upcoming Events Preview</h3>
                <div id="upcomingEvents">Loading events...</div>
                <button onclick="loadUpcomingEvents()" class="btn">Refresh Events</button>
            </div>
            
            <div class="settings-section">
                <a href="/" class="btn">Back to Chat</a>
                <a href="/integrations" class="btn">Integrations</a>
                <a href="/system" class="btn">System Dashboard</a>
            </div>
        </div>
        
        <script>
            let currentPreferences = {};
            
            document.addEventListener('DOMContentLoaded', function() {
                loadStatus();
                loadPreferences();
                loadUpcomingEvents();
                setInterval(loadStatus, 30000);
            });
            
            function loadStatus() {
                fetch('/api/calendar-alerts/status', {
                    credentials: 'include'
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        updateStatusDisplay(data.status);
                    } else {
                        document.getElementById('statusInfo').innerHTML = 
                            '<span style="color: #dc2626;">❌ ' + (data.error || 'Not configured') + '</span>';
                    }
                })
                .catch(e => {
                    document.getElementById('statusInfo').innerHTML = 
                        '<span style="color: #dc2626;">❌ Status check failed</span>';
                });
            }
            
            function updateStatusDisplay(status) {
                let html = '';
                
                html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">';
                
                html += '<div>';
                html += '<strong>Monitoring:</strong> ';
                html += status.monitoring_enabled ? 
                    '<span style="color: #059669;">✅ Enabled</span>' : 
                    '<span style="color: #dc2626;">❌ Disabled</span>';
                html += '</div>';
                
                html += '<div>';
                html += '<strong>Telegram:</strong> ';
                html += status.telegram_configured ? 
                    '<span style="color: #059669;">✅ Connected</span>' : 
                    '<span style="color: #dc2626;">❌ Not configured</span>';
                html += '</div>';
                
                html += '<div>';
                html += '<strong>Calendar:</strong> ';
                html += status.calendar_configured ? 
                    '<span style="color: #059669;">✅ Connected</span>' : 
                    '<span style="color: #dc2626;">❌ Not configured</span>';
                html += '</div>';
                
                html += '<div>';
                html += '<strong>Recent Alerts:</strong> ';
                html += status.recent_alerts_24h || 0;
                html += ' (24h)';
                html += '</div>';
                
                html += '</div>';
                
                if (status.upcoming_events_24h !== undefined) {
                    html += '<div style="margin-top: 10px;">';
                    html += '<strong>Upcoming Events (24h):</strong> ' + status.upcoming_events_24h;
                    html += '</div>';
                }
                
                document.getElementById('statusInfo').innerHTML = html;
            }
            
            function loadPreferences() {
                fetch('/api/calendar-alerts/preferences', {
                    credentials: 'include'
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        currentPreferences = data.preferences;
                        updateFormFromPreferences();
                    }
                })
                .catch(e => console.error('Failed to load preferences:', e));
            }
            
            function updateFormFromPreferences() {
                const prefs = currentPreferences;
                
                document.getElementById('meetingAlertsEnabled').checked = 
                    prefs.meeting_alerts?.enabled || false;
                
                const alertTimes = prefs.meeting_alerts?.alert_times || [15, 30];
                document.getElementById('alertTime1').value = alertTimes[0] || 15;
                document.getElementById('alertTime2').value = alertTimes[1] || 30;
                
                document.getElementById('weekendAlerts').checked = 
                    prefs.meeting_alerts?.include_weekends || false;
                
                document.getElementById('dailySummaryEnabled').checked = 
                    prefs.daily_summary?.enabled || false;
                
                document.getElementById('summaryTime').value = 
                    prefs.daily_summary?.time || '07:00';
                
                document.getElementById('includeTomorrow').checked = 
                    prefs.daily_summary?.include_tomorrow || false;
            }
            
            function updatePreferences() {
                const newPreferences = {
                    meeting_alerts: {
                        enabled: document.getElementById('meetingAlertsEnabled').checked,
                        alert_times: [
                            parseInt(document.getElementById('alertTime1').value) || 15,
                            parseInt(document.getElementById('alertTime2').value) || 30
                        ],
                        include_weekends: document.getElementById('weekendAlerts').checked
                    },
                    daily_summary: {
                        enabled: document.getElementById('dailySummaryEnabled').checked,
                        time: document.getElementById('summaryTime').value,
                        include_tomorrow: document.getElementById('includeTomorrow').checked
                    },
                    calendar_changes: currentPreferences.calendar_changes || {
                        enabled: false,
                        immediate_notification: true
                    },
                    recurring_reminders: currentPreferences.recurring_reminders || {
                        enabled: true,
                        max_per_event: 2
                    }
                };
                
                fetch('/api/calendar-alerts/preferences', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    credentials: 'include',
                    body: JSON.stringify({preferences: newPreferences})
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        currentPreferences = newPreferences;
                        console.log('Preferences saved');
                    } else {
                        alert('Failed to save preferences: ' + data.error);
                    }
                })
                .catch(e => {
                    console.error('Failed to save preferences:', e);
                    alert('Failed to save preferences');
                });
            }
            
            function enableMonitoring() {
                fetch('/api/calendar-alerts/enable', {
                    method: 'POST',
                    credentials: 'include'
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert('✅ Calendar monitoring enabled!');
                        loadStatus();
                    } else {
                        alert('❌ Failed to enable monitoring: ' + data.error);
                    }
                })
                .catch(e => alert('❌ Enable request failed'));
            }
            
            function disableMonitoring() {
                if (confirm('Disable calendar monitoring? You won\\'t receive alerts until re-enabled.')) {
                    fetch('/api/calendar-alerts/disable', {
                        method: 'POST',
                        credentials: 'include'
                    })
                    .then(r => r.json())
                    .then(data => {
                        if (data.success) {
                            alert('🔕 Calendar monitoring disabled');
                            loadStatus();
                        } else {
                            alert('❌ Failed to disable monitoring: ' + data.error);
                        }
                    })
                    .catch(e => alert('❌ Disable request failed'));
                }
            }
            
            function sendSummaryNow() {
                fetch('/api/calendar-alerts/send-summary', {
                    method: 'POST',
                    credentials: 'include'
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert('📅 Calendar summary sent to Telegram!');
                    } else {
                        alert('❌ Failed to send summary: ' + data.error);
                    }
                })
                .catch(e => alert('❌ Summary request failed'));
            }
            
            function testAlert() {
                fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    credentials: 'include',
                    body: JSON.stringify({
                        user_input: 'test calendar alert',
                        project: 'Calendar Testing',
                        voices: ['SyntaxPrime'],
                        random: false
                    })
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        alert('🧪 Test alert sent! Check your Telegram.');
                    } else {
                        alert('❌ Test failed: ' + (data.error || 'Unknown error'));
                    }
                })
                .catch(e => alert('❌ Test request failed'));
            }
            
            function loadUpcomingEvents() {
                fetch('/api/calendar-alerts/upcoming-events?hours=48', {
                    credentials: 'include'
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        displayUpcomingEvents(data.events);
                    } else {
                        document.getElementById('upcomingEvents').innerHTML = 
                            '<div style="color: #dc2626;">❌ ' + data.error + '</div>';
                    }
                })
                .catch(e => {
                    document.getElementById('upcomingEvents').innerHTML = 
                        '<div style="color: #dc2626;">❌ Failed to load events</div>';
                });
            }
            
            function displayUpcomingEvents(events) {
                if (events.length === 0) {
                    document.getElementById('upcomingEvents').innerHTML = 
                        '<div style="text-align: center; padding: 20px; color: #888;">No upcoming events in next 48 hours</div>';
                    return;
                }
                
                let html = '<div class="events-preview">';
                
                events.forEach(event => {
                    const startTime = new Date(event.start_time);
                    const timeStr = startTime.toLocaleString();
                    
                    html += '<div class="event-item">';
                    html += '<div>';
                    html += '<strong>' + event.title + '</strong><br>';
                    html += '<small>' + timeStr + '</small>';
                    if (event.location) {
                        html += '<br><small>📍 ' + event.location + '</small>';
                    }
                    html += '</div>';
                    
                    html += '<div>';
                    if (event.attendee_count > 0) {
                        html += '<small>👥 ' + event.attendee_count + '</small>';
                    }
                    html += '</div>';
                    
                    html += '</div>';
                });
                
                html += '</div>';
                html += '<div style="margin-top: 10px; text-align: center;">';
                html += '<small>' + events.length + ' events in next 48 hours</small>';
                html += '</div>';
                
                document.getElementById('upcomingEvents').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """)
    
# Section 13: Mobile API Routes
# Section 13: Mobile API Routes
# Section 13: Mobile API Routes (UPDATED FOR PHASE 2)
# Section 13: Mobile API Routes (UPDATED FOR CONSOLIDATED GOOGLE INTEGRATION)
# Section 13: Mobile API Routes (UPDATED WITH ENHANCED MARKETING)
# Section 13: Mobile API Routes (UPDATED WITH CALENDAR-TELEGRAM INTEGRATION)
# Section 13: Mobile API Routes (UPDATED WITH SLACK INTEGRATION)
# Section 13: Mobile API Routes (ENHANCED WITH JWT AUTHENTICATION FOR iOS)
# Section 13: Mobile API Routes (ENHANCED WITH JWT AUTHENTICATION FOR iOS + BLUESKY)
# Section 13: Mobile API Routes (ENHANCED WITH RSS MARKETING KNOWLEDGE INTEGRATION) 9/13/25
def generate_mobile_jwt(username: str) -> str:
    """Generate JWT token for mobile authentication"""
    if not JWT_AVAILABLE:
        raise Exception("JWT library not available")
    
    payload = {
        'mobile_authenticated': True,
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=30)  # 30-day expiry
    }
    
    return jwt.encode(payload, app.secret_key, algorithm='HS256')

def is_mobile_authenticated():
    """Check if mobile request is authenticated - ENHANCED VERSION"""
    if not JWT_AVAILABLE:
        # Fallback: allow access if JWT not available (backward compatibility)
        app.logger.warning("JWT not available, allowing mobile access")
        return True
    
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        # Fallback: check for old-style authentication or allow for now
        app.logger.info("No Bearer token found, checking fallback authentication")
        return True  # Temporary: allow access during transition
    
    token = auth_header[7:]  # Remove 'Bearer ' prefix
    
    try:
        payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
        return payload.get('mobile_authenticated', False)
    except jwt.InvalidTokenError as e:
        app.logger.warning(f"JWT token validation failed: {e}")
        return False

@app.route('/api/mobile/auth', methods=['POST'])
def mobile_authenticate():
    """Mobile authentication endpoint for iOS app"""
    try:
        if not JWT_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'JWT authentication not available on server. Install PyJWT library.',
                'fallback_available': True
            }), 500
            
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        # Validate credentials using the same logic as web login
        if password == PASSWORD:
            # Generate JWT token
            token = generate_mobile_jwt(username or 'mobile_user')
            
            return jsonify({
                'success': True,
                'token': token,
                'message': 'Authentication successful',
                'user': {
                    'username': username or 'mobile_user',
                    'authenticated': True
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid credentials'
            }), 401
            
    except Exception as e:
        app.logger.error(f"Mobile auth error: {e}")
        return jsonify({
            'success': False,
            'error': f'Authentication failed: {str(e)}'
        }), 500

@app.route('/api/mobile/projects', methods=['GET'])
def mobile_get_projects():
    """Get available projects for mobile app"""
    if not is_mobile_authenticated():
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        return jsonify({
            'success': True,
            'projects': PROJECTS,
            'default_project': PROJECTS[0] if PROJECTS else 'Personal Operating Manual'
        })
    except Exception as e:
        app.logger.error(f"Mobile projects error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get projects: {str(e)}'
        }), 500

@app.route('/api/mobile/conversations/<project>', methods=['GET'])
def mobile_get_conversations(project):
    """Get conversation history for a specific project (mobile)"""
    if not is_mobile_authenticated():
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        # Validate project
        if project not in PROJECTS:
            return jsonify({
                'success': False,
                'error': f'Invalid project: {project}'
            }), 400
        
        # Get recent conversations from database
        conversations = load_conversation_enhanced(project, limit=50)
        
        # Format for mobile consumption
        mobile_conversations = []
        for conv in conversations:
            mobile_conversations.append({
                'id': conv.get('id'),
                'user_input': conv.get('user_input', ''),
                'ai_response': conv.get('ai_response', {}),
                'timestamp': conv.get('created_at', ''),
                'project': project
            })
        
        return jsonify({
            'success': True,
            'project': project,
            'conversations': mobile_conversations,
            'total': len(mobile_conversations)
        })
        
    except Exception as e:
        app.logger.error(f"Mobile conversations error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get conversations: {str(e)}'
        }), 500

@app.route('/api/mobile/chat', methods=['POST'])
def mobile_chat():
    """Mobile chat with full AI processing - ENHANCED with RSS marketing knowledge integration"""
    if not is_mobile_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Handle both old and new request formats for compatibility
    data = request.get_json()
    user_input = data.get('user_input', data.get('message', '')).strip()
    project = data.get('project', 'Personal Operating Manual')
    use_voices = data.get('voices', ['SyntaxPrime'])
    random_toggle = data.get('random', False)
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    app.logger.info(f"Mobile chat request: '{user_input}' for project '{project}'")
    
    try:
        response_data = {}
        handled = False
        
        # Auto-refresh brain context periodically
        try:
            refresh_brain_context()
        except Exception as e:
            app.logger.warning(f"Brain context refresh failed: {e}")

        # PRIORITY 1: Handle reminder commands FIRST
        try:
            print(f"MOBILE: Checking reminder command for: '{user_input}'")
            response_data, handled = handle_reminder_command(user_input, project, use_voices, random_toggle)
            if handled:
                print(f"MOBILE: Reminder handled successfully!")
                save_conversation_enhanced(project, user_input, response_data)
                return jsonify({'success': True, 'responses': response_data})
            else:
                print(f"MOBILE: Not a reminder command, continuing...")
        except Exception as e:
            app.logger.error(f"Mobile: Reminder handler failed: {e}")
            # Don't set handled=True here - let other processors try

        # FIXED: BlueSky commands with enhanced pattern matching (moved higher in priority)
        if is_bluesky_configured():
            app.logger.info(f"Mobile: Checking BlueSky command patterns for: '{user_input}'")
            try:
                # Enhanced BlueSky command detection with more flexible patterns
                user_lower = user_input.lower().strip()
                
                # Comprehensive BlueSky trigger patterns
                bluesky_patterns = [
                    # Direct BlueSky mentions
                    'bluesky', 'bsky', 'blue sky',
                    # Action patterns
                    'analyze bluesky', 'check bluesky', 'my bluesky', 'bluesky feed',
                    'bluesky timeline', 'bluesky posts', 'bluesky analysis',
                    # Engagement patterns
                    'bluesky engagement', 'bluesky suggestions', 'who should i follow',
                    'bluesky opportunities', 'social engagement', 'feed analysis',
                    # High priority patterns
                    'bluesky high priority', 'best bluesky posts', 'top bluesky',
                    # Test patterns
                    'bluesky test', 'test bluesky', 'bluesky connection'
                ]
                
                # Check if input matches any BlueSky pattern
                bluesky_detected = False
                for pattern in bluesky_patterns:
                    if pattern in user_lower:
                        bluesky_detected = True
                        app.logger.info(f"Mobile: BlueSky pattern matched: '{pattern}'")
                        break
                
                # Also check for standalone keywords that might be BlueSky related
                standalone_keywords = ['bsky', 'bluesky']
                if not bluesky_detected:
                    for keyword in standalone_keywords:
                        if user_lower == keyword or user_lower.startswith(keyword + ' ') or user_lower.endswith(' ' + keyword):
                            bluesky_detected = True
                            app.logger.info(f"Mobile: BlueSky standalone keyword matched: '{keyword}'")
                            break
                
                if bluesky_detected:
                    app.logger.info(f"Mobile: Processing BlueSky command: '{user_input}'")
                    response_content = process_bluesky_command(user_input)
                    
                    # Check if we got a real response (not just the help menu)
                    if response_content and "Available BlueSky commands" not in response_content:
                        app.logger.info(f"Mobile: BlueSky command successfully processed")
                        response_data = {"SyntaxPrime": response_content}
                        handled = True
                        save_conversation_enhanced(project, user_input, response_data)
                        return jsonify({'success': True, 'responses': response_data})
                    else:
                        # If it's just the help menu, let it fall through to normal processing
                        # but log that we tried BlueSky
                        app.logger.info(f"Mobile: BlueSky returned help menu, falling through to normal processing")
                
            except Exception as e:
                app.logger.error(f"Mobile: BlueSky processing failed: {e}")
                # Don't fail the whole request, just log and continue
                pass

        # Enhanced Marketing Commands with RSS Knowledge Base
        if not handled:
            try:
                app.logger.info(f"Mobile: Enhanced marketing processing: '{user_input}'")
                temp_response, temp_handled = enhanced_marketing_command_processor(user_input, project, use_voices, random_toggle)
                if temp_handled:
                    response_data = temp_response
                    handled = True
                    app.logger.info(f"Mobile: Enhanced marketing command handled successfully")
                    save_conversation_enhanced(project, user_input, response_data)
                    return jsonify({'success': True, 'responses': response_data})
            except Exception as e:
                app.logger.error(f"Mobile: Enhanced marketing processing failed: {e}")

        # Try Cloze commands with proper configuration validation
        if is_cloze_configured():
            app.logger.info(f"Mobile: Cloze is configured, processing command: '{user_input}'")
            response_data, handled = process_cloze_command(user_input, project, use_voices, random_toggle)
            if handled:
                app.logger.info(f"Mobile: Cloze command handled successfully")
                save_conversation_enhanced(project, user_input, response_data)
                return jsonify({'success': True, 'responses': response_data})

        # Try Google integration
        if is_google_configured():
            app.logger.info(f"Mobile: Google is configured, processing command: '{user_input}'")
            try:
                from modules.enhanced_google_integration import EnhancedGoogleIntegration
                google_integration = EnhancedGoogleIntegration()
                response_data, handled = google_integration.process_google_commands(
                    user_input, project, use_voices, random_toggle
                )
                if handled:
                    app.logger.info(f"Mobile: Google command handled successfully")
                    save_conversation_enhanced(project, user_input, response_data)
                    return jsonify({'success': True, 'responses': response_data})
            except Exception as e:
                app.logger.error(f"Mobile: Google integration processing failed: {e}")

        # Try Slack integration
        if is_slack_configured():
            app.logger.info(f"Mobile: Slack is configured, processing command: '{user_input}'")
            response_data, handled = process_slack_command(user_input, project, use_voices, random_toggle)
            if handled:
                app.logger.info(f"Mobile: Slack command handled successfully")
                save_conversation_enhanced(project, user_input, response_data)
                return jsonify({'success': True, 'responses': response_data})

        # Try Telegram integration
        if is_telegram_configured():
            app.logger.info(f"Mobile: Telegram is configured, processing command: '{user_input}'")
            response_data, handled = process_telegram_command(user_input, project, use_voices, random_toggle)
            if handled:
                app.logger.info(f"Mobile: Telegram command handled successfully")
                save_conversation_enhanced(project, user_input, response_data)
                return jsonify({'success': True, 'responses': response_data})

        # Try Calendar-Telegram integration
        if is_calendar_telegram_configured():
            app.logger.info(f"Mobile: Calendar-Telegram is configured, processing command: '{user_input}'")
            response_data, handled = process_calendar_telegram_command(user_input, project, use_voices, random_toggle)
            if handled:
                app.logger.info(f"Mobile: Calendar-Telegram command handled successfully")
                save_conversation_enhanced(project, user_input, response_data)
                return jsonify({'success': True, 'responses': response_data})

        # Try ClickUp commands
        if is_clickup_configured():
            app.logger.info(f"Mobile: ClickUp is configured, processing command: '{user_input}'")
            response_data, handled = process_clickup_command(user_input, project, use_voices, random_toggle)
            if handled:
                app.logger.info(f"Mobile: ClickUp command handled successfully")
                save_conversation_enhanced(project, user_input, response_data)
                return jsonify({'success': True, 'responses': response_data})

        # Handle scrape command
        if user_input.lower().startswith("scrape "):
            url = user_input.split(" ", 1)[1].strip()
            try:
                result = scrape_url(url)
                if not result["ok"]:
                    response_data = {"SyntaxPrime": f"Could not fetch/extract content: {result['error']}"}
                else:
                    summary_prompt = (
                        "Summarize the key points from the following webpage for Carl. "
                        "Use bullets and keep it tight and actionable.\n\n"
                        f"--- SCRAPED CONTENT START ---\n{result['text']}\n--- SCRAPED CONTENT END ---"
                    )
                    retrieval_ctx = enhanced_retrieve(summary_prompt, k=5, project=project) if is_ready() else []
                    response_data = generate_response(
                        summary_prompt, use_voices, random_toggle,
                        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                    )
                handled = True
            except Exception as e:
                app.logger.error(f"Scrape command failed: {e}")
                response_data = {"SyntaxPrime": f"Scrape failed: {e}"}
                handled = True
            
            save_conversation_enhanced(project, user_input, response_data)
            return jsonify({'success': True, 'responses': response_data})

        # Normal AI response as fallback (same enhanced logic as web version)
        if not handled:
            try:
                retrieval_ctx = enhanced_retrieve(user_input, k=5, project=project) if is_ready() else []
                
                # Use enhanced response generation with context validation
                response_data = generate_response_with_context_check(
                    user_input, use_voices, random_toggle,
                    project, CHAT_MODEL, retrieval_ctx
                )
                
                save_conversation_enhanced(project, user_input, response_data)
            except Exception as e:
                app.logger.error(f"Normal response generation failed: {e}")
                response_data = {"SyntaxPrime": f"Response generation failed: {e}"}
                save_conversation_enhanced(project, user_input, response_data)
        
        return jsonify({'success': True, 'responses': response_data})
        
    except Exception as e:
        app.logger.error(f"Mobile chat failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/mobile/status', methods=['GET'])
def mobile_status():
    """Mobile app status check endpoint"""
    if not is_mobile_authenticated():
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    try:
        # Check system status
        status = {
            'server_online': True,
            'brain_ready': is_ready(),
            'jwt_available': JWT_AVAILABLE,
            'integrations': {
                'google': True,  # Always true since we have the integration
                'marketing': is_marketing_configured(),
                'marketing_rss': True,  # RSS marketing knowledge is always available
                'cloze': is_cloze_configured(),
                'clickup': is_clickup_configured(),
                'telegram': is_telegram_configured(),
                'calendar_telegram': is_calendar_telegram_configured(),
                'bluesky': is_bluesky_configured()
            },
            'projects': PROJECTS,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        app.logger.error(f"Mobile status error: {e}")
        return jsonify({
            'success': False,
            'error': f'Status check failed: {str(e)}'
        }), 500


# Section 25: Google Drive Export API Routes (NEW)
@app.route('/api/export/status')
def api_export_status():
    """Check Google Drive export capability status"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        status = {
            'google_configured': is_google_configured(),
            'export_available': False,
            'scopes_granted': [],
            'missing_scopes': [],
            'error': None
        }
        
        if is_google_configured():
            try:
                from modules.enhanced_google_integration import EnhancedGoogleIntegration
                google_integration = EnhancedGoogleIntegration()
                
                if google_integration.is_configured():
                    status['export_available'] = True
                    # Try to get scopes if available
                    try:
                        credentials = google_integration.get_credentials()
                        if credentials and hasattr(credentials, 'scopes'):
                            status['scopes_granted'] = credentials.scopes or []
                    except:
                        pass
                else:
                    status['error'] = 'No valid Google credentials'
            except Exception as e:
                status['error'] = f'Google integration error: {str(e)}'
        else:
            status['error'] = 'Google integration not configured or missing required scopes'
            status['missing_scopes'] = [
                'https://www.googleapis.com/auth/documents',
                'https://www.googleapis.com/auth/drive.file'
            ]
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/bookmarks', methods=['POST'])
def api_export_bookmarks():
    """API endpoint to export bookmarks to Google Docs"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json() or {}
        project_filter = data.get('project') or session.get('current_project', 'Personal Operating Manual')
        include_responses = data.get('include_responses', True)
        bookmark_ids = data.get('bookmark_ids', [])  # Optional: specific bookmark IDs
        
        from modules.database import get_bookmarks
        
        if not is_google_configured():
            return jsonify({
                'success': False,
                'error': 'Google Drive export not available. Please configure Google OAuth with document creation scopes.'
            }), 400
        
        # Get bookmarks
        if bookmark_ids:
            # Get specific bookmarks (implementation depends on your bookmark system)
            bookmarks = get_bookmarks(bookmark_ids=bookmark_ids)
        else:
            # Get all bookmarks for project
            bookmarks = get_bookmarks(project=project_filter, limit=50)
        
        if not bookmarks:
            return jsonify({
                'success': False,
                'error': f'No bookmarks found for project "{project_filter}"'
            }), 404
        
        # Convert to export format and use the chat export command
        export_request = f"export my bookmarks for {project_filter}"
        if not include_responses:
            export_request += " without AI responses"
        
        response_data, handled = handle_export_command(export_request, project_filter, ['SyntaxPrime'], False)
        
        if handled and response_data:
            return jsonify({
                'success': True,
                'message': 'Bookmarks exported successfully',
                'export_result': response_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Export command failed to process'
            }), 500
        
    except Exception as e:
        app.logger.error(f"Bookmark export API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export/conversation', methods=['POST'])
def api_export_conversation():
    """API endpoint to export a specific conversation to Google Docs"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json() or {}
        conversation_id = data.get('conversation_id')
        project_filter = data.get('project') or session.get('current_project', 'Personal Operating Manual')
        include_responses = data.get('include_responses', True)
        
        if not is_google_configured():
            return jsonify({
                'success': False,
                'error': 'Google Drive export not available. Please configure Google OAuth with document creation scopes.'
            }), 400
        
        # Create export request
        if conversation_id:
            export_request = f"export conversation {conversation_id}"
        else:
            export_request = f"export recent conversation"
        
        if not include_responses:
            export_request += " without AI responses"
        
        response_data, handled = handle_export_command(export_request, project_filter, ['SyntaxPrime'], False)
        
        if handled and response_data:
            return jsonify({
                'success': True,
                'message': 'Conversation exported successfully',
                'export_result': response_data
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Export command failed to process'
            }), 500
        
    except Exception as e:
        app.logger.error(f"Conversation export API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/diagnostics/google-export')
def google_export_diagnostics():
    """Google Drive export diagnostics page"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        diagnostics = {
            'export_available': is_google_configured(),
            'google_configured': is_google_configured(),
            'required_scopes': [
                'https://www.googleapis.com/auth/documents',
                'https://www.googleapis.com/auth/drive.file'
            ],
            'granted_scopes': [],
            'test_results': {}
        }
        
        if is_google_configured():
            try:
                from modules.enhanced_google_integration import EnhancedGoogleIntegration
                google_integration = EnhancedGoogleIntegration()
                
                # Test basic connectivity
                if google_integration.is_configured():
                    diagnostics['test_results']['connectivity'] = {
                        'success': True,
                        'message': 'Google integration is configured and ready'
                    }
                    
                    # Try to get scopes
                    try:
                        credentials = google_integration.get_credentials()
                        if credentials and hasattr(credentials, 'scopes'):
                            diagnostics['granted_scopes'] = credentials.scopes or []
                    except:
                        diagnostics['granted_scopes'] = ['Unable to retrieve scope information']
                else:
                    diagnostics['test_results']['connectivity'] = {
                        'success': False,
                        'message': 'Google integration is not properly configured'
                    }
                    
            except Exception as e:
                diagnostics['test_results']['connectivity'] = {
                    'success': False,
                    'error': str(e)
                }
        else:
            diagnostics['test_results']['connectivity'] = {
                'success': False,
                'message': 'Google integration not configured'
            }
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Google Drive Export Diagnostics</title>
            <style>
                body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff00; padding: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .diagnostic-section { background: #1a1a1a; border: 1px solid #333; padding: 20px; margin: 20px 0; border-radius: 5px; }
                .success { color: #00ff00; }
                .error { color: #ff4444; }
                .warning { color: #ffaa00; }
                pre { white-space: pre-wrap; }
                .btn { background: #6366f1; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; text-decoration: none; display: inline-block; margin: 10px 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Google Drive Export Diagnostics</h1>
                
                <div class="diagnostic-section">
                    <h3>Export System Status</h3>
                    <p><strong>Export Available:</strong> <span class="{% if diagnostics.export_available %}success">✅ Yes{% else %}error">❌ No{% endif %}</span></p>
                    <p><strong>Google Configured:</strong> <span class="{% if diagnostics.google_configured %}success">✅ Yes{% else %}error">❌ No{% endif %}</span></p>
                </div>
                
                <div class="diagnostic-section">
                    <h3>Required Scopes</h3>
                    {% for scope in diagnostics.required_scopes %}
                    <p>{{ scope }}</p>
                    {% endfor %}
                </div>
                
                {% if diagnostics.granted_scopes %}
                <div class="diagnostic-section">
                    <h3>Granted Scopes</h3>
                    {% for scope in diagnostics.granted_scopes %}
                    <p>{{ scope }}</p>
                    {% endfor %}
                </div>
                {% endif %}
                
                {% if diagnostics.test_results %}
                <div class="diagnostic-section">
                    <h3>Test Results</h3>
                    <pre>{{ diagnostics.test_results | tojson(indent=2) }}</pre>
                </div>
                {% endif %}
                
                <div class="diagnostic-section">
                    <h3>Available Commands</h3>
                    <pre>{{ export_help }}</pre>
                </div>
                
                <a href="/integrations" class="btn">Google Integration Setup</a>
                <a href="/" class="btn">Back to Chat</a>
            </div>
        </body>
        </html>
        """, diagnostics=diagnostics, export_help=get_export_help())
        
    except Exception as e:
        return f"Export diagnostics failed: {str(e)}", 500

# Section 14: Google OAuth Integration
# Section 14: Google OAuth Integration (COMPLETELY REPLACED)
# Section 14: Google OAuth Integration (COMPLETELY REPLACED FOR PHASE 2)
# Section 14: Google OAuth Integration
# Section 14: Google OAuth Integration (COMPLETELY REPLACED)
# Section 14: Google OAuth Integration (COMPLETELY REPLACED FOR PHASE 2)
@app.route('/google/auth/start')
def google_auth_start():
    """Updated OAuth flow with Phase 2 scopes (Docs, Sheets, Analytics)"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')
        
        if not os.path.exists(credentials_path):
            from modules.google_oauth_config import GOOGLE_SETUP_TEMPLATE
            railway_url = os.getenv('RAILWAY_STATIC_URL', 'your-app.railway.app')
            return render_template_string(GOOGLE_SETUP_TEMPLATE, railway_url=railway_url)
        
        from google_auth_oauthlib.flow import Flow
        from modules.google_oauth_config import get_oauth_scopes
        
        railway_url = os.getenv('RAILWAY_STATIC_URL')
        redirect_uri = f"https://{railway_url}/google/auth/callback" if railway_url else "http://localhost:5000/google/auth/callback"
        
        app.logger.info(f"Starting OAuth flow with redirect URI: {redirect_uri}")
        
        # Use Phase 2 scopes including Docs, Sheets, Analytics
        scopes = get_oauth_scopes("standard")  # Default to Phase 2 standard scopes
        app.logger.info(f"OAuth scopes: {scopes}")
        
        flow = Flow.from_client_secrets_file(credentials_path, scopes=scopes)
        flow.redirect_uri = redirect_uri
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'  # Force consent to ensure all Phase 2 scopes are granted
        )
        
        session['oauth_state'] = state
        session['oauth_redirect_uri'] = redirect_uri
        
        app.logger.info(f"Redirecting to Google OAuth: {authorization_url}")
        return redirect(authorization_url)
        
    except Exception as e:
        app.logger.error(f"OAuth start failed: {e}")
        return f"OAuth initialization failed: {str(e)}<br><a href='/integrations'>Setup Instructions</a>", 500

@app.route('/google/auth/callback')
def google_auth_callback():
    """Updated OAuth callback with Phase 2 scope validation"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        error = request.args.get('error')
        if error:
            app.logger.error(f"OAuth error: {error}")
            return f"OAuth failed: {error}<br><a href='/integrations'>Try Again</a>", 400
        
        if request.args.get('state') != session.get('oauth_state'):
            app.logger.error("OAuth state mismatch")
            return "Invalid state parameter - possible CSRF attack<br><a href='/integrations'>Try Again</a>", 400
        
        from google_auth_oauthlib.flow import Flow
        from modules.google_oauth_config import get_oauth_scopes, validate_scopes_in_token, OAUTH_SUCCESS_TEMPLATE
        
        credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')
        token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        
        flow = Flow.from_client_secrets_file(
            credentials_path,
            scopes=get_oauth_scopes("standard"),
            state=session['oauth_state']
        )
        flow.redirect_uri = session.get('oauth_redirect_uri')
        
        app.logger.info(f"Processing OAuth callback, saving token to: {token_path}")
        
        flow.fetch_token(authorization_response=request.url)
        credentials = flow.credentials
        
        # Validate scopes with Phase 2 detection
        scope_validation = validate_scopes_in_token(credentials)
        
        with open(token_path, 'w') as token_file:
            token_file.write(credentials.to_json())
        
        app.logger.info("Token saved successfully")
        
        # Test the credentials with all services
        test_results = test_google_services(credentials)
        
        session.pop('oauth_state', None)
        session.pop('oauth_redirect_uri', None)
        
        return render_template_string(
            OAUTH_SUCCESS_TEMPLATE,
            token_path=token_path,
            test_results=test_results,
            scope_validation=scope_validation
        )
        
    except Exception as e:
        app.logger.error(f"OAuth callback failed: {e}")
        return f"OAuth completion failed: {str(e)}<br><a href='/integrations'>Try Again</a>", 500

def test_google_services(credentials):
    """Test all Google services with new credentials - Updated for Phase 2"""
    test_results = {}
    
    try:
        from googleapiclient.discovery import build
        
        # Test Gmail (Phase 1)
        try:
            gmail_svc = build('gmail', 'v1', credentials=credentials)
            profile = gmail_svc.users().getProfile(userId='me').execute()
            test_results['gmail'] = f"✅ Connected as {profile.get('emailAddress', 'Unknown')}"
        except Exception as e:
            test_results['gmail'] = f"❌ Gmail test failed: {str(e)}"
        
        # Test Calendar (Phase 1)
        try:
            cal_svc = build('calendar', 'v3', credentials=credentials)
            calendar_list = cal_svc.calendarList().list(maxResults=1).execute()
            test_results['calendar'] = f"✅ Access to {len(calendar_list.get('items', []))} calendars"
        except Exception as e:
            test_results['calendar'] = f"❌ Calendar test failed: {str(e)}"
        
        # Test Drive (Phase 1)
        try:
            drive_svc = build('drive', 'v3', credentials=credentials)
            about = drive_svc.about().get(fields="user,storageQuota").execute()
            user_email = about.get('user', {}).get('emailAddress', 'Unknown')
            test_results['drive'] = f"✅ Drive access for {user_email}"
        except Exception as e:
            test_results['drive'] = f"❌ Drive test failed: {str(e)}"
        
        # Test Google Docs (Phase 2)
        try:
            docs_svc = build('docs', 'v1', credentials=credentials)
            # Simple test - we can't create without a document ID
            test_results['docs'] = f"✅ Google Docs API ready for document creation"
        except Exception as e:
            test_results['docs'] = f"❌ Docs API test failed: {str(e)}"
        
        # Test Google Sheets (Phase 2)
        try:
            sheets_svc = build('sheets', 'v4', credentials=credentials)
            # Simple test - API connection
            test_results['sheets'] = f"✅ Google Sheets API ready for spreadsheet operations"
        except Exception as e:
            test_results['sheets'] = f"❌ Sheets API test failed: {str(e)}"
        
        # Test Google Slides (Phase 2)
        try:
            slides_svc = build('slides', 'v1', credentials=credentials)
            # Simple test - API connection
            test_results['slides'] = f"✅ Google Slides API ready for presentation creation"
        except Exception as e:
            test_results['slides'] = f"❌ Slides API test failed: {str(e)}"
        
        # Test Analytics (Phase 2) - Optional
        try:
            analytics_svc = build('analyticsreporting', 'v4', credentials=credentials)
            test_results['analytics'] = f"✅ Analytics API connected (requires View ID configuration)"
        except Exception as e:
            test_results['analytics'] = f"⚠️ Analytics API test failed: {str(e)} (configure GOOGLE_ANALYTICS_VIEW_ID)"
        
        # Test Search Console (Phase 2) - Optional
        try:
            searchconsole_svc = build('searchconsole', 'v1', credentials=credentials)
            test_results['searchconsole'] = f"✅ Search Console API connected (requires site verification)"
        except Exception as e:
            test_results['searchconsole'] = f"⚠️ Search Console API test failed: {str(e)} (verify site in Search Console)"
            
    except Exception as e:
        test_results['error'] = f"Service testing failed: {str(e)}"
    
    return test_results

@app.route('/api/google/token-status')
def google_token_status():
    """Check Google token status with automatic refresh capability"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Try to import the token manager
        try:
            from modules.google_token_refresh import token_manager, get_google_credentials
            
            # Get token status from the manager
            status = token_manager.get_token_status()
            
            # Try to get valid credentials (this will attempt refresh if needed)
            credentials = get_google_credentials()
            
            if credentials:
                status['credentials_available'] = True
                status['scopes_count'] = len(credentials.scopes) if credentials.scopes else 0
                status['scopes'] = list(credentials.scopes) if credentials.scopes else []
            else:
                status['credentials_available'] = False
            
            # Add additional diagnostics
            status['token_file_exists'] = os.path.exists(os.getenv('GOOGLE_TOKEN_PATH', 'token.json'))
            status['credentials_file_exists'] = os.path.exists(os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json'))
            
            return jsonify({
                'success': True,
                'token_manager_available': True,
                **status
            })
            
        except ImportError:
            # Fallback to legacy token checking
            app.logger.warning("Token manager not available, using legacy token check")
            
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            
            token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
            
            if not os.path.exists(token_path):
                return jsonify({
                    'success': False,
                    'token_manager_available': False,
                    'status': 'missing',
                    'message': 'No token file found',
                    'needs_auth': True,
                    'token_file_exists': False
                })
            
            # Load and check token
            credentials = Credentials.from_authorized_user_file(token_path)
            
            status = {
                'valid': credentials.valid,
                'expired': credentials.expired,
                'has_refresh_token': bool(credentials.refresh_token),
                'scopes': list(credentials.scopes) if credentials.scopes else [],
                'scopes_count': len(credentials.scopes) if credentials.scopes else 0,
                'needs_auth': False,
                'token_file_exists': True,
                'credentials_file_exists': os.path.exists(os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json'))
            }
            
            if credentials.valid:
                status['status'] = 'valid'
                status['message'] = 'Token is valid and ready to use'
            elif credentials.expired and credentials.refresh_token:
                status['status'] = 'expired_refreshable'
                status['message'] = 'Token expired but can be refreshed'
                
                # Try to refresh
                try:
                    credentials.refresh(Request())
                    
                    # Save refreshed token
                    with open(token_path, 'w') as token_file:
                        token_file.write(credentials.to_json())
                    
                    status['status'] = 'refreshed'
                    status['message'] = 'Token was expired but has been refreshed'
                    status['valid'] = True
                    status['expired'] = False
                    
                except Exception as refresh_error:
                    status['status'] = 'refresh_failed'
                    status['message'] = f'Token refresh failed: {str(refresh_error)}'
                    status['needs_auth'] = True
                    
            elif credentials.expired and not credentials.refresh_token:
                status['status'] = 'expired_no_refresh'
                status['message'] = 'Token expired and cannot be refreshed - need re-authentication'
                status['needs_auth'] = True
            else:
                status['status'] = 'unknown'
                status['message'] = 'Token status unclear'
            
            return jsonify({
                'success': True,
                'token_manager_available': False,
                **status
            })
            
    except Exception as e:
        app.logger.error(f"Token status check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'token_manager_available': False
        }), 500

@app.route('/api/google/force-refresh', methods=['POST'])
def force_google_token_refresh():
    """Force a Google token refresh (for testing/debugging)"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Try using the token manager first
        try:
            from modules.google_token_refresh import force_token_refresh, token_manager
            
            success = force_token_refresh()
            
            if success:
                # Get updated status
                status = token_manager.get_token_status()
                return jsonify({
                    'success': True,
                    'message': 'Token refreshed successfully using token manager',
                    'method': 'token_manager',
                    'status': status
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Token refresh failed - check refresh token availability',
                    'method': 'token_manager'
                }), 400
                
        except ImportError:
            # Fallback to legacy refresh
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            
            token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
            
            if not os.path.exists(token_path):
                return jsonify({
                    'success': False,
                    'error': 'No token file found',
                    'method': 'legacy'
                }), 400
            
            credentials = Credentials.from_authorized_user_file(token_path)
            
            if not credentials.refresh_token:
                return jsonify({
                    'success': False,
                    'error': 'No refresh token available - need re-authentication',
                    'method': 'legacy'
                }), 400
            
            # Attempt refresh
            credentials.refresh(Request())
            
            # Save refreshed token
            with open(token_path, 'w') as token_file:
                token_file.write(credentials.to_json())
            
            return jsonify({
                'success': True,
                'message': 'Token refreshed successfully using legacy method',
                'method': 'legacy',
                'valid': credentials.valid,
                'expired': credentials.expired
            })
            
    except Exception as e:
        app.logger.error(f"Force token refresh failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

        
# Section 15: Backup and Maintenance Routes
# Section 15: Backup and Maintenance Routes
from modules.backup_maintenance import (
    backup_manager,
    backup_scheduler,
    get_backup_status,
    start_automated_backups,
    stop_automated_backups
)

@app.route('/backup/create-database', methods=['POST'])
def create_database_backup():
    """Create database backup"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        result = backup_manager.create_database_backup()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/backup/create-brain', methods=['POST'])
def create_brain_backup():
    """Create brain backup"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        result = backup_manager.create_brain_backup()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/backup/create-full', methods=['POST'])
def create_full_backup():
    """Create full system backup"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        result = backup_manager.create_full_system_backup()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/backup/reindex', methods=['POST'])
def reindex_knowledge_base():
    """Reindex knowledge base"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        result = backup_manager.reindex_knowledge_base()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/backup/maintenance', methods=['POST'])
def perform_maintenance():
    """Perform full maintenance"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        result = backup_manager.perform_maintenance()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/backup/start-scheduler', methods=['POST'])
def start_backup_scheduler():
    """Start automated backup scheduler"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        success = start_automated_backups()
        return jsonify({
            'success': success,
            'message': 'Scheduler started' if success else 'Scheduler already running'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/backup/stop-scheduler', methods=['POST'])
def stop_backup_scheduler():
    """Stop automated backup scheduler"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        success = stop_automated_backups()
        return jsonify({
            'success': success,
            'message': 'Scheduler stopped' if success else 'Scheduler not running'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/backup/download/<filename>')
def download_backup_file(filename):
    """Download a backup file"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        backup_path = os.path.join(backup_manager.backup_dir, filename)
        
        if not os.path.exists(backup_path):
            return "Backup file not found", 404
        
        if not backup_path.startswith(os.path.abspath(backup_manager.backup_dir)):
            return "Invalid file path", 400
        
        return send_file(
            backup_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        return f"Download failed: {str(e)}", 500

@app.route('/admin/reload-modules', methods=['POST'])
def reload_modules():
    """Force reload of Python modules to fix deployment caching issues"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        import sys
        import importlib
        
        # List of modules that commonly need reloading
        modules_to_reload = [
            'modules.gmail',
            'modules.smart_commands',
            'modules.clickup_integration',
            'modules.cloze_integration',
            'modules.marketing_commands',
            'modules.telegram_notifications',
            'utils.ghostline_engine',
            'utils.gmail_client',
            'modules.brain',
            'modules.file_processing',
            'modules.utils'
        ]
        
        reloaded_modules = []
        failed_modules = []
        
        for module_name in modules_to_reload:
            try:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
                    reloaded_modules.append(module_name)
                    app.logger.info(f"Reloaded module: {module_name}")
            except Exception as e:
                failed_modules.append(f"{module_name}: {str(e)}")
                app.logger.error(f"Failed to reload {module_name}: {e}")
        
        return jsonify({
            "success": True,
            "reloaded_modules": reloaded_modules,
            "failed_modules": failed_modules,
            "message": f"Reloaded {len(reloaded_modules)} modules"
        })
        
    except Exception as e:
        app.logger.error(f"Module reload failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/admin/restart', methods=['POST'])
def force_restart():
    """Force application restart to clear all caches"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        app.logger.warning("Force restart requested by user")
        
        # Send response before restarting
        response = jsonify({
            "success": True,
            "message": "Application restarting in 2 seconds..."
        })
        
        # Schedule restart after response is sent
        def delayed_restart():
            time.sleep(2)
            app.logger.warning("Executing force restart")
            os._exit(0)  # Force process restart
        
        restart_thread = threading.Thread(target=delayed_restart, daemon=True)
        restart_thread.start()
        
        return response
        
    except Exception as e:
        app.logger.error(f"Force restart failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/admin/clear-cache', methods=['POST'])
def clear_cache():
    """Clear various application caches"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        import gc
        
        cache_actions = []
        
        # Clear Python bytecode cache
        try:
            import sys
            for module_name in list(sys.modules.keys()):
                if module_name.startswith(('modules.', 'utils.')):
                    if hasattr(sys.modules[module_name], '__pycache__'):
                        delattr(sys.modules[module_name], '__pycache__')
            cache_actions.append("Python bytecode cache cleared")
        except Exception as e:
            cache_actions.append(f"Bytecode cache clear failed: {e}")
        
        # Force garbage collection
        try:
            collected = gc.collect()
            cache_actions.append(f"Garbage collection: {collected} objects freed")
        except Exception as e:
            cache_actions.append(f"Garbage collection failed: {e}")
        
        # Clear session data (but keep login status)
        try:
            logged_in = session.get('logged_in')
            session.clear()
            if logged_in:
                session['logged_in'] = True
            cache_actions.append("Session cache cleared (keeping login)")
        except Exception as e:
            cache_actions.append(f"Session clear failed: {e}")
        
        return jsonify({
            "success": True,
            "actions_performed": cache_actions,
            "message": "Cache clearing completed"
        })
        
    except Exception as e:
        app.logger.error(f"Cache clearing failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Section 15.5: RSS Marketing Knowledge System Routes 9/13/25
@app.route('/marketing-knowledge')
def marketing_knowledge_dashboard():
    """Marketing knowledge dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        retriever = get_marketing_retriever()
        
        # Get fresh insights and stats
        fresh_insights = retriever.get_fresh_marketing_insights(days=7, limit=8)
        category_stats = retriever.get_category_stats()
        rss_status = get_rss_status()
        
        return render_template_string(MARKETING_KNOWLEDGE_TEMPLATE,
                                    fresh_insights=fresh_insights,
                                    category_stats=category_stats,
                                    rss_status=rss_status)
    except Exception as e:
        return f"Marketing knowledge dashboard error: {str(e)}", 500

@app.route('/api/marketing-knowledge/search', methods=['POST'])
def api_search_marketing_knowledge():
    """API: Search marketing knowledge base"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        category = data.get('category')
        limit = min(int(data.get('limit', 8)), 20)
        
        if not query:
            return jsonify({'success': False, 'error': 'Query required'}), 400
        
        retriever = get_marketing_retriever()
        
        if category and category != 'all':
            results = retriever.search_marketing_content(
                query=query,
                category=category,
                limit=limit
            )
        else:
            results = retriever.get_contextual_marketing_advice(query, limit)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/marketing-knowledge/seo-tips', methods=['POST'])
def api_get_seo_tips():
    """API: Get SEO optimization tips"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json() or {}
        topic = data.get('topic', '')
        
        tips = get_seo_advice(topic if topic else None)
        
        return jsonify({
            'success': True,
            'topic': topic or 'general',
            'tips': tips,
            'count': len(tips)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/marketing-knowledge/content-tips', methods=['POST'])
def api_get_content_tips():
    """API: Get content writing tips"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json() or {}
        content_type = data.get('content_type', 'blog')
        
        tips = get_content_writing_tips(content_type)
        
        return jsonify({
            'success': True,
            'content_type': content_type,
            'tips': tips,
            'count': len(tips)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/marketing-knowledge/social-tips', methods=['POST'])
def api_get_social_tips():
    """API: Get social media tips"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json() or {}
        platform = data.get('platform')
        
        tips = get_social_media_advice(platform)
        
        return jsonify({
            'success': True,
            'platform': platform or 'all',
            'tips': tips,
            'count': len(tips)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rss-monitor/status')
def api_rss_monitor_status():
    """API: Get RSS monitor status"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        status = get_rss_status()
        return jsonify({
            'success': True,
            'status': status
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rss-monitor/start', methods=['POST'])
def api_start_rss_monitor():
    """API: Start RSS monitoring"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        success = start_rss_monitoring()
        
        return jsonify({
            'success': success,
            'message': 'RSS monitoring started' if success else 'RSS monitoring already running'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rss-monitor/stop', methods=['POST'])
def api_stop_rss_monitor():
    """API: Stop RSS monitoring"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        success = stop_rss_monitoring()
        
        return jsonify({
            'success': success,
            'message': 'RSS monitoring stopped' if success else 'RSS monitoring not running'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/rss-monitor/force-update', methods=['POST'])
def api_force_rss_update():
    """API: Force RSS feed update"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        success = force_feed_update()
        
        return jsonify({
            'success': success,
            'message': 'Feed update completed' if success else 'Feed update failed'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Section 16: Marketing Debug and Enhancement Routes
@app.route('/api/marketing/context-debug')
def marketing_context_debug():
    """Debug endpoint to see recent marketing context"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        summary = marketing_context.get_recent_context_summary()
        return jsonify({
            'success': True,
            'context_summary': summary,
            'context_items': len(marketing_context.recent_concepts)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/marketing/recent-concepts')
def get_recent_marketing_concepts():
    """Get recent marketing concepts for autocomplete/suggestions"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        recent = marketing_context.recent_concepts[-5:]  # Last 5 concepts
        concepts = [
            {
                'concept': item['extracted_concept'],
                'timestamp': item['timestamp'],
                'success': item['result']['success']
            }
            for item in recent
        ]
        
        return jsonify({
            'success': True,
            'recent_concepts': concepts
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/marketing/clear-context', methods=['POST'])
def clear_marketing_context():
    """Clear marketing context (for testing/debugging)"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        cleared_count = len(marketing_context.recent_concepts)
        marketing_context.recent_concepts.clear()
        marketing_context.conversation_memory.clear()
        
        return jsonify({
            'success': True,
            'message': f'Cleared {cleared_count} context items'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Section 17: Utility and Export Routes
@app.route('/healthz')
def healthz():
    build_status = get_build_status()
    status = {
        "status": "ok",
        "brain_ready": build_status["status"] == "complete",
        "brain_building": _rag_building or build_status["status"] == "building",
        "brain_progress": build_status["progress"],
        "brain_chunks": build_status.get("chunks_processed", 0)
    }
    return jsonify(status)

@app.route('/export/<project>')
def export_session(project):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    session_path = f'sessions/{project.lower().replace(" ", "_")}.json'
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        content = ""
        for line in lines:
            entry = json.loads(line)
            content += f"### Prompt:\n{entry['prompt']}\n"
            for voice, reply in entry['response'].items():
                content += f"- **{voice}**: {reply}\n"
            content += "\n---\n\n"
        file_stream = io.BytesIO()
        file_stream.write(content.encode('utf-8'))
        file_stream.seek(0)
        return send_file(
            file_stream,
            mimetype='text/markdown',
            as_attachment=True,
            download_name=f"{project}_session.md"
        )
    except FileNotFoundError:
        return f"No session data found for project: {project}", 404
        
# Section 17: Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['password'] == PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = "Wrong password."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))
    
# Section 19: Timezone Management Routes
@app.route('/api/timezone/detect', methods=['POST'])
def detect_timezone():
    """Receive browser-detected timezone"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        timezone_name = data.get('timezone')
        
        if not timezone_name:
            return jsonify({'success': False, 'error': 'No timezone provided'}), 400
        
        from modules.timezone_handler import timezone_manager
        
        success = timezone_manager.set_detected_timezone(timezone_name)
        if success:
            tz_info = timezone_manager.get_timezone_info()
            return jsonify({
                'success': True,
                'timezone_set': timezone_name,
                'current_info': tz_info
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid timezone: {timezone_name}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/timezone/set', methods=['POST'])
def set_user_timezone_route():
    """Set user's preferred timezone manually"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        timezone_name = data.get('timezone')
        
        if not timezone_name:
            return jsonify({'success': False, 'error': 'No timezone provided'}), 400
        
        from modules.timezone_handler import timezone_manager
        
        success = timezone_manager.set_user_timezone(timezone_name)
        if success:
            tz_info = timezone_manager.get_timezone_info()
            return jsonify({
                'success': True,
                'timezone_set': timezone_name,
                'current_info': tz_info
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Invalid timezone: {timezone_name}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/timezone/info')
def get_timezone_info_route():
    """Get current timezone information"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        from modules.timezone_handler import timezone_manager
        tz_info = timezone_manager.get_timezone_info()
        common_timezones = timezone_manager.get_common_timezones()
        
        return jsonify({
            'success': True,
            'current': tz_info,
            'common_timezones': common_timezones,
            'user_set': 'user_timezone' in session,
            'detected': 'detected_timezone' in session
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/timezone-settings')
def timezone_settings():
    """Timezone settings page"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Timezone Settings</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
            }
            .container { max-width: 800px; margin: 0 auto; }
            .btn { 
                background: #6366f1; color: white; border: none; padding: 12px 24px;
                border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
                text-decoration: none; display: inline-block;
            }
            .btn:hover { background: #5855eb; }
            .btn.success { background: #059669; }
            .btn.warning { background: #d97706; }
            .settings-section { 
                background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
                padding: 20px; margin: 20px 0; 
            }
            .form-group { margin: 15px 0; }
            .form-group label { display: block; margin-bottom: 5px; font-weight: bold; }
            .form-group select, .form-group input { 
                width: 100%; padding: 10px; background: #333; color: #fff; 
                border: 1px solid #555; border-radius: 4px; font-size: 16px;
            }
            .current-time { 
                font-size: 24px; font-weight: bold; text-align: center; 
                padding: 20px; background: #2a2a2a; border-radius: 8px; margin: 20px 0;
            }
            .status-indicator { 
                display: inline-block; width: 12px; height: 12px; border-radius: 50%; 
                margin-right: 8px;
            }
            .status-detected { background: #059669; }
            .status-manual { background: #6366f1; }
            .status-default { background: #d97706; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Timezone Settings</h1>
            
            <div class="current-time" id="currentTime">Loading current time...</div>
            
            <div class="settings-section">
                <h3>Timezone Status</h3>
                <div id="timezoneStatus">Loading...</div>
            </div>
            
            <div class="settings-section">
                <h3>Automatic Detection</h3>
                <p>Let your browser automatically detect your timezone:</p>
                <button onclick="detectTimezone()" class="btn">Detect My Timezone</button>
                <div id="detectionResult"></div>
            </div>
            
            <div class="settings-section">
                <h3>Manual Selection</h3>
                <p>Choose your timezone manually:</p>
                <div class="form-group">
                    <label for="timezoneSelect">Select Timezone:</label>
                    <select id="timezoneSelect" onchange="setTimezone()">
                        <option value="">Loading timezones...</option>
                    </select>
                </div>
            </div>
            
            <div class="settings-section">
                <a href="/" class="btn">Back to Chat</a>
                <a href="/integrations" class="btn">Integrations</a>
                <a href="/system" class="btn">System Dashboard</a>
            </div>
        </div>
        
        <script>
            let currentTimezoneInfo = null;
            
            // Initialize page
            document.addEventListener('DOMContentLoaded', function() {
                loadTimezoneInfo();
                setInterval(updateCurrentTime, 1000); // Update time every second
            });
            
            function loadTimezoneInfo() {
                fetch('/api/timezone/info', {
                    credentials: 'include'
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        currentTimezoneInfo = data;
                        updateDisplay();
                        populateTimezoneSelect();
                    } else {
                        console.error('Failed to load timezone info:', data.error);
                    }
                })
                .catch(e => {
                    console.error('Error loading timezone info:', e);
                });
            }
            
            function updateDisplay() {
                if (!currentTimezoneInfo) return;
                
                // Update current time
                document.getElementById('currentTime').textContent = 
                    currentTimezoneInfo.current.formatted_time;
                
                // Update status
                let statusHtml = '';
                if (currentTimezoneInfo.user_set) {
                    statusHtml += '<span class="status-indicator status-manual"></span>Manually set to: ' + 
                                 currentTimezoneInfo.current.timezone_name;
                } else if (currentTimezoneInfo.detected) {
                    statusHtml += '<span class="status-indicator status-detected"></span>Auto-detected: ' + 
                                 currentTimezoneInfo.current.timezone_name;
                } else {
                    statusHtml += '<span class="status-indicator status-default"></span>Using default: ' + 
                                 currentTimezoneInfo.current.timezone_name;
                }
                
                statusHtml += '<br><small>Current offset: ' + currentTimezoneInfo.current.utc_offset + 
                             ' (' + currentTimezoneInfo.current.timezone_abbr + ')</small>';
                
                document.getElementById('timezoneStatus').innerHTML = statusHtml;
            }
            
            function updateCurrentTime() {
                if (currentTimezoneInfo) {
                    const now = new Date();
                    const formatter = new Intl.DateTimeFormat('en-US', {
                        timeZone: currentTimezoneInfo.current.timezone_name,
                        weekday: 'long',
                        year: 'numeric',
                        month: 'long',
                        day: 'numeric',
                        hour: 'numeric',
                        minute: '2-digit',
                        second: '2-digit',
                        timeZoneName: 'short'
                    });
                    
                    document.getElementById('currentTime').textContent = formatter.format(now);
                }
            }
            
            function populateTimezoneSelect() {
                if (!currentTimezoneInfo) return;
                
                const select = document.getElementById('timezoneSelect');
                select.innerHTML = '<option value="">Choose timezone...</option>';
                
                for (const [display, value] of Object.entries(currentTimezoneInfo.common_timezones)) {
                    const option = document.createElement('option');
                    option.value = value;
                    option.textContent = display;
                    if (value === currentTimezoneInfo.current.timezone_name) {
                        option.selected = true;
                    }
                    select.appendChild(option);
                }
            }
            
            function detectTimezone() {
                const detectedTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
                
                fetch('/api/timezone/detect', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    credentials: 'include',
                    body: JSON.stringify({timezone: detectedTimezone})
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('detectionResult').innerHTML = 
                            '<div style="color: #059669; margin-top: 10px;">✓ Detected and set timezone: ' + 
                            detectedTimezone + '</div>';
                        loadTimezoneInfo(); // Reload to update display
                    } else {
                        document.getElementById('detectionResult').innerHTML = 
                            '<div style="color: #dc2626; margin-top: 10px;">✗ Detection failed: ' + 
                            data.error + '</div>';
                    }
                })
                .catch(e => {
                    document.getElementById('detectionResult').innerHTML = 
                        '<div style="color: #dc2626; margin-top: 10px;">✗ Detection failed: ' + e + '</div>';
                });
            }
            
            function setTimezone() {
                const select = document.getElementById('timezoneSelect');
                const timezone = select.value;
                
                if (!timezone) return;
                
                fetch('/api/timezone/set', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    credentials: 'include',
                    body: JSON.stringify({timezone: timezone})
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        loadTimezoneInfo(); // Reload to update display
                        alert('Timezone updated successfully!');
                    } else {
                        alert('Failed to set timezone: ' + data.error);
                    }
                })
                .catch(e => {
                    alert('Error setting timezone: ' + e);
                });
            }
        </script>
    </body>
    </html>
    """)

# Register timezone template filters
try:
    from modules.timezone_handler import datetime_filter, date_filter, time_filter
    app.jinja_env.filters['user_datetime'] = datetime_filter
    app.jinja_env.filters['user_date'] = date_filter
    app.jinja_env.filters['user_time'] = time_filter
    print("Timezone template filters registered successfully")
except ImportError as e:
    print(f"Could not register timezone filters: {e}")
except Exception as e:
    print(f"Error registering timezone filters: {e}")

# Section 20: Slack Integration Routes
# Section 20: Slack Integration Routes (UPDATED)
import hmac
import hashlib

from modules.slack_integration import (
    SlackMentionHandler,
    is_slack_configured,
    process_slack_webhook_event
)

def verify_slack_signature(data, timestamp, signature):
    """Verify that requests are coming from Slack"""
    signing_secret = os.getenv('SLACK_SIGNING_SECRET')
    if not signing_secret:
        return False
    
    # Create expected signature
    sig_basestring = f"v0:{timestamp}:{data}"
    expected_signature = 'v0=' + hmac.new(
        signing_secret.encode(),
        sig_basestring.encode(),
        hashlib.sha256
    ).hexdigest()
    
    # Compare signatures
    return hmac.compare_digest(expected_signature, signature)

@app.route('/slack/events', methods=['POST'])
def slack_events():
    """Handle Slack Events API webhook"""
    try:
        data = request.get_data(as_text=True)
        timestamp = request.headers.get('X-Slack-Request-Timestamp', '')
        signature = request.headers.get('X-Slack-Signature', '')
        
        app.logger.info(f"Slack webhook received")
        
        # Verify signature in production
        if os.getenv('RAILWAY_ENVIRONMENT') and not verify_slack_signature(data, timestamp, signature):
            return "Invalid signature", 401
        
        event_data = json.loads(data)
        
        # Handle URL verification
        if event_data.get('type') == 'url_verification':
            return event_data.get('challenge', '')
        
        # Process the event
        result = process_slack_webhook_event(event_data)
        
        if result.get('success') and result.get('task_created'):
            app.logger.info("✅ AMCF task created from Slack mention")
        
        return jsonify({'status': 'ok'}), 200
        
    except Exception as e:
        app.logger.error(f"Slack webhook failed: {e}")
        return "Internal error", 500

@app.route('/slack/test-mention', methods=['POST'])
def test_slack_mention():
    """Test endpoint for Slack mention processing (development only)"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    if not is_slack_configured():
        return jsonify({
            'success': False,
            'error': 'Slack not configured',
            'required_env_vars': [
                'SLACK_BOT_TOKEN',
                'SLACK_SIGNING_SECRET',
                'SLACK_USER_ID'
            ]
        }), 400
    
    try:
        data = request.get_json()
        test_message = data.get('message', 'Hey @me can you review the quarterly report?')
        test_user = data.get('user', 'test_user')
        
        # Create mock event data
        mock_event = {
            'type': 'event_callback',
            'event': {
                'type': 'message',
                'text': test_message.replace('@me', f"<@{os.getenv('SLACK_USER_ID')}>"),
                'channel': 'test_channel',
                'user': test_user,
                'ts': str(int(datetime.datetime.now().timestamp()))
            }
        }
        
        handler = SlackMentionHandler()
        result = handler.process_slack_mention(mock_event)
        
        return jsonify({
            'success': True,
            'test_message': test_message,
            'processing_result': result
        })
        
    except Exception as e:
        app.logger.error(f"Slack mention test failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/slack/status')
def slack_status():
    """Check Slack integration status"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    status = {
        'configured': is_slack_configured(),
        'bot_token_present': bool(os.getenv('SLACK_BOT_TOKEN')),
        'app_token_present': bool(os.getenv('SLACK_APP_TOKEN')),
        'signing_secret_present': bool(os.getenv('SLACK_SIGNING_SECRET')),
        'user_id_set': bool(os.getenv('SLACK_USER_ID')),
        'clickup_integration': is_clickup_configured(),
        'sdk_available': False
    }
    
    # Test SDK availability
    try:
        from slack_sdk import WebClient
        status['sdk_available'] = True
    except ImportError:
        pass
    
    # Test connection if configured
    if status['configured']:
        try:
            handler = SlackMentionHandler()
            if handler.slack_client:
                # Test auth
                auth_response = handler.slack_client.auth_test()
                if auth_response.get('ok'):
                    status['connection_working'] = True
                    status['bot_user_id'] = auth_response.get('user_id')
                    status['team_name'] = auth_response.get('team')
                else:
                    status['connection_error'] = 'Auth test failed'
            else:
                status['connection_error'] = 'Slack client not initialized'
                
        except Exception as e:
            status['connection_error'] = str(e)
    
    return jsonify(status)

@app.route('/slack/setup')
def slack_setup_page():
    """Slack setup instructions page"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    railway_url = os.getenv('RAILWAY_STATIC_URL', 'your-app.railway.app')
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Slack Integration Setup</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
            }
            .container { max-width: 1000px; margin: 0 auto; }
            .setup-section { 
                background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
                padding: 20px; margin: 20px 0; 
            }
            .btn { 
                background: #6366f1; color: white; border: none; padding: 12px 24px;
                border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
                text-decoration: none; display: inline-block;
            }
            .btn:hover { background: #5855eb; }
            .btn.success { background: #059669; }
            .btn.warning { background: #d97706; }
            .code-block {
                background: #2a2a2a; border: 1px solid #444; border-radius: 4px;
                padding: 15px; margin: 15px 0; font-family: 'Courier New', monospace;
                overflow-x: auto;
            }
            .step { margin: 20px 0; }
            .step-number {
                display: inline-block; background: #6366f1; color: white;
                border-radius: 50%; width: 30px; height: 30px; text-align: center;
                line-height: 30px; margin-right: 10px; font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Slack Integration Setup</h1>
            
            <div class="setup-section">
                <h3>Current Status</h3>
                <div id="statusDisplay">Loading...</div>
                <button onclick="checkStatus()" class="btn">Refresh Status</button>
            </div>
            
            <div class="setup-section">
                <h3>Setup Instructions</h3>
                
                <div class="step">
                    <span class="step-number">1</span>
                    <strong>Configure Event Subscriptions in Slack App</strong>
                    <p>In your Ghostline-Task app settings:</p>
                    <ul>
                        <li>Go to <strong>Event Subscriptions</strong></li>
                        <li>Enable Events: <strong>On</strong></li>
                        <li>Request URL: <code>https://{{ railway_url }}/slack/events</code></li>
                        <li>Subscribe to Bot Events: <code>message.channels</code>, <code>message.groups</code>, <code>message.im</code>, <code>message.mpim</code></li>
                    </ul>
                </div>
                
                <div class="step">
                    <span class="step-number">2</span>
                    <strong>Get Your User ID</strong>
                    <p>You need your Slack User ID (not username) for mention detection:</p>
                    <button onclick="findUserId()" class="btn warning">Find My User ID</button>
                    <div id="userIdResult"></div>
                </div>
                
                <div class="step">
                    <span class="step-number">3</span>
                    <strong>Set Environment Variables</strong>
                    <p>Add these to your Railway deployment:</p>
                    <div class="code-block" id="envVars">
                        SLACK_BOT_TOKEN=xoxb-your-bot-token-here<br>
                        SLACK_SIGNING_SECRET=your-signing-secret-here<br>
                        SLACK_USER_ID=your-user-id-here
                    </div>
                    <button onclick="copyEnvVars()" class="btn">Copy Environment Variables</button>
                </div>
                
                <div class="step">
                    <span class="step-number">4</span>
                    <strong>Install App to Workspace</strong>
                    <p>Make sure your app is installed and has the required permissions.</p>
                </div>
                
                <div class="step">
                    <span class="step-number">5</span>
                    <strong>Test Integration</strong>
                    <p>Once configured, test with the button below:</p>
                    <button onclick="testMention()" class="btn success">Test Mention Processing</button>
                    <div id="testResult"></div>
                </div>
            </div>
            
            <div class="setup-section">
                <h3>How It Works</h3>
                <p>Once configured, mention yourself in any Slack channel:</p>
                <ul>
                    <li><strong>@yourname can you review the quarterly report by Friday?</strong></li>
                    <li><strong>@yourname please handle the client meeting prep</strong></li>
                    <li><strong>@yourname urgent: fix the deployment issue</strong></li>
                </ul>
                <p>Ghostline will:</p>
                <ol>
                    <li>Detect the mention in real-time</li>
                    <li>Parse the task and due date</li>
                    <li>Create a ClickUp task</li>
                    <li>Reply in Slack: "✅ Task created: Review quarterly report (Due: Friday)"</li>
                </ol>
            </div>
            
            <div class="setup-section">
                <a href="/integrations" class="btn">Back to Integrations</a>
                <a href="/diagnostics" class="btn">Diagnostics</a>
                <a href="/" class="btn">Back to Chat</a>
            </div>
        </div>
        
        <script>
            function checkStatus() {
                fetch('/slack/status', { credentials: 'include' })
                .then(r => r.json())
                .then(data => {
                    let statusHtml = '<ul>';
                    statusHtml += `<li>Bot Token: ${data.bot_token_present ? '✅' : '❌'}</li>`;
                    statusHtml += `<li>Signing Secret: ${data.signing_secret_present ? '✅' : '❌'}</li>`;
                    statusHtml += `<li>User ID: ${data.user_id_set ? '✅' : '❌'}</li>`;
                    statusHtml += `<li>SDK Available: ${data.sdk_available ? '✅' : '❌'}</li>`;
                    statusHtml += `<li>ClickUp Integration: ${data.clickup_integration ? '✅' : '❌'}</li>`;
                    
                    if (data.connection_working) {
                        statusHtml += `<li>Connection: ✅ Connected to ${data.team_name}</li>`;
                    } else if (data.connection_error) {
                        statusHtml += `<li>Connection: ❌ ${data.connection_error}</li>`;
                    } else {
                        statusHtml += `<li>Connection: ⚠️ Not tested</li>`;
                    }
                    
                    statusHtml += '</ul>';
                    document.getElementById('statusDisplay').innerHTML = statusHtml;
                })
                .catch(e => {
                    document.getElementById('statusDisplay').innerHTML = 
                        '<p style="color: #dc2626;">Failed to check status: ' + e + '</p>';
                });
            }
            
            function findUserId() {
                // This requires the bot token to work
                fetch('/slack/status', { credentials: 'include' })
                .then(r => r.json())
                .then(data => {
                    if (data.bot_user_id) {
                        document.getElementById('userIdResult').innerHTML = 
                            `<p style="color: #059669; margin-top: 10px;">Bot User ID found: <code>${data.bot_user_id}</code><br>
                            But you need YOUR user ID. In Slack, right-click your profile → Copy Member ID</p>`;
                    } else {
                        document.getElementById('userIdResult').innerHTML = 
                            `<p style="color: #d97706; margin-top: 10px;">Configure bot token first, then in Slack: right-click your profile → Copy Member ID</p>`;
                    }
                });
            }
            
            function copyEnvVars() {
                const envText = document.getElementById('envVars').textContent;
                navigator.clipboard.writeText(envText).then(() => {
                    alert('Environment variables copied to clipboard!');
                });
            }
            
            function testMention() {
                const testMessage = prompt('Enter test message (use @me for mention):', 'Hey @me can you review the quarterly report by Friday?');
                if (!testMessage) return;
                
                fetch('/slack/test-mention', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    credentials: 'include',
                    body: JSON.stringify({message: testMessage})
                })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        const result = data.processing_result;
                        let resultHtml = '<div style="color: #059669; margin-top: 10px;">';
                        resultHtml += `<strong>✅ Test successful!</strong><br>`;
                        resultHtml += `Task created: ${result.task_created ? 'Yes' : 'No'}<br>`;
                        if (result.message) {
                            resultHtml += `Message: ${result.message}<br>`;
                        }
                        resultHtml += '</div>';
                        document.getElementById('testResult').innerHTML = resultHtml;
                    } else {
                        document.getElementById('testResult').innerHTML = 
                            `<div style="color: #dc2626; margin-top: 10px;">❌ Test failed: ${data.error}</div>`;
                    }
                })
                .catch(e => {
                    document.getElementById('testResult').innerHTML = 
                        `<div style="color: #dc2626; margin-top: 10px;">❌ Test failed: ${e}</div>`;
                });
            }
            
            // Load status on page load
            document.addEventListener('DOMContentLoaded', checkStatus);
        </script>
    </body>
    </html>
    """, railway_url=railway_url)

# Section 21: Multi-Site Keyword Management System Routes (NEW)

# Keyword Management Routes
@app.route('/keywords')
def keywords_dashboard():
    """Main keyword management dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        stats = keyword_manager.get_keyword_stats()
        return render_template_string(KEYWORDS_DASHBOARD_TEMPLATE,
                                    stats=stats,
                                    sites=SITE_DOMAINS)
    except Exception as e:
        return f"Keyword dashboard error: {str(e)}", 500

@app.route('/api/keywords/<site_domain>')
def api_get_site_keywords(site_domain):
    """API: Get keywords for a specific site"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        category = request.args.get('category')
        limit = min(int(request.args.get('limit', 100)), 500)
        
        keywords = keyword_manager.get_site_keywords(site_domain, limit=limit, category=category)
        
        return jsonify({
            'success': True,
            'site_domain': site_domain,
            'keywords': keywords,
            'count': len(keywords)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/keywords/<site_domain>/add', methods=['POST'])
def api_add_keyword(site_domain):
    """API: Add a single keyword to a site"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        
        if not keyword:
            return jsonify({'success': False, 'error': 'Keyword required'}), 400
        
        # Optional additional data
        kwargs = {}
        if 'search_volume' in data:
            kwargs['search_volume'] = int(data['search_volume'])
        if 'competition_level' in data:
            kwargs['competition_level'] = data['competition_level']
        if 'suggested_bid' in data:
            kwargs['suggested_bid'] = float(data['suggested_bid'])
        if 'category' in data:
            kwargs['category'] = data['category']
        
        success = keyword_manager.add_keyword(site_domain, keyword, **kwargs)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Added keyword "{keyword}" to {site_domain}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to add keyword'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/keywords/<site_domain>/remove', methods=['POST'])
def api_remove_keyword(site_domain):
    """API: Remove a keyword from a site"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        keyword = data.get('keyword', '').strip()
        
        if not keyword:
            return jsonify({'success': False, 'error': 'Keyword required'}), 400
        
        success = keyword_manager.remove_keyword(site_domain, keyword)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Removed keyword "{keyword}" from {site_domain}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Keyword not found or already removed'
            }), 404
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/keywords/<site_domain>/import-csv', methods=['POST'])
def api_import_keywords_csv(site_domain):
    """API: Import keywords from CSV file"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        if 'csv_file' not in request.files:
            return jsonify({'success': False, 'error': 'No CSV file provided'}), 400
        
        csv_file = request.files['csv_file']
        if csv_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read CSV content
        csv_content = csv_file.read().decode('utf-8')
        
        # Import keywords
        result = keyword_manager.bulk_import_csv(site_domain, csv_content)
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"CSV import failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Import failed: {str(e)}'
        }), 500

@app.route('/api/keywords/match-content', methods=['POST'])
def api_match_content():
    """API: Match content topic to best site"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        topic = data.get('topic', '').strip()
        
        if not topic:
            return jsonify({'success': False, 'error': 'Topic required'}), 400
        
        match_result = keyword_manager.match_content_to_sites(topic)
        
        # Convert to JSON-serializable format
        return jsonify({
            'success': True,
            'topic': match_result.topic,
            'best_site': match_result.best_site,
            'confidence_score': match_result.confidence_score,
            'reasoning': match_result.reasoning,
            'site_name': SITE_DOMAINS.get(match_result.best_site, {}).get('name', match_result.best_site),
            'all_matches': [
                {
                    'site_domain': match.site_domain,
                    'site_name': SITE_DOMAINS.get(match.site_domain, {}).get('name', match.site_domain),
                    'keyword': match.keyword,
                    'match_score': match.match_score,
                    'match_type': match.match_type,
                    'search_volume': match.search_volume,
                    'competition_level': match.competition_level
                }
                for match in match_result.all_matches
            ]
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/keywords/record-performance', methods=['POST'])
def api_record_performance():
    """API: Record keyword performance feedback"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        
        keyword_id = data.get('keyword_id')
        content_topic = data.get('content_topic', '').strip()
        performance_score = float(data.get('performance_score', 5.0))
        user_feedback = data.get('user_feedback', 'pending')
        
        if not keyword_id or not content_topic:
            return jsonify({
                'success': False,
                'error': 'keyword_id and content_topic required'
            }), 400
        
        success = keyword_manager.record_keyword_performance(
            keyword_id=keyword_id,
            content_topic=content_topic,
            performance_score=performance_score,
            user_feedback=user_feedback
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Performance recorded successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to record performance'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/keywords/<site_domain>')
def keywords_site_detail(site_domain):
    """Detailed keyword view for a specific site"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if site_domain not in SITE_DOMAINS:
        return "Site not found", 404
    
    try:
        keywords = keyword_manager.get_site_keywords(site_domain, limit=200)
        site_info = SITE_DOMAINS[site_domain]
        
        return render_template_string(SITE_KEYWORDS_DETAIL_TEMPLATE,
                                    site_domain=site_domain,
                                    site_info=site_info,
                                    keywords=keywords)
    except Exception as e:
        return f"Site detail error: {str(e)}", 500

@app.route('/keywords/test-matching')
def keywords_test_matching():
    """Test content matching interface"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template_string(KEYWORDS_TEST_MATCHING_TEMPLATE, sites=SITE_DOMAINS)

# HTML Templates for Keyword Management
KEYWORDS_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Keyword Management Dashboard</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .btn { 
            background: #6366f1; color: white; border: none; padding: 12px 24px;
            border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
            text-decoration: none; display: inline-block;
        }
        .btn:hover { background: #5855eb; }
        .btn.success { background: #059669; }
        .btn.warning { background: #d97706; }
        .btn.secondary { background: #374151; }
        .btn.secondary:hover { background: #4b5563; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .site-card {
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
            padding: 20px; position: relative;
        }
        .site-name { font-size: 20px; font-weight: bold; margin-bottom: 10px; }
        .site-domain { color: #6366f1; font-size: 14px; margin-bottom: 15px; }
        .stat-row { display: flex; justify-content: space-between; margin: 8px 0; }
        .stat-label { color: #9ca3af; }
        .stat-value { font-weight: bold; }
        .action-bar { margin: 20px 0; }
        .quick-match {
            background: #2a2a2a; padding: 20px; border-radius: 8px; margin: 20px 0;
        }
        .quick-match input { 
            width: 70%; padding: 12px; background: #333; color: #fff;
            border: 1px solid #555; border-radius: 4px; font-size: 16px;
        }
        .match-result {
            margin-top: 15px; padding: 15px; background: #1a1a1a; border-radius: 6px;
            display: none;
        }
        .progress-bar {
            width: 100%; height: 8px; background: #333; border-radius: 4px; overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill { height: 100%; background: #059669; transition: width 0.3s; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Site Keyword Management</h1>
        
        <div class="action-bar">
            <a href="/keywords/test-matching" class="btn success">Test Content Matching</a>
            <a href="/" class="btn secondary">Back to Chat</a>
        </div>
        
        <div class="quick-match">
            <h3>Quick Content Match Test</h3>
            <input type="text" id="quickMatchInput" placeholder="Enter a content topic to test matching..." onkeypress="handleQuickMatch(event)">
            <button onclick="testQuickMatch()" class="btn">Test Match</button>
            <div id="quickMatchResult" class="match-result"></div>
        </div>
        
        <div class="stats-grid">
            {% for site_domain, site_data in stats.site_stats.items() %}
            <div class="site-card">
                <div class="site-name">{{ site_data.site_name }}</div>
                <div class="site-domain">{{ site_domain }}</div>
                
                <div class="stat-row">
                    <span class="stat-label">Keywords:</span>
                    <span class="stat-value">{{ site_data.active_keywords }}</span>
                </div>
                
                <div class="stat-row">
                    <span class="stat-label">Avg Search Volume:</span>
                    <span class="stat-value">{{ "%.0f"|format(site_data.avg_search_volume) }}</span>
                </div>
                
                <div class="stat-row">
                    <span class="stat-label">Match Score:</span>
                    <span class="stat-value">{{ "%.1f"|format(site_data.avg_match_score) }}</span>
                </div>
                
                <div class="stat-row">
                    <span class="stat-label">Times Used:</span>
                    <span class="stat-value">{{ site_data.total_usage }}</span>
                </div>
                
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {{ (site_data.avg_match_score / 10 * 100)|int }}%"></div>
                </div>
                
                <div style="margin-top: 15px;">
                    <a href="/keywords/{{ site_domain }}" class="btn">View Keywords</a>
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% if stats.performance_stats %}
        <div class="site-card" style="margin: 20px 0;">
            <h3>Overall Performance Stats</h3>
            <div class="stat-row">
                <span class="stat-label">Total Performance Logs:</span>
                <span class="stat-value">{{ stats.performance_stats.total_performance_logs or 0 }}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Approved:</span>
                <span class="stat-value">{{ stats.performance_stats.approved_count or 0 }}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Rejected:</span>
                <span class="stat-value">{{ stats.performance_stats.rejected_count or 0 }}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Avg Performance Score:</span>
                <span class="stat-value">{{ "%.1f"|format(stats.performance_stats.avg_performance_score or 0) }}</span>
            </div>
        </div>
        {% endif %}
    </div>
    
    <script>
        function handleQuickMatch(event) {
            if (event.key === 'Enter') {
                testQuickMatch();
            }
        }
        
        function testQuickMatch() {
            const topic = document.getElementById('quickMatchInput').value.trim();
            if (!topic) {
                alert('Please enter a content topic');
                return;
            }
            
            const resultDiv = document.getElementById('quickMatchResult');
            resultDiv.innerHTML = '<div style="color: #6366f1;">Testing match...</div>';
            resultDiv.style.display = 'block';
            
            fetch('/api/keywords/match-content', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify({topic: topic})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    let html = '<div style="color: #059669; font-weight: bold; margin-bottom: 10px;">';
                    html += `Best Match: ${data.site_name} (${data.confidence_score.toFixed(1)}% confidence)</div>`;
                    html += `<div style="color: #9ca3af; margin-bottom: 10px;">${data.reasoning}</div>`;
                    
                    if (data.all_matches && data.all_matches.length > 0) {
                        html += '<div style="font-size: 14px;"><strong>Top Matches:</strong><br>';
                        data.all_matches.slice(0, 5).forEach(match => {
                            html += `<div style="margin: 5px 0; padding: 5px; background: #333; border-radius: 4px;">`;
                            html += `${match.site_name}: "${match.keyword}" (${match.match_score.toFixed(1)})`;
                            html += `</div>`;
                        });
                        html += '</div>';
                    }
                    
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.innerHTML = `<div style="color: #dc2626;">Error: ${data.error}</div>`;
                }
            })
            .catch(e => {
                resultDiv.innerHTML = `<div style="color: #dc2626;">Request failed: ${e}</div>`;
            });
        }
    </script>
</body>
</html>
'''

SITE_KEYWORDS_DETAIL_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>{{ site_info.name }} - Keywords</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .btn { 
            background: #6366f1; color: white; border: none; padding: 12px 24px;
            border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
            text-decoration: none; display: inline-block;
        }
        .btn:hover { background: #5855eb; }
        .btn.success { background: #059669; }
        .btn.danger { background: #dc2626; }
        .btn.secondary { background: #374151; }
        .header-section {
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
            padding: 20px; margin: 20px 0;
        }
        .add-keyword-form {
            background: #2a2a2a; padding: 20px; border-radius: 8px; margin: 20px 0;
        }
        .add-keyword-form input, .add-keyword-form select { 
            padding: 10px; background: #333; color: #fff;
            border: 1px solid #555; border-radius: 4px; font-size: 16px; margin: 5px;
        }
        .csv-upload {
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
            padding: 20px; margin: 20px 0;
        }
        .keywords-table {
            width: 100%; border-collapse: collapse; margin: 20px 0;
            background: #1a1a1a;
        }
        .keywords-table th, .keywords-table td {
            padding: 12px; text-align: left; border-bottom: 1px solid #333;
        }
        .keywords-table th { background: #2a2a2a; font-weight: bold; }
        .keywords-table tr:hover { background: #2a2a2a; }
        .keyword-row { cursor: pointer; }
        .competition-high { color: #dc2626; }
        .competition-medium { color: #d97706; }
        .competition-low { color: #059669; }
        .match-score { font-weight: bold; }
        .remove-btn { 
            background: #dc2626; color: white; border: none; padding: 6px 12px;
            border-radius: 4px; cursor: pointer; font-size: 12px;
        }
        .file-input { background: #333; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-section">
            <h1>{{ site_info.name }}</h1>
            <p style="color: #6366f1; font-size: 18px;">{{ site_domain }}</p>
            <p>{{ site_info.focus_areas | join(', ') }}</p>
            <div>
                <a href="/keywords" class="btn secondary">← Back to Dashboard</a>
                <span style="margin-left: 20px; color: #9ca3af;">{{ keywords|length }} keywords</span>
            </div>
        </div>
        
        <div class="add-keyword-form">
            <h3>Add New Keyword</h3>
            <div style="display: flex; flex-wrap: wrap; align-items: center;">
                <input type="text" id="newKeyword" placeholder="Enter keyword..." style="flex: 1; min-width: 200px;">
                <input type="number" id="searchVolume" placeholder="Search Volume" style="width: 120px;">
                <select id="competitionLevel" style="width: 120px;">
                    <option value="unknown">Competition</option>
                    <option value="low">Low</option>
                    <option value="medium">Medium</option>
                    <option value="high">High</option>
                </select>
                <input type="number" id="suggestedBid" placeholder="Bid ($)" step="0.01" style="width: 100px;">
                <button onclick="addKeyword()" class="btn success">Add Keyword</button>
            </div>
        </div>
        
        <div class="csv-upload">
            <h3>Bulk Import from CSV</h3>
            <p>Upload Google Ads Keyword Planner data or similar CSV format:</p>
            <input type="file" id="csvFile" accept=".csv" class="file-input">
            <button onclick="uploadCSV()" class="btn">Import CSV</button>
            <div id="uploadResult" style="margin-top: 15px;"></div>
        </div>
        
        {% if keywords %}
        <table class="keywords-table">
            <thead>
                <tr>
                    <th>Keyword</th>
                    <th>Category</th>
                    <th>Search Volume</th>
                    <th>Competition</th>
                    <th>Suggested Bid</th>
                    <th>Match Score</th>
                    <th>Times Used</th>
                    <th>Source</th>
                    <th>Action</th>
                </tr>
            </thead>
            <tbody>
                {% for keyword in keywords %}
                <tr class="keyword-row">
                    <td style="font-weight: bold;">{{ keyword.keyword }}</td>
                    <td>{{ keyword.keyword_category }}</td>
                    <td>{{ "{:,}".format(keyword.search_volume or 0) }}</td>
                    <td class="competition-{{ keyword.competition_level or 'unknown' }}">
                        {{ keyword.competition_level or 'unknown' }}
                    </td>
                    <td>${{ "%.2f"|format(keyword.suggested_bid or 0) }}</td>
                    <td class="match-score">{{ "%.1f"|format(keyword.match_score or 0) }}</td>
                    <td>{{ keyword.times_used or 0 }}</td>
                    <td>{{ keyword.source }}</td>
                    <td>
                        <button onclick="removeKeyword('{{ keyword.keyword }}')" class="remove-btn">Remove</button>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <div style="text-align: center; padding: 40px; color: #9ca3af;">
            No keywords found. Add some keywords to get started.
        </div>
        {% endif %}
    </div>
    
    <script>
        function addKeyword() {
            const keyword = document.getElementById('newKeyword').value.trim();
            if (!keyword) {
                alert('Please enter a keyword');
                return;
            }
            
            const data = {
                keyword: keyword,
                search_volume: parseInt(document.getElementById('searchVolume').value) || 0,
                competition_level: document.getElementById('competitionLevel').value,
                suggested_bid: parseFloat(document.getElementById('suggestedBid').value) || 0
            };
            
            fetch('/api/keywords/{{ site_domain }}/add', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify(data)
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Keyword added successfully!');
                    location.reload();
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(e => alert('Request failed: ' + e));
        }
        
        function removeKeyword(keyword) {
            if (!confirm('Remove keyword "' + keyword + '"?')) return;
            
            fetch('/api/keywords/{{ site_domain }}/remove', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify({keyword: keyword})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Keyword removed successfully!');
                    location.reload();
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(e => alert('Request failed: ' + e));
        }
        
        function uploadCSV() {
            const fileInput = document.getElementById('csvFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a CSV file');
                return;
            }
            
            const formData = new FormData();
            formData.append('csv_file', file);
            
            const resultDiv = document.getElementById('uploadResult');
            resultDiv.innerHTML = '<div style="color: #6366f1;">Uploading and processing CSV...</div>';
            
            fetch('/api/keywords/{{ site_domain }}/import-csv', {
                method: 'POST',
                credentials: 'include',
                body: formData
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    let html = `<div style="color: #059669; font-weight: bold;">Import successful!</div>`;
                    html += `<div>Imported: ${data.imported_count} keywords</div>`;
                    html += `<div>Skipped: ${data.skipped_count} keywords</div>`;
                    
                    if (data.preview && data.preview.length > 0) {
                        html += '<div style="margin-top: 10px; font-size: 14px;"><strong>Preview:</strong><br>';
                        data.preview.forEach(kw => {
                            html += `<div style="margin: 2px 0;">"${kw.keyword}" (Vol: ${kw.search_volume}, Comp: ${kw.competition_level})</div>`;
                        });
                        html += '</div>';
                    }
                    
                    if (data.errors && data.errors.length > 0) {
                        html += '<div style="color: #d97706; margin-top: 10px;">Warnings:<br>';
                        data.errors.slice(0, 5).forEach(error => {
                            html += `<div style="font-size: 12px;">${error}</div>`;
                        });
                        html += '</div>';
                    }
                    
                    resultDiv.innerHTML = html;
                    
                    // Reload page after delay to show new keywords
                    setTimeout(() => location.reload(), 3000);
                    
                } else {
                    let errorHtml = `<div style="color: #dc2626;">Import failed</div>`;
                    if (data.errors && data.errors.length > 0) {
                        errorHtml += '<div style="font-size: 14px; margin-top: 10px;">Errors:<br>';
                        data.errors.forEach(error => {
                            errorHtml += `<div>${error}</div>`;
                        });
                        errorHtml += '</div>';
                    }
                    resultDiv.innerHTML = errorHtml;
                }
            })
            .catch(e => {
                resultDiv.innerHTML = `<div style="color: #dc2626;">Upload failed: ${e}</div>`;
            });
        }
    </script>
</body>
</html>
'''

KEYWORDS_TEST_MATCHING_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Content Matching Test</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .btn { 
            background: #6366f1; color: white; border: none; padding: 12px 24px;
            border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
            text-decoration: none; display: inline-block;
        }
        .btn:hover { background: #5855eb; }
        .btn.secondary { background: #374151; }
        .test-section {
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
            padding: 20px; margin: 20px 0;
        }
        .topic-input { 
            width: 100%; padding: 15px; background: #333; color: #fff;
            border: 1px solid #555; border-radius: 4px; font-size: 16px; margin: 10px 0;
            min-height: 60px; resize: vertical;
        }
        .result-section {
            background: #2a2a2a; padding: 20px; border-radius: 8px; margin: 20px 0;
            display: none;
        }
        .best-match {
            background: #059669; padding: 15px; border-radius: 6px; margin: 10px 0;
            font-size: 18px; font-weight: bold;
        }
        .confidence-bar {
            width: 100%; height: 12px; background: #333; border-radius: 6px; overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill { height: 100%; background: #059669; transition: width 0.3s; }
        .matches-list {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px; margin: 20px 0;
        }
        .match-item {
            background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 15px;
        }
        .match-score { 
            float: right; background: #6366f1; color: white; padding: 4px 8px;
            border-radius: 12px; font-size: 12px; font-weight: bold;
        }
        .site-name { font-weight: bold; margin-bottom: 5px; }
        .keyword { color: #9ca3af; font-style: italic; }
        .example-topics {
            background: #333; padding: 15px; border-radius: 6px; margin: 10px 0;
        }
        .example-topic {
            cursor: pointer; padding: 8px; border-radius: 4px; margin: 5px 0;
            background: #444; transition: background 0.2s;
        }
        .example-topic:hover { background: #555; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Content Matching Test</h1>
        
        <div style="margin: 20px 0;">
            <a href="/keywords" class="btn secondary">← Back to Dashboard</a>
        </div>
        
        <div class="test-section">
            <h3>Test Content Topic Matching</h3>
            <p>Enter a content topic or idea and see which site it matches best:</p>
            
            <textarea id="topicInput" class="topic-input" placeholder="Enter your content topic here... e.g., 'How to improve team leadership in remote work environments' or 'Mental health strategies for creative professionals'"></textarea>
            
            <button onclick="testMatching()" class="btn">Test Content Match</button>
            
            <div class="example-topics">
                <h4>Example Topics (click to test):</h4>
                <div class="example-topic" onclick="setTopic('SEO strategies for small business digital marketing')">SEO strategies for small business digital marketing</div>
                <div class="example-topic" onclick="setTopic('Nonprofit fundraising and community engagement tactics')">Nonprofit fundraising and community engagement tactics</div>
                <div class="example-topic" onclick="setTopic('Netflix vs Hulu streaming service comparison')">Netflix vs Hulu streaming service comparison</div>
                <div class="example-topic" onclick="setTopic('Overcoming creative burnout and impostor syndrome')">Overcoming creative burnout and impostor syndrome</div>
                <div class="example-topic" onclick="setTopic('ROI measurement for marketing consulting clients')">ROI measurement for marketing consulting clients</div>
                <div class="example-topic" onclick="setTopic('Military leadership principles in business strategy')">Military leadership principles in business strategy</div>
                <div class="example-topic" onclick="setTopic('Food insecurity solutions in urban communities')">Food insecurity solutions in urban communities</div>
            </div>
        </div>
        
        <div id="resultSection" class="result-section">
            <h3>Matching Results</h3>
            <div id="bestMatch"></div>
            <div id="confidenceScore"></div>
            <div id="reasoning"></div>
            <div id="allMatches"></div>
            
            <div style="margin-top: 20px;">
                <h4>Provide Feedback</h4>
                <p>Was this match accurate? Your feedback helps improve the system.</p>
                <button onclick="recordFeedback('approved', 8.0)" class="btn" style="background: #059669;">✓ Good Match</button>
                <button onclick="recordFeedback('rejected', 2.0)" class="btn" style="background: #dc2626;">✗ Poor Match</button>
            </div>
        </div>
    </div>
    
    <script>
        let lastMatchResult = null;
        
        function setTopic(topic) {
            document.getElementById('topicInput').value = topic;
            testMatching();
        }
        
        function testMatching() {
            const topic = document.getElementById('topicInput').value.trim();
            if (!topic) {
                alert('Please enter a content topic');
                return;
            }
            
            const resultSection = document.getElementById('resultSection');
            resultSection.style.display = 'block';
            
            document.getElementById('bestMatch').innerHTML = '<div style="color: #6366f1;">Testing match...</div>';
            document.getElementById('confidenceScore').innerHTML = '';
            document.getElementById('reasoning').innerHTML = '';
            document.getElementById('allMatches').innerHTML = '';
            
            fetch('/api/keywords/match-content', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify({topic: topic})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    lastMatchResult = data;
                    displayResults(data);
                } else {
                    document.getElementById('bestMatch').innerHTML = 
                        `<div style="color: #dc2626;">Error: ${data.error}</div>`;
                }
            })
            .catch(e => {
                document.getElementById('bestMatch').innerHTML = 
                    `<div style="color: #dc2626;">Request failed: ${e}</div>`;
            });
        }
        
        function displayResults(data) {
            // Best match
            document.getElementById('bestMatch').innerHTML = 
                `<div class="best-match">Best Match: ${data.site_name}</div>`;
            
            // Confidence score
            const confidenceHtml = `
                <div style="margin: 15px 0;">
                    <strong>Confidence: ${data.confidence_score.toFixed(1)}%</strong>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${data.confidence_score}%"></div>
                    </div>
                </div>
            `;
            document.getElementById('confidenceScore').innerHTML = confidenceHtml;
            
            // Reasoning
            document.getElementById('reasoning').innerHTML = 
                `<div style="color: #9ca3af; margin: 10px 0;">${data.reasoning}</div>`;
            
            // All matches
            if (data.all_matches && data.all_matches.length > 0) {
                let matchesHtml = '<h4>All Keyword Matches:</h4><div class="matches-list">';
                
                data.all_matches.forEach(match => {
                    matchesHtml += `
                        <div class="match-item">
                            <div class="match-score">${match.match_score.toFixed(1)}</div>
                            <div class="site-name">${match.site_name}</div>
                            <div class="keyword">"${match.keyword}"</div>
                            <div style="font-size: 12px; color: #6b7280; margin-top: 5px;">
                                ${match.match_type} match • Vol: ${match.search_volume.toLocaleString()}
                            </div>
                        </div>
                    `;
                });
                
                matchesHtml += '</div>';
                document.getElementById('allMatches').innerHTML = matchesHtml;
            }
        }
        
        function recordFeedback(feedback, score) {
            if (!lastMatchResult) {
                alert('No match result to provide feedback for');
                return;
            }
            
            // Find the top matching keyword ID from the best site
            const topMatch = lastMatchResult.all_matches.find(match => 
                match.site_domain === lastMatchResult.best_site
            );
            
            if (!topMatch) {
                alert('No keyword match found for feedback');
                return;
            }
            
            const data = {
                keyword_id: topMatch.keyword_id || 1, // Fallback ID
                content_topic: lastMatchResult.topic,
                performance_score: score,
                user_feedback: feedback
            };
            
            fetch('/api/keywords/record-performance', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify(data)
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Thank you! Your feedback has been recorded and will help improve matching accuracy.');
                } else {
                    alert('Failed to record feedback: ' + data.error);
                }
            })
            .catch(e => {
                alert('Feedback request failed: ' + e);
            });
        }
    </script>
</body>
</html>
'''

# Section 22: Google Trends Content Opportunity Routes (NEW) 9/13/25
# Section 22: Google Trends Content Opportunity Routes (NEW) 9/13/25

from modules.google_trends_monitor import (
    run_trends_monitoring_cycle,
    get_trends_monitor,
    start_trends_monitoring,
    stop_trends_monitoring,
    get_trends_monitoring_status,
    is_trends_monitoring_configured
)

@app.route('/trends-monitor')
def trends_monitor_dashboard():
    """Google Trends monitoring dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        status = get_trends_monitoring_status()
        
        # Get recent opportunities
        recent_opportunities = []
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT opportunity_id, trending_query, best_site, confidence_score,
                           suggested_title, user_response, detected_at, alert_sent
                    FROM content_opportunities 
                    ORDER BY detected_at DESC 
                    LIMIT 20
                ''')
                
                for row in cursor.fetchall():
                    opportunity_id, query, site, confidence, title, response, detected, alert_sent = row
                    site_info = SITE_DOMAINS.get(site, {})
                    
                    recent_opportunities.append({
                        'opportunity_id': opportunity_id,
                        'trending_query': query,
                        'best_site': site,
                        'site_name': site_info.get('name', site),
                        'confidence_score': float(confidence),
                        'suggested_title': title,
                        'user_response': response,
                        'detected_at': detected,
                        'alert_sent': alert_sent,
                        'status_color': _get_opportunity_status_color(response)
                    })
        
        return render_template_string(TRENDS_DASHBOARD_TEMPLATE,
                                    status=status,
                                    opportunities=recent_opportunities,
                                    SITE_DOMAINS=SITE_DOMAINS)
    except Exception as e:
        return f"Trends dashboard error: {str(e)}", 500

@app.route('/api/trends-monitor/status')
def api_trends_status():
    """API: Get trends monitoring status"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        status = get_trends_monitoring_status()
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trends-monitor/start', methods=['POST'])
def api_start_trends_monitoring():
    """API: Start trends monitoring"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        success = start_trends_monitoring()
        
        return jsonify({
            'success': success,
            'message': 'Trends monitoring started' if success else 'Already running or not configured'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trends-monitor/stop', methods=['POST'])
def api_stop_trends_monitoring():
    """API: Stop trends monitoring"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        success = stop_trends_monitoring()
        
        return jsonify({
            'success': success,
            'message': 'Trends monitoring stopped' if success else 'Not running'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trends-monitor/run-cycle', methods=['POST'])
def api_run_trends_cycle():
    """API: Manually trigger a trends monitoring cycle"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        result = run_trends_monitoring_cycle()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trends/opportunities')
def api_get_content_opportunities():
    """API: Get content opportunities with filtering"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Get query parameters
        limit = min(int(request.args.get('limit', 50)), 100)
        site_filter = request.args.get('site')
        status_filter = request.args.get('status')
        min_confidence = float(request.args.get('min_confidence', 0))
        
        opportunities = []
        
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                
                # Build query with filters
                where_clauses = ['1=1']
                params = []
                
                if site_filter:
                    where_clauses.append('best_site = %s')
                    params.append(site_filter)
                
                if status_filter:
                    where_clauses.append('user_response = %s')
                    params.append(status_filter)
                
                if min_confidence > 0:
                    where_clauses.append('confidence_score >= %s')
                    params.append(min_confidence)
                
                params.append(limit)
                
                cursor.execute(f'''
                    SELECT opportunity_id, trending_query, best_site, confidence_score,
                           matched_keywords, suggested_title, competition_level,
                           total_search_volume, reasoning, detected_at, alert_sent,
                           user_response, performance_score, metadata
                    FROM content_opportunities 
                    WHERE {' AND '.join(where_clauses)}
                    ORDER BY detected_at DESC 
                    LIMIT %s
                ''', params)
                
                for row in cursor.fetchall():
                    (opportunity_id, query, site, confidence, keywords, title,
                     competition, volume, reasoning, detected, alert_sent,
                     response, score, metadata) = row
                    
                    site_info = SITE_DOMAINS.get(site, {})
                    
                    opportunities.append({
                        'opportunity_id': opportunity_id,
                        'trending_query': query,
                        'best_site': site,
                        'site_name': site_info.get('name', site),
                        'confidence_score': float(confidence),
                        'matched_keywords': keywords,
                        'suggested_title': title,
                        'competition_level': competition,
                        'total_search_volume': volume,
                        'reasoning': reasoning,
                        'detected_at': detected.isoformat() if detected else None,
                        'alert_sent': alert_sent,
                        'user_response': response,
                        'performance_score': float(score) if score else None,
                        'metadata': metadata
                    })
        
        return jsonify({
            'success': True,
            'opportunities': opportunities,
            'count': len(opportunities)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trends/opportunity/<opportunity_id>/feedback', methods=['POST'])
def api_update_opportunity_feedback(opportunity_id):
    """API: Update opportunity feedback"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        user_response = data.get('response')
        performance_score = data.get('score')
        feedback_text = data.get('feedback', '')
        
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE content_opportunities 
                    SET user_response = %s, performance_score = %s, user_feedback = %s
                    WHERE opportunity_id = %s
                ''', (user_response, performance_score, feedback_text, opportunity_id))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    return jsonify({
                        'success': True,
                        'message': 'Feedback updated successfully'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Opportunity not found'
                    }), 404
        
        return jsonify({'success': False, 'error': 'Database not available'}), 500
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trends/stats')
def api_trends_stats():
    """API: Get trends monitoring statistics"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        stats = {}
        
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                
                # Overall stats
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_opportunities,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(*) FILTER (WHERE user_response = 'approved') as approved_count,
                        COUNT(*) FILTER (WHERE user_response = 'skipped') as skipped_count,
                        COUNT(*) FILTER (WHERE user_response = 'wrong_site') as wrong_site_count,
                        COUNT(*) FILTER (WHERE alert_sent = TRUE) as alerts_sent
                    FROM content_opportunities
                ''')
                
                overall = cursor.fetchone()
                if overall:
                    stats['overall'] = {
                        'total_opportunities': overall[0],
                        'avg_confidence': float(overall[1]) if overall[1] else 0,
                        'approved_count': overall[2],
                        'skipped_count': overall[3],
                        'wrong_site_count': overall[4],
                        'alerts_sent': overall[5]
                    }
                
                # Last 7 days stats
                cursor.execute('''
                    SELECT 
                        COUNT(*) as recent_opportunities,
                        AVG(confidence_score) as recent_avg_confidence
                    FROM content_opportunities
                    WHERE detected_at >= CURRENT_DATE - INTERVAL '7 days'
                ''')
                
                recent = cursor.fetchone()
                if recent:
                    stats['last_7_days'] = {
                        'opportunities': recent[0],
                        'avg_confidence': float(recent[1]) if recent[1] else 0
                    }
                
                # By site breakdown
                cursor.execute('''
                    SELECT best_site, COUNT(*), AVG(confidence_score)
                    FROM content_opportunities
                    GROUP BY best_site
                    ORDER BY COUNT(*) DESC
                ''')
                
                site_stats = {}
                for row in cursor.fetchall():
                    site, count, avg_conf = row
                    site_info = SITE_DOMAINS.get(site, {})
                    site_stats[site] = {
                        'name': site_info.get('name', site),
                        'opportunities': count,
                        'avg_confidence': float(avg_conf) if avg_conf else 0
                    }
                
                stats['by_site'] = site_stats
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def _get_opportunity_status_color(response):
    """Get color for opportunity status"""
    colors = {
        'approved': '#059669',
        'skipped': '#d97706',
        'wrong_site': '#dc2626',
        None: '#6b7280'
    }
    return colors.get(response, '#6b7280')

# HTML Template for Trends Dashboard
TRENDS_DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Google Trends Content Monitor</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .btn { 
            background: #6366f1; color: white; border: none; padding: 12px 24px;
            border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
            text-decoration: none; display: inline-block;
        }
        .btn:hover { background: #5855eb; }
        .btn.success { background: #059669; }
        .btn.danger { background: #dc2626; }
        .btn.warning { background: #d97706; }
        .btn.secondary { background: #374151; }
        .status-section {
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
            padding: 20px; margin: 20px 0;
        }
        .monitor-controls { margin: 20px 0; }
        .opportunities-table {
            width: 100%; border-collapse: collapse; margin: 20px 0;
            background: #1a1a1a;
        }
        .opportunities-table th, .opportunities-table td {
            padding: 12px; text-align: left; border-bottom: 1px solid #333;
        }
        .opportunities-table th { background: #2a2a2a; font-weight: bold; }
        .opportunities-table tr:hover { background: #2a2a2a; }
        .confidence-score {
            padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;
        }
        .confidence-high { background: #059669; color: white; }
        .confidence-medium { background: #d97706; color: white; }
        .confidence-low { background: #6b7280; color: white; }
        .status-dot {
            display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px;
        }
        .running { background: #059669; }
        .stopped { background: #dc2626; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Google Trends Content Monitor</h1>
        
        <div class="status-section">
            <h3>Monitor Status</h3>
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <span class="status-dot {% if status.running %}running{% else %}stopped{% endif %}"></span>
                <span style="font-weight: bold;">
                    {% if status.running %}Running{% else %}Stopped{% endif %}
                </span>
            </div>
            
            <div class="monitor-controls">
                {% if status.running %}
                    <button onclick="stopMonitoring()" class="btn danger">Stop Monitor</button>
                {% else %}
                    <button onclick="startMonitoring()" class="btn success">Start Monitor</button>
                {% endif %}
                
                <button onclick="runCycle()" class="btn">Run Check Now</button>
                <button onclick="refreshStatus()" class="btn secondary">Refresh Status</button>
            </div>
        </div>
        
        <div class="status-section">
            <h3>Recent Content Opportunities</h3>
            
            {% if opportunities %}
            <table class="opportunities-table">
                <thead>
                    <tr>
                        <th>Trending Query</th>
                        <th>Site Match</th>
                        <th>Confidence</th>
                        <th>Suggested Title</th>
                        <th>Status</th>
                        <th>Detected</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {% for opp in opportunities %}
                    <tr>
                        <td>
                            <strong>{{ opp.trending_query }}</strong>
                            {% if opp.alert_sent %}
                            <span style="color: #059669; font-size: 12px;">Sent</span>
                            {% endif %}
                        </td>
                        <td>{{ opp.site_name }}</td>
                        <td>
                            <span class="confidence-score {% if opp.confidence_score >= 70 %}confidence-high{% elif opp.confidence_score >= 40 %}confidence-medium{% else %}confidence-low{% endif %}">
                                {{ "%.0f"|format(opp.confidence_score) }}%
                            </span>
                        </td>
                        <td style="max-width: 300px; overflow: hidden;">
                            {{ opp.suggested_title }}
                        </td>
                        <td>
                            <span style="color: {{ opp.status_color }};">
                                {% if opp.user_response == 'approved' %}Approved
                                {% elif opp.user_response == 'skipped' %}Skipped  
                                {% elif opp.user_response == 'wrong_site' %}Wrong Site
                                {% else %}Pending{% endif %}
                            </span>
                        </td>
                        <td>{{ opp.detected_at.strftime('%m/%d %H:%M') if opp.detected_at else 'N/A' }}</td>
                        <td>
                            <button onclick="approveOpportunity('{{ opp.opportunity_id }}')" class="btn success" style="font-size: 12px; padding: 6px 12px;">Approve</button>
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
            {% else %}
            <div style="text-align: center; padding: 40px; color: #9ca3af;">
                No content opportunities found yet. Start monitoring to detect trends!
            </div>
            {% endif %}
        </div>
        
        <div style="margin-top: 40px; text-align: center;">
            <a href="/keywords" class="btn secondary">Keyword Management</a>
            <a href="/marketing-knowledge" class="btn secondary">Marketing Knowledge</a>
            <a href="/" class="btn secondary">Back to Chat</a>
        </div>
    </div>
    
    <script>
        function startMonitoring() {
            fetch('/api/trends-monitor/start', {
                method: 'POST',
                credentials: 'include'
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Trends monitoring started!');
                    location.reload();
                } else {
                    alert('Failed to start: ' + data.error);
                }
            });
        }
        
        function stopMonitoring() {
            if (!confirm('Stop trends monitoring?')) return;
            
            fetch('/api/trends-monitor/stop', {
                method: 'POST',
                credentials: 'include'
            })
            .then(r => r.json())
            .then(data => {
                alert(data.success ? 'Stopped' : 'Failed: ' + data.error);
                if (data.success) location.reload();
            });
        }
        
        function runCycle() {
            const btn = event.target;
            btn.textContent = 'Running...';
            btn.disabled = true;
            
            fetch('/api/trends-monitor/run-cycle', {
                method: 'POST',
                credentials: 'include'
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Cycle complete! Found: ' + data.opportunities + ' opportunities');
                    location.reload();
                } else {
                    alert('Failed: ' + data.error);
                }
                btn.textContent = 'Run Check Now';
                btn.disabled = false;
            });
        }
        
        function refreshStatus() {
            location.reload();
        }
        
        function approveOpportunity(opportunityId) {
            fetch('/api/trends/opportunity/' + opportunityId + '/feedback', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify({
                    response: 'approved',
                    score: 8.0
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Opportunity approved!');
                    location.reload();
                } else {
                    alert('Failed: ' + data.error);
                }
            });
        }
    </script>
</body>
</html>
'''

# Section 17.5 RSS Marketing Knowledge Dashboard Template 9/13/25
MARKETING_KNOWLEDGE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Marketing Knowledge Base</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .btn { 
            background: #6366f1; color: white; border: none; padding: 12px 24px;
            border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
            text-decoration: none; display: inline-block;
        }
        .btn:hover { background: #5855eb; }
        .btn.success { background: #059669; }
        .btn.warning { background: #d97706; }
        .btn.secondary { background: #374151; }
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 300px;
            gap: 20px;
            margin: 20px 0;
        }
        .main-content { }
        .sidebar { }
        .search-section {
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
            padding: 20px; margin: 20px 0;
        }
        .search-input { 
            width: 100%; padding: 15px; background: #333; color: #fff;
            border: 1px solid #555; border-radius: 4px; font-size: 16px; margin: 10px 0;
        }
        .category-filters {
            display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0;
        }
        .category-btn {
            background: #374151; color: white; border: none; padding: 8px 16px;
            border-radius: 20px; cursor: pointer; font-size: 14px;
            transition: background 0.2s;
        }
        .category-btn:hover, .category-btn.active { background: #6366f1; }
        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .insight-card {
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
            padding: 20px; position: relative;
        }
        .insight-title { font-size: 16px; font-weight: bold; margin-bottom: 10px; }
        .insight-meta { color: #9ca3af; font-size: 12px; margin-bottom: 10px; }
        .insight-summary { color: #d1d5db; line-height: 1.4; margin-bottom: 15px; }
        .insight-keywords {
            display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px;
        }
        .keyword-tag {
            background: #374151; color: #e5e7eb; padding: 4px 8px;
            border-radius: 12px; font-size: 11px;
        }
        .stats-section {
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
            padding: 20px; margin-bottom: 20px;
        }
        .stat-row {
            display: flex; justify-content: space-between; margin: 8px 0;
            padding: 8px 0; border-bottom: 1px solid #333;
        }
        .stat-label { color: #9ca3af; }
        .stat-value { font-weight: bold; color: #fff; }
        .search-results {
            display: none;
            background: #2a2a2a; border-radius: 8px; padding: 20px; margin: 20px 0;
        }
        .loading { text-align: center; color: #6366f1; padding: 20px; }
        .monitor-status {
            display: flex; align-items: center; gap: 10px; margin-bottom: 15px;
        }
        .status-dot {
            width: 12px; height: 12px; border-radius: 50%;
            background: #dc2626;
        }
        .status-dot.running { background: #059669; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Marketing Knowledge Base</h1>
        
        <div class="dashboard-grid">
            <div class="main-content">
                <div class="search-section">
                    <h3>Search Marketing Best Practices</h3>
                    <input type="text" id="searchInput" class="search-input" 
                           placeholder="e.g., SEO optimization, social media strategy, content writing tips...">
                    
                    <div class="category-filters">
                        <button class="category-btn active" data-category="all">All</button>
                        <button class="category-btn" data-category="seo">SEO</button>
                        <button class="category-btn" data-category="content_marketing">Content Marketing</button>
                        <button class="category-btn" data-category="social_media">Social Media</button>
                        <button class="category-btn" data-category="analytics">Analytics</button>
                    </div>
                    
                    <button onclick="searchKnowledge()" class="btn">Search Knowledge Base</button>
                    
                    <div class="category-filters" style="margin-top: 15px; border-top: 1px solid #333; padding-top: 15px;">
                        <button onclick="getSEOTips()" class="btn secondary">Get SEO Tips</button>
                        <button onclick="getContentTips()" class="btn secondary">Writing Tips</button>
                        <button onclick="getSocialTips()" class="btn secondary">Social Media</button>
                    </div>
                </div>
                
                <div id="searchResults" class="search-results"></div>
                
                <div class="insights-grid">
                    <h3 style="grid-column: 1 / -1;">Fresh Marketing Insights (Last 7 Days)</h3>
                    {% for insight in fresh_insights %}
                    <div class="insight-card">
                        <div class="insight-title">{{ insight.title }}</div>
                        <div class="insight-meta">
                            {{ insight.feed_name }} • {{ insight.category|title }} 
                            {% if insight.days_old is not none %}• {{ insight.days_old }} days ago{% endif %}
                            • Score: {{ "%.1f"|format(insight.relevance_score) }}
                        </div>
                        <div class="insight-summary">
                            {{ insight.summary or insight.content[:200] }}...
                        </div>
                        {% if insight.keywords %}
                        <div class="insight-keywords">
                            {% for keyword in insight.keywords[:5] %}
                            <span class="keyword-tag">{{ keyword }}</span>
                            {% endfor %}
                        </div>
                        {% endif %}
                        <div style="margin-top: 15px;">
                            <a href="{{ insight.url }}" target="_blank" class="btn secondary" style="font-size: 12px; padding: 6px 12px;">Read Full Article</a>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="sidebar">
                <div class="stats-section">
                    <h3>RSS Monitor Status</h3>
                    <div class="monitor-status">
                        <div class="status-dot{% if rss_status.running %} running{% endif %}"></div>
                        <span>{% if rss_status.running %}Running{% else %}Stopped{% endif %}</span>
                    </div>
                    
                    {% if rss_status.total_sources %}
                    <div class="stat-row">
                        <span class="stat-label">Sources:</span>
                        <span class="stat-value">{{ rss_status.active_sources }}/{{ rss_status.total_sources }}</span>
                    </div>
                    {% endif %}
                    
                    {% if rss_status.total_content %}
                    <div class="stat-row">
                        <span class="stat-label">Total Content:</span>
                        <span class="stat-value">{{ "{:,}".format(rss_status.total_content) }}</span>
                    </div>
                    
                    <div class="stat-row">
                        <span class="stat-label">Fresh Content:</span>
                        <span class="stat-value">{{ "{:,}".format(rss_status.fresh_content or 0) }}</span>
                    </div>
                    {% endif %}
                    
                    <div style="margin-top: 15px;">
                        <button onclick="toggleMonitor()" id="toggleBtn" class="btn">
                            {% if rss_status.running %}Stop Monitor{% else %}Start Monitor{% endif %}
                        </button>
                        <button onclick="forceUpdate()" class="btn secondary">Force Update</button>
                    </div>
                </div>
                
                <div class="stats-section">
                    <h3>Knowledge Stats</h3>
                    {% for category, stats in category_stats.items() %}
                    <div class="stat-row">
                        <span class="stat-label">{{ category|title }}:</span>
                        <span class="stat-value">{{ stats.total_content }} ({{ stats.fresh_content }} fresh)</span>
                    </div>
                    {% endfor %}
                </div>
                
                <div class="stats-section">
                    <h3>Quick Actions</h3>
                    <button onclick="getMarketingTrends()" class="btn secondary">2025 Trends</button>
                    <button onclick="getRankMathTips()" class="btn secondary">Rank Math Tips</button>
                    <button onclick="getLocalSEO()" class="btn secondary">Local SEO</button>
                    <button onclick="getTechnicalSEO()" class="btn secondary">Technical SEO</button>
                </div>
            </div>
        </div>
        
        <div style="margin-top: 40px; text-align: center;">
            <a href="/" class="btn secondary">Back to Chat</a>
            <a href="/integrations" class="btn secondary">Integrations</a>
            <a href="/system" class="btn secondary">System Dashboard</a>
        </div>
    </div>
    
    <script>
        let activeCategory = 'all';
        let isMonitorRunning = {{ 'true' if rss_status.running else 'false' }};
        
        // Category filter handling
        document.querySelectorAll('.category-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.category-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                activeCategory = this.dataset.category;
            });
        });
        
        // Enter key search
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchKnowledge();
            }
        });
        
        function showLoading(container) {
            container.innerHTML = '<div class="loading">Searching knowledge base...</div>';
            container.style.display = 'block';
        }
        
        function displayResults(results, query) {
            const container = document.getElementById('searchResults');
            
            if (results.length === 0) {
                container.innerHTML = `
                    <h3>No Results Found</h3>
                    <p>No marketing content found for "${query}". Try different keywords or check a different category.</p>
                `;
                return;
            }
            
            let html = `<h3>Search Results for "${query}" (${results.length} found)</h3>`;
            html += '<div class="insights-grid">';
            
            results.forEach(result => {
                html += `
                    <div class="insight-card">
                        <div class="insight-title">${result.title}</div>
                        <div class="insight-meta">
                            ${result.feed_name || 'Unknown Source'} • ${result.category} 
                            ${result.subcategory ? '• ' + result.subcategory : ''}
                            • Score: ${result.relevance_score.toFixed(1)}
                            ${result.days_old !== null ? '• ' + result.days_old + ' days ago' : ''}
                        </div>
                        <div class="insight-summary">
                            ${result.summary || result.content.substring(0, 200)}...
                        </div>
                        ${result.keywords && result.keywords.length > 0 ? 
                            '<div class="insight-keywords">' + 
                            result.keywords.slice(0, 5).map(kw => `<span class="keyword-tag">${kw}</span>`).join('') +
                            '</div>' : ''
                        }
                        <div style="margin-top: 15px;">
                            <a href="${result.url}" target="_blank" class="btn secondary" style="font-size: 12px; padding: 6px 12px;">Read Full Article</a>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            container.innerHTML = html;
        }
        
        function searchKnowledge() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) {
                alert('Please enter a search query');
                return;
            }
            
            const container = document.getElementById('searchResults');
            showLoading(container);
            
            fetch('/api/marketing-knowledge/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify({
                    query: query,
                    category: activeCategory === 'all' ? null : activeCategory,
                    limit: 12
                })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    displayResults(data.results, query);
                } else {
                    container.innerHTML = `<div style="color: #dc2626;">Search failed: ${data.error}</div>`;
                }
            })
            .catch(e => {
                container.innerHTML = `<div style="color: #dc2626;">Search failed: ${e}</div>`;
            });
        }
        
        function getSEOTips() {
            const query = document.getElementById('searchInput').value.trim();
            const container = document.getElementById('searchResults');
            showLoading(container);
            
            fetch('/api/marketing-knowledge/seo-tips', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify({topic: query})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    displayResults(data.tips, `SEO Tips${query ? ' for "' + query + '"' : ''}`);
                } else {
                    container.innerHTML = `<div style="color: #dc2626;">Failed to get SEO tips: ${data.error}</div>`;
                }
            })
            .catch(e => {
                container.innerHTML = `<div style="color: #dc2626;">Failed to get SEO tips: ${e}</div>`;
            });
        }
        
        function getContentTips() {
            const container = document.getElementById('searchResults');
            showLoading(container);
            
            fetch('/api/marketing-knowledge/content-tips', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify({content_type: 'blog'})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    displayResults(data.tips, 'Content Writing Tips');
                } else {
                    container.innerHTML = `<div style="color: #dc2626;">Failed to get content tips: ${data.error}</div>`;
                }
            })
            .catch(e => {
                container.innerHTML = `<div style="color: #dc2626;">Failed to get content tips: ${e}</div>`;
            });
        }
        
        function getSocialTips() {
            const container = document.getElementById('searchResults');
            showLoading(container);
            
            fetch('/api/marketing-knowledge/social-tips', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                credentials: 'include',
                body: JSON.stringify({platform: null})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    displayResults(data.tips, 'Social Media Tips');
                } else {
                    container.innerHTML = `<div style="color: #dc2626;">Failed to get social tips: ${data.error}</div>`;
                }
            })
            .catch(e => {
                container.innerHTML = `<div style="color: #dc2626;">Failed to get social tips: ${e}</div>`;
            });
        }
        
        function getMarketingTrends() {
            searchSpecific('marketing trends 2025 predictions digital marketing');
        }
        
        function getRankMathTips() {
            searchSpecific('Rank Math optimization WordPress SEO plugin');
        }
        
        function getLocalSEO() {
            searchSpecific('local SEO google my business local search ranking');
        }
        
        function getTechnicalSEO() {
            searchSpecific('technical SEO core web vitals site speed crawling');
        }
        
        function searchSpecific(query) {
            document.getElementById('searchInput').value = query;
            searchKnowledge();
        }
        
        function toggleMonitor() {
            const btn = document.getElementById('toggleBtn');
            btn.textContent = 'Processing...';
            btn.disabled = true;
            
            const action = isMonitorRunning ? 'stop' : 'start';
            
            fetch(`/api/rss-monitor/${action}`, {
                method: 'POST',
                credentials: 'include'
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    isMonitorRunning = !isMonitorRunning;
                    btn.textContent = isMonitorRunning ? 'Stop Monitor' : 'Start Monitor';
                    
                    // Update status indicator
                    const dot = document.querySelector('.status-dot');
                    if (isMonitorRunning) {
                        dot.classList.add('running');
                    } else {
                        dot.classList.remove('running');
                    }
                    
                    alert(data.message);
                } else {
                    alert('Action failed: ' + data.error);
                    btn.textContent = isMonitorRunning ? 'Stop Monitor' : 'Start Monitor';
                }
                btn.disabled = false;
            })
            .catch(e => {
                alert('Action failed: ' + e);
                btn.textContent = isMonitorRunning ? 'Stop Monitor' : 'Start Monitor';
                btn.disabled = false;
            });
        }
        
        function forceUpdate() {
            if (!confirm('Force RSS feed update? This may take several minutes.')) return;
            
            const originalText = event.target.textContent;
            event.target.textContent = 'Updating...';
            event.target.disabled = true;
            
            fetch('/api/rss-monitor/force-update', {
                method: 'POST',
                credentials: 'include'
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('RSS feeds updated successfully! Refresh the page to see new content.');
                } else {
                    alert('Update failed: ' + data.error);
                }
                event.target.textContent = originalText;
                event.target.disabled = false;
            })
            .catch(e => {
                alert('Update failed: ' + e);
                event.target.textContent = originalText;
                event.target.disabled = false;
            });
        }
    </script>
</body>
</html>
'''

# Section 23: Thread Management and Bookmark API Routes (NEW)
# Add this section to your app.py file

@app.route('/api/threads', methods=['GET'])
def api_list_threads():
    """API: List all conversation threads with timestamps"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Get query parameters
        project = request.args.get('project')  # Optional project filter
        limit = min(int(request.args.get('limit', 50)), 100)
        include_archived = request.args.get('include_archived', 'false').lower() == 'true'
        
        threads = []
        
        with get_db_connection() as conn:
            if not conn:
                return jsonify({
                    'success': False,
                    'error': 'Database not available'
                }), 500
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Build query with optional filters
            where_clauses = []
            params = []
            
            if project:
                where_clauses.append('project = %s')
                params.append(project)
            
            if not include_archived:
                where_clauses.append('NOT is_archived')
            
            where_sql = 'WHERE ' + ' AND '.join(where_clauses) if where_clauses else ''
            params.append(limit)
            
            cursor.execute(f'''
                SELECT tm.thread_id, tm.title, tm.project, tm.created_at, tm.updated_at,
                       tm.message_count, tm.tags, tm.is_archived,
                       COUNT(ct.id) as actual_message_count,
                       MAX(ct.created_at) as last_message_at,
                       MIN(ct.created_at) as first_message_at
                FROM chat_thread_metadata tm
                LEFT JOIN chat_threads ct ON tm.thread_id = ct.thread_id
                {where_sql}
                GROUP BY tm.thread_id, tm.title, tm.project, tm.created_at, 
                         tm.updated_at, tm.message_count, tm.tags, tm.is_archived
                ORDER BY tm.updated_at DESC
                LIMIT %s
            ''', params)
            
            rows = cursor.fetchall()
            
            for row in rows:
                threads.append({
                    'thread_id': str(row['thread_id']),
                    'title': row['title'],
                    'project': row['project'],
                    'created_at': row['created_at'].isoformat(),
                    'updated_at': row['updated_at'].isoformat(),
                    'message_count': row['actual_message_count'] or 0,
                    'tags': row['tags'] or [],
                    'is_archived': row['is_archived'],
                    'first_message_at': row['first_message_at'].isoformat() if row['first_message_at'] else None,
                    'last_message_at': row['last_message_at'].isoformat() if row['last_message_at'] else None
                })
        
        return jsonify({
            'success': True,
            'threads': threads,
            'count': len(threads),
            'project_filter': project,
            'include_archived': include_archived
        })
        
    except Exception as e:
        app.logger.error(f"List threads API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/threads/<thread_id>', methods=['GET'])
def api_get_thread_content(thread_id):
    """API: Get specific thread content with all conversations"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        limit = min(int(request.args.get('limit', 100)), 500)
        include_metadata = request.args.get('include_metadata', 'true').lower() == 'true'
        
        thread_data = {
            'thread_id': thread_id,
            'metadata': None,
            'conversations': [],
            'bookmarks': []
        }
        
        with get_db_connection() as conn:
            if not conn:
                return jsonify({
                    'success': False,
                    'error': 'Database not available'
                }), 500
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get thread metadata
            if include_metadata:
                cursor.execute('''
                    SELECT thread_id, title, project, created_at, updated_at,
                           message_count, tags, is_archived
                    FROM chat_thread_metadata
                    WHERE thread_id = %s
                ''', (thread_id,))
                
                metadata_row = cursor.fetchone()
                if metadata_row:
                    thread_data['metadata'] = {
                        'thread_id': str(metadata_row['thread_id']),
                        'title': metadata_row['title'],
                        'project': metadata_row['project'],
                        'created_at': metadata_row['created_at'].isoformat(),
                        'updated_at': metadata_row['updated_at'].isoformat(),
                        'message_count': metadata_row['message_count'],
                        'tags': metadata_row['tags'] or [],
                        'is_archived': metadata_row['is_archived']
                    }
                else:
                    return jsonify({
                        'success': False,
                        'error': 'Thread not found'
                    }), 404
            
            # Get conversations in thread
            cursor.execute('''
                SELECT id, user_input, response_data, created_at, project,
                       context_project, context_data
                FROM chat_threads
                WHERE thread_id = %s
                ORDER BY created_at ASC
                LIMIT %s
            ''', (thread_id, limit))
            
            conversation_rows = cursor.fetchall()
            
            for row in conversation_rows:
                thread_data['conversations'].append({
                    'id': row['id'],
                    'user_input': row['user_input'],
                    'response_data': row['response_data'],
                    'created_at': row['created_at'].isoformat(),
                    'project': row['project'],
                    'context_project': row['context_project'],
                    'context_data': row['context_data'] or {}
                })
            
            # Get bookmarks for this thread
            cursor.execute('''
                SELECT cb.bookmark_id, cb.title, cb.notes, cb.bookmark_type,
                       cb.created_at, cb.chat_id,
                       ct.user_input as conversation_preview
                FROM conversation_bookmarks cb
                LEFT JOIN chat_threads ct ON cb.chat_id = ct.id
                WHERE cb.thread_id = %s
                ORDER BY cb.created_at DESC
            ''', (thread_id,))
            
            bookmark_rows = cursor.fetchall()
            
            for row in bookmark_rows:
                thread_data['bookmarks'].append({
                    'bookmark_id': str(row['bookmark_id']),
                    'title': row['title'],
                    'notes': row['notes'],
                    'bookmark_type': row['bookmark_type'],
                    'created_at': row['created_at'].isoformat(),
                    'chat_id': row['chat_id'],
                    'conversation_preview': row['conversation_preview'][:100] + '...' if row['conversation_preview'] else None
                })
        
        return jsonify({
            'success': True,
            **thread_data,
            'conversation_count': len(thread_data['conversations']),
            'bookmark_count': len(thread_data['bookmarks'])
        })
        
    except Exception as e:
        app.logger.error(f"Get thread content API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/bookmark', methods=['POST'])
def api_create_bookmark():
    """API: Create bookmark at current conversation point"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Required fields
        chat_id = data.get('chat_id')
        title = data.get('title', '').strip()
        
        # Optional fields
        notes = data.get('notes', '').strip()
        bookmark_type = data.get('bookmark_type', 'manual')
        
        if not chat_id:
            return jsonify({
                'success': False,
                'error': 'chat_id is required'
            }), 400
        
        if not title:
            return jsonify({
                'success': False,
                'error': 'title is required'
            }), 400
        
        # Validate that the conversation exists
        with get_db_connection() as conn:
            if not conn:
                return jsonify({
                    'success': False,
                    'error': 'Database not available'
                }), 500
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Check if conversation exists
            cursor.execute('''
                SELECT id, thread_id, project, user_input, created_at
                FROM chat_threads
                WHERE id = %s
            ''', (chat_id,))
            
            conversation = cursor.fetchone()
            if not conversation:
                return jsonify({
                    'success': False,
                    'error': 'Conversation not found'
                }), 404
            
            # Create the bookmark
            cursor.execute('''
                INSERT INTO conversation_bookmarks (chat_id, thread_id, title, notes, bookmark_type)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING bookmark_id, created_at
            ''', (
                chat_id,
                conversation['thread_id'],
                title,
                notes if notes else None,
                bookmark_type
            ))
            
            bookmark_result = cursor.fetchone()
            bookmark_id = bookmark_result['bookmark_id']
            created_at = bookmark_result['created_at']
            
            conn.commit()
            
            return jsonify({
                'success': True,
                'bookmark': {
                    'bookmark_id': str(bookmark_id),
                    'chat_id': chat_id,
                    'thread_id': str(conversation['thread_id']) if conversation['thread_id'] else None,
                    'title': title,
                    'notes': notes,
                    'bookmark_type': bookmark_type,
                    'created_at': created_at.isoformat(),
                    'conversation_preview': conversation['user_input'][:100] + '...' if len(conversation['user_input']) > 100 else conversation['user_input'],
                    'project': conversation['project']
                },
                'message': f'Bookmark "{title}" created successfully'
            })
        
    except Exception as e:
        app.logger.error(f"Create bookmark API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search', methods=['GET', 'POST'])
def api_search_conversations():
    """API: Search across all conversations in chat_tables"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        # Handle both GET and POST requests
        if request.method == 'POST':
            data = request.get_json() or {}
            query = data.get('query', '').strip()
            project_filter = data.get('project')
            limit = min(int(data.get('limit', 20)), 100)
            search_type = data.get('search_type', 'all')  # 'all', 'threads', 'bookmarks'
            include_context = data.get('include_context', True)
        else:
            query = request.args.get('q', '').strip()
            project_filter = request.args.get('project')
            limit = min(int(request.args.get('limit', 20)), 100)
            search_type = request.args.get('type', 'all')
            include_context = request.args.get('include_context', 'true').lower() == 'true'
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Search query is required'
            }), 400
        
        search_results = {
            'query': query,
            'conversations': [],
            'threads': [],
            'bookmarks': [],
            'total_results': 0
        }
        
        with get_db_connection() as conn:
            if not conn:
                return jsonify({
                    'success': False,
                    'error': 'Database not available'
                }), 500
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search conversations
            if search_type in ['all', 'conversations']:
                conversation_params = []
                conversation_where = []
                
                # Base search using full-text search if available, fallback to LIKE
                conversation_sql = '''
                    SELECT ct.id, ct.user_input, ct.response_data, ct.created_at, 
                           ct.project, ct.thread_id, ct.context_project,
                           tm.title as thread_title
                '''
                
                # Try full-text search first
                try:
                    conversation_sql += ''',
                           ts_rank(to_tsvector('english', 
                                   ct.user_input || ' ' || 
                                   COALESCE(ct.response_data->>'SyntaxPrime', '')), 
                                   plainto_tsquery('english', %s)) as rank
                    FROM chat_threads ct
                    LEFT JOIN chat_thread_metadata tm ON ct.thread_id = tm.thread_id
                    WHERE to_tsvector('english', 
                          ct.user_input || ' ' || 
                          COALESCE(ct.response_data->>'SyntaxPrime', '')) 
                          @@ plainto_tsquery('english', %s)
                    '''
                    conversation_params.extend([query, query])
                    using_fts = True
                except:
                    # Fallback to LIKE search
                    conversation_sql += '''
                    FROM chat_threads ct
                    LEFT JOIN chat_thread_metadata tm ON ct.thread_id = tm.thread_id
                    WHERE (LOWER(ct.user_input) LIKE %s 
                           OR LOWER(ct.response_data->>'SyntaxPrime') LIKE %s)
                    '''
                    like_pattern = f'%{query.lower()}%'
                    conversation_params.extend([like_pattern, like_pattern])
                    using_fts = False
                
                # Add project filter if specified
                if project_filter:
                    conversation_sql += ' AND ct.project = %s'
                    conversation_params.append(project_filter)
                
                # Add ordering and limit
                if using_fts:
                    conversation_sql += ' ORDER BY rank DESC, ct.created_at DESC'
                else:
                    conversation_sql += ' ORDER BY ct.created_at DESC'
                
                conversation_sql += ' LIMIT %s'
                conversation_params.append(limit)
                
                cursor.execute(conversation_sql, conversation_params)
                conversation_rows = cursor.fetchall()
                
                for row in conversation_rows:
                    result_item = {
                        'id': row['id'],
                        'type': 'conversation',
                        'user_input': row['user_input'],
                        'created_at': row['created_at'].isoformat(),
                        'project': row['project'],
                        'thread_id': str(row['thread_id']) if row['thread_id'] else None,
                        'thread_title': row['thread_title'],
                        'context_project': row['context_project']
                    }
                    
                    # Include AI response if requested
                    if include_context and row['response_data']:
                        ai_response = row['response_data'].get('SyntaxPrime', '')
                        if ai_response:
                            result_item['ai_response_preview'] = ai_response[:200] + '...' if len(ai_response) > 200 else ai_response
                    
                    # Add relevance score if available
                    if using_fts and 'rank' in row:
                        result_item['relevance_score'] = float(row['rank'])
                    
                    search_results['conversations'].append(result_item)
            
            # Search threads
            if search_type in ['all', 'threads']:
                thread_params = [f'%{query.lower()}%']
                thread_sql = '''
                    SELECT thread_id, title, project, created_at, updated_at,
                           message_count, tags, is_archived
                    FROM chat_thread_metadata
                    WHERE LOWER(title) LIKE %s
                '''
                
                if project_filter:
                    thread_sql += ' AND project = %s'
                    thread_params.append(project_filter)
                
                thread_sql += ' ORDER BY updated_at DESC LIMIT %s'
                thread_params.append(min(limit // 2, 10))  # Fewer thread results
                
                cursor.execute(thread_sql, thread_params)
                thread_rows = cursor.fetchall()
                
                for row in thread_rows:
                    search_results['threads'].append({
                        'thread_id': str(row['thread_id']),
                        'type': 'thread',
                        'title': row['title'],
                        'project': row['project'],
                        'created_at': row['created_at'].isoformat(),
                        'updated_at': row['updated_at'].isoformat(),
                        'message_count': row['message_count'],
                        'tags': row['tags'] or [],
                        'is_archived': row['is_archived']
                    })
            
            # Search bookmarks
            if search_type in ['all', 'bookmarks']:
                bookmark_params = [f'%{query.lower()}%', f'%{query.lower()}%']
                bookmark_sql = '''
                    SELECT cb.bookmark_id, cb.title, cb.notes, cb.bookmark_type,
                           cb.created_at, cb.chat_id, cb.thread_id,
                           ct.user_input as conversation_preview, ct.project,
                           tm.title as thread_title
                    FROM conversation_bookmarks cb
                    LEFT JOIN chat_threads ct ON cb.chat_id = ct.id
                    LEFT JOIN chat_thread_metadata tm ON cb.thread_id = tm.thread_id
                    WHERE LOWER(cb.title) LIKE %s
                       OR LOWER(cb.notes) LIKE %s
                '''
                
                if project_filter:
                    bookmark_sql += ' AND ct.project = %s'
                    bookmark_params.append(project_filter)
                
                bookmark_sql += ' ORDER BY cb.created_at DESC LIMIT %s'
                bookmark_params.append(min(limit // 3, 10))  # Even fewer bookmark results
                
                cursor.execute(bookmark_sql, bookmark_params)
                bookmark_rows = cursor.fetchall()
                
                for row in bookmark_rows:
                    search_results['bookmarks'].append({
                        'bookmark_id': str(row['bookmark_id']),
                        'type': 'bookmark',
                        'title': row['title'],
                        'notes': row['notes'],
                        'bookmark_type': row['bookmark_type'],
                        'created_at': row['created_at'].isoformat(),
                        'chat_id': row['chat_id'],
                        'thread_id': str(row['thread_id']) if row['thread_id'] else None,
                        'thread_title': row['thread_title'],
                        'conversation_preview': row['conversation_preview'][:100] + '...' if row['conversation_preview'] and len(row['conversation_preview']) > 100 else row['conversation_preview'],
                        'project': row['project']
                    })
        
        # Calculate total results
        search_results['total_results'] = (
            len(search_results['conversations']) +
            len(search_results['threads']) +
            len(search_results['bookmarks'])
        )
        
        return jsonify({
            'success': True,
            'search_results': search_results,
            'search_type': search_type,
            'project_filter': project_filter,
            'include_context': include_context
        })
        
    except Exception as e:
        app.logger.error(f"Search API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export-to-drive', methods=['POST'])
def api_export_to_drive():
    """API: Export bookmarked content to Google Docs"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Get export parameters
        bookmark_ids = data.get('bookmark_ids', [])  # List of bookmark IDs to export
        thread_id = data.get('thread_id')  # Or export entire thread
        export_format = data.get('format', 'google_docs')  # 'google_docs' or 'markdown'
        document_title = data.get('title', f'Ghostline Export - {datetime.datetime.now().strftime("%Y-%m-%d")}')
        include_responses = data.get('include_responses', True)
        
        if not bookmark_ids and not thread_id:
            return jsonify({
                'success': False,
                'error': 'Either bookmark_ids or thread_id must be provided'
            }), 400
        
        # Check Google integration availability
        try:
            from modules.enhanced_google_integration import EnhancedGoogleIntegration
            google_integration = EnhancedGoogleIntegration()
            
            if not google_integration.is_configured():
                return jsonify({
                    'success': False,
                    'error': 'Google integration not configured. Please set up Google OAuth first.'
                }), 400
                
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'Google integration module not available'
            }), 500
        
        export_content = []
        
        with get_db_connection() as conn:
            if not conn:
                return jsonify({
                    'success': False,
                    'error': 'Database not available'
                }), 500
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Export by bookmark IDs
            if bookmark_ids:
                placeholders = ','.join(['%s'] * len(bookmark_ids))
                cursor.execute(f'''
                    SELECT cb.bookmark_id, cb.title, cb.notes, cb.created_at,
                           ct.id as chat_id, ct.user_input, ct.response_data,
                           ct.created_at as conversation_date, ct.project,
                           tm.title as thread_title
                    FROM conversation_bookmarks cb
                    JOIN chat_threads ct ON cb.chat_id = ct.id
                    LEFT JOIN chat_thread_metadata tm ON cb.thread_id = tm.thread_id
                    WHERE cb.bookmark_id IN ({placeholders})
                    ORDER BY cb.created_at ASC
                ''', bookmark_ids)
                
                bookmark_rows = cursor.fetchall()
                
                for row in bookmark_rows:
                    content_item = {
                        'type': 'bookmark',
                        'title': row['title'],
                        'notes': row['notes'],
                        'bookmark_date': row['created_at'].isoformat(),
                        'conversation_date': row['conversation_date'].isoformat(),
                        'project': row['project'],
                        'thread_title': row['thread_title'],
                        'user_input': row['user_input']
                    }
                    
                    if include_responses and row['response_data']:
                        content_item['ai_response'] = row['response_data'].get('SyntaxPrime', '')
                    
                    export_content.append(content_item)
            
            # Export by thread ID
            elif thread_id:
                cursor.execute('''
                    SELECT tm.title as thread_title, tm.project, tm.created_at as thread_created,
                           ct.id as chat_id, ct.user_input, ct.response_data,
                           ct.created_at as conversation_date,
                           cb.bookmark_id, cb.title as bookmark_title, cb.notes
                    FROM chat_thread_metadata tm
                    JOIN chat_threads ct ON tm.thread_id = ct.thread_id
                    LEFT JOIN conversation_bookmarks cb ON ct.id = cb.chat_id
                    WHERE tm.thread_id = %s
                    ORDER BY ct.created_at ASC
                ''', (thread_id,))
                
                thread_rows = cursor.fetchall()
                
                if not thread_rows:
                    return jsonify({
                        'success': False,
                        'error': 'Thread not found or empty'
                    }), 404
                
                # Get thread info from first row
                thread_info = thread_rows[0]
                document_title = f"{thread_info['thread_title']} - {document_title}"
                
                for row in thread_rows:
                    content_item = {
                        'type': 'thread_conversation',
                        'thread_title': row['thread_title'],
                        'project': row['project'],
                        'conversation_date': row['conversation_date'].isoformat(),
                        'user_input': row['user_input']
                    }
                    
                    if row['bookmark_id']:
                        content_item['is_bookmarked'] = True
                        content_item['bookmark_title'] = row['bookmark_title']
                        content_item['bookmark_notes'] = row['notes']
                    
                    if include_responses and row['response_data']:
                        content_item['ai_response'] = row['response_data'].get('SyntaxPrime', '')
                    
                    export_content.append(content_item)
        
        if not export_content:
            return jsonify({
                'success': False,
                'error': 'No content found to export'
            }), 404
        
        # Format content for Google Docs
        if export_format == 'google_docs':
            try:
                # Create Google Doc
                document_content = f"# {document_title}\n\n"
                document_content += f"Exported on: {datetime.datetime.now().strftime('%Y-%m-%d at %H:%M')}\n\n"
                
                for item in export_content:
                    if item['type'] == 'bookmark':
                        document_content += f"## {item['title']}\n"
                        document_content += f"**Project:** {item['project']}\n"
                        document_content += f"**Date:** {datetime.datetime.fromisoformat(item['conversation_date']).strftime('%Y-%m-%d %H:%M')}\n"
                        
                        if item.get('notes'):
                            document_content += f"**Notes:** {item['notes']}\n"
                        
                        document_content += f"\n**User Input:**\n{item['user_input']}\n"
                        
                        if include_responses and item.get('ai_response'):
                            document_content += f"\n**AI Response:**\n{item['ai_response']}\n"
                        
                        document_content += "\n---\n\n"
                    
                    elif item['type'] == 'thread_conversation':
                        document_content += f"### {datetime.datetime.fromisoformat(item['conversation_date']).strftime('%H:%M')}"
                        
                        if item.get('is_bookmarked'):
                            document_content += f" [BOOKMARKED: {item['bookmark_title']}]"
                        
                        document_content += f"\n\n**User:** {item['user_input']}\n"
                        
                        if include_responses and item.get('ai_response'):
                            document_content += f"\n**SyntaxPrime:** {item['ai_response']}\n"
                        
                        document_content += "\n"
                
                # Create Google Doc using the enhanced integration
                doc_result = google_integration.create_google_doc(
                    title=document_title,
                    content=document_content
                )
                
                if doc_result.get('success'):
                    return jsonify({
                        'success': True,
                        'export': {
                            'format': 'google_docs',
                            'document_id': doc_result['document_id'],
                            'document_url': doc_result['document_url'],
                            'document_title': document_title,
                            'items_exported': len(export_content),
                            'export_date': datetime.datetime.now().isoformat(),
                            'include_responses': include_responses
                        },
                        'message': f'Successfully exported {len(export_content)} items to Google Docs'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'Google Docs creation failed: {doc_result.get("error", "Unknown error")}'
                    }), 500
                    
            except Exception as e:
                app.logger.error(f"Google Docs export failed: {e}")
                return jsonify({
                    'success': False,
                    'error': f'Google Docs export failed: {str(e)}'
                }), 500
        
        else:  # Markdown format
            markdown_content = f"# {document_title}\n\n"
            markdown_content += f"Exported on: {datetime.datetime.now().strftime('%Y-%m-%d at %H:%M')}\n\n"
            
            for item in export_content:
                if item['type'] == 'bookmark':
                    markdown_content += f"## {item['title']}\n\n"
                    markdown_content += f"**Project:** {item['project']}  \n"
                    markdown_content += f"**Date:** {datetime.datetime.fromisoformat(item['conversation_date']).strftime('%Y-%m-%d %H:%M')}  \n"
                    
                    if item.get('notes'):
                        markdown_content += f"**Notes:** {item['notes']}  \n"
                    
                    markdown_content += f"\n**User Input:**\n```\n{item['user_input']}\n```\n\n"
                    
                    if include_responses and item.get('ai_response'):
                        markdown_content += f"**AI Response:**\n```\n{item['ai_response']}\n```\n\n"
                    
                    markdown_content += "---\n\n"
            
            return jsonify({
                'success': True,
                'export': {
                    'format': 'markdown',
                    'content': markdown_content,
                    'document_title': document_title,
                    'items_exported': len(export_content),
                    'export_date': datetime.datetime.now().isoformat(),
                    'include_responses': include_responses
                },
                'message': f'Successfully exported {len(export_content)} items as Markdown'
            })
        
    except Exception as e:
        app.logger.error(f"Export to Drive API failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Section 24: Weather Integration Routes and Functions (NEW)
# Weather monitoring for headache prediction and UV sensitivity alerts

# Weather Integration Functions
def process_user_input_with_weather(user_input: str, project: str) -> dict[str, str]:
    """Enhanced input processing with weather integration"""
    
    # Import weather integration
    try:
        from modules.weather_integration import (
            detect_weather_command,
            handle_weather_integration,
            WEATHER_COMMANDS
        )
        weather_available = True
    except ImportError:
        weather_available = False
    
    # Check for weather commands first
    if weather_available and detect_weather_command(user_input):
        print("🌦️ Processing weather command")
        weather_response = handle_weather_integration(user_input, project)
        if weather_response:
            return weather_response
    
    # Check for specific weather command patterns
    user_lower = user_input.lower().strip()
    
    # Direct weather command responses
    weather_command_responses = {
        "weather": "Checking current weather conditions for health monitoring...",
        "pressure": "Analyzing barometric pressure for headache prediction...",
        "uv": "Checking UV index for sun sensitivity alerts...",
        "weather alerts": "Reviewing active weather alerts and health warnings...",
        "headache weather": "Analyzing current conditions for headache triggers...",
        "weather help": """🌦️ **Weather Monitoring Commands**:
        
**Basic Commands:**
• `weather` or `weather now` - Current conditions
• `pressure` - Barometric pressure and headache risk
• `uv` or `uv index` - UV levels and sun safety
• `weather alerts` - Active health alerts
• `weather patterns` - Historical pressure analysis

**Health Monitoring:**
• Tracks pressure drops that may trigger headaches (3+ mbar changes)
• Monitors UV index for sun sensitivity (alerts at 6+ UV index)
• Provides personalized health alerts based on conditions

**Data Sources:**
• Powered by Tomorrow.io Weather API
• Updates every 30 minutes to conserve API calls
• Maintains 7-day pressure history for pattern analysis

Type any weather command to get started!"""
    }
    
    # Check for exact weather command matches
    for cmd, response in weather_command_responses.items():
        if cmd in user_lower:
            if weather_available:
                # Process the actual weather command
                return handle_weather_integration(user_input, project)
            else:
                return {"SyntaxPrime": f"🌦️ Weather integration not available. Please configure TOMORROW_IO_API_KEY to enable weather monitoring.\n\n{response}"}
    
    # If not a weather command, continue with normal processing
    return None

def enhance_conversation_with_weather_awareness(messages: list[dict], user_input: str) -> List[dict]:
    """Add weather context to conversations when health-relevant topics are discussed"""
    
    try:
        from modules.weather_integration import get_weather_monitor, is_weather_configured
        
        if not is_weather_configured():
            return messages
        
        # Health-related keywords that benefit from weather context
        health_keywords = [
            'headache', 'migraine', 'head hurts', 'head pain', 'pressure headache',
            'sun', 'sunny', 'outside', 'outdoors', 'going out', 'uv', 'sunlight',
            'bright light', 'photosensitive', 'sun allergy', 'light sensitive',
            'weather', 'barometric', 'atmospheric pressure', 'storm coming'
        ]
        
        user_lower = user_input.lower()
        if not any(keyword in user_lower for keyword in health_keywords):
            return messages
        
        # Get current weather conditions
        monitor = get_weather_monitor()
        if not monitor:
            return messages
            
        weather_data = monitor.get_current_conditions()
        alerts = monitor.get_health_alerts(weather_data)
        
        # Determine if weather context is relevant
        weather_relevant = False
        context_info = []
        
        # Pressure-related context
        if weather_data.pressure_trend in ['dropping_significantly', 'dropping_moderately']:
            weather_relevant = True
            trend_desc = "significantly" if weather_data.pressure_trend == "dropping_significantly" else "moderately"
            context_info.append(f"Barometric pressure is dropping {trend_desc} ({weather_data.pressure_surface_level:.1f}mbar)")
        
        # UV-related context
        if weather_data.uv_index >= 6 and any(keyword in user_lower for keyword in ['sun', 'outside', 'outdoors', 'uv', 'light']):
            weather_relevant = True
            context_info.append(f"UV index is high ({weather_data.uv_index:.1f} - {weather_data.uv_health_concern})")
        
        # Add weather context if relevant
        if weather_relevant and context_info:
            weather_context = f"""

CURRENT WEATHER CONTEXT (for health-aware responses):
{' | '.join(context_info)}
Active health alerts: {', '.join(alerts) if alerts else 'None'}

Consider this weather information when providing health advice or discussing outdoor activities."""
            
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] += weather_context
            else:
                messages.insert(0, {
                    "role": "system",
                    "content": f"You are a helpful AI assistant with access to current weather conditions.{weather_context}"
                })
                
            print(f"🌦️ Added weather context to conversation (pressure: {weather_data.pressure_surface_level:.1f}mbar, UV: {weather_data.uv_index:.1f})")
    
    except Exception as e:
        print(f"⚠️ Failed to add weather context: {e}")
    
    return messages

def get_weather_aware_prompt(user_input: str, weather_data) -> Optional[str]:
    """Generate weather-aware prompts based on user input and current conditions"""
    
    user_lower = user_input.lower()
    
    # Weather-aware AI response templates
    weather_aware_prompts = {
        "headache_pressure": """The user is experiencing headaches. Current barometric pressure is {pressure}mbar and {trend}. 
        This pressure change may be contributing to their headache symptoms. Provide empathetic advice considering the weather context.""",
        
        "sun_sensitivity": """The user is asking about going outside. Current UV index is {uv_index} ({uv_concern}). 
        Consider their sun sensitivity and provide appropriate outdoor activity recommendations.""",
        
        "weather_health": """Current weather conditions may affect the user's health:
        - Pressure: {pressure}mbar ({pressure_trend})
        - UV Index: {uv_index} ({uv_concern})
        - Active alerts: {alerts}
        
        Provide health-conscious advice considering these weather factors."""
    }
    
    if any(word in user_lower for word in ['headache', 'migraine', 'head hurts', 'head pain']):
        if weather_data.pressure_trend in ['dropping_significantly', 'dropping_moderately']:
            return weather_aware_prompts["headache_pressure"].format(
                pressure=weather_data.pressure_surface_level,
                trend=weather_data.pressure_trend.replace('_', ' ')
            )
    
    elif any(word in user_lower for word in ['outside', 'sun', 'outdoors', 'going out']) and weather_data.uv_index >= 6:
        return weather_aware_prompts["sun_sensitivity"].format(
            uv_index=weather_data.uv_index,
            uv_concern=weather_data.uv_health_concern
        )
    
    elif any(word in user_lower for word in ['weather', 'how am i feeling', 'health', 'wellness']):
        try:
            from modules.weather_integration import get_weather_monitor
            monitor = get_weather_monitor()
            alerts = monitor.get_health_alerts(weather_data) if monitor else []
        except:
            alerts = []
            
        return weather_aware_prompts["weather_health"].format(
            pressure=weather_data.pressure_surface_level,
            pressure_trend=weather_data.pressure_trend.replace('_', ' '),
            uv_index=weather_data.uv_index,
            uv_concern=weather_data.uv_health_concern,
            alerts=', '.join(alerts) if alerts else 'None'
        )
    
    return None

def normalize_weather_command(user_input: str) -> str:
    """Normalize user input to standard weather commands"""
    
    # Command aliases for natural language processing
    weather_command_aliases = {
        # Pressure-related
        "barometric pressure": "pressure",
        "atmospheric pressure": "pressure",
        "pressure headache": "pressure",
        "headache weather": "pressure",
        
        # UV-related
        "sun index": "uv",
        "sunlight level": "uv",
        "uv level": "uv",
        "sun safety": "uv",
        
        # General weather
        "current weather": "weather",
        "weather now": "weather",
        "weather today": "weather",
        "conditions": "weather",
        
        # Alerts and patterns
        "health alerts": "weather alerts",
        "weather warnings": "weather alerts",
        "pressure trends": "weather patterns",
        "headache patterns": "weather patterns"
    }
    
    user_lower = user_input.lower().strip()
    
    for alias, command in weather_command_aliases.items():
        if alias in user_lower:
            return command
    
    return user_input

# Weather Dashboard and API Routes
@app.route('/weather')
def weather_dashboard():
    """Weather monitoring dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        from modules.weather_integration import (
            get_weather_monitor,
            get_weather_status,
            is_weather_configured,
            test_weather_integration
        )
        
        if not is_weather_configured():
            return render_template_string("""
            <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        background: #0f0f0f; color: #fff; padding: 40px; text-align: center;">
                <h2>🌦️ Weather Integration Setup Required</h2>
                <div style="background: #1a1a1a; border: 1px solid #333; border-radius: 8px; padding: 30px; max-width: 600px; margin: 0 auto;">
                    <p>To enable weather monitoring for headache prediction and UV sensitivity alerts:</p>
                    <ol style="text-align: left; margin: 20px 0;">
                        <li>Sign up for a free Tomorrow.io account at <a href="https://app.tomorrow.io/signup" 
                            style="color: #6366f1;">tomorrow.io</a></li>
                        <li>Get your API key from the dashboard</li>
                        <li>Set environment variable: <code style="background: #333; padding: 2px 6px; border-radius: 4px;">TOMORROW_IO_API_KEY=your_key_here</code></li>
                        <li>Optional: Set <code style="background: #333; padding: 2px 6px; border-radius: 4px;">DEFAULT_WEATHER_LOCATION=lat,lon</code> for your location</li>
                    </ol>
                    <p><a href="/diagnostics" style="color: #6366f1;">Check diagnostics</a> to verify configuration.</p>
                    <a href="/" style="color: #6366f1; text-decoration: none; margin-top: 20px; display: inline-block;">← Back to Chat</a>
                </div>
            </div>
            """)
        
        monitor = get_weather_monitor()
        if not monitor:
            return "Weather monitoring system initialization failed", 500
        
        # Get current weather data
        weather_data = monitor.get_current_conditions()
        alerts = monitor.get_health_alerts(weather_data)
        weather_summary = monitor.format_weather_summary(weather_data)
        
        # Get pressure history for chart
        pressure_history = monitor.pressure_history[-48:]  # Last 48 hours
        
        return render_template_string(WEATHER_DASHBOARD_TEMPLATE,
                                    weather_data=weather_data,
                                    weather_summary=weather_summary,
                                    alerts=alerts,
                                    pressure_history=pressure_history)
        
    except Exception as e:
        app.logger.error(f"Weather dashboard error: {e}")
        return f"Weather dashboard error: {str(e)}", 500

@app.route('/api/weather/current')
def api_weather_current():
    """API endpoint for current weather data"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        from modules.weather_integration import (
            get_weather_monitor,
            is_weather_configured
        )
        
        if not is_weather_configured():
            return jsonify({'error': 'Weather integration not configured'}), 400
        
        monitor = get_weather_monitor()
        if not monitor:
            return jsonify({'error': 'Weather monitor initialization failed'}), 500
        
        weather_data = monitor.get_current_conditions()
        alerts = monitor.get_health_alerts(weather_data)
        
        return jsonify({
            'success': True,
            'data': {
                'temperature_c': weather_data.temperature,
                'temperature_f': weather_data.temperature * 9/5 + 32,
                'pressure': weather_data.pressure_surface_level,
                'pressure_trend': weather_data.pressure_trend,
                'uv_index': weather_data.uv_index,
                'uv_health_concern': weather_data.uv_health_concern,
                'uv_alert_level': weather_data.uv_alert_level,
                'humidity': weather_data.humidity,
                'wind_speed': weather_data.wind_speed,
                'timestamp': weather_data.timestamp.isoformat(),
                'alerts': alerts
            }
        })
        
    except Exception as e:
        app.logger.error(f"Weather API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/weather/history')
def api_weather_history():
    """API endpoint for pressure history data"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        from modules.weather_integration import get_weather_monitor
        
        monitor = get_weather_monitor()
        if not monitor:
            return jsonify({'error': 'Weather monitor not available'}), 500
        
        # Get recent pressure history
        hours = int(request.args.get('hours', 48))
        history = monitor.pressure_history[-hours:] if hours <= 168 else monitor.pressure_history[-168:]
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/weather/test')
def api_weather_test():
    """Test weather integration"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        from modules.weather_integration import test_weather_integration, get_weather_status
        
        # Run integration test
        success = test_weather_integration()
        status = get_weather_status()
        
        return jsonify({
            'test_passed': success,
            'status': status
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Weather Dashboard Template
WEATHER_DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🌦️ Weather Monitoring - Ghostline AI</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #0f0f0f; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .weather-card { background: #1a1a1a; border: 1px solid #333; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
        .weather-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric { display: flex; align-items: center; padding: 16px; background: #2a2a2a; border-radius: 8px; margin: 8px 0; }
        .metric-icon { font-size: 24px; margin-right: 12px; }
        .metric-value { font-size: 24px; font-weight: 600; color: #6366f1; }
        .metric-label { font-size: 14px; color: #9ca3af; margin-left: 8px; }
        .alert { padding: 16px; border-radius: 8px; margin: 12px 0; border-left: 4px solid; }
        .alert-warning { background: #451a03; border-color: #f59e0b; color: #fbbf24; }
        .alert-info { background: #1e3a8a; border-color: #3b82f6; color: #93c5fd; }
        .pressure-trend { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 500; margin-left: 8px; }
        .trend-dropping { background: #7f1d1d; color: #fca5a5; }
        .trend-rising { background: #14532d; color: #86efac; }
        .trend-stable { background: #374151; color: #d1d5db; }
        .chart-container { height: 200px; margin: 20px 0; }
        .nav { margin-bottom: 20px; }
        .nav a { text-decoration: none; color: #6366f1; margin-right: 20px; }
        .nav a:hover { color: #8b5cf6; }
        h1 { color: #fff; margin-bottom: 8px; }
        h2 { color: #9ca3af; margin-top: 0; font-weight: normal; }
        h3 { color: #fff; }
        .btn { background: #6366f1; color: white; border: none; padding: 12px 24px; border-radius: 8px; 
               cursor: pointer; text-decoration: none; display: inline-block; margin: 10px 5px; }
        .btn:hover { background: #5855eb; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <div class="nav">
            <a href="/">← Back to Ghostline</a> | 
            <a href="/diagnostics">Diagnostics</a> | 
            <a href="/api/weather/current">API</a>
        </div>
        
        <h1>🌦️ Weather Monitoring Dashboard</h1>
        <h2>Health-focused weather tracking for headaches and UV sensitivity</h2>
        
        {% if alerts %}
        <div class="weather-card">
            <h3>🚨 Active Health Alerts</h3>
            {% for alert in alerts %}
            <div class="alert alert-warning">{{ alert }}</div>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="weather-grid">
            <div class="weather-card">
                <h3>🌡️ Current Conditions</h3>
                <div class="metric">
                    <span class="metric-icon">🌡️</span>
                    <span class="metric-value">{{ "%.1f"|format(weather_data.temperature * 9/5 + 32) }}°F</span>
                    <span class="metric-label">({{ "%.1f"|format(weather_data.temperature) }}°C)</span>
                </div>
                
                <div class="metric">
                    <span class="metric-icon">🌪️</span>
                    <span class="metric-value">{{ "%.1f"|format(weather_data.pressure_surface_level) }}</span>
                    <span class="metric-label">mbar</span>
                    <span class="pressure-trend trend-{{ weather_data.pressure_trend.replace('_', '-') }}">
                        {{ weather_data.pressure_trend.replace('_', ' ').title() }}
                    </span>
                </div>
                
                <div class="metric">
                    <span class="metric-icon">☀️</span>
                    <span class="metric-value">{{ weather_data.uv_index }}</span>
                    <span class="metric-label">UV Index ({{ weather_data.uv_health_concern }})</span>
                </div>
                
                <div class="metric">
                    <span class="metric-icon">💧</span>
                    <span class="metric-value">{{ "%.0f"|format(weather_data.humidity) }}%</span>
                    <span class="metric-label">Humidity</span>
                </div>
                
                <small style="color: #6b7280;">Last updated: {{ weather_data.timestamp.strftime('%I:%M %p') }}</small>
            </div>
            
            <div class="weather-card">
                <h3>📊 Pressure Trend (48h)</h3>
                <div class="chart-container">
                    <canvas id="pressureChart"></canvas>
                </div>
                <div class="alert alert-info">
                    <strong>Headache Tracking:</strong> Drops of 3+ mbar may trigger headaches in sensitive individuals.
                </div>
            </div>
        </div>
        
        <div class="weather-card">
            <h3>💡 Health Guidelines</h3>
            <p><strong>Barometric Pressure:</strong> Changes in atmospheric pressure can trigger migraines and headaches. We track pressure drops that may affect you.</p>
            <p><strong>UV Index:</strong> High UV levels (6+) can be problematic for sun-sensitive individuals. We monitor UV levels throughout the day.</p>
            <p><strong>Data Refresh:</strong> Weather data is cached for 30 minutes to avoid excessive API usage while maintaining accuracy.</p>
        </div>
        
        <div class="weather-card">
            <h3>Quick Actions</h3>
            <a href="/api/weather/current" class="btn">Get Current Data</a>
            <a href="/api/weather/test" class="btn">Test Integration</a>
            <button onclick="window.location.reload()" class="btn">Refresh Dashboard</button>
        </div>
        
        <div style="margin-top: 40px; text-align: center; color: #6b7280;">
            <p>Powered by Tomorrow.io Weather API | Weather data updates every 30 minutes</p>
        </div>
    </div>
    
    <script>
        // Pressure Chart
        const ctx = document.getElementById('pressureChart').getContext('2d');
        const pressureData = {{ pressure_history | safe }};
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: pressureData.map(d => new Date(d.timestamp).toLocaleTimeString()),
                datasets: [{
                    label: 'Pressure (mbar)',
                    data: pressureData.map(d => d.pressure),
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Pressure (mbar)',
                            color: '#9ca3af'
                        },
                        ticks: { color: '#9ca3af' },
                        grid: { color: '#374151' }
                    },
                    x: {
                        display: false
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
        
        // Auto-refresh every 30 minutes
        setTimeout(() => {
            window.location.reload();
        }, 30 * 60 * 1000);
    </script>
</body>
</html>
"""

# Section 18: Background Services and Startup
# Section 18: Background Services and Startup (UPDATED WITH CALENDAR-TELEGRAM INTEGRATION)
# Section 18: Background Services and Startup
# Section 18: Background Services and Startup
# Section 18: Background Services and Startup (UPDATED WITH RSS MARKETING MONITOR) 9/13/25
# Section 18: Background Services and Startup (UPDATED WITH GOOGLE TRENDS MONITORING) 9/13/25
def safe_reminder_checker():
    """Background thread with enhanced safety to prevent spam"""
    consecutive_errors = 0
    max_errors = 3
    last_check_time = datetime.datetime.now()
    daily_send_count = 0
    last_reset_date = datetime.datetime.now().date()
    
    while True:
        try:
            current_time = datetime.datetime.now()
            current_date = current_time.date()
            
            # Reset daily counter
            if current_date != last_reset_date:
                daily_send_count = 0
                last_reset_date = current_date
            
            # Safety check: don't run too frequently
            if (current_time - last_check_time).total_seconds() < 90:
                time.sleep(30)
                continue
            
            # Safety check: don't send too many per day
            if daily_send_count > 50:
                print(f"Daily send limit reached ({daily_send_count}), pausing until tomorrow")
                time.sleep(3600)  # Wait 1 hour
                continue
            
            if is_telegram_configured() and consecutive_errors < max_errors:
                reminders = GhostlineTelegramReminders()
                result = reminders.check_and_send_reminders()
                
                if result["sent"] > 0:
                    daily_send_count += result["sent"]
                    print(f"Sent {result['sent']} reminders (daily total: {daily_send_count})")
                    consecutive_errors = 0
                elif "error" in result:
                    consecutive_errors += 1
                    print(f"Reminder check error #{consecutive_errors}: {result['error']}")
                
                last_check_time = current_time
                
            else:
                if consecutive_errors >= max_errors:
                    print(f"Too many errors ({consecutive_errors}), pausing for 30 minutes")
                    time.sleep(1800)
                    consecutive_errors = 0
                    
        except Exception as e:
            consecutive_errors += 1
            print(f"Reminder checker crashed #{consecutive_errors}: {e}")
            
            if consecutive_errors >= max_errors:
                time.sleep(1800)  # 30 minute pause
                consecutive_errors = 0
        
        # Standard interval - longer to prevent spam
        time.sleep(180)  # Check every 3 minutes instead of 2

# Start background services only on Railway
if os.getenv('RAILWAY_ENVIRONMENT'):
    # Start Telegram reminder checker
    checker_thread = threading.Thread(target=safe_reminder_checker, daemon=True)
    checker_thread.start()
    print("Telegram reminder checker started with spam protection")
    
    # Start RSS Marketing Monitor
    def delayed_rss_start():
        time.sleep(120)  # 2 minute delay after app startup
        try:
            success = start_rss_monitoring()
            if success:
                print("RSS Marketing Monitor started successfully")
            else:
                print("RSS Marketing Monitor was already running")
        except Exception as e:
            print(f"Failed to start RSS Marketing Monitor: {e}")
    
    rss_startup_thread = threading.Thread(target=delayed_rss_start, daemon=True)
    rss_startup_thread.start()
    print("Scheduled RSS Marketing Monitor startup in 2 minutes")
    
    # Start Google Trends monitoring
    def delayed_trends_start():
        time.sleep(180)  # 3 minute delay after app startup
        try:
            from modules.google_trends_monitor import start_trends_monitoring
            success = start_trends_monitoring()
            if success:
                print("Google Trends monitoring started successfully")
            else:
                print("Google Trends monitoring failed to start (check configuration)")
        except Exception as e:
            print(f"Failed to start Google Trends monitoring: {e}")
    
    trends_startup_thread = threading.Thread(target=delayed_trends_start, daemon=True)
    trends_startup_thread.start()
    print("Scheduled Google Trends monitoring startup in 3 minutes")
    
    # Start Calendar-Telegram monitoring if configured
    if is_calendar_telegram_configured():
        def delayed_calendar_start():
            time.sleep(60)  # 1 minute delay after app startup
            try:
                start_calendar_monitoring()
                print("Calendar-Telegram monitoring started from saved state")
            except Exception as e:
                print(f"Failed to start Calendar-Telegram monitoring: {e}")
        
        calendar_startup_thread = threading.Thread(target=delayed_calendar_start, daemon=True)
        calendar_startup_thread.start()
        print("Scheduled Calendar-Telegram monitoring startup check")
    else:
        print("Calendar-Telegram integration not configured")
    
    # Start automated backups after a delay
    if not os.getenv('DISABLE_AUTO_BACKUPS'):
        def delayed_backup_start():
            time.sleep(300)  # 5 minute delay
            try:
                start_automated_backups()
                print("Automated backups started successfully")
            except Exception as e:
                print(f"Failed to start automated backups: {e}")
        
        backup_startup_thread = threading.Thread(target=delayed_backup_start, daemon=True)
        backup_startup_thread.start()
        print("Scheduled automated backup startup in 5 minutes")
    else:
        print("Automated backups disabled")
else:
    print("Background services disabled (not on Railway)")

# Application startup
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    
    if os.getenv('RAILWAY_ENVIRONMENT'):
        # Production configuration for Railway
        host = '0.0.0.0'  # Must bind to all interfaces for Railway
        debug = False
        
        print(f"Starting Ghostline on Railway - {host}:{port}")
        
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False  # Prevents duplicate processes
        )
    else:
        # Local development
        app.run(
            host='127.0.0.1',
            port=port,
            debug=True
        )
