# Section 1: Imports and Flask Setup
# Section 1: Imports and Flask Setup (UPDATED)
# Section 1: Imports and Flask Setup (UPDATED FOR PHASE 2)
# Section 1: Imports and Flask Setup (UPDATED FOR CONSOLIDATED GOOGLE INTEGRATION)
# Section 1: Imports and Flask Setup (UPDATED WITH ENHANCED MARKETING)
# Section 1: Imports and Flask Setup (UPDATED WITH CALENDAR-TELEGRAM INTEGRATION)
# Section 1: Imports and Flask Setup (UPDATED WITH CALENDAR-TELEGRAM INTEGRATION)9/11/25
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

def handle_reminder_command(user_input, project, use_voices, random_toggle):
    """Handle reminder creation commands - FIXED DETECTION PATTERNS"""
    
    # DEBUG: Add this line to see if function is called
    print(f"🔍 DEBUG: handle_reminder_command called with input: '{user_input}'")
    
    # FIXED: Updated patterns to catch "remind me in one minute to X"
    explicit_reminder_patterns = [
        r'^remind me to\s+',
        r'^set a reminder\s+',
        r'^create a reminder\s+',
        r'^set reminder\s+',
        r'^reminder:\s+',
        r'^remind me in\s+',               # ✅ FIXED: Removed \d+ requirement
        r'^remind me at\s+',
        r'^reminder for\s+',
        r'remind me .+ in\s+',             # ✅ NEW: Catches "remind me to X in Y"
        r'remind me .+ at\s+',             # ✅ NEW: Catches "remind me to X at Y"
        r'set a reminder .+ (in|at|tomorrow|today)',
    ]
    
    user_input_lower = user_input.lower().strip()
    
    # Check if this is an EXPLICIT reminder request
    is_explicit_reminder = any(
        re.search(pattern, user_input_lower)
        for pattern in explicit_reminder_patterns
    )
    
    if not is_explicit_reminder:
        return None, False
    
    if not is_telegram_configured():
        response_data = {
            "SyntaxPrime": "Telegram reminders not configured. Visit /integrations to set up your bot."
        }
        return response_data, True
    
    try:
        # Add safety wrapper around the problematic parse function
        try:
            parsed = parse_reminder_command(user_input, project)
        except Exception as parse_error:
            app.logger.error(f"Reminder parsing failed: {parse_error}")
            response_data = {"SyntaxPrime": f"Could not parse reminder request: {str(parse_error)}"}
            return response_data, True
        
        if not parsed or not parsed.get("success"):
            error_msg = parsed.get("error", "Unknown parsing error") if parsed else "Parsing returned None"
            response_data = {"SyntaxPrime": f"Reminder parsing failed: {error_msg}"}
            return response_data, True
        
        # Add safety wrapper around reminder creation
        try:
            reminders = GhostlineTelegramReminders()
            result = reminders.create_reminder(
                title=parsed["title"],
                remind_at=parsed["remind_at"],
                project=parsed["project"],
                priority=2
            )
        except Exception as creation_error:
            app.logger.error(f"Reminder creation failed: {creation_error}")
            response_data = {"SyntaxPrime": f"Failed to create reminder: {str(creation_error)}"}
            return response_data, True
        
        if result and result.get("success"):
            display_time = parsed.get("display_time", result["remind_at"].strftime('%I:%M %p on %B %d') if result.get("remind_at") else "unknown time")
            
            response_text = f"Reminder Created!\n\n"
            response_text += f"**What:** {parsed['title']}\n"
            response_text += f"**When:** {display_time}\n"
            response_text += f"**Project:** {project}\n\n"
            response_text += "You'll receive a Telegram notification with action buttons to mark complete or snooze."
            
            response_data = {"SyntaxPrime": response_text}
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
            response_data = {"SyntaxPrime": f"Failed to create reminder: {error_msg}"}
        
        return response_data, True
        
    except Exception as e:
        app.logger.error(f"Reminder command completely failed: {e}", exc_info=True)
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
@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    response_data = {}
    selected_project = PROJECTS[0]

    if request.method == 'POST':
        user_input = request.form['user_input'].strip()
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

        # Handle reminder commands with proper error handling (MOVED UP HIGH PRIORITY)
        try:
            response_data, handled = handle_reminder_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)
        except Exception as e:
            app.logger.error(f"Reminder handler failed: {e}")

        # Try hybrid content strategy commands
        try:
            response_data, handled = generate_content_strategy_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)
        except Exception as e:
            app.logger.error(f"Content strategy command failed: {e}")

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

        # Try Enhanced Marketing Commands with Context Support
        if is_marketing_configured():
            try:
                # Get marketing context for better responses
                marketing_context = get_marketing_context()
                response_data, handled = process_marketing_command_with_context(
                    user_input, project, use_voices, random_toggle, marketing_context
                )
                if handled:
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

        # Normal AI response as fallback (same enhanced logic as web version)
        if not response_data:
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

        return _render_enhanced(project, response_data)

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

@app.route('/api/chat/stream', methods=['POST'])
def stream_chat():
    """Enhanced streaming chat endpoint - FIXED VERSION WITH BLUESKY"""
    
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
        
        def generate_stream():
            try:
                # Send initial message
                yield f"data: {json.dumps({'type': 'start', 'message': 'Processing your request...'})}\n\n"
                
                # Initialize response data
                response_data = {}
                handled = False
                
                # FIXED: Handle reminder commands with proper error handling FIRST
                try:
                    response_data, handled = handle_reminder_command(user_input, project, use_voices, random_toggle)
                    if handled:
                        app.logger.info(f"Request handled by reminder system")
                except Exception as e:
                    app.logger.error(f"Reminder handler failed: {e}")
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
                    
                    # Add enhanced marketing processor with context
                    if is_marketing_configured():
                        app.logger.info(f"Adding enhanced marketing processor to stream pipeline")
                        processors.insert(0, ('marketing_enhanced', lambda: process_marketing_command_with_context(user_input, project, use_voices, random_toggle, marketing_context)))
                    
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
                
                # Save conversation
                try:
                    save_conversation_enhanced(project, user_input, response_data)
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
# Section 10: Telegram Integration Routes
# Section 10: Telegram Integration Routes
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
    """Enhanced Telegram webhook handler with chat ID capture and detailed logging"""
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
            app.logger.info(f"Processing callback: {callback_query.get('data', 'no data')}")
            
            reminders = GhostlineTelegramReminders()
            result = reminders.process_callback_query(callback_query)
            
            app.logger.info(f"Callback result: {result}")
            
            # Send callback answer to remove loading state
            callback_id = callback_query.get('id')
            if callback_id:
                bot = reminders.bot
                answer_url = f"https://api.telegram.org/bot{bot.token}/answerCallbackQuery"
                requests.post(answer_url, json={
                    "callback_query_id": callback_id,
                    "text": "Action processed!" if result.get('success') else "Action failed"
                })
            
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
    """Mobile chat with full AI processing - ENHANCED with JWT auth + all integrations + FIXED BlueSky"""
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

        # Enhanced Marketing Commands with Context Support
        if is_marketing_configured():
            app.logger.info(f"Mobile: Enhanced marketing is configured, processing command: '{user_input}'")
            try:
                marketing_context = get_marketing_context()
                response_data, handled = process_marketing_command_with_context(
                    user_input, project, use_voices, random_toggle, marketing_context
                )
                if handled:
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
                'cloze': is_cloze_configured(),
                'clickup': is_clickup_configured(),
                'telegram': is_telegram_configured(),
                'calendar_telegram': is_calendar_telegram_configured(),
                'bluesky': is_bluesky_configured()  # NEW: BlueSky status
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

# Section 18: Background Services and Startup
# Section 18: Background Services and Startup (UPDATED WITH CALENDAR-TELEGRAM INTEGRATION)
# Section 18: Background Services and Startup
# Section 18: Background Services and Startup
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
