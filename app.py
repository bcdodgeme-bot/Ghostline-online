# Section 1: Imports and Flask Setup
from flask import Flask, render_template, request, redirect, session, url_for, send_file, jsonify, render_template_string, Response
from flask_cors import CORS
from utils.ghostline_engine import generate_response, stream_generate
from utils.rag_basic import retrieve, is_ready, load_corpus, get_build_status
from utils.scraper import scrape_url
from utils.gmail_client import (
    list_overnight, search as gmail_search,
    list_today_events, list_tomorrow_events, search_calendar,
    get_next_meeting, format_calendar_summary
)
import os, json, io
import threading
import time
import zipfile
import tempfile
import datetime
import requests

# Module imports for integrated systems
from modules.marketing_commands import process_marketing_command, is_marketing_configured
from modules.cloze_integration import process_cloze_command, is_cloze_configured
from modules.clickup_integration import process_clickup_command, is_clickup_configured
from modules.telegram_notifications import (
    GhostlineTelegramReminders,
    parse_reminder_command,
    is_telegram_configured
)
from modules.smart_commands import process_smart_command
from modules.personalities import GhostlinePersonalities, PersonalityIntegration

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
    enhanced_build_brain_background,
    enhanced_build_new_brain_background,
    build_brain_background,
    build_new_brain_background
)

from modules.file_processing import setup_easyocr_environment, markdown_filter

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

# Initialize database when app starts
with app.app_context():
    init_database()

# Setup EasyOCR environment
setup_easyocr_environment()

# Register markdown filter
app.jinja_env.filters['markdown'] = markdown_filter

# Section 3: Helper Functions for Chat Processing
def handle_reminder_command(user_input, project, use_voices, random_toggle):
    """Handle reminder creation commands"""
    reminder_keywords = [
        'remind me', 'reminder', 'set reminder', 'alert me',
        'don\'t forget', 'remember to', 'remind'
    ]
    
    if not any(keyword in user_input.lower() for keyword in reminder_keywords):
        return None, False
    
    if not is_telegram_configured():
        response_data = {
            "SyntaxPrime": "Telegram reminders not configured. Visit /integrations to set up your bot."
        }
        return response_data, True
    
    try:
        parsed = parse_reminder_command(user_input, project)
        
        if not parsed["success"]:
            response_data = {"SyntaxPrime": parsed["error"]}
            return response_data, True
        
        reminders = GhostlineTelegramReminders()
        result = reminders.create_reminder(
            title=parsed["title"],
            remind_at=parsed["remind_at"],
            project=parsed["project"],
            priority=2
        )
        
        if result["success"]:
            display_time = parsed.get("display_time", result["remind_at"].strftime('%I:%M %p on %B %d'))
            
            response_text = f"Reminder Created!\n\n"
            response_text += f"**What:** {parsed['title']}\n"
            response_text += f"**When:** {display_time}\n"
            response_text += f"**Project:** {project}\n\n"
            response_text += "You'll receive a Telegram notification with action buttons to mark complete or snooze."
            
            response_data = {"SyntaxPrime": response_text}
        else:
            response_data = {"SyntaxPrime": f"Failed to create reminder: {result['error']}"}
        
        return response_data, True
        
    except Exception as e:
        app.logger.error(f"Reminder command failed: {e}")
        response_data = {"SyntaxPrime": f"Reminder creation failed: {str(e)}"}
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
        search_terms = user_input.lower().replace('?', '').split()
        
        # Extract important words (longer than 3 chars, not common words)
        important_words = [w for w in search_terms
                          if len(w) > 3 and w not in [
                              'what', 'does', 'tell', 'about', 'explain', 'describe',
                              'where', 'when', 'who', 'how', 'why', 'the', 'and', 'are',
                              'this', 'that', 'with', 'from', 'they', 'have', 'been'
                          ]]
        
        print(f"Trying search with important words: {important_words}")
        
        for word in important_words[:3]:  # Try up to 3 important words
            try:
                additional_context = enhanced_retrieve(word, k=3, project=project)
                if additional_context:
                    enhanced_context.extend(additional_context)
                    print(f"Found {len(additional_context)} results for '{word}'")
            except Exception as e:
                print(f"Enhanced search failed for '{word}': {e}")
        
        # Also try the full query one more time
        try:
            final_attempt = enhanced_retrieve(user_input, k=5, project=project)
            if final_attempt:
                enhanced_context.extend(final_attempt)
                print(f"Final attempt found {len(final_attempt)} additional results")
        except Exception as e:
            print(f"Final enhanced search attempt failed: {e}")
        
        # Remove duplicates and use enhanced context if better
        if enhanced_context:
            seen_content = set()
            unique_context = []
            for item in enhanced_context:
                content_key = item.get('text', '')[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_context.append(item)
            
            if len(unique_context) > context_quality:
                retrieval_context = unique_context[:10]
                print(f"Using enhanced context: {len(retrieval_context)} unique results")
    
    # Add instruction to be less overly cautious about knowledge
    if is_specific_query and len(retrieval_context) < 2:
        enhanced_prompt = f"""User question: {user_input}

Context from database: {len(retrieval_context)} results found.

Important: Even if database context is limited, please answer using your general knowledge when appropriate. Don't claim you lack information if you actually know about the topic from your training. Only defer to "I don't have information" for very specific or recent topics that genuinely require external sources.

If this is about popular culture, TV shows, movies, books, or well-known topics, please provide a helpful response based on your training knowledge."""
        
        return generate_response(
            enhanced_prompt, use_voices, random_toggle,
            project=project, model=model, retrieval_context=retrieval_context
        )
    
    return generate_response(
        user_input, use_voices, random_toggle,
        project=project, model=model, retrieval_context=retrieval_context
    )
    
    
# Section 4: Main Chat Route
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

        # Auto-refresh brain context periodically
        try:
            refresh_brain_context()
        except Exception as e:
            print(f"Brain context refresh failed: {e}")

        # Try smart commands FIRST (before individual system commands)
        response_data, handled = process_smart_command(user_input, project, use_voices, random_toggle)
        if handled:
            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # Try Gmail/calendar commands (fallback for specific commands)
        response_data, handled = process_gmail_command(user_input, project, use_voices, random_toggle)
        if handled:
            return _render_enhanced(project, response_data)

        # Try reminder commands
        response_data, handled = handle_reminder_command(user_input, project, use_voices, random_toggle)
        if handled:
            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # Try ClickUp commands (with improved detection)
        if is_clickup_configured():
            response_data, handled = process_clickup_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)

        # Try marketing commands (image generation)
        if is_marketing_configured():
            response_data, handled = process_marketing_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)

        # Try Cloze commands
        if is_cloze_configured():
            response_data, handled = process_cloze_command(user_input, project, use_voices, random_toggle)
            if handled:
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)

        # ---- Command: scrape <url> ----
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
            except Exception as e:
                app.logger.error(f"Scrape command failed: {e}")
                response_data = {"SyntaxPrime": f"Scrape failed: {e}"}
            
            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # ---- Normal flow with enhanced context checking ----
        try:
            retrieval_ctx = enhanced_retrieve(user_input, k=5, project=project) if is_ready() else []
            
            # Use enhanced response generation with context validation
            response_data = generate_response_with_context_check(
                user_input, use_voices, random_toggle,
                project, CHAT_MODEL, retrieval_ctx
            )
            
            save_conversation_enhanced(project, user_input, response_data)
        except Exception as e:
            app.logger.error(f"Normal flow failed: {e}")
            response_data = {"SyntaxPrime": f"Response generation failed: {e}"}
            save_conversation_enhanced(project, user_input, response_data)

    return _render_enhanced(selected_project, response_data)
    
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
from modules.file_processing import handle_file_upload

@app.route('/upload', methods=['POST'])
def upload_file():
    """Updated upload handler for integrated chat flow"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    return handle_file_upload()
    
# Section 7: Streaming Chat API
@app.route('/api/chat/stream', methods=['POST'])
def stream_chat():
    """Fixed streaming chat endpoint with enhanced auth debugging"""
    
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
        
        app.logger.info(f"Stream processing: '{user_input[:50]}...' for project '{project}'")
        
        def generate_stream():
            try:
                # Send initial message
                yield f"data: {json.dumps({'type': 'start', 'message': 'Processing your request...'})}\n\n"
                
                # Initialize response data
                response_data = {}
                handled = False
                
                # Try command processors with better error isolation
                processors = [
                    ('smart', lambda: process_smart_command(user_input, project, use_voices, random_toggle)),
                    ('gmail', lambda: process_gmail_command(user_input, project, use_voices, random_toggle)),
                    ('reminder', lambda: handle_reminder_command(user_input, project, use_voices, random_toggle)),
                ]
                
                # Add conditional processors
                if is_clickup_configured():
                    processors.append(('clickup', lambda: process_clickup_command(user_input, project, use_voices, random_toggle)))
                if is_marketing_configured():
                    processors.append(('marketing', lambda: process_marketing_command(user_input, project, use_voices, random_toggle)))
                if is_cloze_configured():
                    processors.append(('cloze', lambda: process_cloze_command(user_input, project, use_voices, random_toggle)))
                
                # Try each processor
                for proc_name, processor in processors:
                    if not handled:
                        try:
                            response_data, handled = processor()
                            if handled:
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
                                "Use bullets and keep it tight and actionable.\n\n"
                                f"--- SCRAPED CONTENT START ---\n{result['text']}\n--- SCRAPED CONTENT END ---"
                            )
                            retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                            response_data = generate_response(
                                summary_prompt, use_voices, random_toggle,
                                project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                            )
                        handled = True
                    except Exception as e:
                        app.logger.error(f"Scrape command failed: {e}")
                        response_data = {"SyntaxPrime": f"Scrape failed: {e}"}
                        handled = True
                
                # Normal AI response as fallback
                if not handled:
                    try:
                        retrieval_ctx = enhanced_retrieve(user_input, k=5) if is_ready() else []
                        response_data = generate_response(
                            user_input, use_voices, random_toggle,
                            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                        )
                    except Exception as e:
                        app.logger.error(f"Normal response generation failed: {e}")
                        response_data = {"SyntaxPrime": f"Response generation failed: {str(e)}"}
                
                # Ensure we have some response
                if not response_data:
                    response_data = {"SyntaxPrime": "I encountered an issue processing your request. Please try again."}
                
                # Save conversation
                try:
                    save_conversation_enhanced(project, user_input, response_data)
                    app.logger.info("Conversation saved successfully")
                except Exception as e:
                    app.logger.error(f"Failed to save conversation: {e}")
                
                # Stream each response with proper chunking
                for voice, content in response_data.items():
                    if not content:
                        continue
                        
                    # Send content in chunks for streaming effect
                    chunk_size = 30
                    for i in range(0, len(content), chunk_size):
                        chunk = content[i:i+chunk_size]
                        yield f"data: {json.dumps({'type': 'content', 'voice': voice, 'chunk': chunk})}\n\n"
                        time.sleep(0.03)  # Small delay for streaming effect
                
                # Send completion signal
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

@app.route('/api/auth/check')
def check_auth_status():
    """Debug endpoint to check authentication status"""
    if not session.get('logged_in'):
        return jsonify({
            'authenticated': False,
            'session_exists': bool(session),
            'session_keys': list(session.keys()),
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown')
        }), 401
    
    return jsonify({
        'authenticated': True,
        'session_keys': list(session.keys()),
        'message': 'Authentication working correctly'
    })
    
# Section 8: Dashboard Routes (Modular)
from modules.dashboard_system import setup_system_routes
from modules.dashboard_diagnostics import setup_diagnostics_routes
from modules.dashboard_integrations import setup_integrations_routes

# Register dashboard routes
setup_system_routes(app)
setup_diagnostics_routes(app)
setup_integrations_routes(app)

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
    
# Section 13: Mobile API Routes
@app.route('/api/mobile/auth', methods=['POST'])
def mobile_auth():
    """JWT authentication for mobile clients"""
    if not session.get('logged_in'):
        data = request.get_json()
        password = data.get('password')
        
        if password == PASSWORD:
            if JWT_AVAILABLE:
                import time
                
                payload = {
                    'authenticated': True,
                    'exp': int(time.time()) + (24 * 60 * 60)  # 24 hours
                }
                token = jwt.encode(payload, app.secret_key, algorithm='HS256')
                
                return jsonify({
                    'success': True,
                    'token': token,
                    'expires_in': 24 * 60 * 60
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'JWT not available - install PyJWT'
                }), 500
        else:
            return jsonify({'success': False, 'error': 'Invalid password'}), 401
    else:
        return jsonify({'success': True, 'message': 'Already authenticated'})

@app.route('/api/mobile/projects')
def mobile_projects():
    """Get projects with conversation counts for mobile"""
    if not is_mobile_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    projects_with_counts = []
    
    with get_db_connection() as conn:
        if conn:
            cursor = conn.cursor()
            for project in PROJECTS:
                cursor.execute('''
                    SELECT COUNT(*) as count,
                           MAX(created_at) as last_activity
                    FROM chat_threads 
                    WHERE project = %s
                ''', (project,))
                
                result = cursor.fetchone()
                projects_with_counts.append({
                    'name': project,
                    'conversation_count': result[0] if result else 0,
                    'last_activity': result[1].isoformat() if result and result[1] else None
                })
        else:
            for project in PROJECTS:
                projects_with_counts.append({
                    'name': project,
                    'conversation_count': 0,
                    'last_activity': None
                })
    
    return jsonify({
        'success': True,
        'projects': projects_with_counts
    })

@app.route('/api/mobile/conversations/<project>')
def mobile_conversations(project):
    """Get conversation history for a project (paginated)"""
    if not is_mobile_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    offset = (page - 1) * limit
    
    conversations = []
    total_count = 0
    
    with get_db_connection() as conn:
        if conn:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM chat_threads WHERE project = %s', (project,))
            total_count = cursor.fetchone()[0]
            
            # Get paginated conversations
            cursor.execute('''
                SELECT user_input, response_data, created_at 
                FROM chat_threads 
                WHERE project = %s 
                ORDER BY created_at DESC 
                LIMIT %s OFFSET %s
            ''', (project, limit, offset))
            
            rows = cursor.fetchall()
            for row in rows:
                conversations.append({
                    'user_input': row[0],
                    'responses': row[1],
                    'timestamp': row[2].isoformat(),
                    'preview': row[0][:100] + '...' if len(row[0]) > 100 else row[0]
                })
    
    return jsonify({
        'success': True,
        'conversations': conversations,
        'pagination': {
            'page': page,
            'limit': limit,
            'total': total_count,
            'has_more': (offset + limit) < total_count
        }
    })

@app.route('/api/mobile/chat', methods=['POST'])
def mobile_chat():
    """Simple test version for Swift app development"""
    if not is_mobile_authenticated():
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    user_input = data.get('user_input', '').strip()
    
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    
    # Simple test response - replace with full pipeline once Swift app is working
    return jsonify({
        'success': True,
        'responses': {
            'SyntaxPrime': f'Test response to: {user_input}'
        }
    })

# TODO: Replace above with full mobile chat implementation once Swift app is tested:
# @app.route('/api/mobile/chat', methods=['POST'])
# def mobile_chat_full():
#     """Full mobile chat endpoint"""
#     if not is_mobile_authenticated():
#         return jsonify({'error': 'Unauthorized'}), 401
#
#     data = request.get_json()
#     user_input = data.get('user_input', '').strip()
#     project = data.get('project', 'Personal Operating Manual')
#     use_voices = data.get('voices', ['SyntaxPrime'])
#     random_toggle = data.get('random', False)
#
#     if not user_input:
#         return jsonify({'error': 'No input provided'}), 400
#
#     try:
#         retrieval_ctx = enhanced_retrieve(user_input, k=5) if is_ready() else []
#         response_data = generate_response(
#             user_input, use_voices, random_toggle,
#             project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
#         )
#
#         save_conversation_enhanced(project, user_input, response_data)
#
#         return jsonify({
#             'success': True,
#             'responses': response_data
#         })
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

def is_mobile_authenticated():
    """Check if mobile client is authenticated via JWT or session"""
    if session.get('logged_in'):
        return True
    
    if JWT_AVAILABLE:
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            try:
                token = auth_header.split(' ')[1]
                payload = jwt.decode(token, app.secret_key, algorithms=['HS256'])
                return payload.get('authenticated', False)
            except:
                return False
    
    return False

# Section 14: Google OAuth Integration
@app.route('/google/auth/start')
def google_auth_start():
    """Initiate Google OAuth flow - Railway-compatible version"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')
        
        if not os.path.exists(credentials_path):
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Google Setup Required</title>
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
                    .setup-steps { background: #1a1a1a; padding: 20px; border-radius: 8px; margin: 15px 0; }
                    .setup-steps ol li { margin: 10px 0; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Google OAuth Setup Required</h1>
                    <div class="setup-steps">
                        <h3>Complete These Steps First:</h3>
                        <ol>
                            <li>Go to <a href="https://console.cloud.google.com/" target="_blank">Google Cloud Console</a></li>
                            <li>Create a new project or select existing</li>
                            <li>Enable <strong>Gmail API</strong> and <strong>Calendar API</strong></li>
                            <li>Go to <strong>APIs & Services → Credentials</strong></li>
                            <li>Create <strong>OAuth 2.0 Client ID</strong> (Web application)</li>
                            <li>Add authorized redirect URI: <code>https://{{ railway_url }}/google/auth/callback</code></li>
                            <li>Download the credentials JSON file</li>
                            <li>Upload it to Railway and set <code>GOOGLE_CREDENTIALS_PATH</code> env var</li>
                            <li>Return here and try again</li>
                        </ol>
                    </div>
                    <a href="/integrations" class="btn">Setup Instructions</a>
                    <a href="/" class="btn">Back to Chat</a>
                </div>
            </body>
            </html>
            """, railway_url=os.getenv('RAILWAY_STATIC_URL', 'your-app.railway.app'))
        
        from google_auth_oauthlib.flow import Flow
        
        railway_url = os.getenv('RAILWAY_STATIC_URL')
        if railway_url:
            redirect_uri = f"https://{railway_url}/google/auth/callback"
        else:
            redirect_uri = "http://localhost:5000/google/auth/callback"
        
        app.logger.info(f"Starting OAuth flow with redirect URI: {redirect_uri}")
        
        flow = Flow.from_client_secrets_file(
            credentials_path,
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly"
            ]
        )
        flow.redirect_uri = redirect_uri
        
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'
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
    """Handle Google OAuth callback and save token"""
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
        
        credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')
        token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        
        flow = Flow.from_client_secrets_file(
            credentials_path,
            scopes=[
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/calendar.readonly"
            ],
            state=session['oauth_state']
        )
        flow.redirect_uri = session.get('oauth_redirect_uri')
        
        app.logger.info(f"Processing OAuth callback, saving token to: {token_path}")
        
        flow.fetch_token(authorization_response=request.url)
        
        credentials = flow.credentials
        with open(token_path, 'w') as token_file:
            token_file.write(credentials.to_json())
        
        app.logger.info("Token saved successfully")
        
        # Test the credentials immediately
        test_results = {}
        try:
            from utils.gmail_client import _gmail_service, _calendar_service
            
            gmail_svc = _gmail_service()
            profile = gmail_svc.users().getProfile(userId='me').execute()
            test_results['gmail'] = f"Connected as {profile.get('emailAddress', 'Unknown')}"
            
            cal_svc = _calendar_service()
            calendar_list = cal_svc.calendarList().list(maxResults=1).execute()
            test_results['calendar'] = f"Access to {len(calendar_list.get('items', []))} calendars"
            
        except Exception as test_error:
            test_results['error'] = str(test_error)
        
        session.pop('oauth_state', None)
        session.pop('oauth_redirect_uri', None)
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Google OAuth Complete</title>
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
                .btn.success { background: #059669; }
                .success-box { 
                    background: #065f46; border: 1px solid #059669; border-radius: 8px; 
                    padding: 20px; margin: 20px 0; 
                }
                .test-results { 
                    background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
                    padding: 15px; margin: 15px 0; 
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="success-box">
                    <h1>Google OAuth Setup Complete!</h1>
                    <p>Your Gmail and Calendar access has been configured successfully.</p>
                    <p><strong>Token saved to:</strong> {{ token_path }}</p>
                </div>
                
                <div class="test-results">
                    <h3>Connection Test Results:</h3>
                    {% if test_results.gmail %}
                        <p>Gmail: {{ test_results.gmail }}</p>
                    {% endif %}
                    {% if test_results.calendar %}
                        <p>Calendar: {{ test_results.calendar }}</p>
                    {% endif %}
                    {% if test_results.error %}
                        <p>Warning: {{ test_results.error }}</p>
                    {% endif %}
                </div>
                
                <div style="text-align: center; margin: 30px 0;">
                    <a href="/" class="btn success">Start Using Gmail Commands</a>
                    <a href="/integrations" class="btn">View Integrations</a>
                </div>
            </div>
        </body>
        </html>
        """, token_path=token_path, test_results=test_results)
        
    except Exception as e:
        app.logger.error(f"OAuth callback failed: {e}")
        return f"OAuth completion failed: {str(e)}<br><a href='/integrations'>Try Again</a>", 500

@app.route('/google/auth/revoke', methods=['POST'])
def google_auth_revoke():
    """Revoke Google authentication and delete token"""
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        
        if os.path.exists(token_path):
            try:
                from utils.gmail_client import _build_creds
                creds = _build_creds()
                
                requests.post(
                    'https://oauth2.googleapis.com/revoke',
                    params={'token': creds.token},
                    headers={'content-type': 'application/x-www-form-urlencoded'}
                )
                app.logger.info("Token revoked with Google")
            except:
                app.logger.warning("Could not revoke token with Google, but will delete local file")
            
            os.remove(token_path)
            app.logger.info("Local token file deleted")
            
            return jsonify({
                'success': True,
                'message': 'Google authentication revoked and token deleted'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No token file found'
            })
            
    except Exception as e:
        app.logger.error(f"Token revocation failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
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
        
# Section 16: Utility and Export Routes
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
