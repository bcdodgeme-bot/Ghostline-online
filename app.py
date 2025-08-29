# Section 1: Imports and Initial Flask Setup

from flask import Flask, render_template, request, redirect, session, url_for, send_file, jsonify
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

# OCR/File Parsing
from PIL import Image
import fitz
import docx

# Markdown support
import markdown
from markupsafe import Markup

# Database imports - NEW
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import urllib.parse

# .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'ghostline-default-key')
PASSWORD = os.getenv('GHOSTLINE_PASSWORD', 'open_the_gate')

# Choose model via env; override on Render with CHAT_MODEL
CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

# sessions dir
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

# Database configuration - NEW
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    # Railway provides postgres:// but psycopg2 needs postgresql://
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# Section 2: Database Connection and Management Functions

# Section 2: Database Imports and Initialization
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

# Initialize database when app starts
with app.app_context():
    init_database()

# Section 3: Brain and RAG System Functions

# Section 3: Brain and RAG System Functions
# Section 3: Brain and RAG System Functions
# Section 3: Brain and RAG System Functions

from modules.brain import (
    enhanced_retrieve,
    enhanced_build_brain_background,
    enhanced_build_new_brain_background,
    build_brain_background,
    build_new_brain_background
)

# Section 4: OCR and File Processing Functions

# Section 4: OCR and File Processing Functions

from modules.file_processing import setup_easyocr_environment, markdown_filter

# Call this right after creating the Flask app
setup_easyocr_environment()

# Register markdown filter
app.jinja_env.filters['markdown'] = markdown_filter

# Section 5: Utility Functions

def _save_daily_log(sync_type: str, content: str):
    """Save daily sync results to log file"""
    try:
        os.makedirs("daily_logs", exist_ok=True)
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_path = f"daily_logs/{today}.md"
        
        timestamp = datetime.datetime.now().strftime("%I:%M %p")
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n## {sync_type.title()} Sync - {timestamp}\n\n")
            f.write(content)
            f.write("\n\n---\n")
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"DEBUG: Daily log save failed: {error_details}")

def load_conversation(project: str, limit: int = 50):
    """Load conversation history for a project"""
    path = f"sessions/{project.lower().replace(' ', '_')}.json"
    if not os.path.exists(path):
        return []
    turns = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for line in lines[-limit:]:
        try:
            row = json.loads(line)
            turns.append({"user": row.get("prompt", ""), "responses": row.get("response", {})})
        except json.JSONDecodeError:
            continue
    return turns

def _append_session(project: str, user_input: str, response_data: dict):
    """Append conversation to session file"""
    path = f"sessions/{project.lower().replace(' ', '_')}.json"
    with open(path, 'a', encoding='utf-8') as f:
        json.dump({'prompt': user_input, 'response': response_data}, f)
        f.write('\n')

def _render_enhanced(project: str, response_data: dict):
    """Render the main template with enhanced conversation data"""
    conversation = load_conversation_enhanced(project, limit=50)
    return render_template(
        'index.html',
        projects=PROJECTS,
        response_data=response_data,
        conversation=conversation,
        current_project=project
    )

# Section 6: Main Route with Enhanced Database Functionality

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

        # ---- Command: Gmail overnight (multiple aliases) ----
        if user_input.lower().strip() in ["overnight", "mail", "emails", "inbox", "check mail"]:
            try:
                msgs = list_overnight(include_unread=True, include_primary=False)
                lines = [f"- {msg.get('sender', 'Unknown')}: {msg.get('subject', 'No Subject')}" for msg in msgs[:25]]
                summary_prompt = (
                    f"Found {len(msgs)} overnight emails. Here's the summary:\n\n"
                    + "\n".join(lines)
                )
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                app.logger.error(f"Gmail overnight check failed: {e}")
                response_data = {"SyntaxPrime": f"Gmail check failed: {e}"}

            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # ---- Command: Gmail search (multiple aliases) ----
        if user_input.lower().startswith(("search ", "find ", "email about ")):
            # Extract query after the command
            for prefix in ["search ", "find ", "email about "]:
                if user_input.lower().startswith(prefix):
                    query_text = user_input[len(prefix):].strip()
                    break
            
            try:
                msgs = gmail_search(query_text)
                lines = [f"- Message ID: {msg.get('id', 'Unknown')}" for msg in msgs[:25]]
                summary_prompt = (
                    f"Found {len(msgs)} messages for search query: '{query_text}'\n\n"
                    + "\n".join(lines)
                )
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                app.logger.error(f"Gmail search failed: {e}")
                response_data = {"SyntaxPrime": f"Gmail search failed: {e}"}

            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # ---- Command: Today's calendar ----
        if user_input.lower().strip() in ["calendar", "today", "meetings", "schedule"]:
            try:
                events = list_today_events(max_results=20)
                calendar_summary = format_calendar_summary(events, "Today's Calendar")
                
                summary_prompt = (
                    f"Here's Carl's calendar for today. Summarize the key meetings and suggest priorities:\n\n"
                    f"{calendar_summary}"
                )
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                app.logger.error(f"Calendar check failed: {e}")
                response_data = {"SyntaxPrime": f"Calendar check failed: {e}"}

            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # ---- Command: Tomorrow's calendar ----
        if user_input.lower().strip() in ["tomorrow", "tomorrow's schedule", "next day"]:
            try:
                events = list_tomorrow_events(max_results=20)
                calendar_summary = format_calendar_summary(events, "Tomorrow's Calendar")
                
                summary_prompt = (
                    f"Here's Carl's calendar for tomorrow. Highlight important meetings and prep needed:\n\n"
                    f"{calendar_summary}"
                )
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                app.logger.error(f"Tomorrow's calendar failed: {e}")
                response_data = {"SyntaxPrime": f"Tomorrow's calendar failed: {e}"}

            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # ---- Command: Next meeting ----
        if user_input.lower().strip() in ["next meeting", "next", "upcoming"]:
            try:
                next_meeting = get_next_meeting()
                if next_meeting and next_meeting.get('summary'):
                    summary_prompt = (
                        f"Carl's next meeting: {next_meeting['summary']} at {next_meeting.get('start_formatted', 'Unknown time')}. "
                        f"Give a brief overview and any prep suggestions."
                    )
                else:
                    summary_prompt = "No upcoming meetings found."
                
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                app.logger.error(f"Next meeting check failed: {e}")
                response_data = {"SyntaxPrime": f"Next meeting check failed: {e}"}

            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # ---- Command: Search calendar ----
        if user_input.lower().startswith(("meeting about ", "calendar search ")):
            # Extract query after the command
            for prefix in ["meeting about ", "calendar search "]:
                if user_input.lower().startswith(prefix):
                    query_text = user_input[len(prefix):].strip()
                    break
            
            try:
                events = search_calendar(query_text, days_ahead=30, max_results=10)
                calendar_summary = format_calendar_summary(events, f"Calendar search: '{query_text}'")
                
                summary_prompt = (
                    f"Carl searched his calendar for '{query_text}'. Here are the relevant meetings:\n\n"
                    f"{calendar_summary}\n\n"
                    f"Summarize the key meetings and any patterns or next steps."
                )
                retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                app.logger.error(f"Calendar search failed: {e}")
                response_data = {"SyntaxPrime": f"Calendar search failed: {e}"}

            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # ---- Command: Good Morning ----
        if user_input.lower().strip() in ["good morning", "morning", "gm"]:
            app.logger.info("Good Morning command triggered")
            try:
                app.logger.info("About to call list_overnight")
                msgs = list_overnight(include_unread=True, include_primary=False)
                app.logger.info(f"Got {len(msgs)} messages")
                
                app.logger.info("About to call list_today_events")
                events = list_today_events(max_results=20)
                app.logger.info(f"Got {len(events)} events")
                
                app.logger.info("About to call get_next_meeting")
                next_meeting = get_next_meeting()
                app.logger.info(f"Got next meeting: {next_meeting}")
                
                # Format briefing
                email_summary = f"Found {len(msgs)} overnight emails"
                calendar_summary = format_calendar_summary(events, "Today's Schedule")
                
                morning_briefing = f"""Good morning! Here's your daily briefing:

**OVERNIGHT EMAILS**
{email_summary}

**TODAY'S CALENDAR**
{calendar_summary}

**NEXT MEETING**
{f"{next_meeting.get('summary', 'Unknown')} at {next_meeting.get('start_formatted', 'Unknown time')}" if next_meeting else "No meetings scheduled"}

**PRIORITIES FOR TODAY**
• Review urgent emails
• Prepare for upcoming meetings
• Check calendar for conflicts"""

                app.logger.info("About to save daily log")
                save_daily_log_enhanced("morning", morning_briefing)
                app.logger.info("Daily log saved")
                
                app.logger.info("About to call retrieve")
                retrieval_ctx = enhanced_retrieve(morning_briefing, k=5) if is_ready() else []
                app.logger.info("Retrieve completed")
                
                app.logger.info("About to call generate_response")
                response_data = generate_response(
                    f"Summarize this morning briefing and suggest 3 key priorities:\n\n{morning_briefing}",
                    use_voices, random_toggle, project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                app.logger.info("generate_response completed")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                app.logger.error(f"Full error trace: {error_details}")
                response_data = {"SyntaxPrime": f"Morning briefing failed: {str(e)} | Type: {type(e).__name__} | Details: {error_details[:200]}"}

            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # ---- Command: Good Evening ----
        if user_input.lower().strip() in ["good evening", "evening", "ge", "wrap up", "day summary"]:
            try:
                # Get today's sent emails and completed meetings
                today_events = list_today_events(max_results=20)
                tomorrow_events = list_tomorrow_events(max_results=15)
                
                # Filter completed events
                now = datetime.datetime.now()
                completed_events = []
                upcoming_events = []
                
                for event in today_events:
                    # Simple time comparison - events that started before now are "completed"
                    if 'T' in event['start']:
                        event_time = datetime.datetime.fromisoformat(event['start'].replace('Z', '+00:00'))
                        if event_time < now:
                            completed_events.append(event)
                        else:
                            upcoming_events.append(event)
                
                evening_summary = f"""Good evening! Here's your day wrap-up:

**TODAY'S COMPLETED MEETINGS ({len(completed_events)})**
{chr(10).join([f"• {e['start_formatted']} — {e['summary']}" for e in completed_events[:5]]) if completed_events else "No meetings completed"}

**STILL UPCOMING TODAY**
{chr(10).join([f"• {e['start_formatted']} — {e['summary']}" for e in upcoming_events]) if upcoming_events else "No more meetings today"}

**TOMORROW'S PREP NEEDED**
{format_calendar_summary(tomorrow_events[:5], "")}

**END OF DAY CHECKLIST**
• Review and respond to urgent emails
• Prepare materials for tomorrow's meetings  
• Set priorities for tomorrow
• Clear desk and close open tasks"""

                # Save to daily log
                save_daily_log_enhanced("evening", evening_summary)
                
                # Generate AI response
                retrieval_ctx = enhanced_retrieve(evening_summary, k=5) if is_ready() else []
                response_data = generate_response(
                    f"Summarize this evening wrap-up and suggest 3 things to prepare for tomorrow:\n\n{evening_summary}",
                    use_voices, random_toggle, project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                
            except Exception as e:
                app.logger.error(f"Evening summary failed: {e}")
                response_data = {"SyntaxPrime": f"Evening summary failed: {e}"}

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
                    retrieval_ctx = enhanced_retrieve(summary_prompt, k=5) if is_ready() else []
                    response_data = generate_response(
                        summary_prompt, use_voices, random_toggle,
                        project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                    )
            except Exception as e:
                app.logger.error(f"Scrape command failed: {e}")
                response_data = {"SyntaxPrime": f"Scrape failed: {e}"}
            
            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

        # ---- Normal flow ----
        try:
            retrieval_ctx = enhanced_retrieve(user_input, k=5) if is_ready() else []
            response_data = generate_response(
                user_input, use_voices, random_toggle,
                project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
            )
            save_conversation_enhanced(project, user_input, response_data)
        except Exception as e:
            app.logger.error(f"Normal flow failed: {e}")
            response_data = {"SyntaxPrime": f"Response generation failed: {e}"}
            save_conversation_enhanced(project, user_input, response_data)

    return _render_enhanced(selected_project, response_data)

# Section 7: Brain Building Endpoints and Dashboard

# Section 7: Brain Building Endpoints and Dashboard

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

@app.route('/brain')
def brain_control():
    """Enhanced brain control dashboard with batch progress"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return get_brain_control_dashboard()
# Section 8: File Upload and Processing Route

# Section 8: File Upload and Processing Route

from modules.file_processing import handle_file_upload

@app.route('/upload', methods=['POST'])
def upload_file():
    return handle_file_upload()

# Section 9: Other Routes - Streaming, Database Dashboard, and Utilities

# --- STREAMING ---
@app.route('/stream', methods=['POST'])
def stream():
    if not session.get('logged_in'):
        return "Unauthorized", 401
    user_input = request.form['user_input'].strip()
    project = request.form['project']
    use_voices = request.form.getlist('voices') or ['SyntaxPrime']
    retrieval_ctx = enhanced_retrieve(user_input, k=5) if is_ready() else []

    def generate():
        for chunk in stream_generate(
            user_input, use_voices, project=project,
            model=CHAT_MODEL, retrieval_context=retrieval_ctx
        ):
            yield chunk

    return app.response_class(generate(), mimetype='text/plain')

# --- DATABASE DASHBOARD - NEW ---
@app.route('/database_status')
def database_status():
    """Check database connection and table status"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    status = {
        "database_url_configured": bool(DATABASE_URL),
        "connection_working": False,
        "tables_exist": False,
        "conversation_count": 0,
        "uploaded_files_count": 0,
        "daily_logs_count": 0
    }
    
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor()
                status["connection_working"] = True
                
                # Check if tables exist
                cursor.execute('''
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name IN ('chat_threads', 'uploaded_files', 'daily_logs', 'user_settings')
                ''')
                table_count = cursor.fetchone()[0]
                status["tables_exist"] = table_count == 4
                
                if status["tables_exist"]:
                    # Get record counts
                    cursor.execute('SELECT COUNT(*) FROM chat_threads')
                    status["conversation_count"] = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM uploaded_files')
                    status["uploaded_files_count"] = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM daily_logs')
                    status["daily_logs_count"] = cursor.fetchone()[0]
                
            except Exception as e:
                app.logger.error(f"Database status check failed: {e}")
    
    return jsonify(status)

@app.route('/database')
def database_dashboard():
    """Simple database dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghostline Database Dashboard</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f0f; 
                color: #fff; 
                margin: 0; 
                padding: 20px; 
            }
            .container { max-width: 900px; margin: 0 auto; }
            .status-box { 
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
            }
            .btn:hover { background: #5855eb; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .stat-box {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-number {
                font-size: 28px;
                font-weight: bold;
                color: #10b981;
            }
            .success { color: #10b981; }
            .error { color: #ef4444; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Database Dashboard</h1>
            
            <div class="status-box">
                <h3>Connection Status</h3>
                <div id="status">Loading...</div>
            </div>
            
            <div class="status-box">
                <h3>Statistics</h3>
                <div id="stats" class="stats-grid">Loading...</div>
            </div>
            
            <div class="status-box">
                <button class="btn" onclick="refreshStatus()">Refresh</button>
                <button class="btn" onclick="window.location.href='/'">&larr; Back to Chat</button>
            </div>
        </div>
        
        <script>
            function refreshStatus() {
                fetch('/database_status')
                    .then(r => r.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        const statsDiv = document.getElementById('stats');
                        
                        // Update status
                        if (data.database_url_configured && data.connection_working && data.tables_exist) {
                            statusDiv.innerHTML = '<span class="success">Database Connected &amp; Ready</span>';
                        } else {
                            statusDiv.innerHTML = '<span class="error">Database Issues Detected</span>';
                        }
                        
                        // Update stats
                        statsDiv.innerHTML = `
                            <div class="stat-box">
                                <div class="stat-number">${data.conversation_count}</div>
                                <div>Conversations</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-number">${data.uploaded_files_count}</div>
                                <div>Files Uploaded</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-number">${data.daily_logs_count}</div>
                                <div>Daily Logs</div>
                            </div>
                        `;
                    })
                    .catch(e => {
                        document.getElementById('status').innerHTML = '<span class="error">Connection Error</span>';
                    });
            }
            
            refreshStatus();
            setInterval(refreshStatus, 5000);
        </script>
    </body>
    </html>
    '''
    return html_content

# --- UTILITY ROUTES ---
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

# Section 10: Debug Routes, Authentication, and App Startup

# --- DEBUG ROUTES ---
@app.route('/debug/rag')
def debug_rag():
    if not session.get('logged_in'):
        return "Unauthorized", 401
    q = request.args.get('query', '').strip()
    k = int(request.args.get('k', 5))
    if not q:
        return jsonify({"ok": False, "error": "missing query"}), 400
    if not is_ready():
        return jsonify({"ok": False, "error": "brain not ready"}), 500
    hits = retrieve(q, k=k)
    return jsonify({"ok": True, "count": len(hits), "results": hits})

@app.route('/debug/ocr')
def debug_ocr():
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        import easyocr
        import numpy as np
        
        reader = easyocr.Reader(['en'])
        
        return "<pre>EasyOCR is working!\n\nSupported languages: English\nReady for image analysis!</pre>"
        
    except ImportError as e:
        return f"<pre>EasyOCR not installed: {str(e)}</pre>"
    except Exception as e:
        return f"<pre>EasyOCR error: {str(e)}</pre>"

# --- AUTHENTICATION ---
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

# --- APP STARTUP ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)



