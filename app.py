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

# Markdown filter for Jinja2
def markdown_filter(text):
    """Convert markdown to HTML"""
    if not text:
        return ""
    # Configure markdown with basic extensions
    md = markdown.Markdown(extensions=['nl2br', 'fenced_code'])
    return Markup(md.convert(text))

# Register markdown filter
app.jinja_env.filters['markdown'] = markdown_filter

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
        app.logger.error(f"Failed to save daily log: {e}")

def build_brain_background():
    """Build the RAG system using batched processing"""
    global _rag_building, _rag_build_error
    
    try:
        _rag_building = True
        _rag_build_error = None
        app.logger.info("Starting batched brain build...")
        
        load_corpus(CORPUS_PATH)
        
        _rag_building = False
        app.logger.info("Batched brain build complete!")
        
    except Exception as e:
        _rag_building = False
        _rag_build_error = str(e)
        app.logger.error(f"Batched brain build failed: {e}")

def build_new_brain_background():
    """Build new brain from raw sources on server"""
    global _brain_building, _brain_build_error
    
    try:
        _brain_building = True
        _brain_build_error = None
        app.logger.info("Starting server-side brain building from raw sources...")
        
        from build_brain_fixed2 import build_new_brain
        result_path = build_new_brain()
        
        # Copy the new brain to the expected location
        import shutil
        shutil.copy(str(result_path), CORPUS_PATH)
        app.logger.info(f"New brain saved to {CORPUS_PATH}")
        
        _brain_building = False
        app.logger.info("Server-side brain build complete!")
        
    except Exception as e:
        _brain_building = False
        _brain_build_error = str(e)
        app.logger.error(f"Server-side brain build failed: {e}")

def load_conversation(project: str, limit: int = 50):
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


@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    response_data = {}
    selected_project = PROJECTS[0]

    if request.method == 'POST':
        user_input = request.form['user_input'].strip()
        project = request.form['project']
        selected_project = project
        use_voices = request.form.getlist('voices') or ['SyntaxPrime']
        random_toggle = 'random' in request.form

        # ---- Command: Gmail overnight (multiple aliases) ----
        if user_input.lower().strip() in ["overnight", "mail", "emails", "inbox", "check mail"]:
            try:
                msgs = list_overnight(max_results=25, unread_only=True)
                lines = [f"- {m['date']} — {m['from']} — {m['subject']}" for m in msgs]
                summary_prompt = (
                    "Summarize these overnight emails into 5–8 concise bullets. "
                    "Group related threads, call out anything urgent, and suggest 3 next actions:\n\n"
                    + "\n".join(lines)
                )
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Gmail check failed: {e}"}

            _append_session(project, user_input, response_data)
            return _render(project, response_data)

        # ---- Command: Gmail search (multiple aliases) ----
        if user_input.lower().startswith(("search ", "find ", "email about ")):
            # Extract query after the command
            for prefix in ["search ", "find ", "email about "]:
                if user_input.lower().startswith(prefix):
                    query_text = user_input[len(prefix):].strip()
                    break
            
            try:
                msgs = gmail_search(query_text, max_results=25)
                lines = [f"- {m['date']} — {m['from']} — {m['subject']}" for m in msgs]
                summary_prompt = (
                    f"Summarize the most relevant messages for query: '{query_text}'. "
                    "Give key points, who it's from, and any required follow‑ups:\n\n"
                    + "\n".join(lines)
                )
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Gmail search failed: {e}"}

            _append_session(project, user_input, response_data)
            return _render(project, response_data)

        # ---- Command: Today's calendar ----
        if user_input.lower().strip() in ["calendar", "today", "meetings", "schedule"]:
            try:
                events = list_today_events(max_results=20)
                calendar_summary = format_calendar_summary(events, "Today's Calendar")
                
                summary_prompt = (
                    f"Here's Carl's calendar for today. Summarize the key meetings and suggest priorities:\n\n"
                    f"{calendar_summary}"
                )
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Calendar check failed: {e}"}

            _append_session(project, user_input, response_data)
            return _render(project, response_data)

        # ---- Command: Tomorrow's calendar ----
        if user_input.lower().strip() in ["tomorrow", "tomorrow's schedule", "next day"]:
            try:
                events = list_tomorrow_events(max_results=20)
                calendar_summary = format_calendar_summary(events, "Tomorrow's Calendar")
                
                summary_prompt = (
                    f"Here's Carl's calendar for tomorrow. Highlight important meetings and prep needed:\n\n"
                    f"{calendar_summary}"
                )
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Tomorrow's calendar failed: {e}"}

            _append_session(project, user_input, response_data)
            return _render(project, response_data)

        # ---- Command: Next meeting ----
        if user_input.lower().strip() in ["next meeting", "next", "upcoming"]:
            try:
                next_meeting = get_next_meeting(hours_ahead=48)
                if next_meeting:
                    summary_prompt = (
                        f"Carl's next meeting: {next_meeting['summary']} at {next_meeting['start_formatted']} "
                        f"on {next_meeting['start'][:10]}. "
                        f"Location: {next_meeting.get('location', 'Not specified')}. "
                        f"Give a brief overview and any prep suggestions."
                    )
                else:
                    summary_prompt = "No upcoming meetings found in the next 48 hours."
                
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Next meeting check failed: {e}"}

            _append_session(project, user_input, response_data)
            return _render(project, response_data)

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
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Calendar search failed: {e}"}

            _append_session(project, user_input, response_data)
            return _render(project, response_data)

        # ---- Command: Good Morning ----
        if user_input.lower().strip() in ["good morning", "morning", "gm"]:
            try:
                # Get overnight emails and today's calendar
                msgs = list_overnight(max_results=25, unread_only=True)
                events = list_today_events(max_results=20)
                next_meeting = get_next_meeting(hours_ahead=24)
                
                # Format briefing
                email_lines = [f"• {m['date']} — {m['from']} — {m['subject']}" for m in msgs[:10]]
                calendar_summary = format_calendar_summary(events, "Today's Schedule")
                
                morning_briefing = f"""Good morning! Here's your daily briefing:

**OVERNIGHT EMAILS ({len(msgs)} total)**
{chr(10).join(email_lines) if email_lines else "No new emails"}

**TODAY'S CALENDAR**
{calendar_summary}

**NEXT MEETING**
{f"{next_meeting['summary']} at {next_meeting['start_formatted']}" if next_meeting else "No meetings scheduled"}

**PRIORITIES FOR TODAY**
• Review urgent emails
• Prepare for upcoming meetings
• Check calendar for conflicts"""

                # Save to daily log
                _save_daily_log("morning", morning_briefing)
                
                # Generate AI response
                retrieval_ctx = retrieve(morning_briefing, k=5) if is_ready() else []
                response_data = generate_response(
                    f"Summarize this morning briefing and suggest 3 key priorities:\n\n{morning_briefing}",
                    use_voices, random_toggle, project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                
            except Exception as e:
                response_data = {"SyntaxPrime": f"Morning briefing failed: {e}"}

            _append_session(project, user_input, response_data)
            return _render(project, response_data)

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
                _save_daily_log("evening", evening_summary)
                
                # Generate AI response
                retrieval_ctx = retrieve(evening_summary, k=5) if is_ready() else []
                response_data = generate_response(
                    f"Summarize this evening wrap-up and suggest 3 things to prepare for tomorrow:\n\n{evening_summary}",
                    use_voices, random_toggle, project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
                
            except Exception as e:
                response_data = {"SyntaxPrime": f"Evening summary failed: {e}"}

            _append_session(project, user_input, response_data)
            return _render(project, response_data)

        # ---- Command: scrape <url> ----
        if user_input.lower().startswith("scrape "):
            url = user_input.split(" ", 1)[1].strip()
            result = scrape_url(url)
            if not result["ok"]:
                response_data = {"SyntaxPrime": f"Could not fetch/extract content: {result['error']}"}
            else:
                summary_prompt = (
                    "Summarize the key points from the following webpage for Carl. "
                    "Use bullets and keep it tight and actionable.\n\n"
                    f"--- SCRAPED CONTENT START ---\n{result['text']}\n--- SCRAPED CONTENT END ---"
                )
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            _append_session(project, user_input, response_data)
            return _render(project, response_data)

        # ---- Normal flow ----
        retrieval_ctx = retrieve(user_input, k=5) if is_ready() else []
        response_data = generate_response(
            user_input, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )
        _append_session(project, user_input, response_data)

    return _render(selected_project, response_data)


def _append_session(project: str, user_input: str, response_data: dict):
    path = f"sessions/{project.lower().replace(' ', '_')}.json"
    with open(path, 'a', encoding='utf-8') as f:
        json.dump({'prompt': user_input, 'response': response_data}, f)
        f.write('\n')


def _render(project: str, response_data: dict):
    conversation = load_conversation(project, limit=50)
    return render_template(
        'index.html',
        projects=PROJECTS,
        response_data=response_data,
        conversation=conversation,
        current_project=project
    )


# --- BACKUP ALL PROJECTS ---
@app.route('/backup_all')
def backup_all():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        # Create temporary file for the zip
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.close()
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            backup_count = 0
            
            # Add all session files
            if os.path.exists('sessions'):
                for filename in os.listdir('sessions'):
                    if filename.endswith('.json'):
                        file_path = os.path.join('sessions', filename)
                        zipf.write(file_path, f"sessions/{filename}")
                        backup_count += 1
            
            # Add daily logs if they exist
            if os.path.exists('daily_logs'):
                for filename in os.listdir('daily_logs'):
                    if filename.endswith('.md'):
                        file_path = os.path.join('daily_logs', filename)
                        zipf.write(file_path, f"daily_logs/{filename}")
            
            # Create backup manifest
            manifest = f"""# Ghostline Backup Manifest
Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Session files backed up: {backup_count}
Projects: {', '.join(PROJECTS)}

## Contents:
- /sessions/ - All conversation history
- /daily_logs/ - Daily sync summaries (if any)

## Restore Instructions:
1. Extract this ZIP file
2. Copy session files to your sessions/ directory
3. Copy daily_logs to your daily_logs/ directory
"""
            zipf.writestr("backup_manifest.md", manifest)
        
        # Send the zip file
        backup_name = f"ghostline_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        return send_file(
            temp_zip.name,
            mimetype='application/zip',
            as_attachment=True,
            download_name=backup_name
        )
        
    except Exception as e:
        return f"Backup failed: {e}", 500


# --- BRAIN BUILDING ENDPOINTS ---
@app.route('/build_brain', methods=['POST'])
def build_brain():
    """Manually trigger batched brain building"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _rag_building
    
    if _rag_building:
        return jsonify({"ok": False, "error": "Brain is already building"}), 400
    
    if is_ready():
        return jsonify({"ok": False, "error": "Brain is already built"}), 400
    
    # Start building in background
    thread = threading.Thread(target=build_brain_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({"ok": True, "message": "Batched brain building started"})

@app.route('/build_new_brain', methods=['POST'])
def build_new_brain():
    """Build brain from raw sources on server"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _brain_building
    
    if _brain_building:
        return jsonify({"ok": False, "error": "Brain is already building"}), 400
    
    # Start server-side building in background
    thread = threading.Thread(target=build_new_brain_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({"ok": True, "message": "Server-side brain building started"})

@app.route('/brain_status')
def brain_status():
    """Enhanced brain status with batch progress"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _rag_building, _rag_build_error, _brain_building, _brain_build_error
    
    # Get detailed build status from the batched system
    build_status = get_build_status()
    
    # Check if server-side building is in progress
    if _brain_building:
        status = {
            "ready": False,
            "building": True,
            "progress": "Building brain from raw sources on server...",
            "error": _brain_build_error,
            "percentage": 50,  # Indeterminate progress
            "chunks": 0,
            "batches_completed": 0,
            "total_batches": 1
        }
    else:
        status = {
            "ready": build_status["status"] == "complete",
            "building": _rag_building or build_status["status"] == "building", 
            "progress": build_status["progress"],
            "error": _rag_build_error or _brain_build_error,
            "percentage": build_status["percentage"],
            "chunks": build_status.get("chunks_processed", 0),
            "batches_completed": build_status.get("batches_completed", 0),
            "total_batches": build_status.get("total_batches", 0)
        }
    
    return jsonify(status)

# --- BRAIN CONTROL PAGE ---
@app.route('/brain')
def brain_control():
    """Enhanced brain control dashboard with batch progress"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghostline Brain Control v0.1.9.7</title>
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
            .btn:disabled { background: #666; cursor: not-allowed; }
            .btn.server-build { background: #059669; }
            .btn.server-build:hover { background: #047857; }
            
            /* Enhanced progress bar for batches */
            .progress-container { 
                margin: 15px 0;
                background: #333; 
                border: 2px inset #666;
                height: 40px; 
                border-radius: 8px;
                position: relative;
                overflow: hidden;
            }
            .progress-bar { 
                background: linear-gradient(90deg, #10b981 0%, #34d399 50%, #10b981 100%);
                height: 100%; 
                transition: width 0.8s ease;
                position: relative;
                min-width: 0;
                border-radius: 6px;
            }
            .progress-bar::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: repeating-linear-gradient(
                    45deg,
                    transparent,
                    transparent 12px,
                    rgba(255,255,255,0.15) 12px,
                    rgba(255,255,255,0.15) 24px
                );
                animation: slide 2s linear infinite;
            }
            @keyframes slide {
                0% { transform: translateX(-24px); }
                100% { transform: translateX(24px); }
            }
            .progress-text {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
                z-index: 10;
                font-size: 16px;
            }
            
            /* Batch progress section */
            .batch-info {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin: 15px 0;
            }
            .batch-stat {
                background: #2a2a2a;
                padding: 12px;
                border-radius: 6px;
                text-align: center;
            }
            .batch-stat .number {
                font-size: 24px;
                font-weight: bold;
                color: #10b981;
            }
            .batch-stat .label {
                font-size: 12px;
                color: #888;
                margin-top: 4px;
            }
            
            #status { font-family: monospace; font-size: 14px; }
            .error { color: #ef4444; }
            .success { color: #10b981; }
            .building { color: #f59e0b; }
            .eta { 
                font-size: 12px; 
                color: #888; 
                margin-top: 8px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Ghostline Brain Control v0.1.9.7</h1>
            <p>Batched RAG system with server-side brain building from raw sources.</p>
            
            <div class="status-box">
                <h3>Brain Status</h3>
                <div id="status">Loading...</div>
                
                <div id="progress-container" class="progress-container" style="display: none;">
                    <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
                    <div class="progress-text" id="progress-text">0%</div>
                </div>
                
                <div id="batch-info" class="batch-info" style="display: none;">
                    <div class="batch-stat">
                        <div class="number" id="chunks-processed">0</div>
                        <div class="label">Chunks Processed</div>
                    </div>
                    <div class="batch-stat">
                        <div class="number" id="batches-completed">0</div>
                        <div class="label">Batches Complete</div>
                    </div>
                </div>
                
                <div id="eta" class="eta"></div>
            </div>
            
            <div class="status-box">
                <h3>Controls</h3>
                <button class="btn" id="build-btn" onclick="buildBrain()">Build Brain (from file)</button>
                <button class="btn server-build" id="server-build-btn" onclick="buildNewBrain()">Build Brain (from sources)</button>
                <button class="btn" onclick="refreshStatus()">Refresh Status</button>
                <button class="btn" onclick="window.location.href='/'">Back to Chat</button>
            </div>
            
            <div class="status-box">
                <h3>Build Options</h3>
                <p><strong>Build Brain (from file):</strong> Uses existing brain file with batched processing.</p>
                <p><strong>Build Brain (from sources):</strong> Creates fresh brain from raw HTML/TXT/JSON files on server.</p>
                <p><strong>Memory Safe:</strong> Both approaches work with Railway's 32GB RAM.</p>
                <p><strong>Auto-Resume:</strong> Batched processing continues from last completed batch.</p>
            </div>
        </div>
        
        <script>
            let statusInterval;
            
            function refreshStatus() {
                fetch('/brain_status')
                    .then(r => r.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        const buildBtn = document.getElementById('build-btn');
                        const serverBuildBtn = document.getElementById('server-build-btn');
                        const progressContainer = document.getElementById('progress-container');
                        const progressBar = document.getElementById('progress-bar');
                        const progressText = document.getElementById('progress-text');
                        const batchInfo = document.getElementById('batch-info');
                        const etaDiv = document.getElementById('eta');
                        
                        let statusText = '';
                        
                        if (data.ready) {
                            statusText = `<span class="success">✓ Brain Ready</span><br>Total chunks: ${data.chunks.toLocaleString()}`;
                            buildBtn.disabled = true;
                            buildBtn.textContent = 'Brain Complete';
                            serverBuildBtn.disabled = true;
                            serverBuildBtn.textContent = 'Brain Complete';
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                            etaDiv.textContent = '';
                            
                        } else if (data.building) {
                            statusText = `<span class="building">⚡ Building Brain...</span><br>${data.progress}`;
                            
                            if (data.percentage > 0) {
                                progressContainer.style.display = 'block';
                                progressBar.style.width = data.percentage + '%';
                                progressText.textContent = `${data.percentage}%`;
                                
                                // Show batch info
                                if (data.total_batches > 0) {
                                    batchInfo.style.display = 'grid';
                                    document.getElementById('chunks-processed').textContent = data.chunks.toLocaleString();
                                    document.getElementById('batches-completed').textContent = `${data.batches_completed}/${data.total_batches}`;
                                    
                                    etaDiv.textContent = `Batch ${data.batches_completed + 1} of ${data.total_batches} in progress`;
                                }
                            } else {
                                progressContainer.style.display = 'none';
                                batchInfo.style.display = 'none';
                            }
                            
                            buildBtn.disabled = true;
                            buildBtn.textContent = 'Building...';
                            serverBuildBtn.disabled = true;
                            serverBuildBtn.textContent = 'Building...';
                            
                        } else if (data.error) {
                            statusText = `<span class="error">✗ Build Failed</span><br>${data.error}`;
                            buildBtn.disabled = false;
                            buildBtn.textContent = 'Retry Build (from file)';
                            serverBuildBtn.disabled = false;
                            serverBuildBtn.textContent = 'Retry Build (from sources)';
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                            etaDiv.textContent = '';
                            
                        } else {
                            statusText = '<span style="color: #fbbf24;">◯ Brain Not Built</span><br>Ready for building';
                            buildBtn.disabled = false;
                            buildBtn.textContent = 'Build Brain (from file)';
                            serverBuildBtn.disabled = false;
                            serverBuildBtn.textContent = 'Build Brain (from sources)';
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                            etaDiv.textContent = '';
                        }
                        
                        statusDiv.innerHTML = statusText;
                    })
                    .catch(e => {
                        document.getElementById('status').innerHTML = `<span class="error">Connection error: ${e}</span>`;
                    });
            }
            
            function buildBrain() {
                fetch('/build_brain', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        if (data.ok) {
                            statusInterval = setInterval(refreshStatus, 3000);
                        } else {
                            alert('Build failed: ' + data.error);
                        }
                    })
                    .catch(e => alert('Build request failed: ' + e));
            }
            
            function buildNewBrain() {
                fetch('/build_new_brain', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        if (data.ok) {
                            statusInterval = setInterval(refreshStatus, 3000);
                        } else {
                            alert('Build failed: ' + data.error);
                        }
                    })
                    .catch(e => alert('Build request failed: ' + e));
            }
            
            // Initial status check
            refreshStatus();
            
            // Auto-refresh every 5 seconds
            setInterval(refreshStatus, 5000);
        </script>
    </body>
    </html>
    """


# --- DEBUG ROUTES ---
@app.route('/debug/sessions')
def debug_sessions():
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        if not os.path.exists('sessions'):
            return "<pre>sessions/ directory does not exist</pre>"
        
        files = os.listdir('sessions')
        result = [f"=== Sessions Directory Debug ===\n"]
        result.append(f"Directory exists: Yes")
        result.append(f"Files found: {len(files)}\n")
        
        for filename in files:
            filepath = f"sessions/{filename}"
            size = os.path.getsize(filepath)
            result.append(f"File: {filename} ({size} bytes)")
            
            # Show first few lines of each file
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:3]  # First 3 lines
                for i, line in enumerate(lines, 1):
                    result.append(f"  Line {i}: {line.strip()[:100]}...")
        
        return "<pre>" + "\n".join(result) + "</pre>"
        
    except Exception as e:
        return f"<pre>Error checking sessions: {e}</pre>"

@app.route('/debug/files')
def debug_files():
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        result = ["=== Debug Files Report ===\n"]
        
        # Check data/cleaned directory
        if os.path.exists('data/cleaned/'):
            files = os.listdir('data/cleaned/')
            file_info = []
            for f in files:
                path = os.path.join('data/cleaned/', f)
                size = os.path.getsize(path) if os.path.isfile(path) else 0
                file_info.append(f"{f} ({size} bytes)")
            result.append(f"Files in data/cleaned/: {file_info}\n")
        else:
            result.append("data/cleaned/ directory not found\n")
        
        # Check for raw data folders
        raw_folders = []
        for item in os.listdir('data/'):
            if item.startswith('raw_') and os.path.isdir(f'data/{item}'):
                file_count = len(list(os.listdir(f'data/{item}')))
                raw_folders.append(f"{item} ({file_count} files)")
        
        if raw_folders:
            result.append(f"Raw data folders: {raw_folders}")
        else:
            result.append("No raw data folders found")
        
        return "<pre>" + "\n".join(result) + "</pre>"
        
    except Exception as e:
        return f"Error checking files: {e}"


# --- STREAMING (plain text) ---
@app.route('/stream', methods=['POST'])
def stream():
    if not session.get('logged_in'):
        return "Unauthorized", 401
    user_input = request.form['user_input'].strip()
    project = request.form['project']
    use_voices = request.form.getlist('voices') or ['SyntaxPrime']
    retrieval_ctx = retrieve(user_input, k=5) if is_ready() else []

    def generate():
        for chunk in stream_generate(
            user_input, use_voices, project=project,
            model=CHAT_MODEL, retrieval_context=retrieval_ctx
        ):
            yield chunk

    return app.response_class(generate(), mimetype='text/plain')


# --- RELOAD BRAIN ---
@app.route('/reload_corpus')
def reload_corpus():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        load_corpus(CORPUS_PATH)
        return "Brain reloaded", 200
    except Exception as e:
        return f"Reload failed: {e}", 500


# --- HEALTH CHECK ---
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


# --- DEBUG RAG: see what the retriever returns ---
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


# --- DEBUG: Sample entries to see data structure ---
@app.route('/debug/sample')
def debug_sample():
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        # Get a few sample entries to see their structure
        import gzip
        samples = []
        with gzip.open('data/cleaned/ghostline_sources.jsonl.gz', 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Just first 5 entries
                    break
                try:
                    entry = json.loads(line)
                    # Only show metadata, not full content
                    sample = {k: v for k, v in entry.items() if k != 'content'}
                    sample['content_length'] = len(entry.get('content', ''))
                    samples.append(sample)
                except:
                    continue
        
        return jsonify({"ok": True, "samples": samples})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


# --- DEBUG: Check EasyOCR status ---
@app.route('/debug/ocr')
def debug_ocr():
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    try:
        import easyocr
        import numpy as np
        
        # Test EasyOCR initialization
        reader = easyocr.Reader(['en'])
        
        return "<pre>EasyOCR is working!\n\nSupported languages: English\nReady for image analysis!</pre>"
        
    except ImportError as e:
        return f"<pre>EasyOCR not installed: {str(e)}</pre>"
    except Exception as e:
        return f"<pre>EasyOCR error: {str(e)}</pre>"


# --- AUTH ---
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


# --- EXPORT SESSION ---
@app.route('/export/<project>')
def export_session(project):
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


# --- UPLOAD / OCR ---
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>Upload & Analyze</title></head>
        <body style="font-family: Arial; padding: 20px; background: #0f0f0f; color: white;">
            <h2>Upload & Analyze File</h2>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="file" required accept="image/*,.pdf,.docx" style="margin: 10px 0;">
                <br>
                <button type="submit" style="background: #6366f1; color: white; border: none; padding: 10px 20px; border-radius: 5px;">Analyze</button>
            </form>
        </body>
        </html>
        '''
    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return "No file uploaded", 400
        
        filename = file.filename.lower()
        text = ""

        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                import easyocr
                import numpy as np
                
                file.stream.seek(0)
                img = Image.open(file.stream)
                img_array = np.array(img)
                
                reader = easyocr.Reader(['en'])
                results = reader.readtext(img_array)
                text = '\n'.join([result[1] for result in results if result[1].strip()])
                
                if not text.strip():
                    text = "No text detected in image"
                    
            except Exception as e:
                return f"OCR Error: {str(e)}. EasyOCR processing failed.", 500
                
        elif filename.endswith('.pdf'):
            try:
                file.stream.seek(0)
                data = file.read()
                doc = fitz.open(stream=data, filetype="pdf")
                text = "".join(page.get_text() for page in doc)
                if not text.strip():
                    text = "No text found in PDF"
            except Exception as e:
                return f"PDF Error: {str(e)}", 500
                
        elif filename.endswith('.docx'):
            try:
                file.stream.seek(0)
                document = docx.Document(file)
                text = "\n".join(p.text for p in document.paragraphs)
                if not text.strip():
                    text = "No text found in Word document"
            except Exception as e:
                return f"Word Document Error: {str(e)}", 500
        else:
            return "Unsupported file type. Supported: PNG, JPG, JPEG, PDF, DOCX", 400

        if len(text) > 10000:
            text = text[:10000] + "\n\n[...truncated...]"
            
        return f"<pre>{text}</pre>"
        
    except Exception as e:
        return f"Upload Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)