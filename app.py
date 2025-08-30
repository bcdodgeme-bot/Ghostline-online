# Section 1: Imports and Initial Flask Setup

from flask import Flask, render_template, request, redirect, session, url_for, send_file, jsonify, render_template_string
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
import requests  # Add this if not already present
from modules.cloze_integration import process_cloze_command, is_cloze_configured

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
# Section 5: Utility Functions

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

# Section 6: Main Route with Enhanced Database Functionality

# Section 6: Main Route with Enhanced Database Functionality
# Section 6: Main Route with Enhanced Database Functionality

from modules.gmail import process_gmail_command
from modules.cloze_integration import process_cloze_command, is_cloze_configured
from utils.scraper import scrape_url

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

        # Try Gmail/calendar commands first
        response_data, handled = process_gmail_command(user_input, project, use_voices, random_toggle)
        if handled:
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

# Section 11: PDF Generation Routes
# Section 11: PDF Generation Routes

from modules.pdf_generation import (
    generate_project_pdf,
    generate_daily_briefing_pdf,
    generate_project_report,
    generate_daily_briefing_report
)
import datetime

@app.route('/reports/<project_name>.pdf')
def project_report_pdf(project_name):
    """Generate project report as PDF"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    try:
        # Get optional date range parameter
        days = request.args.get('days', 30, type=int)
        days = min(days, 365)  # Limit to 1 year max
        
        # Generate PDF
        pdf_bytes, temp_path = generate_project_pdf(project_name, days)
        
        # Safe filename for download
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
        # Parse date
        report_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        
        # Generate PDF
        pdf_bytes, temp_path = generate_daily_briefing_pdf(report_date)
        
        # Safe filename for download
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
        days = min(days, 365)  # Limit to 1 year max
        
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
        # Parse date
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
                <button class="btn secondary" onclick="window.location.href='/brain'">Brain Dashboard</button>
                <button class="btn secondary" onclick="window.location.href='/database'">Database Dashboard</button>
            </div>
        </div>
    </body>
    </html>
    '''
    
    return render_template_string(html_content, projects=PROJECTS)

# Section 12: Cloze CRM Integration

from modules.cloze_integration import (
    process_cloze_command,
    get_cloze_morning_briefing,
    get_cloze_pipeline_summary,
    search_cloze_contacts,
    log_ghostline_interaction_to_cloze,
    is_cloze_configured
)

# Update the main index route to include Cloze command processing
# Add this to the existing index() function after Gmail command processing:

def index_with_cloze():
    """Enhanced index route with Cloze integration"""
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

        # Try Gmail/calendar commands first
        response_data, handled = process_gmail_command(user_input, project, use_voices, random_toggle)
        if handled:
            return _render_enhanced(project, response_data)

        # Try Cloze commands
        if is_cloze_configured():
            response_data, handled = process_cloze_command(user_input, project, use_voices, random_toggle)
            if handled:
                # Log to Cloze if it's a regular interaction (not a Cloze command itself)
                save_conversation_enhanced(project, user_input, response_data)
                return _render_enhanced(project, response_data)

        # ---- Command: scrape <url> ---- (existing code continues...)

@app.route('/cloze/status')
def cloze_status():
    """Check Cloze API configuration and connection"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    status = {
        "configured": is_cloze_configured(),
        "api_key_present": bool(os.getenv('CLOZE_API_KEY')),
        "connection_working": False,
        "user_info": None
    }
    
    if is_cloze_configured():
        try:
            from modules.cloze_integration import ClozeClient
            client = ClozeClient()
            profile = client.get_profile()
            status["connection_working"] = True
            status["user_info"] = {
                "name": profile.get('name', 'Unknown'),
                "email": profile.get('email', 'Unknown')
            }
        except Exception as e:
            status["error"] = str(e)
    
    return jsonify(status)

@app.route('/cloze/briefing')
def cloze_briefing():
    """Get Cloze morning briefing"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if not is_cloze_configured():
        return "Cloze API not configured", 400
    
    try:
        briefing = get_cloze_morning_briefing()
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cloze Briefing</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #0f0f0f; color: #fff; margin: 0; padding: 20px; line-height: 1.6;
                }
                .container { max-width: 800px; margin: 0 auto; }
                .btn { 
                    background: #6366f1; color: white; border: none; padding: 12px 24px;
                    border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
                    text-decoration: none; display: inline-block;
                }
                .btn:hover { background: #5855eb; }
                pre { background: #1a1a1a; padding: 20px; border-radius: 8px; white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cloze Morning Briefing</h1>
                <pre>{{ briefing }}</pre>
                <a href="/" class="btn">← Back to Chat</a>
                <a href="/cloze" class="btn">Cloze Dashboard</a>
            </div>
        </body>
        </html>
        ''', briefing=briefing)
        
    except Exception as e:
        return f"Briefing generation failed: {str(e)}", 500

@app.route('/cloze/pipeline')
def cloze_pipeline():
    """Get Cloze pipeline summary"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    if not is_cloze_configured():
        return "Cloze API not configured", 400
    
    try:
        pipeline = get_cloze_pipeline_summary()
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cloze Pipeline</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #0f0f0f; color: #fff; margin: 0; padding: 20px; line-height: 1.6;
                }
                .container { max-width: 800px; margin: 0 auto; }
                .btn { 
                    background: #6366f1; color: white; border: none; padding: 12px 24px;
                    border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
                    text-decoration: none; display: inline-block;
                }
                .btn:hover { background: #5855eb; }
                pre { background: #1a1a1a; padding: 20px; border-radius: 8px; white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cloze Pipeline Summary</h1>
                <pre>{{ pipeline }}</pre>
                <a href="/" class="btn">← Back to Chat</a>
                <a href="/cloze" class="btn">Cloze Dashboard</a>
            </div>
        </body>
        </html>
        ''', pipeline=pipeline)
        
    except Exception as e:
        return f"Pipeline summary failed: {str(e)}", 500

@app.route('/cloze')
def cloze_dashboard():
    """Cloze integration dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Cloze Integration Dashboard</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f0f; 
                color: #fff; 
                margin: 0; 
                padding: 20px; 
            }
            .container { max-width: 1000px; margin: 0 auto; }
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
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover { background: #5855eb; }
            .btn.secondary { background: #374151; }
            .btn.secondary:hover { background: #4b5563; }
            .commands-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .command-card {
                background: #2a2a2a;
                padding: 15px;
                border-radius: 8px;
            }
            .command-title {
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #6366f1;
            }
            .command-example {
                background: #1a1a1a;
                padding: 10px;
                border-radius: 4px;
                font-family: monospace;
                margin-top: 10px;
            }
            .success { color: #10b981; }
            .error { color: #ef4444; }
            .warning { color: #f59e0b; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Cloze CRM Integration</h1>
            
            <div class="status-box">
                <h3>Connection Status</h3>
                <div id="status">Loading...</div>
            </div>
            
            <div class="status-box">
                <h3>Quick Actions</h3>
                <a href="/cloze/briefing" class="btn">Morning Briefing</a>
                <a href="/cloze/pipeline" class="btn">Pipeline Summary</a>
                <a href="/cloze/status" class="btn secondary">API Status</a>
            </div>
            
            <div class="status-box">
                <h3>Available Chat Commands</h3>
                <div class="commands-grid">
                    <div class="command-card">
                        <div class="command-title">Morning Briefing</div>
                        <p>Get daily activity summary and active projects from Cloze</p>
                        <div class="command-example">cloze morning</div>
                        <div class="command-example">morning cloze</div>
                    </div>
                    
                    <div class="command-card">
                        <div class="command-title">Pipeline Summary</div>
                        <p>View deals and pipeline status by stage</p>
                        <div class="command-example">cloze pipeline</div>
                        <div class="command-example">cloze deals</div>
                    </div>
                    
                    <div class="command-card">
                        <div class="command-title">Contact Search</div>
                        <p>Search for contacts in your Cloze database</p>
                        <div class="command-example">cloze search john smith</div>
                        <div class="command-example">cloze search acme corp</div>
                    </div>
                </div>
            </div>
            
            <div class="status-box">
                <h3>Setup Instructions</h3>
                <ol>
                    <li>Email <strong>support@cloze.com</strong> to request API access</li>
                    <li>Get your API key from Cloze Pro settings</li>
                    <li>Set <strong>CLOZE_API_KEY</strong> environment variable</li>
                    <li>Restart Ghostline to activate integration</li>
                </ol>
            </div>
            
            <div class="status-box">
                <button class="btn secondary" onclick="window.location.href='/'">← Back to Chat</button>
                <button class="btn secondary" onclick="window.location.href='/brain'">Brain Dashboard</button>
                <button class="btn secondary" onclick="window.location.href='/database'">Database Dashboard</button>
            </div>
        </div>
        
        <script>
            function refreshStatus() {
                fetch('/cloze/status')
                    .then(r => r.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        
                        if (!data.configured) {
                            statusDiv.innerHTML = '<span class="warning">API Key Not Configured</span><br>Set CLOZE_API_KEY environment variable';
                        } else if (data.connection_working && data.user_info) {
                            statusDiv.innerHTML = `<span class="success">Connected to Cloze</span><br>User: ${data.user_info.name} (${data.user_info.email})`;
                        } else {
                            statusDiv.innerHTML = `<span class="error">Connection Failed</span><br>${data.error || 'Unknown error'}`;
                        }
                    })
                    .catch(e => {
                        document.getElementById('status').innerHTML = '<span class="error">Status Check Failed</span>';
                    });
            }
            
            refreshStatus();
            setInterval(refreshStatus, 30000);
        </script>
    </body>
    </html>
    '''
    return html_content

# Section 13: Marketing FLUX Integration Routes

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
        
        # Required fields
        concept = data.get('concept')
        if not concept:
            return jsonify({'success': False, 'error': 'Concept description required'}), 400
        
        # Optional parameters with smart defaults
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
        if result['success'] and 'database' in globals():
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

@app.route('/api/marketing/test-concepts', methods=['POST'])
def api_marketing_test_concepts():
    """Rapidly test multiple creative concepts"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
    try:
        data = request.get_json()
        
        concepts = data.get('concepts', [])
        if not concepts or not isinstance(concepts, list):
            return jsonify({
                'success': False, 
                'error': 'Concepts array required'
            }), 400
        
        if len(concepts) > 10:
            return jsonify({
                'success': False,
                'error': 'Maximum 10 concepts per test batch'
            }), 400
        
        style = data.get('style', 'corporate')
        
        generator = MarketingFluxGenerator()
        result = generator.test_concepts(concepts, style)
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Concept testing failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Concept testing failed: {str(e)}'
        }), 500

@app.route('/api/marketing/social-set', methods=['POST'])
def api_marketing_social_set():
    """Create social media asset set"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
    try:
        data = request.get_json()
        
        concept = data.get('concept')
        if not concept:
            return jsonify({'success': False, 'error': 'Post concept required'}), 400
        
        platforms = data.get('platforms', ['instagram', 'facebook', 'linkedin', 'twitter'])
        style = data.get('style', 'corporate')
        
        generator = MarketingFluxGenerator()
        result = generator.create_social_media_set(concept, platforms, style)
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Social media set creation failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Social media set creation failed: {str(e)}'
        }), 500

@app.route('/api/marketing/formats')
def api_marketing_formats():
    """Get available formats and specifications"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
    try:
        generator = MarketingFluxGenerator()
        
        return jsonify({
            'success': True,
            'social_formats': generator.social_specs,
            'styles': generator.marketing_styles,
            'models': generator.models,
            'recommendations': generator.get_marketing_recommendations()
        })
        
    except Exception as e:
        app.logger.error(f"Formats fetch failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/marketing/download', methods=['POST'])
def api_marketing_download():
    """Download generated marketing asset"""
    if not session.get('logged_in'):
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        filename = data.get('filename', 'marketing_asset')
        campaign_name = data.get('campaign_name', 'general')
        
        if not image_url:
            return jsonify({'success': False, 'error': 'Image URL required'}), 400
        
        # Create organized folder structure
        import re
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_campaign = re.sub(r'[^a-zA-Z0-9_-]', '_', campaign_name)
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
        
        folder_path = os.path.join("static", "marketing", safe_campaign)
        file_path = os.path.join(folder_path, f"{safe_filename}_{timestamp}.png")
        
        # Download image
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Ensure directory exists
        os.makedirs(folder_path, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        # Return web-accessible path
        web_path = f"/static/marketing/{safe_campaign}/{os.path.basename(file_path)}"
        
        return jsonify({
            'success': True,
            'local_path': file_path,
            'web_path': web_path,
            'filename': os.path.basename(file_path),
            'campaign_folder': safe_campaign
        })
        
    except Exception as e:
        app.logger.error(f"Download failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Download failed: {str(e)}'
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

@app.route('/marketing/campaigns')
def marketing_campaigns():
    """Campaign management page"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    # Simple campaigns view
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Marketing Campaigns</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Marketing Campaigns</h1>
            <p>Campaign history and management coming soon!</p>
            <a href="/marketing" class="btn">← Back to Marketing Dashboard</a>
            <a href="/" class="btn">Chat Interface</a>
        </div>
    </body>
    </html>
    ''')


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



