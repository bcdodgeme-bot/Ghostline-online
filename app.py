from flask import Flask, render_template, request, redirect, session, url_for, send_file, jsonify
from utils.ghostline_engine import generate_response, stream_generate
from utils.scraper import scrape_url
from utils.gmail_client import (
    list_overnight, search as gmail_search,
    list_today_events, list_tomorrow_events, search_calendar, 
    get_next_meeting, format_calendar_summary
)
import os, json, io
import threading
import time

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
_rag_system = None
_rag_building = False
_rag_build_progress = ""
_rag_build_error = None

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

# RAG functions that work with global state
def retrieve(query: str, k: int = 5):
    """Retrieve relevant context using the RAG system"""
    global _rag_system
    if not _rag_system:
        return []
    
    try:
        # Import here to avoid startup issues
        from utils.rag_basic import SimpleRAG
        results = _rag_system.search(query, top_k=k)
        return [{"text": result["text"], "source": result["source"]} for result in results]
    except Exception as e:
        app.logger.error(f"RAG retrieval error: {e}")
        return []

def is_ready():
    """Check if RAG system is ready"""
    global _rag_system
    return _rag_system is not None

def build_brain_background():
    """Build the RAG system in background"""
    global _rag_system, _rag_building, _rag_build_progress, _rag_build_error
    
    try:
        _rag_building = True
        _rag_build_progress = "Initializing RAG system..."
        app.logger.info("Starting brain build...")
        
        # Import RAG system
        from utils.rag_basic import SimpleRAG
        
        _rag_build_progress = "Creating RAG instance..."
        rag = SimpleRAG()
        
        _rag_build_progress = "Building index from ChatGPT history..."
        rag.build_index(CORPUS_PATH)
        
        _rag_build_progress = "Brain build complete!"
        _rag_system = rag
        _rag_building = False
        
        app.logger.info(f"Brain build successful! Loaded {len(rag.chunks)} chunks")
        
    except Exception as e:
        _rag_building = False
        _rag_build_error = str(e)
        _rag_build_progress = f"Build failed: {e}"
        app.logger.error(f"Brain build failed: {e}")


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
                    "Summarize these overnight emails into 5—8 concise bullets. "
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


# --- BRAIN BUILDING ENDPOINTS ---
@app.route('/build_brain', methods=['POST'])
def build_brain():
    """Manually trigger brain building"""
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
    
    return jsonify({"ok": True, "message": "Brain building started"})

@app.route('/brain_status')
def brain_status():
    """Check brain building status"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _rag_system, _rag_building, _rag_build_progress, _rag_build_error
    
    status = {
        "ready": is_ready(),
        "building": _rag_building,
        "progress": _rag_build_progress,
        "error": _rag_build_error,
        "chunks": len(_rag_system.chunks) if _rag_system else 0
    }
    
    return jsonify(status)

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


# --- HEALTH CHECK ---
@app.route('/healthz')
def healthz():
    global _rag_system, _rag_building, _rag_build_progress
    
    status = {
        "status": "ok",
        "brain_ready": is_ready(),
        "brain_building": _rag_building,
        "brain_progress": _rag_build_progress,
        "brain_chunks": len(_rag_system.chunks) if _rag_system else 0
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


# --- BRAIN CONTROL PAGE ---
@app.route('/brain')
def brain_control():
    """Brain control dashboard"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghostline Brain Control</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f0f; 
                color: #fff; 
                margin: 0; 
                padding: 20px; 
            }
            .container { max-width: 800px; margin: 0 auto; }
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
            .progress { 
                background: #333; 
                border-radius: 4px; 
                height: 8px; 
                margin: 10px 0; 
                overflow: hidden;
            }
            .progress-bar { 
                background: #6366f1; 
                height: 100%; 
                transition: width 0.3s ease;
            }
            #status { font-family: monospace; }
            .error { color: #ef4444; }
            .success { color: #10b981; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Ghostline Brain Control</h1>
            <p>Manage the RAG system that gives Syntax Prime access to your ChatGPT history.</p>
            
            <div class="status-box">
                <h3>Brain Status</h3>
                <div id="status">Loading...</div>
                <div id="progress-container" style="display: none;">
                    <div class="progress">
                        <div class="progress-bar" id="progress-bar"></div>
                    </div>
                </div>
            </div>
            
            <div class="status-box">
                <h3>Controls</h3>
                <button class="btn" id="build-btn" onclick="buildBrain()">Build Brain</button>
                <button class="btn" onclick="refreshStatus()">Refresh Status</button>
                <button class="btn" onclick="window.location.href='/'">Back to Chat</button>
            </div>
            
            <div class="status-box">
                <h3>Info</h3>
                <p><strong>What this does:</strong> Processes your 41MB ChatGPT history file into searchable chunks.</p>
                <p><strong>Time required:</strong> 5-10 minutes on first build.</p>
                <p><strong>Memory usage:</strong> High during build, normal after completion.</p>
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
                        
                        let statusText = '';
                        if (data.ready) {
                            statusText = `<span class="success">✓ Brain Ready</span><br>Chunks loaded: ${data.chunks}`;
                            buildBtn.disabled = true;
                            buildBtn.textContent = 'Brain Already Built';
                        } else if (data.building) {
                            statusText = `<span style="color: #f59e0b;">⚡ Building...</span><br>${data.progress}`;
                            buildBtn.disabled = true;
                            buildBtn.textContent = 'Building...';
                        } else if (data.error) {
                            statusText = `<span class="error">✗ Error</span><br>${data.error}`;
                            buildBtn.disabled = false;
                            buildBtn.textContent = 'Retry Build';
                        } else {
                            statusText = '<span style="color: #fbbf24;">○ Brain Not Built</span><br>Ready to build';
                            buildBtn.disabled = false;
                            buildBtn.textContent = 'Build Brain';
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
                            // Start monitoring progress
                            statusInterval = setInterval(refreshStatus, 2000);
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
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return "No file uploaded", 400
        
        filename = file.filename.lower()
        text = ""

        if filename.endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Use EasyOCR (no system dependencies required)
                import easyocr
                import numpy as np
                
                # Convert PIL image to numpy array
                file.stream.seek(0)  # Reset stream position
                img = Image.open(file.stream)
                img_array = np.array(img)
                
                # Initialize EasyOCR reader
                reader = easyocr.Reader(['en'])
                
                # Extract text
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

        # Truncate very long text
        if len(text) > 10000:
            text = text[:10000] + "\n\n[...truncated...]"
            
        return f"<pre>{text}</pre>"
        
    except Exception as e:
        return f"Upload Error: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
