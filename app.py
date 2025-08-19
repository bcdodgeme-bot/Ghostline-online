from flask import Flask, render_template, request, redirect, session, url_for, send_file, jsonify
from utils.ghostline_engine import generate_response, stream_generate
from utils.rag_basic import retrieve, is_ready, load_corpus
from utils.scraper import scrape_url
from utils.gmail_client import list_overnight, search as gmail_search
import os, json, io

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

def _boot_load_corpus():
    try:
        load_corpus(CORPUS_PATH)
        app.logger.info("✅ Brain loaded from %s", CORPUS_PATH)
    except Exception as e:
        app.logger.warning("⚠️ Brain load failed: %s", e)
_boot_load_corpus()


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

        # ---- Command: gmail overnight ----
        if user_input.lower().strip() == "gmail overnight":
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

        # ---- Command: gmail search <query> ----
        if user_input.lower().startswith("gmail search "):
            query_text = user_input.split(" ", 2)[2].strip()
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
        return "Brain reloaded ✅", 200
    except Exception as e:
        return f"Reload failed: {e}", 500


# --- HEALTH CHECK ---
@app.route('/healthz')
def healthz():
    ok = True
    details = {}
    try:
        details["corpus_loaded"] = bool(is_ready())
    except Exception as e:
        ok = False
        details["corpus_error"] = str(e)
    status = {"status": "ok" if ok else "error", **details}
    return jsonify(status), (200 if ok else 500)


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
        return jsonify({"ok": False, "error": "corpus not loaded"}), 500
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
        
        return "<pre>✅ EasyOCR is working!\n\nSupported languages: English\nReady for image analysis!</pre>"
        
    except ImportError as e:
        return f"<pre>❌ EasyOCR not installed: {str(e)}</pre>"
    except Exception as e:
        return f"<pre>❌ EasyOCR error: {str(e)}</pre>"


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
    app.run(debug=True)

