from flask import Flask, render_template, request, redirect, session, url_for, send_file, jsonify
from utils.ghostline_engine import generate_response, stream_generate
from utils.rag_basic import retrieve, is_ready, load_corpus  # RAG imports
from utils.scraper import scrape_url                         # Scraper
from utils.gmail_client import list_overnight, search as gmail_search
import os
import json
import io

# OCR/File Parsing Imports
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import docx

# Optional: load from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'ghostline-default-key')
PASSWORD = os.getenv('GHOSTLINE_PASSWORD', 'open_the_gate')

# Make sure sessions dir exists
os.makedirs("sessions", exist_ok=True)

PROJECTS = [
    'Personal Operating Manual',
    'AMCF',
    'BCDodgeme',
    'Rose and Angel',
    'Meals N Feelz',
    'TV Signals',
    'Damn It Carl',
    'HalalBot',
    'Kitchen',
    'Health',
    'Side Quests'
]

CORPUS_PATH = "data/cleaned/ghostline_sources.jsonl.gz"

# --- Auto-load the corpus (works with Flask 3.x) ---
_corpus_loaded = False

def _ensure_corpus_loaded():
    """Load once on import and once again on first request if needed."""
    global _corpus_loaded
    if _corpus_loaded:
        return
    try:
        load_corpus(CORPUS_PATH)
        app.logger.info("✅ Brain loaded from %s", CORPUS_PATH)
        _corpus_loaded = True
    except Exception as e:
        app.logger.warning("⚠️ Brain load failed: %s", e)

# Try at import (covers gunicorn worker start, local run)
_ensure_corpus_loaded()

# And ensure on first actual HTTP request (Flask 3.x safe)
@app.before_request
def _load_once_before_request():
    _ensure_corpus_loaded()

@app.route('/', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    response_data = {}
    if request.method == 'POST':
        user_input = request.form['user_input'].strip()
        project = request.form['project']
        use_voices = request.form.getlist('voices') or ['SyntaxPrime']
        random_toggle = 'random' in request.form

        # --- Command: scrape <url> ---
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
                retrieval_ctx = retrieve(summary_prompt, k=5, project_filter=project) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model="llama3", retrieval_context=retrieval_ctx
                )

            # Save and return
            session_path = f"sessions/{project.lower().replace(' ', '_')}.json"
            with open(session_path, 'a') as f:
                json.dump({'prompt': user_input, 'response': response_data}, f)
                f.write('\n')
            return render_template('index.html', projects=PROJECTS, response_data=response_data)

        # --- Command: gmail overnight ---
        if user_input.lower().strip() == "gmail overnight":
            try:
                msgs = list_overnight(max_results=25, unread_only=True)
                lines = [f"- {m['date']} — {m['from']} — {m['subject']}" for m in msgs]
                summary_prompt = (
                    "Summarize these overnight emails into 5–8 concise bullets. "
                    "Group related threads, call out anything urgent, and suggest 3 next actions:\n\n"
                    + "\n".join(lines)
                )
                retrieval_ctx = retrieve(summary_prompt, k=5, project_filter=project) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model="llama3", retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Gmail check failed: {e}"}

            session_path = f"sessions/{project.lower().replace(' ', '_')}.json"
            with open(session_path, 'a') as f:
                json.dump({'prompt': user_input, 'response': response_data}, f)
                f.write('\n')
            return render_template('index.html', projects=PROJECTS, response_data=response_data)

        # --- Command: gmail search <query> ---
        if user_input.lower().startswith("gmail search "):
            query_text = user_input.split(" ", 2)[2].strip()
            try:
                msgs = gmail_search(query_text, max_results=25)
                lines = [f"- {m['date']} — {m['from']} — {m['subject']}" for m in msgs]
                summary_prompt = (
                    f"Summarize the most relevant messages for query: '{query_text}'. "
                    "Give me key points, who it’s from, and any required follow-ups:\n\n"
                    + "\n".join(lines)
                )
                retrieval_ctx = retrieve(summary_prompt, k=5, project_filter=project) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model="llama3", retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Gmail search failed: {e}"}

            session_path = f"sessions/{project.lower().replace(' ', '_')}.json"
            with open(session_path, 'a') as f:
                json.dump({'prompt': user_input, 'response': response_data}, f)
                f.write('\n')
            return render_template('index.html', projects=PROJECTS, response_data=response_data)

        # --- Normal flow: retrieve knowledge + generate ---
        retrieval_ctx = retrieve(user_input, k=5, project_filter=project) if is_ready() else []
        response_data = generate_response(
            user_input, use_voices, random_toggle,
            project=project, model="llama3", retrieval_context=retrieval_ctx
        )

        # Save to project session file
        session_path = f"sessions/{project.lower().replace(' ', '_')}.json"
        with open(session_path, 'a') as f:
            json.dump({'prompt': user_input, 'response': response_data}, f)
            f.write('\n')

    return render_template('index.html', projects=PROJECTS, response_data=response_data)

# --- STREAMING ENDPOINT (plain text stream) ---
@app.route('/stream', methods=['POST'])
def stream():
    if not session.get('logged_in'):
        return "Unauthorized", 401

    user_input = request.form['user_input'].strip()
    project = request.form['project']
    use_voices = request.form.getlist('voices') or ['SyntaxPrime']

    retrieval_ctx = retrieve(user_input, k=5, project_filter=project) if is_ready() else []

    def generate():
        for chunk in stream_generate(
            user_input, use_voices, project=project,
            model="llama3", retrieval_context=retrieval_ctx
        ):
            yield chunk

    return app.response_class(generate(), mimetype='text/plain')

# --- RELOAD BRAIN (no restart needed) ---
@app.route('/reload_corpus')
def reload_corpus():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    try:
        load_corpus(CORPUS_PATH)
        # mark as loaded in case it errored earlier
        global _corpus_loaded
        _corpus_loaded = True
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

@app.route('/export/<project>')
def export_session(project):
    session_path = f"sessions/{project.lower().replace(' ', '_')}.json"
    try:
        with open(session_path, 'r') as f:
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

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file or not file.filename:
        return "No file uploaded", 400

    filename = file.filename.lower()

    if filename.endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(file.stream)
        text = pytesseract.image_to_string(img)
    elif filename.endswith('.pdf'):
        file.stream.seek(0)
        data = file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
    elif filename.endswith('.docx'):
        file.stream.seek(0)
        document = docx.Document(file)
        text = "\n".join(p.text for p in document.paragraphs)
    else:
        return "Unsupported file type", 400

    return f"<pre>{text}</pre>"

if __name__ == '__main__':
    app.run(debug=True)

