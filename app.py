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
except Exception:
    pass

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'ghostline-default-key')
PASSWORD = os.getenv('GHOSTLINE_PASSWORD', 'open_the_gate')

# Central place to pick the chat model (can be overridden in Render env)
CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))

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

# gz OK; rag loader handles .gz
CORPUS_PATH = "data/cleaned/ghostline_sources.jsonl.gz"

# --- Auto-load the corpus at startup / on (re)deploy (Flask 3 safe) ---
def _boot_load_corpus():
    try:
        load_corpus(CORPUS_PATH)
        app.logger.info("✅ Brain loaded from %s", CORPUS_PATH)
    except Exception as e:
        app.logger.warning("⚠️ Brain load failed: %s", e)

_boot_load_corpus()

# --- conversation loader (tail last N turns) ---
def load_conversation(project: str, limit: int = 50):
    """
    Reads sessions/<project>.json and returns a list of turns like:
      {"user": "...", "responses": {"SyntaxPrime": "...", ...}}
    Most recent at the end.
    """
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
                retrieval_ctx = retrieve(summary_prompt, k=5, project_filter=project) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Gmail check failed: {e}"}

            session_path = f"sessions/{project.lower().replace(' ', '_')}.json"
            with open(session_path, 'a', encoding='utf-8') as f:
                json.dump({'prompt': user_input, 'response': response_data}, f); f.write('\n')

            conversation = load_conversation(project, limit=50)
            return render_template(
                'index.html',
                projects=PROJECTS,
                response_data=response_data,
                conversation=conversation,
                current_project=project
            )

        # ---- Command: gmail search <query> ----
        if user_input.lower().startswith("gmail search "):
            query_text = user_input.split(" ", 2)[2].strip()
            try:
                msgs = gmail_search(query_text, max_results=25)
                lines = [f"- {m['date']} — {m['from']} — {m['subject']}" for m in msgs]
                summary_prompt = (
                    f"Summarize the most relevant messages for query: '{query_text}'. "
                    "Give key points, who it’s from, and any required follow‑ups:\n\n"
                    + "\n".join(lines)
                )
                retrieval_ctx = retrieve(summary_prompt, k=5, project_filter=project) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                response_data = {"SyntaxPrime": f"Gmail search failed: {e}"}

            session_path = f"sessions/{project.lower().replace(' ', '_')}.json"
            with open(session_path, 'a', encoding='utf-8') as f:
                json.dump({'prompt': user_input, 'response': response_data}, f); f.write('\n')

            conversation = load_conversation(project, limit=50)
            return render_template(
                'index.html',
                projects=PROJECTS,
                response_data=response_data,
                conversation=conversation,
                current_project=project
            )

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
                retrieval_ctx = retrieve(summary_prompt, k=5, project_filter=project) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )

            # Save and return
            session_path = f"sessions/{project.lower().replace(' ', '_')}.json"
            with open(session_path, 'a', encoding='utf-8') as f:
                json.dump({'prompt': user_input, 'response': response_data}, f)
                f.write('\n')

            conversation = load_conversation(project, limit=50)
            return render_template(
                'index.html',
                projects=PROJECTS,
                response_data=response_data,
                conversation=conversation,
                current_project=project
            )

        # ---- Normal flow: retrieve + generate ----
        retrieval_ctx = retrieve(user_input, k=5, project_filter=project) if is_ready() else []
        response_data = generate_response(
            user_input, use_voices, random_toggle,
            project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
        )

        # Save to project session file
        session_path = f"sessions/{project.lower().replace(' ', '_')}.json"
        with open(session_path, 'a', encoding='utf-8') as f:
            json.dump({'prompt': user_input, 'response': response_data}, f)
            f.write('\n')

    conversation = load_conversation(selected_project, limit=50)
    return render_template(
        'index.html',
        projects=PROJECTS,
        response_data=response_data,
        conversation=conversation,
        current_project=selected_project
    )

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
            model=CHAT_MODEL, retrieval_context=retrieval_ctx
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




