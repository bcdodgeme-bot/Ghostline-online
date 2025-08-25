from flask import Flask, render_template, request, redirect, session, url_for, send_file, jsonify
from utils.ghostline_engine import generate_response, stream_generate
from utils.rag_basic import retrieve, is_ready, load_corpus, get_build_status
from utils.scraper import scrape_url
from utils.gmail_client import (
    list_overnight, search as gmail_search,
    list_today_events, list_tomorrow_events, search_calendar,
    get_next_meeting, format_calendar_summary
)
import os, json, io, zipfile, tempfile, datetime
from zoneinfo import ZoneInfo  # tz-aware times

# ---------- Safe markdown filter ----------
try:
    import markdown as _mdpkg
    from markupsafe import Markup as _Markup
except Exception:
    _mdpkg = None
    _Markup = lambda x: x  # passthrough if markdown unavailable

def markdown_filter(text: str):
    if not text:
        return ""
    if _mdpkg:
        md = _mdpkg.Markdown(extensions=['nl2br', 'fenced_code'])
        return _Markup(md.convert(text))
    return text

# ---------- App setup ----------
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'ghostline-default-key')
app.jinja_env.filters['markdown'] = markdown_filter

# No login gate by default; set GHOSTLINE_PASSWORD to enable it.
PASSWORD = os.getenv('GHOSTLINE_PASSWORD')

CHAT_MODEL = os.getenv("CHAT_MODEL", os.getenv("OPENROUTER_MODEL", "openrouter/auto"))
os.makedirs("sessions", exist_ok=True)

# Local timezone for Gmail/Calendar and summaries (overridable)
LOCAL_TZ = ZoneInfo(os.getenv('APP_TIMEZONE', 'America/New_York'))

PROJECTS = [
    'Personal Operating Manual',
    'AMCF',
    'Meals N Feelz',
    'Damn It Carl',
    'TV Signals',
    'Health'
]

# ---------- Persistence ----------
def _append_session(project, user, data):
    try:
        fname = os.path.join("sessions", f"{project}.jsonl")
        with open(fname, "a", encoding="utf-8") as f:
            f.write(json.dumps({"user": user, "data": data}) + "\n")
    except Exception as e:
        print("append_session error:", e)

def _load_session(project):
    fname = os.path.join("sessions", f"{project}.jsonl")
    if not os.path.exists(fname):
        return []
    out = []
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

# ---------- Calendar/Gmail helpers ----------
def _event_iso_to_local(iso_str: str):
    """Convert RFC3339/ISO timestamps from Google to LOCAL_TZ-aware datetime."""
    try:
        if not iso_str:
            return None
        dt = datetime.datetime.fromisoformat(iso_str.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo('UTC'))
        return dt.astimezone(LOCAL_TZ)
    except Exception:
        return None

def _save_daily_log(sync_type: str, content: str):
    try:
        os.makedirs("daily_logs", exist_ok=True)
        day = datetime.datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
        fn = os.path.join("daily_logs", f"{day}_{sync_type}.md")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print("save_daily_log error:", e)

def _render(project, response_data):
    try:
        convo = []
        for item in _load_session(project):
            if 'user' in item and 'data' in item and isinstance(item['data'], dict):
                u = item['user']
                d = item['data']
                convo.append({"user": u, "responses": d.get("responses", {})})
        return render_template(
            "index.html",
            conversation=convo,
            response=response_data,
            projects=PROJECTS,
            current_project=project
        )
    except Exception as e:
        return render_template(
            "index.html",
            conversation=[],
            response={"responses": {"SyntaxPrime": f"Render failed: {e}"}},
            projects=PROJECTS,
            current_project=project
        )

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def home():
    project = request.form.get("project") or request.args.get("project") or "Personal Operating Manual"
    session.setdefault('authed', True)  # no-login default

    if request.method == "POST":
        user_input = (request.form.get("user_input") or "").strip()

        # voices + random toggle from the form (defaults safe)
        use_voices = request.form.getlist("voices") or ["SyntaxPrime"]
        random_toggle = ("random" in request.form) or (request.form.get("random") == "on")

        # Optional tiny login if PASSWORD is set
        if PASSWORD and user_input.lower().startswith("login:"):
            pwd = user_input.split("login:", 1)[-1].strip()
            if pwd == PASSWORD:
                session['authed'] = True
                return redirect(url_for('home', project=project))
            resp = {"responses": {"SyntaxPrime": "Invalid password."}}
            _append_session(project, user_input, resp)
            return _render(project, resp)

        if PASSWORD and not session.get('authed'):
            resp = {"responses": {"SyntaxPrime": "Please log in to use Ghostline."}}
            _append_session(project, user_input, resp)
            return _render(project, resp)

        low = user_input.lower()

        # ----- Good Morning -----
        if low in ["good morning", "morning", "gm"]:
            try:
                overnight = list_overnight(include_unread=False, include_primary=False)
                today_events = list_today_events(max_results=10)
                summary = (
                    f"**Good morning!**\n"
                    f"Overnight emails: {len(overnight)}\n"
                    f"Today's meetings: {len(today_events)}\n\n"
                    f"{format_calendar_summary(today_events[:5], 'Top of day:')}"
                )
                _save_daily_log("morning", summary)
                retrieval_ctx = retrieve("morning briefing summary")
                combined = f"{summary}\n\n{retrieval_ctx or ''}"
                bot = generate_response(
                    prompt=combined,
                    voices=use_voices,
                    random_toggle=random_toggle,
                    project=project,
                    model=CHAT_MODEL,
                    retrieval_context=retrieval_ctx
                )
                resp = {"responses": bot if isinstance(bot, dict) else {"SyntaxPrime": str(bot)}}
            except Exception as e:
                resp = {"responses": {"SyntaxPrime": f"Morning briefing failed: {e}"}}
            _append_session(project, user_input, resp)
            return _render(project, resp)

        # ----- Good Evening / Wrap up -----
        if low in ["good evening", "evening", "ge", "wrap up", "day summary"]:
            try:
                today_events = list_today_events(max_results=20)
                tomorrow_events = list_tomorrow_events(max_results=15)

                now = datetime.datetime.now(LOCAL_TZ)
                completed_events, upcoming_events = [], []
                for event in today_events:
                    start_iso = event.get('start')
                    start_dt_local = _event_iso_to_local(start_iso)
                    if start_dt_local and start_dt_local <= now:
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
                _save_daily_log("evening", evening_summary)
                retrieval_ctx = retrieve("evening wrap up")
                combined = f"{evening_summary}\n\n{retrieval_ctx or ''}"
                bot = generate_response(
                    prompt=combined,
                    voices=use_voices,
                    random_toggle=random_toggle,
                    project=project,
                    model=CHAT_MODEL,
                    retrieval_context=retrieval_ctx
                )
                resp = {"responses": bot if isinstance(bot, dict) else {"SyntaxPrime": str(bot)}}
            except Exception as e:
                resp = {"responses": {"SyntaxPrime": f"Evening wrap-up failed: {e}"}}
            _append_session(project, user_input, resp)
            return _render(project, resp)

        # ----- Default chat flow -----
        try:
            retrieval_ctx = retrieve(user_input) if user_input else ""
            prompt = f"{user_input}\n\n{retrieval_ctx or ''}"
            bot = generate_response(
                prompt=prompt,
                voices=use_voices,
                random_toggle=random_toggle,
                project=project,
                model=CHAT_MODEL,
                retrieval_context=retrieval_ctx
            )
            resp = {"responses": bot if isinstance(bot, dict) else {"SyntaxPrime": str(bot)}}
        except Exception as e:
            resp = {"responses": {"SyntaxPrime": f"Chat failed: {e}"}}

        _append_session(project, user_input, resp)
        return _render(project, resp)

    # GET
    return _render(project, {})

# ---------- Upload / OCR (your handler can replace the placeholder) ----------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    try:
        if request.method == "POST":
            f = request.files.get("file")
            if not f:
                return "No file uploaded", 400

            # >>> Plug your real OCR pipeline here <<<
            data = f.read()
            size_kb = round(len(data) / 1024, 1)
            return f"Upload received ({size_kb} KB). Your OCR handler can run here."
        return "Use POST with multipart/form-data"
    except Exception as e:
        return f"Upload failed: {e}", 500

# ---------- Convenience endpoints ----------
@app.route("/next_meeting")
def next_meeting():
    try:
        meeting = get_next_meeting()
        return jsonify(meeting)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/export/<project>")
def export_project(project):
    try:
        content = []
        for item in _load_session(project):
            if 'user' in item and 'data' in item:
                content.append(f"You: {item['user']}")
                responses = item['data'].get('responses', {})
                for voice, reply in responses.items():
                    content.append(f"{voice}:\n{reply}\n")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        tmp.write("\n".join(content).encode("utf-8"))
        tmp.flush()
        tmp.seek(0)
        path = tmp.name
        return send_file(path, as_attachment=True, download_name=f"{project}.txt")
    except Exception as e:
        return f"Export failed: {e}", 500

@app.route("/backup_all")
def backup_all():
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp.name, 'w', zipfile.ZIP_DEFLATED) as z:
            for proj in PROJECTS:
                p = os.path.join("sessions", f"{proj}.jsonl")
                if os.path.exists(p):
                    z.write(p, arcname=os.path.basename(p))
        return send_file(tmp.name, as_attachment=True, download_name="ghostline_sessions.zip")
    except Exception as e:
        return f"Backup failed: {e}", 500

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("DEBUG", "0") == "1")

