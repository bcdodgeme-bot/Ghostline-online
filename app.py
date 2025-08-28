# Part 1: Complete imports and initial Flask setup

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
# Part 2: Database connection and initialization functions

# Database connection helper - NEW
@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        if DATABASE_URL:
            conn = psycopg2.connect(DATABASE_URL)
            yield conn
        else:
            app.logger.warning("No DATABASE_URL found - using file storage only")
            yield None
    except Exception as e:
        app.logger.error(f"Database connection failed: {e}")
        if conn:
            conn.rollback()
        yield None
    finally:
        if conn:
            conn.close()

# Database initialization - NEW
def init_database():
    """Create necessary database tables"""
    if not DATABASE_URL:
        app.logger.info("No database URL - running in file-only mode")
        return
    
    with get_db_connection() as conn:
        if not conn:
            return
            
        cursor = conn.cursor()
        
        try:
            # Create chat_threads table for conversation storage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_threads (
                    id SERIAL PRIMARY KEY,
                    project VARCHAR(100) NOT NULL,
                    user_input TEXT NOT NULL,
                    response_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_chat_threads_project_date 
                ON chat_threads (project, created_at DESC)
            ''')
            
            # Create uploaded_files table for file tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    content_preview TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    project VARCHAR(100) NOT NULL,
                    processing_status VARCHAR(50) DEFAULT 'completed'
                )
            ''')
            
            # Create user_settings table for preferences
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_settings (
                    id SERIAL PRIMARY KEY,
                    setting_key VARCHAR(100) UNIQUE NOT NULL,
                    setting_value JSONB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create daily_logs table for briefings
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_logs (
                    id SERIAL PRIMARY KEY,
                    log_date DATE NOT NULL,
                    log_type VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(log_date, log_type)
                )
            ''')
            
            # Create brain_documents table for RAG system storage
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS brain_documents (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    title VARCHAR(500),
                    content TEXT NOT NULL,
                    embedding_vector FLOAT8[] NULL,
                    chunk_index INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for brain_documents
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_brain_docs_id ON brain_documents (document_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_brain_docs_title ON brain_documents (title)')
            
            conn.commit()
            app.logger.info("Database tables initialized successfully")
            
        except Exception as e:
            conn.rollback()
            app.logger.error(f"Database initialization failed: {e}")


# Enhanced session management with database backup - NEW
def load_conversation_enhanced(project: str, limit: int = 50):
    """Load conversation history from database first, then fallback to file"""
    conversations = []
    
    # Try database first
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute('''
                    SELECT user_input, response_data, created_at 
                    FROM chat_threads 
                    WHERE project = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                ''', (project, limit))
                
                rows = cursor.fetchall()
                for row in rows:
                    conversations.append({
                        "user": row['user_input'],
                        "responses": row['response_data'],
                        "timestamp": row['created_at'].isoformat()
                    })
                
                # Reverse to get chronological order
                conversations.reverse()
                app.logger.info(f"Loaded {len(conversations)} conversations from database for {project}")
                return conversations
                
            except Exception as e:
                app.logger.error(f"Failed to load conversations from database: {e}")
    
    # Fallback to file system (your existing function)
    app.logger.info(f"Falling back to file system for {project} conversations")
    return load_conversation(project, limit)

def save_conversation_enhanced(project: str, user_input: str, response_data: dict):
    """Save conversation to both database and file"""
    
    # Save to database first
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO chat_threads (project, user_input, response_data)
                    VALUES (%s, %s, %s)
                ''', (project, user_input, psycopg2.extras.Json(response_data)))
                
                conn.commit()
                app.logger.info(f"Conversation saved to database for {project}")
                
            except Exception as e:
                app.logger.error(f"Failed to save conversation to database: {e}")
                conn.rollback()
    
    # Also save to file as backup (your existing function)
    _append_session(project, user_input, response_data)

def save_daily_log_enhanced(sync_type: str, content: str):
    """Save daily log to both database and file"""
    today = datetime.datetime.now().date()
    
    # Save to database
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO daily_logs (log_date, log_type, content)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (log_date, log_type) 
                    DO UPDATE SET content = EXCLUDED.content, created_at = CURRENT_TIMESTAMP
                ''', (today, sync_type, content))
                
                conn.commit()
                app.logger.info(f"Daily log saved to database: {sync_type}")
                
            except Exception as e:
                app.logger.error(f"Failed to save daily log to database: {e}")
                conn.rollback()
    
    # Also save to file as backup
    _save_daily_log(sync_type, content)

def track_uploaded_file(filename: str, file_type: str, project: str, content_preview: str = ""):
    """Track uploaded files in database"""
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO uploaded_files (filename, file_type, project, content_preview)
                    VALUES (%s, %s, %s, %s)
                ''', (filename, file_type, project, content_preview[:500]))  # Limit preview length
                
                conn.commit()
                app.logger.info(f"File upload tracked: {filename}")
                
            except Exception as e:
                app.logger.error(f"Failed to track uploaded file: {e}")
                conn.rollback()
def save_brain_to_database(corpus_data):
    """Save processed brain corpus to database"""
    with get_db_connection() as conn:
        if not conn:
            app.logger.warning("No database connection - brain will only be saved to file")
            return False
            
        try:
            cursor = conn.cursor()
            
            # Clear existing brain data
            cursor.execute('DELETE FROM brain_documents')
            app.logger.info("Cleared existing brain documents from database")
            
            # Insert new brain data
            saved_count = 0
            for item in corpus_data:
                cursor.execute('''
                    INSERT INTO brain_documents (document_id, title, content, chunk_index, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                ''', (
                    item.get('id', 'unknown'),
                    item.get('title', '')[:500],  # Limit title length
                    item.get('content', ''),
                    item.get('chunk_index', 0),
                    psycopg2.extras.Json(item.get('metadata', {}))
                ))
                saved_count += 1
            
            conn.commit()
            app.logger.info(f"Saved {saved_count} brain documents to database")
            return True
            
        except Exception as e:
            app.logger.error(f"Failed to save brain to database: {e}")
            conn.rollback()
            return False

def load_brain_from_database():
    """Load brain corpus from database"""
    with get_db_connection() as conn:
        if not conn:
            return None
            
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('''
                SELECT document_id, title, content, chunk_index, metadata 
                FROM brain_documents 
                ORDER BY document_id, chunk_index
            ''')
            
            rows = cursor.fetchall()
            corpus_data = []
            
            for row in rows:
                corpus_data.append({
                    'id': row['document_id'],
                    'title': row['title'],
                    'content': row['content'],
                    'chunk_index': row['chunk_index'],
                    'metadata': row['metadata'] or {}
                })
            
            app.logger.info(f"Loaded {len(corpus_data)} brain documents from database")
            return corpus_data
            
        except Exception as e:
            app.logger.error(f"Failed to load brain from database: {e}")
            return None

def enhanced_build_brain_background():
    """Enhanced brain building with database storage - works with chunked files"""
    global _rag_building, _rag_build_error
    
    try:
        _rag_building = True
        _rag_build_error = None
        app.logger.info("Starting enhanced brain build with database integration...")
        
        # Build the brain using existing corpus (this handles the chunked files)
        load_corpus(CORPUS_PATH)
        
        # Now save to database by extracting data from the loaded RAG system
        try:
            # Import your RAG system to access the loaded data
            from utils.rag_basic import _rag_system
            
            if _rag_system and hasattr(_rag_system, 'chunks') and _rag_system.chunks:
                app.logger.info(f"Found {len(_rag_system.chunks)} chunks in loaded RAG system")
                
                # Convert RAG chunks to database format
                corpus_data = []
                for i, chunk in enumerate(_rag_system.chunks):
                    # Adapt to your RAG system's chunk format
                    corpus_item = {
                        'id': str(chunk.get('id', f'chunk_{i}')),
                        'title': chunk.get('source', f'chunk_{i}'),
                        'content': chunk.get('text', ''),
                        'chunk_index': i,
                        'metadata': {
                            'created_at': chunk.get('created_at', ''),
                            'source': chunk.get('source', ''),
                            'batch': chunk.get('batch', 0)
                        }
                    }
                    corpus_data.append(corpus_item)
                
                # Save to database
                if save_brain_to_database(corpus_data):
                    app.logger.info("Brain successfully saved to database from RAG system")
                else:
                    app.logger.warning("Brain build completed but database save failed")
            else:
                app.logger.warning("No chunks found in RAG system - skipping database save")
        
        except Exception as db_error:
            app.logger.error(f"Database save failed during brain build: {db_error}")
        
        _rag_building = False
        app.logger.info("Enhanced brain build complete!")
        
    except Exception as e:
        _rag_building = False
        _rag_build_error = str(e)
        app.logger.error(f"Enhanced brain build failed: {e}")

def enhanced_build_new_brain_background():
    """Enhanced new brain building from sources with database storage"""
    global _brain_building, _brain_build_error
    
    try:
        _brain_building = True
        _brain_build_error = None
        app.logger.info("Starting enhanced new brain build from raw sources...")
        
        from build_brain_fixed2 import build_new_brain
        result_path = build_new_brain()
        
        app.logger.info(f"New brain built with result path: {result_path}")
        
        # Load the new brain into the RAG system
        load_corpus(CORPUS_PATH)
        
        # Now try to save to database
        try:
            from utils.rag_basic import _rag_system
            
            if _rag_system and hasattr(_rag_system, 'chunks') and _rag_system.chunks:
                app.logger.info(f"Found {len(_rag_system.chunks)} chunks in newly built RAG system")
                
                corpus_data = []
                for i, chunk in enumerate(_rag_system.chunks):
                    corpus_item = {
                        'id': str(chunk.get('id', f'chunk_{i}')),
                        'title': chunk.get('source', f'chunk_{i}'),
                        'content': chunk.get('text', ''),
                        'chunk_index': i,
                        'metadata': {
                            'created_at': chunk.get('created_at', ''),
                            'source': chunk.get('source', ''),
                            'batch': chunk.get('batch', 0)
                        }
                    }
                    corpus_data.append(corpus_item)
                
                if save_brain_to_database(corpus_data):
                    app.logger.info("New brain successfully saved to database")
                else:
                    app.logger.warning("New brain build completed but database save failed")
            else:
                app.logger.warning("No chunks found in newly built RAG system")
        
        except Exception as db_error:
            app.logger.error(f"Database save failed during new brain build: {db_error}")
        
       
# Part 3: EasyOCR setup and processing functions (UNCHANGED)

# Initialize database when app starts - NEW
with app.app_context():
    init_database()

# Fix EasyOCR model directory permissions
def setup_easyocr_environment():
    """Setup writable directory for EasyOCR models"""
    try:
        # Create a writable temp directory for EasyOCR
        easyocr_dir = os.path.join(tempfile.gettempdir(), 'easyocr_models')
        os.makedirs(easyocr_dir, exist_ok=True)
        
        # Set environment variable to override EasyOCR's default path
        os.environ['EASYOCR_MODULE_PATH'] = easyocr_dir
        
        app.logger.info(f"EasyOCR model path set to: {easyocr_dir}")
        return True
    except Exception as e:
        app.logger.error(f"Failed to setup EasyOCR environment: {e}")
        return False

# Call this right after creating the Flask app
setup_easyocr_environment()

# Enhanced OCR processing function
def process_image_ocr(file_stream, filename):
    """Process image with EasyOCR, handling model download issues"""
    try:
        import easyocr
        import numpy as np
        
        app.logger.info(f"Starting OCR processing for: {filename}")
        
        # Reset stream position
        file_stream.seek(0)
        
        # Open and convert image
        img = Image.open(file_stream)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img)
        app.logger.info(f"Image loaded: {img.size}, mode: {img.mode}")
        
        # Initialize EasyOCR with error handling for model downloads
        try:
            # Try to create reader with custom model path
            reader = easyocr.Reader(
                ['en'], 
                gpu=False,  # Disable GPU for server compatibility
                download_enabled=True,  # Allow model downloads
                model_storage_directory=os.environ.get('EASYOCR_MODULE_PATH')
            )
            app.logger.info("EasyOCR reader initialized successfully")
            
        except Exception as model_error:
            app.logger.error(f"EasyOCR model initialization failed: {model_error}")
            
            # Fallback: try without custom directory
            reader = easyocr.Reader(['en'], gpu=False)
            app.logger.info("EasyOCR reader initialized with default settings")
        
        # Perform OCR
        results = reader.readtext(img_array)
        
        if results:
            text = '\n'.join([result[1] for result in results if result[1].strip()])
            app.logger.info(f"OCR extracted {len(results)} text regions, {len(text)} characters")
            return text
        else:
            app.logger.warning("No OCR results found")
            return "No text detected in image"
            
    except ImportError as e:
        app.logger.error(f"EasyOCR not installed: {e}")
        raise Exception("EasyOCR not installed. Please install with: pip install easyocr opencv-python-headless")
    
    except Exception as e:
        app.logger.error(f"OCR processing failed: {e}")
        raise Exception(f"OCR processing failed: {str(e)}")

# Vision analysis function for when OCR fails
def analyze_image_with_vision(file_stream, filename):
    """Analyze image using GPT-4 Vision when OCR results are poor"""
    try:
        import base64
        import requests
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            return "OpenAI API key not configured for vision analysis"
        
        # Reset stream and encode image
        file_stream.seek(0)
        image_data = base64.b64encode(file_stream.read()).decode('utf-8')
        
        # Determine image format for data URL
        file_extension = filename.split('.')[-1].lower()
        mime_type = f"image/{file_extension}" if file_extension in ['png', 'jpg', 'jpeg', 'gif', 'bmp'] else "image/jpeg"
        
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json"
        }
        
        # Create vision analysis prompt
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Analyze this image in detail. If it contains charts, graphs, screenshots, or data visualizations, describe the key insights, trends, and important information. If it's a photo, describe what you see. Be specific and actionable in your analysis."
                        },
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1
        }
        
        app.logger.info(f"Sending image to GPT-4 Vision for analysis: {filename}")
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            vision_analysis = result['choices'][0]['message']['content']
            app.logger.info(f"GPT-4 Vision analysis successful: {len(vision_analysis)} characters")
            return vision_analysis
        else:
            app.logger.error(f"GPT-4 Vision API error: {response.status_code} - {response.text}")
            return f"Vision analysis failed: API error {response.status_code}"
            
    except Exception as e:
        app.logger.error(f"Vision analysis failed: {e}")
        return f"Vision analysis error: {str(e)}"

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

# Part 4: Utility functions for brain building and session management

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

def build_brain_background():
    """Build the RAG system using batched processing - WITH PROGRESS TRACKING!"""
    global _rag_building, _rag_build_error
    
    try:
        _rag_building = True
        _rag_build_error = None
        app.logger.info("Starting batched brain build with progress tracking...")
        
        # Load corpus with progress tracking - this will show your loading bar!
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

# Part 5: Main route with enhanced database functionality

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
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
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
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
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
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
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
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
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
                
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
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
                retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
                response_data = generate_response(
                    summary_prompt, use_voices, random_toggle,
                    project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
                )
            except Exception as e:
                app.logger.error(f"Calendar search failed: {e}")
                response_data = {"SyntaxPrime": f"Calendar search failed: {e}"}

            save_conversation_enhanced(project, user_input, response_data)
            return _render_enhanced(project, response_data)

# Part 6: Continue main route - Good Morning, Evening, Scrape commands and normal flow

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
                retrieval_ctx = retrieve(morning_briefing, k=5) if is_ready() else []
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
                retrieval_ctx = retrieve(evening_summary, k=5) if is_ready() else []
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
                    retrieval_ctx = retrieve(summary_prompt, k=5) if is_ready() else []
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
            retrieval_ctx = retrieve(user_input, k=5) if is_ready() else []
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

# Part 7: Brain building endpoints and backup functionality

@app.route('/build_brain', methods=['POST'])
def build_brain():
    """Manually trigger enhanced brain building with database storage"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _rag_building
    
    if _rag_building:
        return jsonify({"ok": False, "error": "Brain is already building"}), 400
    
    if is_ready():
        return jsonify({"ok": False, "error": "Brain is already built"}), 400
    
    # Start enhanced building in background
    thread = threading.Thread(target=enhanced_build_brain_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({"ok": True, "message": "Enhanced brain building with database storage started"})

@app.route('/build_new_brain', methods=['POST'])
def build_new_brain():
    """Build new brain from raw sources with database storage"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _brain_building
    
    if _brain_building:
        return jsonify({"ok": False, "error": "Brain is already building"}), 400
    
    # Start enhanced building in background
    thread = threading.Thread(target=enhanced_build_new_brain_background)
    thread.daemon = True
    thread.start()
    
    return jsonify({"ok": True, "message": "Enhanced new brain building with database storage started"})

# Part 8: Brain control dashboard with enhanced loading bar

@app.route('/brain_status')
def brain_status():
    """Enhanced brain status with batch progress - YOUR LOADING BAR IS BACK!"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _rag_building, _rag_build_error, _brain_building, _brain_build_error
    
    # Get detailed build status from the batched system
    build_status = get_build_status()
    
    # Check if server-side building is in progress
    if _brain_building:
        status = {
            "ready": build_status["status"] == "complete",
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

@app.route('/brain')
def brain_control():
    """Enhanced brain control dashboard with batch progress - YOUR LOADING BAR IS BACK!"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghostline Brain Control v0.2.0</title>
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
                transition: all 0.3s ease;
            }
            .btn:hover { background: #5855eb; transform: translateY(-2px); }
            .btn:disabled { background: #666; cursor: not-allowed; transform: none; }
            .btn.server-build { background: #059669; }
            .btn.server-build:hover { background: #047857; }
            
            .progress-container { 
                margin: 20px 0;
                background: #333; 
                border: 2px solid #444;
                height: 50px; 
                border-radius: 12px;
                position: relative;
                overflow: hidden;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
            }
            .progress-bar { 
                background: linear-gradient(90deg, #10b981 0%, #34d399 30%, #6ee7b7 60%, #34d399 100%);
                height: 100%; 
                transition: width 0.8s ease;
                position: relative;
                min-width: 0;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
            }
            
            .progress-bar::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
                animation: shimmer 2s infinite;
            }
            
            @keyframes shimmer {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(100%); }
            }
            
            .progress-text {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-weight: bold;
                color: #fff;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
                z-index: 1;
            }
            
            .batch-info {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .batch-stat {
                background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #333;
                transition: transform 0.2s ease;
            }
            .batch-stat:hover { transform: translateY(-2px); }
            
            .batch-stat .number {
                font-size: 28px;
                font-weight: bold;
                color: #10b981;
                margin-bottom: 5px;
            }
            .batch-stat .label {
                font-size: 12px;
                color: #888;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            #status { 
                font-family: 'SF Mono', 'Monaco', 'Cascadia Code', monospace; 
                font-size: 16px;
                padding: 10px;
                border-radius: 6px;
                background: #000;
                border: 1px solid #333;
            }
            .error { color: #ef4444; }
            .success { color: #10b981; }
            .building { color: #f59e0b; }
            
            .pulse { animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Ghostline Brain Control v0.2.0</h1>
            <p>Enhanced RAG system with real-time progress tracking and batch processing.</p>
            
            <div class="status-box">
                <h3>🔍 Brain Status</h3>
                <div id="status">Loading brain status...</div>
                
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
                    <div class="batch-stat">
                        <div class="number" id="total-batches">0</div>
                        <div class="label">Total Batches</div>
                    </div>
                    <div class="batch-stat">
                        <div class="number" id="percentage">0%</div>
                        <div class="label">Progress</div>
                    </div>
                </div>
            </div>
            
            <div class="status-box">
                <h3>🎛️ Controls</h3>
                <button class="btn" id="build-btn" onclick="buildBrain()">Build Brain (from file)</button>
                <button class="btn server-build" id="server-build-btn" onclick="buildNewBrain()">Build Brain (from sources)</button>
                <button class="btn" onclick="refreshStatus()">🔄 Refresh Status</button>
                <button class="btn" onclick="window.location.href='/'">← Back to Chat</button>
            </div>
        </div>
        
        <script>
            function refreshStatus() {
                fetch('/brain_status')
                    .then(r => r.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        const buildBtn = document.getElementById('build-btn');
                        const serverBuildBtn = document.getElementById('server-build-btn');
                        const progressContainer = document.getElementById('progress-container');
                        const batchInfo = document.getElementById('batch-info');
                        
                        // Update basic status
                        if (data.ready) {
                            statusDiv.innerHTML = '<span class="success">✅ Brain Ready & Loaded</span>';
                            buildBtn.disabled = true;
                            serverBuildBtn.disabled = true;
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                        } else if (data.building) {
                            statusDiv.innerHTML = '<span class="building pulse">⚡ Building Brain...</span>';
                            buildBtn.disabled = true;
                            serverBuildBtn.disabled = true;
                            showProgress(data);
                        } else if (data.error) {
                            statusDiv.innerHTML = '<span class="error">❌ Build Error: ' + data.error + '</span>';
                            buildBtn.disabled = false;
                            serverBuildBtn.disabled = false;
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                        } else {
                            statusDiv.innerHTML = '<span style="color: #fbbf24;">⭕ Brain Not Built</span>';
                            buildBtn.disabled = false;
                            serverBuildBtn.disabled = false;
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                        }
                    })
                    .catch(e => {
                        document.getElementById('status').innerHTML = '<span class="error">❌ Connection Error</span>';
                    });
            }
            
            function showProgress(data) {
                const progressContainer = document.getElementById('progress-container');
                const batchInfo = document.getElementById('batch-info');
                const progressBar = document.getElementById('progress-bar');
                const progressText = document.getElementById('progress-text');
                
                // Show progress elements
                progressContainer.style.display = 'block';
                batchInfo.style.display = 'grid';
                
                // Update progress bar
                const percentage = Math.max(0, Math.min(100, data.percentage || 0));
                progressBar.style.width = percentage + '%';
                progressText.textContent = percentage + '%';
                
                // Update batch info
                document.getElementById('chunks-processed').textContent = data.chunks || 0;
                document.getElementById('batches-completed').textContent = data.batches_completed || 0;
                document.getElementById('total-batches').textContent = data.total_batches || 0;
                document.getElementById('percentage').textContent = percentage + '%';
            }
            
            function buildBrain() {
                fetch('/build_brain', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        if (!data.ok) alert('Build failed: ' + data.error);
                    })
                    .catch(e => alert('Request failed: ' + e));
            }
            
            function buildNewBrain() {
                fetch('/build_new_brain', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        if (!data.ok) alert('Build failed: ' + data.error);
                    })
                    .catch(e => alert('Request failed: ' + e));
            }
            
            // Auto-refresh every 2 seconds
            refreshStatus();
            setInterval(refreshStatus, 2000);
        </script>
    </body>
    </html>
    '''
    return html_content

# Part 9: Enhanced upload route with database integration (COMPLETE)

# --- STREAMING ---
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

# --- ENHANCED UPLOAD / OCR ---
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return "No file uploaded", 400
        
        # Get current project from form or session
        project = request.form.get('project', PROJECTS[0])
        filename = file.filename.lower()
        text = ""
        
        app.logger.info(f"Processing file: {filename} for project: {project}")

        # Process different file types
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            text = process_image_ocr(file.stream, filename)
                
        elif filename.endswith('.pdf'):
            try:
                file.stream.seek(0)
                data = file.read()
                
                if not data:
                    return "PDF file appears to be empty", 400
                
                doc = fitz.open(stream=data, filetype="pdf")
                
                if doc.page_count == 0:
                    return "PDF has no pages", 400
                
                text_parts = []
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text_parts.append(f"=== Page {page_num + 1} ===\n{page_text}")
                
                text = "\n\n".join(text_parts)
                doc.close()
                
                if not text.strip():
                    text = "No text found in PDF (may be image-based or encrypted)"
                    
            except Exception as e:
                app.logger.error(f"PDF processing failed: {e}")
                return f"PDF Error: {str(e)}", 500
                
        elif filename.endswith('.docx'):
            try:
                file.stream.seek(0)
                file_data = file.read()
                
                if not file_data:
                    return "Word document appears to be empty", 400
                
                import io
                file_stream = io.BytesIO(file_data)
                document = docx.Document(file_stream)
                
                paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
                
                tables_text = []
                for table in document.tables:
                    for row in table.rows:
                        row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_text:
                            tables_text.append(" | ".join(row_text))
                
                all_text = []
                if paragraphs:
                    all_text.extend(paragraphs)
                if tables_text:
                    all_text.append("\n=== Tables ===")
                    all_text.extend(tables_text)
                
                text = "\n".join(all_text)
                
                if not text.strip():
                    text = "No readable text found in Word document"
                    
            except Exception as e:
                app.logger.error(f"Word document processing failed: {e}")
                return f"Word Document Error: {str(e)}", 500
                
        else:
            return "Unsupported file type. Supported: PNG, JPG, JPEG, GIF, BMP, PDF, DOCX", 400

        # Truncate if too long
        if len(text) > 15000:
            text = text[:15000] + "\n\n[...Content truncated...]"
        
        # Check if OCR results are meaningful (fallback logic)
        text_words = len(text.split()) if text else 0
        is_image_file = filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        
        # Create analysis prompt based on OCR quality
        if is_image_file and text_words < 5:
            # Poor OCR results - switch to GPT-4 Vision analysis
            app.logger.info(f"OCR extracted only {text_words} words, switching to vision analysis")
            
            # Get vision analysis
            vision_description = analyze_image_with_vision(file.stream, file.filename)
            
            analysis_prompt = f"""I've uploaded an image file '{file.filename}'. OCR only extracted: "{text.strip()}"

Since the text was minimal, I analyzed the image visually instead:

=== VISUAL ANALYSIS ===
{vision_description}

=== FILE DETAILS ===
- Filename: {file.filename}
- Type: IMAGE ({filename.split('.')[-1].upper()})
- Analysis Method: GPT-4 Vision (due to minimal text extraction)

Please provide insights based on this visual analysis."""

        else:
            # Good OCR results - proceed with text analysis
            analysis_prompt = f"""I've uploaded and processed the file '{file.filename}'. Here's what was extracted:

=== EXTRACTED CONTENT ===
{text}

=== FILE DETAILS ===
- Filename: {file.filename}
- Type: {filename.split('.')[-1].upper()}
- Characters: {len(text):,}
- Words: {len(text.split()):,}
- Lines: {len(text.splitlines()):,}

Please analyze this content and provide insights, summaries, or answer any questions about what you see."""

        app.logger.info(f"File processing successful: {len(text)} characters extracted")

        # Get AI voices from form or use default
        use_voices = ['SyntaxPrime']  # Default to SyntaxPrime for file analysis
        random_toggle = False

        # Generate AI analysis
        try:
            retrieval_ctx = retrieve(analysis_prompt, k=5) if is_ready() else []
            response_data = generate_response(
                analysis_prompt, use_voices, random_toggle,
                project=project, model=CHAT_MODEL, retrieval_context=retrieval_ctx
            )
        except Exception as e:
            app.logger.error(f"AI analysis failed: {e}")
            response_data = {"SyntaxPrime": f"File processed successfully, but AI analysis failed: {e}"}

        # Save to database and track file upload
        user_message = f"[File Upload] {file.filename}"
        save_conversation_enhanced(project, user_message, response_data)
        
        # Track the uploaded file in database
        file_extension = filename.split('.')[-1].upper() if '.' in filename else 'UNKNOWN'
        content_summary = text[:500] if text else "No text extracted"
        track_uploaded_file(file.filename, file_extension, project, content_summary)

        # Redirect back to main chat with the analysis
        return redirect(f'/?project={project}#bottom-anchor')
        
    except Exception as e:
        app.logger.error(f"Upload route failed: {e}")
        import traceback
        app.logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"Upload Error: {str(e)}", 500

# Part 10: Database dashboard and utility routes

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
            <h1>📊 Database Dashboard</h1>
            
            <div class="status-box">
                <h3>Connection Status</h3>
                <div id="status">Loading...</div>
            </div>
            
            <div class="status-box">
                <h3>Statistics</h3>
                <div id="stats" class="stats-grid">Loading...</div>
            </div>
            
            <div class="status-box">
                <button class="btn" onclick="refreshStatus()">🔄 Refresh</button>
                <button class="btn" onclick="window.location.href='/'">← Back to Chat</button>
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
                            statusDiv.innerHTML = '<span class="success">✅ Database Connected & Ready</span>';
                        } else {
                            statusDiv.innerHTML = '<span class="error">❌ Database Issues Detected</span>';
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
                        document.getElementById('status').innerHTML = '<span class="error">❌ Connection Error</span>';
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

# Part 11: Debug routes and authentication (FINAL PART)

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

