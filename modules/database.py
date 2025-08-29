# modules/database.py - Database Operations Module

import os
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    # Railway provides postgres:// but psycopg2 needs postgresql://
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = None
    try:
        if DATABASE_URL:
            conn = psycopg2.connect(DATABASE_URL)
            yield conn
        else:
            print("No DATABASE_URL found - using file storage only")
            yield None
    except Exception as e:
        print(f"Database connection failed: {e}")
        if conn:
            conn.rollback()
        yield None
    finally:
        if conn:
            conn.close()

def init_database():
    """Create necessary database tables"""
    if not DATABASE_URL:
        print("No database URL - running in file-only mode")
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
            print("Database tables initialized successfully")
            
        except Exception as e:
            conn.rollback()
            print(f"Database initialization failed: {e}")

def search_brain_database(query_text, k=5):
    """Search brain documents in database using PostgreSQL full-text search"""
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Use PostgreSQL's full-text search with ranking
            search_sql = '''
                SELECT document_id, title, content, metadata,
                       ts_rank(to_tsvector('english', content || ' ' || COALESCE(title, '')), 
                               plainto_tsquery('english', %s)) as rank
                FROM brain_documents 
                WHERE to_tsvector('english', content || ' ' || COALESCE(title, '')) 
                      @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s
            '''
            
            cursor.execute(search_sql, (query_text, query_text, k))
            rows = cursor.fetchall()
            
            # Convert to format expected by RAG system
            results = []
            for row in rows:
                results.append({
                    'text': row['content'][:1000],  # Limit chunk size
                    'source': row['title'] or f"Document {row['document_id']}",
                    'id': row['document_id'],
                    'score': float(row['rank']),
                    'metadata': row['metadata'] or {}
                })
            
            print(f"Database search found {len(results)} results for: {query_text}")
            return results
            
        except Exception as e:
            print(f"Database search failed: {e}")
            return []

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
                print(f"Loaded {len(conversations)} conversations from database for {project}")
                return conversations
                
            except Exception as e:
                print(f"Failed to load conversations from database: {e}")
    
    # Fallback to file system - need to import this function from utils
    print(f"Falling back to file system for {project} conversations")
    # Import will be handled when this module is properly integrated
    return []

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
                print(f"Conversation saved to database for {project}")
                
            except Exception as e:
                print(f"Failed to save conversation to database: {e}")
                conn.rollback()
    
    # File backup will be handled when module is integrated
    # _append_session(project, user_input, response_data)

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
                print(f"Daily log saved to database: {sync_type}")
                
            except Exception as e:
                print(f"Failed to save daily log to database: {e}")
                conn.rollback()
    
    # File backup will be handled when module is integrated
    # _save_daily_log(sync_type, content)

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
                print(f"File upload tracked: {filename}")
                
            except Exception as e:
                print(f"Failed to track uploaded file: {e}")
                conn.rollback()

def save_brain_to_database(corpus_data):
    """Save processed brain corpus to database"""
    with get_db_connection() as conn:
        if not conn:
            print("No database connection - brain will only be saved to file")
            return False
            
        try:
            cursor = conn.cursor()
            
            # Clear existing brain data
            cursor.execute('DELETE FROM brain_documents')
            print("Cleared existing brain documents from database")
            
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
            print(f"Saved {saved_count} brain documents to database")
            return True
            
        except Exception as e:
            print(f"Failed to save brain to database: {e}")
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
            
            print(f"Loaded {len(corpus_data)} brain documents from database")
            return corpus_data
            
        except Exception as e:
            print(f"Failed to load brain from database: {e}")
            return None

def get_database_status():
    """Check database connection and table status"""
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
                print(f"Database status check failed: {e}")
    
    return status