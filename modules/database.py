# modules/database.py - Enhanced Database Operations Module
# Complete replacement file with smart context routing and brain health monitoring

import os
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import List, Dict, Any

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

def get_content_tier(content: str) -> str:
    """Classify content by length and substance into tiers"""
    if not content or not content.strip():
        return "minimal"
    
    content = content.strip()
    length = len(content)
    word_count = len(content.split())
    
    # Count sentences (rough approximation)
    sentence_count = content.count('.') + content.count('!') + content.count('?')
    
    # Minimal: Very short responses, greetings, confirmations
    if (length < 50 or word_count < 8 or
        any(greeting in content.lower() for greeting in ['hi', 'hello', 'thanks', 'ok', 'yes', 'no'])):
        return "minimal"
    
    # Basic: Short but substantial responses
    elif length < 200 or word_count < 30 or sentence_count <= 2:
        return "basic"
    
    # Substantial: Medium-length informative content
    elif length < 800 or word_count < 120 or sentence_count <= 5:
        return "substantial"
    
    # Comprehensive: Long, detailed responses
    else:
        return "comprehensive"

def should_include_content(content: str, context_type: str = "mixed") -> bool:
    """Smart filter to determine if content should be included based on tier and context"""
    tier = get_content_tier(content)
    
    # Context-aware filtering
    if context_type == "personal_context":
        # For personal context, include more content types
        return tier in ["basic", "substantial", "comprehensive"]
    
    elif context_type == "knowledge_base":
        # For knowledge base, prioritize substantial content
        return tier in ["substantial", "comprehensive"]
    
    elif context_type == "recent_priority":
        # For recent high-priority searches, include most content
        return tier in ["basic", "substantial", "comprehensive"]
    
    else:  # mixed or default
        # Standard filtering - exclude only minimal content
        return tier != "minimal"

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
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_brain_docs_content_fts ON brain_documents USING gin(to_tsvector(\'english\', content))')
            
            # Create brain_health table for monitoring
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS brain_health (
                    id SERIAL PRIMARY KEY,
                    last_refresh TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_documents INTEGER DEFAULT 0,
                    last_search_query TEXT,
                    last_search_results INTEGER DEFAULT 0,
                    health_status VARCHAR(50) DEFAULT 'healthy',
                    error_log TEXT
                )
            ''')
            
            conn.commit()
            print("Database tables initialized successfully")
            
        except Exception as e:
            conn.rollback()
            print(f"Database initialization failed: {e}")

def classify_search_intent(query_text: str, conversation_context: List[str] = None) -> str:
    """Classify whether a search needs personal context vs knowledge base"""
    
    query_lower = query_text.lower().strip()
    
    # Personal context indicators (recent conversation continuation)
    personal_patterns = [
        # Conversational continuity
        "we were talking about", "you mentioned", "earlier you said", "as we discussed",
        "my situation", "my project", "our conversation", "what i told you",
        
        # Current status/updates
        "update me", "catch me up", "where are we", "current status", "latest on",
        "how are things", "what's happening with",
        
        # Recent activities (contextual)
        "today", "yesterday", "this week", "recently", "just now", "right now",
        "currently", "at the moment", "these days",
        
        # Personal references
        "my family", "my daughter", "my work", "my company", "shazeen", "ghada",
        "amcf", "my mom", "my projects"
    ]
    
    # Knowledge base indicators (factual/reference information)
    knowledge_patterns = [
        # Factual queries
        "what is", "what does", "tell me about", "explain", "describe", "define",
        "how does", "why does", "when was", "where is", "who is", "who was",
        
        # TV shows, entertainment, general knowledge
        "dead like me", "happy time", "tv show", "television", "movie", "book",
        "actor", "character", "episode", "season", "plot",
        
        # Technical/procedural
        "how to", "tutorial", "instructions", "guide", "documentation",
        "best practices", "examples of", "tips for",
        
        # General topics (not personal)
        "marketing strategy", "seo", "algorithm", "technology", "history of"
    ]
    
    # Greeting/casual patterns (minimal context needed)
    casual_patterns = [
        "hello", "hi", "good morning", "good afternoon", "hey", "what's up",
        "how are you", "thanks", "thank you", "ok", "okay", "cool", "great"
    ]
    
    # Check patterns in order of specificity
    if any(pattern in query_lower for pattern in casual_patterns):
        return "casual"
    elif any(pattern in query_lower for pattern in personal_patterns):
        return "personal_context"
    elif any(pattern in query_lower for pattern in knowledge_patterns):
        return "knowledge_base"
    
    # Context-based classification
    if conversation_context:
        recent_topics = " ".join(conversation_context[-3:]).lower()  # Last 3 exchanges
        
        # If recent conversation mentioned personal topics, lean personal
        if any(term in recent_topics for term in ["shazeen", "ghada", "amcf", "daughter", "project"]):
            return "personal_context"
    
    # Default to knowledge base for specific questions, personal for general
    if len(query_text.split()) <= 3:
        return "personal_context"  # Short queries often reference ongoing conversation
    else:
        return "knowledge_base"   # Longer queries often seek information

def search_recent_conversations(query_text: str, k: int = 3, days: int = 7) -> List[Dict[str, Any]]:
    """Search only recent conversation history with smart filtering"""
    
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search conversations from last N days only
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            search_sql = '''
                SELECT user_input, response_data, created_at, project,
                       ts_rank(to_tsvector('english', user_input || ' ' || 
                               COALESCE(response_data->>'SyntaxPrime', '')), 
                               plainto_tsquery('english', %s)) as rank
                FROM chat_threads 
                WHERE created_at >= %s
                AND to_tsvector('english', user_input || ' ' || 
                    COALESCE(response_data->>'SyntaxPrime', '')) 
                    @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC, created_at DESC
                LIMIT %s
            '''
            
            cursor.execute(search_sql, (query_text, cutoff_date, query_text, k * 2))  # Get more for filtering
            rows = cursor.fetchall()
            
            print(f"Recent conversation search found {len(rows)} results from last {days} days")
            
            # Convert to RAG format with smart filtering
            results = []
            for row in rows:
                # Get response content for filtering
                response_content = row['response_data'].get('SyntaxPrime', '') if row['response_data'] else ''
                
                # Apply smart filtering - prioritize substantial content for recent conversations
                if should_include_content(response_content, "personal_context"):
                    # Combine user input and AI response for context
                    combined_text = f"User: {row['user_input']}\nResponse: {response_content}"
                    
                    results.append({
                        'text': combined_text[:1200],  # Reasonable chunk size
                        'source': f"Recent conversation - {row['project']} ({row['created_at'].strftime('%m/%d')})",
                        'id': f"conversation_{row['created_at'].timestamp()}",
                        'score': float(row['rank']),
                        'metadata': {
                            'type': 'recent_conversation',
                            'project': row['project'],
                            'date': row['created_at'].isoformat(),
                            'content_tier': get_content_tier(response_content)
                        }
                    })
                    
                    # Stop when we have enough quality results
                    if len(results) >= k:
                        break
            
            print(f"After smart filtering: {len(results)} quality results retained")
            return results
            
        except Exception as e:
            print(f"Recent conversation search failed: {e}")
            return []

def search_knowledge_base_only(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search only knowledge base documents with smart filtering"""
    
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search brain documents, excluding conversation-like content
            search_sql = '''
                SELECT document_id, title, content, metadata,
                       ts_rank(to_tsvector('english', content || ' ' || COALESCE(title, '')), 
                               plainto_tsquery('english', %s)) as rank
                FROM brain_documents 
                WHERE to_tsvector('english', content || ' ' || COALESCE(title, '')) 
                      @@ plainto_tsquery('english', %s)
                -- Exclude conversation-like content
                AND NOT (content LIKE '%User:%' OR content LIKE '%Assistant:%' OR content LIKE '%Response:%')
                ORDER BY rank DESC
                LIMIT %s
            '''
            
            cursor.execute(search_sql, (query_text, query_text, k * 2))  # Get more for filtering
            rows = cursor.fetchall()
            
            if not rows:
                # Fallback: try simpler search without conversation filtering
                fallback_sql = '''
                    SELECT document_id, title, content, metadata, 1.0 as rank
                    FROM brain_documents 
                    WHERE LOWER(content) LIKE %s OR LOWER(title) LIKE %s
                    ORDER BY LENGTH(content) DESC
                    LIMIT %s
                '''
                
                like_pattern = f'%{query_text.lower()}%'
                cursor.execute(fallback_sql, (like_pattern, like_pattern, k * 2))
                rows = cursor.fetchall()
                
                print(f"Knowledge base fallback search found {len(rows)} results")
            else:
                print(f"Knowledge base search found {len(rows)} results")
            
            # Convert to RAG format with smart filtering
            results = []
            for row in rows:
                content = row['content']
                
                # Apply smart filtering - prioritize substantial content for knowledge base
                if should_include_content(content, "knowledge_base"):
                    results.append({
                        'text': content[:1500],
                        'source': row['title'] or f"Document {row['document_id']}",
                        'id': row['document_id'],
                        'score': float(row['rank']),
                        'metadata': {
                            'type': 'knowledge_base',
                            'content_tier': get_content_tier(content),
                            **(row['metadata'] or {})
                        }
                    })
                    
                    # Stop when we have enough quality results
                    if len(results) >= k:
                        break
            
            print(f"After smart filtering: {len(results)} quality knowledge base results")
            return results
            
        except Exception as e:
            print(f"Knowledge base search failed: {e}")
            return []

def smart_context_search(query_text: str, k: int = 5, conversation_context: List[str] = None) -> List[Dict[str, Any]]:
    """Intelligent search routing based on query intent and context"""
    
    # Classify the search intent
    intent = classify_search_intent(query_text, conversation_context)
    print(f"Smart search classified query '{query_text}' as: {intent}")
    
    if intent == "casual":
        # For greetings and casual chat, minimal or no context needed
        return []
    
    elif intent == "personal_context":
        # Search recent conversations first, then add some knowledge base if needed
        recent_results = search_recent_conversations(query_text, k=max(3, k//2), days=7)
        
        if len(recent_results) < 2:
            # If not enough recent context, add some knowledge base results
            kb_results = search_knowledge_base_only(query_text, k=2)
            recent_results.extend(kb_results)
            print(f"Personal context search: {len(recent_results)} total results (recent + knowledge)")
        
        return recent_results[:k]
    
    elif intent == "knowledge_base":
        # Search knowledge base primarily, minimal recent context
        kb_results = search_knowledge_base_only(query_text, k=k)
        
        # Add one recent conversation result for continuity if available
        recent_results = search_recent_conversations(query_text, k=1, days=3)
        if recent_results:
            kb_results.insert(0, recent_results[0])  # Put recent context first
        
        print(f"Knowledge base search: {len(kb_results)} results")
        return kb_results[:k]
    
    else:
        # Default balanced approach
        recent_results = search_recent_conversations(query_text, k=2, days=7)
        kb_results = search_knowledge_base_only(query_text, k=3)
        
        combined = recent_results + kb_results
        print(f"Balanced search: {len(combined)} results (recent + knowledge)")
        return combined[:k]

def get_conversation_context(project: str, limit: int = 5) -> List[str]:
    """Get recent conversation context for intent classification"""
    
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_input, response_data 
                FROM chat_threads 
                WHERE project = %s 
                ORDER BY created_at DESC 
                LIMIT %s
            ''', (project, limit))
            
            rows = cursor.fetchall()
            context = []
            
            for row in rows:
                context.append(row[0])  # user input
                if row[1] and 'SyntaxPrime' in row[1]:
                    response_content = row[1]['SyntaxPrime']
                    # Only include substantial responses in context
                    if should_include_content(response_content, "recent_priority"):
                        context.append(response_content[:200])  # truncated response
            
            return context
            
        except Exception as e:
            print(f"Failed to get conversation context: {e}")
            return []

def search_brain_database(query_text, k=5):
    """Enhanced search with debugging, smart filtering, and fallback strategies"""
    with get_db_connection() as conn:
        if not conn:
            print(f"No DB connection for query: '{query_text}'")
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            print(f"Searching database for: '{query_text}'")
            
            # Try multiple search strategies
            results = []
            
            # Strategy 1: Full-text search with ranking
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
            
            cursor.execute(search_sql, (query_text, query_text, k * 2))  # Get more for filtering
            rows = cursor.fetchall()
            
            if rows:
                print(f"Full-text search returned {len(rows)} results")
                print(f"Top result: {rows[0]['title']} (rank: {rows[0]['rank']:.4f})")
            else:
                print("Full-text search found no results")
                
                # Strategy 2: Fallback to LIKE search for partial matches
                fallback_sql = '''
                    SELECT document_id, title, content, metadata, 1.0 as rank
                    FROM brain_documents 
                    WHERE LOWER(content) LIKE %s OR LOWER(title) LIKE %s
                    ORDER BY LENGTH(title) ASC
                    LIMIT %s
                '''
                
                like_pattern = f'%{query_text.lower()}%'
                cursor.execute(fallback_sql, (like_pattern, like_pattern, k * 2))
                rows = cursor.fetchall()
                
                if rows:
                    print(f"Fallback LIKE search returned {len(rows)} results")
                else:
                    print("No results found with any search strategy")
            
            # Convert to format expected by RAG system with smart filtering
            for row in rows:
                content = row['content']
                
                # Apply smart filtering for general brain database search
                if should_include_content(content, "mixed"):
                    results.append({
                        'text': content[:1500],  # Increased chunk size
                        'source': row['title'] or f"Document {row['document_id']}",
                        'id': row['document_id'],
                        'score': float(row['rank']),
                        'metadata': {
                            'content_tier': get_content_tier(content),
                            **(row['metadata'] or {})
                        }
                    })
                    
                    # Stop when we have enough quality results
                    if len(results) >= k:
                        break
            
            print(f"After smart filtering: {len(results)} quality results retained")
            
            # Log search results for monitoring
            update_brain_health(query_text, len(results))
            
            return results
            
        except Exception as e:
            print(f"Database search failed: {e}")
            update_brain_health(query_text, 0, error=str(e))
            return []

def update_brain_health(query=None, results_count=0, error=None):
    """Update brain health monitoring"""
    with get_db_connection() as conn:
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            # Get current document count
            cursor.execute('SELECT COUNT(*) FROM brain_documents')
            doc_count = cursor.fetchone()[0]
            
            # Update or insert health record
            cursor.execute('''
                INSERT INTO brain_health (last_refresh, total_documents, last_search_query, 
                                        last_search_results, health_status, error_log)
                VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            ''', (doc_count, query, results_count, 'healthy' if not error else 'error', error))
            
            # Also update the most recent record
            cursor.execute('''
                UPDATE brain_health 
                SET last_refresh = CURRENT_TIMESTAMP,
                    total_documents = %s,
                    last_search_query = %s,
                    last_search_results = %s,
                    health_status = %s,
                    error_log = %s
                WHERE id = (SELECT MAX(id) FROM brain_health)
            ''', (doc_count, query, results_count, 'healthy' if not error else 'error', error))
            
            conn.commit()
            
        except Exception as e:
            print(f"Failed to update brain health: {e}")

def get_brain_health_status():
    """Get current brain health status"""
    with get_db_connection() as conn:
        if not conn:
            return {"status": "no_database", "message": "No database connection"}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get latest health record
            cursor.execute('''
                SELECT * FROM brain_health 
                ORDER BY last_refresh DESC 
                LIMIT 1
            ''')
            
            health_record = cursor.fetchone()
            
            if not health_record:
                return {"status": "unknown", "message": "No health records found"}
            
            return {
                "status": health_record['health_status'],
                "last_refresh": health_record['last_refresh'].isoformat(),
                "total_documents": health_record['total_documents'],
                "last_search_query": health_record['last_search_query'],
                "last_search_results": health_record['last_search_results'],
                "error_log": health_record['error_log']
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

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
    
    # Fallback to file system will be handled by calling function
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

def track_uploaded_file(filename: str, file_type: str, project: str, content_preview: str = ""):
    """Track uploaded files in database"""
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO uploaded_files (filename, file_type, project, content_preview)
                    VALUES (%s, %s, %s, %s)
                ''', (filename, file_type, project, content_preview[:500]))
                
                conn.commit()
                print(f"File upload tracked: {filename}")
                
            except Exception as e:
                print(f"Failed to track uploaded file: {e}")
                conn.rollback()

def save_brain_to_database(corpus_data):
    """Save processed brain corpus to database with progress tracking and smart filtering"""
    with get_db_connection() as conn:
        if not conn:
            print("No database connection - brain will only be saved to file")
            return False
            
        try:
            cursor = conn.cursor()
            
            # Clear existing brain data
            cursor.execute('DELETE FROM brain_documents')
            print("Cleared existing brain documents from database")
            
            # Insert new brain data in batches with smart filtering
            saved_count = 0
            filtered_count = 0
            batch_size = 100
            
            for i in range(0, len(corpus_data), batch_size):
                batch = corpus_data[i:i + batch_size]
                
                for item in batch:
                    content = item.get('content', '')
                    
                    # Apply smart filtering during save process
                    if should_include_content(content, "knowledge_base"):
                        cursor.execute('''
                            INSERT INTO brain_documents (document_id, title, content, chunk_index, metadata)
                            VALUES (%s, %s, %s, %s, %s)
                        ''', (
                            item.get('id', 'unknown'),
                            item.get('title', '')[:500],
                            content,
                            item.get('chunk_index', 0),
                            psycopg2.extras.Json({
                                **item.get('metadata', {}),
                                'content_tier': get_content_tier(content)
                            })
                        ))
                        saved_count += 1
                    else:
                        filtered_count += 1
                
                # Commit each batch
                conn.commit()
                print(f"Saved batch {i//batch_size + 1}: {saved_count} saved, {filtered_count} filtered")
            
            # Update brain health
            update_brain_health(results_count=saved_count)
            
            print(f"Successfully saved {saved_count} brain documents to database ({filtered_count} filtered out)")
            return True
            
        except Exception as e:
            print(f"Failed to save brain to database: {e}")
            conn.rollback()
            update_brain_health(error=str(e))
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
    """Check database connection and table status with enhanced info"""
    status = {
        "database_url_configured": bool(DATABASE_URL),
        "connection_working": False,
        "tables_exist": False,
        "conversation_count": 0,
        "uploaded_files_count": 0,
        "daily_logs_count": 0,
        "brain_documents_count": 0,
        "brain_health": None
    }
    
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor()
                status["connection_working"] = True
                
                # Check if tables exist
                cursor.execute('''
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name IN ('chat_threads', 'uploaded_files', 'daily_logs', 'user_settings', 'brain_documents', 'brain_health')
                ''')
                table_count = cursor.fetchone()[0]
                status["tables_exist"] = table_count >= 4
                
                if status["tables_exist"]:
                    # Get record counts
                    cursor.execute('SELECT COUNT(*) FROM chat_threads')
                    status["conversation_count"] = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM uploaded_files')
                    status["uploaded_files_count"] = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM daily_logs')
                    status["daily_logs_count"] = cursor.fetchone()[0]
                    
                    # Brain-specific counts
                    try:
                        cursor.execute('SELECT COUNT(*) FROM brain_documents')
                        status["brain_documents_count"] = cursor.fetchone()[0]
                    except:
                        status["brain_documents_count"] = 0
                
                # Get brain health status
                status["brain_health"] = get_brain_health_status()
                
            except Exception as e:
                print(f"Database status check failed: {e}")
    
    return status
