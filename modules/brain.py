# modules/brain.py - Enhanced Brain System with Conversation History + Database Retrieval
# Complete replacement with dual-table search functionality

import os
import datetime
import threading
from flask import jsonify
from psycopg2.extras import RealDictCursor
from modules.database import (
    get_db_connection, save_brain_to_database, search_brain_database,
    get_brain_health_status, update_brain_health,
    smart_context_search, get_conversation_context,
    get_database_status
)

# Global brain system state
_brain_building = False
_brain_build_error = None
_last_brain_refresh = None

CORPUS_PATH = "data/cleaned/ghostline_sources.jsonl.gz"

class DatabaseBrainSystem:
    """Database-only brain system with dual-table search capability"""
    
    def __init__(self):
        self.ready = False
        self.document_count = 0
        self.conversation_count = 0
        self._check_brain_status()
    
    def _check_brain_status(self):
        """Check if brain documents and conversations are available"""
        try:
            db_status = get_database_status()
            self.document_count = db_status.get('brain_documents', 0)
            
            # Also check conversation threads
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM chat_threads WHERE user_input IS NOT NULL")
                    self.conversation_count = cursor.fetchone()[0]
            
            self.ready = (self.document_count > 0) or (self.conversation_count > 0)
            
            if self.ready:
                print(f"Database brain system ready: {self.document_count} documents, {self.conversation_count} conversations")
            else:
                print("Database brain system: No documents or conversations found")
                
        except Exception as e:
            print(f"Failed to check brain status: {e}")
            self.ready = False
    
    def get_status(self):
        """Get brain system status"""
        return {
            "ready": self.ready,
            "document_count": self.document_count,
            "conversation_count": self.conversation_count,
            "status": "complete" if self.ready else "empty",
            "method": "database_dual"
        }

# Global brain system instance
_brain_system = DatabaseBrainSystem()

def enhanced_retrieve(query_text, k=5, project=None):
    """Enhanced retrieve with conversation history + brain documents search"""
    print(f"Enhanced retrieve: searching for '{query_text}'")
    
    all_results = []
    
    # PRIORITY 1: Search conversation history in chat_threads
    # This is where personal context lives (like who Ghada is)
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                # Smart multi-tier search filtering SQL
                conversation_sql = """
                SELECT 
                    'conversation_' || id as source_id,
                    'Conversation History' as title,
                    user_input as content,
                    'conversation_history' as source_type,
                    created_at::text as timestamp,
                    project,
                    1.0 as base_relevance
                FROM chat_threads 
                WHERE (user_input ILIKE %s OR response_data::text ILIKE %s)
                    AND user_input IS NOT NULL
                    AND (
                        -- Tier 1: High-value short queries (questions about people, places, things)
                        (LENGTH(user_input) BETWEEN 10 AND 100 AND (
                            user_input ILIKE '%%who is%%' OR 
                            user_input ILIKE '%%what is%%' OR 
                            user_input ILIKE '%%where is%%' OR 
                            user_input ILIKE '%%when is%%' OR 
                            user_input ILIKE '%%how is%%' OR
                            user_input ILIKE '%%tell me about%%' OR
                            user_input ILIKE '%%ghada%%' OR
                            user_input ILIKE '%%dead like me%%' OR
                            user_input ~* '\\b(favorite|love|like|enjoy|watch|show|movie|book|person|friend|family)\\b'
                        ))
                        OR
                        -- Tier 2: Medium-length conversational content (detailed but not overwhelming)
                        (LENGTH(user_input) BETWEEN 100 AND 500 AND (
                            user_input ~* '\\b(remember|mentioned|told|said|discussed|talked about)\\b' OR
                            user_input ~* '\\b(preference|opinion|think|feel|believe)\\b' OR
                            user_input ~* '\\b(project|work|plan|idea|goal)\\b'
                        ))
                        OR
                        -- Tier 3: Substantial content with context markers (your detailed brain dumps)
                        (LENGTH(user_input) > 500 AND (
                            user_input ~* '\\b(context|background|history|detail|explain|describe)\\b' OR
                            user_input ~* '\\b(important|significant|key|main|primary)\\b' OR
                            (user_input ~* '\\b(i|me|my|mine)\\b' AND LENGTH(user_input) > 200)
                        ))
                    )
                ORDER BY created_at DESC
                LIMIT %s
                """
                
                search_pattern = f"%{query_text}%"
                cursor.execute(conversation_sql, (search_pattern, search_pattern, k))
                
                conversation_results = cursor.fetchall()
                
                if conversation_results:
                    print(f"Found {len(conversation_results)} conversation history results")
                    
                    for row in conversation_results:
                        # Extract relevant section around the query match
                        content = row['content']
                        query_lower = query_text.lower()
                        content_lower = content.lower()
                        
                        # Find the position of the match and extract context
                        match_pos = content_lower.find(query_lower)
                        if match_pos != -1:
                            # Extract 500 chars around the match for context
                            start = max(0, match_pos - 250)
                            end = min(len(content), match_pos + 250)
                            context_snippet = content[start:end]
                            
                            if start > 0:
                                context_snippet = "..." + context_snippet
                            if end < len(content):
                                context_snippet = context_snippet + "..."
                        else:
                            # Fallback: take first 500 chars
                            context_snippet = content[:500] + ("..." if len(content) > 500 else "")
                        
                        result = {
                            'text': context_snippet,
                            'source': f"Conversation ({row['project']}) - {row['timestamp'][:10]}",
                            'title': 'Personal Conversation History',
                            'source_type': 'conversation',
                            'similarity': 0.95  # High relevance for personal context
                        }
                        all_results.append(result)
                
    except Exception as e:
        print(f"Conversation history search failed: {e}")
    
    # PRIORITY 2: Search brain documents for knowledge base content
    try:
        brain_results = search_brain_database(query_text, k)
        if brain_results:
            print(f"Found {len(brain_results)} brain document results")
            
            for result in brain_results:
                result['source_type'] = 'knowledge_base'
                # Lower similarity score than conversation history
                if 'similarity' in result:
                    result['similarity'] = result['similarity'] * 0.8
                else:
                    result['similarity'] = 0.7
                
                all_results.append(result)
    except Exception as e:
        print(f"Brain documents search failed: {e}")
    
    # PRIORITY 3: Smart context search as fallback with improved filtering
    if len(all_results) < k:
        try:
            conversation_context = []
            if project:
                conversation_context = get_conversation_context(project, limit=5)
            
            # Apply smart filtering to fallback search as well
            smart_results = smart_context_search(
                query_text,
                k=k-len(all_results),
                conversation_context=conversation_context,
                apply_smart_filter=True  # Enable smart filtering for fallback
            )
            
            if smart_results:
                print(f"Found {len(smart_results)} smart context results")
                for result in smart_results:
                    result['source_type'] = 'smart_context'
                    if 'similarity' not in result:
                        result['similarity'] = 0.6
                    all_results.append(result)
        except Exception as e:
            print(f"Smart context search failed: {e}")
    
    # Sort by relevance: conversation history first, then by similarity
    all_results.sort(key=lambda x: (
        0 if x.get('source_type') == 'conversation' else 1,  # Conversation first
        -x.get('similarity', 0)  # Then by similarity descending
    ))
    
    # Return top results
    final_results = all_results[:k]
    
    print(f"Enhanced retrieve returning {len(final_results)} total results:")
    for i, result in enumerate(final_results):
        source_type = result.get('source_type', 'unknown')
        similarity = result.get('similarity', 0)
        print(f"  {i+1}. {source_type} (similarity: {similarity:.2f})")
    
    # Update health tracking
    try:
        update_brain_health(
            query=query_text[:100],
            results_count=len(final_results)
        )
    except Exception as e:
        print(f"Health tracking update failed: {e}")
    
    return final_results

def refresh_brain_context():
    """Refresh brain context by checking database status"""
    global _last_brain_refresh, _brain_system
    
    current_time = datetime.datetime.now()
    
    # Check if we need to refresh (every 4 hours or first time)
    if (_last_brain_refresh is None or
        (current_time - _last_brain_refresh).total_seconds() > 14400):
        
        try:
            print("Refreshing brain context...")
            
            # Refresh database status
            _brain_system._check_brain_status()
            _last_brain_refresh = current_time
            
            print(f"Brain context refreshed: {_brain_system.document_count} documents, {_brain_system.conversation_count} conversations")
            
            # Update health status
            update_brain_health(
                query="refresh_context",
                results_count=_brain_system.document_count + _brain_system.conversation_count
            )
                
        except Exception as e:
            print(f"Brain refresh failed: {e}")
            update_brain_health(
                query="refresh_context",
                results_count=0,
                error=str(e)
            )

def build_brain_from_corpus():
    """Build brain by processing corpus file and saving to database"""
    global _brain_building, _brain_build_error
    
    try:
        _brain_building = True
        _brain_build_error = None
        print("Starting brain build from corpus file...")
        
        # Check if corpus file exists
        if not os.path.exists(CORPUS_PATH):
            raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")
        
        # Import the corpus processing logic
        try:
            import gzip
            import json
            
            print(f"Processing corpus file: {CORPUS_PATH}")
            
            corpus_data = []
            chunk_id = 0
            
            with gzip.open(CORPUS_PATH, 'rt', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        
                        # Extract text content from JSON structure
                        text_content = extract_text_from_json(data)
                        
                        if text_content and len(text_content.strip()) > 50:
                            # Create chunks if text is long
                            chunks = chunk_text(text_content, max_words=500)
                            
                            for chunk_index, chunk_text in enumerate(chunks):
                                corpus_item = {
                                    'id': f'corpus_{chunk_id}',
                                    'title': f'line_{line_num + 1}_{chunk_index}',
                                    'content': chunk_text,
                                    'chunk_index': chunk_index,
                                    'metadata': {
                                        'source': f'line_{line_num + 1}',
                                        'created_at': datetime.datetime.now().isoformat(),
                                        'build_timestamp': datetime.datetime.now().isoformat(),
                                        'chunk_index': chunk_index
                                    }
                                }
                                corpus_data.append(corpus_item)
                                chunk_id += 1
                    
                    except json.JSONDecodeError:
                        continue
                    
                    # Progress update
                    if line_num % 1000 == 0 and line_num > 0:
                        print(f"Processed {line_num} lines, created {len(corpus_data)} chunks")
            
            print(f"Corpus processing complete: {len(corpus_data)} total chunks")
            
            # Save to database
            if corpus_data:
                if save_brain_to_database(corpus_data):
                    print("Brain successfully saved to database")
                    update_brain_health(results_count=len(corpus_data))
                    
                    # Refresh brain system status
                    _brain_system._check_brain_status()
                else:
                    raise Exception("Database save failed")
            else:
                raise Exception("No valid chunks created from corpus")
                
        except Exception as process_error:
            raise Exception(f"Corpus processing failed: {process_error}")
        
        _brain_building = False
        print("Brain build from corpus complete!")
        
    except Exception as e:
        _brain_building = False
        _brain_build_error = str(e)
        print(f"Brain build from corpus failed: {e}")
        update_brain_health(error=str(e))

def build_brain_from_sources():
    """Build new brain from raw sources"""
    global _brain_building, _brain_build_error
    
    try:
        _brain_building = True
        _brain_build_error = None
        print("Starting brain build from raw sources...")
        
        # Import the brain building module
        try:
            from build_brain_fixed2 import build_new_brain
            result_path = build_new_brain()
            print(f"New brain built at: {result_path}")
            
            # Copy to expected location if different
            if str(result_path) != CORPUS_PATH:
                import shutil
                shutil.copy(str(result_path), CORPUS_PATH)
                print(f"New brain copied to {CORPUS_PATH}")
            
        except ImportError:
            raise Exception("build_brain_fixed2 module not available")
        except Exception as e:
            raise Exception(f"Brain building from sources failed: {e}")
        
        # Now process the new corpus file
        build_brain_from_corpus()
        
    except Exception as e:
        _brain_building = False
        _brain_build_error = str(e)
        print(f"Brain build from sources failed: {e}")
        update_brain_health(error=str(e))

def extract_text_from_json(json_obj):
    """Extract meaningful text from a JSON object"""
    texts = []
    text_fields = ['text', 'content', 'message', 'body', 'description', 'title', 'question', 'answer']
    
    def extract_recursive(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in text_fields and isinstance(value, str) and len(value.strip()) > 20:
                    context = f"[{prefix}{key}] " if prefix or key != 'text' else ""
                    texts.append(context + value.strip())
                elif isinstance(value, (dict, list)):
                    extract_recursive(value, f"{prefix}{key}.")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    extract_recursive(item, f"{prefix}[{i}].")
                elif isinstance(item, str) and len(item.strip()) > 20:
                    texts.append(item.strip())
    
    extract_recursive(json_obj)
    return " ".join(texts) if texts else ""

def chunk_text(text, max_words=500):
    """Break text into smaller chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
    
    return chunks if chunks else [text]

def get_brain_diagnostics():
    """Get comprehensive brain system diagnostics"""
    diagnostics = {
        "database": {},
        "brain_system": {},
        "health_status": {},
        "file_system": {},
        "test_searches": {},
        "conversation_search": {},
        "smart_filtering": {}
    }
    
    # Check database
    try:
        diagnostics["database"] = get_database_status()
    except Exception as e:
        diagnostics["database"]["error"] = str(e)
    
    # Check brain system
    try:
        global _brain_system
        diagnostics["brain_system"] = _brain_system.get_status()
    except Exception as e:
        diagnostics["brain_system"]["error"] = str(e)
    
    # Check file system
    try:
        diagnostics["file_system"]["corpus_exists"] = os.path.exists(CORPUS_PATH)
        if os.path.exists(CORPUS_PATH):
            diagnostics["file_system"]["corpus_size"] = os.path.getsize(CORPUS_PATH)
            diagnostics["file_system"]["corpus_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(CORPUS_PATH)).isoformat()
    except Exception as e:
        diagnostics["file_system"]["error"] = str(e)
    
    # Get health status
    try:
        diagnostics["health_status"] = get_brain_health_status()
    except Exception as e:
        diagnostics["health_status"]["error"] = str(e)
    
    # Test conversation search specifically
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chat_threads WHERE user_input ILIKE %s", ('%ghada%',))
                ghada_conversations = cursor.fetchone()[0]
                
                diagnostics["conversation_search"] = {
                    "ghada_mentions": ghada_conversations,
                    "search_working": ghada_conversations > 0
                }
    except Exception as e:
        diagnostics["conversation_search"]["error"] = str(e)
    
    # Test smart filtering effectiveness
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                
                # Test different tier queries
                test_queries = [
                    ("who is ghada", "tier1_question"),
                    ("what is dead like me", "tier1_question"),
                    ("tell me about", "tier1_context"),
                    ("remember when we talked", "tier2_conversational"),
                    ("project details and background", "tier2_work"),
                    ("detailed explanation and context", "tier3_substantial")
                ]
                
                smart_filter_results = {}
                
                for query, query_type in test_queries:
                    # Test the smart filtering SQL
                    test_sql = """
                    SELECT COUNT(*) FROM chat_threads 
                    WHERE user_input ILIKE %s
                        AND user_input IS NOT NULL
                        AND (
                            -- Tier 1: High-value short queries
                            (LENGTH(user_input) BETWEEN 10 AND 100 AND (
                                user_input ILIKE '%%who is%%' OR 
                                user_input ILIKE '%%what is%%' OR 
                                user_input ILIKE '%%tell me about%%' OR
                                user_input ILIKE '%%ghada%%' OR
                                user_input ILIKE '%%dead like me%%'
                            ))
                            OR
                            -- Tier 2: Medium-length conversational content
                            (LENGTH(user_input) BETWEEN 100 AND 500 AND (
                                user_input ~* '\\b(remember|mentioned|project|work)\\b'
                            ))
                            OR
                            -- Tier 3: Substantial content
                            (LENGTH(user_input) > 500 AND (
                                user_input ~* '\\b(context|background|detail|explain)\\b'
                            ))
                        )
                    """
                    
                    cursor.execute(test_sql, (f'%{query}%',))
                    filtered_count = cursor.fetchone()[0]
                    
                    # Also get unfiltered count for comparison
                    cursor.execute("SELECT COUNT(*) FROM chat_threads WHERE user_input ILIKE %s AND user_input IS NOT NULL", (f'%{query}%',))
                    total_count = cursor.fetchone()[0]
                    
                    smart_filter_results[query_type] = {
                        "query": query,
                        "filtered_results": filtered_count,
                        "total_results": total_count,
                        "filter_effectiveness": f"{filtered_count}/{total_count}"
                    }
                
                diagnostics["smart_filtering"] = {
                    "test_results": smart_filter_results,
                    "filtering_active": True
                }
                
    except Exception as e:
        diagnostics["smart_filtering"]["error"] = str(e)
    
    # Test searches with focus on personal queries
    test_queries = ["Ghada", "who is ghada", "Dead Like Me", "tv show", "personal"]
    for query in test_queries:
        try:
            results = enhanced_retrieve(query, k=3)
            diagnostics["test_searches"][query] = {
                "results_count": len(results),
                "has_content": len(results) > 0,
                "source_types": [r.get('source_type', 'unknown') for r in results],
                "sample_sources": [r.get('source', 'unknown')[:50] for r in results[:2]]
            }
        except Exception as e:
            diagnostics["test_searches"][query] = {"error": str(e)}
    
    return diagnostics

# Control endpoints
def handle_build_brain(session):
    """Handle brain building from corpus file"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _brain_building
    
    if _brain_building:
        return jsonify({"ok": False, "error": "Brain is already building"}), 400
    
    # Start building in background
    thread = threading.Thread(target=build_brain_from_corpus)
    thread.daemon = True
    thread.start()
    
    return jsonify({"ok": True, "message": "Brain building from corpus started"})

def handle_build_new_brain(session):
    """Handle building new brain from raw sources"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _brain_building
    
    if _brain_building:
        return jsonify({"ok": False, "error": "Brain is already building"}), 400
    
    # Start building in background
    thread = threading.Thread(target=build_brain_from_sources)
    thread.daemon = True
    thread.start()
    
    return jsonify({"ok": True, "message": "Brain building from sources started"})

def get_brain_status():
    """Get comprehensive brain status"""
    global _brain_building, _brain_build_error, _brain_system
    
    # Get health information
    try:
        health_status = get_brain_health_status()
    except Exception as e:
        health_status = {"error": str(e)}
    
    # Get brain system status
    brain_status = _brain_system.get_status()
    
    status = {
        "ready": brain_status["ready"],
        "building": _brain_building,
        "progress": "Building brain..." if _brain_building else (
            f"Ready: {brain_status['document_count']} documents, {brain_status['conversation_count']} conversations" if brain_status["ready"]
            else "Brain not built"
        ),
        "error": _brain_build_error,
        "percentage": 100 if brain_status["ready"] else (50 if _brain_building else 0),
        "chunks": brain_status["document_count"],
        "conversations": brain_status.get("conversation_count", 0),
        "method": "database_dual_search_smart_filtered",
        "health": health_status,
        "last_refresh": _last_brain_refresh.isoformat() if _last_brain_refresh else None
    }
    
    return status

def get_brain_control_dashboard():
    """Generate brain control dashboard HTML"""
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghostline Brain Control - Smart Filtered Dual Search</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f0f; 
                color: #fff; 
                margin: 0; 
                padding: 20px; 
            }
            .container { max-width: 1200px; margin: 0 auto; }
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
            .btn.diagnostic { background: #dc2626; }
            .btn.diagnostic:hover { background: #b91c1c; }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .stat-box {
                background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #333;
            }
            .stat-number {
                font-size: 24px;
                font-weight: bold;
                color: #10b981;
                margin-bottom: 5px;
            }
            .stat-label {
                font-size: 12px;
                color: #888;
                text-transform: uppercase;
            }
            
            #status { 
                font-family: 'SF Mono', 'Monaco', 'Cascadia Code', monospace; 
                font-size: 16px;
                padding: 15px;
                border-radius: 6px;
                background: #000;
                border: 1px solid #333;
            }
            .error { color: #ef4444; }
            .success { color: #10b981; }
            .building { color: #f59e0b; }
            .warning { color: #f59e0b; }
            
            .pulse { animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
            
            .hidden { display: none; }
            
            .feature-badge {
                display: inline-block;
                background: #10b981;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 8px;
            }
            
            .progress-simple {
                background: #333;
                height: 30px;
                border-radius: 15px;
                overflow: hidden;
                margin: 15px 0;
                position: relative;
            }
            .progress-simple.building {
                background: linear-gradient(90deg, #333 25%, #444 50%, #333 75%);
                animation: shimmer 1.5s infinite;
            }
            @keyframes shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }
            .progress-text {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-weight: bold;
                z-index: 1;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Brain Control - Smart Filtered Dual Search <span class="feature-badge">SMART</span></h1>
            <p>Enhanced brain system with intelligent multi-tier search filtering. Catches "Who is Ghada?" while preserving detailed context.</p>
            
            <div class="status-box">
                <h3>Brain Status</h3>
                <div id="status">Loading brain status...</div>
                
                <div id="progress-simple" class="progress-simple hidden">
                    <div class="progress-text">Building...</div>
                </div>
                
                <div id="stats" class="stats-grid hidden">
                    <div class="stat-box">
                        <div class="stat-number" id="document-count">0</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" id="conversation-count">0</div>
                        <div class="stat-label">Conversations</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" id="method">Smart Filter</div>
                        <div class="stat-label">Method</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" id="health-status">Unknown</div>
                        <div class="stat-label">Health</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number" id="last-refresh">Never</div>
                        <div class="stat-label">Last Refresh</div>
                    </div>
                </div>
            </div>
            
            <div class="status-box">
                <h3>Controls</h3>
                <button class="btn" id="build-btn" onclick="buildBrain()">Build from Corpus File</button>
                <button class="btn server-build" id="server-build-btn" onclick="buildNewBrain()">Build from Raw Sources</button>
                <button class="btn" onclick="refreshStatus()">Refresh Status</button>
                <button class="btn diagnostic" onclick="showDiagnostics()">System Diagnostics</button>
                <button class="btn" onclick="window.location.href='/'">&larr; Back to Chat</button>
            </div>
            
            <div id="diagnostics-panel" class="status-box hidden">
                <h3>System Diagnostics</h3>
                <div id="diagnostics-content">Loading diagnostics...</div>
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
                        const progressDiv = document.getElementById('progress-simple');
                        const statsDiv = document.getElementById('stats');
                        
                        if (data.ready) {
                            statusDiv.innerHTML = '<span class="success">Smart Filtered Dual Search Brain Ready</span><br><small>Multi-tier conversation filtering + knowledge base search active</small>';
                            buildBtn.disabled = false;  // Allow rebuilds
                            serverBuildBtn.disabled = false;
                            progressDiv.classList.add('hidden');
                            statsDiv.classList.remove('hidden');
                            updateStats(data);
                        } else if (data.building) {
                            statusDiv.innerHTML = '<span class="building pulse">Building Brain...</span>';
                            buildBtn.disabled = true;
                            serverBuildBtn.disabled = true;
                            progressDiv.classList.remove('hidden');
                            progressDiv.classList.add('building');
                            statsDiv.classList.add('hidden');
                        } else if (data.error) {
                            statusDiv.innerHTML = '<span class="error">Build Error: ' + data.error + '</span>';
                            buildBtn.disabled = false;
                            serverBuildBtn.disabled = false;
                            progressDiv.classList.add('hidden');
                            statsDiv.classList.add('hidden');
                        } else {
                            statusDiv.innerHTML = '<span class="warning">Brain Not Built</span><br><small>No documents found in database</small>';
                            buildBtn.disabled = false;
                            serverBuildBtn.disabled = false;
                            progressDiv.classList.add('hidden');
                            statsDiv.classList.add('hidden');
                        }
                    })
                    .catch(e => {
                        document.getElementById('status').innerHTML = '<span class="error">Connection Error</span>';
                    });
            }
            
            function updateStats(data) {
                document.getElementById('document-count').textContent = data.chunks || 0;
                document.getElementById('conversation-count').textContent = data.conversations || 0;
                document.getElementById('method').textContent = 'Smart Filter';
                
                const healthElement = document.getElementById('health-status');
                if (data.health && data.health.status) {
                    healthElement.textContent = data.health.status === 'healthy' ? 'Healthy' : 'Issues';
                } else {
                    healthElement.textContent = 'Unknown';
                }
                
                const refreshElement = document.getElementById('last-refresh');
                if (data.last_refresh) {
                    const refreshTime = new Date(data.last_refresh);
                    refreshElement.textContent = refreshTime.toLocaleTimeString();
                } else {
                    refreshElement.textContent = 'Never';
                }
            }
            
            function buildBrain() {
                if (confirm('Build brain from corpus file? This will process the existing corpus.')) {
                    fetch('/build_brain', { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                            if (!data.ok) alert('Build failed: ' + data.error);
                        })
                        .catch(e => alert('Request failed: ' + e));
                }
            }
            
            function buildNewBrain() {
                if (confirm('Build new brain from raw sources? This will take longer.')) {
                    fetch('/build_new_brain', { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                            if (!data.ok) alert('Build failed: ' + data.error);
                        })
                        .catch(e => alert('Request failed: ' + e));
                }
            }
            
            function showDiagnostics() {
                const panel = document.getElementById('diagnostics-panel');
                const content = document.getElementById('diagnostics-content');
                
                panel.classList.remove('hidden');
                content.innerHTML = 'Loading diagnostics...';
                
                fetch('/debug/brain_diagnostics')
                    .then(r => r.json())
                    .then(data => {
                        let html = '<pre style="white-space: pre-wrap; word-wrap: break-word;">' + 
                                  JSON.stringify(data, null, 2) + '</pre>';
                        content.innerHTML = html;
                    })
                    .catch(e => {
                        content.innerHTML = '<span class="error">Diagnostics failed: ' + e + '</span>';
                    });
            }
            
            // Auto-refresh every 3 seconds
            refreshStatus();
            setInterval(refreshStatus, 3000);
        </script>
    </body>
    </html>
    '''
    return html_content

# Compatibility functions for existing code
def is_ready():
    """Check if brain system is ready"""
    global _brain_system
    return _brain_system.ready

def load_corpus(path):
    """Compatibility function - triggers database status refresh"""
    global _brain_system
    print("Database brain system: Refreshing status instead of loading corpus")
    _brain_system._check_brain_status()

def get_build_status():
    """Get build status in expected format"""
    global _brain_building, _brain_system
    
    brain_status = _brain_system.get_status()
    
    return {
        "status": "building" if _brain_building else brain_status["status"],
        "progress": "Building..." if _brain_building else f"{brain_status['document_count']} documents, {brain_status['conversation_count']} conversations ready",
        "percentage": 50 if _brain_building else (100 if brain_status["ready"] else 0),
        "chunks_processed": brain_status["document_count"],
        "embeddings_created": brain_status["document_count"],
        "conversations_available": brain_status["conversation_count"],
        "method": "database_dual_search_smart_filtered"
    }
