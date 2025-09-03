# modules/brain.py - Enhanced Brain System with Database-Only Retrieval
# Complete rewrite eliminating file-based RAG system dependencies

import os
import datetime
import threading
from flask import jsonify
from modules.database import (
    save_brain_to_database, search_brain_database,
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
    """Database-only brain system - no file-based RAG dependencies"""
    
    def __init__(self):
        self.ready = False
        self.document_count = 0
        self._check_brain_status()
    
    def _check_brain_status(self):
        """Check if brain documents are available in database"""
        try:
            db_status = get_database_status()
            self.document_count = db_status.get('brain_documents', 0)
            self.ready = self.document_count > 0
            
            if self.ready:
                print(f"Database brain system ready with {self.document_count} documents")
            else:
                print("Database brain system: No documents found")
                
        except Exception as e:
            print(f"Failed to check brain status: {e}")
            self.ready = False
    
    def search(self, query_text, k=5, project=None):
        """Enhanced search with smart context routing"""
        if not self.ready:
            return []
        
        print(f"Database brain search: '{query_text}'")
        
        # Get conversation context for intent classification
        conversation_context = []
        if project:
            try:
                conversation_context = get_conversation_context(project, limit=5)
            except Exception as e:
                print(f"Failed to get conversation context: {e}")
        
        # Primary: Smart context search
        try:
            results = smart_context_search(
                query_text,
                k=k,
                conversation_context=conversation_context
            )
            
            if results:
                print(f"Smart context search returned {len(results)} results")
                return results
            else:
                print("Smart context search returned no results")
        except Exception as e:
            print(f"Smart context search failed: {e}")
        
        # Fallback: Basic database search
        try:
            results = search_brain_database(query_text, k)
            print(f"Database search returned {len(results)} results")
            return results
        except Exception as e:
            print(f"Database search failed: {e}")
            return []
    
    def get_status(self):
        """Get brain system status"""
        return {
            "ready": self.ready,
            "document_count": self.document_count,
            "status": "complete" if self.ready else "empty",
            "method": "database"
        }

# Global brain system instance
_brain_system = DatabaseBrainSystem()

def enhanced_retrieve(query_text, k=5, project=None):
    """Enhanced retrieve using database-only brain system"""
    global _brain_system
    
    try:
        results = _brain_system.search(query_text, k=k, project=project)
        
        # Update health tracking
        update_brain_health(
            query=query_text[:100],
            results_count=len(results)
        )
        
        return results
        
    except Exception as e:
        print(f"Enhanced retrieve failed: {e}")
        update_brain_health(
            query=query_text[:100],
            results_count=0,
            error=str(e)
        )
        return []

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
            
            print(f"Brain context refreshed: {_brain_system.document_count} documents available")
            
            # Update health status
            update_brain_health(
                query="refresh_context",
                results_count=_brain_system.document_count
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
        "test_searches": {}
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
    
    # Test searches
    test_queries = ["Dead Like Me", "Happy Time", "tv show", "television series", "project management"]
    for query in test_queries:
        try:
            results = enhanced_retrieve(query, k=3)
            diagnostics["test_searches"][query] = {
                "results_count": len(results),
                "has_content": len(results) > 0,
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
            f"Ready with {brain_status['document_count']} documents" if brain_status["ready"]
            else "Brain not built"
        ),
        "error": _brain_build_error,
        "percentage": 100 if brain_status["ready"] else (50 if _brain_building else 0),
        "chunks": brain_status["document_count"],
        "method": "database",
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
        <title>Ghostline Brain Control - Database Edition</title>
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
                background: #dc2626;
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
            <h1>Brain Control - Database Edition <span class="feature-badge">DATABASE</span></h1>
            <p>Streamlined brain system using direct database access. No file-based RAG dependencies.</p>
            
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
                        <div class="stat-number" id="method">Database</div>
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
                            statusDiv.innerHTML = '<span class="success">Database Brain Ready</span><br><small>Direct database access active</small>';
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
                document.getElementById('method').textContent = data.method || 'Database';
                
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
        "progress": "Building..." if _brain_building else f"{brain_status['document_count']} documents ready",
        "percentage": 50 if _brain_building else (100 if brain_status["ready"] else 0),
        "chunks_processed": brain_status["document_count"],
        "embeddings_created": brain_status["document_count"],
        "method": "database"
    }
