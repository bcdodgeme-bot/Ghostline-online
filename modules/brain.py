# modules/brain.py - Enhanced Brain and RAG System Module
# Complete replacement file with smart context routing and auto-refresh

import os
import datetime
import threading
from flask import jsonify, redirect, url_for
from utils.rag_basic import retrieve, is_ready, load_corpus, get_build_status
from modules.database import (
    save_brain_to_database, search_brain_database, 
    get_brain_health_status, update_brain_health,
    smart_context_search, get_conversation_context
)

# Global RAG system state
_rag_building = False
_rag_build_error = None
_brain_building = False
_brain_build_error = None
_last_brain_refresh = None

CORPUS_PATH = "data/cleaned/ghostline_sources.jsonl.gz"

def enhanced_retrieve(query_text, k=5, project=None):
    """Enhanced retrieve with smart context routing and recency prioritization"""
    print(f"Starting enhanced retrieve with smart routing for: '{query_text}'")
    
    # Get conversation context for intent classification
    conversation_context = []
    if project:
        try:
            conversation_context = get_conversation_context(project, limit=5)
        except Exception as e:
            print(f"Failed to get conversation context: {e}")
    
    # Use smart context search
    try:
        results = smart_context_search(query_text, k=k, conversation_context=conversation_context)
        
        if results:
            print(f"Smart context search returned {len(results)} results")
            # Log the types of results for debugging
            result_types = [r['metadata'].get('type', 'unknown') for r in results]
            print(f"Result types: {result_types}")
            return results
        else:
            print("Smart context search returned no results")
    except Exception as e:
        print(f"Smart context search failed: {e}")
    
    # Fallback to original enhanced retrieve logic
    print("Falling back to original enhanced retrieve")
    
    # Strategy 1: Try database search with original logic
    db_results = search_brain_database(query_text, k)
    
    if db_results and len(db_results) >= 2:
        print(f"Using {len(db_results)} database results (fallback)")
        return db_results
    
    # Strategy 2: Fallback to file-based RAG system
    try:
        file_results = retrieve(query_text, k)
        if file_results:
            print(f"File-based search returned {len(file_results)} results (fallback)")
            return file_results
    except Exception as e:
        print(f"File-based retrieve failed: {e}")
    
    # Strategy 3: Return whatever we got
    all_results = db_results if db_results else []
    print(f"Returning {len(all_results)} results total (final fallback)")
    return all_results

def refresh_brain_context():
    """Refresh brain context periodically to maintain performance"""
    global _last_brain_refresh
    
    current_time = datetime.datetime.now()
    
    # Check if we need to refresh (every 4 hours or first time)
    if (_last_brain_refresh is None or 
        (current_time - _last_brain_refresh).total_seconds() > 14400):
        
        try:
            print("Refreshing brain context...")
            
            # Only refresh if the corpus file exists and RAG is ready
            if os.path.exists(CORPUS_PATH) and is_ready():
                load_corpus(CORPUS_PATH)
                _last_brain_refresh = current_time
                print("Brain context refreshed successfully")
                
                # Update health status
                update_brain_health(query="refresh_context", results_count=1)
                
            else:
                print("Brain corpus not available for refresh")
                
        except Exception as e:
            print(f"Brain refresh failed: {e}")
            update_brain_health(query="refresh_context", results_count=0, error=str(e))

def enhanced_build_brain_background():
    """Enhanced brain building with database storage and health monitoring"""
    global _rag_building, _rag_build_error
    
    try:
        _rag_building = True
        _rag_build_error = None
        print("Starting enhanced brain build with database integration...")
        
        # Build the brain using existing corpus
        load_corpus(CORPUS_PATH)
        print("RAG system loaded from corpus file")
        
        # Now save to database by extracting data from the loaded RAG system
        try:
            from utils.rag_basic import _rag_system
            
            if _rag_system and hasattr(_rag_system, 'chunks') and _rag_system.chunks:
                print(f"Found {len(_rag_system.chunks)} chunks in loaded RAG system")
                
                # Convert RAG chunks to database format
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
                            'batch': chunk.get('batch', 0),
                            'build_timestamp': datetime.datetime.now().isoformat()
                        }
                    }
                    corpus_data.append(corpus_item)
                
                # Save to database with progress tracking
                if save_brain_to_database(corpus_data):
                    print("Brain successfully saved to database from RAG system")
                    update_brain_health(results_count=len(corpus_data))
                else:
                    print("Brain build completed but database save failed")
                    update_brain_health(error="Database save failed")
                    
            else:
                print("No chunks found in RAG system - skipping database save")
                update_brain_health(error="No chunks in RAG system")
        
        except Exception as db_error:
            print(f"Database save failed during brain build: {db_error}")
            update_brain_health(error=str(db_error))
        
        _rag_building = False
        print("Enhanced brain build complete!")
        
    except Exception as e:
        _rag_building = False
        _rag_build_error = str(e)
        print(f"Enhanced brain build failed: {e}")
        update_brain_health(error=str(e))

def enhanced_build_new_brain_background():
    """Enhanced new brain building from sources with database storage"""
    global _brain_building, _brain_build_error
    
    try:
        _brain_building = True
        _brain_build_error = None
        print("Starting enhanced new brain build from raw sources...")
        
        from build_brain_fixed2 import build_new_brain
        result_path = build_new_brain()
        
        print(f"New brain built with result path: {result_path}")
        
        # Load the new brain into the RAG system
        load_corpus(CORPUS_PATH)
        
        # Save to database
        try:
            from utils.rag_basic import _rag_system
            
            if _rag_system and hasattr(_rag_system, 'chunks') and _rag_system.chunks:
                print(f"Found {len(_rag_system.chunks)} chunks in newly built RAG system")
                
                corpus_data = []
                for i, chunk in enumerate(_rag_system.chunks):
                    corpus_item = {
                        'id': str(chunk.get('id', f'new_chunk_{i}')),
                        'title': chunk.get('source', f'new_chunk_{i}'),
                        'content': chunk.get('text', ''),
                        'chunk_index': i,
                        'metadata': {
                            'created_at': chunk.get('created_at', ''),
                            'source': chunk.get('source', ''),
                            'batch': chunk.get('batch', 0),
                            'rebuild_timestamp': datetime.datetime.now().isoformat()
                        }
                    }
                    corpus_data.append(corpus_item)
                
                if save_brain_to_database(corpus_data):
                    print("New brain successfully saved to database")
                    update_brain_health(results_count=len(corpus_data))
                else:
                    print("New brain build completed but database save failed")
                    update_brain_health(error="New brain database save failed")
            else:
                print("No chunks found in newly built RAG system")
                update_brain_health(error="No chunks in rebuilt RAG system")
        
        except Exception as db_error:
            print(f"Database save failed during new brain build: {db_error}")
            update_brain_health(error=str(db_error))
        
        _brain_building = False
        print("Enhanced new brain build complete!")
        
    except Exception as e:
        _brain_building = False
        _brain_build_error = str(e)
        print(f"Enhanced new brain build failed: {e}")
        update_brain_health(error=str(e))

def build_brain_background():
    """Build the RAG system using batched processing with progress tracking"""
    global _rag_building, _rag_build_error
    
    try:
        _rag_building = True
        _rag_build_error = None
        print("Starting batched brain build with progress tracking...")
        
        # Load corpus with progress tracking
        load_corpus(CORPUS_PATH)
        
        _rag_building = False
        print("Batched brain build complete!")
        
        # Update health status
        update_brain_health(query="build_complete", results_count=1)
        
    except Exception as e:
        _rag_building = False
        _rag_build_error = str(e)
        print(f"Batched brain build failed: {e}")
        update_brain_health(error=str(e))

def build_new_brain_background():
    """Build new brain from raw sources on server"""
    global _brain_building, _brain_build_error
    
    try:
        _brain_building = True
        _brain_build_error = None
        print("Starting server-side brain building from raw sources...")
        
        from build_brain_fixed2 import build_new_brain
        result_path = build_new_brain()
        
        # Copy the new brain to the expected location
        import shutil
        shutil.copy(str(result_path), CORPUS_PATH)
        print(f"New brain saved to {CORPUS_PATH}")
        
        _brain_building = False
        print("Server-side brain build complete!")
        
        # Update health status
        update_brain_health(query="rebuild_complete", results_count=1)
        
    except Exception as e:
        _brain_building = False
        _brain_build_error = str(e)
        print(f"Server-side brain build failed: {e}")
        update_brain_health(error=str(e))

def get_brain_diagnostics():
    """Get comprehensive brain system diagnostics"""
    diagnostics = {
        "file_system": {},
        "database": {},
        "rag_system": {},
        "health_status": {},
        "test_searches": {}
    }
    
    # Check file system
    try:
        diagnostics["file_system"]["corpus_exists"] = os.path.exists(CORPUS_PATH)
        if os.path.exists(CORPUS_PATH):
            diagnostics["file_system"]["corpus_size"] = os.path.getsize(CORPUS_PATH)
            diagnostics["file_system"]["corpus_modified"] = datetime.datetime.fromtimestamp(
                os.path.getmtime(CORPUS_PATH)).isoformat()
    except Exception as e:
        diagnostics["file_system"]["error"] = str(e)
    
    # Check RAG system
    try:
        from utils.rag_basic import _rag_system
        diagnostics["rag_system"]["ready"] = is_ready()
        if _rag_system and hasattr(_rag_system, 'chunks'):
            diagnostics["rag_system"]["chunks_loaded"] = len(_rag_system.chunks)
        else:
            diagnostics["rag_system"]["chunks_loaded"] = 0
    except Exception as e:
        diagnostics["rag_system"]["error"] = str(e)
    
    # Check database
    try:
        from modules.database import get_database_status
        diagnostics["database"] = get_database_status()
    except Exception as e:
        diagnostics["database"]["error"] = str(e)
    
    # Get health status
    try:
        diagnostics["health_status"] = get_brain_health_status()
    except Exception as e:
        diagnostics["health_status"]["error"] = str(e)
    
    # Test searches with known problematic queries
    test_queries = ["Dead Like Me", "Happy Time", "tv show", "television series"]
    for query in test_queries:
        try:
            results = enhanced_retrieve(query, k=3)
            diagnostics["test_searches"][query] = {
                "results_count": len(results),
                "has_content": len(results) > 0
            }
        except Exception as e:
            diagnostics["test_searches"][query] = {"error": str(e)}
    
    return diagnostics

# Brain control endpoints
def handle_build_brain(session):
    """Handle brain building endpoint"""
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

def handle_build_new_brain(session):
    """Handle new brain building endpoint"""
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

def get_brain_status():
    """Get brain status with enhanced diagnostics"""
    global _rag_building, _rag_build_error, _brain_building, _brain_build_error
    
    # Get detailed build status from the batched system
    build_status = get_build_status()
    
    # Get health information
    health_status = get_brain_health_status()
    
    # Check if server-side building is in progress
    if _brain_building:
        status = {
            "ready": build_status["status"] == "complete",
            "building": True,
            "progress": "Building brain from raw sources on server...",
            "error": _brain_build_error,
            "percentage": 50,
            "chunks": 0,
            "batches_completed": 0,
            "total_batches": 1,
            "health": health_status,
            "last_refresh": _last_brain_refresh.isoformat() if _last_brain_refresh else None
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
            "total_batches": build_status.get("total_batches", 0),
            "health": health_status,
            "last_refresh": _last_brain_refresh.isoformat() if _last_brain_refresh else None
        }
    
    return status

def get_brain_control_dashboard():
    """Generate enhanced brain control dashboard HTML with diagnostics"""
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghostline Brain Control v0.4.0</title>
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
            
            .diagnostics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            
            .diagnostic-card {
                background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #333;
            }
            
            .diagnostic-title {
                font-size: 16px;
                font-weight: bold;
                color: #6366f1;
                margin-bottom: 10px;
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
            .warning { color: #f59e0b; }
            
            .pulse { animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
            
            .hidden { display: none; }
            
            .feature-badge {
                display: inline-block;
                background: #059669;
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
                margin-left: 8px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Brain Control v0.4.0 <span class="feature-badge">SMART ROUTING</span></h1>
            <p>Enhanced RAG system with smart context routing, auto-refresh, and comprehensive diagnostics.</p>
            
            <div class="status-box">
                <h3>Brain Status</h3>
                <div id="status">Loading brain status...</div>
                
                <div id="progress-container" class="progress-container hidden">
                    <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
                    <div class="progress-text" id="progress-text">0%</div>
                </div>
                
                <div id="batch-info" class="batch-info hidden">
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
                
                <div id="health-info" class="diagnostics-grid hidden">
                    <div class="diagnostic-card">
                        <div class="diagnostic-title">Health Status</div>
                        <div id="health-status">Loading...</div>
                    </div>
                    <div class="diagnostic-card">
                        <div class="diagnostic-title">Last Refresh</div>
                        <div id="last-refresh">Loading...</div>
                    </div>
                    <div class="diagnostic-card">
                        <div class="diagnostic-title">Smart Routing</div>
                        <div>Personal vs Knowledge Base context separation enabled</div>
                    </div>
                </div>
            </div>
            
            <div class="status-box">
                <h3>Controls</h3>
                <button class="btn" id="build-btn" onclick="buildBrain()">Build Brain (from file)</button>
                <button class="btn server-build" id="server-build-btn" onclick="buildNewBrain()">Build Brain (from sources)</button>
                <button class="btn" onclick="refreshStatus()">Refresh Status</button>
                <button class="btn diagnostic" onclick="showDiagnostics()">Full Diagnostics</button>
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
                        const progressContainer = document.getElementById('progress-container');
                        const batchInfo = document.getElementById('batch-info');
                        const healthInfo = document.getElementById('health-info');
                        
                        // Update basic status
                        if (data.ready) {
                            statusDiv.innerHTML = '<span class="success">✅ Brain Ready & Loaded</span><br><small>Smart context routing active</small>';
                            buildBtn.disabled = true;
                            serverBuildBtn.disabled = true;
                            progressContainer.classList.add('hidden');
                            batchInfo.classList.add('hidden');
                            healthInfo.classList.remove('hidden');
                            updateHealthInfo(data);
                        } else if (data.building) {
                            statusDiv.innerHTML = '<span class="building pulse">⚡ Building Brain...</span>';
                            buildBtn.disabled = true;
                            serverBuildBtn.disabled = true;
                            showProgress(data);
                        } else if (data.error) {
                            statusDiv.innerHTML = '<span class="error">❌ Build Error: ' + data.error + '</span>';
                            buildBtn.disabled = false;
                            serverBuildBtn.disabled = false;
                            progressContainer.classList.add('hidden');
                            batchInfo.classList.add('hidden');
                            healthInfo.classList.add('hidden');
                        } else {
                            statusDiv.innerHTML = '<span class="warning">⚠️ Brain Not Built</span>';
                            buildBtn.disabled = false;
                            serverBuildBtn.disabled = false;
                            progressContainer.classList.add('hidden');
                            batchInfo.classList.add('hidden');
                            healthInfo.classList.add('hidden');
                        }
                    })
                    .catch(e => {
                        document.getElementById('status').innerHTML = '<span class="error">❌ Connection Error</span>';
                    });
            }
            
            function updateHealthInfo(data) {
                const healthStatus = document.getElementById('health-status');
                const lastRefresh = document.getElementById('last-refresh');
                
                if (data.health) {
                    const status = data.health.status || 'unknown';
                    healthStatus.innerHTML = status === 'healthy' ? 
                        '<span class="success">✅ Healthy</span>' : 
                        '<span class="error">❌ ' + status + '</span>';
                    
                    if (data.health.last_refresh) {
                        const refreshTime = new Date(data.health.last_refresh).toLocaleString();
                        lastRefresh.innerHTML = refreshTime;
                    }
                }
                
                if (data.last_refresh) {
                    const autoRefresh = new Date(data.last_refresh).toLocaleString();
                    lastRefresh.innerHTML += '<br><small>Auto-refresh: ' + autoRefresh + '</small>';
                }
            }
            
            function showProgress(data) {
                const progressContainer = document.getElementById('progress-container');
                const batchInfo = document.getElementById('batch-info');
                const healthInfo = document.getElementById('health-info');
                const progressBar = document.getElementById('progress-bar');
                const progressText = document.getElementById('progress-text');
                
                // Show progress elements
                progressContainer.classList.remove('hidden');
                batchInfo.classList.remove('hidden');
                healthInfo.classList.add('hidden');
                
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
            
            function showDiagnostics() {
                const panel = document.getElementById('diagnostics-panel');
                const content = document.getElementById('diagnostics-content');
                
                panel.classList.remove('hidden');
                content.innerHTML = 'Loading comprehensive diagnostics...';
                
                fetch('/debug/brain_diagnostics')
                    .then(r => r.json())
                    .then(data => {
                        let html = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                        content.innerHTML = html;
                    })
                    .catch(e => {
                        content.innerHTML = '<span class="error">Diagnostics failed: ' + e + '</span>';
                    });
            }
            
            // Auto-refresh every 2 seconds
            refreshStatus();
            setInterval(refreshStatus, 2000);
        </script>
    </body>
    </html>
    '''
    return html_content