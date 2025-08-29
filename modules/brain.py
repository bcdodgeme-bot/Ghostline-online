# modules/brain.py - Brain and RAG System Module

import os
import threading
from flask import jsonify, redirect, url_for
from utils.rag_basic import retrieve, is_ready, load_corpus, get_build_status
from modules.database import save_brain_to_database, search_brain_database

# Global RAG system state
_rag_building = False
_rag_build_error = None
_brain_building = False
_brain_build_error = None

CORPUS_PATH = "data/cleaned/ghostline_sources.jsonl.gz"

def enhanced_retrieve(query_text, k=5):
    """Enhanced retrieve function that searches database first, then falls back to files"""
    # Try database first
    db_results = search_brain_database(query_text, k)
    
    if db_results:
        print(f"Using {len(db_results)} database results for query: {query_text}")
        return db_results
    
    # Fallback to file-based RAG system
    print(f"No database results, falling back to file search for: {query_text}")
    try:
        return retrieve(query_text, k)
    except Exception as e:
        print(f"File-based retrieve also failed: {e}")
        return []

def enhanced_build_brain_background():
    """Enhanced brain building with database storage - works with chunked files"""
    global _rag_building, _rag_build_error
    
    try:
        _rag_building = True
        _rag_build_error = None
        print("Starting enhanced brain build with database integration...")
        
        # Build the brain using existing corpus (this handles the chunked files)
        load_corpus(CORPUS_PATH)
        
        # Now save to database by extracting data from the loaded RAG system
        try:
            # Import your RAG system to access the loaded data - FIXED IMPORT
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
                            'batch': chunk.get('batch', 0)
                        }
                    }
                    corpus_data.append(corpus_item)
                
                # Save to database using the imported module function
                if save_brain_to_database(corpus_data):
                    print("Brain successfully saved to database from RAG system")
                else:
                    print("Brain build completed but database save failed")
            else:
                print("No chunks found in RAG system - skipping database save")
        
        except Exception as db_error:
            print(f"Database save failed during brain build: {db_error}")
        
        _rag_building = False
        print("Enhanced brain build complete!")
        
    except Exception as e:
        _rag_building = False
        _rag_build_error = str(e)
        print(f"Enhanced brain build failed: {e}")

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
        
        # Now try to save to database
        try:
            from utils.rag_basic import _rag_system
            
            if _rag_system and hasattr(_rag_system, 'chunks') and _rag_system.chunks:
                print(f"Found {len(_rag_system.chunks)} chunks in newly built RAG system")
                
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
                    print("New brain successfully saved to database")
                else:
                    print("New brain build completed but database save failed")
            else:
                print("No chunks found in newly built RAG system")
        
        except Exception as db_error:
            print(f"Database save failed during new brain build: {db_error}")
        
        _brain_building = False
        print("Enhanced new brain build complete!")
        
    except Exception as e:
        _brain_building = False
        _brain_build_error = str(e)
        print(f"Enhanced new brain build failed: {e}")

def build_brain_background():
    """Build the RAG system using batched processing - WITH PROGRESS TRACKING!"""
    global _rag_building, _rag_build_error
    
    try:
        _rag_building = True
        _rag_build_error = None
        print("Starting batched brain build with progress tracking...")
        
        # Load corpus with progress tracking - this will show your loading bar!
        load_corpus(CORPUS_PATH)
        
        _rag_building = False
        print("Batched brain build complete!")
        
    except Exception as e:
        _rag_building = False
        _rag_build_error = str(e)
        print(f"Batched brain build failed: {e}")

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
        
    except Exception as e:
        _brain_building = False
        _brain_build_error = str(e)
        print(f"Server-side brain build failed: {e}")

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
    """Get brain status with batch progress"""
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
    
    return status

def get_brain_control_dashboard():
    """Generate brain control dashboard HTML"""
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
            <h1>Brain Control v0.2.0</h1>
            <p>Enhanced RAG system with real-time progress tracking and batch processing.</p>
            
            <div class="status-box">
                <h3>Brain Status</h3>
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
                <h3>Controls</h3>
                <button class="btn" id="build-btn" onclick="buildBrain()">Build Brain (from file)</button>
                <button class="btn server-build" id="server-build-btn" onclick="buildNewBrain()">Build Brain (from sources)</button>
                <button class="btn" onclick="refreshStatus()">Refresh Status</button>
                <button class="btn" onclick="window.location.href='/'">&larr; Back to Chat</button>
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
                            statusDiv.innerHTML = '<span class="success">Brain Ready &amp; Loaded</span>';
                            buildBtn.disabled = true;
                            serverBuildBtn.disabled = true;
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                        } else if (data.building) {
                            statusDiv.innerHTML = '<span class="building pulse">Building Brain...</span>';
                            buildBtn.disabled = true;
                            serverBuildBtn.disabled = true;
                            showProgress(data);
                        } else if (data.error) {
                            statusDiv.innerHTML = '<span class="error">Build Error: ' + data.error + '</span>';
                            buildBtn.disabled = false;
                            serverBuildBtn.disabled = false;
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                        } else {
                            statusDiv.innerHTML = '<span style="color: #fbbf24;">Brain Not Built</span>';
                            buildBtn.disabled = false;
                            serverBuildBtn.disabled = false;
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                        }
                    })
                    .catch(e => {
                        document.getElementById('status').innerHTML = '<span class="error">Connection Error</span>';
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