# Updated /brain route and supporting functions for app.py

# Add this import at the top
from utils.rag_basic import get_build_status

# Update the global state variables
_rag_system = None
_rag_building = False
_rag_build_progress = ""
_rag_build_error = None

def build_brain_background():
    """Build the RAG system using batched processing"""
    global _rag_system, _rag_building, _rag_build_progress, _rag_build_error
    
    try:
        _rag_building = True
        _rag_build_progress = "Initializing batched RAG system..."
        _rag_build_error = None
        
        from utils.rag_basic import load_corpus
        load_corpus(CORPUS_PATH)
        
        _rag_building = False
        _rag_build_progress = "Batched build complete!"
        
    except Exception as e:
        _rag_building = False
        _rag_build_error = str(e)
        _rag_build_progress = f"Build failed: {e}"

@app.route('/brain_status')
def brain_status():
    """Enhanced brain status with batch progress"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _rag_building, _rag_build_error
    
    # Get detailed build status from the batched system
    build_status = get_build_status()
    
    status = {
        "ready": build_status["status"] == "complete",
        "building": _rag_building or build_status["status"] == "building", 
        "progress": build_status["progress"],
        "error": _rag_build_error,
        "percentage": build_status["percentage"],
        "chunks": build_status.get("chunks_processed", 0),
        "batches_completed": build_status.get("batches_completed", 0),
        "total_batches": build_status.get("total_batches", 0)
    }
    
    return jsonify(status)

@app.route('/brain')
def brain_control():
    """Enhanced brain control dashboard with batch progress"""
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ghostline Brain Control v0.1.9.6</title>
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
            .btn:disabled { background: #666; cursor: not-allowed; }
            
            /* Enhanced progress bar for batches */
            .progress-container { 
                margin: 15px 0;
                background: #333; 
                border: 2px inset #666;
                height: 40px; 
                border-radius: 8px;
                position: relative;
                overflow: hidden;
            }
            .progress-bar { 
                background: linear-gradient(90deg, #10b981 0%, #34d399 50%, #10b981 100%);
                height: 100%; 
                transition: width 0.8s ease;
                position: relative;
                min-width: 0;
                border-radius: 6px;
            }
            .progress-bar::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: repeating-linear-gradient(
                    45deg,
                    transparent,
                    transparent 12px,
                    rgba(255,255,255,0.15) 12px,
                    rgba(255,255,255,0.15) 24px
                );
                animation: slide 2s linear infinite;
            }
            @keyframes slide {
                0% { transform: translateX(-24px); }
                100% { transform: translateX(24px); }
            }
            .progress-text {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
                z-index: 10;
                font-size: 16px;
            }
            
            /* Batch progress section */
            .batch-info {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
                margin: 15px 0;
            }
            .batch-stat {
                background: #2a2a2a;
                padding: 12px;
                border-radius: 6px;
                text-align: center;
            }
            .batch-stat .number {
                font-size: 24px;
                font-weight: bold;
                color: #10b981;
            }
            .batch-stat .label {
                font-size: 12px;
                color: #888;
                margin-top: 4px;
            }
            
            #status { font-family: monospace; font-size: 14px; }
            .error { color: #ef4444; }
            .success { color: #10b981; }
            .building { color: #f59e0b; }
            .eta { 
                font-size: 12px; 
                color: #888; 
                margin-top: 8px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Ghostline Brain Control v0.1.9.6</h1>
            <p>Batched RAG system - processes your ChatGPT history in memory-safe chunks.</p>
            
            <div class="status-box">
                <h3>Brain Status</h3>
                <div id="status">Loading...</div>
                
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
                </div>
                
                <div id="eta" class="eta"></div>
            </div>
            
            <div class="status-box">
                <h3>Controls</h3>
                <button class="btn" id="build-btn" onclick="buildBrain()">Build Brain</button>
                <button class="btn" onclick="refreshStatus()">Refresh Status</button>
                <button class="btn" onclick="window.location.href='/'">Back to Chat</button>
            </div>
            
            <div class="status-box">
                <h3>Batched Processing Info</h3>
                <p><strong>New Approach:</strong> Processes 20,000 lines per batch to prevent memory crashes.</p>
                <p><strong>Auto-Resume:</strong> If interrupted, continues from last completed batch.</p>
                <p><strong>Memory Safe:</strong> Clears data between batches to stay under 2GB limit.</p>
                <p><strong>Persistent:</strong> Each batch saved separately, combined when complete.</p>
            </div>
        </div>
        
        <script>
            let statusInterval;
            let startTime = null;
            
            function refreshStatus() {
                fetch('/brain_status')
                    .then(r => r.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        const buildBtn = document.getElementById('build-btn');
                        const progressContainer = document.getElementById('progress-container');
                        const progressBar = document.getElementById('progress-bar');
                        const progressText = document.getElementById('progress-text');
                        const batchInfo = document.getElementById('batch-info');
                        const etaDiv = document.getElementById('eta');
                        
                        let statusText = '';
                        
                        if (data.ready) {
                            statusText = `<span class="success">✓ Brain Ready</span><br>Total chunks: ${data.chunks.toLocaleString()}`;
                            buildBtn.disabled = true;
                            buildBtn.textContent = 'Brain Complete';
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                            etaDiv.textContent = '';
                            
                        } else if (data.building) {
                            statusText = `<span class="building">⚡ Building Brain...</span><br>${data.progress}`;
                            
                            if (data.percentage > 0) {
                                progressContainer.style.display = 'block';
                                progressBar.style.width = data.percentage + '%';
                                progressText.textContent = `${data.percentage}%`;
                                
                                // Show batch info
                                if (data.total_batches > 0) {
                                    batchInfo.style.display = 'grid';
                                    document.getElementById('chunks-processed').textContent = data.chunks.toLocaleString();
                                    document.getElementById('batches-completed').textContent = `${data.batches_completed}/${data.total_batches}`;
                                    
                                    etaDiv.textContent = `Batch ${data.batches_completed + 1} of ${data.total_batches} in progress`;
                                }
                            } else {
                                progressContainer.style.display = 'none';
                                batchInfo.style.display = 'none';
                            }
                            
                            buildBtn.disabled = true;
                            buildBtn.textContent = 'Building...';
                            
                        } else if (data.error) {
                            statusText = `<span class="error">✗ Build Failed</span><br>${data.error}`;
                            buildBtn.disabled = false;
                            buildBtn.textContent = 'Retry Build';
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                            etaDiv.textContent = '';
                            
                        } else {
                            statusText = '<span style="color: #fbbf24;">○ Brain Not Built</span><br>Ready for batched processing';
                            buildBtn.disabled = false;
                            buildBtn.textContent = 'Build Brain';
                            progressContainer.style.display = 'none';
                            batchInfo.style.display = 'none';
                            etaDiv.textContent = '';
                        }
                        
                        statusDiv.innerHTML = statusText;
                    })
                    .catch(e => {
                        document.getElementById('status').innerHTML = `<span class="error">Connection error: ${e}</span>`;
                    });
            }
            
            function buildBrain() {
                fetch('/build_brain', { method: 'POST' })
                    .then(r => r.json())
                    .then(data => {
                        if (data.ok) {
                            startTime = Date.now();
                            statusInterval = setInterval(refreshStatus, 3000);
                        } else {
                            alert('Build failed: ' + data.error);
                        }
                    })
                    .catch(e => alert('Build request failed: ' + e));
            }
            
            // Initial status check
            refreshStatus();
            
            // Auto-refresh every 5 seconds
            setInterval(refreshStatus, 5000);
        </script>
    </body>
    </html>
    """
