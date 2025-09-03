# modules/dashboard_system.py
# System Management Dashboard - Database, Brain & Backup Management

from flask import session, redirect, url_for, jsonify, render_template_string
from modules.database import get_db_connection
from modules.brain import get_brain_status
from modules.backup_maintenance import get_backup_status, backup_manager, start_automated_backups, stop_automated_backups
import os

def setup_system_routes(app):
    """Register all system dashboard routes"""
    
    @app.route('/system')
    def system_dashboard():
        """Unified system management dashboard with accordion sections"""
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ghostline System Management</title>
            <style>
                body { 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
                }
                .container { max-width: 1400px; margin: 0 auto; }
                
                /* Accordion Styles */
                .accordion {
                    background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
                    margin: 15px 0; overflow: hidden;
                }
                .accordion-header {
                    background: #2a2a2a; padding: 20px; cursor: pointer;
                    border-bottom: 1px solid #333; display: flex; 
                    justify-content: space-between; align-items: center;
                    transition: background 0.2s;
                }
                .accordion-header:hover { background: #333; }
                .accordion-header.active { background: #374151; }
                .accordion-title { font-size: 18px; font-weight: bold; }
                .accordion-status { font-size: 14px; color: #9ca3af; }
                .accordion-arrow {
                    font-size: 14px; transition: transform 0.2s;
                    color: #6366f1;
                }
                .accordion-arrow.active { transform: rotate(90deg); }
                .accordion-content {
                    padding: 0; max-height: 0; overflow: hidden;
                    transition: max-height 0.3s ease-out, padding 0.3s ease-out;
                }
                .accordion-content.active {
                    padding: 20px; max-height: 2000px;
                }
                
                /* Button Styles */
                .btn { 
                    background: #6366f1; color: white; border: none; padding: 12px 24px;
                    border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
                    text-decoration: none; display: inline-block;
                }
                .btn:hover { background: #5855eb; }
                .btn.success { background: #059669; }
                .btn.success:hover { background: #047857; }
                .btn.warning { background: #d97706; }
                .btn.danger { background: #dc2626; }
                .btn.secondary { background: #374151; }
                .btn.secondary:hover { background: #4b5563; }
                
                /* Status and Grid Styles */
                .success { color: #10b981; }
                .error { color: #ef4444; }
                .warning { color: #f59e0b; }
                .info { color: #3b82f6; }
                
                .stats-grid {
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px; margin: 20px 0;
                }
                .stat-box {
                    background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center;
                }
                .stat-number { font-size: 24px; font-weight: bold; color: #10b981; }
                .stat-label { color: #9ca3af; margin-top: 5px; }
                
                .actions-grid {
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px; margin: 20px 0;
                }
                .action-group {
                    background: #2a2a2a; padding: 15px; border-radius: 8px;
                }
                .action-title { font-size: 16px; font-weight: bold; margin-bottom: 10px; }
                
                .log-output {
                    background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
                    padding: 15px; font-family: monospace; font-size: 12px;
                    max-height: 300px; overflow-y: auto; white-space: pre-wrap;
                    margin: 15px 0;
                }
                
                .progress-bar {
                    width: 100%; height: 20px; background: #1a1a1a; border-radius: 10px;
                    overflow: hidden; margin: 10px 0;
                }
                .progress-fill {
                    height: 100%; background: #059669; transition: width 0.3s;
                }
                
                .backup-grid {
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px; margin: 20px 0;
                }
                .backup-card {
                    background: #2a2a2a; padding: 15px; border-radius: 8px;
                    border: 1px solid #404040;
                }
                .backup-filename { 
                    font-family: monospace; font-size: 12px; 
                    background: #1a1a1a; padding: 5px; border-radius: 4px; margin: 5px 0;
                    word-break: break-all;
                }
                .backup-size { color: #10b981; font-weight: bold; }
                .backup-date { color: #6b7280; font-size: 12px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>System Management Dashboard</h1>
                <p>Unified control center for database, brain, and backup operations.</p>
                
                <!-- Database Management Section -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('database')">
                        <div>
                            <div class="accordion-title">Database Management</div>
                            <div class="accordion-status" id="db-status-summary">Loading...</div>
                        </div>
                        <div class="accordion-arrow" id="database-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="database-content">
                        <div id="database-details">Loading database status...</div>
                        
                        <div class="stats-grid" id="database-stats">
                            <!-- Database stats will load here -->
                        </div>
                        
                        <div class="actions-grid">
                            <div class="action-group">
                                <div class="action-title">Connection Status</div>
                                <div id="db-connection-status">Checking...</div>
                            </div>
                            <div class="action-group">
                                <div class="action-title">Quick Actions</div>
                                <button class="btn secondary" onclick="refreshAllStatus()">Refresh Status</button>
                                <a href="/diagnostics" class="btn secondary">Health Check</a>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Brain System Section -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('brain')">
                        <div>
                            <div class="accordion-title">Brain & Knowledge System</div>
                            <div class="accordion-status" id="brain-status-summary">Loading...</div>
                        </div>
                        <div class="accordion-arrow" id="brain-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="brain-content">
                        <div id="brain-details">Loading brain status...</div>
                        
                        <div class="stats-grid" id="brain-stats">
                            <!-- Brain stats will load here -->
                        </div>
                        
                        <div class="actions-grid">
                            <div class="action-group">
                                <div class="action-title">Brain Operations</div>
                                <button class="btn success" onclick="buildBrain()">Build Brain</button>
                                <button class="btn warning" onclick="buildNewBrain()">Rebuild from Sources</button>
                                <button class="btn" onclick="refreshBrain()">Refresh Context</button>
                            </div>
                            <div class="action-group">
                                <div class="action-title">Diagnostics</div>
                                <button class="btn secondary" onclick="testSearch()">Test Search</button>
                                <a href="/diagnostics" class="btn secondary">Full Diagnostics</a>
                            </div>
                        </div>
                        
                        <div id="brain-progress-section" style="display: none;">
                            <h4>Operation Progress</h4>
                            <div class="progress-bar">
                                <div class="progress-fill" id="brain-progress-fill" style="width: 0%;"></div>
                            </div>
                            <div id="brain-progress-text">Ready...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Backup System Section -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('backup')">
                        <div>
                            <div class="accordion-title">Backup & Maintenance</div>
                            <div class="accordion-status" id="backup-status-summary">Loading...</div>
                        </div>
                        <div class="accordion-arrow" id="backup-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="backup-content">
                        <div id="backup-details">Loading backup status...</div>
                        
                        <div class="stats-grid" id="backup-stats">
                            <!-- Backup stats will load here -->
                        </div>
                        
                        <div class="actions-grid">
                            <div class="action-group">
                                <div class="action-title">Manual Backups</div>
                                <button class="btn success" onclick="createDatabaseBackup()">Database Backup</button>
                                <button class="btn success" onclick="createBrainBackup()">Brain Backup</button>
                                <button class="btn warning" onclick="createFullBackup()">Full System Backup</button>
                            </div>
                            <div class="action-group">
                                <div class="action-title">Maintenance</div>
                                <button class="btn" onclick="reindexKnowledge()">Reindex Knowledge</button>
                                <button class="btn warning" onclick="performMaintenance()">Full Maintenance</button>
                                <button class="btn secondary" id="scheduler-btn" onclick="toggleScheduler()">Start Auto-Backup</button>
                            </div>
                        </div>
                        
                        <div id="backup-progress-section" style="display: none;">
                            <h4>Operation Progress</h4>
                            <div class="progress-bar">
                                <div class="progress-fill" id="backup-progress-fill" style="width: 0%;"></div>
                            </div>
                            <div id="backup-progress-text">Ready...</div>
                        </div>
                        
                        <div class="log-output" id="operation-log">
System ready for operations...
Click any section above to expand and view detailed status information.
                        </div>
                        
                        <div id="recent-backups">
                            <h4>Recent Backups</h4>
                            <div class="backup-grid" id="backup-grid">
                                <!-- Recent backups will load here -->
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Navigation -->
                <div style="text-align: center; margin: 30px 0; padding: 20px; background: #1a1a1a; border-radius: 8px;">
                    <a href="/" class="btn secondary">← Back to Chat</a>
                    <a href="/reports" class="btn secondary">Reports Dashboard</a>
                    <a href="/integrations" class="btn secondary">Integrations</a>
                    <a href="/diagnostics" class="btn secondary">Diagnostics</a>
                    <button class="btn secondary" onclick="refreshAllStatus()">Refresh All Status</button>
                </div>
            </div>
            
            <script>
                let operationInProgress = false;
                let refreshInterval = null;
                
                // Accordion functionality
                function toggleAccordion(section) {
                    const content = document.getElementById(section + '-content');
                    const arrow = document.getElementById(section + '-arrow');
                    const header = content.previousElementSibling;
                    
                    const isActive = content.classList.contains('active');
                    
                    if (isActive) {
                        content.classList.remove('active');
                        arrow.classList.remove('active');
                        header.classList.remove('active');
                    } else {
                        content.classList.add('active');
                        arrow.classList.add('active');
                        header.classList.add('active');
                        
                        // Load section data when opened
                        if (section === 'database') loadDatabaseStatus();
                        if (section === 'brain') loadBrainStatus();
                        if (section === 'backup') loadBackupStatus();
                    }
                }
                
                // Status loading functions
                function loadDatabaseStatus() {
                    fetch('/api/system/database-status')
                        .then(r => r.json())
                        .then(data => {
                            const summary = document.getElementById('db-status-summary');
                            const details = document.getElementById('database-details');
                            const stats = document.getElementById('database-stats');
                            
                            if (data.database_url_configured && data.connection_working && data.tables_exist) {
                                summary.innerHTML = '<span class="success">✅ Connected & Ready</span>';
                                details.innerHTML = '<span class="success">✅ Database Connected & Ready</span>';
                            } else {
                                summary.innerHTML = '<span class="error">❌ Issues Detected</span>';
                                details.innerHTML = '<span class="error">❌ Database Issues Detected</span>';
                            }
                            
                            stats.innerHTML = `
                                <div class="stat-box">
                                    <div class="stat-number">${data.conversation_count}</div>
                                    <div class="stat-label">Conversations</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.uploaded_files_count}</div>
                                    <div class="stat-label">Files</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.daily_logs_count}</div>
                                    <div class="stat-label">Daily Logs</div>
                                </div>
                            `;
                        })
                        .catch(e => {
                            document.getElementById('db-status-summary').innerHTML = '<span class="error">❌ Status Error</span>';
                        });
                }
                
                function loadBrainStatus() {
                    fetch('/brain_status')
                        .then(r => r.json())
                        .then(data => {
                            const summary = document.getElementById('brain-status-summary');
                            const details = document.getElementById('brain-details');
                            const stats = document.getElementById('brain-stats');
                            
                            if (data.status === 'complete') {
                                summary.innerHTML = '<span class="success">✅ Ready</span>';
                                details.innerHTML = '<span class="success">✅ Brain System Ready</span>';
                            } else if (data.status === 'building') {
                                summary.innerHTML = '<span class="warning">🔄 Building</span>';
                                details.innerHTML = '<span class="warning">🔄 Brain Building in Progress</span>';
                            } else {
                                summary.innerHTML = '<span class="error">❌ Not Ready</span>';
                                details.innerHTML = '<span class="error">❌ Brain System Not Ready</span>';
                            }
                            
                            stats.innerHTML = `
                                <div class="stat-box">
                                    <div class="stat-number">${data.chunks_processed || 0}</div>
                                    <div class="stat-label">Knowledge Chunks</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${Math.round(data.progress || 0)}%</div>
                                    <div class="stat-label">Build Progress</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.batch_size || 0}</div>
                                    <div class="stat-label">Batch Size</div>
                                </div>
                            `;
                        })
                        .catch(e => {
                            document.getElementById('brain-status-summary').innerHTML = '<span class="error">❌ Status Error</span>';
                        });
                }
                
                function loadBackupStatus() {
                    fetch('/api/system/backup-status')
                        .then(r => r.json())
                        .then(data => {
                            const summary = document.getElementById('backup-status-summary');
                            const details = document.getElementById('backup-details');
                            const stats = document.getElementById('backup-stats');
                            const schedulerBtn = document.getElementById('scheduler-btn');
                            
                            if (data.scheduler_running) {
                                summary.innerHTML = '<span class="success">✅ Auto-Backup Active</span>';
                                details.innerHTML = '<span class="success">✅ Automated backups running</span>';
                                schedulerBtn.textContent = '⏸️ Stop Auto-Backup';
                                schedulerBtn.onclick = () => toggleScheduler(false);
                            } else {
                                summary.innerHTML = '<span class="warning">⚠️ Manual Mode</span>';
                                details.innerHTML = '<span class="warning">⚠️ Automated backups stopped</span>';
                                schedulerBtn.textContent = '▶️ Start Auto-Backup';
                                schedulerBtn.onclick = () => toggleScheduler(true);
                            }
                            
                            const totalBackups = data.recent_backups ? data.recent_backups.length : 0;
                            const totalSize = data.recent_backups ? 
                                data.recent_backups.reduce((sum, backup) => sum + (backup.size_bytes || 0), 0) : 0;
                            const lastBackup = data.recent_backups && data.recent_backups.length > 0 ? 
                                new Date(data.recent_backups[0].created).toLocaleDateString() : 'Never';
                            
                            stats.innerHTML = `
                                <div class="stat-box">
                                    <div class="stat-number">${totalBackups}</div>
                                    <div class="stat-label">Backups</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${formatBytes(totalSize)}</div>
                                    <div class="stat-label">Total Size</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${lastBackup}</div>
                                    <div class="stat-label">Last Backup</div>
                                </div>
                            `;
                            
                            updateBackupList(data.recent_backups);
                        })
                        .catch(e => {
                            document.getElementById('backup-status-summary').innerHTML = '<span class="error">❌ Status Error</span>';
                        });
                }
                
                function updateBackupList(backups) {
                    const gridDiv = document.getElementById('backup-grid');
                    
                    if (!backups || backups.length === 0) {
                        gridDiv.innerHTML = '<p>No backups found</p>';
                        return;
                    }
                    
                    gridDiv.innerHTML = backups.slice(0, 6).map(backup => `
                        <div class="backup-card">
                            <div class="backup-filename">${backup.filename}</div>
                            <div class="backup-size">${formatBytes(backup.size_bytes)}</div>
                            <div class="backup-date">${new Date(backup.created).toLocaleString()}</div>
                            <button class="btn secondary" onclick="downloadBackup('${backup.filename}')" 
                                    style="margin-top: 10px; padding: 8px 16px; font-size: 12px;">
                                💾 Download
                            </button>
                        </div>
                    `).join('');
                }
                
                // Operation functions
                function refreshAllStatus() {
                    loadDatabaseStatus();
                    loadBrainStatus();
                    loadBackupStatus();
                    addToLog('🔄 All status refreshed');
                }
                
                function buildBrain() {
                    if (operationInProgress) return;
                    executeOperation('/build_brain', 'Building brain system...');
                }
                
                function buildNewBrain() {
                    if (operationInProgress) return;
                    executeOperation('/build_new_brain', 'Rebuilding brain from sources...');
                }
                
                function refreshBrain() {
                    addToLog('🧠 Refreshing brain context...');
                    setTimeout(() => {
                        addToLog('✅ Brain context refreshed');
                        loadBrainStatus();
                    }, 1000);
                }
                
                function testSearch() {
                    const query = prompt('Enter search query:');
                    if (query) {
                        window.open(`/diagnostics?test=search&q=${encodeURIComponent(query)}`, '_blank');
                    }
                }
                
                function createDatabaseBackup() {
                    if (operationInProgress) return;
                    executeOperation('/backup/create-database', 'Creating database backup...');
                }
                
                function createBrainBackup() {
                    if (operationInProgress) return;
                    executeOperation('/backup/create-brain', 'Creating brain backup...');
                }
                
                function createFullBackup() {
                    if (operationInProgress) return;
                    executeOperation('/backup/create-full', 'Creating full system backup...');
                }
                
                function reindexKnowledge() {
                    if (operationInProgress) return;
                    executeOperation('/backup/reindex', 'Reindexing knowledge base...');
                }
                
                function performMaintenance() {
                    if (operationInProgress) return;
                    executeOperation('/backup/maintenance', 'Performing full maintenance...');
                }
                
                function toggleScheduler(start = null) {
                    if (operationInProgress) return;
                    
                    const action = start !== null ? start : !document.getElementById('scheduler-btn').textContent.includes('Stop');
                    const endpoint = action ? '/backup/start-scheduler' : '/backup/stop-scheduler';
                    
                    operationInProgress = true;
                    showProgress('Updating scheduler...', 0);
                    
                    fetch(endpoint, { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                addToLog(`✅ Scheduler ${action ? 'started' : 'stopped'} successfully`);
                                loadBackupStatus();
                            } else {
                                addToLog(`❌ Failed to ${action ? 'start' : 'stop'} scheduler: ${data.error}`);
                            }
                        })
                        .catch(e => {
                            addToLog(`❌ Scheduler operation failed: ${e.message}`);
                        })
                        .finally(() => {
                            operationInProgress = false;
                            hideProgress();
                        });
                }
                
                function executeOperation(endpoint, message) {
                    operationInProgress = true;
                    showProgress(message, 10);
                    addToLog(`🔄 ${message}`);
                    
                    fetch(endpoint, { method: 'POST' })
                        .then(response => {
                            showProgress(message, 50);
                            return response.json();
                        })
                        .then(data => {
                            showProgress('Processing results...', 90);
                            
                            if (data.success) {
                                addToLog(`✅ Operation completed successfully`);
                                
                                if (data.backup_files) {
                                    data.backup_files.forEach(file => {
                                        addToLog(`📁 Created: ${file}`);
                                    });
                                }
                                
                                if (data.size_bytes) {
                                    addToLog(`💾 Size: ${formatBytes(data.size_bytes)}`);
                                }
                                
                                refreshAllStatus();
                            } else {
                                addToLog(`❌ Operation failed: ${data.error || 'Unknown error'}`);
                            }
                        })
                        .catch(e => {
                            addToLog(`❌ Operation failed: ${e.message}`);
                        })
                        .finally(() => {
                            operationInProgress = false;
                            hideProgress();
                        });
                }
                
                function showProgress(text, percent) {
                    const progressSection = document.getElementById('backup-progress-section');
                    const progressFill = document.getElementById('backup-progress-fill');
                    const progressText = document.getElementById('backup-progress-text');
                    
                    progressSection.style.display = 'block';
                    progressFill.style.width = percent + '%';
                    progressText.textContent = text;
                }
                
                function hideProgress() {
                    const progressSection = document.getElementById('backup-progress-section');
                    progressSection.style.display = 'none';
                }
                
                function addToLog(message) {
                    const logDiv = document.getElementById('operation-log');
                    const timestamp = new Date().toLocaleTimeString();
                    logDiv.textContent += `\\n[${timestamp}] ${message}`;
                    logDiv.scrollTop = logDiv.scrollHeight;
                }
                
                function downloadBackup(filename) {
                    window.open(`/backup/download/${encodeURIComponent(filename)}`, '_blank');
                }
                
                function formatBytes(bytes) {
                    if (bytes === 0) return '0 B';
                    const k = 1024;
                    const sizes = ['B', 'KB', 'MB', 'GB'];
                    const i = Math.floor(Math.log(bytes) / Math.log(k));
                    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
                }
                
                // Initialize on load
                document.addEventListener('DOMContentLoaded', function() {
                    refreshAllStatus();
                    refreshInterval = setInterval(refreshAllStatus, 60000);
                });
            </script>
        </body>
        </html>
        """)

    # API endpoints for system dashboard
    @app.route('/api/system/database-status')
    def api_database_status():
        """Check database connection and table status"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        DATABASE_URL = os.getenv('DATABASE_URL')
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

    @app.route('/api/system/backup-status')
    def api_backup_status():
        """Get backup system status"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            return jsonify(get_backup_status())
        except Exception as e:
            return jsonify({'error': str(e)}), 500