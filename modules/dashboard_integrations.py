# modules/dashboard_integrations.py
# Integrations Dashboard - Google Services, CRM, and Project Management

from flask import session, redirect, url_for, jsonify, render_template_string, request
import os
import datetime

def setup_integrations_routes(app):
    """Register all integrations dashboard routes"""
    
    @app.route('/integrations')
    def integrations_dashboard():
        """Unified integrations management dashboard"""
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ghostline Integrations</title>
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
                    font-size: 14px; transition: transform 0.2s; color: #6366f1;
                }
                .accordion-arrow.active { transform: rotate(90deg); }
                .accordion-content {
                    padding: 0; max-height: 0; overflow: hidden;
                    transition: max-height 0.3s ease-out, padding 0.3s ease-out;
                }
                .accordion-content.active { padding: 20px; max-height: 2000px; }
                
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
                
                /* Status Styles */
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
                .stat-number { font-size: 20px; font-weight: bold; color: #10b981; }
                .stat-label { color: #9ca3af; margin-top: 5px; }
                
                .actions-grid {
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px; margin: 20px 0;
                }
                .action-group {
                    background: #2a2a2a; padding: 15px; border-radius: 8px;
                }
                .action-title { font-size: 16px; font-weight: bold; margin-bottom: 10px; }
                
                .commands-grid {
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px; margin: 20px 0;
                }
                .command-card {
                    background: #2a2a2a; padding: 15px; border-radius: 8px;
                }
                .command-title {
                    font-size: 16px; font-weight: bold; margin-bottom: 8px; color: #6366f1;
                }
                .command-example {
                    background: #1a1a1a; padding: 8px; border-radius: 4px;
                    font-family: monospace; margin: 5px 0; font-size: 14px;
                }
                
                .setup-steps {
                    background: #1a1a1a; padding: 20px; border-radius: 8px; margin: 15px 0;
                }
                .setup-steps ol li { margin: 10px 0; line-height: 1.4; }
                
                .callback-url {
                    background: #2a2a2a; padding: 10px; border-radius: 4px;
                    font-family: monospace; word-break: break-all; margin: 10px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Integrations Management</h1>
                <p>Configure and manage external service connections for enhanced Ghostline functionality.</p>
                
                <!-- Google Services Section -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('google')">
                        <div>
                            <div class="accordion-title">Google Services (Gmail & Calendar)</div>
                            <div class="accordion-status" id="google-status-summary">Loading...</div>
                        </div>
                        <div class="accordion-arrow" id="google-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="google-content">
                        <div id="google-details">Loading Google services status...</div>
                        
                        <div class="stats-grid" id="google-stats">
                            <!-- Google stats will load here -->
                        </div>
                        
                        <div class="actions-grid">
                            <div class="action-group">
                                <div class="action-title">OAuth Setup</div>
                                <a href="/google/auth/start" class="btn success" id="oauth-btn">🔐 Start OAuth Setup</a>
                                <button class="btn secondary" onclick="refreshGoogleStatus()">🔄 Refresh Status</button>
                                <button class="btn danger" onclick="revokeGoogleAuth()" id="revoke-btn" style="display: none;">🗑️ Revoke Access</button>
                            </div>
                            <div class="action-group">
                                <div class="action-title">Quick Tests</div>
                                <button class="btn" onclick="testGmailConnection()">📧 Test Gmail</button>
                                <button class="btn" onclick="testCalendarConnection()">📅 Test Calendar</button>
                                <a href="/api/diagnostics/google-auth" class="btn secondary" target="_blank">🔧 Debug Auth</a>
                            </div>
                        </div>
                        
                        <div class="commands-grid">
                            <div class="command-card">
                                <div class="command-title">Daily Briefings</div>
                                <div class="command-example">good morning</div>
                                <div class="command-example">good evening</div>
                                <p>Complete daily summaries with emails and calendar</p>
                            </div>
                            
                            <div class="command-card">
                                <div class="command-title">Email Commands</div>
                                <div class="command-example">overnight</div>
                                <div class="command-example">search project alpha</div>
                                <p>Check overnight emails and search Gmail</p>
                            </div>
                            
                            <div class="command-card">
                                <div class="command-title">Calendar Commands</div>
                                <div class="command-example">calendar</div>
                                <div class="command-example">tomorrow</div>
                                <div class="command-example">next meeting</div>
                                <p>View schedules and upcoming meetings</p>
                            </div>
                        </div>
                        
                        <div class="setup-steps" id="google-setup" style="display: none;">
                            <h4>🔧 Google OAuth Setup Instructions</h4>
                            <ol>
                                <li>Go to <a href="https://console.cloud.google.com/" target="_blank">Google Cloud Console</a></li>
                                <li>Create/select project and enable Gmail API + Calendar API</li>
                                <li>Create OAuth 2.0 Client ID (Web application)</li>
                                <li>Add authorized redirect URI: <div class="callback-url" id="callback-url">Loading...</div></li>
                                <li>Download credentials JSON and set GOOGLE_CREDENTIALS_PATH</li>
                                <li>Click "Start OAuth Setup" above</li>
                            </ol>
                        </div>
                    </div>
                </div>
                
                <!-- CRM Integration Section -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('cloze')">
                        <div>
                            <div class="accordion-title">CRM Integration (Cloze)</div>
                            <div class="accordion-status" id="cloze-status-summary">Loading...</div>
                        </div>
                        <div class="accordion-arrow" id="cloze-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="cloze-content">
                        <div id="cloze-details">Loading Cloze CRM status...</div>
                        
                        <div class="stats-grid" id="cloze-stats">
                            <!-- Cloze stats will load here -->
                        </div>
                        
                        <div class="actions-grid">
                            <div class="action-group">
                                <div class="action-title">Quick Actions</div>
                                <button class="btn success" onclick="getClozeB

riefing()">📊 Morning Briefing</button>
                                <button class="btn" onclick="getClozePipeline()">🔄 Pipeline Summary</button>
                                <button class="btn secondary" onclick="refreshClozeStatus()">🔄 Refresh Status</button>
                            </div>
                            <div class="action-group">
                                <div class="action-title">Testing</div>
                                <button class="btn" onclick="testClozeConnection()">🔗 Test Connection</button>
                                <button class="btn" onclick="testClozeSearch()">🔍 Test Search</button>
                            </div>
                        </div>
                        
                        <div class="commands-grid">
                            <div class="command-card">
                                <div class="command-title">Morning Briefing</div>
                                <div class="command-example">cloze morning</div>
                                <div class="command-example">morning cloze</div>
                                <p>Get daily activity summary from Cloze</p>
                            </div>
                            
                            <div class="command-card">
                                <div class="command-title">Pipeline Summary</div>
                                <div class="command-example">cloze pipeline</div>
                                <div class="command-example">cloze deals</div>
                                <p>View deals and pipeline status</p>
                            </div>
                            
                            <div class="command-card">
                                <div class="command-title">Contact Search</div>
                                <div class="command-example">cloze search john smith</div>
                                <div class="command-example">cloze search acme corp</div>
                                <p>Search contacts in Cloze database</p>
                            </div>
                        </div>
                        
                        <div class="setup-steps">
                            <h4>🔧 Cloze CRM Setup Instructions</h4>
                            <ol>
                                <li>Email <strong>support@cloze.com</strong> to request API access</li>
                                <li>Get your API key from Cloze Pro settings</li>
                                <li>Set environment variable: <code>CLOZE_API_KEY=your_api_key</code></li>
                                <li>Restart Ghostline to activate integration</li>
                            </ol>
                        </div>
                    </div>
                </div>
                
                <!-- Project Management Section -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('clickup')">
                        <div>
                            <div class="accordion-title">Project Management (ClickUp)</div>
                            <div class="accordion-status" id="clickup-status-summary">Loading...</div>
                        </div>
                        <div class="accordion-arrow" id="clickup-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="clickup-content">
                        <div id="clickup-details">Loading ClickUp status...</div>
                        
                        <div class="stats-grid" id="clickup-stats">
                            <!-- ClickUp stats will load here -->
                        </div>
                        
                        <div class="actions-grid">
                            <div class="action-group">
                                <div class="action-title">Quick Actions</div>
                                <button class="btn success" onclick="getClickUpBriefing()">📋 Morning Briefing</button>
                                <button class="btn" onclick="getClickUpTimeToday()">⏰ Today's Time</button>
                                <button class="btn" onclick="getClickUpTasks()">📝 Tasks Summary</button>
                            </div>
                            <div class="action-group">
                                <div class="action-title">Testing</div>
                                <button class="btn" onclick="testClickUpConnection()">🔗 Test Connection</button>
                                <button class="btn secondary" onclick="refreshClickUpStatus()">🔄 Refresh Status</button>
                            </div>
                        </div>
                        
                        <div class="commands-grid">
                            <div class="command-card">
                                <div class="command-title">Daily Overview</div>
                                <div class="command-example">clickup morning</div>
                                <div class="command-example">clickup briefing</div>
                                <p>Daily task overview and time summary</p>
                            </div>
                            
                            <div class="command-card">
                                <div class="command-title">Time Tracking</div>
                                <div class="command-example">clickup time today</div>
                                <div class="command-example">clickup time week</div>
                                <p>View logged hours and productivity</p>
                            </div>
                            
                            <div class="command-card">
                                <div class="command-title">Task Management</div>
                                <div class="command-example">create clickup task: review proposal</div>
                                <div class="command-example">clickup tasks</div>
                                <p>Create tasks and view current workload</p>
                            </div>
                            
                            <div class="command-card">
                                <div class="command-title">Timer Controls</div>
                                <div class="command-example">start timer on project alpha</div>
                                <div class="command-example">stop timer</div>
                                <p>Control time tracking remotely</p>
                            </div>
                        </div>
                        
                        <div class="setup-steps">
                            <h4>🔧 ClickUp Setup Instructions</h4>
                            <ol>
                                <li>Go to your <strong>ClickUp Settings</strong></li>
                                <li>Navigate to <strong>Apps → API</strong></li>
                                <li>Generate a <strong>Personal API Token</strong></li>
                                <li>Set environment variable: <code>CLICKUP_API_TOKEN=your_token</code></li>
                                <li>Restart Ghostline to activate integration</li>
                            </ol>
                            <p><strong>Note:</strong> ClickUp API access may require a paid plan.</p>
                        </div>
                    </div>
                </div>
                
                <!-- Navigation -->
                <div style="text-align: center; margin: 30px 0; padding: 20px; background: #1a1a1a; border-radius: 8px;">
                    <a href="/" class="btn secondary">← Back to Chat</a>
                    <a href="/system" class="btn secondary">System Dashboard</a>
                    <a href="/diagnostics" class="btn secondary">Diagnostics</a>
                    <a href="/reports" class="btn secondary">Reports</a>
                    <button class="btn secondary" onclick="refreshAllStatus()">🔄 Refresh All</button>
                </div>
            </div>
            
            <script>
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
                        if (section === 'google') loadGoogleStatus();
                        if (section === 'cloze') loadClozeStatus();
                        if (section === 'clickup') loadClickUpStatus();
                    }
                }
                
                // Google Services Functions
                function loadGoogleStatus() {
                    fetch('/api/integrations/google-status')
                        .then(r => r.json())
                        .then(data => {
                            const summary = document.getElementById('google-status-summary');
                            const details = document.getElementById('google-details');
                            const stats = document.getElementById('google-stats');
                            const setupDiv = document.getElementById('google-setup');
                            const oauthBtn = document.getElementById('oauth-btn');
                            const revokeBtn = document.getElementById('revoke-btn');
                            
                            // Update callback URL
                            document.getElementById('callback-url').textContent = data.callback_url || 'Loading...';
                            
                            if (!data.credentials_file_exists) {
                                summary.innerHTML = '<span class="error">❌ Setup Required</span>';
                                details.innerHTML = '<span class="error">❌ Credentials file missing</span>';
                                setupDiv.style.display = 'block';
                                oauthBtn.style.display = 'none';
                                revokeBtn.style.display = 'none';
                            } else if (!data.token_file_exists) {
                                summary.innerHTML = '<span class="warning">⚠️ OAuth Needed</span>';
                                details.innerHTML = '<span class="warning">⚠️ OAuth authorization required</span>';
                                setupDiv.style.display = 'none';
                                oauthBtn.style.display = 'inline-block';
                                revokeBtn.style.display = 'none';
                            } else if (data.gmail_working && data.calendar_working) {
                                summary.innerHTML = '<span class="success">✅ Connected & Working</span>';
                                details.innerHTML = '<span class="success">✅ Gmail and Calendar connected</span>';
                                setupDiv.style.display = 'none';
                                oauthBtn.innerHTML = '🔄 Re-authorize';
                                revokeBtn.style.display = 'inline-block';
                            } else {
                                summary.innerHTML = '<span class="error">❌ Connection Issues</span>';
                                details.innerHTML = '<span class="error">❌ Services not responding</span>';
                                setupDiv.style.display = 'none';
                                oauthBtn.innerHTML = '🔄 Re-authorize';
                                oauthBtn.style.display = 'inline-block';
                                revokeBtn.style.display = 'inline-block';
                            }
                            
                            stats.innerHTML = `
                                <div class="stat-box">
                                    <div class="stat-number">${data.gmail_working ? 'Connected' : 'Disconnected'}</div>
                                    <div class="stat-label">Gmail Status</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.calendar_working ? 'Connected' : 'Disconnected'}</div>
                                    <div class="stat-label">Calendar Status</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.gmail_email || 'Unknown'}</div>
                                    <div class="stat-label">Email Account</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.calendar_count || 0}</div>
                                    <div class="stat-label">Calendars</div>
                                </div>
                            `;
                        })
                        .catch(e => {
                            document.getElementById('google-status-summary').innerHTML = '<span class="error">❌ Status Error</span>';
                        });
                }
                
                function refreshGoogleStatus() {
                    loadGoogleStatus();
                }
                
                function revokeGoogleAuth() {
                    if (confirm('This will revoke Google access. Continue?')) {
                        fetch('/google/auth/revoke', { method: 'POST' })
                            .then(r => r.json())
                            .then(data => {
                                if (data.success) {
                                    alert('✅ Google access revoked');
                                    loadGoogleStatus();
                                } else {
                                    alert('❌ Revocation failed: ' + data.error);
                                }
                            });
                    }
                }
                
                function testGmailConnection() {
                    alert('Testing Gmail connection...');
                    // Implementation would go here
                }
                
                function testCalendarConnection() {
                    alert('Testing Calendar connection...');
                    // Implementation would go here
                }
                
                // Cloze Functions
                function loadClozeStatus() {
                    fetch('/api/integrations/cloze-status')
                        .then(r => r.json())
                        .then(data => {
                            const summary = document.getElementById('cloze-status-summary');
                            const details = document.getElementById('cloze-details');
                            const stats = document.getElementById('cloze-stats');
                            
                            if (!data.configured) {
                                summary.innerHTML = '<span class="warning">⚠️ API Key Missing</span>';
                                details.innerHTML = '<span class="warning">⚠️ Set CLOZE_API_KEY environment variable</span>';
                            } else if (data.connection_working && data.user_info) {
                                summary.innerHTML = '<span class="success">✅ Connected</span>';
                                details.innerHTML = '<span class="success">✅ Connected to Cloze CRM</span>';
                            } else {
                                summary.innerHTML = '<span class="error">❌ Connection Failed</span>';
                                details.innerHTML = `<span class="error">❌ ${data.error || 'Unknown error'}</span>`;
                            }
                            
                            stats.innerHTML = `
                                <div class="stat-box">
                                    <div class="stat-number">${data.connection_working ? 'Connected' : 'Disconnected'}</div>
                                    <div class="stat-label">API Status</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.user_info?.name || 'Unknown'}</div>
                                    <div class="stat-label">User</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.user_info?.email || 'Unknown'}</div>
                                    <div class="stat-label">Email</div>
                                </div>
                            `;
                        })
                        .catch(e => {
                            document.getElementById('cloze-status-summary').innerHTML = '<span class="error">❌ Status Error</span>';
                        });
                }
                
                function refreshClozeStatus() {
                    loadClozeStatus();
                }
                
                function getClozeB

riefing() {
                    window.open('/cloze/briefing', '_blank');
                }
                
                function getClozePipeline() {
                    window.open('/cloze/pipeline', '_blank');
                }
                
                function testClozeConnection() {
                    alert('Testing Cloze connection...');
                }
                
                function testClozeSearch() {
                    const query = prompt('Enter search query:');
                    if (query) {
                        alert(`Searching Cloze for: ${query}`);
                    }
                }
                
                // ClickUp Functions
                function loadClickUpStatus() {
                    fetch('/api/integrations/clickup-status')
                        .then(r => r.json())
                        .then(data => {
                            const summary = document.getElementById('clickup-status-summary');
                            const details = document.getElementById('clickup-details');
                            const stats = document.getElementById('clickup-stats');
                            
                            if (!data.configured) {
                                summary.innerHTML = '<span class="warning">⚠️ API Token Missing</span>';
                                details.innerHTML = '<span class="warning">⚠️ Set CLICKUP_API_TOKEN environment variable</span>';
                            } else if (data.connection_working && data.user_info) {
                                summary.innerHTML = '<span class="success">✅ Connected</span>';
                                details.innerHTML = '<span class="success">✅ Connected to ClickUp</span>';
                            } else {
                                summary.innerHTML = '<span class="error">❌ Connection Failed</span>';
                                details.innerHTML = `<span class="error">❌ ${data.error || 'Unknown error'}</span>`;
                            }
                            
                            stats.innerHTML = `
                                <div class="stat-box">
                                    <div class="stat-number">${data.connection_working ? 'Connected' : 'Disconnected'}</div>
                                    <div class="stat-label">API Status</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.user_info?.username || 'Unknown'}</div>
                                    <div class="stat-label">User</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.team_info?.name || 'Unknown'}</div>
                                    <div class="stat-label">Team</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-number">${data.team_info?.members || 0}</div>
                                    <div class="stat-label">Members</div>
                                </div>
                            `;
                        })
                        .catch(e => {
                            document.getElementById('clickup-status-summary').innerHTML = '<span class="error">❌ Status Error</span>';
                        });
                }
                
                function refreshClickUpStatus() {
                    loadClickUpStatus();
                }
                
                function getClickUpBriefing() {
                    window.open('/clickup/briefing', '_blank');
                }
                
                function getClickUpTimeToday() {
                    window.open('/clickup/time/today', '_blank');
                }
                
                function getClickUpTasks() {
                    window.open('/clickup/tasks', '_blank');
                }
                
                function testClickUpConnection() {
                    alert('Testing ClickUp connection...');
                }
                
                // Utility Functions
                function refreshAllStatus() {
                    loadGoogleStatus();
                    loadClozeStatus();
                    loadClickUpStatus();
                }
                
                // Initialize on load
                document.addEventListener('DOMContentLoaded', function() {
                    refreshAllStatus();
                });
            </script>
        </body>
        </html>
        """)

    # API endpoints for integrations dashboard
    @app.route('/api/integrations/google-status')
    def api_google_status():
        """Get Google services status"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')
        token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        railway_url = os.getenv('RAILWAY_STATIC_URL', 'your-app.railway.app')
        
        status = {
            "credentials_file_exists": os.path.exists(credentials_path),
            "token_file_exists": os.path.exists(token_path),
            "gmail_working": False,
            "calendar_working": False,
            "callback_url": f"https://{railway_url}/google/auth/callback"
        }
        
        if status["token_file_exists"]:
            try:
                from utils.gmail_client import _build_creds, _gmail_service, _calendar_service
                
                # Test credentials
                creds = _build_creds()
                
                # Test Gmail API
                try:
                    gmail_svc = _gmail_service()
                    profile = gmail_svc.users().getProfile(userId='me').execute()
                    status["gmail_working"] = True
                    status["gmail_email"] = profile.get('emailAddress', 'Unknown')
                    status["gmail_total_messages"] = profile.get('messagesTotal', 0)
                except Exception:
                    pass
                
                # Test Calendar API
                try:
                    cal_svc = _calendar_service()
                    calendar_list = cal_svc.calendarList().list(maxResults=5).execute()
                    status["calendar_working"] = True
                    calendars = calendar_list.get('items', [])
                    status["calendar_count"] = len(calendars)
                    if calendars:
                        status["primary_calendar"] = next(
                            (cal.get('summary', 'Unknown') for cal in calendars if cal.get('primary')),
                            calendars[0].get('summary', 'Unknown')
                        )
                except Exception:
                    pass
                    
            except Exception as e:
                status["auth_error"] = str(e)
        
        return jsonify(status)

    @app.route('/api/integrations/cloze-status')
    def api_cloze_status():
        """Get Cloze CRM status"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        from modules.cloze_integration import is_cloze_configured
        
        status = {
            "configured": is_cloze_configured(),
            "api_key_present": bool(os.getenv('CLOZE_API_KEY')),
            "connection_working": False,
            "user_info": None
        }
        
        if is_cloze_configured():
            try:
                from modules.cloze_integration import ClozeClient
                client = ClozeClient()
                profile = client.get_profile()
                status["connection_working"] = True
                status["user_info"] = {
                    "name": profile.get('name', 'Unknown'),
                    "email": profile.get('email', 'Unknown')
                }
            except Exception as e:
                status["error"] = str(e)
        
        return jsonify(status)

    @app.route('/api/integrations/clickup-status')
    def api_clickup_status():
        """Get ClickUp status"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        from modules.clickup_integration import is_clickup_configured, ClickUpClient
        
        status = {
            "configured": is_clickup_configured(),
            "api_token_present": bool(os.getenv('CLICKUP_API_TOKEN')),
            "connection_working": False,
            "user_info": None,
            "team_info": None
        }
        
        if is_clickup_configured():
            try:
                client = ClickUpClient()
                
                # Test connection and get user info
                user_info = client.get_user_info()
                status["connection_working"] = True
                status["user_info"] = {
                    "username": user_info.get('user', {}).get('username', 'Unknown'),
                    "email": user_info.get('user', {}).get('email', 'Unknown'),
                    "id": user_info.get('user', {}).get('id', 'Unknown')
                }
                
                # Get team info
                teams = client.get_teams()
                if teams.get('teams'):
                    status["team_info"] = {
                        "name": teams['teams'][0].get('name', 'Unknown'),
                        "id": teams['teams'][0].get('id', 'Unknown'),
                        "members": len(teams['teams'][0].get('members', []))
                    }
                
            except Exception as e:
                status["error"] = str(e)
        
        return jsonify(status)