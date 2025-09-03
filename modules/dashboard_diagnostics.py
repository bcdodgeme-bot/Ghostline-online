# modules/dashboard_diagnostics.py
# Debug & Diagnostics Dashboard - All testing and debugging tools

from flask import session, redirect, url_for, jsonify, render_template_string, request
from modules.database import get_db_connection
from modules.brain import enhanced_retrieve, get_brain_diagnostics
import datetime
import os

def setup_diagnostics_routes(app):
    """Register all diagnostics dashboard routes"""
    
    @app.route('/diagnostics')
    def diagnostics_dashboard():
        """Unified diagnostics and testing dashboard"""
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        
        return render_template_string("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ghostline Diagnostics & Testing</title>
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
                
                /* Button and Form Styles */
                .btn { 
                    background: #6366f1; color: white; border: none; padding: 12px 24px;
                    border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
                    text-decoration: none; display: inline-block;
                }
                .btn:hover { background: #5855eb; }
                .btn.success { background: #059669; }
                .btn.warning { background: #d97706; }
                .btn.danger { background: #dc2626; }
                .btn.secondary { background: #374151; }
                .btn.secondary:hover { background: #4b5563; }
                
                .test-input {
                    background: #1a1a1a; color: #fff; border: 1px solid #333; 
                    padding: 10px; border-radius: 8px; margin: 5px; width: 300px;
                }
                
                /* Status Styles */
                .success { color: #10b981; }
                .error { color: #ef4444; }
                .warning { color: #f59e0b; }
                .info { color: #3b82f6; }
                
                .test-grid {
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 15px; margin: 20px 0;
                }
                .test-card {
                    background: #2a2a2a; padding: 15px; border-radius: 8px;
                }
                .test-title { font-size: 16px; font-weight: bold; margin-bottom: 10px; }
                
                .result-output {
                    background: #1a1a1a; border: 1px solid #333; border-radius: 8px;
                    padding: 15px; font-family: monospace; font-size: 12px;
                    max-height: 300px; overflow-y: auto; white-space: pre-wrap;
                    margin: 15px 0;
                }
                
                .health-report {
                    background: #1a1a1a; padding: 20px; border-radius: 8px;
                    margin: 15px 0; line-height: 1.4;
                }
                
                .stats-grid {
                    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px; margin: 20px 0;
                }
                .stat-box {
                    background: #2a2a2a; padding: 15px; border-radius: 8px; text-align: center;
                }
                .stat-number { font-size: 24px; font-weight: bold; color: #10b981; }
                .stat-label { color: #9ca3af; margin-top: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Diagnostics & Testing Dashboard</h1>
                <p>Comprehensive system testing, debugging tools, and health monitoring for all Ghostline components.</p>
                
                <!-- Brain & Search Testing -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('brain')">
                        <div>
                            <div class="accordion-title">Brain & Search Diagnostics</div>
                            <div class="accordion-status" id="brain-status-summary">Click to test brain system</div>
                        </div>
                        <div class="accordion-arrow" id="brain-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="brain-content">
                        <div class="test-grid">
                            <div class="test-card">
                                <div class="test-title">Interactive Search Test</div>
                                <input type="text" id="search-query" class="test-input" placeholder="Enter search query..." value="Dead Like Me">
                                <input type="number" id="search-k" class="test-input" placeholder="Results" value="5" min="1" max="20" style="width: 100px;">
                                <button class="btn" onclick="testSearch()">Search</button>
                                <button class="btn secondary" onclick="testCommonQueries()">Test Common Queries</button>
                            </div>
                            
                            <div class="test-card">
                                <div class="test-title">Brain Health Check</div>
                                <button class="btn success" onclick="runHealthCheck()">Full Health Check</button>
                                <button class="btn" onclick="getBrainDiagnostics()">Detailed Diagnostics</button>
                                <button class="btn warning" onclick="testProblematicQueries()">Test Problem Queries</button>
                            </div>
                            
                            <div class="test-card">
                                <div class="test-title">Context & Indexing</div>
                                <button class="btn" onclick="refreshBrainContext()">Refresh Context</button>
                                <button class="btn secondary" onclick="checkIndexHealth()">Index Health</button>
                                <button class="btn warning" onclick="validateEmbeddings()">Validate Embeddings</button>
                            </div>
                        </div>
                        
                        <div id="brain-results" class="result-output">
Ready for brain and search testing...
Use the tools above to test search functionality, validate brain health, and diagnose any issues.
                        </div>
                    </div>
                </div>
                
                <!-- Database Testing -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('database')">
                        <div>
                            <div class="accordion-title">Database Diagnostics</div>
                            <div class="accordion-status" id="database-status-summary">Click to test database</div>
                        </div>
                        <div class="accordion-arrow" id="database-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="database-content">
                        <div class="stats-grid" id="database-stats">
                            <!-- Database stats will load here -->
                        </div>
                        
                        <div class="test-grid">
                            <div class="test-card">
                                <div class="test-title">Connection Tests</div>
                                <button class="btn success" onclick="testDatabaseConnection()">Test Connection</button>
                                <button class="btn" onclick="validateTables()">Validate Tables</button>
                                <button class="btn warning" onclick="checkIndexes()">Check Indexes</button>
                            </div>
                            
                            <div class="test-card">
                                <div class="test-title">Performance Tests</div>
                                <button class="btn" onclick="runQueryPerformanceTest()">Query Performance</button>
                                <button class="btn" onclick="checkTableSizes()">Table Sizes</button>
                                <button class="btn secondary" onclick="analyzeSlowQueries()">Slow Query Analysis</button>
                            </div>
                            
                            <div class="test-card">
                                <div class="test-title">Data Integrity</div>
                                <button class="btn" onclick="validateDataIntegrity()">Data Validation</button>
                                <button class="btn warning" onclick="checkDuplicates()">Check Duplicates</button>
                                <button class="btn danger" onclick="findCorruptedRecords()">Find Corrupted Data</button>
                            </div>
                        </div>
                        
                        <div id="database-results" class="result-output">
Ready for database testing...
Use the tools above to test connections, validate data integrity, and analyze performance.
                        </div>
                    </div>
                </div>
                
                <!-- Integration Testing -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('integrations')">
                        <div>
                            <div class="accordion-title">Integration Testing</div>
                            <div class="accordion-status" id="integrations-status-summary">Test external services</div>
                        </div>
                        <div class="accordion-arrow" id="integrations-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="integrations-content">
                        <div class="test-grid">
                            <div class="test-card">
                                <div class="test-title">Google Services</div>
                                <button class="btn success" onclick="testGoogleAuth()">Test OAuth</button>
                                <button class="btn" onclick="testGmailAPI()">Test Gmail API</button>
                                <button class="btn" onclick="testCalendarAPI()">Test Calendar API</button>
                            </div>
                            
                            <div class="test-card">
                                <div class="test-title">AI Services</div>
                                <button class="btn" onclick="testOpenRouterAPI()">Test OpenRouter</button>
                                <button class="btn" onclick="testElevenLabsAPI()">Test ElevenLabs</button>
                                <button class="btn" onclick="testReplicateAPI()">Test Replicate (FLUX)</button>
                            </div>
                            
                            <div class="test-card">
                                <div class="test-title">Business Integrations</div>
                                <button class="btn" onclick="testClozeAPI()">Test Cloze CRM</button>
                                <button class="btn" onclick="testClickUpAPI()">Test ClickUp</button>
                                <button class="btn" onclick="testTelegramBot()">Test Telegram Bot</button>
                            </div>
                        </div>
                        
                        <div id="integrations-results" class="result-output">
Ready for integration testing...
Use the tools above to test external API connections and service availability.
                        </div>
                    </div>
                </div>
                
                <!-- File Processing & OCR -->
                <div class="accordion">
                    <div class="accordion-header" onclick="toggleAccordion('files')">
                        <div>
                            <div class="accordion-title">File Processing & OCR</div>
                            <div class="accordion-status" id="files-status-summary">Test file handling</div>
                        </div>
                        <div class="accordion-arrow" id="files-arrow">▶</div>
                    </div>
                    <div class="accordion-content" id="files-content">
                        <div class="test-grid">
                            <div class="test-card">
                                <div class="test-title">OCR Testing</div>
                                <button class="btn success" onclick="testOCRSetup()">Test OCR Setup</button>
                                <button class="btn" onclick="testImageProcessing()">Test Image Processing</button>
                                <button class="btn warning" onclick="benchmarkOCR()">OCR Benchmark</button>
                            </div>
                            
                            <div class="test-card">
                                <div class="test-title">Document Processing</div>
                                <button class="btn" onclick="testPDFProcessing()">Test PDF Processing</button>
                                <button class="btn" onclick="testDocxProcessing()">Test DOCX Processing</button>
                                <button class="btn" onclick="testMarkdownProcessing()">Test Markdown</button>
                            </div>
                            
                            <div class="test-card">
                                <div class="test-title">Upload System</div>
                                <button class="btn" onclick="testUploadEndpoint()">Test Upload Endpoint</button>
                                <button class="btn" onclick="validateFileTypes()">Validate File Types</button>
                                <button class="btn secondary" onclick="checkStorageLimits()">Check Storage Limits</button>
                            </div>
                        </div>
                        
                        <div id="files-results" class="result-output">
Ready for file processing tests...
Use the tools above to test OCR, document parsing, and upload functionality.
                        </div>
                    </div>
                </div>
                
                <!-- System Health Overview -->
                <div class="health-report" id="system-health">
                    <h3>System Health Overview</h3>
                    <div id="health-summary">Click any test section above to begin diagnostics...</div>
                </div>
                
                <!-- Navigation -->
                <div style="text-align: center; margin: 30px 0; padding: 20px; background: #1a1a1a; border-radius: 8px;">
                    <a href="/" class="btn secondary">← Back to Chat</a>
                    <a href="/system" class="btn secondary">System Dashboard</a>
                    <a href="/integrations" class="btn secondary">Integrations</a>
                    <button class="btn success" onclick="runFullDiagnostics()">🔍 Run Full Diagnostics</button>
                    <button class="btn secondary" onclick="exportDiagnostics()">📋 Export Results</button>
                </div>
            </div>
            
            <script>
                let testResults = {};
                
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
                        if (section === 'database') loadDatabaseStats();
                    }
                }
                
                // Brain & Search Tests
                function testSearch() {
                    const query = document.getElementById('search-query').value.trim();
                    const k = parseInt(document.getElementById('search-k').value) || 5;
                    
                    if (!query) {
                        addResult('brain', '❌ Please enter a search query');
                        return;
                    }
                    
                    addResult('brain', `🔍 Searching for: "${query}" (k=${k})`);
                    
                    fetch(`/api/diagnostics/search?q=${encodeURIComponent(query)}&k=${k}`)
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                addResult('brain', `✅ Found ${data.results.length} results`);
                                data.results.forEach((result, i) => {
                                    const preview = result.text ? result.text.substring(0, 100) + '...' : 'No content';
                                    addResult('brain', `  ${i+1}. Source: ${result.source || 'Unknown'} (Score: ${result.score?.toFixed(4) || 'N/A'})`);
                                    addResult('brain', `     Preview: ${preview}`);
                                });
                            } else {
                                addResult('brain', `❌ Search failed: ${data.error}`);
                            }
                        })
                        .catch(e => {
                            addResult('brain', `❌ Search request failed: ${e.message}`);
                        });
                }
                
                function testCommonQueries() {
                    const queries = ['Dead Like Me', 'Happy Time', 'Carl', 'marketing', 'project management'];
                    addResult('brain', '🧪 Testing common problematic queries...');
                    
                    queries.forEach((query, i) => {
                        setTimeout(() => {
                            document.getElementById('search-query').value = query;
                            testSearch();
                        }, i * 2000); // 2 second delay between tests
                    });
                }
                
                function runHealthCheck() {
                    addResult('brain', '🩺 Running comprehensive brain health check...');
                    
                    fetch('/api/diagnostics/brain-health')
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                addResult('brain', '✅ Brain health check completed');
                                
                                // Display health metrics
                                Object.entries(data.health_metrics).forEach(([key, value]) => {
                                    const status = value.healthy ? '✅' : '❌';
                                    addResult('brain', `  ${status} ${key}: ${value.message}`);
                                });
                                
                                // Display search test results
                                if (data.search_tests) {
                                    addResult('brain', '\\n📊 Search Test Results:');
                                    data.search_tests.forEach(test => {
                                        const status = test.success ? '✅' : '❌';
                                        addResult('brain', `  ${status} "${test.query}": ${test.results} results`);
                                    });
                                }
                            } else {
                                addResult('brain', `❌ Health check failed: ${data.error}`);
                            }
                        })
                        .catch(e => {
                            addResult('brain', `❌ Health check failed: ${e.message}`);
                        });
                }
                
                function getBrainDiagnostics() {
                    addResult('brain', '🔬 Getting detailed brain diagnostics...');
                    
                    fetch('/api/diagnostics/brain-diagnostics')
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                addResult('brain', '✅ Brain diagnostics completed');
                                
                                // Display system info
                                if (data.system_info) {
                                    addResult('brain', '\\n💾 System Information:');
                                    Object.entries(data.system_info).forEach(([key, value]) => {
                                        addResult('brain', `  ${key}: ${value}`);
                                    });
                                }
                                
                                // Display index stats
                                if (data.index_stats) {
                                    addResult('brain', '\\n📈 Index Statistics:');
                                    Object.entries(data.index_stats).forEach(([key, value]) => {
                                        addResult('brain', `  ${key}: ${value}`);
                                    });
                                }
                            } else {
                                addResult('brain', `❌ Diagnostics failed: ${data.error}`);
                            }
                        })
                        .catch(e => {
                            addResult('brain', `❌ Diagnostics failed: ${e.message}`);
                        });
                }
                
                // Database Tests
                function loadDatabaseStats() {
                    fetch('/api/system/database-status')
                        .then(r => r.json())
                        .then(data => {
                            const statsDiv = document.getElementById('database-stats');
                            
                            statsDiv.innerHTML = `
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
                                <div class="stat-box">
                                    <div class="stat-number">${data.connection_working ? 'Connected' : 'Disconnected'}</div>
                                    <div class="stat-label">Status</div>
                                </div>
                            `;
                            
                            const status = data.connection_working && data.tables_exist ? 'Connected & Ready' : 'Issues Detected';
                            document.getElementById('database-status-summary').innerHTML = status;
                        })
                        .catch(e => {
                            document.getElementById('database-status-summary').innerHTML = 'Status Error';
                        });
                }
                
                function testDatabaseConnection() {
                    addResult('database', '🔗 Testing database connection...');
                    
                    fetch('/api/diagnostics/database-test')
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                addResult('database', '✅ Database connection successful');
                                addResult('database', `  Connection time: ${data.connection_time}ms`);
                                addResult('database', `  Database version: ${data.version || 'Unknown'}`);
                            } else {
                                addResult('database', `❌ Database connection failed: ${data.error}`);
                            }
                        })
                        .catch(e => {
                            addResult('database', `❌ Connection test failed: ${e.message}`);
                        });
                }
                
                // Integration Tests
                function testGoogleAuth() {
                    addResult('integrations', '🔐 Testing Google OAuth...');
                    
                    fetch('/api/diagnostics/google-auth')
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                addResult('integrations', '✅ Google OAuth working');
                                addResult('integrations', `  Gmail: ${data.gmail_working ? 'Working' : 'Failed'}`);
                                addResult('integrations', `  Calendar: ${data.calendar_working ? 'Working' : 'Failed'}`);
                            } else {
                                addResult('integrations', `❌ Google OAuth failed: ${data.error}`);
                            }
                        })
                        .catch(e => {
                            addResult('integrations', `❌ Google auth test failed: ${e.message}`);
                        });
                }
                
                function testOCRSetup() {
                    addResult('files', '👁️ Testing OCR setup...');
                    
                    fetch('/api/diagnostics/ocr-test')
                        .then(r => r.json())
                        .then(data => {
                            if (data.success) {
                                addResult('files', '✅ OCR system working');
                                addResult('files', `  EasyOCR available: ${data.easyocr_available ? 'Yes' : 'No'}`);
                                addResult('files', `  Supported languages: ${data.supported_languages?.join(', ') || 'Unknown'}`);
                            } else {
                                addResult('files', `❌ OCR test failed: ${data.error}`);
                            }
                        })
                        .catch(e => {
                            addResult('files', `❌ OCR test failed: ${e.message}`);
                        });
                }
                
                // Utility Functions
                function addResult(section, message) {
                    const resultDiv = document.getElementById(section + '-results');
                    const timestamp = new Date().toLocaleTimeString();
                    resultDiv.textContent += `\\n[${timestamp}] ${message}`;
                    resultDiv.scrollTop = resultDiv.scrollHeight;
                }
                
                function runFullDiagnostics() {
                    addResult('brain', '🚀 Starting comprehensive system diagnostics...');
                    
                    // Run all major tests
                    setTimeout(() => runHealthCheck(), 500);
                    setTimeout(() => testDatabaseConnection(), 2000);
                    setTimeout(() => testGoogleAuth(), 4000);
                    setTimeout(() => testOCRSetup(), 6000);
                    
                    // Update health summary
                    setTimeout(() => {
                        document.getElementById('health-summary').innerHTML = `
                            <strong>Full diagnostics completed at ${new Date().toLocaleString()}</strong><br>
                            Check individual sections above for detailed results.
                        `;
                    }, 8000);
                }
                
                function exportDiagnostics() {
                    const brainResults = document.getElementById('brain-results').textContent;
                    const dbResults = document.getElementById('database-results').textContent;
                    const intResults = document.getElementById('integrations-results').textContent;
                    const fileResults = document.getElementById('files-results').textContent;
                    
                    const fullReport = `GHOSTLINE DIAGNOSTICS REPORT
Generated: ${new Date().toISOString()}

=== BRAIN & SEARCH TESTS ===
${brainResults}

=== DATABASE TESTS ===
${dbResults}

=== INTEGRATION TESTS ===
${intResults}

=== FILE PROCESSING TESTS ===
${fileResults}

=== END REPORT ===`;
                    
                    const blob = new Blob([fullReport], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `ghostline_diagnostics_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.txt`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }
                
                // Placeholder functions for remaining tests
                function testProblematicQueries() { addResult('brain', '🧪 Testing problematic queries...'); }
                function refreshBrainContext() { addResult('brain', '🔄 Refreshing brain context...'); }
                function checkIndexHealth() { addResult('brain', '🔍 Checking index health...'); }
                function validateEmbeddings() { addResult('brain', '📊 Validating embeddings...'); }
                function validateTables() { addResult('database', '📋 Validating database tables...'); }
                function checkIndexes() { addResult('database', '🏃 Checking database indexes...'); }
                function runQueryPerformanceTest() { addResult('database', '⚡ Running query performance tests...'); }
                function testGmailAPI() { addResult('integrations', '📧 Testing Gmail API...'); }
                function testCalendarAPI() { addResult('integrations', '📅 Testing Calendar API...'); }
                function testOpenRouterAPI() { addResult('integrations', '🤖 Testing OpenRouter API...'); }
                function testElevenLabsAPI() { addResult('integrations', '🔊 Testing ElevenLabs API...'); }
                function testImageProcessing() { addResult('files', '🖼️ Testing image processing...'); }
                function testPDFProcessing() { addResult('files', '📄 Testing PDF processing...'); }
            </script>
        </body>
        </html>
        """)

    # API endpoints for diagnostics dashboard
    @app.route('/api/diagnostics/search')
    def api_diagnostic_search():
        """Test search functionality with detailed results"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        query = request.args.get('q', '').strip()
        k = int(request.args.get('k', 5))
        
        if not query:
            return jsonify({'success': False, 'error': 'No query provided'})
        
        try:
            results = enhanced_retrieve(query, k=k)
            return jsonify({
                'success': True,
                'query': query,
                'k': k,
                'results': results or [],
                'count': len(results) if results else 0
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/diagnostics/brain-health')
    def api_brain_health():
        """Comprehensive brain health check"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            health_metrics = {}
            search_tests = []
            
            # Test various search queries
            test_queries = [
                'Dead Like Me', 'Happy Time', 'television show', 
                'Carl', 'project management', 'marketing'
            ]
            
            for query in test_queries:
                try:
                    results = enhanced_retrieve(query, k=3)
                    search_tests.append({
                        'query': query,
                        'success': bool(results),
                        'results': len(results) if results else 0
                    })
                    
                    # Health metric for this query
                    health_metrics[f'search_{query.replace(" ", "_").lower()}'] = {
                        'healthy': bool(results and len(results) > 0),
                        'message': f'Found {len(results) if results else 0} results'
                    }
                except Exception as e:
                    search_tests.append({
                        'query': query,
                        'success': False,
                        'error': str(e),
                        'results': 0
                    })
                    
                    health_metrics[f'search_{query.replace(" ", "_").lower()}'] = {
                        'healthy': False,
                        'message': f'Search failed: {str(e)}'
                    }
            
            return jsonify({
                'success': True,
                'health_metrics': health_metrics,
                'search_tests': search_tests,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/diagnostics/brain-diagnostics')
    def api_brain_diagnostics():
        """Get detailed brain diagnostics"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            diagnostics = get_brain_diagnostics()
            return jsonify({
                'success': True,
                'system_info': diagnostics.get('system_info', {}),
                'index_stats': diagnostics.get('index_stats', {}),
                'performance_metrics': diagnostics.get('performance_metrics', {}),
                'timestamp': datetime.datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/diagnostics/database-test')
    def api_database_test():
        """Test database connection with timing"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        start_time = datetime.datetime.now()
        
        try:
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('SELECT version()')
                    version = cursor.fetchone()[0] if cursor.fetchone() else 'Unknown'
                    
                    connection_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
                    
                    return jsonify({
                        'success': True,
                        'connection_time': round(connection_time, 2),
                        'version': version,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                else:
                    return jsonify({'success': False, 'error': 'Failed to establish connection'})
                    
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/diagnostics/google-auth')
    def api_google_auth_test():
        """Test Google authentication and services"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            from utils.gmail_client import _build_creds, _gmail_service, _calendar_service
            
            # Test credential building
            creds = _build_creds()
            
            gmail_working = False
            calendar_working = False
            
            try:
                # Test Gmail
                gmail_svc = _gmail_service()
                profile = gmail_svc.users().getProfile(userId='me').execute()
                gmail_working = True
                gmail_email = profile.get('emailAddress', 'Unknown')
            except Exception as e:
                gmail_error = str(e)
            
            try:
                # Test Calendar
                cal_svc = _calendar_service()
                calendar_list = cal_svc.calendarList().list(maxResults=1).execute()
                calendar_working = True
            except Exception as e:
                calendar_error = str(e)
            
            return jsonify({
                'success': True,
                'credentials_valid': creds.valid if creds else False,
                'gmail_working': gmail_working,
                'calendar_working': calendar_working,
                'gmail_email': gmail_email if gmail_working else None,
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

    @app.route('/api/diagnostics/ocr-test')
    def api_ocr_test():
        """Test OCR system availability"""
        if not session.get('logged_in'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            import easyocr
            
            # Try to create a reader instance
            reader = easyocr.Reader(['en'])
            
            return jsonify({
                'success': True,
                'easyocr_available': True,
                'supported_languages': ['en'],  # Could be expanded to check reader.lang_list
                'timestamp': datetime.datetime.now().isoformat()
            })
            
        except ImportError:
            return jsonify({
                'success': False,
                'error': 'EasyOCR not installed',
                'easyocr_available': False
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e),
                'easyocr_available': False
            })