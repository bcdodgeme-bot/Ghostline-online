# modules/google_oauth_config.py - Complete Google Ecosystem OAuth Configuration (Phase 2)

import os
from typing import List, Dict, Any

# =============================================================================
# GOOGLE OAUTH SCOPES - PHASE 2 COMPLETE
# =============================================================================

# Phase 2: Complete Google Ecosystem Scopes with Content Creation & Analytics
GOOGLE_OAUTH_SCOPES = [
    # Core Integration (Phase 1) - Email, Calendar, Drive
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    
    # Content Creation APIs (Phase 2) - Documents, Sheets, Slides
    "https://www.googleapis.com/auth/documents",  # Full Google Docs access
    "https://www.googleapis.com/auth/spreadsheets",  # Full Google Sheets access
    "https://www.googleapis.com/auth/presentations",  # Full Google Slides access
    "https://www.googleapis.com/auth/drive.file",  # Create/edit files created by app
    
    # Analytics & Performance APIs (Phase 2) - Analytics, Search Console
    "https://www.googleapis.com/auth/analytics.readonly",  # Google Analytics read access
    "https://www.googleapis.com/auth/webmasters.readonly",  # Search Console read access
]

# Alternative scope configurations for different security requirements
GOOGLE_OAUTH_SCOPES_READONLY = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/documents.readonly",
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/presentations.readonly",
    "https://www.googleapis.com/auth/analytics.readonly",
    "https://www.googleapis.com/auth/webmasters.readonly",
]

# Maximum permissions (for advanced use cases)
GOOGLE_OAUTH_SCOPES_FULL = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar",  # Full calendar access (read/write)
    "https://www.googleapis.com/auth/drive",  # Full drive access
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/analytics.readonly",
    "https://www.googleapis.com/auth/webmasters.readonly",
]

# Legacy Phase 1 scopes (for backwards compatibility)
GOOGLE_OAUTH_SCOPES_PHASE1 = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata.readonly"
]

def get_oauth_scopes(access_level: str = "standard") -> List[str]:
    """Get appropriate OAuth scopes based on access level needed
    
    Args:
        access_level: 'readonly', 'standard', 'full', or 'phase1'
        
    Returns:
        List of OAuth scope URLs
    """
    if access_level == "readonly":
        return GOOGLE_OAUTH_SCOPES_READONLY
    elif access_level == "full":
        return GOOGLE_OAUTH_SCOPES_FULL
    elif access_level == "phase1":
        return GOOGLE_OAUTH_SCOPES_PHASE1
    else:
        return GOOGLE_OAUTH_SCOPES  # Default to Phase 2 standard scopes

def validate_scopes_in_token(credentials) -> Dict[str, Any]:
    """Validate that current token has required scopes and determine integration phase
    
    Args:
        credentials: Google OAuth credentials object
        
    Returns:
        Dictionary with validation results and phase information
    """
    if not credentials or not hasattr(credentials, 'scopes'):
        return {
            'valid': False,
            'phase': 'none',
            'missing_scopes': GOOGLE_OAUTH_SCOPES,
            'message': 'No credentials or scopes available',
            'recommendations': ['Complete initial OAuth setup']
        }
    
    current_scopes = set(credentials.scopes or [])
    
    # Define scope sets for each phase
    phase1_core = {
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    }
    
    phase1_complete = set(GOOGLE_OAUTH_SCOPES_PHASE1)
    
    phase2_content = {
        "https://www.googleapis.com/auth/documents",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/presentations",
        "https://www.googleapis.com/auth/drive.file",
    }
    
    phase2_analytics = {
        "https://www.googleapis.com/auth/analytics.readonly",
        "https://www.googleapis.com/auth/webmasters.readonly",
    }
    
    phase2_complete = set(GOOGLE_OAUTH_SCOPES)
    
    # Calculate missing scopes for each level
    missing_phase1_core = phase1_core - current_scopes
    missing_phase1_complete = phase1_complete - current_scopes
    missing_phase2_content = phase2_content - current_scopes
    missing_phase2_analytics = phase2_analytics - current_scopes
    missing_phase2_complete = phase2_complete - current_scopes
    
    # Determine current integration phase and status
    if len(missing_phase2_complete) == 0:
        phase = "phase2_complete"
        message = "✅ Full Google ecosystem integration - all APIs available"
        valid = True
        recommendations = []
    elif len(missing_phase1_complete) == 0:
        if len(missing_phase2_content) == 0:
            phase = "phase2_content"
            message = f"✅ Content creation ready, missing {len(missing_phase2_analytics)} analytics scopes"
            valid = False
            recommendations = ["Add Analytics and Search Console scopes for full Phase 2"]
        elif len(missing_phase2_analytics) == 0:
            phase = "phase2_analytics"
            message = f"✅ Analytics ready, missing {len(missing_phase2_content)} content creation scopes"
            valid = False
            recommendations = ["Add Docs, Sheets, Slides scopes for content creation"]
        else:
            phase = "phase1_complete"
            message = f"✅ Phase 1 complete, missing {len(missing_phase2_complete)} Phase 2 scopes"
            valid = False
            recommendations = ["Re-authenticate with Phase 2 scopes for document creation and analytics"]
    elif len(missing_phase1_core) == 0:
        phase = "phase1_partial"
        message = f"⚠️ Core integration working, missing {len(missing_phase1_complete)} Phase 1 scopes"
        valid = False
        recommendations = ["Complete Phase 1 setup before proceeding to Phase 2"]
    else:
        phase = "incomplete"
        message = f"❌ Basic integration incomplete, missing {len(missing_phase1_core)} core scopes"
        valid = False
        recommendations = ["Complete initial OAuth setup with core Gmail/Calendar/Drive access"]
    
    return {
        'valid': valid,
        'phase': phase,
        'current_scopes': sorted(list(current_scopes)),
        'missing_scopes': sorted(list(missing_phase2_complete)),
        'phase1_missing': sorted(list(missing_phase1_complete)),
        'phase2_content_missing': sorted(list(missing_phase2_content)),
        'phase2_analytics_missing': sorted(list(missing_phase2_analytics)),
        'message': message,
        'recommendations': recommendations,
        'scope_breakdown': {
            'phase1_core': len(phase1_core & current_scopes),
            'phase1_total': len(phase1_complete & current_scopes),
            'phase2_content': len(phase2_content & current_scopes),
            'phase2_analytics': len(phase2_analytics & current_scopes),
            'total_granted': len(current_scopes)
        }
    }

def get_scope_descriptions() -> Dict[str, Dict[str, str]]:
    """Get human-readable descriptions of what each scope enables
    
    Returns:
        Nested dictionary with scope URLs, descriptions, and categories
    """
    return {
        # Phase 1 - Core Integration
        "https://www.googleapis.com/auth/gmail.readonly": {
            "description": "Read Gmail messages, labels, and search emails",
            "category": "Email Access",
            "phase": "1",
            "level": "readonly",
            "features": ["Email search", "Morning briefings", "Message summaries"]
        },
        "https://www.googleapis.com/auth/calendar.readonly": {
            "description": "Read calendar events, schedules, and meeting details",
            "category": "Calendar Access",
            "phase": "1",
            "level": "readonly",
            "features": ["Schedule viewing", "Meeting reminders", "Calendar search"]
        },
        "https://www.googleapis.com/auth/drive.readonly": {
            "description": "Read files and folders in Google Drive",
            "category": "File Access",
            "phase": "1",
            "level": "readonly",
            "features": ["File search", "Document preview", "Folder browsing"]
        },
        "https://www.googleapis.com/auth/drive.metadata.readonly": {
            "description": "Read Drive file metadata (names, dates, sizes)",
            "category": "File Metadata",
            "phase": "1",
            "level": "readonly",
            "features": ["File information", "Search optimization", "Organization tools"]
        },
        
        # Phase 2 - Content Creation
        "https://www.googleapis.com/auth/documents": {
            "description": "Create, read, and edit Google Docs documents",
            "category": "Document Creation",
            "phase": "2",
            "level": "read/write",
            "features": ["Create documents", "Edit content", "Format text", "Insert elements"]
        },
        "https://www.googleapis.com/auth/spreadsheets": {
            "description": "Create, read, and edit Google Sheets spreadsheets",
            "category": "Spreadsheet Operations",
            "phase": "2",
            "level": "read/write",
            "features": ["Create sheets", "Data analysis", "Chart creation", "Formula automation"]
        },
        "https://www.googleapis.com/auth/presentations": {
            "description": "Create, read, and edit Google Slides presentations",
            "category": "Presentation Creation",
            "phase": "2",
            "level": "read/write",
            "features": ["Create slides", "Add content", "Design layouts", "Media insertion"]
        },
        "https://www.googleapis.com/auth/drive.file": {
            "description": "Create and edit files through the application",
            "category": "File Creation",
            "phase": "2",
            "level": "limited write",
            "features": ["App-created files", "Shared documents", "Template generation"]
        },
        
        # Phase 2 - Analytics & Performance
        "https://www.googleapis.com/auth/analytics.readonly": {
            "description": "Read Google Analytics website traffic and user data",
            "category": "Website Analytics",
            "phase": "2",
            "level": "readonly",
            "features": ["Traffic reports", "User behavior", "Conversion tracking", "Performance metrics"]
        },
        "https://www.googleapis.com/auth/webmasters.readonly": {
            "description": "Read Search Console SEO and search performance data",
            "category": "SEO Analytics",
            "phase": "2",
            "level": "readonly",
            "features": ["Search rankings", "Keyword performance", "Site health", "Index status"]
        },
    }

def get_required_apis() -> List[Dict[str, Any]]:
    """Get comprehensive list of Google Cloud APIs that need to be enabled
    
    Returns:
        List of API configurations with setup details
    """
    return [
        # Phase 1 APIs
        {
            'name': 'Gmail API',
            'api_name': 'gmail.googleapis.com',
            'description': 'Email access, search, and message reading',
            'phase': 1,
            'required': True,
            'enable_url': 'https://console.cloud.google.com/apis/library/gmail.googleapis.com',
            'documentation': 'https://developers.google.com/gmail/api',
            'quota_limits': 'Basic: 1 billion requests/day'
        },
        {
            'name': 'Google Calendar API',
            'api_name': 'calendar-json.googleapis.com',
            'description': 'Calendar events, schedules, and meeting management',
            'phase': 1,
            'required': True,
            'enable_url': 'https://console.cloud.google.com/apis/library/calendar-json.googleapis.com',
            'documentation': 'https://developers.google.com/calendar',
            'quota_limits': 'Basic: 1 million requests/day'
        },
        {
            'name': 'Google Drive API',
            'api_name': 'drive.googleapis.com',
            'description': 'File storage, document access, and folder management',
            'phase': 1,
            'required': True,
            'enable_url': 'https://console.cloud.google.com/apis/library/drive.googleapis.com',
            'documentation': 'https://developers.google.com/drive',
            'quota_limits': 'Basic: 1 billion requests/day'
        },
        
        # Phase 2 Content APIs
        {
            'name': 'Google Docs API',
            'api_name': 'docs.googleapis.com',
            'description': 'Document creation, editing, and formatting',
            'phase': 2,
            'required': False,
            'enable_url': 'https://console.cloud.google.com/apis/library/docs.googleapis.com',
            'documentation': 'https://developers.google.com/docs/api',
            'quota_limits': 'Basic: 300 requests/minute'
        },
        {
            'name': 'Google Sheets API',
            'api_name': 'sheets.googleapis.com',
            'description': 'Spreadsheet creation, data manipulation, and analysis',
            'phase': 2,
            'required': False,
            'enable_url': 'https://console.cloud.google.com/apis/library/sheets.googleapis.com',
            'documentation': 'https://developers.google.com/sheets/api',
            'quota_limits': 'Basic: 300 requests/minute'
        },
        {
            'name': 'Google Slides API',
            'api_name': 'slides.googleapis.com',
            'description': 'Presentation creation, slide management, and design',
            'phase': 2,
            'required': False,
            'enable_url': 'https://console.cloud.google.com/apis/library/slides.googleapis.com',
            'documentation': 'https://developers.google.com/slides/api',
            'quota_limits': 'Basic: 300 requests/minute'
        },
        
        # Phase 2 Analytics APIs
        {
            'name': 'Google Analytics Reporting API',
            'api_name': 'analyticsreporting.googleapis.com',
            'description': 'Website traffic analysis and user behavior insights',
            'phase': 2,
            'required': False,
            'enable_url': 'https://console.cloud.google.com/apis/library/analyticsreporting.googleapis.com',
            'documentation': 'https://developers.google.com/analytics/devguides/reporting/core/v4',
            'quota_limits': 'Basic: 10,000 requests/day',
            'setup_notes': 'Requires Analytics account and View ID configuration'
        },
        {
            'name': 'Google Search Console API',
            'api_name': 'searchconsole.googleapis.com',
            'description': 'Search performance, SEO data, and site health monitoring',
            'phase': 2,
            'required': False,
            'enable_url': 'https://console.cloud.google.com/apis/library/searchconsole.googleapis.com',
            'documentation': 'https://developers.google.com/webmaster-tools',
            'quota_limits': 'Basic: 1,200 requests/day',
            'setup_notes': 'Requires verified website in Search Console'
        }
    ]

def generate_oauth_redirect_uri(railway_url: str = None, local_port: int = 5000) -> str:
    """Generate the appropriate OAuth redirect URI for current environment
    
    Args:
        railway_url: Railway deployment URL (if deployed)
        local_port: Local development port
        
    Returns:
        Complete OAuth redirect URI
    """
    if railway_url:
        return f"https://{railway_url}/google/auth/callback"
    else:
        return f"http://localhost:{local_port}/google/auth/callback"

def validate_environment_config() -> Dict[str, Any]:
    """Validate current environment configuration for Google integration
    
    Returns:
        Dictionary with configuration status and recommendations
    """
    config_status = {
        'credentials_file': {
            'present': os.path.exists(os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')),
            'path': os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json'),
            'required': True
        },
        'token_file': {
            'present': os.path.exists(os.getenv('GOOGLE_TOKEN_PATH', 'token.json')),
            'path': os.getenv('GOOGLE_TOKEN_PATH', 'token.json'),
            'required': False  # Created during OAuth flow
        },
        'railway_url': {
            'present': bool(os.getenv('RAILWAY_STATIC_URL')),
            'value': os.getenv('RAILWAY_STATIC_URL'),
            'required': True  # For production deployment
        },
        'analytics_view_id': {
            'present': bool(os.getenv('GOOGLE_ANALYTICS_VIEW_ID')),
            'value': os.getenv('GOOGLE_ANALYTICS_VIEW_ID'),
            'required': False  # Only for Analytics features
        },
        'search_console_url': {
            'present': bool(os.getenv('SEARCH_CONSOLE_SITE_URL')),
            'value': os.getenv('SEARCH_CONSOLE_SITE_URL'),
            'required': False  # Only for Search Console features
        }
    }
    
    # Calculate overall status
    required_configs = [k for k, v in config_status.items() if v['required']]
    missing_required = [k for k in required_configs if not config_status[k]['present']]
    
    optional_configs = [k for k, v in config_status.items() if not v['required']]
    missing_optional = [k for k in optional_configs if not config_status[k]['present']]
    
    overall_status = {
        'ready_for_oauth': len(missing_required) == 0,
        'phase2_ready': config_status['analytics_view_id']['present'] and config_status['search_console_url']['present'],
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'config_details': config_status
    }
    
    return overall_status

# =============================================================================
# OAUTH FLOW TEMPLATES (Updated for Phase 2)
# =============================================================================

GOOGLE_SETUP_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Google OAuth Setup - Phase 2 Integration</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .btn { 
            background: #6366f1; color: white; border: none; padding: 12px 24px;
            border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
            text-decoration: none; display: inline-block;
        }
        .btn:hover { background: #5855eb; }
        .btn.success { background: #059669; }
        .btn.warning { background: #d97706; }
        .setup-section { background: #1a1a1a; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .setup-section h3 { color: #6366f1; margin-top: 0; }
        .setup-section ol li, .setup-section ul li { margin: 10px 0; line-height: 1.6; }
        .code-block { 
            background: #2a2a2a; padding: 15px; border-radius: 4px; 
            font-family: 'Courier New', monospace; margin: 10px 0; 
            border-left: 4px solid #6366f1;
        }
        .phase-banner { 
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            padding: 20px; border-radius: 8px; margin: 20px 0; text-align: center;
        }
        .phase-banner h2 { margin: 0; font-size: 28px; }
        .api-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px; margin: 20px 0;
        }
        .api-card {
            background: #2a2a2a; padding: 15px; border-radius: 8px;
            border-left: 4px solid #059669;
        }
        .api-card.phase2 { border-left-color: #6366f1; }
        .api-card h4 { margin: 0 0 10px 0; color: #fff; }
        .api-card p { margin: 5px 0; color: #d1d5db; font-size: 14px; }
        .warning-box { 
            background: #92400e; padding: 15px; border-radius: 8px; margin: 15px 0;
            border-left: 4px solid #f59e0b;
        }
        .feature-list {
            background: #1e3a8a; padding: 15px; border-radius: 8px; margin: 15px 0;
            border-left: 4px solid #3b82f6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="phase-banner">
            <h2>🚀 Google Ecosystem Integration - Phase 2</h2>
            <p>Complete productivity suite with document creation, analytics, and SEO monitoring</p>
        </div>
        
        <div class="warning-box">
            <strong>⚠️ Setup Required:</strong> This integration requires Google Cloud Console configuration and OAuth setup with expanded permissions for document creation and analytics access.
        </div>
        
        <div class="setup-section">
            <h3>📋 Step 1: Google Cloud Console Setup</h3>
            <ol>
                <li><strong>Create/Select Project:</strong>
                    <ul>
                        <li>Go to <a href="https://console.cloud.google.com/" target="_blank" style="color: #6366f1;">Google Cloud Console</a></li>
                        <li>Create a new project or select existing one</li>
                        <li>Note your Project ID for reference</li>
                    </ul>
                </li>
                <li><strong>Enable Required APIs:</strong>
                    <div class="api-grid">
                        <div class="api-card">
                            <h4>📧 Gmail API</h4>
                            <p><strong>ID:</strong> gmail.googleapis.com</p>
                            <p>Email access and search functionality</p>
                        </div>
                        <div class="api-card">
                            <h4>📅 Calendar API</h4>
                            <p><strong>ID:</strong> calendar-json.googleapis.com</p>
                            <p>Schedule management and meeting data</p>
                        </div>
                        <div class="api-card">
                            <h4>📁 Drive API</h4>
                            <p><strong>ID:</strong> drive.googleapis.com</p>
                            <p>File storage and document access</p>
                        </div>
                        <div class="api-card phase2">
                            <h4>📝 Docs API (NEW)</h4>
                            <p><strong>ID:</strong> docs.googleapis.com</p>
                            <p>Document creation and editing</p>
                        </div>
                        <div class="api-card phase2">
                            <h4>📊 Sheets API (NEW)</h4>
                            <p><strong>ID:</strong> sheets.googleapis.com</p>
                            <p>Spreadsheet operations and data analysis</p>
                        </div>
                        <div class="api-card phase2">
                            <h4>🎯 Analytics API (NEW)</h4>
                            <p><strong>ID:</strong> analyticsreporting.googleapis.com</p>
                            <p>Website traffic and user insights</p>
                        </div>
                    </div>
                </li>
                <li><strong>Configure OAuth 2.0:</strong>
                    <ul>
                        <li>Go to <strong>APIs & Services → Credentials</strong></li>
                        <li>Click <strong>+ CREATE CREDENTIALS → OAuth 2.0 Client ID</strong></li>
                        <li>Choose <strong>Web application</strong></li>
                        <li>Add authorized redirect URI:</li>
                    </ul>
                    <div class="code-block">https://{{ railway_url }}/google/auth/callback</div>
                </li>
                <li><strong>Download Credentials:</strong>
                    <ul>
                        <li>Download the credentials JSON file</li>
                        <li>Upload to Railway as <code>credentials.json</code></li>
                        <li>Set environment variable: <code>GOOGLE_CREDENTIALS_PATH=credentials.json</code></li>
                    </ul>
                </li>
            </ol>
        </div>
        
        <div class="setup-section">
            <h3>🔐 Step 2: Environment Variables</h3>
            <p>Configure these variables in Railway:</p>
            <div class="code-block">
# Required for OAuth
GOOGLE_CREDENTIALS_PATH=credentials.json
GOOGLE_TOKEN_PATH=token.json

# Optional Phase 2 configurations
GOOGLE_ANALYTICS_VIEW_ID=your_analytics_view_id
SEARCH_CONSOLE_SITE_URL=https://your-website.com
            </div>
        </div>
        
        <div class="feature-list">
            <h3>✨ Available Features After Setup</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div>
                    <h4>📧 Email & Calendar</h4>
                    <ul>
                        <li>Morning email briefings</li>
                        <li>Calendar event summaries</li>
                        <li>Email search and analysis</li>
                        <li>Meeting preparation</li>
                    </ul>
                </div>
                <div>
                    <h4>📝 Document Creation</h4>
                    <ul>
                        <li>Create Google Docs via chat</li>
                        <li>Generate spreadsheets</li>
                        <li>Auto-populate templates</li>
                        <li>Content collaboration</li>
                    </ul>
                </div>
                <div>
                    <h4>📊 Analytics & SEO</h4>
                    <ul>
                        <li>Website traffic reports</li>
                        <li>Search performance data</li>
                        <li>Keyword ranking monitoring</li>
                        <li>Automated insights</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <div class="setup-section">
            <h3>🎯 Step 3: Complete Setup</h3>
            <ol>
                <li>Ensure all APIs are enabled in Google Cloud Console</li>
                <li>Upload credentials.json to Railway</li>
                <li>Set required environment variables</li>
                <li>Return here and click "Start OAuth Flow"</li>
                <li>Complete authentication with expanded permissions</li>
                <li>Test integration with sample commands</li>
            </ol>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="/google/auth/start" class="btn success">🚀 Start OAuth Flow</a>
                <a href="/integrations" class="btn">📚 Setup Instructions</a>
                <a href="/" class="btn">🏠 Back to Chat</a>
            </div>
        </div>
        
        <div class="setup-section">
            <h3>📖 Sample Commands to Try</h3>
            <div class="code-block">
# Document Management
"create document 'Meeting Notes'"
"create spreadsheet 'Budget 2024'"
"add to document DOC_ID: New content here"

# Analytics & SEO  
"analytics report last month"
"search console data"
"website traffic summary"

# Email & Calendar (existing)
"good morning" 
"calendar for tomorrow"
"search emails about project"
            </div>
        </div>
    </div>
</body>
</html>
'''

OAUTH_SUCCESS_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Google Integration Complete - Phase 2</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
        }
        .container { max-width: 1000px; margin: 0 auto; }
        .btn { 
            background: #6366f1; color: white; border: none; padding: 12px 24px;
            border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
            text-decoration: none; display: inline-block;
        }
        .btn:hover { background: #5855eb; }
        .btn.success { background: #059669; }
        .btn.warning { background: #d97706; }
        .success-banner { 
            background: linear-gradient(135deg, #059669, #10b981);
            padding: 30px; border-radius: 8px; margin: 20px 0; text-align: center;
        }
        .success-banner h1 { margin: 0; font-size: 32px; }
        .test-results { 
            background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
            padding: 20px; margin: 20px 0; 
        }
        .scope-validation {
            background: #1e3a8a; border: 1px solid #3b82f6; border-radius: 8px;
            padding: 20px; margin: 20px 0;
        }
        .scope-breakdown {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin: 15px 0;
        }
        .scope-category {
            background: #2a2a2a; padding: 15px; border-radius: 8px;
            border-left: 4px solid #6366f1;
        }
        .scope-category.complete { border-left-color: #059669; }
        .scope-category.partial { border-left-color: #d97706; }
        .scope-category.missing { border-left-color: #dc2626; }
        .commands-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px; margin: 20px 0;
        }
        .command-category {
            background: #2a2a2a; padding: 15px; border-radius: 8px;
        }
        .command-category h4 { color: #6366f1; margin: 0 0 10px 0; }
        .command-category code {
            background: #1a1a1a; padding: 8px; border-radius: 4px;
            display: block; margin: 5px 0; color: #10b981;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="success-banner">
            <h1>🎉 Google Integration Complete!</h1>
            <p>Your complete Google ecosystem is now connected and ready for advanced automation</p>
        </div>
        
        <div class="test-results">
            <h3>📊 Integration Status</h3>
            <p><strong>Token saved to:</strong> {{ token_path }}</p>
            <p><strong>Total scopes granted:</strong> {{ scope_validation.scope_breakdown.total_granted }}</p>
            <p><strong>Integration phase:</strong> {{ scope_validation.phase }}</p>
            <p><strong>Status:</strong> {{ scope_validation.message }}</p>
        </div>
        
        <div class="scope-validation">
            <h3>🔐 Scope Validation Results</h3>
            <div class="scope-breakdown">
                <div class="scope-category {% if scope_validation.scope_breakdown.phase1_total >= 4 %}complete{% elif scope_validation.scope_breakdown.phase1_total > 0 %}partial{% else %}missing{% endif %}">
                    <h4>Phase 1 - Core Integration</h4>
                    <p>{{ scope_validation.scope_breakdown.phase1_total }}/4 scopes granted</p>
                    <small>Email, Calendar, Drive access</small>
                </div>
                <div class="scope-category {% if scope_validation.scope_breakdown.phase2_content >= 4 %}complete{% elif scope_validation.scope_breakdown.phase2_content > 0 %}partial{% else %}missing{% endif %}">
                    <h4>Phase 2 - Content Creation</h4>
                    <p>{{ scope_validation.scope_breakdown.phase2_content }}/4 scopes granted</p>
                    <small>Docs, Sheets, Slides creation</small>
                </div>
                <div class="scope-category {% if scope_validation.scope_breakdown.phase2_analytics >= 2 %}complete{% elif scope_validation.scope_breakdown.phase2_analytics > 0 %}partial{% else %}missing{% endif %}">
                    <h4>Phase 2 - Analytics</h4>
                    <p>{{ scope_validation.scope_breakdown.phase2_analytics }}/2 scopes granted</p>
                    <small>Analytics, Search Console</small>
                </div>
            </div>
            
            {% if not scope_validation.valid %}
            <div style="background: #92400e; padding: 15px; border-radius: 8px; margin: 15px 0;">
                <h4>⚠️ Incomplete Scope Setup</h4>
                <p>Some features may not be available. Missing scopes:</p>
                <ul>
                    {% for scope in scope_validation.missing_scopes %}
                    <li><code>{{ scope }}</code></li>
                    {% endfor %}
                </ul>
                <p><strong>Recommendations:</strong></p>
                <ul>
                    {% for rec in scope_validation.recommendations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        
        <div class="test-results">
            <h3>🔧 Service Connection Tests</h3>
            {% for service, result in test_results.items() %}
                <p><strong>{{ service.title() }}:</strong> {{ result }}</p>
            {% endfor %}
        </div>
        
        <div class="test-results">
            <h3>🚀 Available Commands</h3>
            <div class="commands-grid">
                <div class="command-category">
                    <h4>📧 Email & Calendar</h4>
                    <code>"good morning"</code>
                    <code>"calendar for tomorrow"</code>
                    <code>"search emails about budget"</code>
                    <code>"next meeting"</code>
                </div>
                <div class="command-category">
                    <h4>📝 Document Creation</h4>
                    <code>"create document 'Meeting Notes'"</code>
                    <code>"create spreadsheet 'Budget 2024'"</code>
                    <code>"add to document DOC_ID: content"</code>
                </div>
                <div class="command-category">
                    <h4>📊 Analytics & SEO</h4>
                    <code>"analytics report"</code>
                    <code>"website traffic last month"</code>
                    <code>"search console data"</code>
                    <code>"seo performance"</code>
                </div>
                <div class="command-category">
                    <h4>📁 File Management</h4>
                    <code>"drive search project proposal"</code>
                    <code>"find document quarterly report"</code>
                    <code>"read sheet SHEET_ID"</code>
                </div>
            </div>
        </div>
        
        <div style="text-align: center; margin: 40px 0;">
            <a href="/" class="btn success">🎯 Start Using Google Commands</a>
            <a href="/integrations" class="btn">🔧 View All Integrations</a>
            <a href="/google/auth/start" class="btn warning">🔄 Re-authenticate</a>
        </div>
        
        <div class="test-results">
            <h3>📚 Next Steps</h3>
            <ul>
                <li><strong>Test Basic Commands:</strong> Try "good morning" or "calendar for today" to verify core functionality</li>
                <li><strong>Create Your First Document:</strong> Use "create document 'Test Doc'" to test content creation</li>
                <li><strong>Set up Analytics:</strong> Configure GOOGLE_ANALYTICS_VIEW_ID for website insights</li>
                <li><strong>Verify Search Console:</strong> Set SEARCH_CONSOLE_SITE_URL for SEO monitoring</li>
                <li><strong>Explore Advanced Features:</strong> Try combining commands for powerful automation workflows</li>
            </ul>
        </div>
    </div>
</body>
</html>
'''
