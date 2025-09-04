# modules/google_diagnostics.py - Google Integration Diagnostics

import os
import json
import requests
import datetime
from typing import Dict, Any, List, Tuple

def check_google_integration_health() -> Dict[str, Any]:
    """Comprehensive Google integration health check"""
    
    results = {
        'timestamp': datetime.datetime.now().isoformat(),
        'overall_status': 'unknown',
        'authentication': {},
        'gmail_api': {},
        'calendar_api': {},
        'permissions': {},
        'recommendations': []
    }
    
    # Check authentication files
    results['authentication'] = check_authentication_files()
    
    # Check API connectivity
    if results['authentication']['token_valid']:
        results['gmail_api'] = test_gmail_api()
        results['calendar_api'] = test_calendar_api()
        results['permissions'] = check_api_permissions()
    else:
        results['gmail_api'] = {'status': 'cannot_test', 'reason': 'no_valid_token'}
        results['calendar_api'] = {'status': 'cannot_test', 'reason': 'no_valid_token'}
        results['permissions'] = {'status': 'cannot_test', 'reason': 'no_valid_token'}
    
    # Determine overall status and recommendations
    results['overall_status'] = determine_overall_status(results)
    results['recommendations'] = generate_recommendations(results)
    
    return results

def check_authentication_files() -> Dict[str, Any]:
    """Check for presence and validity of authentication files"""
    
    auth_check = {
        'credentials_file': {'exists': False, 'valid': False, 'path': None},
        'token_file': {'exists': False, 'valid': False, 'path': None, 'expires': None},
        'token_valid': False,
        'scopes_granted': []
    }
    
    # Check credentials file
    credentials_path = os.getenv('GOOGLE_CREDENTIALS_PATH', 'credentials.json')
    auth_check['credentials_file']['path'] = credentials_path
    
    if os.path.exists(credentials_path):
        auth_check['credentials_file']['exists'] = True
        try:
            with open(credentials_path, 'r') as f:
                creds_data = json.load(f)
                # Validate it's a proper OAuth credentials file
                if 'web' in creds_data and 'client_id' in creds_data['web']:
                    auth_check['credentials_file']['valid'] = True
                elif 'installed' in creds_data and 'client_id' in creds_data['installed']:
                    auth_check['credentials_file']['valid'] = True
        except Exception as e:
            auth_check['credentials_file']['error'] = str(e)
    
    # Check token file
    token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
    auth_check['token_file']['path'] = token_path
    
    if os.path.exists(token_path):
        auth_check['token_file']['exists'] = True
        try:
            with open(token_path, 'r') as f:
                token_data = json.load(f)
                
                # Check for required fields
                if all(field in token_data for field in ['token', 'refresh_token', 'token_uri']):
                    auth_check['token_file']['valid'] = True
                    
                    # Check expiration
                    if 'expiry' in token_data:
                        expiry = datetime.datetime.fromisoformat(token_data['expiry'].replace('Z', '+00:00'))
                        auth_check['token_file']['expires'] = expiry.isoformat()
                        auth_check['token_valid'] = expiry > datetime.datetime.now(datetime.timezone.utc)
                    else:
                        auth_check['token_valid'] = True  # No expiry means it doesn't expire
                    
                    # Extract granted scopes
                    if 'scopes' in token_data:
                        auth_check['scopes_granted'] = token_data['scopes']
                        
        except Exception as e:
            auth_check['token_file']['error'] = str(e)
    
    return auth_check

def test_gmail_api() -> Dict[str, Any]:
    """Test Gmail API connectivity and permissions"""
    
    gmail_test = {
        'status': 'unknown',
        'profile_access': False,
        'message_access': False,
        'search_access': False,
        'errors': [],
        'profile_data': None
    }
    
    try:
        from utils.gmail_client import _gmail_service
        
        # Test basic profile access
        try:
            gmail = _gmail_service()
            profile = gmail.users().getProfile(userId='me').execute()
            gmail_test['profile_access'] = True
            gmail_test['profile_data'] = {
                'email': profile.get('emailAddress'),
                'messages_total': profile.get('messagesTotal'),
                'history_id': profile.get('historyId')
            }
        except Exception as e:
            gmail_test['errors'].append(f"Profile access failed: {str(e)}")
        
        # Test message listing
        try:
            messages = gmail.users().messages().list(userId='me', maxResults=1).execute()
            if 'messages' in messages:
                gmail_test['message_access'] = True
            else:
                gmail_test['errors'].append("No messages returned from API")
        except Exception as e:
            gmail_test['errors'].append(f"Message access failed: {str(e)}")
        
        # Test search functionality
        try:
            search_results = gmail.users().messages().list(
                userId='me', 
                q='in:inbox', 
                maxResults=1
            ).execute()
            gmail_test['search_access'] = True
        except Exception as e:
            gmail_test['errors'].append(f"Search access failed: {str(e)}")
        
        # Determine overall status
        if gmail_test['profile_access'] and gmail_test['message_access']:
            gmail_test['status'] = 'healthy'
        elif gmail_test['profile_access']:
            gmail_test['status'] = 'limited'
        else:
            gmail_test['status'] = 'failed'
            
    except Exception as e:
        gmail_test['status'] = 'error'
        gmail_test['errors'].append(f"Gmail service initialization failed: {str(e)}")
    
    return gmail_test

def test_calendar_api() -> Dict[str, Any]:
    """Test Calendar API connectivity and permissions"""
    
    calendar_test = {
        'status': 'unknown',
        'calendar_list_access': False,
        'event_access': False,
        'primary_calendar_access': False,
        'errors': [],
        'calendars_found': 0,
        'calendar_names': []
    }
    
    try:
        from utils.gmail_client import _calendar_service
        
        # Test calendar list access
        try:
            calendar = _calendar_service()
            calendar_list = calendar.calendarList().list().execute()
            
            if 'items' in calendar_list:
                calendar_test['calendar_list_access'] = True
                calendar_test['calendars_found'] = len(calendar_list['items'])
                calendar_test['calendar_names'] = [
                    cal.get('summary', 'Unknown') for cal in calendar_list['items'][:5]
                ]
                
                # Check for primary calendar
                primary_found = any(cal.get('primary', False) for cal in calendar_list['items'])
                calendar_test['primary_calendar_access'] = primary_found
                
        except Exception as e:
            calendar_test['errors'].append(f"Calendar list access failed: {str(e)}")
        
        # Test event access on primary calendar
        try:
            now = datetime.datetime.utcnow().isoformat() + 'Z'
            tomorrow = (datetime.datetime.utcnow() + datetime.timedelta(days=1)).isoformat() + 'Z'
            
            events_result = calendar.events().list(
                calendarId='primary',
                timeMin=now,
                timeMax=tomorrow,
                maxResults=1,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            calendar_test['event_access'] = True
            
        except Exception as e:
            calendar_test['errors'].append(f"Event access failed: {str(e)}")
        
        # Determine overall status
        if calendar_test['calendar_list_access'] and calendar_test['event_access']:
            calendar_test['status'] = 'healthy'
        elif calendar_test['calendar_list_access']:
            calendar_test['status'] = 'limited'
        else:
            calendar_test['status'] = 'failed'
            
    except Exception as e:
        calendar_test['status'] = 'error'
        calendar_test['errors'].append(f"Calendar service initialization failed: {str(e)}")
    
    return calendar_test

def check_api_permissions() -> Dict[str, Any]:
    """Check what permissions/scopes are actually granted"""
    
    permissions = {
        'required_scopes': [
            'https://www.googleapis.com/auth/gmail.readonly',
            'https://www.googleapis.com/auth/calendar.readonly'
        ],
        'granted_scopes': [],
        'missing_scopes': [],
        'scope_status': 'unknown'
    }
    
    try:
        token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                token_data = json.load(f)
                granted = token_data.get('scopes', [])
                permissions['granted_scopes'] = granted
                
                # Check for missing scopes
                for required_scope in permissions['required_scopes']:
                    if required_scope not in granted:
                        permissions['missing_scopes'].append(required_scope)
                
                # Determine status
                if not permissions['missing_scopes']:
                    permissions['scope_status'] = 'complete'
                elif permissions['granted_scopes']:
                    permissions['scope_status'] = 'partial'
                else:
                    permissions['scope_status'] = 'none'
                    
    except Exception as e:
        permissions['error'] = str(e)
        permissions['scope_status'] = 'error'
    
    return permissions

def determine_overall_status(results: Dict[str, Any]) -> str:
    """Determine overall integration health status"""
    
    auth = results['authentication']
    gmail = results['gmail_api']
    calendar = results['calendar_api']
    
    # Critical issues
    if not auth['credentials_file']['exists']:
        return 'setup_required'
    
    if not auth['token_file']['exists'] or not auth['token_valid']:
        return 'authentication_required'
    
    # API issues
    if gmail.get('status') == 'failed' and calendar.get('status') == 'failed':
        return 'apis_failed'
    
    if gmail.get('status') == 'healthy' and calendar.get('status') == 'healthy':
        return 'healthy'
    
    if gmail.get('status') in ['healthy', 'limited'] or calendar.get('status') in ['healthy', 'limited']:
        return 'partially_working'
    
    return 'degraded'

def generate_recommendations(results: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations based on diagnostic results"""
    
    recommendations = []
    
    auth = results['authentication']
    gmail = results['gmail_api']
    calendar = results['calendar_api']
    permissions = results['permissions']
    
    # Authentication issues
    if not auth['credentials_file']['exists']:
        recommendations.append(
            "Download OAuth credentials from Google Cloud Console and set GOOGLE_CREDENTIALS_PATH"
        )
    elif not auth['credentials_file']['valid']:
        recommendations.append(
            "Replace corrupted credentials file with valid OAuth credentials from Google Cloud Console"
        )
    
    if not auth['token_file']['exists']:
        recommendations.append(
            "Complete OAuth flow by visiting /google/auth/start to generate access token"
        )
    elif not auth['token_valid']:
        recommendations.append(
            "Token expired or invalid - re-authenticate by visiting /google/auth/start"
        )
    
    # Scope/permission issues
    if permissions.get('missing_scopes'):
        recommendations.append(
            f"Missing API scopes: {', '.join(permissions['missing_scopes'])}. Re-authenticate with full permissions."
        )
    
    # API-specific issues
    if gmail.get('status') == 'failed':
        if 'insufficient privileges' in str(gmail.get('errors', [])).lower():
            recommendations.append(
                "Gmail API access denied - ensure Gmail API is enabled in Google Cloud Console"
            )
        else:
            recommendations.append(
                "Gmail API connectivity issues - check network and API quotas"
            )
    
    if calendar.get('status') == 'failed':
        if 'insufficient privileges' in str(calendar.get('errors', [])).lower():
            recommendations.append(
                "Calendar API access denied - ensure Calendar API is enabled in Google Cloud Console"
            )
        else:
            recommendations.append(
                "Calendar API connectivity issues - check network and API quotas"
            )
    
    # Configuration recommendations
    if not recommendations:
        if gmail.get('status') == 'limited' or calendar.get('status') == 'limited':
            recommendations.append(
                "APIs working but with limited functionality - check specific error messages for details"
            )
        else:
            recommendations.append(
                "Google integrations appear to be working correctly"
            )
    
    return recommendations

def test_specific_functions() -> Dict[str, Any]:
    """Test specific Gmail/Calendar functions that are failing"""
    
    function_tests = {
        'list_overnight': {'status': 'unknown', 'error': None, 'result_count': 0},
        'list_today_events': {'status': 'unknown', 'error': None, 'result_count': 0},
        'get_next_meeting': {'status': 'unknown', 'error': None, 'has_data': False},
        'gmail_search': {'status': 'unknown', 'error': None, 'result_count': 0}
    }
    
    # Test list_overnight
    try:
        from utils.gmail_client import list_overnight
        msgs = list_overnight(include_unread=True, include_primary=False)
        
        if msgs is None:
            function_tests['list_overnight']['status'] = 'returns_none'
        elif isinstance(msgs, list):
            function_tests['list_overnight']['status'] = 'success'
            function_tests['list_overnight']['result_count'] = len(msgs)
        else:
            function_tests['list_overnight']['status'] = 'unexpected_type'
            function_tests['list_overnight']['error'] = f"Returned {type(msgs)}"
            
    except Exception as e:
        function_tests['list_overnight']['status'] = 'exception'
        function_tests['list_overnight']['error'] = str(e)
    
    # Test list_today_events
    try:
        from utils.gmail_client import list_today_events
        events = list_today_events(max_results=20)
        
        if events is None:
            function_tests['list_today_events']['status'] = 'returns_none'
        elif isinstance(events, list):
            function_tests['list_today_events']['status'] = 'success'
            function_tests['list_today_events']['result_count'] = len(events)
        else:
            function_tests['list_today_events']['status'] = 'unexpected_type'
            function_tests['list_today_events']['error'] = f"Returned {type(events)}"
            
    except Exception as e:
        function_tests['list_today_events']['status'] = 'exception'
        function_tests['list_today_events']['error'] = str(e)
    
    # Test get_next_meeting
    try:
        from utils.gmail_client import get_next_meeting
        next_meeting = get_next_meeting()
        
        if next_meeting is None:
            function_tests['get_next_meeting']['status'] = 'returns_none'
        elif isinstance(next_meeting, dict):
            function_tests['get_next_meeting']['status'] = 'success'
            function_tests['get_next_meeting']['has_data'] = bool(next_meeting.get('summary'))
        else:
            function_tests['get_next_meeting']['status'] = 'unexpected_type'
            function_tests['get_next_meeting']['error'] = f"Returned {type(next_meeting)}"
            
    except Exception as e:
        function_tests['get_next_meeting']['status'] = 'exception'
        function_tests['get_next_meeting']['error'] = str(e)
    
    # Test gmail_search
    try:
        from utils.gmail_client import search as gmail_search
        results = gmail_search("test")
        
        if results is None:
            function_tests['gmail_search']['status'] = 'returns_none'
        elif isinstance(results, list):
            function_tests['gmail_search']['status'] = 'success'
            function_tests['gmail_search']['result_count'] = len(results)
        else:
            function_tests['gmail_search']['status'] = 'unexpected_type'
            function_tests['gmail_search']['error'] = f"Returned {type(results)}"
            
    except Exception as e:
        function_tests['gmail_search']['status'] = 'exception'
        function_tests['gmail_search']['error'] = str(e)
    
    return function_tests

def generate_diagnostic_report() -> str:
    """Generate a comprehensive diagnostic report"""
    
    health_check = check_google_integration_health()
    function_tests = test_specific_functions()
    
    report = []
    report.append("# Google Integration Diagnostic Report")
    report.append(f"Generated: {health_check['timestamp']}")
    report.append(f"Overall Status: **{health_check['overall_status'].upper()}**")
    report.append("")
    
    # Authentication Status
    report.append("## Authentication Status")
    auth = health_check['authentication']
    
    report.append(f"- Credentials file: {'✓' if auth['credentials_file']['exists'] else '✗'} "
                 f"({auth['credentials_file']['path']})")
    report.append(f"- Token file: {'✓' if auth['token_file']['exists'] else '✗'} "
                 f"({auth['token_file']['path']})")
    report.append(f"- Token valid: {'✓' if auth['token_valid'] else '✗'}")
    
    if auth['scopes_granted']:
        report.append(f"- Granted scopes: {', '.join(auth['scopes_granted'])}")
    
    report.append("")
    
    # API Status
    report.append("## API Connectivity")
    
    gmail = health_check['gmail_api']
    calendar = health_check['calendar_api']
    
    report.append(f"### Gmail API: {gmail['status']}")
    report.append(f"- Profile access: {'✓' if gmail.get('profile_access') else '✗'}")
    report.append(f"- Message access: {'✓' if gmail.get('message_access') else '✗'}")
    report.append(f"- Search access: {'✓' if gmail.get('search_access') else '✗'}")
    
    if gmail.get('errors'):
        report.append("- Errors:")
        for error in gmail['errors']:
            report.append(f"  - {error}")
    
    report.append("")
    report.append(f"### Calendar API: {calendar['status']}")
    report.append(f"- Calendar list access: {'✓' if calendar.get('calendar_list_access') else '✗'}")
    report.append(f"- Event access: {'✓' if calendar.get('event_access') else '✗'}")
    report.append(f"- Primary calendar access: {'✓' if calendar.get('primary_calendar_access') else '✗'}")
    report.append(f"- Calendars found: {calendar.get('calendars_found', 0)}")
    
    if calendar.get('errors'):
        report.append("- Errors:")
        for error in calendar['errors']:
            report.append(f"  - {error}")
    
    report.append("")
    
    # Function Tests
    report.append("## Function Test Results")
    
    for func_name, test_result in function_tests.items():
        status_icon = {
            'success': '✓',
            'exception': '✗',
            'returns_none': '⚠',
            'unexpected_type': '⚠',
            'unknown': '?'
        }.get(test_result['status'], '?')
        
        report.append(f"- {func_name}: {status_icon} {test_result['status']}")
        
        if test_result.get('error'):
            report.append(f"  - Error: {test_result['error']}")
        if test_result.get('result_count') is not None:
            report.append(f"  - Results: {test_result['result_count']}")
        if test_result.get('has_data') is not None:
            report.append(f"  - Has data: {test_result['has_data']}")
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    for i, recommendation in enumerate(health_check['recommendations'], 1):
        report.append(f"{i}. {recommendation}")
    
    return "\n".join(report)