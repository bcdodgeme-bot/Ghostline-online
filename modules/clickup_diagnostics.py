# modules/clickup_diagnostics.py - ClickUp Workspace Configuration and Diagnostics

import os
import json
import datetime
import requests
from typing import Dict, List, Any, Optional

class ClickUpDiagnostics:
    """Diagnostic and configuration tool for ClickUp integration"""
    
    def __init__(self):
        self.api_token = os.getenv('CLICKUP_API_TOKEN')
        self.base_url = "https://api.clickup.com/api/v2"
        self.headers = {
            "Authorization": self.api_token,
            "Content-Type": "application/json"
        }
    
    def _make_request(self, method, endpoint, data=None):
        """Make API request with detailed error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=30)
            
            # Log the response for debugging
            print(f"ClickUp API {method} {endpoint}: {response.status_code}")
            
            if response.status_code == 401:
                return {"error": "Invalid API token - check CLICKUP_API_TOKEN", "status_code": 401}
            elif response.status_code == 403:
                return {"error": "Access forbidden - check API token permissions", "status_code": 403}
            elif response.status_code == 429:
                return {"error": "Rate limit exceeded - try again later", "status_code": 429}
            elif not response.ok:
                return {"error": f"API error: {response.status_code} - {response.text}", "status_code": response.status_code}
            
            return response.json()
            
        except requests.exceptions.Timeout:
            return {"error": "Request timed out - check internet connection"}
        except requests.exceptions.ConnectionError:
            return {"error": "Connection failed - check internet connection"}
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from ClickUp API"}
    
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive ClickUp diagnostic"""
        
        results = {
            'timestamp': datetime.datetime.now().isoformat(),
            'api_configured': bool(self.api_token),
            'api_connection': 'unknown',
            'user_info': None,
            'teams': [],
            'workspaces': [],
            'recommended_config': None,
            'errors': [],
            'suggestions': []
        }
        
        # Check API token
        if not self.api_token:
            results['errors'].append("CLICKUP_API_TOKEN environment variable not set")
            results['suggestions'].append("Get API token from ClickUp Settings → Apps → API")
            return results
        
        # Test API connection
        user_info = self._make_request('GET', '/user')
        if 'error' in user_info:
            results['api_connection'] = 'failed'
            results['errors'].append(f"API connection failed: {user_info['error']}")
            return results
        
        results['api_connection'] = 'success'
        results['user_info'] = {
            'username': user_info.get('user', {}).get('username'),
            'email': user_info.get('user', {}).get('email'),
            'id': user_info.get('user', {}).get('id')
        }
        
        # Get teams/workspaces
        teams_response = self._make_request('GET', '/team')
        if 'error' in teams_response:
            results['errors'].append(f"Failed to get teams: {teams_response['error']}")
            return results
        
        teams = teams_response.get('teams', [])
        results['teams'] = len(teams)
        
        # Analyze each team/workspace
        for team in teams:
            team_id = team['id']
            team_name = team['name']
            
            workspace_info = {
                'team_id': team_id,
                'team_name': team_name,
                'spaces': [],
                'total_lists': 0,
                'suitable_for_ghostline': False
            }
            
            # Get spaces in this team
            spaces_response = self._make_request('GET', f'/team/{team_id}/space')
            if 'error' not in spaces_response:
                spaces = spaces_response.get('spaces', [])
                
                for space in spaces:
                    space_id = space['id']
                    space_name = space['name']
                    
                    space_info = {
                        'space_id': space_id,
                        'space_name': space_name,
                        'lists': []
                    }
                    
                    # Get lists in this space
                    lists_response = self._make_request('GET', f'/space/{space_id}/list')
                    if 'error' not in lists_response:
                        lists = lists_response.get('lists', [])
                        
                        for list_item in lists:
                            list_info = {
                                'list_id': list_item['id'],
                                'list_name': list_item['name'],
                                'task_count': list_item.get('task_count', 0)
                            }
                            space_info['lists'].append(list_info)
                            workspace_info['total_lists'] += 1
                    
                    workspace_info['spaces'].append(space_info)
            
            # Determine if suitable for Ghostline
            if workspace_info['total_lists'] > 0:
                workspace_info['suitable_for_ghostline'] = True
                
            results['workspaces'].append(workspace_info)
        
        # Generate recommendation
        results['recommended_config'] = self._generate_recommendation(results['workspaces'])
        
        # Generate suggestions
        results['suggestions'] = self._generate_suggestions(results)
        
        return results
    
    def _generate_recommendation(self, workspaces: List[Dict]) -> Optional[Dict[str, str]]:
        """Generate recommended configuration"""
        
        # Find the best workspace/list combination
        best_option = None
        best_score = 0
        
        for workspace in workspaces:
            if not workspace['suitable_for_ghostline']:
                continue
                
            for space in workspace['spaces']:
                for list_item in space['lists']:
                    # Score based on naming and task count
                    score = 0
                    
                    # Prefer lists with task-oriented names
                    list_name_lower = list_item['list_name'].lower()
                    if any(keyword in list_name_lower for keyword in ['inbox', 'task', 'todo', 'general', 'main']):
                        score += 3
                    
                    # Prefer lists with some but not too many tasks
                    task_count = list_item.get('task_count', 0)
                    if 0 <= task_count <= 50:
                        score += 2
                    elif task_count <= 100:
                        score += 1
                    
                    # Prefer workspaces with simple names
                    if any(keyword in workspace['team_name'].lower() for keyword in ['personal', 'main', 'default']):
                        score += 1
                    
                    if score > best_score:
                        best_score = score
                        best_option = {
                            'team_id': workspace['team_id'],
                            'team_name': workspace['team_name'],
                            'space_id': space['space_id'],
                            'space_name': space['space_name'],
                            'list_id': list_item['list_id'],
                            'list_name': list_item['list_name'],
                            'confidence': 'high' if score >= 4 else 'medium' if score >= 2 else 'low'
                        }
        
        return best_option
    
    def _generate_suggestions(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable suggestions"""
        suggestions = []
        
        if results['api_connection'] != 'success':
            suggestions.append("Fix API connection before proceeding")
            return suggestions
        
        if not results['workspaces']:
            suggestions.append("Create a ClickUp workspace first")
            return suggestions
        
        suitable_workspaces = [w for w in results['workspaces'] if w['suitable_for_ghostline']]
        if not suitable_workspaces:
            suggestions.append("Create at least one Space and List in your ClickUp workspace")
            suggestions.append("Consider creating a 'Ghostline' space with an 'Inbox' list")
            return suggestions
        
        if results['recommended_config']:
            config = results['recommended_config']
            suggestions.append(f"Use recommended configuration: {config['team_name']} → {config['space_name']} → {config['list_name']}")
            
            if config['confidence'] == 'low':
                suggestions.append("Consider creating a dedicated 'Ghostline' or 'Inbox' list for better organization")
        
        return suggestions
    
    def test_task_creation(self, list_id: str) -> Dict[str, Any]:
        """Test task creation in a specific list"""
        
        test_task_data = {
            "name": f"Ghostline Test Task - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "description": "Auto-generated test task from Ghostline integration diagnostic. Safe to delete.",
            "priority": 4  # Low priority
        }
        
        result = self._make_request('POST', f'/list/{list_id}/task', test_task_data)
        
        if 'error' in result:
            return {
                'success': False,
                'error': result['error'],
                'suggestion': 'Check list permissions and API token scope'
            }
        
        task_id = result.get('id')
        task_url = result.get('url', '')
        
        return {
            'success': True,
            'task_id': task_id,
            'task_url': task_url,
            'message': f"Successfully created test task: {test_task_data['name']}"
        }
    
    def create_ghostline_workspace_setup(self, team_id: str) -> Dict[str, Any]:
        """Create ideal Ghostline workspace setup"""
        
        # This would create a "Ghostline" space and "Inbox" list
        # Note: ClickUp API has limited space/list creation capabilities
        # Users may need to create these manually
        
        return {
            'success': False,
            'message': 'Automatic workspace creation not fully supported by ClickUp API',
            'manual_steps': [
                'Go to your ClickUp workspace',
                'Create a new Space called "Ghostline"',
                'Within that Space, create a List called "Inbox"',
                'Run diagnostics again to get the IDs'
            ]
        }

def generate_clickup_diagnostic_report() -> str:
    """Generate comprehensive diagnostic report"""
    
    try:
        diagnostics = ClickUpDiagnostics()
        results = diagnostics.run_full_diagnostic()
        
        report = []
        report.append("# ClickUp Integration Diagnostic Report")
        report.append(f"Generated: {results['timestamp']}")
        report.append("")
        
        # API Status
        report.append("## API Connection Status")
        report.append(f"- API Token Configured: {'✓' if results['api_configured'] else '✗'}")
        report.append(f"- API Connection: {'✓' if results['api_connection'] == 'success' else '✗'}")
        
        if results['user_info']:
            report.append(f"- Connected as: {results['user_info']['username']} ({results['user_info']['email']})")
        
        report.append("")
        
        # Workspace Analysis
        report.append("## Workspace Analysis")
        report.append(f"- Teams/Workspaces found: {results['teams']}")
        
        suitable_count = len([w for w in results['workspaces'] if w['suitable_for_ghostline']])
        report.append(f"- Suitable for Ghostline: {suitable_count}")
        
        for workspace in results['workspaces']:
            report.append(f"\n### {workspace['team_name']}")
            report.append(f"- Spaces: {len(workspace['spaces'])}")
            report.append(f"- Total Lists: {workspace['total_lists']}")
            report.append(f"- Suitable: {'✓' if workspace['suitable_for_ghostline'] else '✗'}")
            
            for space in workspace['spaces'][:3]:  # Show first 3 spaces
                report.append(f"  - **{space['space_name']}**: {len(space['lists'])} lists")
                for list_item in space['lists'][:2]:  # Show first 2 lists per space
                    report.append(f"    - {list_item['list_name']} ({list_item['task_count']} tasks)")
        
        report.append("")
        
        # Recommendation
        if results['recommended_config']:
            config = results['recommended_config']
            report.append("## Recommended Configuration")
            report.append(f"- **Team:** {config['team_name']}")
            report.append(f"- **Space:** {config['space_name']}")
            report.append(f"- **List:** {config['list_name']}")
            report.append(f"- **List ID:** `{config['list_id']}`")
            report.append(f"- **Confidence:** {config['confidence']}")
            report.append("")
            
            report.append("### Environment Variables to Set:")
            report.append(f"```")
            report.append(f"CLICKUP_DEFAULT_LIST_ID={config['list_id']}")
            report.append(f"CLICKUP_DEFAULT_TEAM_ID={config['team_id']}")
            report.append(f"```")
        else:
            report.append("## No Suitable Configuration Found")
            report.append("Manual workspace setup required.")
        
        report.append("")
        
        # Errors and Suggestions
        if results['errors']:
            report.append("## Errors Found")
            for error in results['errors']:
                report.append(f"- ❌ {error}")
            report.append("")
        
        if results['suggestions']:
            report.append("## Suggestions")
            for i, suggestion in enumerate(results['suggestions'], 1):
                report.append(f"{i}. {suggestion}")
        
        return "\n".join(report)
        
    except Exception as e:
        return f"Diagnostic failed: {str(e)}\n\nPlease check your CLICKUP_API_TOKEN and internet connection."

def get_clickup_workspace_tree() -> Dict[str, Any]:
    """Get hierarchical view of ClickUp workspace structure"""
    
    try:
        diagnostics = ClickUpDiagnostics()
        
        # Get teams
        teams_response = diagnostics._make_request('GET', '/team')
        if 'error' in teams_response:
            return {"error": teams_response['error']}
        
        workspace_tree = {}
        
        for team in teams_response.get('teams', []):
            team_id = team['id']
            team_name = team['name']
            
            team_data = {
                'team_id': team_id,
                'spaces': {}
            }
            
            # Get spaces
            spaces_response = diagnostics._make_request('GET', f'/team/{team_id}/space')
            if 'error' not in spaces_response:
                for space in spaces_response.get('spaces', []):
                    space_id = space['id']
                    space_name = space['name']
                    
                    space_data = {
                        'space_id': space_id,
                        'lists': {}
                    }
                    
                    # Get lists
                    lists_response = diagnostics._make_request('GET', f'/space/{space_id}/list')
                    if 'error' not in lists_response:
                        for list_item in lists_response.get('lists', []):
                            list_id = list_item['id']
                            list_name = list_item['name']
                            
                            space_data['lists'][list_name] = {
                                'list_id': list_id,
                                'task_count': list_item.get('task_count', 0)
                            }
                    
                    team_data['spaces'][space_name] = space_data
            
            workspace_tree[team_name] = team_data
        
        return {"workspace_tree": workspace_tree, "timestamp": datetime.datetime.now().isoformat()}
        
    except Exception as e:
        return {"error": str(e)}