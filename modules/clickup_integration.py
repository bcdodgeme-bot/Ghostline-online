# modules/clickup_integration.py
# Updated ClickUp API integration with environment variable configuration

import os
import json
import datetime
import requests
import re
from flask import current_app
from modules.database import get_db_connection, save_conversation_enhanced

class ClickUpClient:
    def __init__(self):
        self.api_token = os.getenv('CLICKUP_API_TOKEN')
        if not self.api_token:
            raise ValueError("CLICKUP_API_TOKEN not configured - visit /diagnostics/clickup for setup")
        
        self.base_url = "https://api.clickup.com/api/v2"
        self.headers = {
            "Authorization": self.api_token,
            "Content-Type": "application/json"
        }
        
        # Try to get preconfigured IDs from environment
        self._team_id = os.getenv('CLICKUP_DEFAULT_TEAM_ID')
        self._default_list_id = os.getenv('CLICKUP_DEFAULT_LIST_ID')
        
        # Fallback cache
        self._cached_team_id = None
        self._cached_list_id = None
    
    def _make_request(self, method, endpoint, data=None):
        """Make API request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data, timeout=30)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, json=data, timeout=30)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=self.headers, timeout=30)
            
            if response.status_code == 401:
                raise Exception("Invalid ClickUp API token - check CLICKUP_API_TOKEN")
            elif response.status_code == 403:
                raise Exception("ClickUp API access forbidden - check token permissions")
            elif response.status_code == 429:
                raise Exception("ClickUp API rate limit exceeded - try again later")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise Exception("ClickUp API request timed out")
        except requests.exceptions.RequestException as e:
            current_app.logger.error(f"ClickUp API error: {e}")
            raise Exception(f"ClickUp API request failed: {str(e)}")
    
    def get_teams(self):
        """Get user's teams/workspaces"""
        return self._make_request('GET', '/team')
    
    def get_team_id(self):
        """Get the team ID (from env var or cache)"""
        # First try environment variable
        if self._team_id:
            return self._team_id
            
        # Then try cache
        if self._cached_team_id:
            return self._cached_team_id
        
        # Finally, query API
        try:
            teams = self.get_teams()
            if teams['teams']:
                self._cached_team_id = teams['teams'][0]['id']
                return self._cached_team_id
        except Exception as e:
            current_app.logger.error(f"Failed to get team ID: {e}")
            
        return None
    
    def get_spaces(self, team_id=None):
        """Get spaces in team"""
        if not team_id:
            team_id = self.get_team_id()
        
        if not team_id:
            raise Exception("No ClickUp team found - check workspace setup")
            
        return self._make_request('GET', f'/team/{team_id}/space')
    
    def get_lists(self, space_id):
        """Get lists in space"""
        return self._make_request('GET', f'/space/{space_id}/list')
    
    def get_default_list_id(self):
        """Get default list ID with better error messaging"""
        # First try environment variable
        if self._default_list_id:
            return self._default_list_id
            
        # Then try cache
        if self._cached_list_id:
            return self._cached_list_id
        
        # Finally, try to find automatically
        try:
            team_id = self.get_team_id()
            if not team_id:
                raise Exception("No ClickUp team found")
                
            spaces = self.get_spaces(team_id)
            
            if not spaces.get('spaces'):
                raise Exception("No ClickUp spaces found - create a Space in your workspace")
            
            # Look for the first space with lists
            for space in spaces['spaces']:
                space_id = space['id']
                try:
                    lists = self.get_lists(space_id)
                    if lists.get('lists'):
                        self._cached_list_id = lists['lists'][0]['id']
                        current_app.logger.info(f"Using ClickUp list: {lists['lists'][0]['name']} in {space['name']}")
                        return self._cached_list_id
                except Exception:
                    continue
            
            raise Exception("No ClickUp lists found - create a List in your workspace")
                        
        except Exception as e:
            current_app.logger.error(f"Failed to get default list: {e}")
            raise Exception(f"ClickUp workspace configuration error: {str(e)}\n\nVisit /diagnostics/clickup for setup help")
    
    def create_task(self, name, description="", due_date=None, priority=3, list_id=None):
        """Create a new task with better error handling"""
        try:
            if not list_id:
                list_id = self.get_default_list_id()
            
            task_data = {
                "name": name,
                "description": description,
                "priority": priority
            }
            
            if due_date:
                # Convert datetime to timestamp (milliseconds)
                if isinstance(due_date, datetime.datetime):
                    task_data["due_date"] = int(due_date.timestamp() * 1000)
            
            result = self._make_request('POST', f'/list/{list_id}/task', task_data)
            current_app.logger.info(f"Created ClickUp task: {name} (ID: {result.get('id')})")
            return result
            
        except Exception as e:
            error_msg = str(e)
            if "configuration error" in error_msg:
                # Pass through configuration errors with diagnostic link
                raise e
            else:
                raise Exception(f"Failed to create ClickUp task: {error_msg}")
    
    def get_tasks(self, list_id=None, **filters):
        """Get tasks with filters"""
        try:
            if not list_id:
                list_id = self.get_default_list_id()
            
            # Build query parameters
            params = {}
            if 'due_date_gt' in filters:
                params['due_date_gt'] = int(filters['due_date_gt'].timestamp() * 1000)
            if 'due_date_lt' in filters:
                params['due_date_lt'] = int(filters['due_date_lt'].timestamp() * 1000)
            
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            endpoint = f'/list/{list_id}/task'
            if query_string:
                endpoint += f'?{query_string}'
            
            return self._make_request('GET', endpoint)
            
        except Exception as e:
            current_app.logger.error(f"Failed to get tasks: {e}")
            return {"tasks": []}
    
    def get_time_entries(self, team_id=None, start_date=None, end_date=None):
        """Get time tracking entries"""
        try:
            if not team_id:
                team_id = self.get_team_id()
                
            if not team_id:
                raise Exception("No team ID available for time tracking")
            
            params = {}
            if start_date:
                params['start_date'] = int(start_date.timestamp() * 1000)
            if end_date:
                params['end_date'] = int(end_date.timestamp() * 1000)
            
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            endpoint = f'/team/{team_id}/time_entries'
            if query_string:
                endpoint += f'?{query_string}'
            
            return self._make_request('GET', endpoint)
            
        except Exception as e:
            current_app.logger.error(f"Failed to get time entries: {e}")
            return {"data": []}
    
    def start_time_tracking(self, task_id, description=""):
        """Start time tracking on a task"""
        data = {}
        if description:
            data['description'] = description
        
        return self._make_request('POST', f'/task/{task_id}/time', data)
    
    def stop_time_tracking(self, team_id=None):
        """Stop current time tracking"""
        if not team_id:
            team_id = self.get_team_id()
        
        if not team_id:
            raise Exception("No team ID available to stop timer")
        
        return self._make_request('DELETE', f'/team/{team_id}/time_entries/current')
    
    def get_user_info(self):
        """Get current user information"""
        return self._make_request('GET', '/user')

def is_clickup_configured():
    """Check if ClickUp API is configured"""
    return bool(os.getenv('CLICKUP_API_TOKEN'))

def process_clickup_command(user_input, project, use_voices, random_toggle):
    """Process ClickUp-related commands with better error handling"""
    if not is_clickup_configured():
        return None, False
    
    user_input_lower = user_input.lower()
    
    # ClickUp command patterns - more comprehensive
    clickup_patterns = [
        'clickup', 'click up', 'cu ', ' cu ', 'time track', 'my tasks', 'create task',
        'task list', 'task status', 'work timer', 'productivity'
    ]
    
    if not any(pattern in user_input_lower for pattern in clickup_patterns):
        return None, False
    
    try:
        client = ClickUpClient()
        
        # Setup/diagnostic command
        if any(keyword in user_input_lower for keyword in ['setup', 'configure', 'diagnostic']):
            response_data = {
                "SyntaxPrime": "**ClickUp Setup Required**\n\n"
                              "Visit `/diagnostics/clickup` for complete setup wizard.\n\n"
                              "Quick setup:\n"
                              "1. Get API token from ClickUp Settings → Apps → API\n"
                              "2. Set `CLICKUP_API_TOKEN` in Railway environment\n"
                              "3. Create workspace: Space → List structure\n"
                              "4. Run diagnostics to get configuration\n\n"
                              "**Current Status:** " + ("API Token Set" if is_clickup_configured() else "API Token Missing")
            }
            return response_data, True
        
        # Morning briefing / status
        if any(keyword in user_input_lower for keyword in ['morning', 'briefing', 'status', 'today']):
            briefing = get_clickup_morning_briefing(client)
            response_data = {"SyntaxPrime": briefing}
            return response_data, True
        
        # Time tracking queries
        if any(keyword in user_input_lower for keyword in ['time', 'hours', 'logged', 'tracking']):
            if 'today' in user_input_lower:
                time_summary = get_clickup_time_today(client)
            elif 'week' in user_input_lower:
                time_summary = get_clickup_time_week(client)
            else:
                time_summary = get_clickup_time_today(client)
            
            response_data = {"SyntaxPrime": time_summary}
            return response_data, True
        
        # Task creation with better parsing
        if any(keyword in user_input_lower for keyword in ['create', 'add', 'new task']):
            # Extract task name after various patterns
            patterns = [
                r'(?:create|add|new)\s+(?:clickup\s+)?task:?\s*(.+)',
                r'(?:create|add|new)\s+(.+)\s+(?:task|to-do)',
                r'task:?\s*(.+)'
            ]
            
            task_name = None
            for pattern in patterns:
                task_match = re.search(pattern, user_input, re.IGNORECASE)
                if task_match:
                    task_name = task_match.group(1).strip()
                    break
            
            if task_name:
                result = create_clickup_task(client, task_name, project)
                response_data = {"SyntaxPrime": result}
                return response_data, True
            else:
                response_data = {"SyntaxPrime": "Please specify a task name. Example: `create clickup task: Review quarterly report`"}
                return response_data, True
        
        # Task queries
        if any(keyword in user_input_lower for keyword in ['tasks', 'due', 'deadline', 'todo']):
            tasks_summary = get_clickup_tasks_summary(client)
            response_data = {"SyntaxPrime": tasks_summary}
            return response_data, True
        
        # Time tracking controls
        if 'start timer' in user_input_lower or 'start tracking' in user_input_lower:
            timer_match = re.search(r'start (?:timer|tracking) (?:on\s+)?(.+)', user_input, re.IGNORECASE)
            if timer_match:
                task_identifier = timer_match.group(1).strip()
                result = start_clickup_timer(client, task_identifier)
                response_data = {"SyntaxPrime": result}
                return response_data, True
        
        if 'stop timer' in user_input_lower or 'stop tracking' in user_input_lower:
            result = stop_clickup_timer(client)
            response_data = {"SyntaxPrime": result}
            return response_data, True
        
        # Default help response
        response_data = {
            "SyntaxPrime": "**ClickUp Commands Available:**\n\n"
                          "📋 **Tasks:**\n"
                          "• `create clickup task: [name]` - Create new task\n"
                          "• `clickup tasks` - View due tasks\n"
                          "• `clickup morning` - Daily briefing\n\n"
                          "⏱️ **Time Tracking:**\n"
                          "• `clickup time today` - Today's hours\n"
                          "• `clickup time week` - Weekly summary\n"
                          "• `start timer on [task]` - Start tracking\n"
                          "• `stop timer` - Stop current timer\n\n"
                          "⚙️ **Setup:**\n"
                          "• `clickup setup` - Configuration help\n"
                          "• Visit `/diagnostics/clickup` for full setup"
        }
        return response_data, True
        
    except Exception as e:
        current_app.logger.error(f"ClickUp command failed: {e}")
        
        error_message = str(e)
        if "configuration error" in error_message or "workspace setup" in error_message:
            # Configuration-related errors
            response_data = {
                "SyntaxPrime": f"**ClickUp Configuration Issue:**\n\n{error_message}\n\n"
                              "**Quick Fix:**\n"
                              "1. Visit `/diagnostics/clickup` for setup wizard\n"
                              "2. Or set these environment variables:\n"
                              "   - `CLICKUP_API_TOKEN` (from ClickUp Settings)\n"
                              "   - `CLICKUP_DEFAULT_LIST_ID` (from diagnostics)\n\n"
                              "Need help? Check the integration dashboard."
            }
        else:
            # Other API errors
            response_data = {"SyntaxPrime": f"ClickUp integration error: {error_message}"}
        
        return response_data, True

# Rest of your existing functions remain the same, but with improved error handling

def create_clickup_task(client, task_name, project=None):
    """Create a new ClickUp task with better error handling"""
    try:
        description = f"Created from Ghostline chat"
        if project:
            description += f" (Project: {project})"
        
        task = client.create_task(
            name=task_name,
            description=description,
            priority=3
        )
        
        task_id = task.get('id')
        task_url = task.get('url', '')
        
        response = f"✅ **Task Created:** {task_name}\n"
        response += f"🆔 **ID:** {task_id}\n"
        if task_url:
            response += f"🔗 **URL:** {task_url}"
        
        return response
        
    except Exception as e:
        error_msg = str(e)
        if "configuration error" in error_msg:
            return error_msg  # Pass through configuration errors
        else:
            return f"❌ **Task creation failed:** {error_msg}\n\nTry visiting `/diagnostics/clickup` for help."

# Keep all your existing helper functions (get_clickup_morning_briefing, etc.)
# but add this improved error handling pattern to each one

def get_clickup_morning_briefing(client=None):
    """Generate morning briefing with enhanced error handling"""
    if not client:
        try:
            client = ClickUpClient()
        except Exception as e:
            return f"**ClickUp Morning Briefing Unavailable**\n\nConfiguration error: {str(e)}\n\nVisit `/diagnostics/clickup` for setup help."
    
    try:
        briefing = ["📋 **CLICKUP MORNING BRIEFING**", ""]
        
        # Today's date range
        today = datetime.datetime.now()
        start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = today.replace(hour=23, minute=59, second=59, microsecond=59)
        
        # Get tasks due today
        today_tasks = client.get_tasks(
            due_date_gt=start_of_day,
            due_date_lt=end_of_day
        )
        
        briefing.append(f"**📅 Tasks Due Today ({len(today_tasks.get('tasks', []))}):**")
        if today_tasks.get('tasks'):
            for task in today_tasks['tasks'][:5]:  # Limit to 5 tasks
                status = "✅" if task.get('status', {}).get('status') == 'complete' else "📲"
                priority_map = {1: "🔴", 2: "🟡", 3: "🟢", 4: "🔵"}
                priority_icon = priority_map.get(task.get('priority', {}).get('priority', 3), "🟢")
                briefing.append(f"  {status} {priority_icon} {task['name']}")
        else:
            briefing.append("  No tasks due today")
        
        briefing.append("")
        
        # Get overdue tasks
        yesterday = today - datetime.timedelta(days=1)
        overdue_tasks = client.get_tasks(due_date_lt=yesterday)
        overdue_count = len([t for t in overdue_tasks.get('tasks', [])
                           if t.get('status', {}).get('status') != 'complete'])
        
        if overdue_count > 0:
            briefing.append(f"⚠️ **{overdue_count} Overdue Tasks**")
            briefing.append("")
        
        # Get yesterday's time tracking
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=59)
        
        time_entries = client.get_time_entries(
            start_date=yesterday_start,
            end_date=yesterday_end
        )
        
        total_time = sum(int(entry.get('duration', 0)) for entry in time_entries.get('data', []))
        hours = total_time // 3600000  # Convert milliseconds to hours
        minutes = (total_time % 3600000) // 60000
        
        briefing.append(f"⏱️ **Yesterday's Time:** {hours}h {minutes}m")
        briefing.append("")
        
        # Week summary
        week_start = today - datetime.timedelta(days=today.weekday())
        week_time = client.get_time_entries(start_date=week_start)
        week_total = sum(int(entry.get('duration', 0)) for entry in week_time.get('data', []))
        week_hours = week_total // 3600000
        
        briefing.append(f"📊 **This Week:** {week_hours}h total")
        
        return "\n".join(briefing)
        
    except Exception as e:
        current_app.logger.error(f"ClickUp briefing failed: {e}")
        return f"**ClickUp Morning Briefing Error:**\n{str(e)}\n\nTry visiting `/diagnostics/clickup` for help."

# Copy the rest of your existing helper functions here with similar error handling improvements...
