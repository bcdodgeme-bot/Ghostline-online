# modules/clickup_integration.py - FIXED VERSION with proper pattern matching and Flask context handling

import os
import json
import datetime
import requests
import re
import logging

# FIXED: Import Flask properly but handle context gracefully
try:
    from flask import current_app
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

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
        
        # FIXED: Setup logging that works with or without Flask context
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger that works with or without Flask context"""
        try:
            if FLASK_AVAILABLE:
                # Try to use Flask's current_app logger if in context
                return current_app.logger
        except RuntimeError:
            # Not in Flask context, use standard logging
            pass
        except Exception:
            # Flask not available or other error
            pass
        
        # Fallback to standard Python logging
        logger = logging.getLogger('clickup_integration')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - ClickUp - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _log_info(self, message):
        """Safe logging that works in any context"""
        try:
            self.logger.info(message)
        except Exception:
            # Fallback to print if logging fails
            print(f"ClickUp: {message}")
    
    def _log_error(self, message):
        """Safe error logging that works in any context"""
        try:
            self.logger.error(message)
        except Exception:
            # Fallback to print if logging fails
            print(f"ClickUp ERROR: {message}")
    
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
            self._log_error(f"ClickUp API error: {e}")
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
            self._log_error(f"Failed to get team ID: {e}")
            
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
        # First try environment variable (recommended configuration from diagnostics)
        if self._default_list_id:
            self._log_info(f"Using configured list ID: {self._default_list_id}")
            return self._default_list_id
            
        # Then try cache
        if self._cached_list_id:
            return self._cached_list_id
        
        # FIXED: Better error message suggesting the specific configuration
        raise Exception("""ClickUp workspace configuration needed.

From your diagnostics, add these environment variables to Railway:

CLICKUP_DEFAULT_LIST_ID=901306635049
CLICKUP_DEFAULT_TEAM_ID=9013453647

This will use your "Personal Time Management → List" for Ghostline tasks.

Visit /diagnostics/clickup for setup wizard.""")
    
    def create_task(self, name, description="", due_date=None, priority=3, list_id=None):
        """Create a new task with better error handling and proper context"""
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
            
            self._log_info(f"Creating ClickUp task: {name} in list {list_id}")
            result = self._make_request('POST', f'/list/{list_id}/task', task_data)
            
            # FIXED: Safe logging that doesn't require Flask context
            task_id = result.get('id')
            self._log_info(f"Created ClickUp task: {name} (ID: {task_id})")
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            self._log_error(f"Task creation failed: {error_msg}")
            
            if "configuration needed" in error_msg:
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
            self._log_error(f"Failed to get tasks: {e}")
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
            self._log_error(f"Failed to get time entries: {e}")
            return {"data": []}

def is_clickup_configured():
    """Check if ClickUp API is configured"""
    return bool(os.getenv('CLICKUP_API_TOKEN'))

def process_clickup_command(user_input, project, use_voices, random_toggle):
    """Process ClickUp-related commands with FIXED pattern matching and context handling"""
    if not is_clickup_configured():
        return None, False
    
    user_input_lower = user_input.lower()
    
    # PATCH: Tighten ClickUp pattern matching to reduce false positives
    # FIXED: More specific regex patterns to prevent false positives
    clickup_patterns = [
        r'\bclickup\b',                    # "clickup" as whole word
        r'\bclick\s+up\b',                # "click up" with space
        r'\bcreate\s+clickup\s+task\b',   # "create clickup task"
        r'\bmy\s+clickup\s+tasks\b',      # "my clickup tasks"
        r'\bclickup\s+time\b',            # "clickup time"
        r'\bclickup\s+morning\b',         # "clickup morning"
        r'\bclickup\s+setup\b',           # "clickup setup"
        r'\bwork\s+timer\b',              # "work timer"
        r'\bmy\s+task\s+list\b',          # FIXED: More specific - "my task list"
        r'\bmy\s+task\s+status\b',        # FIXED: More specific - "my task status"
        r'\bclickup\s+tasks?\b',          # ADDED: "clickup tasks" or "clickup task"
        r'\bshow\s+clickup\s+tasks?\b',   # ADDED: "show clickup tasks"
        r'\blist\s+clickup\s+tasks?\b'    # ADDED: "list clickup tasks"
    ]
    
    # FIXED: Use regex search instead of simple string containment
    if not any(re.search(pattern, user_input_lower) for pattern in clickup_patterns):
        return None, False
    
    try:
        client = ClickUpClient()
        
        # Setup/diagnostic command
        if any(re.search(pattern, user_input_lower) for pattern in [r'\bsetup\b', r'\bconfigure\b', r'\bdiagnostic\b']):
            response_data = {
                "SyntaxPrime": "**ClickUp Setup Status**\n\n"
                              "From your recent diagnostics, you need to set these environment variables:\n\n"
                              "```\n"
                              "CLICKUP_DEFAULT_LIST_ID=901306635049\n"
                              "CLICKUP_DEFAULT_TEAM_ID=9013453647\n"
                              "```\n\n"
                              "This configures: Rose and Angel Consulting → Personal Time Management → List\n\n"
                              "Visit `/diagnostics/clickup` for complete setup wizard."
            }
            return response_data, True
        
        # Morning briefing command
        if re.search(r'\bmorning\b', user_input_lower):
            briefing = get_clickup_morning_briefing(client)
            response_data = {"SyntaxPrime": briefing}
            return response_data, True
        
        # Time tracking commands
        if re.search(r'\btime\b', user_input_lower):
            if re.search(r'\btoday\b', user_input_lower):
                time_summary = get_clickup_time_summary(client, 'today')
            elif re.search(r'\bweek\b', user_input_lower):
                time_summary = get_clickup_time_summary(client, 'week')
            else:
                time_summary = get_clickup_time_summary(client, 'today')
            
            response_data = {"SyntaxPrime": time_summary}
            return response_data, True
        
        # Task listing commands
        if re.search(r'\btasks?\b', user_input_lower) and not re.search(r'\bcreate\b|\badd\b|\bnew\b', user_input_lower):
            tasks_summary = get_clickup_tasks_summary(client)
            response_data = {"SyntaxPrime": tasks_summary}
            return response_data, True
        
        # Task creation with better parsing
        if any(re.search(pattern, user_input_lower) for pattern in [r'\bcreate\b', r'\badd\b', r'\bnew\s+task\b']):
            # FIXED: Better task name extraction patterns
            task_patterns = [
                r'(?:create|add|new)\s+(?:clickup\s+)?task:?\s*(.+)',
                r'(?:create|add|new)\s+(.+?)\s+(?:task|to-do|item)',
                r'task:?\s*(.+)',
                r'clickup:?\s*(.+)'
            ]
            
            task_name = None
            for pattern in task_patterns:
                task_match = re.search(pattern, user_input, re.IGNORECASE)
                if task_match:
                    task_name = task_match.group(1).strip()
                    # Clean up common suffixes
                    task_name = re.sub(r'\s+(task|to-do|item)$', '', task_name, flags=re.IGNORECASE)
                    break
            
            if task_name and len(task_name) > 2:  # Ensure meaningful task name
                result = create_clickup_task(client, task_name, project)
                response_data = {"SyntaxPrime": result}
                return response_data, True
            else:
                response_data = {"SyntaxPrime": "Please specify a task name. Example: `create clickup task: Review quarterly report`"}
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
                          "• `clickup time week` - Weekly summary\n\n"
                          "⚙️ **Setup:**\n"
                          "• `clickup setup` - Configuration help\n"
                          "• Visit `/diagnostics/clickup` for full setup\n\n"
                          "**Status:** API connected, workspace detected, but needs environment variables set"
        }
        return response_data, True
        
    except Exception as e:
        # FIXED: Use print instead of current_app.logger to avoid context errors
        print(f"ClickUp command failed: {e}")
        
        error_message = str(e)
        if "configuration needed" in error_message:
            # Configuration-related errors - show the specific fix needed
            response_data = {
                "SyntaxPrime": f"**ClickUp Configuration Needed:**\n\n{error_message}"
            }
        else:
            # Other API errors
            response_data = {"SyntaxPrime": f"ClickUp integration error: {error_message}"}
        
        return response_data, True

def create_clickup_task(client, task_name, project=None):
    """Create a new ClickUp task with FIXED context handling"""
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
        if "configuration needed" in error_msg:
            return error_msg  # Pass through configuration errors with the specific fix
        else:
            return f"❌ **Task creation failed:** {error_msg}\n\nTry visiting `/diagnostics/clickup` for help."

def get_clickup_morning_briefing(client=None):
    """Generate morning briefing with FIXED context handling"""
    if not client:
        try:
            client = ClickUpClient()
        except Exception as e:
            return f"**ClickUp Morning Briefing Unavailable**\n\nConfiguration error: {str(e)}"
    
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
                status = "✅" if task.get('status', {}).get('status') == 'complete' else "🔲"
                priority_map = {1: "🔴", 2: "🟡", 3: "🟢", 4: "🔵"}
                priority_icon = priority_map.get(task.get('priority', {}).get('priority', 3), "🟢")
                briefing.append(f"  {status} {priority_icon} {task['name']}")
        else:
            briefing.append("  No tasks due today")
        
        return "\n".join(briefing)
        
    except Exception as e:
        # FIXED: Use print instead of current_app.logger
        print(f"ClickUp briefing failed: {e}")
        return f"**ClickUp Morning Briefing Error:**\n{str(e)}"

def get_clickup_tasks_summary(client=None):
    """Get summary of current tasks"""
    if not client:
        try:
            client = ClickUpClient()
        except Exception as e:
            return f"**ClickUp Tasks Unavailable**\n\nConfiguration error: {str(e)}"
    
    try:
        # Get upcoming tasks (next 7 days)
        today = datetime.datetime.now()
        week_end = today + datetime.timedelta(days=7)
        
        upcoming_tasks = client.get_tasks(
            due_date_gt=today,
            due_date_lt=week_end
        )
        
        summary = ["📋 **CLICKUP TASKS SUMMARY**", ""]
        
        if upcoming_tasks.get('tasks'):
            summary.append(f"**📅 Upcoming Tasks ({len(upcoming_tasks['tasks'])}):**")
            for task in upcoming_tasks['tasks'][:10]:  # Limit to 10 tasks
                due_date = ""
                if task.get('due_date'):
                    due_timestamp = int(task['due_date']) / 1000
                    due_dt = datetime.datetime.fromtimestamp(due_timestamp)
                    due_date = f" (Due: {due_dt.strftime('%m/%d')})"
                
                status = "✅" if task.get('status', {}).get('status') == 'complete' else "🔲"
                priority_map = {1: "🔴", 2: "🟡", 3: "🟢", 4: "🔵"}
                priority_icon = priority_map.get(task.get('priority', {}).get('priority', 3), "🟢")
                
                summary.append(f"  {status} {priority_icon} {task['name']}{due_date}")
        else:
            summary.append("No upcoming tasks found")
        
        return "\n".join(summary)
        
    except Exception as e:
        print(f"ClickUp tasks summary failed: {e}")
        return f"**ClickUp Tasks Summary Error:**\n{str(e)}"

def get_clickup_time_summary(client=None, period='today'):
    """Get time tracking summary"""
    if not client:
        try:
            client = ClickUpClient()
        except Exception as e:
            return f"**ClickUp Time Summary Unavailable**\n\nConfiguration error: {str(e)}"
    
    try:
        today = datetime.datetime.now()
        
        if period == 'today':
            start_date = today.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = today.replace(hour=23, minute=59, second=59, microsecond=59)
            title = "TODAY'S TIME"
        elif period == 'week':
            # Start of week (Monday)
            start_date = today - datetime.timedelta(days=today.weekday())
            start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = today.replace(hour=23, minute=59, second=59, microsecond=59)
            title = "THIS WEEK'S TIME"
        
        time_entries = client.get_time_entries(
            start_date=start_date,
            end_date=end_date
        )
        
        summary = [f"⏱️ **CLICKUP {title} SUMMARY**", ""]
        
        if time_entries.get('data'):
            total_duration = sum(int(entry.get('duration', 0)) for entry in time_entries['data'])
            hours = total_duration // 3600000  # Convert from milliseconds to hours
            minutes = (total_duration % 3600000) // 60000
            
            summary.append(f"**⏰ Total Time:** {hours}h {minutes}m")
            summary.append(f"**📊 Entries:** {len(time_entries['data'])}")
            
            # Group by task
            task_times = {}
            for entry in time_entries['data']:
                task_name = entry.get('task', {}).get('name', 'Unknown Task')
                duration = int(entry.get('duration', 0))
                task_times[task_name] = task_times.get(task_name, 0) + duration
            
            if task_times:
                summary.append("\n**📋 Time by Task:**")
                for task_name, duration in sorted(task_times.items(), key=lambda x: x[1], reverse=True)[:5]:
                    task_hours = duration // 3600000
                    task_minutes = (duration % 3600000) // 60000
                    summary.append(f"  • {task_name}: {task_hours}h {task_minutes}m")
        else:
            summary.append(f"No time entries found for {period}")
        
        return "\n".join(summary)
        
    except Exception as e:
        print(f"ClickUp time summary failed: {e}")
        return f"**ClickUp Time Summary Error:**\n{str(e)}"
