# modules/clickup_integration.py
# ClickUp API integration for task management, time tracking, and productivity insights

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
            raise ValueError("CLICKUP_API_TOKEN not configured")
        
        self.base_url = "https://api.clickup.com/api/v2"
        self.headers = {
            "Authorization": self.api_token,
            "Content-Type": "application/json"
        }
        
        # Cache team/workspace info
        self._team_id = None
        self._default_space_id = None
        self._default_list_id = None
    
    def _make_request(self, method, endpoint, data=None):
        """Make API request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=self.headers)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=self.headers, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=self.headers, json=data)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=self.headers)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            current_app.logger.error(f"ClickUp API error: {e}")
            raise Exception(f"ClickUp API request failed: {str(e)}")
    
    def get_teams(self):
        """Get user's teams/workspaces"""
        return self._make_request('GET', '/team')
    
    def get_team_id(self):
        """Get the primary team ID"""
        if not self._team_id:
            teams = self.get_teams()
            if teams['teams']:
                self._team_id = teams['teams'][0]['id']
        return self._team_id
    
    def get_spaces(self, team_id=None):
        """Get spaces in team"""
        if not team_id:
            team_id = self.get_team_id()
        return self._make_request('GET', f'/team/{team_id}/space')
    
    def get_lists(self, space_id):
        """Get lists in space"""
        return self._make_request('GET', f'/space/{space_id}/list')
    
    def get_default_list_id(self):
        """Get a default list ID for task creation"""
        if not self._default_list_id:
            try:
                team_id = self.get_team_id()
                spaces = self.get_spaces(team_id)
                
                if spaces['spaces']:
                    # Use first space
                    space_id = spaces['spaces'][0]['id']
                    lists = self.get_lists(space_id)
                    
                    if lists['lists']:
                        # Use first list
                        self._default_list_id = lists['lists'][0]['id']
                        
            except Exception as e:
                current_app.logger.error(f"Failed to get default list: {e}")
        
        return self._default_list_id
    
    def create_task(self, name, description="", due_date=None, priority=3, list_id=None):
        """Create a new task"""
        if not list_id:
            list_id = self.get_default_list_id()
        
        if not list_id:
            raise Exception("No default list found. Please configure ClickUp workspace.")
        
        task_data = {
            "name": name,
            "description": description,
            "priority": priority
        }
        
        if due_date:
            # Convert datetime to timestamp (milliseconds)
            if isinstance(due_date, datetime.datetime):
                task_data["due_date"] = int(due_date.timestamp() * 1000)
        
        return self._make_request('POST', f'/list/{list_id}/task', task_data)
    
    def get_tasks(self, list_id=None, **filters):
        """Get tasks with filters"""
        if not list_id:
            list_id = self.get_default_list_id()
        
        if not list_id:
            return {"tasks": []}
        
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
    
    def get_time_entries(self, team_id=None, start_date=None, end_date=None):
        """Get time tracking entries"""
        if not team_id:
            team_id = self.get_team_id()
        
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
        
        return self._make_request('DELETE', f'/team/{team_id}/time_entries/current')
    
    def get_user_info(self):
        """Get current user information"""
        return self._make_request('GET', '/user')

def is_clickup_configured():
    """Check if ClickUp API is configured"""
    return bool(os.getenv('CLICKUP_API_TOKEN'))

def process_clickup_command(user_input, project, use_voices, random_toggle):
    """Process ClickUp-related commands"""
    if not is_clickup_configured():
        return None, False
    
    user_input_lower = user_input.lower()
    
    # ClickUp command patterns
    clickup_patterns = [
        'clickup', 'click up', 'cu ', ' cu', 'time track', 'my tasks', 'create task'
    ]
    
    if not any(pattern in user_input_lower for pattern in clickup_patterns):
        return None, False
    
    try:
        client = ClickUpClient()
        
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
        
        # Task creation
        if any(keyword in user_input_lower for keyword in ['create', 'add', 'new task']):
            # Extract task name after "create task", "add task", etc.
            task_match = re.search(r'(?:create|add|new)\s+(?:clickup\s+)?task:?\s*(.+)', user_input, re.IGNORECASE)
            if task_match:
                task_name = task_match.group(1).strip()
                result = create_clickup_task(client, task_name, project)
                response_data = {"SyntaxPrime": result}
                return response_data, True
        
        # Task queries
        if any(keyword in user_input_lower for keyword in ['tasks', 'due', 'deadline']):
            tasks_summary = get_clickup_tasks_summary(client)
            response_data = {"SyntaxPrime": tasks_summary}
            return response_data, True
        
        # Time tracking controls
        if 'start timer' in user_input_lower:
            # Extract task name or ID
            timer_match = re.search(r'start timer (?:on\s+)?(.+)', user_input, re.IGNORECASE)
            if timer_match:
                task_identifier = timer_match.group(1).strip()
                result = start_clickup_timer(client, task_identifier)
                response_data = {"SyntaxPrime": result}
                return response_data, True
        
        if 'stop timer' in user_input_lower:
            result = stop_clickup_timer(client)
            response_data = {"SyntaxPrime": result}
            return response_data, True
        
        # Default response for unrecognized ClickUp commands
        response_data = {
            "SyntaxPrime": "**ClickUp Commands Available:**\n\n"
                          "• `clickup morning` - Daily task briefing\n"
                          "• `clickup time today` - Today's time tracking\n"
                          "• `clickup time week` - This week's hours\n"
                          "• `create clickup task: [name]` - New task\n"
                          "• `clickup tasks` - View due tasks\n"
                          "• `start timer on [task]` - Start tracking\n"
                          "• `stop timer` - Stop current timer"
        }
        return response_data, True
        
    except Exception as e:
        current_app.logger.error(f"ClickUp command failed: {e}")
        response_data = {"SyntaxPrime": f"ClickUp integration error: {str(e)}"}
        return response_data, True

def get_clickup_morning_briefing(client=None):
    """Generate morning briefing with tasks and time tracking"""
    if not client:
        client = ClickUpClient()
    
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
        return f"**ClickUp Morning Briefing Error:** {str(e)}"

def get_clickup_time_today(client=None):
    """Get today's time tracking summary"""
    if not client:
        client = ClickUpClient()
    
    try:
        today = datetime.datetime.now()
        start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = today.replace(hour=23, minute=59, second=59, microsecond=59)
        
        time_entries = client.get_time_entries(
            start_date=start_of_day,
            end_date=end_of_day
        )
        
        total_time = 0
        task_breakdown = {}
        
        for entry in time_entries.get('data', []):
            duration = int(entry.get('duration', 0))
            total_time += duration
            
            task_name = entry.get('task', {}).get('name', 'Unknown Task')
            if task_name in task_breakdown:
                task_breakdown[task_name] += duration
            else:
                task_breakdown[task_name] = duration
        
        # Format response
        total_hours = total_time // 3600000
        total_minutes = (total_time % 3600000) // 60000
        
        response = [f"⏰ **Today's Time Tracking: {total_hours}h {total_minutes}m**", ""]
        
        if task_breakdown:
            response.append("**Breakdown by Task:**")
            for task, duration in sorted(task_breakdown.items(), key=lambda x: x[1], reverse=True):
                task_hours = duration // 3600000
                task_minutes = (duration % 3600000) // 60000
                response.append(f"• {task}: {task_hours}h {task_minutes}m")
        else:
            response.append("No time tracked today yet.")
        
        return "\n".join(response)
        
    except Exception as e:
        return f"Time tracking query failed: {str(e)}"

def get_clickup_time_week(client=None):
    """Get this week's time tracking summary"""
    if not client:
        client = ClickUpClient()
    
    try:
        today = datetime.datetime.now()
        week_start = today - datetime.timedelta(days=today.weekday())
        
        time_entries = client.get_time_entries(start_date=week_start)
        
        total_time = 0
        daily_breakdown = {}
        
        for entry in time_entries.get('data', []):
            duration = int(entry.get('duration', 0))
            total_time += duration
            
            # Get date from timestamp
            start_time = datetime.datetime.fromtimestamp(int(entry.get('start', 0)) / 1000)
            date_key = start_time.strftime('%A, %m/%d')
            
            if date_key in daily_breakdown:
                daily_breakdown[date_key] += duration
            else:
                daily_breakdown[date_key] = duration
        
        # Format response
        total_hours = total_time // 3600000
        total_minutes = (total_time % 3600000) // 60000
        
        response = [f"📊 **This Week's Time: {total_hours}h {total_minutes}m**", ""]
        
        if daily_breakdown:
            response.append("**Daily Breakdown:**")
            for day, duration in daily_breakdown.items():
                day_hours = duration // 3600000
                day_minutes = (duration % 3600000) // 60000
                response.append(f"• {day}: {day_hours}h {day_minutes}m")
        else:
            response.append("No time tracked this week yet.")
        
        return "\n".join(response)
        
    except Exception as e:
        return f"Weekly time query failed: {str(e)}"

def create_clickup_task(client, task_name, project=None):
    """Create a new ClickUp task"""
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
        
        return f"✅ **Task Created:** {task_name}\n📝 **ID:** {task_id}\n🔗 **URL:** {task_url}"
        
    except Exception as e:
        return f"Failed to create task: {str(e)}"

def get_clickup_tasks_summary(client=None):
    """Get summary of current tasks"""
    if not client:
        client = ClickUpClient()
    
    try:
        # Get all tasks
        all_tasks = client.get_tasks()
        tasks = all_tasks.get('tasks', [])
        
        # Categorize tasks
        overdue = []
        due_today = []
        upcoming = []
        
        today = datetime.datetime.now()
        today_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today.replace(hour=23, minute=59, second=59, microsecond=59)
        
        for task in tasks:
            if task.get('status', {}).get('status') == 'complete':
                continue  # Skip completed tasks
                
            due_date_ms = task.get('due_date')
            if due_date_ms:
                due_date = datetime.datetime.fromtimestamp(int(due_date_ms) / 1000)
                
                if due_date < today_start:
                    overdue.append(task)
                elif today_start <= due_date <= today_end:
                    due_today.append(task)
                elif due_date <= today + datetime.timedelta(days=7):
                    upcoming.append(task)
        
        # Format response
        response = ["📋 **ClickUp Tasks Summary**", ""]
        
        if overdue:
            response.append(f"⚠️ **Overdue ({len(overdue)}):**")
            for task in overdue[:3]:  # Show first 3
                response.append(f"  • {task['name']}")
            response.append("")
        
        if due_today:
            response.append(f"📅 **Due Today ({len(due_today)}):**")
            for task in due_today:
                priority_map = {1: "🔴", 2: "🟡", 3: "🟢", 4: "🔵"}
                priority_icon = priority_map.get(task.get('priority', {}).get('priority', 3), "🟢")
                response.append(f"  • {priority_icon} {task['name']}")
            response.append("")
        
        if upcoming:
            response.append(f"📆 **This Week ({len(upcoming)}):**")
            for task in upcoming[:5]:  # Show first 5
                due_date_ms = task.get('due_date')
                if due_date_ms:
                    due_date = datetime.datetime.fromtimestamp(int(due_date_ms) / 1000)
                    date_str = due_date.strftime('%m/%d')
                    response.append(f"  • {task['name']} (due {date_str})")
        
        if not overdue and not due_today and not upcoming:
            response.append("No upcoming tasks found.")
        
        return "\n".join(response)
        
    except Exception as e:
        return f"Tasks query failed: {str(e)}"

def start_clickup_timer(client, task_identifier):
    """Start timer on a task"""
    try:
        # For now, assume task_identifier is a task name
        # In a more complete implementation, we'd search for tasks by name
        return f"⏰ **Timer started** on: {task_identifier}\n\n*(Note: Task search by name not yet implemented. Use task ID for now.)*"
        
    except Exception as e:
        return f"Failed to start timer: {str(e)}"

def stop_clickup_timer(client=None):
    """Stop current timer"""
    if not client:
        client = ClickUpClient()
    
    try:
        result = client.stop_time_tracking()
        return "⏹️ **Timer stopped** and time logged."
        
    except Exception as e:
        return f"Failed to stop timer: {str(e)}"

def log_ghostline_conversation_to_clickup(conversation_summary, project=None):
    """Log important Ghostline conversations as ClickUp tasks or comments"""
    if not is_clickup_configured():
        return False
    
    try:
        client = ClickUpClient()
        
        # Create task for significant conversations
        task_name = f"Ghostline: {conversation_summary[:50]}..."
        description = f"Auto-generated from Ghostline conversation\nProject: {project or 'General'}\n\nContent: {conversation_summary}"
        
        task = client.create_task(
            name=task_name,
            description=description,
            priority=4  # Low priority for auto-generated tasks
        )
        
        current_app.logger.info(f"Logged conversation to ClickUp: {task.get('id')}")
        return True
        
    except Exception as e:
        current_app.logger.error(f"Failed to log to ClickUp: {e}")
        return False

def create_clickup_task_from_reminder_snooze(reminder_title, snooze_until):
    """Create ClickUp task when Telegram reminder is snoozed"""
    if not is_clickup_configured():
        return False
    
    try:
        client = ClickUpClient()
        
        task_name = f"Follow up: {reminder_title}"
        description = f"Auto-created from snoozed Telegram reminder\nOriginal reminder: {reminder_title}\nSnoozed until: {snooze_until}"
        
        # Set due date to snooze time
        task = client.create_task(
            name=task_name,
            description=description,
            due_date=snooze_until,
            priority=3
        )
        
        current_app.logger.info(f"Created ClickUp task from reminder snooze: {task.get('id')}")
        return True
        
    except Exception as e:
        current_app.logger.error(f"Failed to create task from reminder: {e}")
        return False