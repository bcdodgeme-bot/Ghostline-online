# modules/slack_integration.py - Simple AMCF-focused Slack integration

import os
import json
import re
import datetime
import logging
from typing import Dict, Any, Optional, Tuple

# Slack SDK imports
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_SDK_AVAILABLE = True
except ImportError:
    SLACK_SDK_AVAILABLE = False

# Flask imports (with graceful handling)
try:
    from flask import current_app
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Import your existing ClickUp integration
from modules.clickup_integration import ClickUpClient, is_clickup_configured
from modules.database import save_conversation_enhanced

# SIMPLE CONFIGURATION - Everything goes to AMCF
AMCF_LIST_ID = os.getenv('SLACK_AMCF_LIST_ID', '901306635069')
DEFAULT_ASSIGNEE_ID = int(os.getenv('CLICKUP_DEFAULT_ASSIGNEE_ID', '120284829'))

# DUE DATE RULES
DUE_DATE_RULES = {
    'urgent': {
        'patterns': ['urgent', 'asap', 'immediately', 'now', 'critical'],
        'due_hours': 4,
        'priority': 1
    },
    'lisa': {
        'patterns': ['lisa'],
        'due_days': int(os.getenv('SLACK_LISA_DUE_DAYS', '2')),
        'priority': 2
    },
    'design': {
        'patterns': ['design', 'logo', 'branding', 'graphics', 'hyperlink'],
        'due_days': 10,
        'priority': 3
    },
    'review': {
        'patterns': ['review', 'check', 'look at', 'verify', 'move', 'update'],
        'due_days': 3,
        'priority': 3
    },
    'project': {
        'patterns': ['project', 'campaign', 'initiative'],
        'due_days': 14,
        'priority': 3
    },
    'default': {
        'due_days': int(os.getenv('SLACK_DEFAULT_DUE_DAYS', '7')),
        'priority': 3
    }
}


class SlackMentionHandler:
    """Simple Slack mention handler for AMCF tasks"""
    
    def __init__(self):
        # Slack configuration
        self.bot_token = os.getenv('SLACK_BOT_TOKEN')
        self.app_token = os.getenv('SLACK_APP_TOKEN')
        self.signing_secret = os.getenv('SLACK_SIGNING_SECRET')
        self.user_id = os.getenv('SLACK_USER_ID')
        
        # Initialize Slack client
        self.slack_client = None
        if SLACK_SDK_AVAILABLE and self.bot_token:
            self.slack_client = WebClient(token=self.bot_token)
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # ClickUp integration
        self.clickup_client = None
        if is_clickup_configured():
            try:
                self.clickup_client = ClickUpClient()
            except Exception as e:
                self.logger.error(f"ClickUp client initialization failed: {e}")
    
    def _setup_logger(self):
        """Setup logger that works with or without Flask context"""
        try:
            if FLASK_AVAILABLE:
                return current_app.logger
        except (RuntimeError, AttributeError):
            pass
        
        logger = logging.getLogger('slack_integration')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - Slack - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def is_configured(self) -> bool:
        """Check if Slack integration is properly configured"""
        return bool(
            self.bot_token and
            self.signing_secret and
            self.user_id and
            (SLACK_SDK_AVAILABLE or self.bot_token)
        )
    
    def extract_mention_from_message(self, message_text: str) -> Optional[str]:
        """Extract mention of the configured user from message text"""
        if not self.user_id:
            return None
        
        mention_pattern = f"<@{self.user_id}>"
        
        if mention_pattern in message_text:
            cleaned_text = message_text.replace(mention_pattern, "").strip()
            cleaned_text = re.sub(r'<@[A-Z0-9]+>', '', cleaned_text)  # Remove other mentions
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            return cleaned_text
        
        return None
    
    def determine_due_date_and_priority(self, message_text: str, sender_name: str = '') -> Tuple[datetime.datetime, int]:
        """Determine due date and priority for AMCF tasks"""
        
        now = datetime.datetime.now()
        message_lower = message_text.lower()
        sender_lower = sender_name.lower()
        
        # Check for explicit dates first
        explicit_due = self._parse_explicit_due_date(message_text)
        if explicit_due:
            return explicit_due, 2
        
        # Check if it's from Lisa (highest priority)
        if 'lisa' in sender_lower or 'lisa' in message_lower:
            due_date = now + datetime.timedelta(days=DUE_DATE_RULES['lisa']['due_days'])
            due_date = due_date.replace(hour=17, minute=0, second=0, microsecond=0)
            return due_date, DUE_DATE_RULES['lisa']['priority']
        
        # Check other patterns
        for rule_name, rule in DUE_DATE_RULES.items():
            if rule_name in ['lisa', 'default']:
                continue
                
            for pattern in rule['patterns']:
                if pattern in message_lower:
                    if 'due_hours' in rule:
                        due_date = now + datetime.timedelta(hours=rule['due_hours'])
                    else:
                        due_date = now + datetime.timedelta(days=rule['due_days'])
                    
                    due_date = due_date.replace(hour=17, minute=0, second=0, microsecond=0)
                    return due_date, rule['priority']
        
        # Default
        default_due = now + datetime.timedelta(days=DUE_DATE_RULES['default']['due_days'])
        default_due = default_due.replace(hour=17, minute=0, second=0, microsecond=0)
        return default_due, DUE_DATE_RULES['default']['priority']
    
    def _parse_explicit_due_date(self, message_text: str) -> Optional[datetime.datetime]:
        """Parse explicit month/date references"""
        
        now = datetime.datetime.now()
        message_lower = message_text.lower()
        
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        
        for month_name, month_num in months.items():
            if month_name in message_lower:
                year = now.year
                if month_num < now.month or (month_num == now.month and now.day > 15):
                    year += 1
                
                last_day = 31
                if month_num in [4, 6, 9, 11]:
                    last_day = 30
                elif month_num == 2:
                    last_day = 29 if year % 4 == 0 else 28
                
                return datetime.datetime(year, month_num, last_day, 17, 0, 0)
        
        return None
    
    def clean_task_name(self, message_text: str) -> str:
        """Clean up message for task name"""
        
        cleaned = message_text
        
        # Remove common prefixes
        prefixes = [
            r'^(hey|hi|hello),?\s*',
            r'^(can you|could you|please)\s*',
            r'^(question\s*-?\s*)',
            r'^(thank you for\s+.*?\.?\s*)',
            r'^[a-zA-Z]+\s+',  # Remove person names at start
        ]
        
        for prefix in prefixes:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Remove trailing punctuation
        cleaned = re.sub(r'[?!.]*$', '', cleaned).strip()
        
        # Ensure minimum length
        if len(cleaned) < 5:
            cleaned = f"AMCF Task: {message_text[:50]}"
        
        return cleaned
    
    def parse_task_from_mention(self, mention_text: str, sender_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse task information from mention text"""
        
        sender_name = sender_info.get('display_name', 'Unknown')
        
        # Get due date and priority
        due_date, priority = self.determine_due_date_and_priority(mention_text, sender_name)
        
        # Clean task name
        task_name = self.clean_task_name(mention_text)
        
        # Create description
        description = f"""AMCF Task from Slack

Original message: {mention_text}
Requested by: {sender_name}
Auto-assigned due date: {due_date.strftime('%A, %B %d at %I:%M %p')}
Priority: {['', 'High', 'Medium', 'Normal', 'Low'][priority]}

Created automatically via Slack integration."""
        
        return {
            'raw_text': mention_text,
            'sender': sender_info,
            'task_name': task_name,
            'due_date': due_date,
            'priority': priority,
            'description': description,
            'list_id': AMCF_LIST_ID
        }
    
    def create_task_from_mention(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create AMCF ClickUp task from mention"""
        
        if not self.clickup_client:
            return {
                'success': False,
                'error': 'ClickUp not configured',
                'message': 'ClickUp integration not available'
            }
        
        try:
            self.logger.info(f"Creating AMCF task: {task_info['task_name']}")
            
            # Create the task with assignee
            task_result = self.clickup_client.create_task(
                name=task_info['task_name'],
                description=task_info['description'],
                due_date=task_info.get('due_date'),
                priority=task_info.get('priority', 3),
                list_id=task_info['list_id']
            )
            
            # Add assignee if task was created successfully
            if task_result and task_result.get('id'):
                try:
                    # Use ClickUp API to assign task
                    import requests
                    task_id = task_result.get('id')
                    url = f"https://api.clickup.com/api/v2/task/{task_id}"
                    
                    headers = {
                        "Authorization": self.clickup_client.api_token,
                        "Content-Type": "application/json"
                    }
                    
                    update_data = {
                        "assignees": {
                            "add": [DEFAULT_ASSIGNEE_ID]
                        }
                    }
                    
                    response = requests.put(url, headers=headers, json=update_data)
                    if response.status_code == 200:
                        self.logger.info(f"Task assigned to user {DEFAULT_ASSIGNEE_ID}")
                    else:
                        self.logger.warning(f"Assignment failed: {response.status_code}")
                        
                except Exception as e:
                    self.logger.warning(f"Task assignment failed: {e}")
            
            # Format enhanced response message
            due_text = ""
            if task_info.get('due_date'):
                due_date = task_info['due_date']
                due_text = f" (Due: {due_date.strftime('%A, %b %d at %I:%M %p')})"
            
            priority_icons = {1: "🔴 High", 2: "🟡 Medium", 3: "🟢 Normal", 4: "🔵 Low"}
            priority_text = priority_icons.get(task_info['priority'], "🟢 Normal")
            
            # Determine the trigger keyword for auto-routing message
            trigger_reason = "Default routing"
            message_lower = task_info['raw_text'].lower()
            
            if 'lisa' in message_lower:
                trigger_reason = "Lisa request (high priority)"
            elif any(word in message_lower for word in ['urgent', 'asap', 'immediately']):
                trigger_reason = "Urgent keyword"
            elif any(word in message_lower for word in ['design', 'logo', 'branding']):
                trigger_reason = "Design work"
            elif any(word in message_lower for word in ['review', 'check', 'verify']):
                trigger_reason = "Review task"
            
            success_message = f"""✅ Task Created
📋 {task_info['task_name']}{due_text}
📁 Space: AMCF
⚡ Priority: {priority_text}
🤖 Auto-routed: {trigger_reason}"""
            
            self.logger.info(f"AMCF task created successfully: {task_result.get('id')}")
            
            return {
                'success': True,
                'task_id': task_result.get('id'),
                'task_url': task_result.get('url'),
                'message': success_message,
                'task_info': task_info
            }
            
        except Exception as e:
            self.logger.error(f"AMCF task creation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"❌ Failed to create AMCF task: {str(e)}"
            }
    
    def send_slack_response(self, channel: str, thread_ts: str, message: str) -> bool:
        """Send response back to Slack"""
        
        if not self.slack_client:
            self.logger.error("Slack client not available")
            return False
        
        try:
            response = self.slack_client.chat_postMessage(
                channel=channel,
                text=message,
                thread_ts=thread_ts
            )
            
            success = response.get('ok', False)
            if success:
                self.logger.info("Slack response sent successfully")
            else:
                self.logger.error(f"Slack response failed: {response}")
            
            return success
            
        except SlackApiError as e:
            self.logger.error(f"Slack API error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send Slack response: {e}")
            return False
    
    def process_slack_mention(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main function to process Slack mention events"""
        
        try:
            event = event_data.get('event', {})
            message_text = event.get('text', '')
            channel = event.get('channel', '')
            thread_ts = event.get('ts', '')
            user_id = event.get('user', '')
            
            self.logger.info(f"Processing AMCF mention: {message_text}")
            
            # Extract mention
            mention_text = self.extract_mention_from_message(message_text)
            if not mention_text:
                return {'success': False, 'error': 'No valid mention found'}
            
            # Get sender info
            sender_info = self._get_user_info(user_id)
            
            # Parse task information
            task_info = self.parse_task_from_mention(mention_text, sender_info)
            
            # Create AMCF task
            task_result = self.create_task_from_mention(task_info)
            
            # Send response to Slack
            response_sent = self.send_slack_response(
                channel=channel,
                thread_ts=thread_ts,
                message=task_result['message']
            )
            
            # Log to database
            try:
                save_conversation_enhanced(
                    'slack_amcf_mentions',
                    f"@mention: {mention_text}",
                    {
                        'task_created': task_result['success'],
                        'task_name': task_info['task_name'],
                        'sender': sender_info.get('display_name', 'Unknown'),
                        'channel': channel,
                        'response_sent': response_sent,
                        'space': 'AMCF',
                        'priority': task_info['priority'],
                        'due_date': task_info['due_date'].isoformat() if task_info['due_date'] else None
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to save to database: {e}")
            
            return {
                'success': True,
                'task_created': task_result['success'],
                'response_sent': response_sent,
                'task_info': task_info,
                'message': task_result['message']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process Slack mention: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user information from Slack API"""
        
        if not self.slack_client:
            return {'user_id': user_id, 'display_name': 'Unknown User'}
        
        try:
            response = self.slack_client.users_info(user=user_id)
            
            if response.get('ok'):
                user = response.get('user', {})
                profile = user.get('profile', {})
                
                return {
                    'user_id': user_id,
                    'display_name': profile.get('display_name') or profile.get('real_name') or user.get('name', 'Unknown'),
                    'email': profile.get('email'),
                    'real_name': profile.get('real_name')
                }
        
        except Exception as e:
            self.logger.warning(f"Failed to get user info for {user_id}: {e}")
        
        return {'user_id': user_id, 'display_name': 'Unknown User'}


def is_slack_configured() -> bool:
    """Check if Slack integration is configured"""
    required_vars = ['SLACK_BOT_TOKEN', 'SLACK_SIGNING_SECRET', 'SLACK_USER_ID']
    return all(os.getenv(var) for var in required_vars)


def process_slack_webhook_event(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process incoming Slack webhook events"""
    
    if not is_slack_configured():
        return {
            'success': False,
            'error': 'Slack not configured'
        }
    
    handler = SlackMentionHandler()
    
    event_type = event_data.get('type')
    
    if event_type == 'url_verification':
        return {
            'success': True,
            'challenge': event_data.get('challenge')
        }
    
    elif event_type == 'event_callback':
        event = event_data.get('event', {})
        
        if event.get('type') == 'message' and not event.get('subtype'):
            user_id = handler.user_id
            message_text = event.get('text', '')
            
            # Skip bot messages
            if event.get('user') == event.get('bot_id'):
                return {'success': True, 'message': 'Bot message ignored'}
            
            # Check for mention
            if f"<@{user_id}>" in message_text:
                return handler.process_slack_mention(event_data)
            else:
                return {'success': True, 'message': 'No mention found'}
    
    return {'success': True, 'message': 'Event ignored'}
