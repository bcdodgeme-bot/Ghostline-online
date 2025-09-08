# modules/slack_integration.py - Slack mention detection and ClickUp task creation

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


class SlackMentionHandler:
    """Handles Slack mention detection and ClickUp task creation"""
    
    def __init__(self):
        # Slack configuration
        self.bot_token = os.getenv('SLACK_BOT_TOKEN')  # Your xoxb- token
        self.app_token = os.getenv('SLACK_APP_TOKEN')  # Your xapp- token
        self.signing_secret = os.getenv('SLACK_SIGNING_SECRET')
        self.user_id = os.getenv('SLACK_USER_ID')  # Your user ID for mention detection
        
        # Initialize Slack client if SDK available
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
        
        # Fallback to standard Python logging
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
        
        # Slack mentions come as <@U1234567890>
        mention_pattern = f"<@{self.user_id}>"
        
        if mention_pattern in message_text:
            # Remove the mention and return cleaned text
            cleaned_text = message_text.replace(mention_pattern, "").strip()
            # Remove extra whitespace
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            return cleaned_text
        
        return None
    
    def parse_task_from_mention(self, mention_text: str, sender_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse task information from mention text using AI-like processing"""
        
        # Extract key information patterns
        task_info = {
            'raw_text': mention_text,
            'sender': sender_info,
            'task_name': '',
            'due_date': None,
            'priority': 3,  # Default priority
            'description': ''
        }
        
        # Clean up common prefixes
        cleaned_text = mention_text
        prefixes_to_remove = [
            r'^(hey|hi|hello),?\s*',
            r'^(can you|could you|please)\s*',
            r'^(would you|will you)\s*'
        ]
        
        for prefix in prefixes_to_remove:
            cleaned_text = re.sub(prefix, '', cleaned_text, flags=re.IGNORECASE).strip()
        
        # Extract due date patterns
        due_date_patterns = [
            r'by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'by\s+(tomorrow|today)',
            r'by\s+(\d{1,2}/\d{1,2})',
            r'by\s+(end of week|eow)',
            r'due\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'deadline\s+(\w+)',
            r'before\s+(\w+)'
        ]
        
        due_date_match = None
        for pattern in due_date_patterns:
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                due_date_match = match.group(1).lower()
                # Remove the due date from the task text
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE).strip()
                break
        
        # Convert due date to datetime
        if due_date_match:
            task_info['due_date'] = self._parse_due_date(due_date_match)
        
        # Extract priority indicators
        priority_patterns = [
            (r'\b(urgent|asap|immediately|now)\b', 1),
            (r'\b(high priority|important|critical)\b', 2),
            (r'\b(when you can|no rush|low priority)\b', 4)
        ]
        
        for pattern, priority in priority_patterns:
            if re.search(pattern, cleaned_text, re.IGNORECASE):
                task_info['priority'] = priority
                # Remove priority indicators from task text
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE).strip()
                break
        
        # Clean up the remaining text for task name
        # Remove question marks and trailing punctuation
        cleaned_text = re.sub(r'[?!.]*$', '', cleaned_text).strip()
        
        # Remove common task prefixes that aren't part of the actual task
        task_prefixes_to_remove = [
            r'^(review|check|look at|handle|take care of|deal with)\s+',
        ]
        
        # But keep them if they're the main verb
        task_name = cleaned_text
        
        # Ensure we have a meaningful task name
        if len(task_name) < 3:
            task_name = f"Task from {sender_info.get('display_name', 'colleague')}: {mention_text[:50]}"
        
        task_info['task_name'] = task_name
        task_info['description'] = f"Request from {sender_info.get('display_name', 'Unknown')}: {mention_text}"
        
        return task_info
    
    def _parse_due_date(self, due_date_text: str) -> Optional[datetime.datetime]:
        """Parse due date text into datetime object"""
        now = datetime.datetime.now()
        
        # Handle day names
        day_names = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
            'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        if due_date_text in day_names:
            target_day = day_names[due_date_text]
            days_ahead = target_day - now.weekday()
            
            # If it's today or past, assume next week
            if days_ahead <= 0:
                days_ahead += 7
            
            due_date = now + datetime.timedelta(days=days_ahead)
            return due_date.replace(hour=17, minute=0, second=0, microsecond=0)  # 5 PM
        
        # Handle relative dates
        if due_date_text == 'tomorrow':
            return (now + datetime.timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
        elif due_date_text == 'today':
            return now.replace(hour=23, minute=59, second=59, microsecond=0)
        elif due_date_text in ['end of week', 'eow']:
            # Friday at 5 PM
            days_to_friday = (4 - now.weekday()) % 7
            if days_to_friday == 0 and now.hour >= 17:  # If it's Friday after 5 PM
                days_to_friday = 7
            due_date = now + datetime.timedelta(days=days_to_friday)
            return due_date.replace(hour=17, minute=0, second=0, microsecond=0)
        
        # Handle MM/DD format
        date_match = re.match(r'(\d{1,2})/(\d{1,2})', due_date_text)
        if date_match:
            try:
                month, day = int(date_match.group(1)), int(date_match.group(2))
                year = now.year
                
                # If the date is in the past, assume next year
                test_date = datetime.datetime(year, month, day)
                if test_date < now:
                    year += 1
                
                return datetime.datetime(year, month, day, 17, 0, 0)
            except ValueError:
                pass
        
        return None
    
    def create_task_from_mention(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create ClickUp task from parsed mention info"""
        
        if not self.clickup_client:
            return {
                'success': False,
                'error': 'ClickUp not configured',
                'message': 'ClickUp integration not available'
            }
        
        try:
            # Create the task
            task_result = self.clickup_client.create_task(
                name=task_info['task_name'],
                description=task_info['description'],
                due_date=task_info.get('due_date'),
                priority=task_info.get('priority', 3)
            )
            
            # Format response message
            due_text = ""
            if task_info.get('due_date'):
                due_date = task_info['due_date']
                due_text = f" (Due: {due_date.strftime('%A, %B %d')})"
            
            success_message = f"✅ Task created: {task_info['task_name']}{due_text}"
            
            return {
                'success': True,
                'task_id': task_result.get('id'),
                'task_url': task_result.get('url'),
                'message': success_message,
                'task_info': task_info
            }
            
        except Exception as e:
            self.logger.error(f"Task creation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"Failed to create task: {str(e)}"
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
                thread_ts=thread_ts  # Reply in thread
            )
            
            return response.get('ok', False)
            
        except SlackApiError as e:
            self.logger.error(f"Slack API error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send Slack response: {e}")
            return False
    
    def process_slack_mention(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main function to process Slack mention events"""
        
        try:
            # Extract event information
            event = event_data.get('event', {})
            message_text = event.get('text', '')
            channel = event.get('channel', '')
            thread_ts = event.get('ts', '')
            user_id = event.get('user', '')
            
            self.logger.info(f"Processing mention in channel {channel} from user {user_id}")
            
            # Check if this is a mention of our user
            mention_text = self.extract_mention_from_message(message_text)
            if not mention_text:
                return {'success': False, 'error': 'No valid mention found'}
            
            # Get sender information
            sender_info = self._get_user_info(user_id)
            
            # Parse task information
            task_info = self.parse_task_from_mention(mention_text, sender_info)
            
            # Create ClickUp task
            task_result = self.create_task_from_mention(task_info)
            
            # Send response to Slack
            response_sent = self.send_slack_response(
                channel=channel,
                thread_ts=thread_ts,
                message=task_result['message']
            )
            
            # Log to database if available
            try:
                save_conversation_enhanced(
                    'slack_mentions',
                    f"@mention: {mention_text}",
                    {
                        'task_created': task_result['success'],
                        'task_name': task_info['task_name'],
                        'sender': sender_info.get('display_name', 'Unknown'),
                        'channel': channel
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
            self.logger.error(f"Failed to process Slack mention: {e}")
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
    
    # Handle different event types
    event_type = event_data.get('type')
    
    if event_type == 'url_verification':
        # Slack webhook verification
        return {
            'success': True,
            'challenge': event_data.get('challenge')
        }
    
    elif event_type == 'event_callback':
        # Actual event processing
        event = event_data.get('event', {})
        
        # Only process message events that mention our user
        if event.get('type') == 'message' and not event.get('subtype'):
            user_id = handler.user_id
            message_text = event.get('text', '')
            
            # Check if message contains our user mention
            if f"<@{user_id}>" in message_text:
                return handler.process_slack_mention(event_data)
    
    return {'success': True, 'message': 'Event ignored'}