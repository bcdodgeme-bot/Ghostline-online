# modules/telegram_notifications.py
# Telegram Bot notification system with reminder persistence and timezone handling

import os
import json
import datetime
import hashlib
import requests
import re
import psycopg2.extras
from modules.database import get_db_connection
from flask import current_app

class TelegramBot:
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN not configured")
        
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        
        # Auto-detect chat_id if not set
        if not self.chat_id:
            self.chat_id = self._get_my_chat_id()
    
    def _get_my_chat_id(self):
        """Get chat ID from recent messages (for setup)"""
        try:
            response = requests.get(f"{self.base_url}/getUpdates")
            data = response.json()
            
            if data['ok'] and data['result']:
                # Get the most recent message's chat_id
                return str(data['result'][-1]['message']['chat']['id'])
            
            return None
            
        except Exception as e:
            print(f"Failed to auto-detect chat_id: {e}")
            return None
    
    def send_message(self, message, parse_mode='Markdown', reply_markup=None):
        """Send message to Telegram"""
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            if reply_markup:
                payload['reply_markup'] = json.dumps(reply_markup)
            
            response = requests.post(f"{self.base_url}/sendMessage", data=payload)
            result = response.json()
            
            if result['ok']:
                current_app.logger.info(f"Telegram message sent: {result['result']['message_id']}")
                return {"success": True, "message_id": result['result']['message_id']}
            else:
                current_app.logger.error(f"Telegram API error: {result}")
                return {"success": False, "error": result.get('description', 'Unknown error')}
                
        except Exception as e:
            current_app.logger.error(f"Telegram send failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_updates(self, offset=None):
        """Get new messages from Telegram"""
        try:
            params = {}
            if offset:
                params['offset'] = offset
            
            response = requests.get(f"{self.base_url}/getUpdates", params=params)
            return response.json()
            
        except Exception as e:
            current_app.logger.error(f"Failed to get Telegram updates: {e}")
            return {"ok": False, "error": str(e)}

class GhostlineTelegramReminders:
    def __init__(self):
        self.bot = TelegramBot() if self._is_telegram_configured() else None
        self._ensure_reminder_tables()
    
    def _is_telegram_configured(self):
        """Check if Telegram is properly configured"""
        return bool(os.getenv('TELEGRAM_BOT_TOKEN'))
    
    def _ensure_reminder_tables(self):
        """Create reminder tables if they don't exist"""
        with get_db_connection() as conn:
            if conn:
                try:
                    cursor = conn.cursor()
                    
                    # Main reminders table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS telegram_reminders (
                            id SERIAL PRIMARY KEY,
                            reminder_id VARCHAR(255) UNIQUE NOT NULL,
                            reminder_type VARCHAR(100) NOT NULL,
                            title VARCHAR(500) NOT NULL,
                            content TEXT,
                            remind_at TIMESTAMP NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            status VARCHAR(50) DEFAULT 'pending',
                            project VARCHAR(100),
                            priority INTEGER DEFAULT 3,
                            metadata JSONB DEFAULT '{}',
                            repeat_pattern VARCHAR(100),
                            snooze_until TIMESTAMP
                        )
                    ''')
                    
                    # Sent messages tracking
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS telegram_messages (
                            id SERIAL PRIMARY KEY,
                            reminder_id VARCHAR(255),
                            message_id INTEGER,
                            message_content TEXT NOT NULL,
                            sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            status VARCHAR(50) DEFAULT 'sent',
                            response_received BOOLEAN DEFAULT FALSE,
                            response_content TEXT,
                            response_at TIMESTAMP
                        )
                    ''')
                    
                    # Create indexes
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_telegram_reminders_status_time 
                        ON telegram_reminders (status, remind_at)
                    ''')
                    
                    conn.commit()
                    
                except Exception as e:
                    current_app.logger.error(f"Failed to create telegram tables: {e}")
    
    def create_reminder(self, title, content="", remind_at=None, reminder_type='general', 
                       project=None, priority=3, repeat_pattern=None, metadata=None):
        """Create a persistent reminder"""
        
        if remind_at is None:
            remind_at = datetime.datetime.now() + datetime.timedelta(hours=1)
        
        # Generate unique reminder ID
        reminder_id = hashlib.md5(
            f"{title}{remind_at}{datetime.datetime.now()}".encode()
        ).hexdigest()[:16]
        
        with get_db_connection() as conn:
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO telegram_reminders 
                        (reminder_id, reminder_type, title, content, remind_at, 
                         project, priority, repeat_pattern, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (reminder_id, reminder_type, title, content, remind_at,
                         project, priority, repeat_pattern, 
                         psycopg2.extras.Json(metadata or {})))
                    
                    conn.commit()
                    
                    current_app.logger.info(f"Telegram reminder created: {reminder_id}")
                    return {"success": True, "reminder_id": reminder_id, "remind_at": remind_at}
                    
                except Exception as e:
                    current_app.logger.error(f"Failed to create reminder: {e}")
                    return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Database not available"}
    
    def check_and_send_reminders(self):
        """Check for due reminders and send via Telegram - FIXED TO PREVENT SPAM"""
        if not self.bot:
            return {"sent": 0, "error": "Telegram not configured"}
        
        with get_db_connection() as conn:
            if not conn:
                return {"sent": 0, "error": "Database not available"}
            
            try:
                cursor = conn.cursor()
                
                # Get pending reminders that are due (not snoozed)
                # CRITICAL FIX: Only get 'pending' status, not 'sent'
                cursor.execute('''
                    SELECT reminder_id, reminder_type, title, content, priority, 
                           project, remind_at, repeat_pattern
                    FROM telegram_reminders 
                    WHERE status = 'pending' 
                    AND remind_at <= %s
                    AND (snooze_until IS NULL OR snooze_until <= %s)
                    ORDER BY priority ASC, remind_at ASC
                    LIMIT 10
                ''', (datetime.datetime.now(), datetime.datetime.now()))
                
                due_reminders = cursor.fetchall()
                sent_count = 0
                
                for reminder in due_reminders:
                    (reminder_id, reminder_type, title, content, priority, 
                     project, remind_at, repeat_pattern) = reminder
                    
                    # CRITICAL FIX: Immediately mark as 'sent' to prevent duplicates
                    cursor.execute('''
                        UPDATE telegram_reminders 
                        SET status = 'sent' 
                        WHERE reminder_id = %s
                    ''', (reminder_id,))
                    
                    # Format Telegram message with markdown
                    message_parts = ["🔔 *GHOSTLINE REMINDER*"]
                    
                    if project:
                        message_parts.append(f"📁 *Project:* {project}")
                    
                    # Priority indicators
                    priority_indicators = {1: "🔴 *URGENT*", 2: "🟡 *HIGH*", 3: "🟢 *NORMAL*"}
                    message_parts.append(f"⚡ *Priority:* {priority_indicators.get(priority, '🟢 NORMAL')}")
                    
                    message_parts.append(f"\n*{title}*")
                    
                    if content and content.strip():
                        message_parts.append(f"\n{content}")
                    
                    # Time info - convert UTC to Eastern for display
                    eastern_time = self._utc_to_eastern(remind_at)
                    time_str = eastern_time.strftime('%I:%M %p on %b %d')
                    message_parts.append(f"\n⏰ *Scheduled:* {time_str}")
                    
                    # Add quick action buttons
                    reply_markup = {
                        "inline_keyboard": [
                            [
                                {"text": "✅ Done", "callback_data": f"done_{reminder_id}"},
                                {"text": "⏰ Snooze 15m", "callback_data": f"snooze15_{reminder_id}"}
                            ],
                            [
                                {"text": "⏰ Snooze 1h", "callback_data": f"snooze60_{reminder_id}"},
                                {"text": "🔍 More Info", "callback_data": f"info_{reminder_id}"}
                            ]
                        ]
                    }
                    
                    telegram_message = "\n".join(message_parts)
                    
                    # Send via Telegram
                    result = self.bot.send_message(telegram_message, reply_markup=reply_markup)
                    
                    if result["success"]:
                        # Track the sent message
                        cursor.execute('''
                            INSERT INTO telegram_messages 
                            (reminder_id, message_id, message_content)
                            VALUES (%s, %s, %s)
                        ''', (reminder_id, result["message_id"], telegram_message))
                        
                        sent_count += 1
                        current_app.logger.info(f"Sent Telegram reminder: {title}")
                        
                        # Handle repeating reminders
                        if repeat_pattern:
                            self._schedule_repeat(reminder_id, repeat_pattern, remind_at, cursor)
                    else:
                        # If send failed, revert to pending
                        cursor.execute('''
                            UPDATE telegram_reminders 
                            SET status = 'pending' 
                            WHERE reminder_id = %s
                        ''', (reminder_id,))
                        current_app.logger.error(f"Failed to send reminder {reminder_id}: {result['error']}")
                
                conn.commit()
                
                return {"sent": sent_count, "total_due": len(due_reminders)}
                
            except Exception as e:
                current_app.logger.error(f"Telegram reminder check failed: {e}")
                return {"sent": 0, "error": str(e)}
    
    def _utc_to_eastern(self, utc_time):
        """Convert UTC time to Eastern time for display"""
        # Simple timezone conversion - Eastern is UTC-5 (EST) or UTC-4 (EDT)
        # This is approximate - doesn't handle DST transitions perfectly
        import calendar
        
        # Check if we're in DST period (rough approximation)
        now = datetime.datetime.now()
        is_dst = now.month >= 3 and now.month <= 10  # March through October
        
        if is_dst:
            offset = -4  # EDT
        else:
            offset = -5  # EST
        
        return utc_time + datetime.timedelta(hours=offset)
    
    def _schedule_repeat(self, reminder_id, repeat_pattern, original_time, cursor):
        """Schedule next occurrence for repeating reminders"""
        try:
            next_time = None
            
            if repeat_pattern == 'daily':
                next_time = original_time + datetime.timedelta(days=1)
            elif repeat_pattern == 'weekly':
                next_time = original_time + datetime.timedelta(weeks=1)
            elif repeat_pattern == 'workdays':
                # Next workday (Mon-Fri)
                next_time = original_time + datetime.timedelta(days=1)
                while next_time.weekday() > 4:  # 0-4 is Mon-Fri
                    next_time += datetime.timedelta(days=1)
            
            if next_time:
                # Create new reminder for next occurrence
                cursor.execute('''
                    INSERT INTO telegram_reminders 
                    (reminder_id, reminder_type, title, content, remind_at, 
                     project, priority, repeat_pattern, metadata, status)
                    SELECT 
                        %s || '_' || extract(epoch from %s)::text,
                        reminder_type, title, content, %s,
                        project, priority, repeat_pattern, metadata, 'pending'
                    FROM telegram_reminders 
                    WHERE reminder_id = %s
                ''', (reminder_id, next_time, next_time, reminder_id))
                
                current_app.logger.info(f"Scheduled repeat reminder for {next_time}")
                
        except Exception as e:
            current_app.logger.error(f"Failed to schedule repeat: {e}")
    
    def process_callback_query(self, callback_query):
        """Handle button presses from Telegram"""
        try:
            data = callback_query.get('data', '')
            message_id = callback_query.get('message', {}).get('message_id')
            
            current_app.logger.info(f"Processing callback: {data} for message: {message_id}")
            
            if data.startswith('done_'):
                reminder_id = data[5:]
                return self._handle_done(reminder_id, message_id)
            elif data.startswith('snooze15_'):
                reminder_id = data[9:]
                return self._handle_snooze(reminder_id, 15, message_id)
            elif data.startswith('snooze60_'):
                reminder_id = data[9:]
                return self._handle_snooze(reminder_id, 60, message_id)
            elif data.startswith('info_'):
                reminder_id = data[5:]
                return self._handle_info_request(reminder_id)
            
            return {"success": False, "error": "Unknown callback"}
            
        except Exception as e:
            current_app.logger.error(f"Callback processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_done(self, reminder_id, message_id):
        """Mark reminder as completed"""
        with get_db_connection() as conn:
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE telegram_reminders 
                        SET status = 'completed' 
                        WHERE reminder_id = %s
                    ''', (reminder_id,))
                    
                    conn.commit()
                    
                    # Send confirmation
                    self.bot.send_message("✅ *Reminder marked as completed!*")
                    current_app.logger.info(f"Reminder {reminder_id} marked as completed")
                    return {"success": True, "action": "completed"}
                    
                except Exception as e:
                    current_app.logger.error(f"Failed to mark reminder as done: {e}")
                    return {"success": False, "error": str(e)}
    
    def _handle_snooze(self, reminder_id, minutes, message_id):
        """Snooze reminder for specified minutes"""
        snooze_until = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
        
        with get_db_connection() as conn:
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE telegram_reminders 
                        SET status = 'pending', snooze_until = %s 
                        WHERE reminder_id = %s
                    ''', (snooze_until, reminder_id))
                    
                    conn.commit()
                    
                    # Convert to Eastern for display
                    eastern_snooze = self._utc_to_eastern(snooze_until)
                    time_str = eastern_snooze.strftime('%I:%M %p')
                    self.bot.send_message(f"⏰ *Reminder snoozed until {time_str}*")
                    current_app.logger.info(f"Reminder {reminder_id} snoozed until {snooze_until}")
                    return {"success": True, "action": "snoozed", "until": snooze_until}
                    
                except Exception as e:
                    current_app.logger.error(f"Failed to snooze reminder: {e}")
                    return {"success": False, "error": str(e)}
    
    def _handle_info_request(self, reminder_id):
        """Handle more info request"""
        with get_db_connection() as conn:
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT title, content, project, created_at, priority
                        FROM telegram_reminders 
                        WHERE reminder_id = %s
                    ''', (reminder_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        title, content, project, created_at, priority = result
                        
                        info_parts = [f"🔍 *Reminder Details*"]
                        info_parts.append(f"*Title:* {title}")
                        
                        if content:
                            info_parts.append(f"*Details:* {content}")
                        
                        if project:
                            info_parts.append(f"*Project:* {project}")
                        
                        priority_names = {1: "URGENT", 2: "HIGH", 3: "NORMAL"}
                        info_parts.append(f"*Priority:* {priority_names.get(priority, 'NORMAL')}")
                        
                        # Convert created time to Eastern
                        eastern_created = self._utc_to_eastern(created_at)
                        created_str = eastern_created.strftime('%I:%M %p on %b %d')
                        info_parts.append(f"*Created:* {created_str}")
                        
                        info_message = "\n".join(info_parts)
                        self.bot.send_message(info_message)
                        
                        return {"success": True, "action": "info_sent"}
                    else:
                        self.bot.send_message("❌ *Reminder not found*")
                        return {"success": False, "error": "Reminder not found"}
                        
                except Exception as e:
                    current_app.logger.error(f"Failed to get reminder info: {e}")
                    return {"success": False, "error": str(e)}
    
    def emergency_stop_all(self):
        """Emergency function to stop all pending reminders"""
        with get_db_connection() as conn:
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE telegram_reminders 
                        SET status = 'emergency_stopped', 
                            snooze_until = NULL
                        WHERE status IN ('pending', 'sent')
                    ''')
                    
                    stopped_count = cursor.rowcount
                    conn.commit()
                    
                    current_app.logger.info(f"Emergency stop: {stopped_count} reminders stopped")
                    return {"success": True, "stopped_count": stopped_count}
                    
                except Exception as e:
                    current_app.logger.error(f"Emergency stop failed: {e}")
                    return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Database not available"}
    
    def quick_reminder(self, message, minutes_from_now=60, project=None):
        """Create a quick reminder for X minutes from now"""
        remind_time = datetime.datetime.now() + datetime.timedelta(minutes=minutes_from_now)
        
        return self.create_reminder(
            title=f"Quick Reminder",
            content=message,
            remind_at=remind_time,
            reminder_type="quick",
            project=project,
            priority=2
        )
    
    def get_active_reminders(self, limit=10):
        """Get list of active reminders"""
        with get_db_connection() as conn:
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT reminder_id, title, remind_at, priority, project, status
                        FROM telegram_reminders 
                        WHERE status IN ('pending', 'sent')
                        ORDER BY remind_at ASC
                        LIMIT %s
                    ''', (limit,))
                    
                    results = cursor.fetchall()
                    
                    reminders = []
                    for row in results:
                        reminders.append({
                            'reminder_id': row[0],
                            'title': row[1],
                            'remind_at': row[2],
                            'priority': row[3],
                            'project': row[4],
                            'status': row[5]
                        })
                    
                    return reminders
                    
                except Exception as e:
                    current_app.logger.error(f"Failed to get active reminders: {e}")
                    return []

def parse_reminder_command(user_input, project=None):
    """Parse natural language reminder commands with Eastern timezone"""
    
    # Clean up the input
    original_input = user_input
    user_input = user_input.lower().strip()
    
    # Time parsing patterns with more flexibility
    time_patterns = [
        (r'in (\d+) minutes?', lambda m: datetime.timedelta(minutes=int(m.group(1)))),
        (r'in (\d+) hours?', lambda m: datetime.timedelta(hours=int(m.group(1)))),
        (r'in (\d+) days?', lambda m: datetime.timedelta(days=int(m.group(1)))),
        (r'tomorrow at (\d+)(am|pm)', lambda m: parse_tomorrow_time(m.group(1), m.group(2))),
        (r'at (\d+)(am|pm)', lambda m: parse_today_time(m.group(1), m.group(2))),
        (r'in (\d+)m', lambda m: datetime.timedelta(minutes=int(m.group(1)))),
        (r'in (\d+)h', lambda m: datetime.timedelta(hours=int(m.group(1)))),
    ]
    
    # Extract timing information
    remind_delta = datetime.timedelta(hours=1)  # Default 1 hour
    reminder_text = original_input
    
    for pattern, time_func in time_patterns:
        match = re.search(pattern, user_input)
        if match:
            try:
                remind_delta = time_func(match)
                # Remove time portion from reminder text
                reminder_text = re.sub(pattern, '', original_input, flags=re.IGNORECASE).strip()
                break
            except:
                continue
    
    # Clean up reminder text - remove command prefixes
    prefixes_to_remove = [
        r'^(remind me to?|reminder:?|set reminder for?|alert me to?)\s*',
        r'^(remember to?|don\'t forget to?)\s*'
    ]
    
    for prefix_pattern in prefixes_to_remove:
        reminder_text = re.sub(prefix_pattern, '', reminder_text, flags=re.IGNORECASE).strip()
    
    if not reminder_text:
        return {
            "success": False, 
            "error": "No reminder content found. Try: 'remind me to call John in 30 minutes'"
        }
    
    # Calculate Eastern time first, then store as UTC
    eastern_time = datetime.datetime.now() + remind_delta
    
    # Convert Eastern to UTC for storage (approximate)
    now = datetime.datetime.now()
    is_dst = now.month >= 3 and now.month <= 10  # Rough DST check
    utc_offset = 4 if is_dst else 5  # EDT is UTC-4, EST is UTC-5
    utc_time = eastern_time + datetime.timedelta(hours=utc_offset)
    
    return {
        "success": True,
        "title": reminder_text,
        "remind_at": utc_time,  # Store in UTC
        "project": project,
        "remind_delta": remind_delta,
        "display_time": eastern_time.strftime('%I:%M %p on %B %d')  # Eastern display
    }

def parse_tomorrow_time(hour, period):
    """Parse tomorrow at specific time"""
    hour = int(hour)
    if period.lower() == 'pm' and hour != 12:
        hour += 12
    elif period.lower() == 'am' and hour == 12:
        hour = 0
    
    tomorrow = datetime.datetime.now().replace(
        hour=hour, minute=0, second=0, microsecond=0
    ) + datetime.timedelta(days=1)
    
    return tomorrow - datetime.datetime.now()

def parse_today_time(hour, period):
    """Parse today at specific time"""
    hour = int(hour)
    if period.lower() == 'pm' and hour != 12:
        hour += 12
    elif period.lower() == 'am' and hour == 12:
        hour = 0
    
    today = datetime.datetime.now().replace(
        hour=hour, minute=0, second=0, microsecond=0
    )
    
    if today <= datetime.datetime.now():
        today += datetime.timedelta(days=1)
    
    return today - datetime.datetime.now()

def is_telegram_configured():
    """Check if Telegram bot is configured"""
    return bool(os.getenv('TELEGRAM_BOT_TOKEN'))