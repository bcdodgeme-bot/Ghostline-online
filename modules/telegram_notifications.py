# modules/telegram_notifications.py
# Telegram Bot notification system with reminder persistence and timezone handling
# DEBUG VERSION - Shows exact database errors to diagnose reminder creation failure

#-- Section 1: Imports and Configuration
import os
import json
import datetime
import hashlib
import requests
import re
import psycopg2.extras
import traceback
from modules.database import get_db_connection

#-- Section 2: TelegramBot Class
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
                # Safe logging without Flask context dependency
                try:
                    from flask import current_app
                    current_app.logger.info(f"Telegram message sent: {result['result']['message_id']}")
                except RuntimeError:
                    print(f"Telegram message sent: {result['result']['message_id']}")
                
                return {"success": True, "message_id": result['result']['message_id']}
            else:
                try:
                    from flask import current_app
                    current_app.logger.error(f"Telegram API error: {result}")
                except RuntimeError:
                    print(f"Telegram API error: {result}")
                
                return {"success": False, "error": result.get('description', 'Unknown error')}
                
        except Exception as e:
            try:
                from flask import current_app
                current_app.logger.error(f"Telegram send failed: {e}")
            except RuntimeError:
                print(f"Telegram send failed: {e}")
            
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
            try:
                from flask import current_app
                current_app.logger.error(f"Failed to get Telegram updates: {e}")
            except RuntimeError:
                print(f"Failed to get Telegram updates: {e}")
            
            return {"ok": False, "error": str(e)}

#-- Section 3: GhostlineTelegramReminders Class - Core Functionality
class GhostlineTelegramReminders:
    def __init__(self):
        self.bot = TelegramBot() if self._is_telegram_configured() else None
        self._ensure_reminder_tables()
    
    def _is_telegram_configured(self):
        """Check if Telegram is properly configured"""
        return bool(os.getenv('TELEGRAM_BOT_TOKEN'))
    
    def _ensure_reminder_tables(self):
        """Create reminder tables if they don't exist"""
        try:
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
                        self._safe_log_error(f"Failed to create telegram tables: {e}")
        except Exception as e:
            self._safe_log_error(f"Database connection failed during table creation: {e}")
    
    def _safe_log_error(self, message):
        """Log error safely without Flask context dependency"""
        try:
            from flask import current_app
            current_app.logger.error(message)
        except RuntimeError:
            print(f"ERROR: {message}")
    
    def _safe_log_info(self, message):
        """Log info safely without Flask context dependency"""
        try:
            from flask import current_app
            current_app.logger.info(message)
        except RuntimeError:
            print(f"INFO: {message}")

#-- Section 4: Reminder Creation (DEBUG VERSION)
    def create_reminder(self, title: str, remind_at: datetime.datetime,
                       project: str = None, priority: int = 3, content: str = None,
                       reminder_type: str = "user_reminder", repeat_pattern: str = None,
                       metadata: dict = None) -> dict:
        """Create a reminder - DEBUG VERSION with detailed logging"""
        
        print(f"🔍 DEBUG: create_reminder called with:")
        print(f"   title: {title}")
        print(f"   remind_at: {remind_at}")
        print(f"   project: {project}")
        print(f"   priority: {priority}")
        print(f"   content: {content}")
        print(f"   reminder_type: {reminder_type}")
        print(f"   repeat_pattern: {repeat_pattern}")
        print(f"   metadata: {metadata}")
        
        if not self.bot:
            print(f"❌ DEBUG: Bot not configured")
            return {"success": False, "error": "Telegram not configured"}
        
        try:
            # Generate unique reminder ID
            reminder_id = hashlib.md5(f"{title}{remind_at}{datetime.datetime.now()}".encode()).hexdigest()
            print(f"🔍 DEBUG: Generated reminder_id: {reminder_id}")
            
            result = self._actually_create_reminder(
                reminder_id=reminder_id,
                reminder_type=reminder_type,
                title=title,
                content=content,
                remind_at=remind_at,
                project=project,
                priority=priority,
                repeat_pattern=repeat_pattern,
                metadata=metadata
            )
            
            print(f"🔍 DEBUG: _actually_create_reminder returned: {result}")
            return result
            
        except Exception as e:
            print(f"❌ DEBUG: Exception in create_reminder: {e}")
            print(f"❌ DEBUG: Exception type: {type(e)}")
            traceback.print_exc()
            return {"success": False, "error": f"create_reminder failed: {str(e)}"}
    
    def schedule_reminder(self, title: str, remind_at: datetime.datetime,
                         project: str = None, priority: int = 2, content: str = None,
                         repeat_pattern: str = None, metadata: dict = None) -> dict:
        """Schedule a reminder - DEBUG VERSION with detailed logging"""
        
        print(f"🔍 DEBUG: schedule_reminder called with:")
        print(f"   title: {title}")
        print(f"   remind_at: {remind_at}")
        print(f"   project: {project}")
        print(f"   priority: {priority}")
        print(f"   content: {content}")
        print(f"   repeat_pattern: {repeat_pattern}")
        print(f"   metadata: {metadata}")
        
        try:
            # Use the existing create_reminder method internally
            result = self.create_reminder(
                title=title,
                content=content,
                remind_at=remind_at,
                project=project,
                priority=priority,
                repeat_pattern=repeat_pattern,
                metadata=metadata
            )
            
            print(f"🔍 DEBUG: create_reminder returned: {result}")
            
            if result.get('success'):
                print(f"✅ DEBUG: Scheduled reminder successfully: {title} at {remind_at}")
                self._safe_log_info(f"Scheduled reminder via schedule_reminder: {title} at {remind_at}")
            else:
                print(f"❌ DEBUG: create_reminder failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ DEBUG: Exception in schedule_reminder: {e}")
            print(f"❌ DEBUG: Exception type: {type(e)}")
            traceback.print_exc()
            self._safe_log_error(f"Failed to schedule reminder: {e}")
            return {
                "success": False,
                "error": f"Reminder scheduling failed: {str(e)}"
            }
    
    def _actually_create_reminder(self, reminder_id: str, reminder_type: str, title: str,
                                 content: str, remind_at: datetime.datetime, project: str,
                                 priority: int, repeat_pattern: str, metadata: dict) -> dict:
        """Actually create the reminder in database - DEBUG VERSION"""
        print(f"🔍 DEBUG: _actually_create_reminder called")
        print(f"   reminder_id: {reminder_id}")
        print(f"   reminder_type: {reminder_type}")
        print(f"   title: {title}")
        print(f"   content: {content}")
        print(f"   remind_at: {remind_at}")
        print(f"   project: {project}")
        print(f"   priority: {priority}")
        print(f"   repeat_pattern: {repeat_pattern}")
        print(f"   metadata: {metadata}")
        
        try:
            with get_db_connection() as conn:
                if not conn:
                    print(f"❌ DEBUG: No database connection")
                    return {"success": False, "error": "Database not available"}
                
                print(f"✅ DEBUG: Got database connection")
                
                try:
                    cursor = conn.cursor()
                    print(f"✅ DEBUG: Got database cursor")
                    
                    # Print the exact SQL we're about to execute
                    sql = '''
                        INSERT INTO telegram_reminders 
                        (reminder_id, reminder_type, title, content, remind_at, 
                         project, priority, repeat_pattern, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    '''
                    params = (reminder_id, reminder_type, title, content, remind_at,
                             project, priority, repeat_pattern,
                             psycopg2.extras.Json(metadata or {}))
                    
                    print(f"🔍 DEBUG: Executing SQL:")
                    print(f"   SQL: {sql}")
                    print(f"   Params: {params}")
                    
                    cursor.execute(sql, params)
                    print(f"✅ DEBUG: SQL executed successfully")
                    
                    conn.commit()
                    print(f"✅ DEBUG: Transaction committed")
                    
                    self._safe_log_info(f"Telegram reminder created: {reminder_id}")
                    print(f"✅ DEBUG: Reminder created successfully: {reminder_id}")
                    
                    return {"success": True, "reminder_id": reminder_id, "remind_at": remind_at}
                    
                except Exception as db_error:
                    print(f"❌ DEBUG: Database operation failed: {db_error}")
                    print(f"❌ DEBUG: Database error type: {type(db_error)}")
                    traceback.print_exc()
                    
                    conn.rollback()
                    print(f"🔄 DEBUG: Transaction rolled back")
                    
                    self._safe_log_error(f"Failed to create reminder: {db_error}")
                    return {"success": False, "error": str(db_error)}
            
            print(f"❌ DEBUG: Database connection block exited - this shouldn't happen")
            return {"success": False, "error": "Database not available"}
            
        except Exception as outer_error:
            print(f"❌ DEBUG: Outer exception in _actually_create_reminder: {outer_error}")
            print(f"❌ DEBUG: Outer exception type: {type(outer_error)}")
            traceback.print_exc()
            
            self._safe_log_error(f"Database connection failed: {outer_error}")
            return {"success": False, "error": f"Database connection failed: {str(outer_error)}"}

#-- Section 5: Reminder Checking and Sending (Anti-Spam)
    def check_and_send_reminders(self):
        """Check for due reminders and send via Telegram - SPAM PREVENTION FIXED"""
        if not self.bot:
            return {"sent": 0, "error": "Telegram not configured"}
        
        try:
            with get_db_connection() as conn:
                if not conn:
                    return {"sent": 0, "error": "Database not available"}
                
                try:
                    cursor = conn.cursor()
                    
                    # CRITICAL FIX: Only get 'pending' status, NEVER 'sent'
                    # This prevents the infinite loop that caused spam
                    cursor.execute('''
                        SELECT reminder_id, reminder_type, title, content, priority, 
                               project, remind_at, repeat_pattern
                        FROM telegram_reminders 
                        WHERE status = 'pending' 
                        AND remind_at <= %s
                        AND (snooze_until IS NULL OR snooze_until <= %s)
                        ORDER BY priority ASC, remind_at ASC
                        LIMIT 5
                    ''', (datetime.datetime.now(), datetime.datetime.now()))
                    
                    due_reminders = cursor.fetchall()
                    sent_count = 0
                    
                    for reminder in due_reminders:
                        (reminder_id, reminder_type, title, content, priority,
                         project, remind_at, repeat_pattern) = reminder
                        
                        # IMMEDIATELY mark as 'sent' to prevent duplicates
                        cursor.execute('''
                            UPDATE telegram_reminders 
                            SET status = 'sent' 
                            WHERE reminder_id = %s AND status = 'pending'
                        ''', (reminder_id,))
                        
                        # Only proceed if we actually updated a pending reminder
                        if cursor.rowcount == 0:
                            continue  # Skip if another process already processed this
                        
                        # Commit the status change immediately
                        conn.commit()
                        
                        # Format Telegram message with markdown
                        message_parts = ["📱 *GHOSTLINE REMINDER*"]
                        
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
                                    {"text": "📝 More Info", "callback_data": f"info_{reminder_id}"}
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
                            self._safe_log_info(f"Sent Telegram reminder: {title}")
                            
                            # Handle repeating reminders
                            if repeat_pattern:
                                self._schedule_repeat(reminder_id, repeat_pattern, remind_at, cursor)
                        else:
                            # If send failed, revert to pending for retry
                            cursor.execute('''
                                UPDATE telegram_reminders 
                                SET status = 'pending' 
                                WHERE reminder_id = %s
                            ''', (reminder_id,))
                            self._safe_log_error(f"Failed to send reminder {reminder_id}: {result['error']}")
                    
                    conn.commit()
                    return {"sent": sent_count, "total_due": len(due_reminders)}
                    
                except Exception as e:
                    self._safe_log_error(f"Telegram reminder check failed: {e}")
                    conn.rollback()
                    return {"sent": 0, "error": str(e)}
                    
        except Exception as e:
            self._safe_log_error(f"Database connection failed during reminder check: {e}")
            return {"sent": 0, "error": f"Database error: {str(e)}"}

#-- Section 6: Timezone and Utility Functions
    def _utc_to_eastern(self, utc_time):
        """Convert UTC time to Eastern time for display - FIXED"""
        # Eastern Time is BEHIND UTC: EST = UTC-5, EDT = UTC-4
        # To convert FROM UTC TO Eastern, we SUBTRACT the offset
        
        now = datetime.datetime.now()
        is_dst = now.month >= 3 and now.month <= 10  # March through October
        
        if is_dst:
            # EDT = UTC-4, so UTC - 4 hours = Eastern
            offset = -4
        else:
            # EST = UTC-5, so UTC - 5 hours = Eastern
            offset = -5
        
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
                
                self._safe_log_info(f"Scheduled repeat reminder for {next_time}")
                
        except Exception as e:
            self._safe_log_error(f"Failed to schedule repeat: {e}")

#-- Section 7: Callback Handling (Button Presses)
    def process_callback_query(self, callback_query):
        """Handle button presses from Telegram"""
        try:
            data = callback_query.get('data', '')
            message_id = callback_query.get('message', {}).get('message_id')
            
            self._safe_log_info(f"Processing callback: {data} for message: {message_id}")
            
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
            self._safe_log_error(f"Callback processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_done(self, reminder_id, message_id):
        """Mark reminder as completed"""
        try:
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
                        
                        # Send confirmation using the existing bot instance
                        if self.bot:
                            self.bot.send_message("✅ *Reminder marked as completed!*\n\nNice work getting that done.")
                        
                        return {"success": True, "action": "completed"}
                        
                    except Exception as e:
                        self._safe_log_error(f"Failed to mark reminder complete: {e}")
                        return {"success": False, "error": str(e)}
            
            return {"success": False, "error": "Database not available"}
            
        except Exception as e:
            self._safe_log_error(f"Database connection failed in _handle_done: {e}")
            return {"success": False, "error": "Database not available"}
    
    def _handle_snooze(self, reminder_id, minutes, message_id):
        """Snooze reminder for specified minutes"""
        try:
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
                        
                        # Send confirmation
                        eastern_snooze_time = self._utc_to_eastern(snooze_until)
                        snooze_display = eastern_snooze_time.strftime('%I:%M %p')
                        
                        if self.bot:
                            self.bot.send_message(f"⏰ *Reminder snoozed until {snooze_display}*\n\nI'll remind you again then.")
                        
                        return {"success": True, "action": f"snoozed_{minutes}min", "snooze_until": snooze_until}
                        
                    except Exception as e:
                        self._safe_log_error(f"Failed to snooze reminder: {e}")
                        return {"success": False, "error": str(e)}
            
            return {"success": False, "error": "Database not available"}
            
        except Exception as e:
            self._safe_log_error(f"Database connection failed in _handle_snooze: {e}")
            return {"success": False, "error": "Database not available"}
    
    def _handle_info_request(self, reminder_id):
        """Send detailed info about reminder"""
        try:
            with get_db_connection() as conn:
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            SELECT title, content, project, priority, created_at, remind_at, status
                            FROM telegram_reminders 
                            WHERE reminder_id = %s
                        ''', (reminder_id,))
                        
                        result = cursor.fetchone()
                        
                        if result:
                            title, content, project, priority, created_at, remind_at, status = result
                            
                            info_parts = [f"📝 *Reminder Details*\n"]
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
                            if self.bot:
                                self.bot.send_message(info_message)
                            
                            return {"success": True, "action": "info_sent"}
                        else:
                            if self.bot:
                                self.bot.send_message("❌ *Reminder not found*")
                            return {"success": False, "error": "Reminder not found"}
                            
                    except Exception as e:
                        self._safe_log_error(f"Failed to get reminder info: {e}")
                        return {"success": False, "error": str(e)}
            
            return {"success": False, "error": "Database not available"}
            
        except Exception as e:
            self._safe_log_error(f"Database connection failed in _handle_info_request: {e}")
            return {"success": False, "error": "Database not available"}

#-- Section 8: Emergency and Management Functions
    def emergency_stop_all(self):
        """Emergency function to stop all pending reminders"""
        try:
            with get_db_connection() as conn:
                if conn:
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            UPDATE telegram_reminders 
                            SET status = 'cancelled' 
                            WHERE status = 'pending'
                        ''')
                        
                        stopped_count = cursor.rowcount
                        conn.commit()
                        
                        self._safe_log_info(f"Emergency stop: cancelled {stopped_count} pending reminders")
                        
                        if self.bot:
                            self.bot.send_message(f"🛑 *EMERGENCY STOP EXECUTED*\n\nCancelled {stopped_count} pending reminders.")
                        
                        return {"success": True, "stopped_count": stopped_count}
                        
                    except Exception as e:
                        self._safe_log_error(f"Emergency stop failed: {e}")
                        return {"success": False, "error": str(e)}
            
            return {"success": False, "error": "Database not available"}
            
        except Exception as e:
            self._safe_log_error(f"Database connection failed during emergency stop: {e}")
            return {"success": False, "error": "Database not available"}

#-- Section 9: Reminder Parsing System
def parse_reminder_command(user_input: str, project: str) -> dict:
    """Parse natural language reminder commands - ENHANCED"""
    original_input = user_input.strip()
    user_input = user_input.lower().strip()
    
    # Enhanced time parsing helpers
    def parse_today_time(hour_str, ampm):
        """Parse time for today"""
        try:
            hour = int(hour_str)
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            
            now = datetime.datetime.now()
            target_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # If the time has passed today, assume tomorrow
            if target_time <= now:
                target_time += datetime.timedelta(days=1)
            
            return target_time - now
            
        except ValueError:
            return datetime.timedelta(hours=1)  # Default fallback
    
    def parse_tomorrow_time(hour_str, ampm):
        """Parse time for tomorrow"""
        try:
            hour = int(hour_str)
            if ampm == 'pm' and hour != 12:
                hour += 12
            elif ampm == 'am' and hour == 12:
                hour = 0
            
            tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
            target_time = tomorrow.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            return target_time - datetime.datetime.now()
            
        except ValueError:
            return datetime.timedelta(days=1)  # Default tomorrow
    
    # Enhanced time pattern matching with natural language support
    time_patterns = [
        # Natural language times
        (r'in five minutes?', lambda m: datetime.timedelta(minutes=5)),
        (r'in ten minutes?', lambda m: datetime.timedelta(minutes=10)),
        (r'in fifteen minutes?', lambda m: datetime.timedelta(minutes=15)),
        (r'in thirty minutes?', lambda m: datetime.timedelta(minutes=30)),
        (r'in one hours?', lambda m: datetime.timedelta(hours=1)),
    
        # Existing digit patterns
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
            "error": "No reminder content found. Please specify what you want to be reminded about."
        }
    
    # Calculate target time
    remind_at = datetime.datetime.now() + remind_delta
    
    # Format display time
    display_time = remind_at.strftime('%I:%M %p on %B %d')
    
    return {
        "success": True,
        "title": reminder_text,
        "remind_at": remind_at,
        "project": project,
        "display_time": display_time,
        "original_input": original_input
    }

#-- Section 10: Utility Functions
def is_telegram_configured() -> bool:
    """Check if Telegram is properly configured"""
    return bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))

# Export the classes and functions for import
__all__ = [
    'TelegramBot',
    'GhostlineTelegramReminders',
    'parse_reminder_command',
    'is_telegram_configured'
]
