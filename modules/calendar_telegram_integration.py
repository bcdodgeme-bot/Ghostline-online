# modules/calendar_telegram_integration_hotfix.py - HOTFIX version that works without database

import os
import datetime
import json
import re
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta
import pytz

from utils.gmail_client import (
    list_today_events, list_tomorrow_events, search_calendar,
    get_next_meeting, format_calendar_summary
)
from modules.telegram_notifications import TelegramBot

class CalendarTelegramAlertsHotfix:
    """Calendar monitoring and Telegram notification system - HOTFIX version without database dependency"""
    
    def __init__(self):
        self.bot = TelegramBot() if self._is_telegram_configured() else None
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.monitoring_enabled = False
        self.last_calendar_hash = None
        self.sent_alerts = set()  # Track sent alerts to prevent duplicates
        
        # Default settings (stored in memory for now)
        self.preferences = {
            'meeting_alerts': {
                'enabled': True,
                'alert_times': [15, 30],
                'include_weekends': False
            },
            'daily_summary': {
                'enabled': True,
                'time': '07:00',
                'include_tomorrow': True
            },
            'calendar_changes': {
                'enabled': False,
                'immediate_notification': True
            }
        }
        
        self.timezone = self._get_user_timezone()
    
    def _is_telegram_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
    
    def _get_user_timezone(self):
        """Get user's timezone with proper fallback"""
        try:
            tz_name = os.getenv('USER_TIMEZONE', 'America/New_York')
            return pytz.timezone(tz_name)
        except:
            return pytz.timezone('America/New_York')
    
    def get_alert_preferences(self) -> Dict:
        """Get current alert preferences (from memory)"""
        return self.preferences.copy()
    
    def save_alert_preferences(self, preferences: Dict) -> bool:
        """Save alert preferences (to memory for now)"""
        try:
            self.preferences.update(preferences)
            print(f"Preferences saved to memory: {preferences}")
            return True
        except Exception as e:
            print(f"Failed to save preferences: {e}")
            return False
    
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[Dict]:
        """Get upcoming calendar events within specified time window - FIXED event processing"""
        try:
            # Get today's and tomorrow's events
            today_events = list_today_events(max_results=50) or []
            tomorrow_events = list_tomorrow_events(max_results=50) or []
            
            all_events = today_events + tomorrow_events
            
            # Filter to events within the time window
            now = datetime.datetime.now(self.timezone)
            cutoff_time = now + datetime.timedelta(hours=hours_ahead)
            
            upcoming_events = []
            
            for event in all_events:
                try:
                    # FIXED: Handle both dict and string event formats
                    if isinstance(event, str):
                        print(f"Skipping string event: {event}")
                        continue
                    
                    if not isinstance(event, dict):
                        print(f"Skipping non-dict event: {type(event)}")
                        continue
                    
                    # Parse event start time with better error handling
                    start_info = event.get('start', {})
                    if not start_info:
                        print(f"Event missing start time: {event.get('summary', 'Unknown')}")
                        continue
                    
                    start_time_str = None
                    if isinstance(start_info, dict):
                        start_time_str = start_info.get('dateTime') or start_info.get('date')
                    elif isinstance(start_info, str):
                        start_time_str = start_info
                    
                    if not start_time_str:
                        print(f"Could not extract start time from: {start_info}")
                        continue
                    
                    start_time = date_parser.parse(start_time_str)
                    if start_time.tzinfo is None:
                        start_time = self.timezone.localize(start_time)
                    
                    # Check if event is within our time window
                    if now <= start_time <= cutoff_time:
                        # Parse end time safely
                        end_info = event.get('end', {})
                        end_time = start_time  # Default to start time
                        
                        if isinstance(end_info, dict):
                            end_time_str = end_info.get('dateTime') or end_info.get('date')
                            if end_time_str:
                                try:
                                    end_time = date_parser.parse(end_time_str)
                                except:
                                    end_time = start_time
                        
                        upcoming_events.append({
                            'id': event.get('id', f"event_{len(upcoming_events)}"),
                            'title': event.get('summary', 'Untitled Event'),
                            'start_time': start_time,
                            'end_time': end_time,
                            'location': event.get('location', ''),
                            'description': event.get('description', ''),
                            'attendees': event.get('attendees', []) or [],
                            'original_event': event
                        })
                        
                except Exception as e:
                    print(f"Error processing individual event: {e}")
                    print(f"Event data: {event}")
                    continue
            
            # Sort by start time
            upcoming_events.sort(key=lambda x: x['start_time'])
            print(f"Successfully processed {len(upcoming_events)} upcoming events")
            return upcoming_events
            
        except Exception as e:
            print(f"Failed to get upcoming events: {e}")
            return []
    
    def check_for_meeting_alerts(self) -> Dict:
        """Check for meetings that need alerts and send them"""
        if not self.bot or not self.chat_id:
            return {'success': False, 'error': 'Telegram not configured'}
        
        preferences = self.get_alert_preferences()
        if not preferences['meeting_alerts']['enabled']:
            return {'success': True, 'alerts_sent': 0, 'message': 'Meeting alerts disabled'}
        
        try:
            upcoming_events = self.get_upcoming_events(hours_ahead=2)  # Check next 2 hours
            alerts_sent = 0
            
            now = datetime.datetime.now(self.timezone)
            alert_times = preferences['meeting_alerts']['alert_times']
            
            for event in upcoming_events:
                event_start = event['start_time']
                
                # Check if we should send an alert for this event
                for alert_minutes in alert_times:
                    alert_time = event_start - datetime.timedelta(minutes=alert_minutes)
                    
                    # If we're within 2 minutes of the alert time
                    time_until_alert = (alert_time - now).total_seconds()
                    
                    if -120 <= time_until_alert <= 120:  # Within 2 minutes
                        alert_key = f"{event['id']}_{alert_minutes}"
                        
                        if alert_key not in self.sent_alerts:
                            success = self._send_meeting_alert(event, alert_minutes)
                            if success:
                                self.sent_alerts.add(alert_key)
                                alerts_sent += 1
                                print(f"Sent alert for {event['title']} ({alert_minutes} min before)")
            
            return {
                'success': True,
                'alerts_sent': alerts_sent,
                'events_checked': len(upcoming_events)
            }
            
        except Exception as e:
            print(f"Meeting alert check failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _send_meeting_alert(self, event: Dict, minutes_before: int) -> bool:
        """Send a Telegram alert for an upcoming meeting"""
        try:
            # Format the alert message
            start_time = event['start_time']
            formatted_time = start_time.strftime('%I:%M %p')
            
            message = f"📅 **Upcoming Meeting Alert**\n\n"
            message += f"**{event['title']}**\n"
            message += f"⏰ Starts in {minutes_before} minutes ({formatted_time})\n"
            
            if event['location']:
                message += f"📍 {event['location']}\n"
            
            if event['attendees']:
                attendee_names = []
                for att in event['attendees'][:3]:
                    if isinstance(att, dict):
                        name = att.get('displayName', att.get('email', 'Unknown'))
                    else:
                        name = str(att)
                    attendee_names.append(name)
                
                message += f"👥 With: {', '.join(attendee_names)}"
                if len(event['attendees']) > 3:
                    message += f" +{len(event['attendees'])-3} others"
                message += "\n"
            
            # Add quick action buttons
            message += f"\n🔕 Reply 'snooze' for 5min reminder"
            message += f"\n✅ Reply 'ready' to mark as acknowledged"
            
            result = self.bot.send_message(self.chat_id, message)
            return result.get('success', False)
            
        except Exception as e:
            print(f"Failed to send meeting alert: {e}")
            return False
    
    def send_daily_calendar_summary(self) -> Dict:
        """Send daily calendar summary via Telegram"""
        if not self.bot or not self.chat_id:
            return {'success': False, 'error': 'Telegram not configured'}
        
        preferences = self.get_alert_preferences()
        if not preferences['daily_summary']['enabled']:
            return {'success': True, 'message': 'Daily summary disabled'}
        
        try:
            # Get today's events
            today_events = list_today_events(max_results=20) or []
            
            # Get tomorrow's events if enabled
            tomorrow_events = []
            if preferences['daily_summary']['include_tomorrow']:
                tomorrow_events = list_tomorrow_events(max_results=10) or []
            
            # Format the summary message
            message = f"🗓️ **Daily Calendar Summary**\n"
            message += f"📅 {datetime.datetime.now().strftime('%A, %B %d, %Y')}\n\n"
            
            # Today's events with better parsing
            valid_today_events = []
            for event in today_events:
                if isinstance(event, dict) and event.get('summary'):
                    valid_today_events.append(event)
            
            if valid_today_events:
                message += f"**Today ({len(valid_today_events)} events):**\n"
                for event in valid_today_events[:8]:  # Limit to 8 events
                    start_time = event.get('start', {})
                    if isinstance(start_time, dict):
                        start_time_str = start_time.get('dateTime', '')
                    else:
                        start_time_str = str(start_time) if start_time else ''
                    
                    if start_time_str:
                        try:
                            dt = date_parser.parse(start_time_str)
                            time_str = dt.strftime('%I:%M %p')
                        except:
                            time_str = "All day"
                    else:
                        time_str = "All day"
                    
                    title = event.get('summary', 'Untitled Event')
                    message += f"• {time_str} - {title}\n"
                
                if len(valid_today_events) > 8:
                    message += f"... and {len(valid_today_events)-8} more events\n"
            else:
                message += "**Today:** No events scheduled\n"
            
            # Tomorrow's events with better parsing
            valid_tomorrow_events = []
            for event in tomorrow_events:
                if isinstance(event, dict) and event.get('summary'):
                    valid_tomorrow_events.append(event)
            
            if valid_tomorrow_events:
                message += f"\n**Tomorrow ({len(valid_tomorrow_events)} events):**\n"
                for event in valid_tomorrow_events[:5]:  # Limit to 5 events
                    start_time = event.get('start', {})
                    if isinstance(start_time, dict):
                        start_time_str = start_time.get('dateTime', '')
                    else:
                        start_time_str = str(start_time) if start_time else ''
                    
                    if start_time_str:
                        try:
                            dt = date_parser.parse(start_time_str)
                            time_str = dt.strftime('%I:%M %p')
                        except:
                            time_str = "All day"
                    else:
                        time_str = "All day"
                    
                    title = event.get('summary', 'Untitled Event')
                    message += f"• {time_str} - {title}\n"
                
                if len(valid_tomorrow_events) > 5:
                    message += f"... and {len(valid_tomorrow_events)-5} more events\n"
            
            # Add motivational footer
            total_events = len(valid_today_events) + len(valid_tomorrow_events)
            if total_events > 0:
                message += f"\n🎯 {total_events} total events ahead. Have a productive day!"
            else:
                message += f"\n😌 Light schedule ahead. Perfect for deep work!"
            
            result = self.bot.send_message(self.chat_id, message)
            return result
            
        except Exception as e:
            print(f"Failed to send daily summary: {e}")
            return {'success': False, 'error': str(e)}
    
    def enable_monitoring(self) -> Dict:
        """Enable calendar monitoring"""
        self.monitoring_enabled = True
        print("Calendar monitoring enabled (in-memory)")
        return {'success': True, 'message': 'Calendar monitoring enabled'}
    
    def disable_monitoring(self) -> Dict:
        """Disable calendar monitoring"""
        self.monitoring_enabled = False
        print("Calendar monitoring disabled (in-memory)")
        return {'success': True, 'message': 'Calendar monitoring disabled'}
    
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status and statistics"""
        try:
            # Get upcoming events count
            upcoming_events = self.get_upcoming_events(hours_ahead=24)
            
            return {
                'monitoring_enabled': self.monitoring_enabled,
                'last_check': datetime.datetime.now(),
                'recent_alerts_24h': len(self.sent_alerts),  # Approximate
                'upcoming_events_24h': len(upcoming_events),
                'telegram_configured': bool(self.bot and self.chat_id),
                'calendar_configured': True,  # Assuming it's configured if we got here
                'preferences': self.get_alert_preferences()
            }
                
        except Exception as e:
            print(f"Failed to get monitoring status: {e}")
            return {
                'monitoring_enabled': False,
                'error': str(e),
                'telegram_configured': bool(self.bot and self.chat_id),
                'calendar_configured': True
            }

# Background monitoring service
class CalendarTelegramMonitorHotfix:
    """Background service for calendar monitoring - HOTFIX version"""
    
    def __init__(self):
        self.alerts = CalendarTelegramAlertsHotfix()
        self.running = False
        self.thread = None
        self.check_interval = 120  # 2 minutes
        self.daily_summary_sent_today = False
    
    def start_monitoring(self):
        """Start the background monitoring service"""
        if self.running:
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        print("Calendar-Telegram monitoring started (HOTFIX version)")
        return True
    
    def stop_monitoring(self):
        """Stop the background monitoring service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Calendar-Telegram monitoring stopped")
        return True
    
    def _monitoring_loop(self):
        """Main monitoring loop with enhanced error handling"""
        consecutive_errors = 0
        max_errors = 5
        
        while self.running:
            try:
                current_time = datetime.datetime.now(self.alerts.timezone)
                
                # Check for meeting alerts
                result = self.alerts.check_for_meeting_alerts()
                if result.get('alerts_sent', 0) > 0:
                    print(f"Calendar monitoring: sent {result['alerts_sent']} alerts")
                
                # Check for daily summary (once per day)
                self._check_daily_summary()
                
                # Reset error counter on success
                consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                print(f"Calendar monitoring error #{consecutive_errors}: {e}")
                
                if consecutive_errors >= max_errors:
                    print(f"Too many consecutive errors, pausing for 10 minutes")
                    time.sleep(600)  # 10 minute pause
                    consecutive_errors = 0
            
            # Wait before next check
            time.sleep(self.check_interval)
    
    def _check_daily_summary(self):
        """Check if it's time to send daily summary"""
        preferences = self.alerts.get_alert_preferences()
        if not preferences['daily_summary']['enabled']:
            return
        
        current_time = datetime.datetime.now(self.alerts.timezone)
        summary_time = preferences['daily_summary']['time']
        
        try:
            # Parse summary time (format: "HH:MM")
            summary_hour, summary_minute = map(int, summary_time.split(':'))
            
            # Check if it's time to send summary
            if (current_time.hour == summary_hour and
                current_time.minute == summary_minute and
                not self.daily_summary_sent_today):
                
                result = self.alerts.send_daily_calendar_summary()
                if result.get('success'):
                    self.daily_summary_sent_today = True
                    print("Daily calendar summary sent successfully")
            
            # Reset daily flag at midnight
            if current_time.hour == 0 and current_time.minute == 0:
                self.daily_summary_sent_today = False
                
        except Exception as e:
            print(f"Daily summary check failed: {e}")

# Global monitor instance for HOTFIX
calendar_monitor_hotfix = CalendarTelegramMonitorHotfix()

def is_calendar_telegram_configured() -> bool:
    """Check if calendar-telegram integration is configured"""
    telegram_ok = bool(os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'))
    return telegram_ok

def process_calendar_telegram_command(user_input: str, project: str, use_voices: list, random_toggle: bool) -> Tuple[Dict, bool]:
    """Process calendar-telegram integration commands - HOTFIX version"""
    user_lower = user_input.lower().strip()
    
    if not is_calendar_telegram_configured():
        return {}, False
    
    # Calendar alert commands
    calendar_triggers = [
        'calendar alerts', 'meeting alerts', 'calendar notifications',
        'telegram calendar', 'calendar telegram', 'meeting reminders',
        'calendar summary', 'daily calendar', 'enable calendar alerts',
        'disable calendar alerts', 'calendar settings'
    ]
    
    if any(trigger in user_lower for trigger in calendar_triggers):
        try:
            alerts = CalendarTelegramAlertsHotfix()
            
            # Enable calendar alerts
            if any(phrase in user_lower for phrase in ['enable', 'start', 'turn on']):
                result = alerts.enable_monitoring()
                calendar_monitor_hotfix.start_monitoring()
                
                response_text = "📅 **Calendar Alerts Enabled!**\n\n"
                response_text += "✅ Meeting reminders activated\n"
                response_text += "✅ Daily calendar summaries enabled\n"
                response_text += "✅ Background monitoring started\n\n"
                response_text += "You'll receive Telegram notifications for:\n"
                response_text += "• Meetings 15 & 30 minutes before they start\n"
                response_text += "• Daily calendar summary at 7:00 AM\n\n"
                response_text += "**Note:** Using in-memory storage (database not available)\n"
                response_text += "Try: 'calendar settings' to view status"
            
            # Disable calendar alerts
            elif any(phrase in user_lower for phrase in ['disable', 'stop', 'turn off']):
                result = alerts.disable_monitoring()
                calendar_monitor_hotfix.stop_monitoring()
                
                response_text = "📅 **Calendar Alerts Disabled**\n\n"
                response_text += "🔕 Meeting reminders stopped\n"
                response_text += "🔕 Daily summaries paused\n"
                response_text += "🔕 Background monitoring stopped\n\n"
                response_text += "You can re-enable anytime with 'enable calendar alerts'"
            
            # Send daily summary now
            elif 'summary' in user_lower or 'daily' in user_lower:
                result = alerts.send_daily_calendar_summary()
                
                if result.get('success'):
                    response_text = "📅 **Daily Calendar Summary Sent!**\n\n"
                    response_text += "Check your Telegram for the complete schedule overview."
                else:
                    response_text = f"❌ Failed to send daily summary: {result.get('error', 'Unknown error')}"
            
            # Calendar settings/status
            elif 'settings' in user_lower or 'status' in user_lower:
                status = alerts.get_monitoring_status()
                
                response_text = "📅 **Calendar Alert Status**\n\n"
                response_text += f"**Monitoring:** {'✅ Enabled' if status['monitoring_enabled'] else '❌ Disabled'}\n"
                response_text += f"**Telegram:** {'✅ Connected' if status['telegram_configured'] else '❌ Not configured'}\n"
                response_text += f"**Calendar:** {'✅ Connected' if status['calendar_configured'] else '❌ Not configured'}\n"
                response_text += f"**Storage:** ⚠️ In-memory only (database not available)\n\n"
                
                if status.get('upcoming_events_24h') is not None:
                    response_text += f"**Activity:**\n"
                    response_text += f"• Upcoming events (24h): {status['upcoming_events_24h']}\n"
                    response_text += f"• Alerts sent (session): {status.get('recent_alerts_24h', 0)}\n\n"
                
                preferences = status.get('preferences', {})
                response_text += f"**Current Settings:**\n"
                response_text += f"• Meeting alerts: {'Enabled' if preferences.get('meeting_alerts', {}).get('enabled') else 'Disabled'}\n"
                response_text += f"• Daily summary: {'Enabled' if preferences.get('daily_summary', {}).get('enabled') else 'Disabled'}\n"
                response_text += f"• Summary time: {preferences.get('daily_summary', {}).get('time', '07:00')}\n"
                response_text += f"• Alert times: {preferences.get('meeting_alerts', {}).get('alert_times', [15, 30])} minutes before\n\n"
                response_text += "**Available Commands:**\n"
                response_text += "• 'enable calendar alerts' - Start monitoring\n"
                response_text += "• 'disable calendar alerts' - Stop monitoring\n"
                response_text += "• 'calendar summary' - Send summary now\n"
                response_text += "• 'test calendar alert' - Send test notification"
            
            # Test calendar alert
            elif 'test' in user_lower:
                test_event = {
                    'id': 'test_event',
                    'title': 'Test Meeting Alert',
                    'start_time': datetime.datetime.now(alerts.timezone) + datetime.timedelta(minutes=15),
                    'location': 'Test Location',
                    'attendees': []
                }
                
                success = alerts._send_meeting_alert(test_event, 15)
                
                if success:
                    response_text = "📅 **Test Alert Sent!**\n\n"
                    response_text += "Check your Telegram to see how meeting alerts will look."
                else:
                    response_text = "❌ **Test Alert Failed**\n\n"
                    response_text += "Check your Telegram configuration. Verify TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set correctly."
            
            else:
                # General calendar alerts info
                response_text = "📅 **Calendar → Telegram Integration**\n\n"
                response_text += "Get proactive calendar notifications via Telegram:\n\n"
                response_text += "**Available Commands:**\n"
                response_text += "• 'enable calendar alerts' - Start monitoring\n"
                response_text += "• 'calendar settings' - View current status\n"
                response_text += "• 'calendar summary' - Get daily overview\n"
                response_text += "• 'test calendar alert' - Send test notification\n\n"
                response_text += "**Features:**\n"
                response_text += "✅ Pre-meeting Telegram alerts (15 & 30 min)\n"
                response_text += "✅ Daily calendar summary at 7 AM\n"
                response_text += "✅ Smart duplicate prevention\n"
                response_text += "✅ Timezone-aware scheduling\n\n"
                response_text += "⚠️ **Note:** Database not available, using in-memory storage"
            
            response_data = {"SyntaxPrime": response_text}
            return response_data, True
            
        except Exception as e:
            response_text = f"❌ **Calendar Alert Error:**\n{str(e)}\n\n"
            response_text += "Check your Google Calendar and Telegram configuration.\n"
            response_text += "Also verify DATABASE_URL is set if you want persistent storage."
            response_data = {"SyntaxPrime": response_text}
            return response_data, True
    
    return {}, False

def start_calendar_monitoring():
    """Start calendar monitoring service - HOTFIX version"""
    if not is_calendar_telegram_configured():
        print("Calendar-Telegram integration not configured")
        return False
    
    return calendar_monitor_hotfix.start_monitoring()

def stop_calendar_monitoring():
    """Stop calendar monitoring service - HOTFIX version"""
    return calendar_monitor_hotfix.stop_monitoring()
