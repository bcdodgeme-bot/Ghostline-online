# modules/timezone_handler.py - Comprehensive timezone management

import os
import datetime
import pytz
from typing import Optional, Union, Dict, Any
from flask import request, session
import json

class TimezoneManager:
    """Centralized timezone management for the application"""
    
    def __init__(self):
        # Default timezone - can be overridden by user preference or detection
        self.default_timezone = os.getenv('DEFAULT_TIMEZONE', 'America/New_York')
        self.fallback_timezone = 'UTC'
        
        # Common timezone mappings for easier detection
        self.timezone_mappings = {
            # US Timezones
            'EST': 'America/New_York',
            'EDT': 'America/New_York', 
            'CST': 'America/Chicago',
            'CDT': 'America/Chicago',
            'MST': 'America/Denver',
            'MDT': 'America/Denver',
            'PST': 'America/Los_Angeles',
            'PDT': 'America/Los_Angeles',
            
            # Other common zones
            'GMT': 'Europe/London',
            'BST': 'Europe/London',
            'CET': 'Europe/Paris',
            'CEST': 'Europe/Paris',
            'JST': 'Asia/Tokyo',
            'AEST': 'Australia/Sydney',
            'AEDT': 'Australia/Sydney'
        }
    
    def get_user_timezone(self) -> pytz.BaseTzInfo:
        """Get user's timezone from various sources in order of preference"""
        
        # 1. Check session for explicitly set timezone
        if 'user_timezone' in session:
            try:
                return pytz.timezone(session['user_timezone'])
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"Invalid timezone in session: {session['user_timezone']}")
        
        # 2. Check for browser-detected timezone in session
        if 'detected_timezone' in session:
            try:
                return pytz.timezone(session['detected_timezone'])
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"Invalid detected timezone: {session['detected_timezone']}")
        
        # 3. Try to detect from HTTP headers (limited accuracy)
        detected_tz = self._detect_timezone_from_headers()
        if detected_tz:
            try:
                return pytz.timezone(detected_tz)
            except pytz.exceptions.UnknownTimeZoneError:
                print(f"Invalid detected timezone from headers: {detected_tz}")
        
        # 4. Use default timezone
        try:
            return pytz.timezone(self.default_timezone)
        except pytz.exceptions.UnknownTimeZoneError:
            print(f"Invalid default timezone: {self.default_timezone}")
            return pytz.timezone(self.fallback_timezone)
    
    def _detect_timezone_from_headers(self) -> Optional[str]:
        """Attempt timezone detection from HTTP headers (limited accuracy)"""
        
        # This is a basic implementation - browser JavaScript detection is more accurate
        if not request:
            return None
            
        # Check Accept-Language header for country hints
        accept_language = request.headers.get('Accept-Language', '')
        
        # Basic country to timezone mapping (rough approximation)
        country_tz_map = {
            'en-us': 'America/New_York',
            'en-gb': 'Europe/London',
            'fr-fr': 'Europe/Paris',
            'de-de': 'Europe/Berlin',
            'ja-jp': 'Asia/Tokyo',
            'zh-cn': 'Asia/Shanghai',
            'es-es': 'Europe/Madrid',
            'it-it': 'Europe/Rome',
            'pt-br': 'America/Sao_Paulo',
            'ru-ru': 'Europe/Moscow',
            'ko-kr': 'Asia/Seoul',
            'nl-nl': 'Europe/Amsterdam',
            'sv-se': 'Europe/Stockholm',
            'da-dk': 'Europe/Copenhagen',
            'no-no': 'Europe/Oslo',
            'fi-fi': 'Europe/Helsinki'
        }
        
        # Extract primary language-country code
        if accept_language:
            primary = accept_language.split(',')[0].lower().strip()
            return country_tz_map.get(primary)
        
        return None
    
    def set_user_timezone(self, timezone_name: str) -> bool:
        """Set user's timezone preference"""
        try:
            # Validate timezone
            pytz.timezone(timezone_name)
            session['user_timezone'] = timezone_name
            return True
        except pytz.exceptions.UnknownTimeZoneError:
            return False
    
    def set_detected_timezone(self, timezone_name: str) -> bool:
        """Set browser-detected timezone"""
        try:
            # Validate timezone
            pytz.timezone(timezone_name)
            session['detected_timezone'] = timezone_name
            return True
        except pytz.exceptions.UnknownTimeZoneError:
            return False
    
    def to_user_time(self, dt: Union[datetime.datetime, str], from_timezone: str = 'UTC') -> datetime.datetime:
        """Convert datetime to user's local timezone"""
        
        # Handle string input (ISO format)
        if isinstance(dt, str):
            try:
                # Try parsing ISO format with timezone
                if dt.endswith('Z'):
                    dt = datetime.datetime.fromisoformat(dt.replace('Z', '+00:00'))
                elif '+' in dt[-6:] or dt[-6:-3] in ['-05', '-06', '-07', '-08', '-04']:
                    dt = datetime.datetime.fromisoformat(dt)
                else:
                    # Assume UTC if no timezone info
                    dt = datetime.datetime.fromisoformat(dt)
                    dt = dt.replace(tzinfo=pytz.UTC)
            except ValueError:
                print(f"Could not parse datetime string: {dt}")
                return datetime.datetime.now(self.get_user_timezone())
        
        # If datetime is naive (no timezone), assume it's in from_timezone
        if dt.tzinfo is None:
            source_tz = pytz.timezone(from_timezone)
            dt = source_tz.localize(dt)
        
        # Convert to user's timezone
        user_tz = self.get_user_timezone()
        return dt.astimezone(user_tz)
    
    def to_utc(self, dt: Union[datetime.datetime, str], from_timezone: Optional[str] = None) -> datetime.datetime:
        """Convert datetime to UTC"""
        
        # Handle string input
        if isinstance(dt, str):
            try:
                if dt.endswith('Z'):
                    return datetime.datetime.fromisoformat(dt.replace('Z', '+00:00')).astimezone(pytz.UTC)
                elif '+' in dt[-6:] or dt[-6:-3] in ['-05', '-06', '-07', '-08', '-04']:
                    return datetime.datetime.fromisoformat(dt).astimezone(pytz.UTC)
                else:
                    dt = datetime.datetime.fromisoformat(dt)
            except ValueError:
                print(f"Could not parse datetime string: {dt}")
                return datetime.datetime.now(pytz.UTC)
        
        # If datetime is naive, localize it first
        if dt.tzinfo is None:
            source_tz = from_timezone or self.get_user_timezone().zone
            source_tz = pytz.timezone(source_tz)
            dt = source_tz.localize(dt)
        
        return dt.astimezone(pytz.UTC)
    
    def format_user_time(self, dt: Union[datetime.datetime, str], 
                        format_string: str = "%A, %B %d, %Y at %I:%M %p %Z",
                        from_timezone: str = 'UTC') -> str:
        """Format datetime in user's timezone with custom format"""
        
        user_dt = self.to_user_time(dt, from_timezone)
        return user_dt.strftime(format_string)
    
    def get_timezone_info(self) -> Dict[str, Any]:
        """Get current timezone information"""
        user_tz = self.get_user_timezone()
        now = datetime.datetime.now(user_tz)
        
        return {
            'timezone_name': user_tz.zone,
            'timezone_abbr': now.strftime('%Z'),
            'utc_offset': now.strftime('%z'),
            'is_dst': now.dst() != datetime.timedelta(0),
            'current_time': now.isoformat(),
            'formatted_time': now.strftime("%A, %B %d, %Y at %I:%M %p %Z")
        }
    
    def get_common_timezones(self) -> Dict[str, str]:
        """Get list of common timezones for user selection"""
        return {
            'US Eastern': 'America/New_York',
            'US Central': 'America/Chicago', 
            'US Mountain': 'America/Denver',
            'US Pacific': 'America/Los_Angeles',
            'US Alaska': 'America/Anchorage',
            'US Hawaii': 'Pacific/Honolulu',
            'UK': 'Europe/London',
            'France/Germany': 'Europe/Paris',
            'Japan': 'Asia/Tokyo',
            'Australia Sydney': 'Australia/Sydney',
            'China': 'Asia/Shanghai',
            'India': 'Asia/Kolkata',
            'Brazil': 'America/Sao_Paulo',
            'UTC': 'UTC'
        }

# Global timezone manager instance
timezone_manager = TimezoneManager()

# Convenience functions
def get_user_timezone():
    """Get user's timezone"""
    return timezone_manager.get_user_timezone()

def to_user_time(dt, from_timezone='UTC'):
    """Convert to user's local time"""
    return timezone_manager.to_user_time(dt, from_timezone)

def to_utc(dt, from_timezone=None):
    """Convert to UTC"""
    return timezone_manager.to_utc(dt, from_timezone)

def format_user_time(dt, format_string="%A, %B %d, %Y at %I:%M %p %Z", from_timezone='UTC'):
    """Format in user's timezone"""
    return timezone_manager.format_user_time(dt, format_string, from_timezone)

def now_user_time():
    """Get current time in user's timezone"""
    return datetime.datetime.now(get_user_timezone())

def today_user_date():
    """Get today's date in user's timezone"""
    return now_user_time().date()

def set_user_timezone(timezone_name):
    """Set user's preferred timezone"""
    return timezone_manager.set_user_timezone(timezone_name)

def get_timezone_info():
    """Get timezone information"""
    return timezone_manager.get_timezone_info()

# Template filters for Jinja2
def datetime_filter(dt, format_string="%A, %B %d, %Y at %I:%M %p %Z", from_timezone='UTC'):
    """Jinja2 filter for datetime formatting"""
    if not dt:
        return ''
    return format_user_time(dt, format_string, from_timezone)

def date_filter(dt, format_string="%A, %B %d, %Y", from_timezone='UTC'):
    """Jinja2 filter for date formatting"""
    if not dt:
        return ''
    return format_user_time(dt, format_string, from_timezone)

def time_filter(dt, format_string="%I:%M %p %Z", from_timezone='UTC'):
    """Jinja2 filter for time formatting"""
    if not dt:
        return ''
    return format_user_time(dt, format_string, from_timezone)