#!/usr/bin/env python3
"""
Detailed calendar debugging - find out why we're missing events
"""

import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pytz

# Allow localhost OAuth for testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def detailed_calendar_debug():
    """Comprehensive calendar debugging to find missing events"""
    
    print("=== Detailed Calendar Debug ===\n")
    
    try:
        # Load credentials
        creds = Credentials.from_authorized_user_file('token.json')
        calendar_service = build('calendar', 'v3', credentials=creds)
        
        # Get all calendars first
        print("📅 Available Calendars:")
        calendar_list = calendar_service.calendarList().list().execute()
        calendars = calendar_list.get('items', [])
        
        for cal in calendars:
            cal_id = cal.get('id', 'No ID')
            cal_name = cal.get('summary', 'No name')
            is_primary = cal.get('primary', False)
            print(f"   {'[PRIMARY]' if is_primary else '[SECONDARY]'} {cal_name}")
            print(f"      ID: {cal_id}")
        
        # Calculate next Tuesday
        today = datetime.now()
        current_weekday = today.weekday()
        
        if current_weekday == 1:  # Today is Tuesday
            days_ahead = 7
        elif current_weekday < 1:
            days_ahead = 1 - current_weekday
        else:
            days_ahead = 7 - current_weekday + 1
        
        next_tuesday = today + timedelta(days=days_ahead)
        print(f"\n🎯 Target Date: {next_tuesday.strftime('%A, %B %d, %Y')}")
        
        # Test each calendar individually
        tz = pytz.timezone('America/New_York')
        start_time = tz.localize(next_tuesday.replace(hour=0, minute=0, second=0, microsecond=0))
        end_time = start_time + timedelta(days=1)
        
        print(f"🕐 Search Window: {start_time} to {end_time}")
        print(f"🌐 UTC Window: {start_time.astimezone(pytz.UTC)} to {end_time.astimezone(pytz.UTC)}")
        
        all_events = []
        
        for cal in calendars:
            cal_id = cal.get('id')
            cal_name = cal.get('summary', 'Unnamed')
            
            print(f"\n🔍 Checking calendar: {cal_name}")
            
            try:
                # Query this specific calendar
                events_result = calendar_service.events().list(
                    calendarId=cal_id,
                    timeMin=start_time.astimezone(pytz.UTC).isoformat(),
                    timeMax=end_time.astimezone(pytz.UTC).isoformat(),
                    maxResults=50,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
                
                events = events_result.get('items', [])
                print(f"   📊 Found {len(events)} events in this calendar")
                
                for event in events:
                    summary = event.get('summary', 'No title')
                    start = event.get('start', {})
                    event_id = event.get('id', 'No ID')
                    
                    # Parse event time
                    if 'dateTime' in start:
                        # Timed event
                        event_time = datetime.fromisoformat(start['dateTime'].replace('Z', '+00:00'))
                        local_time = event_time.astimezone(tz)
                        time_str = local_time.strftime('%I:%M %p')
                        event_type = "TIMED"
                    elif 'date' in start:
                        # All-day event
                        time_str = 'All day'
                        event_type = "ALL-DAY"
                    else:
                        time_str = 'Unknown time'
                        event_type = "UNKNOWN"
                    
                    print(f"      [{event_type}] {time_str}: {summary}")
                    print(f"         Event ID: {event_id}")
                    print(f"         Raw start: {start}")
                    
                    all_events.append({
                        'calendar': cal_name,
                        'calendar_id': cal_id,
                        'summary': summary,
                        'time_str': time_str,
                        'event_type': event_type,
                        'raw_start': start
                    })
                    
            except Exception as e:
                print(f"   ❌ Error accessing calendar: {e}")
        
        # Summary
        print(f"\n📋 SUMMARY:")
        print(f"   Total events found across all calendars: {len(all_events)}")
        
        if all_events:
            print(f"   Events by type:")
            timed_events = [e for e in all_events if e['event_type'] == 'TIMED']
            allday_events = [e for e in all_events if e['event_type'] == 'ALL-DAY']
            
            print(f"      Timed events: {len(timed_events)}")
            for event in timed_events:
                print(f"         {event['time_str']}: {event['summary']} (in {event['calendar']})")
            
            print(f"      All-day events: {len(allday_events)}")
            for event in allday_events:
                print(f"         {event['summary']} (in {event['calendar']})")
        
        # Test primary calendar only (what your app probably uses)
        print(f"\n🎯 Testing PRIMARY calendar only (what your app uses):")
        primary_events = calendar_service.events().list(
            calendarId='primary',
            timeMin=start_time.astimezone(pytz.UTC).isoformat(),
            timeMax=end_time.astimezone(pytz.UTC).isoformat(),
            maxResults=50,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        primary_event_list = primary_events.get('items', [])
        print(f"   📊 Primary calendar has {len(primary_event_list)} events")
        
        if len(primary_event_list) != len(all_events):
            print(f"   ⚠️  DISCREPANCY: Primary ({len(primary_event_list)}) vs All calendars ({len(all_events)})")
            print(f"   This suggests your events are in secondary calendars!")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    detailed_calendar_debug()