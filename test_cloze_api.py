#!/usr/bin/env python3
"""
Cloze API Terminal Test Script
Run this locally to explore your Cloze API capabilities before implementing in Ghostline
"""

import requests
import json
import os
from datetime import datetime, timedelta
from pprint import pprint

class ClozeAPITester:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('CLOZE_API_KEY')
        self.base_url = "https://api.cloze.com/v1"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        if not self.api_key:
            raise ValueError("CLOZE_API_KEY not found. Set it as environment variable or pass directly.")
    
    def test_connection(self):
        """Test basic API connectivity"""
        print("=" * 60)
        print("TESTING CLOZE API CONNECTION")
        print("=" * 60)
        
        try:
            response = requests.get(f"{self.base_url}/user/profile", headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                profile = response.json()
                print("✅ Connection successful!")
                print(f"User: {profile.get('name', 'Unknown')}")
                print(f"Email: {profile.get('email', 'Unknown')}")
                print(f"Company: {profile.get('company', 'Not set')}")
                return True
            else:
                print(f"❌ Connection failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def explore_user_activity(self, days_back=3):
        """Explore what user activity data is available"""
        print("\n" + "=" * 60)
        print(f"EXPLORING USER ACTIVITY (Last {days_back} days)")
        print("=" * 60)
        
        try:
            # Test different activity queries
            params = {'days': days_back}
            response = requests.get(f"{self.base_url}/user/activity", headers=self.headers, params=params, timeout=15)
            
            if response.status_code == 200:
                activity_data = response.json()
                
                print(f"✅ Activity data retrieved")
                print(f"Total activities: {len(activity_data.get('data', []))}")
                
                # Analyze activity types
                activities = activity_data.get('data', [])
                if activities:
                    activity_types = {}
                    for activity in activities:
                        activity_type = activity.get('type', 'unknown')
                        activity_types[activity_type] = activity_types.get(activity_type, 0) + 1
                    
                    print("\nActivity breakdown:")
                    for activity_type, count in activity_types.items():
                        print(f"  {activity_type}: {count}")
                    
                    # Show first few activities as examples
                    print(f"\nFirst 3 activities (structure):")
                    for i, activity in enumerate(activities[:3]):
                        print(f"\n--- Activity {i+1} ---")
                        # Remove potentially sensitive content for display
                        safe_activity = {k: v for k, v in activity.items() if k not in ['content', 'body', 'message']}
                        pprint(safe_activity)
                
                return activity_data
            else:
                print(f"❌ Activity query failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Activity query error: {e}")
            return None
    
    def explore_agenda_data(self):
        """Check what's available in the agenda/reminders"""
        print("\n" + "=" * 60)
        print("EXPLORING AGENDA/REMINDER DATA")
        print("=" * 60)
        
        # Try different agenda-related endpoints
        agenda_endpoints = [
            '/user/agenda',
            '/user/reminders', 
            '/user/todos',
            '/user/schedule'
        ]
        
        for endpoint in agenda_endpoints:
            try:
                print(f"\nTesting endpoint: {endpoint}")
                response = requests.get(f"{self.base_url}{endpoint}", headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ {endpoint} - Success")
                    print(f"   Data keys: {list(data.keys()) if isinstance(data, dict) else 'List with ' + str(len(data)) + ' items'}")
                    
                    # Show structure of first item if it's a list
                    if isinstance(data, list) and data:
                        print(f"   First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                    elif isinstance(data, dict) and data:
                        # Show some sample data
                        for key, value in list(data.items())[:3]:
                            if isinstance(value, (str, int, bool)):
                                print(f"   {key}: {value}")
                
                elif response.status_code == 404:
                    print(f"⚠️  {endpoint} - Not found (endpoint may not exist)")
                else:
                    print(f"❌ {endpoint} - Failed ({response.status_code})")
                    
            except Exception as e:
                print(f"❌ {endpoint} - Error: {e}")
    
    def explore_people_data(self):
        """Test people/contacts data access"""
        print("\n" + "=" * 60)
        print("EXPLORING PEOPLE/CONTACTS DATA")
        print("=" * 60)
        
        try:
            # Get people with stages
            response = requests.get(f"{self.base_url}/user/stages/people", 
                                  headers=self.headers, 
                                  params={'limit': 10}, 
                                  timeout=15)
            
            if response.status_code == 200:
                people_data = response.json()
                people = people_data.get('data', [])
                
                print(f"✅ People data retrieved")
                print(f"Total people (limited to 10): {len(people)}")
                
                if people:
                    # Show structure of first person
                    print(f"\nFirst person structure:")
                    first_person = people[0]
                    safe_person = {k: v for k, v in first_person.items() 
                                 if k not in ['email', 'phone', 'address']}  # Hide PII
                    pprint(safe_person)
                    
                    # Analyze stages
                    stages = {}
                    for person in people:
                        stage = person.get('stage', 'No Stage')
                        stages[stage] = stages.get(stage, 0) + 1
                    
                    print(f"\nStage distribution:")
                    for stage, count in stages.items():
                        print(f"  {stage}: {count}")
                
                return people_data
            else:
                print(f"❌ People query failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ People query error: {e}")
            return None
    
    def explore_project_data(self):
        """Test project/pipeline data access"""
        print("\n" + "=" * 60)
        print("EXPLORING PROJECT/PIPELINE DATA")
        print("=" * 60)
        
        try:
            # Try pipeline data query
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            params = {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
            
            response = requests.get(f"{self.base_url}/user/pipeline", 
                                  headers=self.headers, 
                                  params=params, 
                                  timeout=15)
            
            if response.status_code == 200:
                pipeline_data = response.json()
                projects = pipeline_data.get('data', [])
                
                print(f"✅ Pipeline data retrieved")
                print(f"Total projects: {len(projects)}")
                
                if projects:
                    print(f"\nFirst project structure:")
                    pprint(projects[0])
                    
                    # Analyze project stages
                    stages = {}
                    for project in projects:
                        stage = project.get('stage', 'No Stage')
                        stages[stage] = stages.get(stage, 0) + 1
                    
                    print(f"\nProject stage distribution:")
                    for stage, count in stages.items():
                        print(f"  {stage}: {count}")
                
                return pipeline_data
            else:
                print(f"❌ Pipeline query failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Pipeline query error: {e}")
            return None
    
    def test_email_opens(self):
        """Test email tracking data access"""
        print("\n" + "=" * 60)
        print("EXPLORING EMAIL OPENS/TRACKING DATA")
        print("=" * 60)
        
        try:
            params = {'days': 7, 'limit': 20}
            response = requests.get(f"{self.base_url}/messages/opens", 
                                  headers=self.headers, 
                                  params=params, 
                                  timeout=15)
            
            if response.status_code == 200:
                opens_data = response.json()
                opens = opens_data.get('data', [])
                
                print(f"✅ Email opens data retrieved")
                print(f"Recent opens (7 days): {len(opens)}")
                
                if opens:
                    print(f"\nFirst email open structure:")
                    first_open = opens[0]
                    safe_open = {k: v for k, v in first_open.items() 
                               if k not in ['subject', 'sender', 'recipient']}
                    pprint(safe_open)
                
                return opens_data
            else:
                print(f"❌ Email opens query failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Email opens query error: {e}")
            return None
    
    def test_api_capabilities_summary(self):
        """Run all tests and provide a summary"""
        print("\n" + "=" * 60)
        print("CLOZE API CAPABILITIES SUMMARY")
        print("=" * 60)
        
        capabilities = {
            'connection': self.test_connection(),
            'user_activity': self.explore_user_activity() is not None,
            'people_data': self.explore_people_data() is not None,
            'project_data': self.explore_project_data() is not None,
            'email_tracking': self.test_email_opens() is not None
        }
        
        # Test agenda separately as it might have different endpoints
        self.explore_agenda_data()
        
        print(f"\n" + "=" * 60)
        print("FINAL CAPABILITY ASSESSMENT")
        print("=" * 60)
        
        for capability, status in capabilities.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {capability.replace('_', ' ').title()}: {'Available' if status else 'Not Available/Failed'}")
        
        # Provide recommendations based on what's available
        print(f"\n📋 INTEGRATION RECOMMENDATIONS:")
        
        if capabilities['user_activity']:
            print("✅ Communication-driven task flow is possible (user activity data available)")
        
        if capabilities['people_data']:
            print("✅ Relationship-based prioritization is possible (people data with stages available)")
        
        if capabilities['project_data']:
            print("✅ Project synchronization is possible (pipeline data available)")
        
        if capabilities['email_tracking']:
            print("✅ Email engagement tracking is possible (opens data available)")
        
        working_capabilities = sum(capabilities.values())
        if working_capabilities >= 3:
            print(f"\n🚀 RECOMMENDATION: Proceed with Option 4 (Hybrid Intelligence Flow)")
            print("   You have sufficient API access for a comprehensive integration")
        elif working_capabilities >= 2:
            print(f"\n⚠️  RECOMMENDATION: Start with Option 1 or 3")
            print("   Some API limitations detected, but basic integration is possible")
        else:
            print(f"\n❌ RECOMMENDATION: Contact Cloze support")
            print("   Limited API access detected - may need additional permissions")

def main():
    """Main testing function"""
    print("CLOZE API TESTING SCRIPT")
    print("This will test your Cloze API access and explore available data")
    print("Make sure CLOZE_API_KEY is set as an environment variable")
    print("-" * 60)
    
    try:
        # Initialize tester
        tester = ClozeAPITester()
        
        # Run comprehensive tests
        tester.test_api_capabilities_summary()
        
        print(f"\n" + "=" * 60)
        print("TESTING COMPLETE")
        print("Review the results above to understand your API capabilities")
        print("=" * 60)
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nTo fix this:")
        print("1. Get your API key from Cloze")
        print("2. Run: export CLOZE_API_KEY='your-api-key-here'")
        print("3. Run this script again")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()