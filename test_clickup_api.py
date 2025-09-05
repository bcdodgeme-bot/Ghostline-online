#!/usr/bin/env python3
"""
ClickUp API Terminal Test Script
Run this locally to explore your ClickUp API capabilities before implementing data pipeline
"""

import requests
import json
import os
from datetime import datetime, timedelta
from pprint import pprint

class ClickUpAPITester:
    def __init__(self, api_token=None):
        self.api_token = api_token or os.getenv('CLICKUP_API_TOKEN')
        self.base_url = "https://api.clickup.com/api/v2"
        self.headers = {
            "Authorization": self.api_token,
            "Content-Type": "application/json"
        }
        
        if not self.api_token:
            raise ValueError("CLICKUP_API_TOKEN not found. Set it as environment variable or pass directly.")
    
    def test_connection(self):
        """Test basic API connectivity and get user info"""
        print("=" * 60)
        print("TESTING CLICKUP API CONNECTION")
        print("=" * 60)
        
        try:
            response = requests.get(f"{self.base_url}/user", headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                user_data = response.json()
                user = user_data.get('user', {})
                print("✅ Connection successful!")
                print(f"User: {user.get('username', 'Unknown')}")
                print(f"Email: {user.get('email', 'Unknown')}")
                print(f"ID: {user.get('id', 'Unknown')}")
                return user_data
            else:
                print(f"❌ Connection failed: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None
    
    def explore_workspace_structure(self):
        """Map out your complete ClickUp workspace structure"""
        print("\n" + "=" * 60)
        print("EXPLORING CLICKUP WORKSPACE STRUCTURE")
        print("=" * 60)
        
        workspace_tree = {}
        
        try:
            # Get teams (workspaces)
            teams_response = requests.get(f"{self.base_url}/team", headers=self.headers, timeout=15)
            
            if teams_response.status_code == 200:
                teams_data = teams_response.json()
                teams = teams_data.get('teams', [])
                print(f"✅ Found {len(teams)} team(s)/workspace(s)")
                
                for team in teams:
                    team_name = team.get('name', 'Unnamed Team')
                    team_id = team.get('id')
                    print(f"\n📁 TEAM: {team_name} (ID: {team_id})")
                    
                    workspace_tree[team_id] = {
                        'name': team_name,
                        'spaces': {}
                    }
                    
                    # Get spaces for this team
                    try:
                        spaces_response = requests.get(f"{self.base_url}/team/{team_id}/space", 
                                                     headers=self.headers, timeout=10)
                        
                        if spaces_response.status_code == 200:
                            spaces_data = spaces_response.json()
                            spaces = spaces_data.get('spaces', [])
                            print(f"   📂 {len(spaces)} space(s)")
                            
                            for space in spaces:
                                space_name = space.get('name', 'Unnamed Space')
                                space_id = space.get('id')
                                print(f"      📂 SPACE: {space_name} (ID: {space_id})")
                                
                                workspace_tree[team_id]['spaces'][space_id] = {
                                    'name': space_name,
                                    'lists': {}
                                }
                                
                                # Get lists for this space
                                try:
                                    lists_response = requests.get(f"{self.base_url}/space/{space_id}/list", 
                                                                headers=self.headers, timeout=10)
                                    
                                    if lists_response.status_code == 200:
                                        lists_data = lists_response.json()
                                        lists = lists_data.get('lists', [])
                                        print(f"         📋 {len(lists)} list(s)")
                                        
                                        for list_item in lists:
                                            list_name = list_item.get('name', 'Unnamed List')
                                            list_id = list_item.get('id')
                                            print(f"            📋 LIST: {list_name} (ID: {list_id})")
                                            
                                            workspace_tree[team_id]['spaces'][space_id]['lists'][list_id] = {
                                                'name': list_name,
                                                'task_count': list_item.get('task_count', 0)
                                            }
                                    
                                except Exception as e:
                                    print(f"         ❌ Error getting lists for space {space_name}: {e}")
                        
                    except Exception as e:
                        print(f"   ❌ Error getting spaces for team {team_name}: {e}")
                
                return workspace_tree
            else:
                print(f"❌ Teams query failed: {teams_response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Workspace exploration error: {e}")
            return None
    
    def test_task_operations(self, list_id=None):
        """Test task creation, retrieval, and updates"""
        print("\n" + "=" * 60)
        print("TESTING TASK OPERATIONS")
        print("=" * 60)
        
        if not list_id:
            print("⚠️  No list ID provided - will try to find one automatically")
            workspace = self.explore_workspace_structure()
            if workspace:
                # Try to find any list ID
                for team_id, team_data in workspace.items():
                    for space_id, space_data in team_data['spaces'].items():
                        for list_id_found, list_data in space_data['lists'].items():
                            list_id = list_id_found
                            print(f"📋 Using list: {list_data['name']} (ID: {list_id})")
                            break
                        if list_id:
                            break
                    if list_id:
                        break
        
        if not list_id:
            print("❌ No list ID available for testing")
            return None
        
        task_operations_results = {}
        
        # Test 1: Get existing tasks
        try:
            print(f"\n📋 Getting existing tasks from list {list_id}...")
            tasks_response = requests.get(f"{self.base_url}/list/{list_id}/task", 
                                        headers=self.headers, timeout=15)
            
            if tasks_response.status_code == 200:
                tasks_data = tasks_response.json()
                tasks = tasks_data.get('tasks', [])
                print(f"✅ Found {len(tasks)} existing tasks")
                
                if tasks:
                    print("   First task structure:")
                    first_task = tasks[0]
                    # Show key fields without sensitive data
                    safe_task = {k: v for k, v in first_task.items() 
                               if k in ['id', 'name', 'status', 'priority', 'due_date', 'assignees']}
                    pprint(safe_task)
                
                task_operations_results['get_tasks'] = True
            else:
                print(f"❌ Failed to get tasks: {tasks_response.status_code}")
                task_operations_results['get_tasks'] = False
                
        except Exception as e:
            print(f"❌ Error getting tasks: {e}")
            task_operations_results['get_tasks'] = False
        
        # Test 2: Create a test task
        try:
            print(f"\n➕ Creating test task...")
            test_task_data = {
                "name": f"API Test Task - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "description": "Created by ClickUp API test script - safe to delete",
                "priority": 3,
                "status": "to do"
            }
            
            create_response = requests.post(f"{self.base_url}/list/{list_id}/task", 
                                          headers=self.headers, 
                                          json=test_task_data, 
                                          timeout=15)
            
            if create_response.status_code == 200:
                created_task = create_response.json()
                task_id = created_task.get('id')
                print(f"✅ Test task created successfully!")
                print(f"   Task ID: {task_id}")
                print(f"   Task URL: {created_task.get('url', 'N/A')}")
                
                task_operations_results['create_task'] = {
                    'success': True,
                    'task_id': task_id,
                    'task_data': created_task
                }
                
                # Test 3: Update the test task
                try:
                    print(f"\n✏️  Updating test task...")
                    update_data = {
                        "name": f"API Test Task - Updated {datetime.now().strftime('%H:%M')}",
                        "description": "Updated by API test script",
                        "status": "in progress"
                    }
                    
                    update_response = requests.put(f"{self.base_url}/task/{task_id}", 
                                                 headers=self.headers, 
                                                 json=update_data, 
                                                 timeout=10)
                    
                    if update_response.status_code == 200:
                        print("✅ Task updated successfully!")
                        task_operations_results['update_task'] = True
                    else:
                        print(f"❌ Task update failed: {update_response.status_code}")
                        task_operations_results['update_task'] = False
                        
                except Exception as e:
                    print(f"❌ Error updating task: {e}")
                    task_operations_results['update_task'] = False
                
                # Test 4: Delete the test task (cleanup)
                try:
                    print(f"\n🗑️  Cleaning up test task...")
                    delete_response = requests.delete(f"{self.base_url}/task/{task_id}", 
                                                     headers=self.headers, 
                                                     timeout=10)
                    
                    if delete_response.status_code == 200:
                        print("✅ Test task deleted successfully!")
                        task_operations_results['delete_task'] = True
                    else:
                        print(f"⚠️  Test task deletion failed: {delete_response.status_code}")
                        print("   You may need to manually delete the test task")
                        task_operations_results['delete_task'] = False
                        
                except Exception as e:
                    print(f"❌ Error deleting task: {e}")
                    task_operations_results['delete_task'] = False
                
            else:
                print(f"❌ Task creation failed: {create_response.status_code}")
                print(f"Response: {create_response.text}")
                task_operations_results['create_task'] = {'success': False}
                
        except Exception as e:
            print(f"❌ Error creating task: {e}")
            task_operations_results['create_task'] = {'success': False}
        
        return task_operations_results
    
    def test_time_tracking(self, team_id=None):
        """Test time tracking capabilities"""
        print("\n" + "=" * 60)
        print("TESTING TIME TRACKING CAPABILITIES")
        print("=" * 60)
        
        if not team_id:
            # Try to get a team ID
            try:
                teams_response = requests.get(f"{self.base_url}/team", headers=self.headers, timeout=10)
                if teams_response.status_code == 200:
                    teams = teams_response.json().get('teams', [])
                    if teams:
                        team_id = teams[0]['id']
                        print(f"📊 Using team: {teams[0]['name']} (ID: {team_id})")
            except:
                pass
        
        if not team_id:
            print("❌ No team ID available for time tracking test")
            return None
        
        try:
            # Get time entries for the last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            params = {
                'start_date': int(start_date.timestamp() * 1000),
                'end_date': int(end_date.timestamp() * 1000)
            }
            
            time_response = requests.get(f"{self.base_url}/team/{team_id}/time_entries", 
                                       headers=self.headers, 
                                       params=params, 
                                       timeout=15)
            
            if time_response.status_code == 200:
                time_data = time_response.json()
                entries = time_data.get('data', [])
                print(f"✅ Time tracking data retrieved")
                print(f"   Time entries (last 7 days): {len(entries)}")
                
                if entries:
                    total_time = sum(int(entry.get('duration', 0)) for entry in entries)
                    hours = total_time / (1000 * 60 * 60)  # Convert ms to hours
                    print(f"   Total time tracked: {hours:.2f} hours")
                    
                    print(f"\n   First time entry structure:")
                    first_entry = entries[0]
                    safe_entry = {k: v for k, v in first_entry.items() 
                                if k in ['id', 'duration', 'start', 'end', 'task', 'user']}
                    pprint(safe_entry)
                
                return time_data
            else:
                print(f"❌ Time tracking query failed: {time_response.status_code}")
                print(f"Response: {time_response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Time tracking error: {e}")
            return None
    
    def test_webhooks_capability(self, team_id=None):
        """Test webhook setup capabilities"""
        print("\n" + "=" * 60)
        print("TESTING WEBHOOK CAPABILITIES")
        print("=" * 60)
        
        if not team_id:
            # Try to get a team ID
            try:
                teams_response = requests.get(f"{self.base_url}/team", headers=self.headers, timeout=10)
                if teams_response.status_code == 200:
                    teams = teams_response.json().get('teams', [])
                    if teams:
                        team_id = teams[0]['id']
            except:
                pass
        
        if not team_id:
            print("❌ No team ID available for webhook test")
            return None
        
        try:
            # Get existing webhooks
            webhooks_response = requests.get(f"{self.base_url}/team/{team_id}/webhook", 
                                           headers=self.headers, 
                                           timeout=10)
            
            if webhooks_response.status_code == 200:
                webhooks_data = webhooks_response.json()
                webhooks = webhooks_data.get('webhooks', [])
                print(f"✅ Webhook API accessible")
                print(f"   Existing webhooks: {len(webhooks)}")
                
                if webhooks:
                    print(f"\n   Webhook structure:")
                    first_webhook = webhooks[0]
                    safe_webhook = {k: v for k, v in first_webhook.items() 
                                  if k in ['id', 'events', 'endpoint', 'status']}
                    pprint(safe_webhook)
                
                return webhooks_data
            else:
                print(f"❌ Webhook query failed: {webhooks_response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Webhook error: {e}")
            return None
    
    def generate_pipeline_capabilities_report(self):
        """Run all tests and provide integration capabilities assessment"""
        print("\n" + "=" * 60)
        print("CLICKUP API PIPELINE CAPABILITIES ASSESSMENT")
        print("=" * 60)
        
        results = {}
        
        # Test basic connection
        user_data = self.test_connection()
        results['connection'] = user_data is not None
        
        # Map workspace structure
        workspace = self.explore_workspace_structure()
        results['workspace_access'] = workspace is not None
        
        # Test task operations
        task_ops = self.test_task_operations()
        results['task_operations'] = task_ops is not None and task_ops.get('create_task', {}).get('success', False)
        
        # Test time tracking
        time_data = self.test_time_tracking()
        results['time_tracking'] = time_data is not None
        
        # Test webhooks
        webhook_data = self.test_webhooks_capability()
        results['webhooks'] = webhook_data is not None
        
        # Generate integration assessment
        print(f"\n" + "=" * 60)
        print("INTEGRATION PIPELINE ASSESSMENT")
        print("=" * 60)
        
        for capability, status in results.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {capability.replace('_', ' ').title()}: {'Available' if status else 'Limited/Failed'}")
        
        # Provide data pipeline recommendations
        print(f"\n📊 DATA PIPELINE CAPABILITIES:")
        
        if results['task_operations']:
            print("✅ Bi-directional task sync possible (create/read/update/delete)")
        
        if results['time_tracking']:
            print("✅ Time tracking data integration possible")
        
        if results['webhooks']:
            print("✅ Real-time webhooks available for instant sync")
        
        if results['workspace_access']:
            print("✅ Full workspace structure mapping possible")
        
        # Generate specific recommendations for Cloze integration
        working_capabilities = sum(results.values())
        
        print(f"\n🔗 CLOZE INTEGRATION RECOMMENDATIONS:")
        
        if working_capabilities >= 4:
            print("🚀 FULL PIPELINE READY:")
            print("   • Cloze → ClickUp task automation")
            print("   • ClickUp → Cloze outcome tracking")
            print("   • Real-time sync via webhooks")
            print("   • Time tracking correlation with relationship data")
            
        elif working_capabilities >= 3:
            print("⚡ STRONG PIPELINE POSSIBLE:")
            print("   • Core task automation available")
            print("   • Manual sync methods required")
            print("   • Most integration features achievable")
            
        elif working_capabilities >= 2:
            print("⚠️  BASIC PIPELINE ONLY:")
            print("   • Limited automation possible")
            print("   • Focus on simple task creation from Cloze data")
            
        else:
            print("❌ PIPELINE NOT VIABLE:")
            print("   • API access too limited")
            print("   • Contact ClickUp support for API permissions")
        
        # Provide workspace configuration recommendations
        if workspace:
            print(f"\n⚙️  WORKSPACE CONFIGURATION SUGGESTIONS:")
            total_lists = 0
            suggested_list = None
            
            for team_id, team_data in workspace.items():
                for space_id, space_data in team_data['spaces'].items():
                    for list_id, list_data in space_data['lists'].items():
                        total_lists += 1
                        if not suggested_list and 'inbox' in list_data['name'].lower():
                            suggested_list = (list_id, list_data['name'])
                        elif not suggested_list and 'task' in list_data['name'].lower():
                            suggested_list = (list_id, list_data['name'])
            
            print(f"   Total lists available: {total_lists}")
            
            if suggested_list:
                print(f"   Recommended list for Cloze integration: {suggested_list[1]} ({suggested_list[0]})")
                print(f"   Add this to your environment: CLICKUP_DEFAULT_LIST_ID={suggested_list[0]}")
            
            if total_lists > 10:
                print("   Consider creating a dedicated 'Cloze Actions' list for clarity")
        
        return results

def main():
    """Main testing function"""
    print("CLICKUP API TESTING SCRIPT")
    print("This will test your ClickUp API access and explore data pipeline capabilities")
    print("Make sure CLICKUP_API_TOKEN is set as an environment variable")
    print("-" * 60)
    
    try:
        # Initialize tester
        tester = ClickUpAPITester()
        
        # Run comprehensive pipeline assessment
        results = tester.generate_pipeline_capabilities_report()
        
        print(f"\n" + "=" * 60)
        print("CLICKUP PIPELINE TESTING COMPLETE")
        print("Save this output to compare with your Cloze API test results")
        print("=" * 60)
        
        return results
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\nTo fix this:")
        print("1. Get your ClickUp API token from ClickUp Settings > Apps > API")
        print("2. Run: export CLICKUP_API_TOKEN='your-api-token-here'")
        print("3. Run this script again")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()