#!/usr/bin/env python3
"""
Detailed Gmail debugging - comprehensive email analysis
"""

import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pytz
import base64

# Allow localhost OAuth for testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

def detailed_email_debug():
    """Comprehensive Gmail debugging to understand email processing"""
    
    print("=== Detailed Gmail Debug ===\n")
    
    try:
        # Load credentials
        creds = Credentials.from_authorized_user_file('token.json')
        gmail_service = build('gmail', 'v1', credentials=creds)
        
        # Get user profile
        profile = gmail_service.users().getProfile(userId='me').execute()
        print(f"Gmail Profile: {profile.get('emailAddress')}")
        print(f"Total Messages: {profile.get('messagesTotal', 'Unknown')}")
        print(f"Total Threads: {profile.get('threadsTotal', 'Unknown')}")
        
        # Test different query types
        queries_to_test = [
            "in:inbox is:unread",
            "in:inbox newer_than:1d",
            "after:2025/09/06 in:inbox is:unread",  # Your app's overnight query
            "in:inbox OR in:sent newer_than:1d",    # Enhanced query from your app
            "in:inbox",  # Simple inbox query
            "is:unread"  # All unread
        ]
        
        for query in queries_to_test:
            print(f"\n🔍 Testing query: '{query}'")
            
            try:
                # Search with this query
                results = gmail_service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=10
                ).execute()
                
                messages = results.get('messages', [])
                print(f"   📊 Found {len(messages)} messages")
                
                if messages:
                    print(f"   📋 First few messages:")
                    
                    # Get details for first 3 messages
                    for i, msg in enumerate(messages[:3]):
                        try:
                            full_msg = gmail_service.users().messages().get(
                                userId='me',
                                id=msg['id'],
                                format='full'
                            ).execute()
                            
                            # Extract basic info
                            headers = full_msg['payload'].get('headers', [])
                            subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                            sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
                            date_str = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
                            
                            # Clean up sender
                            if '<' in sender and '>' in sender:
                                sender_name = sender.split('<')[0].strip()
                                sender_email = sender.split('<')[1].split('>')[0].strip()
                                sender_display = sender_name if sender_name else sender_email
                            else:
                                sender_display = sender
                            
                            # Truncate long subjects
                            if len(subject) > 60:
                                subject = subject[:60] + "..."
                            
                            print(f"      {i+1}. {sender_display}: {subject}")
                            print(f"         ID: {msg['id']}")
                            print(f"         Date: {date_str}")
                            print(f"         Labels: {full_msg.get('labelIds', [])}")
                            
                        except Exception as e:
                            print(f"      {i+1}. Failed to get message details: {e}")
                            
                else:
                    print(f"   📭 No messages found")
                    
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        # Test your app's specific overnight query in detail
        print(f"\n🎯 DETAILED ANALYSIS: Your app's overnight query")
        overnight_query = "after:2025/09/06 in:inbox is:unread"
        print(f"Query: {overnight_query}")
        
        try:
            results = gmail_service.users().messages().list(
                userId='me',
                q=overnight_query,
                maxResults=25  # Match your app's limit
            ).execute()
            
            messages = results.get('messages', [])
            print(f"Found {len(messages)} overnight messages")
            
            if messages:
                print(f"\nProcessing all {len(messages)} messages (like your app does):")
                
                processed_count = 0
                failed_count = 0
                
                for i, msg in enumerate(messages):
                    try:
                        full_msg = gmail_service.users().messages().get(
                            userId='me',
                            id=msg['id'],
                            format='full'
                        ).execute()
                        
                        # Extract info exactly like your app
                        headers = full_msg['payload'].get('headers', [])
                        subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
                        sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown Sender')
                        date_str = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
                        
                        # Check if this would be processed successfully
                        if sender and subject:
                            processed_count += 1
                            
                            # Show first 5 in detail
                            if i < 5:
                                print(f"   ✅ {i+1}. {sender[:30]}...: {subject[:40]}...")
                        else:
                            failed_count += 1
                            print(f"   ❌ {i+1}. Missing sender or subject")
                            
                    except Exception as e:
                        failed_count += 1
                        print(f"   ❌ {i+1}. Processing failed: {e}")
                
                print(f"\nSUMMARY:")
                print(f"   Total found: {len(messages)}")
                print(f"   Successfully processed: {processed_count}")
                print(f"   Processing failures: {failed_count}")
                print(f"   Success rate: {processed_count/len(messages)*100:.1f}%")
                
                if processed_count > 0:
                    print(f"   ✅ Your email processing should work correctly")
                else:
                    print(f"   ⚠️  Email processing has issues")
                    
        except Exception as e:
            print(f"Overnight query analysis failed: {e}")
        
        # Test specific email content extraction
        print(f"\n🔍 TESTING EMAIL CONTENT EXTRACTION")
        
        try:
            # Get one recent message for content analysis
            recent_results = gmail_service.users().messages().list(
                userId='me',
                q='in:inbox',
                maxResults=1
            ).execute()
            
            recent_messages = recent_results.get('messages', [])
            
            if recent_messages:
                msg_id = recent_messages[0]['id']
                print(f"Analyzing message ID: {msg_id}")
                
                full_msg = gmail_service.users().messages().get(
                    userId='me',
                    id=msg_id,
                    format='full'
                ).execute()
                
                # Show full message structure
                print(f"Message structure:")
                print(f"   ID: {full_msg.get('id')}")
                print(f"   Thread ID: {full_msg.get('threadId')}")
                print(f"   Label IDs: {full_msg.get('labelIds', [])}")
                print(f"   Snippet: {full_msg.get('snippet', 'No snippet')[:100]}...")
                
                # Show payload structure
                payload = full_msg.get('payload', {})
                print(f"   Payload MIME type: {payload.get('mimeType')}")
                print(f"   Payload parts: {len(payload.get('parts', []))}")
                
                # Show headers
                headers = payload.get('headers', [])
                print(f"   Headers count: {len(headers)}")
                
                important_headers = ['from', 'to', 'subject', 'date']
                for header_name in important_headers:
                    header_value = next((h['value'] for h in headers if h['name'].lower() == header_name), 'Not found')
                    print(f"   {header_name.title()}: {header_value[:100]}...")
                
        except Exception as e:
            print(f"Content extraction test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Email debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    detailed_email_debug()