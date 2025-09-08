#!/usr/bin/env python3
"""
Test multi-site Analytics configuration (SITE_1_NAME format)
"""

import os
import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def load_multi_site_config():
    """Load sites from SITE_X_NAME environment variables"""
    sites = {}
    
    site_index = 1
    while True:
        site_name = os.getenv(f'SITE_{site_index}_NAME')
        if not site_name:
            break
        
        analytics_view_id = os.getenv(f'SITE_{site_index}_ANALYTICS_VIEW_ID')
        search_console_url = os.getenv(f'SITE_{site_index}_SEARCH_CONSOLE_URL')
        aliases_str = os.getenv(f'SITE_{site_index}_ALIASES', '')
        aliases = [alias.strip() for alias in aliases_str.split(',') if alias.strip()]
        
        sites[site_name.lower().replace(' ', '_')] = {
            'name': site_name,
            'analytics_view_id': analytics_view_id,
            'search_console_url': search_console_url,
            'aliases': aliases
        }
        
        site_index += 1
    
    return sites

def test_multisite_analytics():
    """Test Analytics access for all configured sites"""
    
    # Load token
    token_path = os.getenv('GOOGLE_TOKEN_PATH', 'token.json')
    if not os.path.exists(token_path):
        print("❌ No token.json found")
        return False
    
    try:
        credentials = Credentials.from_authorized_user_file(token_path)
        if not credentials.valid:
            print("❌ Token is invalid")
            return False
        
        print("✅ Token loaded successfully")
        
        # Load multi-site configuration
        sites = load_multi_site_config()
        
        if not sites:
            print("❌ No sites found in environment variables")
            print("   Looking for SITE_1_NAME, SITE_2_NAME, etc.")
            return False
        
        print(f"\n📋 Found {len(sites)} configured sites:")
        for key, site in sites.items():
            print(f"   - {site['name']} (key: {key})")
            print(f"     Analytics ID: {site['analytics_view_id']}")
            print(f"     Search Console: {site['search_console_url']}")
            print(f"     Aliases: {', '.join(site['aliases'])}")
        
        # Initialize Analytics service
        print("\n🔍 Testing GA4 Analytics Data API...")
        try:
            analytics_service = build('analyticsdata', 'v1beta', credentials=credentials)
            print("✅ GA4 Analytics Data API connection successful")
        except Exception as e:
            print(f"❌ GA4 Analytics API connection failed: {e}")
            return False
        
        # Test each site's Analytics
        print("\n📊 Testing Analytics data for each site:")
        
        for key, site in sites.items():
            site_name = site['name']
            property_id = site['analytics_view_id']
            
            print(f"\n🔍 Testing {site_name} (Property ID: {property_id})...")
            
            if not property_id:
                print(f"   ⚠️  No Analytics Property ID configured")
                continue
            
            try:
                request_body = {
                    'dateRanges': [{'startDate': '7daysAgo', 'endDate': 'today'}],
                    'metrics': [
                        {'name': 'sessions'},
                        {'name': 'totalUsers'},
                        {'name': 'screenPageViews'}
                    ],
                    'dimensions': [{'name': 'date'}],
                    'orderBys': [{'dimension': {'dimensionName': 'date'}}]
                }
                
                response = analytics_service.properties().runReport(
                    property=f'properties/{property_id}',
                    body=request_body
                ).execute()
                
                rows = response.get('rows', [])
                print(f"   ✅ Success! Got {len(rows)} data points")
                
                if rows:
                    # Calculate totals
                    total_sessions = 0
                    total_users = 0
                    total_pageviews = 0
                    
                    for row in rows:
                        metrics = row.get('metricValues', [])
                        if len(metrics) >= 3:
                            total_sessions += int(metrics[0].get('value', '0'))
                            total_users += int(metrics[1].get('value', '0'))
                            total_pageviews += int(metrics[2].get('value', '0'))
                    
                    print(f"   📈 Last 7 days: {total_sessions:,} sessions, {total_users:,} users, {total_pageviews:,} pageviews")
                    
                    # Show recent data
                    print(f"   📅 Recent activity:")
                    for row in rows[-3:]:  # Last 3 days
                        date_value = row.get('dimensionValues', [{}])[0].get('value', 'Unknown')
                        sessions = row.get('metricValues', [{}])[0].get('value', '0')
                        print(f"      {date_value}: {sessions} sessions")
                else:
                    print(f"   ⚠️  No data available for this period")
                
            except Exception as e:
                print(f"   ❌ Analytics request failed: {e}")
                print(f"      This might mean:")
                print(f"      - Property ID {property_id} is incorrect")
                print(f"      - You don't have access to this property")
                print(f"      - Property might be using Universal Analytics (old format)")
        
        # Test Search Console
        print(f"\n🔍 Testing Search Console access...")
        try:
            searchconsole_service = build('searchconsole', 'v1', credentials=credentials)
            print("✅ Search Console API connection successful")
            
            # Test each site's Search Console
            for key, site in sites.items():
                site_name = site['name']
                site_url = site['search_console_url']
                
                if not site_url:
                    print(f"   ⚠️  {site_name}: No Search Console URL configured")
                    continue
                
                print(f"\n🔍 Testing Search Console for {site_name} ({site_url})...")
                
                try:
                    request_body = {
                        'startDate': '2024-01-01',  # Recent date range
                        'endDate': '2024-01-07',
                        'dimensions': ['query'],
                        'rowLimit': 5
                    }
                    
                    response = searchconsole_service.searchanalytics().query(
                        siteUrl=site_url,
                        body=request_body
                    ).execute()
                    
                    rows = response.get('rows', [])
                    print(f"   ✅ Success! Found {len(rows)} search queries")
                    
                    if rows:
                        print(f"   🔍 Top search queries:")
                        for i, row in enumerate(rows[:3], 1):
                            query = row['keys'][0]
                            clicks = row.get('clicks', 0)
                            impressions = row.get('impressions', 0)
                            print(f"      {i}. '{query}' - {clicks} clicks, {impressions} impressions")
                    
                except Exception as e:
                    print(f"   ❌ Search Console request failed: {e}")
                    print(f"      - Check if {site_url} is verified in Search Console")
                    print(f"      - URL format should be exactly as shown in Search Console")
                
        except Exception as e:
            print(f"❌ Search Console API connection failed: {e}")
        
        print(f"\n🎉 Testing complete!")
        print(f"\n📋 Ghostline Commands you can now use:")
        for key, site in sites.items():
            site_name = site['name']
            aliases = site['aliases']
            print(f"   - 'analytics for {site_name}'")
            if aliases:
                print(f"   - 'analytics for {aliases[0]}' (using alias)")
            print(f"   - 'search console for {site_name}'")
        
        print(f"\n   - 'all sites analytics'")
        print(f"   - 'list sites'")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-site analytics test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Multi-Site Analytics Configuration")
    print("=" * 60)
    test_multisite_analytics()