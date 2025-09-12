# universal_search_console_diagnostic.py - Works with both old and new config formats

import os
import json
import datetime
from typing import Dict, Any

def load_sites_configuration():
    """Load sites configuration from either JSON or legacy environment variables"""
    
    sites = {}
    config_method = "none"
    
    # Method 1: Try JSON configuration first
    sites_json = os.getenv('GOOGLE_SITES_CONFIG')
    if sites_json:
        try:
            sites = json.loads(sites_json)
            config_method = "json"
            print(f"✅ Using JSON configuration: {len(sites)} sites")
            return sites, config_method
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON config found but invalid: {e}")
    
    # Method 2: Try legacy individual environment variables
    # For your current setup - fixing the missing SITE_1_NAME
    known_sites = [
        {
            'index': 1,
            'name': 'BCDodge.me',  # You didn't export SITE_1_NAME, so we'll use this
            'analytics_view_id': os.getenv('SITE_1_ANALYTICS_VIEW_ID'),
            'search_console_url': os.getenv('SITE_1_SEARCH_CONSOLE_URL'),
            'aliases_str': os.getenv('SITE_1_ALIASES', '')
        },
        {
            'index': 2,
            'name': os.getenv('SITE_2_NAME'),
            'analytics_view_id': os.getenv('SITE_2_ANALYTICS_VIEW_ID'),
            'search_console_url': os.getenv('SITE_2_SEARCH_CONSOLE_URL'),
            'aliases_str': os.getenv('SITE_2_ALIASES', '')
        },
        {
            'index': 3,
            'name': os.getenv('SITE_3_NAME'),
            'analytics_view_id': os.getenv('SITE_3_ANALYTICS_VIEW_ID'),
            'search_console_url': os.getenv('SITE_3_SEARCH_CONSOLE_URL'),
            'aliases_str': os.getenv('SITE_3_ALIASES', '')
        },
        {
            'index': 4,
            'name': os.getenv('SITE_4_NAME'),
            'analytics_view_id': os.getenv('SITE_4_ANALYTICS_VIEW_ID'),
            'search_console_url': os.getenv('SITE_4_SEARCH_CONSOLE_URL'),
            'aliases_str': os.getenv('SITE_4_ALIASES', '')
        },
        {
            'index': 5,
            'name': os.getenv('SITE_5_NAME'),
            'analytics_view_id': os.getenv('SITE_5_ANALYTICS_VIEW_ID'),
            'search_console_url': os.getenv('SITE_5_SEARCH_CONSOLE_URL'),
            'aliases_str': os.getenv('SITE_5_ALIASES', '')
        }
    ]
    
    for site_info in known_sites:
        site_name = site_info['name']
        if not site_name:
            continue
            
        analytics_view_id = site_info['analytics_view_id']
        search_console_url = site_info['search_console_url']
        aliases_str = site_info['aliases_str']
        aliases = [alias.strip() for alias in aliases_str.split(',') if alias.strip()]
        
        if analytics_view_id or search_console_url:
            site_key = site_name.lower().replace(' ', '_').replace('&', 'and').replace('.', '_')
            sites[site_key] = {
                'name': site_name,
                'analytics_view_id': analytics_view_id,
                'search_console_url': search_console_url,
                'aliases': aliases
            }
    
    if sites:
        config_method = "legacy"
        print(f"✅ Using legacy configuration: {len(sites)} sites")
        print("💡 Consider converting to JSON format for better management")
    
    return sites, config_method

def diagnose_search_console_universal():
    """Universal diagnostic that works with both config formats"""
    
    print("🔍 UNIVERSAL SEARCH CONSOLE DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Load configuration (supports both formats)
    print("\n📋 STEP 1: Configuration Detection")
    print("-" * 40)
    
    sites_config, config_method = load_sites_configuration()
    
    if not sites_config:
        print("❌ No site configuration found")
        print("\n💡 Set up either:")
        print("   Option 1 (Recommended): GOOGLE_SITES_CONFIG as JSON")
        print("   Option 2 (Legacy): Individual SITE_X_ variables")
        return False
    
    print(f"✅ Configuration loaded via {config_method} method")
    print(f"📊 Found {len(sites_config)} sites configured")
    
    # Show what we found
    for site_key, site_config in sites_config.items():
        print(f"\n🔸 {site_config['name']} ({site_key}):")
        print(f"   Analytics: {site_config.get('analytics_view_id', 'Not configured')}")
        print(f"   Search Console: {site_config.get('search_console_url', 'Not configured')}")
        if site_config.get('aliases'):
            print(f"   Aliases: {', '.join(site_config['aliases'])}")
    
    # 2. Test Google Integration Class
    print(f"\n🔧 STEP 2: Google Integration Test")
    print("-" * 40)
    
    try:
        # Import and test integration
        import sys
        import importlib.util
        
        # Try to import the GoogleIntegration class
        try:
            from modules.enhanced_google_integration import GoogleIntegration
            google_integration = GoogleIntegration()
            print("✅ GoogleIntegration class imported and initialized")
        except ImportError as e:
            print(f"❌ Could not import GoogleIntegration: {e}")
            return False
        except Exception as e:
            print(f"❌ GoogleIntegration initialization failed: {e}")
            return False
        
        # Test if sites loaded correctly
        loaded_sites = google_integration.sites_config
        if loaded_sites:
            print(f"✅ Integration loaded {len(loaded_sites)} sites")
            
            # Check if loaded sites match our detected sites
            loaded_keys = set(loaded_sites.keys())
            detected_keys = set(sites_config.keys())
            
            if loaded_keys == detected_keys:
                print("✅ Site keys match between detection and integration")
            else:
                print("⚠️ Site key mismatch:")
                print(f"   Detected: {detected_keys}")
                print(f"   Loaded: {loaded_keys}")
        else:
            print("❌ No sites loaded in GoogleIntegration")
        
        # Test find_site_by_name
        test_site = list(sites_config.keys())[0] if sites_config else None
        if test_site:
            found_site = google_integration.find_site_by_name(test_site)
            if found_site:
                print(f"✅ find_site_by_name working: '{test_site}' → '{found_site}'")
            else:
                print(f"❌ find_site_by_name failed for '{test_site}'")
    
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False
    
    # 3. Test Search Console Methods (Dry Run)
    print(f"\n📡 STEP 3: Search Console Method Testing")
    print("-" * 40)
    
    for site_key, site_config in sites_config.items():
        search_console_url = site_config.get('search_console_url')
        if not search_console_url:
            print(f"⚠️ {site_config['name']}: No Search Console URL configured")
            continue
        
        print(f"\n🔸 Testing {site_config['name']} ({site_key})")
        
        try:
            # Test the fixed method
            result = google_integration.get_search_console_data_for_site(
                site_key, 
                start_date='2024-01-01', 
                end_date='2024-01-07'
            )
            
            # Check return type (this is where the tuple error would show)
            print(f"   📊 Return type: {type(result)}")
            
            if isinstance(result, tuple):
                print("   🚨 TUPLE ERROR DETECTED!")
                print(f"   📊 Tuple contents: {result}")
                print("   🔧 This means the fix hasn't been applied yet")
            elif isinstance(result, dict):
                print("   ✅ Returns dictionary correctly")
                
                success = result.get('success', False)
                print(f"   📈 API Success: {success}")
                
                if success:
                    data = result.get('data', [])
                    print(f"   📊 Data entries: {len(data) if isinstance(data, list) else 'Invalid type'}")
                else:
                    error = result.get('error', 'No error message')
                    print(f"   ❌ API Error: {error}")
            else:
                print(f"   ⚠️ Unexpected return type: {type(result)}")
                
        except Exception as e:
            print(f"   ❌ Method test exception: {e}")
    
    # 4. Test Command Parsing
    print(f"\n🎯 STEP 4: Command Parsing Test")  
    print("-" * 40)
    
    test_commands = []
    for site_key, site_config in sites_config.items():
        site_name = site_config['name']
        test_commands.extend([
            f"search console for {site_name.lower()}",
            f"seo for {site_key}"
        ])
        
        # Add alias tests
        for alias in site_config.get('aliases', [])[:1]:  # Test first alias
            test_commands.append(f"search console for {alias}")
    
    for command in test_commands[:6]:  # Test first 6 commands
        print(f"\n🔸 Testing: '{command}'")
        
        # Parse site from command (same logic as in the code)
        user_lower = command.lower().strip()
        site_key = None
        
        import re
        site_patterns = [
            r'search console for (.+?)(?:\s+last|\s+this|\s+from|\s+past|$)',
            r'seo for (.+?)(?:\s+last|\s+this|\s+from|\s+past|$)',
            r'(.+?) search console',
            r'(.+?) seo'
        ]
        
        for pattern in site_patterns:
            match = re.search(pattern, user_lower)
            if match:
                potential_site = match.group(1).strip()
                found_site = google_integration.find_site_by_name(potential_site)
                if found_site:
                    site_key = found_site
                    break
        
        if site_key:
            print(f"   ✅ Parsed site: '{site_key}'")
        else:
            print(f"   ❌ Could not parse site from command")
    
    # 5. Summary
    print(f"\n📋 DIAGNOSTIC SUMMARY")
    print("-" * 40)
    
    print(f"✅ Configuration: {config_method} format with {len(sites_config)} sites")
    print(f"✅ Integration: GoogleIntegration class working")
    
    # Check for specific issues
    tuple_error_found = False
    missing_urls = 0
    
    for site_key, site_config in sites_config.items():
        if not site_config.get('search_console_url'):
            missing_urls += 1
    
    if missing_urls > 0:
        print(f"⚠️ Configuration: {missing_urls} sites missing Search Console URLs")
    
    print(f"\n🔧 NEXT STEPS:")
    print("1. Apply the Section 6 fix to enhanced_google_integration.py")
    print("2. Test with: 'search console for [site name]' commands")
    print("3. Fix any remaining API authentication/permission issues")
    
    if config_method == "legacy":
        print("4. Consider converting to JSON configuration format")
    
    return True

if __name__ == "__main__":
    diagnose_search_console_universal()