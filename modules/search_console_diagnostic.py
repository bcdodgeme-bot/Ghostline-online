# search_console_diagnostic.py - Complete diagnostic for tuple error and configuration issues

import os
import json
import datetime
from typing import Dict, Any

def diagnose_search_console_configuration():
    """Complete diagnostic for Search Console configuration and tuple error"""
    
    print("🔍 SEARCH CONSOLE DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # 1. Check Environment Configuration
    print("\n📋 STEP 1: Environment Configuration Check")
    print("-" * 40)
    
    sites_config_raw = os.getenv('GOOGLE_SITES_CONFIG')
    if not sites_config_raw:
        print("❌ GOOGLE_SITES_CONFIG not found in environment")
        return False
    
    try:
        sites_config = json.loads(sites_config_raw)
        print(f"✅ GOOGLE_SITES_CONFIG parsed successfully")
        print(f"   📊 Found {len(sites_config)} sites configured")
    except json.JSONDecodeError as e:
        print(f"❌ GOOGLE_SITES_CONFIG JSON parse error: {e}")
        return False
    
    # 2. Validate Each Site Configuration
    print("\n🏢 STEP 2: Individual Site Configuration")
    print("-" * 40)
    
    for site_key, site_config in sites_config.items():
        print(f"\n🔸 Site: {site_key}")
        print(f"   Name: {site_config.get('name', 'Missing name')}")
        
        # Check Search Console URL
        search_console_url = site_config.get('search_console_url')
        if search_console_url:
            print(f"   ✅ Search Console URL: {search_console_url}")
            
            # Validate URL format
            valid_prefixes = ['https://www.', 'https://', 'sc-domain:']
            is_valid_format = any(search_console_url.startswith(prefix) for prefix in valid_prefixes)
            if is_valid_format:
                print(f"      ✅ URL format is valid")
            else:
                print(f"      ⚠️  URL format might be incorrect")
                print(f"      💡 Should start with: https:// or sc-domain:")
        else:
            print(f"   ❌ No search_console_url configured")
            continue
        
        # Check Analytics configuration
        analytics_id = site_config.get('analytics_view_id')
        if analytics_id:
            print(f"   ✅ Analytics ID: {analytics_id}")
        else:
            print(f"   ⚠️  No analytics_view_id configured")
    
    # 3. Test Google Integration Class Initialization
    print("\n🔧 STEP 3: Google Integration Class Test")
    print("-" * 40)
    
    try:
        from modules.enhanced_google_integration import GoogleIntegration
        google_integration = GoogleIntegration()
        print("✅ GoogleIntegration class initialized successfully")
        
        # Test sites loading
        loaded_sites = google_integration.sites_config
        print(f"✅ Loaded sites in integration: {list(loaded_sites.keys())}")
        
        # Test find_site_by_name method
        test_site = list(loaded_sites.keys())[0] if loaded_sites else None
        if test_site:
            found_site = google_integration.find_site_by_name(test_site)
            if found_site:
                print(f"✅ find_site_by_name working: '{test_site}' → '{found_site}'")
            else:
                print(f"❌ find_site_by_name failed for '{test_site}'")
        
    except Exception as e:
        print(f"❌ GoogleIntegration initialization failed: {e}")
        return False
    
    # 4. Test Search Console Data Retrieval (DRY RUN)
    print("\n📡 STEP 4: Search Console API Test (Dry Run)")
    print("-" * 40)
    
    for site_key, site_config in sites_config.items():
        search_console_url = site_config.get('search_console_url')
        if not search_console_url:
            continue
            
        print(f"\n🔸 Testing {site_config['name']} ({site_key})")
        
        try:
            # Test get_search_console_data_for_site method
            result = google_integration.get_search_console_data_for_site(
                site_key, 
                start_date='2024-01-01', 
                end_date='2024-01-07'
            )
            
            # CRITICAL: Check return type
            print(f"   📊 Return type: {type(result)}")
            print(f"   📊 Is dictionary: {isinstance(result, dict)}")
            
            if isinstance(result, dict):
                print(f"   ✅ Returned dictionary correctly")
                print(f"   📈 Success: {result.get('success', 'Unknown')}")
                
                if result.get('success'):
                    data = result.get('data', [])
                    print(f"   📊 Data type: {type(data)}")
                    print(f"   📊 Data length: {len(data) if isinstance(data, (list, dict)) else 'N/A'}")
                    
                    if isinstance(data, list) and data:
                        sample_row = data[0]
                        print(f"   📊 Sample row type: {type(sample_row)}")
                        if isinstance(sample_row, dict):
                            print(f"   📊 Sample row keys: {list(sample_row.keys())}")
                        else:
                            print(f"   ⚠️  Sample row is not a dictionary: {sample_row}")
                else:
                    error_msg = result.get('error', 'No error message')
                    print(f"   ❌ API call failed: {error_msg}")
            else:
                print(f"   🚨 CRITICAL: Method returned {type(result)} instead of dictionary!")
                print(f"   📊 Actual return value: {result}")
                # This is where the tuple error would occur
            
        except Exception as e:
            print(f"   ❌ Exception during test: {e}")
            print(f"   📊 Exception type: {type(e)}")
    
    # 5. Test Command Handler
    print("\n🎯 STEP 5: Command Handler Test")
    print("-" * 40)
    
    test_commands = [
        "search console for meals n feelz",
        "search console for bcdodge",
        "seo for tv signals",
        "search console for rose and angel"
    ]
    
    for command in test_commands:
        print(f"\n🔸 Testing command: '{command}'")
        
        try:
            # Parse which site this should match
            user_lower = command.lower().strip()
            site_key = None
            
            # Look for site specification patterns (same as in the code)
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
                
                # Test the actual command handler (but don't save to avoid side effects)
                try:
                    # This is where the tuple error would occur
                    result = google_integration.get_search_console_data_for_site(site_key)
                    print(f"   📊 Handler result type: {type(result)}")
                    
                    if isinstance(result, tuple):
                        print(f"   🚨 FOUND THE TUPLE ERROR!")
                        print(f"   📊 Tuple contents: {result}")
                        print(f"   📊 Tuple length: {len(result)}")
                    elif isinstance(result, dict):
                        print(f"   ✅ Handler returned dictionary correctly")
                    else:
                        print(f"   ⚠️  Handler returned unexpected type: {type(result)}")
                        
                except Exception as e:
                    print(f"   ❌ Handler exception: {e}")
            else:
                print(f"   ❌ Could not parse site from command")
                
        except Exception as e:
            print(f"   ❌ Command test exception: {e}")
    
    # 6. Summary and Recommendations
    print("\n📋 STEP 6: Summary and Recommendations")
    print("-" * 40)
    
    print("\n✅ Configuration Status Summary:")
    for site_key, site_config in sites_config.items():
        has_search_console = bool(site_config.get('search_console_url'))
        has_analytics = bool(site_config.get('analytics_view_id'))
        
        status = []
        if has_search_console:
            status.append("Search Console")
        if has_analytics:
            status.append("Analytics")
        
        status_str = " + ".join(status) if status else "Not configured"
        print(f"   🔸 {site_config['name']}: {status_str}")
    
    print("\n🔧 Required Fixes:")
    print("   1. Replace handle_multi_site_search_console_command method")
    print("   2. Replace get_search_console_data_for_site method") 
    print("   3. Fix Rose and Angel Analytics property ID")
    print("   4. Test all sites after fixes applied")
    
    print("\n💡 Next Steps:")
    print("   1. Apply the fixed methods from the artifact")
    print("   2. Update Rose and Angel configuration")
    print("   3. Run 'search console for [site]' commands to test")
    print("   4. Verify blog suggestions work correctly with validation")
    
    return True

if __name__ == "__main__":
    diagnose_search_console_configuration()