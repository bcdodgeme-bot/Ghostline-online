# config_converter.py - Convert old individual env vars to new JSON format

import os
import json

def convert_legacy_env_to_json():
    """Convert legacy SITE_X_ environment variables to GOOGLE_SITES_CONFIG JSON"""
    
    print("🔄 CONVERTING LEGACY ENVIRONMENT VARIABLES")
    print("=" * 50)
    
    sites = {}
    site_index = 1
    
    # Scan for legacy environment variables
    while True:
        site_name = os.getenv(f'SITE_{site_index}_NAME')
        if not site_name:
            break
        
        analytics_view_id = os.getenv(f'SITE_{site_index}_ANALYTICS_VIEW_ID')
        search_console_url = os.getenv(f'SITE_{site_index}_SEARCH_CONSOLE_URL')
        aliases_str = os.getenv(f'SITE_{site_index}_ALIASES', '')
        aliases = [alias.strip() for alias in aliases_str.split(',') if alias.strip()]
        
        if analytics_view_id or search_console_url:
            site_key = site_name.lower().replace(' ', '_').replace('&', 'and')
            sites[site_key] = {
                'name': site_name,
                'analytics_view_id': analytics_view_id,
                'search_console_url': search_console_url,
                'aliases': aliases
            }
            
            print(f"✅ Found Site {site_index}: {site_name}")
            print(f"   Analytics: {analytics_view_id or 'Not configured'}")
            print(f"   Search Console: {search_console_url or 'Not configured'}")
            print(f"   Aliases: {aliases or 'None'}")
        
        site_index += 1
    
    if not sites:
        print("❌ No legacy sites found in environment variables")
        return None
    
    # Create JSON configuration
    json_config = json.dumps(sites, indent=2)
    
    print(f"\n📋 GENERATED JSON CONFIGURATION:")
    print("-" * 40)
    print(json_config)
    
    print(f"\n🔧 TO UPDATE YOUR ENVIRONMENT:")
    print("-" * 40)
    print("Add this to your Railway environment variables:")
    print(f"GOOGLE_SITES_CONFIG={json_config}")
    
    print(f"\n⚠️  NOTE: You can remove these old variables after switching:")
    for i in range(1, site_index):
        if os.getenv(f'SITE_{i}_NAME'):
            print(f"- SITE_{i}_NAME")
            print(f"- SITE_{i}_ANALYTICS_VIEW_ID") 
            print(f"- SITE_{i}_SEARCH_CONSOLE_URL")
            print(f"- SITE_{i}_ALIASES")
    
    return json_config

def verify_current_configuration():
    """Check what configuration method is currently active"""
    
    print("🔍 CURRENT CONFIGURATION STATUS")
    print("=" * 40)
    
    # Check for JSON config
    json_config = os.getenv('GOOGLE_SITES_CONFIG')
    if json_config:
        try:
            sites = json.loads(json_config)
            print(f"✅ JSON Configuration Active: {len(sites)} sites")
            for key, site in sites.items():
                print(f"   - {site['name']} ({key})")
        except json.JSONDecodeError:
            print("❌ JSON Configuration Present but INVALID")
    else:
        print("❌ No JSON configuration found (GOOGLE_SITES_CONFIG)")
    
    # Check for legacy config
    legacy_sites = 0
    site_index = 1
    
    while True:
        site_name = os.getenv(f'SITE_{site_index}_NAME')
        if not site_name:
            break
        legacy_sites += 1
        site_index += 1
    
    if legacy_sites > 0:
        print(f"⚠️  Legacy Configuration Active: {legacy_sites} sites")
        print("   This is the old format - consider converting to JSON")
    else:
        print("✅ No legacy configuration found")
    
    return bool(json_config), legacy_sites > 0

if __name__ == "__main__":
    print("🏗️  GOOGLE SITES CONFIGURATION CONVERTER")
    print("=" * 60)
    
    has_json, has_legacy = verify_current_configuration()
    
    if has_legacy and not has_json:
        print(f"\n🔄 Converting legacy configuration...")
        json_config = convert_legacy_env_to_json()
        
        if json_config:
            # Save to a file for easy copying
            with open('google_sites_config.json', 'w') as f:
                f.write(json_config)
            print(f"\n💾 Configuration saved to: google_sites_config.json")
            print("You can copy this file content to set GOOGLE_SITES_CONFIG")
    
    elif has_json:
        print(f"\n✅ JSON configuration is already active - no conversion needed")
        
    elif not has_json and not has_legacy:
        print(f"\n❌ No configuration found - please set up either:")
        print("   1. Individual SITE_X_ variables, or")  
        print("   2. GOOGLE_SITES_CONFIG JSON variable")
        
    else:
        print(f"\n⚠️  Both configurations present - JSON takes priority")