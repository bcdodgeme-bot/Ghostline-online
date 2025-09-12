# simple_search_console_test.py - Basic test without full imports

import os
import sys
import json

def test_configuration():
    print("🔍 SIMPLE SEARCH CONSOLE CONFIGURATION TEST")
    print("=" * 60)
    
    # Load sites using same logic as your integration
    sites = {}
    
    # Your current environment variable setup
    known_sites = [
        ('BCDodge.me', '1'),
        ('Rose and Angel', '2'), 
        ('Meals N Feelz', '3'),
        ('TV Signals', '4'),
        ('Damn It Carl', '5')
    ]
    
    for site_name, index in known_sites:
        analytics_id = os.getenv(f'SITE_{index}_ANALYTICS_VIEW_ID')
        search_url = os.getenv(f'SITE_{index}_SEARCH_CONSOLE_URL')
        aliases = os.getenv(f'SITE_{index}_ALIASES', '').split(',')
        
        if analytics_id or search_url:
            site_key = site_name.lower().replace(' ', '_').replace('.', '_')
            sites[site_key] = {
                'name': site_name,
                'analytics_view_id': analytics_id,
                'search_console_url': search_url,
                'aliases': [a.strip() for a in aliases if a.strip()]
            }
    
    print(f"Found {len(sites)} configured sites:")
    for key, config in sites.items():
        print(f"\n🔸 {config['name']} ({key})")
        print(f"   Analytics ID: {config.get('analytics_view_id', 'Missing')}")
        print(f"   Search Console: {config.get('search_console_url', 'Missing')}")
        print(f"   Aliases: {config.get('aliases', [])}")
    
    # Test the issue - generate JSON for GOOGLE_SITES_CONFIG
    print(f"\n📋 RECOMMENDED: Convert to JSON configuration")
    print("-" * 50)
    
    json_config = json.dumps(sites, indent=2)
    print("Set this as your GOOGLE_SITES_CONFIG environment variable:")
    print(json_config)
    
    print(f"\n🔧 IMMEDIATE FIXES NEEDED:")
    print("1. Apply Section 6 fix to modules/enhanced_google_integration.py")
    print("2. Set GOOGLE_SITES_CONFIG to the JSON above")
    print("3. Test: 'search console for meals n feelz' command")
    
    return sites

if __name__ == "__main__":
    test_configuration()