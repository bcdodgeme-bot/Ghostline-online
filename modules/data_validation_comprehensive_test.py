# data_validation_comprehensive_test.py - Test all sites for data quality issues

import os
import json
import datetime
from typing import Dict, List, Tuple

def test_all_sites_data_validation():
    """Test all configured sites for data validation issues"""
    
    print("🔍 COMPREHENSIVE DATA VALIDATION TEST")
    print("=" * 60)
    
    # Load sites configuration
    sites_config_raw = os.getenv('GOOGLE_SITES_CONFIG')
    if not sites_config_raw:
        print("❌ GOOGLE_SITES_CONFIG not found")
        return False
    
    try:
        sites_config = json.loads(sites_config_raw)
    except json.JSONDecodeError as e:
        print(f"❌ Config parse error: {e}")
        return False
    
    # Initialize Google Integration
    try:
        from modules.enhanced_google_integration import GoogleIntegration
        from modules.analytics_validation_enhanced import validate_analytics_data_comprehensive, should_block_ai_suggestions_enhanced
        google_integration = GoogleIntegration()
        print("✅ Google Integration initialized")
    except Exception as e:
        print(f"❌ Integration setup failed: {e}")
        return False
    
    print(f"\n📊 Testing {len(sites_config)} configured sites")
    print("-" * 40)
    
    validation_summary = {}
    
    for site_key, site_config in sites_config.items():
        print(f"\n🏢 TESTING: {site_config['name']} ({site_key})")
        print("=" * 50)
        
        site_results = {
            'name': site_config['name'],
            'analytics_configured': bool(site_config.get('analytics_view_id')),
            'search_console_configured': bool(site_config.get('search_console_url')),
            'analytics_result': None,
            'search_console_result': None,
            'validation_results': None,
            'should_block_blog_suggestions': False,
            'issues': []
        }
        
        # Test Analytics if configured
        if site_results['analytics_configured']:
            print("📈 Testing Google Analytics...")
            try:
                analytics_result = google_integration.get_analytics_data(site_key, '30daysAgo', 'today')
                site_results['analytics_result'] = analytics_result
                
                if analytics_result.get('success'):
                    data = analytics_result.get('data', [])
                    total_sessions = sum(row.get('sessions', 0) for row in data if isinstance(row, dict))
                    total_users = sum(row.get('users', 0) for row in data if isinstance(row, dict))
                    
                    print(f"   ✅ Analytics working: {total_sessions} sessions, {total_users} users")
                    
                    # Check for data quality issues
                    if total_sessions == 0:
                        site_results['issues'].append("Analytics shows zero sessions")
                    elif total_users > total_sessions:
                        site_results['issues'].append("Users exceed sessions (data integrity issue)")
                else:
                    error = analytics_result.get('error', 'Unknown error')
                    print(f"   ❌ Analytics failed: {error}")
                    site_results['issues'].append(f"Analytics API error: {error}")
                    
            except Exception as e:
                print(f"   ❌ Analytics exception: {e}")
                site_results['issues'].append(f"Analytics exception: {str(e)}")
        else:
            print("   ⚠️  Analytics not configured")
        
        # Test Search Console if configured  
        if site_results['search_console_configured']:
            print("🔍 Testing Search Console...")
            try:
                # Use proper date format for Search Console
                end_date = datetime.date.today().strftime('%Y-%m-%d')
                start_date = (datetime.date.today() - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
                
                sc_result = google_integration.get_search_console_data_for_site(site_key, start_date, end_date)
                site_results['search_console_result'] = sc_result
                
                if isinstance(sc_result, tuple):
                    print(f"   🚨 TUPLE ERROR FOUND! Returned: {type(sc_result)}")
                    site_results['issues'].append("CRITICAL: Search Console returns tuple instead of dict")
                elif isinstance(sc_result, dict) and sc_result.get('success'):
                    data = sc_result.get('data', [])
                    total_clicks = sum(row.get('clicks', 0) for row in data if isinstance(row, dict))
                    total_impressions = sum(row.get('impressions', 0) for row in data if isinstance(row, dict))
                    
                    print(f"   ✅ Search Console working: {total_clicks} clicks, {total_impressions} impressions")
                    print(f"   📊 Query count: {len(data)}")
                    
                    # Show sample queries for relevance check
                    if data and isinstance(data, list):
                        sample_queries = [row.get('query', 'Unknown') for row in data[:3] if isinstance(row, dict)]
                        print(f"   🔍 Sample queries: {sample_queries}")
                        
                        # Manual relevance check
                        expected_keywords = site_config.get('expected_keywords', [])
                        if expected_keywords:
                            relevant_queries = []
                            for row in data[:5]:
                                if isinstance(row, dict):
                                    query = row.get('query', '').lower()
                                    is_relevant = any(keyword.lower() in query for keyword in expected_keywords)
                                    if is_relevant:
                                        relevant_queries.append(row.get('query'))
                            
                            relevance_score = len(relevant_queries) / min(len(data), 5) if data else 0
                            print(f"   📊 Relevance score: {relevance_score:.2f} ({len(relevant_queries)}/5 relevant)")
                            
                            if relevance_score < 0.3:
                                site_results['issues'].append(f"Low query relevance: {relevance_score:.2f}")
                    
                    # Check for zero clicks (common issue)
                    if total_clicks == 0:
                        site_results['issues'].append("Search Console shows zero clicks")
                        
                elif isinstance(sc_result, dict):
                    error = sc_result.get('error', 'Unknown error')
                    print(f"   ❌ Search Console failed: {error}")
                    site_results['issues'].append(f"Search Console API error: {error}")
                else:
                    print(f"   ❌ Search Console returned unexpected type: {type(sc_result)}")
                    site_results['issues'].append(f"Search Console returned {type(sc_result)}")
                    
            except Exception as e:
                print(f"   ❌ Search Console exception: {e}")
                site_results['issues'].append(f"Search Console exception: {str(e)}")
        else:
            print("   ⚠️  Search Console not configured")
        
        # Run comprehensive validation if we have data
        if site_results['analytics_result'] or site_results['search_console_result']:
            print("🛡️  Running comprehensive validation...")
            try:
                validation_results = validate_analytics_data_comprehensive(
                    site_config,
                    analytics_result=site_results['analytics_result'],
                    search_console_result=site_results['search_console_result']
                )
                site_results['validation_results'] = validation_results
                
                # Check if blog suggestions should be blocked
                should_block, block_reason, overall_confidence = should_block_ai_suggestions_enhanced(validation_results)
                site_results['should_block_blog_suggestions'] = should_block
                
                if should_block:
                    print(f"   🚨 BLOG SUGGESTIONS BLOCKED: {block_reason}")
                    site_results['issues'].append(f"Blog suggestions blocked: {block_reason}")
                else:
                    print(f"   ✅ Blog suggestions allowed (confidence: {overall_confidence:.2f})")
                
                # Show validation details
                for data_type, result in validation_results.items():
                    status = 'Valid' if result.is_valid else 'Invalid'
                    confidence = result.confidence_score
                    relevance = result.site_relevance_score
                    
                    print(f"   📊 {data_type.title()}: {status} (C:{confidence:.2f}, R:{relevance:.2f})")
                    
                    if result.validation_errors:
                        for error in result.validation_errors:
                            print(f"      ❌ {error}")
                            site_results['issues'].append(f"{data_type} error: {error}")
                    
                    if result.warnings:
                        for warning in result.warnings:
                            print(f"      ⚠️  {warning}")
                            
            except Exception as e:
                print(f"   ❌ Validation exception: {e}")
                site_results['issues'].append(f"Validation exception: {str(e)}")
        
        validation_summary[site_key] = site_results
        
        # Summary for this site
        issue_count = len(site_results['issues'])
        if issue_count == 0:
            print(f"   ✅ Site status: HEALTHY")
        elif issue_count <= 2:
            print(f"   ⚠️  Site status: MINOR ISSUES ({issue_count})")
        else:
            print(f"   ❌ Site status: MAJOR ISSUES ({issue_count})")
    
    # Overall Summary
    print(f"\n📋 VALIDATION SUMMARY")
    print("=" * 60)
    
    healthy_sites = 0
    minor_issue_sites = 0
    major_issue_sites = 0
    tuple_error_sites = []
    blocked_blog_sites = []
    
    for site_key, results in validation_summary.items():
        issue_count = len(results['issues'])
        site_name = results['name']
        
        # Check for specific critical issues
        has_tuple_error = any('tuple' in issue.lower() for issue in results['issues'])
        if has_tuple_error:
            tuple_error_sites.append(site_name)
        
        if results['should_block_blog_suggestions']:
            blocked_blog_sites.append(site_name)
        
        # Categorize by issue severity
        if issue_count == 0:
            healthy_sites += 1
            status_icon = "✅"
        elif issue_count <= 2:
            minor_issue_sites += 1
            status_icon = "⚠️ "
        else:
            major_issue_sites += 1
            status_icon = "❌"
        
        print(f"{status_icon} {site_name}: {issue_count} issues")
        for issue in results['issues'][:3]:  # Show top 3 issues
            print(f"     • {issue}")
        if len(results['issues']) > 3:
            print(f"     • ... and {len(results['issues']) - 3} more issues")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   ✅ Healthy sites: {healthy_sites}")
    print(f"   ⚠️  Minor issues: {minor_issue_sites}")
    print(f"   ❌ Major issues: {major_issue_sites}")
    
    if tuple_error_sites:
        print(f"   🚨 TUPLE ERROR SITES: {', '.join(tuple_error_sites)}")
    
    if blocked_blog_sites:
        print(f"   🛡️  Blog suggestions blocked: {', '.join(blocked_blog_sites)}")
    
    print(f"\n🔧 CRITICAL FIXES NEEDED:")
    if tuple_error_sites:
        print(f"   1. Fix tuple error for: {', '.join(tuple_error_sites)}")
    
    # Find specific configuration issues
    permission_denied_sites = []
    zero_data_sites = []
    
    for site_key, results in validation_summary.items():
        for issue in results['issues']:
            if 'permission denied' in issue.lower():
                permission_denied_sites.append(results['name'])
            elif 'zero' in issue.lower():
                zero_data_sites.append(results['name'])
    
    if permission_denied_sites:
        print(f"   2. Fix Analytics permissions for: {', '.join(set(permission_denied_sites))}")
    
    if zero_data_sites:
        print(f"   3. Investigate zero data for: {', '.join(set(zero_data_sites))}")
    
    print(f"\n💡 RECOMMENDED ACTIONS:")
    print(f"   1. Apply the fixed Search Console handler code")
    print(f"   2. Update Rose and Angel Analytics property ID")
    print(f"   3. Verify Search Console site URLs match exactly")
    print(f"   4. Test 'blog suggestions' commands after fixes")
    
    return validation_summary

if __name__ == "__main__":
    test_all_sites_data_validation()