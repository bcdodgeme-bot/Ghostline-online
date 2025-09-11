# modules/analytics_validation_enhanced.py
"""
Enhanced Analytics Validation - Integrates with existing enhanced_google_integration.py
Prevents fabricated data from causing inappropriate AI suggestions
"""

import re
import hashlib
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class DataValidationResult:
    """Enhanced validation result with confidence scoring"""
    is_valid: bool
    confidence_score: float  # 0.0 to 1.0
    validation_errors: List[str]
    warnings: List[str]
    raw_data_sample: str
    checksum: str
    site_relevance_score: float  # 0.0 to 1.0
    recommendation: str

class EnhancedAnalyticsValidator:
    """Enhanced validator that integrates with existing Google integration"""
    
    def __init__(self, site_config: Dict):
        self.site_config = site_config
        self.site_name = site_config.get('name', 'Unknown Site')
        
    def validate_analytics_data_enhanced(self, analytics_result: Dict) -> DataValidationResult:
        """Enhanced Analytics validation with site relevance checking"""
        
        validation_errors = []
        warnings = []
        confidence_score = 1.0
        
        # Basic validation from existing system
        if not analytics_result.get('success'):
            return DataValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_errors=[f"Analytics API call failed: {analytics_result.get('error')}"],
                warnings=[],
                raw_data_sample="No data returned",
                checksum="",
                site_relevance_score=0.0,
                recommendation="Fix API connectivity before proceeding"
            )
        
        data = analytics_result.get('data', [])
        
        if not data:
            return DataValidationResult(
                is_valid=True,  # Empty data can be valid for new sites
                confidence_score=0.8,
                validation_errors=[],
                warnings=["No analytics data - could indicate new site or no traffic"],
                raw_data_sample="Empty dataset",
                checksum="",
                site_relevance_score=1.0,  # Can't judge relevance with no data
                recommendation="Monitor for data in the coming days"
            )
        
        # Enhanced data integrity checks
        total_sessions = sum(row.get('sessions', 0) for row in data)
        total_users = sum(row.get('users', 0) for row in data)
        total_pageviews = sum(row.get('pageviews', 0) for row in data)
        
        # Impossible value checks
        if total_sessions > total_pageviews:
            validation_errors.append("Sessions cannot exceed pageviews - data integrity issue")
            confidence_score -= 0.4
            
        if total_users > total_sessions:
            validation_errors.append("Users cannot exceed sessions - data integrity issue") 
            confidence_score -= 0.4
        
        # Check for suspicious spikes (could indicate fabricated data)
        if len(data) > 7:
            daily_sessions = [row.get('sessions', 0) for row in data]
            avg_sessions = sum(daily_sessions) / len(daily_sessions)
            
            for i, sessions in enumerate(daily_sessions):
                if sessions > avg_sessions * 15 and avg_sessions > 0:  # 15x spike threshold
                    warnings.append(f"Extremely unusual traffic spike on day {i}: {sessions} sessions (avg: {avg_sessions:.1f})")
                    confidence_score -= 0.2
        
        # Validate bounce rates
        bounce_rates = [row.get('bounce_rate', 0) for row in data if row.get('bounce_rate') is not None]
        if bounce_rates:
            for rate in bounce_rates:
                if rate < 0 or rate > 100:
                    validation_errors.append(f"Invalid bounce rate: {rate}% (must be 0-100%)")
                    confidence_score -= 0.3
        
        # Generate data checksum for integrity
        data_str = json.dumps(data, sort_keys=True)
        checksum = hashlib.md5(data_str.encode()).hexdigest()
        
        # Sample for manual verification
        sample_data = {
            'total_sessions': total_sessions,
            'total_users': total_users,
            'total_pageviews': total_pageviews,
            'date_range': analytics_result.get('date_range', 'Unknown'),
            'property_id': analytics_result.get('property_id', 'Unknown'),
            'sample_days': data[:3]
        }
        
        raw_data_sample = json.dumps(sample_data, indent=2)
        
        # Generate recommendation
        if len(validation_errors) > 0:
            recommendation = "CRITICAL: Manual verification required before using for AI suggestions"
        elif confidence_score < 0.7:
            recommendation = "CAUTION: Review data quality before generating content"
        else:
            recommendation = "Data appears valid for AI content generation"
        
        return DataValidationResult(
            is_valid=len(validation_errors) == 0,
            confidence_score=max(0.0, confidence_score),
            validation_errors=validation_errors,
            warnings=warnings,
            raw_data_sample=raw_data_sample,
            checksum=checksum,
            site_relevance_score=1.0,  # Analytics doesn't have query relevance
            recommendation=recommendation
        )

    def validate_search_console_data_enhanced(self, sc_result: Dict) -> DataValidationResult:
        """Enhanced Search Console validation with CRITICAL site relevance checking"""
        
        validation_errors = []
        warnings = []
        confidence_score = 1.0
        
        # Basic validation
        if not sc_result.get('success'):
            return DataValidationResult(
                is_valid=False,
                confidence_score=0.0,
                validation_errors=[f"Search Console API call failed: {sc_result.get('error')}"],
                warnings=[],
                raw_data_sample="No data returned",
                checksum="",
                site_relevance_score=0.0,
                recommendation="Fix API connectivity before proceeding"
            )
        
        data = sc_result.get('data', [])
        
        if not data:
            return DataValidationResult(
                is_valid=True,
                confidence_score=0.8,
                validation_errors=[],
                warnings=["No Search Console data - normal for new sites"],
                raw_data_sample="Empty dataset",
                checksum="",
                site_relevance_score=1.0,
                recommendation="Monitor for search visibility in coming weeks"
            )
        
        # Data integrity checks
        total_clicks = sum(row.get('clicks', 0) for row in data)
        total_impressions = sum(row.get('impressions', 0) for row in data)
        
        if total_clicks > total_impressions:
            validation_errors.append("Clicks cannot exceed impressions - CRITICAL data integrity issue")
            confidence_score -= 0.5
        
        # Validate individual queries
        for i, row in enumerate(data):
            clicks = row.get('clicks', 0)
            impressions = row.get('impressions', 0)
            ctr = row.get('ctr', 0)
            position = row.get('position', 0)
            
            # CTR validation
            if impressions > 0:
                calculated_ctr = (clicks / impressions) * 100
                if abs(calculated_ctr - ctr) > 2.0:  # Allow 2% tolerance
                    warnings.append(f"CTR mismatch for query {i}: reported {ctr:.2f}%, calculated {calculated_ctr:.2f}%")
                    confidence_score -= 0.1
            
            # Position validation
            if position < 1 or position > 100:
                warnings.append(f"Unusual position for query {i}: {position}")
                confidence_score -= 0.1
        
        # CRITICAL: Site relevance validation (prevents "Christianity B2B" disasters)
        site_relevance_score = self._validate_query_relevance_enhanced(data)
        confidence_score *= site_relevance_score
        
        if site_relevance_score < 0.5:
            validation_errors.append(f"CRITICAL: Query relevance extremely low ({site_relevance_score:.2f}) - likely data from wrong site")
        elif site_relevance_score < 0.7:
            warnings.append(f"Query relevance below expected ({site_relevance_score:.2f}) - verify site configuration")
        
        # Generate data checksum
        data_str = json.dumps(data, sort_keys=True)
        checksum = hashlib.md5(data_str.encode()).hexdigest()
        
        # Sample for manual verification
        sample_data = {
            'total_clicks': total_clicks,
            'total_impressions': total_impressions,
            'total_queries': len(data),
            'avg_position': sum(row.get('position', 0) for row in data) / len(data) if data else 0,
            'date_range': sc_result.get('date_range', 'Unknown'),
            'site_url': sc_result.get('site_url', 'Unknown'),
            'top_queries': [row.get('query') for row in data[:5]],
            'relevance_score': site_relevance_score
        }
        
        raw_data_sample = json.dumps(sample_data, indent=2)
        
        # Generate recommendation based on relevance
        if site_relevance_score < 0.5:
            recommendation = "🚨 BLOCK AI SUGGESTIONS - Data appears to be from wrong site/topic"
        elif len(validation_errors) > 0:
            recommendation = "CRITICAL: Manual verification required before AI suggestions"
        elif confidence_score < 0.7:
            recommendation = "CAUTION: Review data quality and site relevance"
        else:
            recommendation = "Data validated - safe for AI content generation"
        
        return DataValidationResult(
            is_valid=len(validation_errors) == 0 and site_relevance_score >= 0.5,
            confidence_score=max(0.0, confidence_score),
            validation_errors=validation_errors,
            warnings=warnings,
            raw_data_sample=raw_data_sample,
            checksum=checksum,
            site_relevance_score=site_relevance_score,
            recommendation=recommendation
        )

    def _validate_query_relevance_enhanced(self, search_data: List[Dict]) -> float:
        """CRITICAL: Validate that search queries match the expected site content"""
        
        if not search_data:
            return 1.0  # No data to validate
        
        # Get expected keywords for this site
        expected_keywords = self._get_expected_site_keywords()
        
        if not expected_keywords:
            return 0.8  # Neutral score if no keywords configured
        
        relevant_queries = 0
        total_clicks_relevant = 0
        total_clicks = sum(row.get('clicks', 0) for row in search_data)
        
        # Analyze each query for relevance
        for row in search_data:
            query = row.get('query', '').lower()
            clicks = row.get('clicks', 0)
            
            # Check if query contains any expected keywords
            is_relevant = any(keyword.lower() in query for keyword in expected_keywords)
            
            if is_relevant:
                relevant_queries += 1
                total_clicks_relevant += clicks
        
        # Calculate relevance score based on both query count and traffic weight
        query_relevance = relevant_queries / len(search_data) if search_data else 0
        traffic_relevance = total_clicks_relevant / total_clicks if total_clicks > 0 else 0
        
        # Weight traffic relevance more heavily (people clicking = more important)
        overall_relevance = (query_relevance * 0.3) + (traffic_relevance * 0.7)
        
        return min(1.0, overall_relevance)

    def _get_expected_site_keywords(self) -> List[str]:
        """Get expected keywords for Carl's actual sites based on real SEO plans"""
        
        site_name_lower = self.site_name.lower()
        
        # Carl's actual site keyword mapping from SEO plans
        site_keyword_map = {
            # BC Dodge Personal Blog
            'bc dodge': [
                'creative productivity', 'burnout recovery tips', 'automation fails',
                'wordpress mistakes', 'solo consultant strategy', 'marketing strategy for consultants',
                'digital marketing in 2025', 'content marketing ROI', 'real productivity hacks',
                'bad marketing advice', 'work burnout recovery', 'personal operating manual',
                'productivity habits', 'mental health balance', 'website mistakes',
                'automation humor', 'mindfulness for busy people', 'career journey struggles'
            ],
            'bcdodge': [
                'creative productivity', 'burnout recovery tips', 'automation fails',
                'wordpress mistakes', 'solo consultant strategy', 'marketing strategy',
                'digital marketing', 'content marketing', 'productivity hacks', 'carl dodge'
            ],
            
            # Rose and Angle Non-Profit Consulting
            'rose and angle': [
                'nonprofit marketing', 'marketing consultant', 'fractional CMO',
                'nonprofit marketing consultant', 'data-driven marketing', 'digital marketing strategy',
                'marketing consulting services', 'nonprofit email marketing', 'lead generation',
                'marketing performance', 'marketing ROI', 'nonprofit lead generation',
                'authentic lead magnet', 'marketing consultation', 'convert leads'
            ],
            'roseandangle': [
                'nonprofit marketing', 'marketing consultant', 'fractional CMO',
                'nonprofit consulting', 'marketing strategy', 'lead generation',
                'email marketing', 'digital marketing', 'rose angle'
            ],
            
            # Meals N Feelz - Food Program Fundraising (NOT cooking!)
            'meals n feelz': [
                'fidya donation', 'fidya 2026', 'fidya calculator', 'fidya USA',
                'pay fidya online', 'missed fast', 'fidya vs kaffarah', 'fidya amount',
                'where can I get food', 'I need food', 'community food help',
                'food desert', 'community pantry', 'feed the community', 'mutual aid',
                'food pantry near me', 'hunger relief', 'charitable giving',
                'ripple effect of fidya', 'fidya in action', 'food insecurity'
            ],
            'mealsnfeelz': [
                'fidya donation', 'fidya', 'food program', 'hunger relief',
                'charity', 'nonprofit', 'community food', 'food insecurity',
                'meals n feelz', 'food pantry', 'donate food'
            ],
            
            # TV Signals - TV Reviews & Opinions
            'tv signals': [
                'my watchlist', 'tv show', 'streaming tv', 'streaming apps', 'binge tv',
                'tv show review', 'streaming tv show', 'watchlist', 'recommended for you',
                'binge tv fatigue', 'streaming app', 'tv commentary', 'episode review',
                'what to watch', 'streaming service', 'tv critic', 'show recommendations'
            ],
            'tvsignals': [
                'tv show', 'streaming', 'watchlist', 'binge tv', 'tv review',
                'streaming apps', 'television', 'tv signals', 'show review'
            ],
            
            # Damnit Carl - Creative Burnout Therapy
            'damnit carl': [
                'emotional support', 'burnout support', 'healing burnout', 'burned out',
                'creative burnout', 'mental health', 'work life balance', 'creative therapy',
                'emotional labor', 'burnout humor', 'creative first aid', 'creative block',
                'burnout recovery', 'creative wellness', 'artistic burnout', 'design therapy',
                'freelance burnout', 'creative mental health', 'burnout survival'
            ],
            'damnitcarl': [
                'creative burnout', 'burnout support', 'emotional support',
                'creative therapy', 'mental health', 'work life balance',
                'damnit carl', 'creative wellness', 'burnout recovery'
            ]
        }
        
        # Try to match site name to keywords
        for site_pattern, keywords in site_keyword_map.items():
            if site_pattern in site_name_lower:
                return keywords
        
        # Check if keywords are in site config
        if 'expected_keywords' in self.site_config:
            return self.site_config['expected_keywords']
        
        # Fallback - use site name words
        return site_name_lower.split()


def validate_analytics_data_comprehensive(site_config: Dict, analytics_result: Dict = None, search_console_result: Dict = None) -> Dict[str, DataValidationResult]:
    """Main validation function that integrates with existing enhanced_google_integration.py"""
    
    validator = EnhancedAnalyticsValidator(site_config)
    results = {}
    
    if analytics_result:
        results['analytics'] = validator.validate_analytics_data_enhanced(analytics_result)
    
    if search_console_result:
        results['search_console'] = validator.validate_search_console_data_enhanced(search_console_result)
    
    return results

def should_block_ai_suggestions_enhanced(validation_results: Dict[str, DataValidationResult]) -> Tuple[bool, str, float]:
    """Enhanced function to determine if AI suggestions should be blocked"""
    
    blocking_issues = []
    min_confidence = 1.0
    min_relevance = 1.0
    
    for data_type, result in validation_results.items():
        min_confidence = min(min_confidence, result.confidence_score)
        min_relevance = min(min_relevance, result.site_relevance_score)
        
        if not result.is_valid:
            blocking_issues.append(f"{data_type} data has validation errors")
        elif result.confidence_score < 0.6:
            blocking_issues.append(f"{data_type} data has low confidence ({result.confidence_score:.2f})")
        elif result.site_relevance_score < 0.5:
            blocking_issues.append(f"{data_type} data appears to be from wrong site/topic ({result.site_relevance_score:.2f} relevance)")
    
    should_block = len(blocking_issues) > 0
    reason = "; ".join(blocking_issues) if should_block else ""
    
    return should_block, reason, min(min_confidence, min_relevance)
