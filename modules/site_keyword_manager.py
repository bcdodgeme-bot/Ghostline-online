# modules/site_keyword_manager.py
# Multi-Site Keyword Management System for Content Matching

import csv
import io
import re
import datetime
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from modules.database import get_db_connection
import psycopg2
from psycopg2.extras import RealDictCursor

# Site configuration
SITE_DOMAINS = {
    'bcdodge.me': {
        'name': 'BC Dodge Digital Marketing',
        'focus_areas': ['digital marketing', 'SEO', 'leadership', 'strategy', 'military', 'Georgetown'],
        'primary_categories': ['marketing', 'seo', 'leadership', 'strategy']
    },
    'roseandangel.com': {
        'name': 'Rose & Angel Consulting',
        'focus_areas': ['business consulting', 'ROI', 'client success', 'marketing strategy'],
        'primary_categories': ['consulting', 'roi', 'client-success', 'marketing']
    },
    'mealsnfeelz.org': {
        'name': 'Meals N Feelz',
        'focus_areas': ['food security', 'hunger relief', 'nonprofit operations', 'community impact'],
        'primary_categories': ['food-security', 'hunger-relief', 'nonprofit', 'community']
    },
    'tvsignals.com': {
        'name': 'TV Signals',
        'focus_areas': ['TV reviews', 'streaming services', 'entertainment analysis', 'cultural commentary'],
        'primary_categories': ['reviews', 'streaming', 'analysis', 'culture']
    },
    'damnitcarl.com': {
        'name': 'Damn It Carl',
        'focus_areas': ['creative burnout', 'mental health', 'work-life balance', 'authenticity'],
        'primary_categories': ['burnout', 'mental-health', 'work-life', 'authenticity']
    },
    'amcf.org': {
        'name': 'AMCF',
        'focus_areas': ['nonprofit management', 'community development', 'organizational strategy', 'social impact'],
        'primary_categories': ['nonprofit', 'community', 'strategy', 'social-impact']
    }
}

@dataclass
class KeywordMatch:
    """Represents a keyword match with scoring"""
    site_domain: str
    keyword: str
    match_score: float
    match_type: str  # exact, partial, semantic
    keyword_id: int
    search_volume: int = 0
    competition_level: str = 'unknown'

@dataclass
class ContentMatchResult:
    """Results from matching content against all sites"""
    topic: str
    best_site: str
    confidence_score: float
    all_matches: List[KeywordMatch]
    reasoning: str

class SiteKeywordManager:
    """Main keyword management system"""
    
    def __init__(self):
        self.sites = SITE_DOMAINS
    
    def add_keyword(self, site_domain: str, keyword: str, **kwargs) -> bool:
        """Add a single keyword to a site"""
        if site_domain not in self.sites:
            raise ValueError(f"Unknown site domain: {site_domain}")
        
        with get_db_connection() as conn:
            if not conn:
                return False
            
            try:
                cursor = conn.cursor()
                
                # Prepare keyword data
                search_volume = kwargs.get('search_volume', 0)
                competition_level = kwargs.get('competition_level', 'unknown')
                suggested_bid = kwargs.get('suggested_bid', 0.00)
                keyword_category = kwargs.get('category', self._guess_category(site_domain, keyword))
                source = kwargs.get('source', 'manual')
                
                cursor.execute('''
                    INSERT INTO site_keywords 
                    (site_domain, keyword, search_volume, competition_level, suggested_bid, 
                     keyword_category, source, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    ON CONFLICT (site_domain, keyword) 
                    DO UPDATE SET 
                        search_volume = EXCLUDED.search_volume,
                        competition_level = EXCLUDED.competition_level,
                        suggested_bid = EXCLUDED.suggested_bid,
                        keyword_category = EXCLUDED.keyword_category,
                        updated_at = CURRENT_TIMESTAMP
                ''', (site_domain, keyword.lower().strip(), search_volume, 
                      competition_level, suggested_bid, keyword_category, source))
                
                conn.commit()
                print(f"Added keyword '{keyword}' to {site_domain}")
                return True
                
            except Exception as e:
                print(f"Failed to add keyword: {e}")
                conn.rollback()
                return False
    
    def bulk_import_csv(self, site_domain: str, csv_content: str) -> Dict[str, Any]:
        """Bulk import keywords from Google Ads Keyword Planner CSV"""
        if site_domain not in self.sites:
            raise ValueError(f"Unknown site domain: {site_domain}")
        
        results = {
            'success': False,
            'imported_count': 0,
            'skipped_count': 0,
            'errors': [],
            'preview': []
        }
        
        try:
            # Parse CSV content
            csv_reader = csv.DictReader(io.StringIO(csv_content))
            
            # Try to detect column names (Google Ads exports vary)
            fieldnames = csv_reader.fieldnames
            print(f"CSV fieldnames: {fieldnames}")
            
            # Map common column name variations
            keyword_col = None
            volume_col = None
            competition_col = None
            bid_col = None
            
            for field in fieldnames:
                field_lower = field.lower()
                if any(term in field_lower for term in ['keyword', 'search term', 'query']):
                    keyword_col = field
                elif any(term in field_lower for term in ['volume', 'searches', 'search volume']):
                    volume_col = field
                elif any(term in field_lower for term in ['competition', 'comp', 'difficulty']):
                    competition_col = field
                elif any(term in field_lower for term in ['bid', 'cpc', 'cost per click', 'suggested bid']):
                    bid_col = field
            
            if not keyword_col:
                results['errors'].append("Could not find keyword column in CSV")
                return results
            
            # Process rows
            for i, row in enumerate(csv_reader):
                if i >= 500:  # Limit to 500 keywords per import
                    break
                    
                try:
                    keyword = row.get(keyword_col, '').strip()
                    if not keyword or len(keyword) < 2:
                        results['skipped_count'] += 1
                        continue
                    
                    # Extract other data if available
                    search_volume = self._parse_number(row.get(volume_col, '0'))
                    competition_level = self._parse_competition(row.get(competition_col, ''))
                    suggested_bid = self._parse_currency(row.get(bid_col, '0'))
                    
                    # Add to database
                    if self.add_keyword(
                        site_domain=site_domain,
                        keyword=keyword,
                        search_volume=search_volume,
                        competition_level=competition_level,
                        suggested_bid=suggested_bid,
                        source='csv_import'
                    ):
                        results['imported_count'] += 1
                        
                        # Add to preview for first 10
                        if len(results['preview']) < 10:
                            results['preview'].append({
                                'keyword': keyword,
                                'search_volume': search_volume,
                                'competition_level': competition_level,
                                'suggested_bid': suggested_bid
                            })
                    else:
                        results['skipped_count'] += 1
                        
                except Exception as e:
                    results['errors'].append(f"Row {i+1}: {str(e)}")
                    results['skipped_count'] += 1
            
            results['success'] = results['imported_count'] > 0
            return results
            
        except Exception as e:
            results['errors'].append(f"CSV parsing error: {str(e)}")
            return results
    
    def get_site_keywords(self, site_domain: str, limit: int = 100, category: str = None) -> List[Dict[str, Any]]:
        """Get all keywords for a specific site"""
        if site_domain not in self.sites:
            raise ValueError(f"Unknown site domain: {site_domain}")
        
        with get_db_connection() as conn:
            if not conn:
                return []
            
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                base_query = '''
                    SELECT id, keyword, search_volume, competition_level, suggested_bid,
                           keyword_category, match_score, times_used, last_used,
                           source, created_at, is_active
                    FROM site_keywords
                    WHERE site_domain = %s AND is_active = true
                '''
                
                params = [site_domain]
                
                if category:
                    base_query += ' AND keyword_category = %s'
                    params.append(category)
                
                base_query += ' ORDER BY match_score DESC, search_volume DESC LIMIT %s'
                params.append(limit)
                
                cursor.execute(base_query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
            except Exception as e:
                print(f"Failed to get site keywords: {e}")
                return []
    
    def match_content_to_sites(self, topic: str) -> ContentMatchResult:
        """Score content topic against all sites and find best match"""
        
        topic_lower = topic.lower()
        topic_words = self._extract_keywords(topic_lower)
        
        all_matches = []
        site_scores = {}
        
        # Get keywords for all sites
        for site_domain in self.sites:
            site_keywords = self.get_site_keywords(site_domain, limit=200)
            site_match_score = 0
            site_matches = []
            
            for kw_data in site_keywords:
                keyword = kw_data['keyword']
                match_score, match_type = self._calculate_match_score(topic_lower, keyword, topic_words)
                
                if match_score > 0:
                    keyword_match = KeywordMatch(
                        site_domain=site_domain,
                        keyword=keyword,
                        match_score=match_score,
                        match_type=match_type,
                        keyword_id=kw_data['id'],
                        search_volume=kw_data.get('search_volume', 0),
                        competition_level=kw_data.get('competition_level', 'unknown')
                    )
                    site_matches.append(keyword_match)
                    all_matches.append(keyword_match)
                    
                    # Weight by search volume and historical performance
                    weighted_score = match_score
                    if kw_data.get('search_volume', 0) > 0:
                        weighted_score *= (1 + kw_data['search_volume'] / 10000)
                    if kw_data.get('match_score', 0) > 0:
                        weighted_score *= (1 + kw_data['match_score'] / 10)
                    
                    site_match_score += weighted_score
            
            if site_matches:
                # Normalize by number of matches to avoid bias toward sites with more keywords
                site_scores[site_domain] = site_match_score / len(site_matches)
                print(f"{site_domain}: {site_match_score:.2f} from {len(site_matches)} matches")
        
        # Determine best site
        if not site_scores:
            best_site = 'bcdodge.me'  # Default fallback
            confidence_score = 0.0
            reasoning = "No keyword matches found - defaulting to primary site"
        else:
            best_site = max(site_scores, key=site_scores.get)
            max_score = site_scores[best_site]
            
            # Calculate confidence based on score separation
            scores_list = sorted(site_scores.values(), reverse=True)
            if len(scores_list) > 1:
                confidence_score = min(100.0, (max_score - scores_list[1]) / max_score * 100)
            else:
                confidence_score = 100.0
            
            reasoning = f"Best match with score {max_score:.2f} - {confidence_score:.1f}% confidence"
        
        # Sort all matches by score
        all_matches.sort(key=lambda x: x.match_score, reverse=True)
        
        return ContentMatchResult(
            topic=topic,
            best_site=best_site,
            confidence_score=confidence_score,
            all_matches=all_matches[:20],  # Top 20 matches
            reasoning=reasoning
        )
    
    def record_keyword_performance(self, keyword_id: int, content_topic: str, 
                                 performance_score: float, user_feedback: str = 'pending') -> bool:
        """Record performance feedback for a keyword match"""
        with get_db_connection() as conn:
            if not conn:
                return False
            
            try:
                cursor = conn.cursor()
                
                # Log the performance
                cursor.execute('''
                    INSERT INTO keyword_performance_logs 
                    (site_keyword_id, content_topic, performance_score, user_feedback)
                    VALUES (%s, %s, %s, %s)
                ''', (keyword_id, content_topic, performance_score, user_feedback))
                
                # Update the keyword's match score based on feedback
                if user_feedback in ['approved', 'rejected']:
                    score_adjustment = performance_score if user_feedback == 'approved' else -performance_score/2
                    
                    cursor.execute('''
                        UPDATE site_keywords 
                        SET match_score = GREATEST(0, match_score + %s),
                            times_used = times_used + 1,
                            last_used = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    ''', (score_adjustment, keyword_id))
                
                conn.commit()
                return True
                
            except Exception as e:
                print(f"Failed to record keyword performance: {e}")
                conn.rollback()
                return False
    
    def get_keyword_stats(self) -> Dict[str, Any]:
        """Get overall keyword statistics"""
        with get_db_connection() as conn:
            if not conn:
                return {}
            
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                # Overall stats
                cursor.execute('''
                    SELECT 
                        site_domain,
                        COUNT(*) as total_keywords,
                        COUNT(*) FILTER (WHERE is_active = true) as active_keywords,
                        AVG(search_volume) as avg_search_volume,
                        AVG(match_score) as avg_match_score,
                        SUM(times_used) as total_usage
                    FROM site_keywords
                    GROUP BY site_domain
                    ORDER BY site_domain
                ''')
                
                site_stats = {}
                for row in cursor.fetchall():
                    site_stats[row['site_domain']] = {
                        'total_keywords': row['total_keywords'],
                        'active_keywords': row['active_keywords'],
                        'avg_search_volume': float(row['avg_search_volume'] or 0),
                        'avg_match_score': float(row['avg_match_score'] or 0),
                        'total_usage': row['total_usage'] or 0,
                        'site_name': self.sites.get(row['site_domain'], {}).get('name', row['site_domain'])
                    }
                
                # Performance stats
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_performance_logs,
                        COUNT(*) FILTER (WHERE user_feedback = 'approved') as approved_count,
                        COUNT(*) FILTER (WHERE user_feedback = 'rejected') as rejected_count,
                        AVG(performance_score) as avg_performance_score
                    FROM keyword_performance_logs
                ''')
                
                perf_stats = cursor.fetchone()
                
                return {
                    'site_stats': site_stats,
                    'performance_stats': dict(perf_stats) if perf_stats else {},
                    'last_updated': datetime.datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"Failed to get keyword stats: {e}")
                return {}
    
    def remove_keyword(self, site_domain: str, keyword: str) -> bool:
        """Remove/deactivate a keyword"""
        with get_db_connection() as conn:
            if not conn:
                return False
            
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE site_keywords 
                    SET is_active = false, updated_at = CURRENT_TIMESTAMP
                    WHERE site_domain = %s AND keyword = %s
                ''', (site_domain, keyword.lower().strip()))
                
                conn.commit()
                return cursor.rowcount > 0
                
            except Exception as e:
                print(f"Failed to remove keyword: {e}")
                conn.rollback()
                return False
    
    # Helper methods
    def _guess_category(self, site_domain: str, keyword: str) -> str:
        """Guess keyword category based on site and keyword content"""
        site_info = self.sites.get(site_domain, {})
        primary_categories = site_info.get('primary_categories', ['general'])
        
        keyword_lower = keyword.lower()
        
        # Simple keyword-based category detection
        category_keywords = {
            'marketing': ['marketing', 'seo', 'content', 'social', 'ads', 'promotion'],
            'leadership': ['leadership', 'management', 'team', 'strategy', 'executive'],
            'consulting': ['consulting', 'advisory', 'guidance', 'expertise'],
            'roi': ['roi', 'return', 'profit', 'revenue', 'performance', 'metrics'],
            'mental-health': ['mental', 'health', 'burnout', 'stress', 'anxiety', 'wellness'],
            'food-security': ['food', 'hunger', 'nutrition', 'meals', 'pantry', 'security'],
            'entertainment': ['tv', 'show', 'movie', 'streaming', 'entertainment', 'review']
        }
        
        for category, category_keywords_list in category_keywords.items():
            if any(cat_kw in keyword_lower for cat_kw in category_keywords_list):
                if category in primary_categories:
                    return category
        
        return primary_categories[0] if primary_categories else 'general'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        return [word for word in words if len(word) > 2 and word not in stop_words]
    
    def _calculate_match_score(self, topic: str, keyword: str, topic_words: List[str]) -> Tuple[float, str]:
        """Calculate how well a keyword matches the topic"""
        keyword_lower = keyword.lower()
        keyword_words = self._extract_keywords(keyword_lower)
        
        # Exact match
        if keyword_lower in topic or topic in keyword_lower:
            return 10.0, 'exact'
        
        # Partial phrase match
        if any(phrase in topic for phrase in keyword_lower.split()) or any(phrase in keyword_lower for phrase in topic.split()):
            return 8.0, 'partial'
        
        # Word overlap scoring
        if topic_words and keyword_words:
            common_words = set(topic_words) & set(keyword_words)
            if common_words:
                overlap_ratio = len(common_words) / max(len(topic_words), len(keyword_words))
                score = overlap_ratio * 6.0  # Max 6.0 for semantic matches
                return score, 'semantic'
        
        return 0.0, 'none'
    
    def _parse_number(self, value: str) -> int:
        """Parse number from string, handling various formats"""
        if not value:
            return 0
        
        # Remove common formatting
        cleaned = re.sub(r'[,\s]', '', str(value))
        
        # Extract number
        match = re.search(r'(\d+)', cleaned)
        if match:
            return int(match.group(1))
        
        return 0
    
    def _parse_competition(self, value: str) -> str:
        """Parse competition level from various formats"""
        if not value:
            return 'unknown'
        
        value_lower = str(value).lower()
        
        if any(term in value_lower for term in ['high', '3', 'hard']):
            return 'high'
        elif any(term in value_lower for term in ['medium', 'med', '2', 'moderate']):
            return 'medium'
        elif any(term in value_lower for term in ['low', '1', 'easy']):
            return 'low'
        
        return 'unknown'
    
    def _parse_currency(self, value: str) -> float:
        """Parse currency value from string"""
        if not value:
            return 0.0
        
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[$€£¥,\s]', '', str(value))
        
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

# Global instance
keyword_manager = SiteKeywordManager()