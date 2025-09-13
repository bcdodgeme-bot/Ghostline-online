# modules/marketing_retrieval.py - Marketing Best Practices Retrieval Functions
import os
import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

class MarketingKnowledgeRetriever:
    def __init__(self):
        self.db_url = self._get_database_url()
    
    def _get_database_url(self) -> str:
        """Get database URL with Railway compatibility"""
        db_url = os.getenv('DATABASE_URL')
        if db_url and db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        return db_url
    
    @contextmanager
    def get_db_connection(self):
        """Database connection context manager"""
        conn = None
        try:
            if self.db_url:
                conn = psycopg2.connect(self.db_url)
                yield conn
            else:
                print("No DATABASE_URL configured")
                yield None
        except Exception as e:
            print(f"Database connection failed: {e}")
            if conn:
                conn.rollback()
            yield None
        finally:
            if conn:
                conn.close()
    
    def search_marketing_content(self,
                                query: str,
                                category: str = None,
                                subcategory: str = None,
                                limit: int = 5,
                                fresh_only: bool = True,
                                min_relevance: float = 5.0) -> List[Dict[str, Any]]:
        """Search marketing best practices with flexible filtering"""
        
        with self.get_db_connection() as conn:
            if not conn:
                return []
            
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                # Build dynamic query
                where_conditions = [
                    "relevance_score >= %s"
                ]
                params = [min_relevance]
                
                if fresh_only:
                    where_conditions.append("is_fresh = true")
                
                if category:
                    where_conditions.append("category = %s")
                    params.append(category)
                
                if subcategory:
                    where_conditions.append("subcategory = %s")
                    params.append(subcategory)
                
                # Full-text search or keyword search
                if query:
                    where_conditions.append("""
                        (to_tsvector('english', title || ' ' || content || ' ' || COALESCE(summary, '')) 
                         @@ plainto_tsquery('english', %s)
                         OR %s = ANY(keywords))
                    """)
                    params.extend([query, query.lower()])
                
                where_clause = " AND ".join(where_conditions)
                
                search_sql = f'''
                    SELECT 
                        mbp.*,
                        rs.feed_name,
                        ts_rank(to_tsvector('english', mbp.title || ' ' || mbp.content || ' ' || COALESCE(mbp.summary, '')), 
                                plainto_tsquery('english', %s)) as search_rank,
                        CASE 
                            WHEN mbp.published_date IS NOT NULL THEN 
                                EXTRACT(DAYS FROM (CURRENT_DATE - mbp.published_date::timestamp))::integer
                            ELSE NULL 
                        END as days_old
                    FROM marketing_best_practices mbp
                    JOIN rss_sources rs ON mbp.rss_source_id = rs.id
                    WHERE {where_clause}
                    ORDER BY search_rank DESC, mbp.relevance_score DESC, mbp.published_date DESC
                    LIMIT %s
                '''
                
                # Add query parameter for ranking and limit
                all_params = [query or ''] + params + [limit]
                
                cursor.execute(search_sql, all_params)
                results = cursor.fetchall()
                
                # Convert to list of dicts
                formatted_results = []
                for row in results:
                    formatted_results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'][:1500],  # Truncate for RAG
                        'summary': row['summary'],
                        'url': row['url'],
                        'author': row['author'],
                        'published_date': row['published_date'].isoformat() if row['published_date'] else None,
                        'category': row['category'],
                        'subcategory': row['subcategory'],
                        'keywords': row['keywords'],
                        'relevance_score': float(row['relevance_score']),
                        'content_type': row['content_type'],
                        'feed_name': row['feed_name'],
                        'days_old': int(row['days_old']) if row['days_old'] else None,
                        'search_rank': float(row['search_rank']) if row['search_rank'] else 0.0
                    })
                
                print(f"Marketing search for '{query}' returned {len(formatted_results)} results")
                return formatted_results
                
            except Exception as e:
                print(f"Marketing content search failed: {e}")
                return []
    
    def get_seo_best_practices(self, topic: str = None, limit: int = 8) -> List[Dict[str, Any]]:
        """Get current SEO best practices"""
        
        # If no specific topic, search broadly
        if not topic:
            search_terms = "SEO optimization ranking search engine"
        else:
            search_terms = f"SEO {topic}"
        
        return self.search_marketing_content(
            query=search_terms,
            category='seo',
            limit=limit,
            fresh_only=True,
            min_relevance=6.0
        )
    
    def get_content_writing_tips(self, content_type: str = "blog", limit: int = 6) -> List[Dict[str, Any]]:
        """Get content writing and optimization tips"""
        
        search_terms = f"content writing {content_type} blog post optimization"
        
        return self.search_marketing_content(
            query=search_terms,
            category='content_marketing',
            limit=limit,
            fresh_only=True,
            min_relevance=5.5
        )
    
    def get_social_media_strategies(self, platform: str = None, limit: int = 6) -> List[Dict[str, Any]]:
        """Get social media marketing strategies"""
        
        if platform:
            search_terms = f"social media {platform} marketing strategy engagement"
        else:
            search_terms = "social media marketing strategy 2025 engagement"
        
        return self.search_marketing_content(
            query=search_terms,
            category='social_media',
            limit=limit,
            fresh_only=True,
            min_relevance=5.0
        )
    
    def get_rank_math_tips(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get Rank Math specific optimization tips"""
        
        return self.search_marketing_content(
            query="Rank Math optimization WordPress SEO plugin",
            limit=limit,
            fresh_only=True,
            min_relevance=6.0
        )
    
    def get_local_seo_advice(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Get local SEO best practices"""
        
        return self.search_marketing_content(
            query="local SEO google my business local search ranking",
            subcategory='local-seo',
            limit=limit,
            fresh_only=True,
            min_relevance=6.0
        )
    
    def get_technical_seo_guidance(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Get technical SEO best practices"""
        
        return self.search_marketing_content(
            query="technical SEO core web vitals site speed crawling",
            subcategory='technical-seo',
            limit=limit,
            fresh_only=True,
            min_relevance=6.5
        )
    
    def get_email_marketing_strategies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get email marketing best practices"""
        
        return self.search_marketing_content(
            query="email marketing newsletter automation campaigns",
            subcategory='email-marketing',
            limit=limit,
            fresh_only=True,
            min_relevance=5.5
        )
    
    def get_conversion_optimization_tips(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get conversion rate optimization advice"""
        
        return self.search_marketing_content(
            query="conversion rate optimization CRO landing page A/B testing",
            subcategory='conversion-optimization',
            limit=limit,
            fresh_only=True,
            min_relevance=6.0
        )
    
    def get_marketing_trends_2025(self, limit: int = 8) -> List[Dict[str, Any]]:
        """Get current marketing trends and predictions"""
        
        current_year = datetime.datetime.now().year
        
        return self.search_marketing_content(
            query=f"marketing trends {current_year} predictions future digital marketing",
            limit=limit,
            fresh_only=True,
            min_relevance=6.0
        )
    
    def get_contextual_marketing_advice(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get contextual marketing advice for any query"""
        
        # Smart query enhancement based on common patterns
        enhanced_queries = []
        
        query_lower = query.lower()
        
        # Detect intent and enhance query
        if 'seo' in query_lower or 'ranking' in query_lower or 'search' in query_lower:
            enhanced_queries.append(f"SEO {query}")
        
        if 'content' in query_lower or 'blog' in query_lower or 'writing' in query_lower:
            enhanced_queries.append(f"content marketing {query}")
        
        if 'social' in query_lower or any(platform in query_lower for platform in ['facebook', 'instagram', 'twitter', 'linkedin']):
            enhanced_queries.append(f"social media {query}")
        
        if not enhanced_queries:
            enhanced_queries = [query]
        
        # Search with enhanced queries
        all_results = []
        for enhanced_query in enhanced_queries:
            results = self.search_marketing_content(
                query=enhanced_query,
                limit=limit,
                fresh_only=True,
                min_relevance=5.0
            )
            all_results.extend(results)
        
        # Remove duplicates and sort by relevance
        unique_results = {}
        for result in all_results:
            if result['id'] not in unique_results:
                unique_results[result['id']] = result
        
        sorted_results = sorted(
            unique_results.values(),
            key=lambda x: (x['search_rank'], x['relevance_score']),
            reverse=True
        )
        
        return sorted_results[:limit]
    
    def record_content_usage(self, content_id: int, query: str, context: str = 'content_generation'):
        """Record when marketing content is used"""
        
        with self.get_db_connection() as conn:
            if not conn:
                return
            
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO marketing_content_usage (content_id, query_text, usage_context)
                    VALUES (%s, %s, %s)
                ''', (content_id, query[:500], context))
                
                conn.commit()
                
            except Exception as e:
                print(f"Failed to record content usage: {e}")
    
    def get_fresh_marketing_insights(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get fresh marketing insights from the last N days"""
        
        with self.get_db_connection() as conn:
            if not conn:
                return []
            
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute('''
                    SELECT 
                        mbp.*,
                        rs.feed_name,
                        EXTRACT(DAYS FROM (CURRENT_DATE - mbp.published_date::date)) as days_old
                    FROM marketing_best_practices mbp
                    JOIN rss_sources rs ON mbp.rss_source_id = rs.id
                    WHERE mbp.published_date >= CURRENT_DATE - INTERVAL '%s days'
                    AND mbp.relevance_score >= 6.0
                    ORDER BY mbp.published_date DESC, mbp.relevance_score DESC
                    LIMIT %s
                ''', (days, limit))
                
                results = cursor.fetchall()
                
                formatted_results = []
                for row in results:
                    formatted_results.append({
                        'id': row['id'],
                        'title': row['title'],
                        'summary': row['summary'] or row['content'][:300],
                        'url': row['url'],
                        'author': row['author'],
                        'published_date': row['published_date'].isoformat(),
                        'category': row['category'],
                        'subcategory': row['subcategory'],
                        'keywords': row['keywords'],
                        'relevance_score': float(row['relevance_score']),
                        'feed_name': row['feed_name'],
                        'days_old': int(row['days_old'])
                    })
                
                return formatted_results
                
            except Exception as e:
                print(f"Fresh insights query failed: {e}")
                return []
    
    def get_category_stats(self) -> Dict[str, Any]:
        """Get statistics about marketing content by category"""
        
        with self.get_db_connection() as conn:
            if not conn:
                return {}
            
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                cursor.execute('''
                    SELECT 
                        category,
                        COUNT(*) as total_content,
                        COUNT(*) FILTER (WHERE is_fresh = true) as fresh_content,
                        AVG(relevance_score) as avg_relevance,
                        MAX(published_date) as latest_content
                    FROM marketing_best_practices
                    GROUP BY category
                    ORDER BY total_content DESC
                ''')
                
                results = cursor.fetchall()
                stats = {}
                
                for row in results:
                    stats[row['category']] = {
                        'total_content': row['total_content'],
                        'fresh_content': row['fresh_content'],
                        'avg_relevance': float(row['avg_relevance']) if row['avg_relevance'] else 0.0,
                        'latest_content': row['latest_content'].isoformat() if row['latest_content'] else None
                    }
                
                return stats
                
            except Exception as e:
                print(f"Category stats query failed: {e}")
                return {}

# Global retriever instance
_marketing_retriever = None

def get_marketing_retriever() -> MarketingKnowledgeRetriever:
    """Get or create global marketing retriever instance"""
    global _marketing_retriever
    if _marketing_retriever is None:
        _marketing_retriever = MarketingKnowledgeRetriever()
    return _marketing_retriever

# Convenience functions for easy integration
def get_seo_advice(topic: str = None) -> List[Dict[str, Any]]:
    """Get SEO advice - convenience function"""
    retriever = get_marketing_retriever()
    return retriever.get_seo_best_practices(topic)

def get_content_writing_tips(content_type: str = "blog") -> List[Dict[str, Any]]:
    """Get content writing tips - convenience function"""
    retriever = get_marketing_retriever()
    return retriever.get_content_writing_tips(content_type)

def get_social_media_advice(platform: str = None) -> List[Dict[str, Any]]:
    """Get social media advice - convenience function"""
    retriever = get_marketing_retriever()
    return retriever.get_social_media_strategies(platform)

def search_marketing_knowledge(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search marketing knowledge - convenience function"""
    retriever = get_marketing_retriever()
    return retriever.get_contextual_marketing_advice(query, limit)

def get_fresh_marketing_updates(days: int = 7) -> List[Dict[str, Any]]:
    """Get fresh marketing updates - convenience function"""
    retriever = get_marketing_retriever()
    return retriever.get_fresh_marketing_insights(days)
