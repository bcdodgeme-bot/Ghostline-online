# modules/rss_marketing_monitor.py - RSS Marketing Best Practices Monitor
import os
import time
import datetime
import threading
import requests
import feedparser
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import re
import hashlib
from urllib.parse import urljoin, urlparse

# AI/ML imports for content analysis
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class MarketingRSSMonitor:
    def __init__(self):
        self.db_url = self._get_database_url()
        self.openai_client = self._init_openai()
        self.running = False
        self.monitor_thread = None
        
        # Content categorization patterns
        self.category_patterns = {
            'seo': [
                'search engine optimization', 'SEO', 'SERP', 'keyword research', 'backlink',
                'organic search', 'google algorithm', 'page rank', 'meta description',
                'title tag', 'schema markup', 'technical seo', 'local seo', 'rank math'
            ],
            'content_marketing': [
                'content marketing', 'content strategy', 'blog writing', 'storytelling',
                'editorial calendar', 'content creation', 'brand voice', 'copywriting',
                'content optimization', 'content distribution', 'content planning'
            ],
            'social_media': [
                'social media', 'facebook marketing', 'instagram', 'twitter', 'linkedin',
                'social media strategy', 'social engagement', 'influencer marketing',
                'social analytics', 'social media calendar', 'community management'
            ],
            'analytics': [
                'google analytics', 'marketing analytics', 'conversion tracking', 'KPIs',
                'marketing metrics', 'ROI measurement', 'attribution modeling',
                'data analysis', 'marketing dashboard', 'performance tracking'
            ]
        }
        
        # Subcategory patterns for more specific classification
        self.subcategory_patterns = {
            'on-page-seo': ['on-page', 'meta tags', 'title optimization', 'internal linking'],
            'off-page-seo': ['backlinks', 'link building', 'domain authority', 'off-page'],
            'technical-seo': ['site speed', 'core web vitals', 'crawling', 'indexing', 'structured data'],
            'local-seo': ['local search', 'google my business', 'local citations', 'local ranking'],
            'blog-writing': ['blog post', 'article writing', 'editorial', 'publishing'],
            'video-marketing': ['video content', 'youtube', 'video seo', 'video strategy'],
            'email-marketing': ['email campaign', 'newsletter', 'email automation', 'email list'],
            'paid-advertising': ['google ads', 'facebook ads', 'ppc', 'advertising', 'ad copy'],
            'conversion-optimization': ['conversion rate', 'CRO', 'landing page', 'A/B testing']
        }
    
    def _get_database_url(self) -> str:
        """Get database URL with Railway compatibility"""
        db_url = os.getenv('DATABASE_URL')
        if db_url and db_url.startswith('postgres://'):
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        return db_url
    
    def _init_openai(self):
        """Initialize OpenAI client if available"""
        if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        return None
    
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
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from content"""
        text_lower = text.lower()
        keywords = []
        
        # Extract from all category patterns
        for category, patterns in self.category_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    keywords.append(pattern)
        
        # Extract from subcategory patterns
        for subcat, patterns in self.subcategory_patterns.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    keywords.append(pattern)
        
        # Remove duplicates and return
        return list(set(keywords))
    
    def categorize_content(self, title: str, content: str) -> tuple[str, str]:
        """Categorize content into main category and subcategory"""
        text = f"{title} {content}".lower()
        
        # Score each category
        category_scores = {}
        for category, patterns in self.category_patterns.items():
            score = sum(1 for pattern in patterns if pattern.lower() in text)
            if score > 0:
                category_scores[category] = score
        
        # Get primary category
        main_category = max(category_scores.items(), key=lambda x: x[1])[0] if category_scores else 'general'
        
        # Find best subcategory
        subcategory_scores = {}
        for subcat, patterns in self.subcategory_patterns.items():
            score = sum(1 for pattern in patterns if pattern.lower() in text)
            if score > 0:
                subcategory_scores[subcat] = score
        
        subcategory = max(subcategory_scores.items(), key=lambda x: x[1])[0] if subcategory_scores else None
        
        return main_category, subcategory
    
    def calculate_relevance_score(self, title: str, content: str, category: str) -> float:
        """Calculate relevance score (1-10) based on content quality indicators"""
        score = 5.0  # Base score
        
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Boost for actionable content
        actionable_terms = ['how to', 'guide', 'tutorial', 'step by step', 'best practices', 'tips', 'strategies']
        score += sum(0.5 for term in actionable_terms if term in title_lower or term in content_lower)
        
        # Boost for current year content
        current_year = str(datetime.datetime.now().year)
        if current_year in title or current_year in content:
            score += 1.0
        
        # Boost for comprehensive content (longer articles)
        if len(content) > 2000:
            score += 1.0
        elif len(content) > 1000:
            score += 0.5
        
        # Boost for specific tools/platforms mentioned
        tools = ['google', 'facebook', 'instagram', 'linkedin', 'youtube', 'wordpress', 'shopify']
        score += sum(0.3 for tool in tools if tool in content_lower) * 0.5  # Max 1.5 boost
        
        # Penalty for very short content
        if len(content) < 200:
            score -= 1.0
        
        # Ensure score is within bounds
        return max(1.0, min(10.0, score))
    
    def generate_ai_summary(self, title: str, content: str) -> Optional[str]:
        """Generate AI summary if OpenAI is available"""
        if not self.openai_client:
            return None
        
        try:
            # Truncate content if too long for API
            content_excerpt = content[:2000] if len(content) > 2000 else content
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a marketing expert. Summarize this marketing/SEO content in 2-3 sentences, focusing on the key actionable insights and best practices."
                    },
                    {
                        "role": "user",
                        "content": f"Title: {title}\n\nContent: {content_excerpt}"
                    }
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"AI summary generation failed: {e}")
            return None
    
    def fetch_and_parse_feed(self, feed_url: str) -> List[Dict[str, Any]]:
        """Fetch and parse RSS feed"""
        try:
            print(f"Fetching RSS feed: {feed_url}")
            
            # Set user agent to avoid being blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(feed_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse feed
            feed = feedparser.parse(response.content)
            
            if feed.bozo:
                print(f"Feed parsing warning for {feed_url}: {feed.bozo_exception}")
            
            items = []
            for entry in feed.entries[:20]:  # Limit to 20 most recent items
                # Extract content
                content = ""
                if hasattr(entry, 'content') and entry.content:
                    content = entry.content[0].value if isinstance(entry.content, list) else entry.content
                elif hasattr(entry, 'description'):
                    content = entry.description
                elif hasattr(entry, 'summary'):
                    content = entry.summary
                
                # Clean HTML tags
                content = re.sub(r'<[^>]+>', '', content)
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Extract published date
                published_date = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_date = datetime.datetime(*entry.published_parsed[:6])
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    published_date = datetime.datetime(*entry.updated_parsed[:6])
                
                items.append({
                    'title': getattr(entry, 'title', 'No Title'),
                    'content': content,
                    'url': getattr(entry, 'link', ''),
                    'author': getattr(entry, 'author', ''),
                    'published_date': published_date,
                    'guid': getattr(entry, 'id', getattr(entry, 'link', ''))
                })
            
            print(f"Parsed {len(items)} items from {feed_url}")
            return items
        
        except Exception as e:
            print(f"Failed to fetch/parse feed {feed_url}: {e}")
            return []
    
    def process_feed_item(self, item: Dict[str, Any], source_id: int, source_category: str) -> bool:
        """Process and store a single feed item"""
        try:
            # Skip items with insufficient content
            if len(item.get('content', '')) < 100:
                return False
            
            # Categorize content
            main_category, subcategory = self.categorize_content(item['title'], item['content'])
            
            # Use source category as fallback
            if main_category == 'general':
                main_category = source_category
            
            # Extract keywords
            keywords = self.extract_keywords(f"{item['title']} {item['content']}")
            
            # Calculate relevance score with item data for potential boosts
            relevance_score = self.calculate_relevance_score(
                item['title'],
                item['content'],
                main_category,
                item  # Pass item data for relevance boosts
            )
            
            # Generate AI summary
            summary = self.generate_ai_summary(item['title'], item['content'])
            
            # Determine content type
            title_lower = item['title'].lower()
            if 'guide' in title_lower or 'tutorial' in title_lower:
                content_type = 'guide'
            elif 'case study' in title_lower:
                content_type = 'case_study'
            elif 'news' in title_lower or 'update' in title_lower or 'announcement' in title_lower:
                content_type = 'news'
            elif 'how to' in title_lower:
                content_type = 'tutorial'
            elif 'checklist' in title_lower:
                content_type = 'checklist'
            elif 'framework' in title_lower or 'strategy' in title_lower:
                content_type = 'framework'
            else:
                content_type = 'article'
            
            # Store in database
            with self.get_db_connection() as conn:
                if not conn:
                    return False
                
                cursor = conn.cursor()
                
                # Check if item already exists (by GUID or URL)
                cursor.execute('''
                    SELECT id FROM marketing_best_practices 
                    WHERE guid = %s OR url = %s
                ''', (item['guid'], item['url']))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing item with improved data
                    cursor.execute('''
                        UPDATE marketing_best_practices 
                        SET title = %s, content = %s, summary = %s, author = %s,
                            published_date = %s, category = %s, subcategory = %s,
                            keywords = %s, relevance_score = %s, content_type = %s,
                            is_fresh = true, last_updated = CURRENT_TIMESTAMP
                        WHERE id = %s
                    ''', (
                        item['title'][:500], item['content'], summary, item['author'][:200],
                        item['published_date'], main_category, subcategory,
                        keywords, relevance_score, content_type, existing[0]
                    ))
                    print(f"Updated existing item: {item['title'][:50]}... (Score: {relevance_score:.1f})")
                else:
                    # Insert new item
                    cursor.execute('''
                        INSERT INTO marketing_best_practices 
                        (rss_source_id, title, content, summary, url, author, published_date,
                         category, subcategory, keywords, relevance_score, content_type, guid)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ''', (
                        source_id, item['title'][:500], item['content'], summary, item['url'],
                        item['author'][:200], item['published_date'], main_category, subcategory,
                        keywords, relevance_score, content_type, item['guid']
                    ))
                    print(f"Inserted new item: {item['title'][:50]}... (Score: {relevance_score:.1f})")
                
                conn.commit()
                return True
        
        except Exception as e:
            print(f"Failed to process feed item: {e}")
            return False
    
    def update_source_status(self, source_id: int, success: bool, error: str = None):
        """Update RSS source fetch status"""
        with self.get_db_connection() as conn:
            if not conn:
                return
            
            cursor = conn.cursor()
            
            if success:
                cursor.execute('''
                    UPDATE rss_sources 
                    SET last_fetched = CURRENT_TIMESTAMP, error_count = 0, last_error = NULL
                    WHERE id = %s
                ''', (source_id,))
            else:
                cursor.execute('''
                    UPDATE rss_sources 
                    SET error_count = error_count + 1, last_error = %s
                    WHERE id = %s
                ''', (error, source_id))
            
            conn.commit()
    
    def fetch_all_feeds(self):
        """Fetch all active RSS feeds"""
        with self.get_db_connection() as conn:
            if not conn:
                print("No database connection for feed fetching")
                return
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get active RSS sources that haven't been fetched recently
            cursor.execute('''
                SELECT id, feed_name, feed_url, category, fetch_interval, error_count
                FROM rss_sources 
                WHERE active = true 
                AND (last_fetched IS NULL OR 
                     last_fetched < CURRENT_TIMESTAMP - INTERVAL '1 second' * fetch_interval)
                AND error_count < 5
                ORDER BY last_fetched ASC NULLS FIRST
            ''')
            
            sources = cursor.fetchall()
            
            print(f"Found {len(sources)} RSS sources to update")
            
            for source in sources:
                try:
                    print(f"\nProcessing: {source['feed_name']}")
                    
                    # Fetch feed items
                    items = self.fetch_and_parse_feed(source['feed_url'])
                    
                    if items:
                        processed_count = 0
                        for item in items:
                            if self.process_feed_item(item, source['id'], source['category']):
                                processed_count += 1
                        
                        print(f"Processed {processed_count}/{len(items)} items from {source['feed_name']}")
                        self.update_source_status(source['id'], True)
                    else:
                        print(f"No items found in {source['feed_name']}")
                        self.update_source_status(source['id'], False, "No items found")
                
                except Exception as e:
                    print(f"Failed to process source {source['feed_name']}: {e}")
                    self.update_source_status(source['id'], False, str(e))
                
                # Small delay between feeds to be respectful
                time.sleep(2)
    
    def cleanup_old_content(self, days_old: int = 90):
        """Remove very old content to prevent database bloat"""
        with self.get_db_connection() as conn:
            if not conn:
                return
            
            cursor = conn.cursor()
            
            # Delete content older than specified days with low relevance
            cursor.execute('''
                DELETE FROM marketing_best_practices 
                WHERE published_date < CURRENT_DATE - INTERVAL '%s days'
                AND relevance_score < 4.0
            ''', (days_old,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                print(f"Cleaned up {deleted_count} old marketing content items")
    
    def monitor_loop(self):
        """Main monitoring loop - runs weekly for marketing content"""
        print("Starting RSS marketing monitor loop (weekly schedule)")
        
        while self.running:
            try:
                # Fetch all feeds
                self.fetch_all_feeds()
                
                # Update content freshness
                with self.get_db_connection() as conn:
                    if conn:
                        cursor = conn.cursor()
                        cursor.execute('SELECT update_content_freshness()')
                        conn.commit()
                
                # Cleanup old content (during weekly run)
                self.cleanup_old_content(days_old=90)  # Keep 3 months of content
                
                print(f"RSS monitor cycle completed. Sleeping for 1 week...")
                
                # Sleep for 1 week between cycles (604800 seconds = 7 days)
                for _ in range(604800):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                print(f"RSS monitor loop error: {e}")
                time.sleep(3600)  # 1 hour error recovery delay
    
    def start_monitoring(self):
        """Start background RSS monitoring"""
        if self.running:
            print("RSS monitor already running")
            return False
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print("RSS marketing monitor started")
        return True
    
    def stop_monitoring(self):
        """Stop background RSS monitoring"""
        if not self.running:
            return False
        
        self.running = False
        if self.monitor_thread:
            print("Stopping RSS marketing monitor...")
            # Thread will stop on next cycle check
        
        return True
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """Get current monitor status"""
        status = {
            'running': self.running,
            'openai_available': bool(self.openai_client),
            'database_connected': bool(self.db_url)
        }
        
        # Get source statistics
        with self.get_db_connection() as conn:
            if conn:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                # Source counts
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_sources,
                        COUNT(*) FILTER (WHERE active = true) as active_sources,
                        COUNT(*) FILTER (WHERE error_count > 0) as error_sources
                    FROM rss_sources
                ''')
                source_stats = cursor.fetchone()
                status.update(source_stats)
                
                # Content statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_content,
                        COUNT(*) FILTER (WHERE is_fresh = true) as fresh_content,
                        AVG(relevance_score) as avg_relevance,
                        MAX(fetch_date) as last_content_fetch
                    FROM marketing_best_practices
                ''')
                content_stats = cursor.fetchone()
                status.update(content_stats)
        
        return status

# Global monitor instance
_rss_monitor = None

def get_rss_monitor() -> MarketingRSSMonitor:
    """Get or create global RSS monitor instance"""
    global _rss_monitor
    if _rss_monitor is None:
        _rss_monitor = MarketingRSSMonitor()
    return _rss_monitor

def start_rss_monitoring():
    """Start RSS monitoring service"""
    monitor = get_rss_monitor()
    return monitor.start_monitoring()

def stop_rss_monitoring():
    """Stop RSS monitoring service"""
    monitor = get_rss_monitor()
    return monitor.stop_monitoring()

def get_rss_status():
    """Get RSS monitoring status"""
    monitor = get_rss_monitor()
    return monitor.get_monitor_status()

def force_feed_update():
    """Force immediate feed update"""
    monitor = get_rss_monitor()
    if monitor.db_url:
        monitor.fetch_all_feeds()
        return True
    return False
