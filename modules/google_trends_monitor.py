# modules/google_trends_monitor.py
# Google Trends monitoring system with content opportunity detection
# Integrates with existing keyword database and Telegram alerts

import os
import json
import time
import datetime
import threading
from typing import List, Dict, Optional, Tuple
import requests
import psycopg2.extras
from dataclasses import dataclass
from modules.database import get_db_connection

# Import pytrends for Google Trends API
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    print("PyTrends not available. Install with: pip install pytrends")
    PYTRENDS_AVAILABLE = False

# Import existing systems
from modules.site_keyword_manager import SITE_DOMAINS
from modules.telegram_notifications import GhostlineTelegramReminders

@dataclass
class TrendingTopic:
    """Data class for trending topic information"""
    query: str
    search_volume: int
    spike_percentage: float
    category: str
    geo: str = "US"
    timestamp: datetime.datetime = None
    related_queries: List[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now()
        if self.related_queries is None:
            self.related_queries = []

@dataclass
class ContentOpportunity:
    """Data class for detected content opportunities"""
    trending_topic: TrendingTopic
    best_site: str
    confidence_score: float
    matched_keywords: List[Dict]
    suggested_title: str
    competition_level: str
    total_search_volume: int
    reasoning: str
    opportunity_id: str = None
    
    def __post_init__(self):
        if self.opportunity_id is None:
            import hashlib
            content = f"{self.trending_topic.query}{self.best_site}{self.trending_topic.timestamp}"
            self.opportunity_id = hashlib.md5(content.encode()).hexdigest()[:12]

class GoogleTrendsMonitor:
    """Main Google Trends monitoring class"""
    
    def __init__(self):
        self.pytrends = None
        self.running = False
        self.last_check = None
        self.monitor_thread = None
        self.telegram_bot = None
        
        # Rate limiting
        self.requests_today = 0
        self.last_request_date = datetime.date.today()
        self.max_requests_per_day = 100
        
        # Alert limiting
        self.alerts_sent_today = 0
        self.max_alerts_per_day = 5
        
        # Initialize components
        self._init_pytrends()
        self._init_telegram()
        self._ensure_database_tables()
    
    def _init_pytrends(self):
        """Initialize PyTrends with error handling"""
        if not PYTRENDS_AVAILABLE:
            return False
        
        try:
            self.pytrends = TrendReq(
                hl='en-US',
                tz=360,  # Eastern Time
                timeout=(10, 25),
                retries=2,
                backoff_factor=0.1,
                requests_args={'verify': False}
            )
            return True
        except Exception as e:
            self._log_error(f"Failed to initialize PyTrends: {e}")
            return False
    
    def _init_telegram(self):
        """Initialize Telegram bot for alerts"""
        try:
            from modules.telegram_notifications import is_telegram_configured
            if is_telegram_configured():
                self.telegram_bot = GhostlineTelegramReminders()
        except Exception as e:
            self._log_error(f"Failed to initialize Telegram: {e}")
    
    def _ensure_database_tables(self):
        """Create necessary database tables"""
        try:
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    
                    # Trending topics table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS trending_topics (
                            id SERIAL PRIMARY KEY,
                            query VARCHAR(500) NOT NULL,
                            search_volume INTEGER,
                            spike_percentage DECIMAL(10,2),
                            category VARCHAR(100),
                            geo VARCHAR(10) DEFAULT 'US',
                            related_queries JSONB DEFAULT '[]',
                            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            processed BOOLEAN DEFAULT FALSE,
                            metadata JSONB DEFAULT '{}'
                        )
                    ''')
                    
                    # Content opportunities table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS content_opportunities (
                            id SERIAL PRIMARY KEY,
                            opportunity_id VARCHAR(50) UNIQUE NOT NULL,
                            trending_query VARCHAR(500) NOT NULL,
                            best_site VARCHAR(100) NOT NULL,
                            confidence_score DECIMAL(5,2) NOT NULL,
                            matched_keywords JSONB DEFAULT '[]',
                            suggested_title VARCHAR(500),
                            competition_level VARCHAR(50),
                            total_search_volume INTEGER,
                            reasoning TEXT,
                            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            alert_sent BOOLEAN DEFAULT FALSE,
                            user_response VARCHAR(50),
                            user_feedback TEXT,
                            content_created BOOLEAN DEFAULT FALSE,
                            performance_score DECIMAL(3,1),
                            metadata JSONB DEFAULT '{}'
                        )
                    ''')
                    
                    # Trends monitoring stats
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS trends_monitor_stats (
                            id SERIAL PRIMARY KEY,
                            check_date DATE DEFAULT CURRENT_DATE,
                            trends_checked INTEGER DEFAULT 0,
                            opportunities_found INTEGER DEFAULT 0,
                            alerts_sent INTEGER DEFAULT 0,
                            api_requests INTEGER DEFAULT 0,
                            errors_count INTEGER DEFAULT 0,
                            avg_confidence_score DECIMAL(5,2),
                            metadata JSONB DEFAULT '{}'
                        )
                    ''')
                    
                    # Create indexes
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_trending_topics_detected 
                        ON trending_topics (detected_at, processed)
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_content_opportunities_alert 
                        ON content_opportunities (alert_sent, detected_at)
                    ''')
                    
                    conn.commit()
                    
        except Exception as e:
            self._log_error(f"Failed to create database tables: {e}")
    
    def _log_error(self, message: str):
        """Safe logging without Flask context dependency"""
        try:
            from flask import current_app
            current_app.logger.error(message)
        except RuntimeError:
            print(f"TRENDS ERROR: {message}")
    
    def _log_info(self, message: str):
        """Safe info logging"""
        try:
            from flask import current_app
            current_app.logger.info(message)
        except RuntimeError:
            print(f"TRENDS INFO: {message}")

class TrendsDataCollector:
    """Handles data collection from Google Trends"""
    
    def __init__(self, monitor):
        self.monitor = monitor
        self.categories = {
            'business': 12,      # Business & Industrial
            'technology': 5,     # Computers & Electronics  
            'news': 16,          # News
            'health': 45,        # Health
            'marketing': 12,     # Business (includes marketing)
        }
    
    def get_trending_searches(self, geo: str = "US", max_results: int = 20) -> List[TrendingTopic]:
        """Get currently trending searches"""
        if not self.monitor.pytrends:
            return []
        
        trending_topics = []
        
        try:
            # Check rate limits
            if not self._check_rate_limits():
                self.monitor._log_error("Rate limit exceeded, skipping trends check")
                return []
            
            # Get trending searches by category
            for category_name, category_id in self.categories.items():
                try:
                    # Get trending searches for this category
                    trending_searches = self.monitor.pytrends.trending_searches(pn=geo)
                    
                    if trending_searches is not None and not trending_searches.empty:
                        for idx, query in enumerate(trending_searches[0][:max_results//len(self.categories)]):
                            # Get interest over time for spike detection
                            spike_data = self._get_spike_data(query, geo)
                            
                            if spike_data['spike_percentage'] > 50:  # Only significant spikes
                                topic = TrendingTopic(
                                    query=query,
                                    search_volume=spike_data['current_volume'],
                                    spike_percentage=spike_data['spike_percentage'],
                                    category=category_name,
                                    geo=geo,
                                    related_queries=spike_data['related_queries']
                                )
                                trending_topics.append(topic)
                    
                    # Rate limiting delay
                    time.sleep(2)
                    
                except Exception as e:
                    self.monitor._log_error(f"Failed to get trends for {category_name}: {e}")
                    continue
        
        except Exception as e:
            self.monitor._log_error(f"Trending searches collection failed: {e}")
        
        return trending_topics
    
    def _get_spike_data(self, query: str, geo: str) -> Dict:
        """Get detailed spike data for a query"""
        try:
            # Build payload for interest over time
            self.monitor.pytrends.build_payload([query], timeframe='now 7-d', geo=geo)
            
            # Get interest over time data
            interest_df = self.monitor.pytrends.interest_over_time()
            
            if interest_df.empty:
                return {
                    'current_volume': 0,
                    'spike_percentage': 0,
                    'related_queries': []
                }
            
            # Calculate spike percentage
            recent_values = interest_df[query].tail(3).values
            earlier_values = interest_df[query].head(3).values
            
            current_avg = recent_values.mean() if len(recent_values) > 0 else 0
            baseline_avg = earlier_values.mean() if len(earlier_values) > 0 else 1
            
            spike_percentage = ((current_avg - baseline_avg) / baseline_avg * 100) if baseline_avg > 0 else 0
            
            # Get related queries
            related_queries = []
            try:
                related_df = self.monitor.pytrends.related_queries()
                if query in related_df and related_df[query]['rising'] is not None:
                    related_queries = related_df[query]['rising']['query'].head(5).tolist()
            except:
                pass
            
            return {
                'current_volume': int(current_avg),
                'spike_percentage': float(spike_percentage),
                'related_queries': related_queries
            }
            
        except Exception as e:
            self.monitor._log_error(f"Failed to get spike data for {query}: {e}")
            return {
                'current_volume': 0,
                'spike_percentage': 0,
                'related_queries': []
            }
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        today = datetime.date.today()
        
        # Reset daily counter if new day
        if today != self.monitor.last_request_date:
            self.monitor.requests_today = 0
            self.monitor.alerts_sent_today = 0
            self.monitor.last_request_date = today
        
        # Check if we're under the daily limit
        if self.monitor.requests_today >= self.monitor.max_requests_per_day:
            return False
        
        self.monitor.requests_today += 1
        return True

class ContentOpportunityDetector:
    """Detects content opportunities by matching trends with keywords"""
    
    def __init__(self, monitor):
        self.monitor = monitor
        self.confidence_threshold = 15.0  # Minimum confidence for alerts
    
    def analyze_trending_topics(self, trending_topics: List[TrendingTopic]) -> List[ContentOpportunity]:
        """Analyze trending topics for content opportunities"""
        opportunities = []
        
        for topic in trending_topics:
            try:
                # Match against existing keywords
                matches = self._match_against_keywords(topic)
                
                if matches:
                    best_match = max(matches, key=lambda x: x['confidence'])
                    
                    if best_match['confidence'] >= self.confidence_threshold:
                        opportunity = self._create_opportunity(topic, best_match, matches)
                        opportunities.append(opportunity)
                        
            except Exception as e:
                self.monitor._log_error(f"Failed to analyze topic '{topic.query}': {e}")
        
        return opportunities
    
    def _match_against_keywords(self, topic: TrendingTopic) -> List[Dict]:
        """Match trending topic against existing keywords"""
        matches = []
        
        try:
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    
                    # Get all keywords for matching
                    cursor.execute('''
                        SELECT site_domain, keyword, search_volume, competition_level,
                               keyword_category, match_score, times_used
                        FROM site_keywords 
                        WHERE active = TRUE
                        ORDER BY search_volume DESC
                    ''')
                    
                    keywords = cursor.fetchall()
                    
                    for row in keywords:
                        site_domain, keyword, search_volume, competition_level, category, match_score, times_used = row
                        
                        # Calculate similarity score
                        confidence = self._calculate_similarity(topic.query, keyword, topic.related_queries)
                        
                        if confidence >= 10.0:  # Minimum relevance threshold
                            site_info = SITE_DOMAINS.get(site_domain, {})
                            
                            matches.append({
                                'site_domain': site_domain,
                                'site_name': site_info.get('name', site_domain),
                                'keyword': keyword,
                                'confidence': confidence,
                                'search_volume': search_volume or 0,
                                'competition_level': competition_level or 'unknown',
                                'category': category,
                                'existing_match_score': match_score or 0,
                                'times_used': times_used or 0
                            })
        
        except Exception as e:
            self.monitor._log_error(f"Failed to match keywords: {e}")
        
        return sorted(matches, key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_similarity(self, trend_query: str, keyword: str, related_queries: List[str]) -> float:
        """Calculate similarity score between trend and keyword"""
        import difflib
        
        # Direct similarity
        direct_similarity = difflib.SequenceMatcher(None, trend_query.lower(), keyword.lower()).ratio() * 100
        
        # Word overlap similarity
        trend_words = set(trend_query.lower().split())
        keyword_words = set(keyword.lower().split())
        
        common_words = trend_words & keyword_words
        total_words = len(trend_words | keyword_words)
        
        word_overlap = (len(common_words) / total_words * 100) if total_words > 0 else 0
        
        # Related queries bonus
        related_bonus = 0
        for related in related_queries[:3]:  # Check top 3 related queries
            related_sim = difflib.SequenceMatcher(None, related.lower(), keyword.lower()).ratio() * 10
            related_bonus = max(related_bonus, related_sim)
        
        # Combined score with weights
        final_score = (
            direct_similarity * 0.4 +
            word_overlap * 0.5 +
            related_bonus * 0.1
        )
        
        return min(final_score, 100.0)
    
    def _create_opportunity(self, topic: TrendingTopic, best_match: Dict, all_matches: List[Dict]) -> ContentOpportunity:
        """Create a content opportunity from matches"""
        
        # Generate suggested title
        suggested_title = self._generate_title_suggestion(topic, best_match)
        
        # Calculate total search volume from top matches
        top_matches = [m for m in all_matches if m['site_domain'] == best_match['site_domain']][:3]
        total_volume = sum(m['search_volume'] for m in top_matches)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(topic, best_match, len(top_matches))
        
        return ContentOpportunity(
            trending_topic=topic,
            best_site=best_match['site_domain'],
            confidence_score=best_match['confidence'],
            matched_keywords=top_matches,
            suggested_title=suggested_title,
            competition_level=best_match['competition_level'],
            total_search_volume=total_volume,
            reasoning=reasoning
        )
    
    def _generate_title_suggestion(self, topic: TrendingTopic, best_match: Dict) -> str:
        """Generate content title suggestion"""
        
        site_info = SITE_DOMAINS.get(best_match['site_domain'], {})
        focus_areas = site_info.get('focus_areas', [])
        
        # Title templates based on site focus
        templates = {
            'productivity': [
                f"How {topic.query.title()} Is Changing Productivity",
                f"The Hidden Impact of {topic.query.title()} on Work Efficiency",
                f"Why {topic.query.title()} Might Be Making You Less Productive"
            ],
            'health': [
                f"The Health Truth About {topic.query.title()}",
                f"What {topic.query.title()} Really Does to Your Wellbeing",
                f"{topic.query.title()}: Benefits vs. Hidden Risks"
            ],
            'business': [
                f"How Smart Businesses Are Using {topic.query.title()}",
                f"The {topic.query.title()} Strategy That's Actually Working",
                f"Why {topic.query.title()} Is the Future of Business"
            ],
            'technology': [
                f"The Real Story Behind {topic.query.title()}",
                f"What Tech Experts Won't Tell You About {topic.query.title()}",
                f"{topic.query.title()}: Hype vs. Reality"
            ]
        }
        
        # Pick template based on site focus
        for focus in focus_areas:
            if focus.lower() in templates:
                import random
                return random.choice(templates[focus.lower()])
        
        # Default template
        return f"The Complete Guide to {topic.query.title()}"
    
    def _generate_reasoning(self, topic: TrendingTopic, best_match: Dict, match_count: int) -> str:
        """Generate reasoning for the match"""
        
        reasoning_parts = []
        
        reasoning_parts.append(
            f"Trending query '{topic.query}' shows {topic.spike_percentage:.0f}% spike in searches"
        )
        
        reasoning_parts.append(
            f"Best keyword match: '{best_match['keyword']}' ({best_match['confidence']:.1f}% similarity)"
        )
        
        reasoning_parts.append(
            f"Site '{best_match['site_name']}' has {match_count} relevant keywords"
        )
        
        if best_match['search_volume'] > 1000:
            reasoning_parts.append(f"High search volume: {best_match['search_volume']:,} monthly searches")
        
        if best_match['competition_level'] == 'low':
            reasoning_parts.append("Low competition level - good opportunity for ranking")
        
        return ". ".join(reasoning_parts) + "."

class TelegramAlertSystem:
    """Handles sending content opportunity alerts via Telegram"""
    
    def __init__(self, monitor):
        self.monitor = monitor
        self.alert_template = self._load_alert_template()
    
    def send_opportunity_alert(self, opportunity: ContentOpportunity) -> bool:
        """Send content opportunity alert to Telegram"""
        
        if not self.monitor.telegram_bot:
            self.monitor._log_error("Telegram bot not available for alerts")
            return False
        
        # Check daily alert limit
        if self.monitor.alerts_sent_today >= self.monitor.max_alerts_per_day:
            self.monitor._log_info("Daily alert limit reached, skipping alert")
            return False
        
        try:
            # Format alert message
            message = self._format_alert_message(opportunity)
            
            # Create action buttons
            reply_markup = {
                "inline_keyboard": [
                    [
                        {"text": "📝 Generate Draft", "callback_data": f"draft_{opportunity.opportunity_id}"},
                        {"text": "⏭️ Skip", "callback_data": f"skip_{opportunity.opportunity_id}"}
                    ],
                    [
                        {"text": "🔄 Wrong Site", "callback_data": f"wrong_{opportunity.opportunity_id}"},
                        {"text": "📊 More Data", "callback_data": f"data_{opportunity.opportunity_id}"}
                    ]
                ]
            }
            
            # Send via Telegram
            result = self.monitor.telegram_bot.bot.send_message(
                message, 
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
            
            if result.get("success"):
                # Update database
                self._record_alert_sent(opportunity, result.get("message_id"))
                self.monitor.alerts_sent_today += 1
                self.monitor._log_info(f"Content opportunity alert sent: {opportunity.trending_topic.query}")
                return True
            else:
                self.monitor._log_error(f"Failed to send alert: {result.get('error')}")
                return False
                
        except Exception as e:
            self.monitor._log_error(f"Alert sending failed: {e}")
            return False
    
    def _format_alert_message(self, opportunity: ContentOpportunity) -> str:
        """Format the alert message"""
        
        topic = opportunity.trending_topic
        site_info = SITE_DOMAINS.get(opportunity.best_site, {})
        
        # Build message components
        header = "🚨 *Content Opportunity Detected*"
        
        trending_info = (
            f"*Trending:* {topic.query} "
            f"(+{topic.spike_percentage:.0f}% search volume)"
        )
        
        match_info = (
            f"*Best Match:* {site_info.get('name', opportunity.best_site)} "
            f"({opportunity.confidence_score:.0f}% confidence)"
        )
        
        # Top keywords
        keywords_text = "*Your Keywords:* "
        top_keywords = opportunity.matched_keywords[:2]  # Top 2 keywords
        keyword_parts = []
        
        for kw in top_keywords:
            volume_text = f"{kw['search_volume']:,}" if kw['search_volume'] > 0 else "N/A"
            keyword_parts.append(f'"{kw["keyword"]}" ({volume_text} searches)')
        
        keywords_text += ", ".join(keyword_parts)
        
        competition_info = f"*Competition:* {opportunity.competition_level.title()}"
        
        suggestion = f"*Suggested:* {opportunity.suggested_title}"
        
        # Reasoning
        context = f"*Why:* {opportunity.reasoning}"
        
        # Combine all parts
        message_parts = [
            header,
            trending_info,
            match_info,
            keywords_text,
            competition_info,
            suggestion,
            context
        ]
        
        return "\n".join(message_parts)
    
    def _record_alert_sent(self, opportunity: ContentOpportunity, message_id: int):
        """Record that alert was sent"""
        try:
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    
                    # Store opportunity in database
                    cursor.execute('''
                        INSERT INTO content_opportunities 
                        (opportunity_id, trending_query, best_site, confidence_score,
                         matched_keywords, suggested_title, competition_level,
                         total_search_volume, reasoning, alert_sent, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (opportunity_id) DO UPDATE SET
                        alert_sent = TRUE, metadata = %s
                    ''', (
                        opportunity.opportunity_id,
                        opportunity.trending_topic.query,
                        opportunity.best_site,
                        opportunity.confidence_score,
                        psycopg2.extras.Json(opportunity.matched_keywords),
                        opportunity.suggested_title,
                        opportunity.competition_level,
                        opportunity.total_search_volume,
                        opportunity.reasoning,
                        True,
                        psycopg2.extras.Json({
                            'message_id': message_id,
                            'category': opportunity.trending_topic.category,
                            'spike_percentage': opportunity.trending_topic.spike_percentage,
                            'related_queries': opportunity.trending_topic.related_queries
                        }),
                        psycopg2.extras.Json({'message_id': message_id})
                    ))
                    
                    conn.commit()
                    
        except Exception as e:
            self.monitor._log_error(f"Failed to record alert: {e}")
    
    def _load_alert_template(self) -> str:
        """Load alert template (placeholder for future customization)"""
        return "default"

    def process_alert_callback(self, callback_data: str, callback_query) -> Dict:
        """Process user responses to alert buttons"""
        try:
            action, opportunity_id = callback_data.split('_', 1)
            
            if action == 'draft':
                return self._handle_draft_request(opportunity_id)
            elif action == 'skip':
                return self._handle_skip_response(opportunity_id)
            elif action == 'wrong':
                return self._handle_wrong_site_response(opportunity_id)
            elif action == 'data':
                return self._handle_data_request(opportunity_id)
            
            return {"success": False, "error": "Unknown action"}
            
        except Exception as e:
            self.monitor._log_error(f"Callback processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _handle_draft_request(self, opportunity_id: str) -> Dict:
        """Handle draft generation request"""
        try:
            # Update opportunity status
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE content_opportunities 
                        SET user_response = 'approved', performance_score = 8.0
                        WHERE opportunity_id = %s
                    ''', (opportunity_id,))
                    conn.commit()
            
            # Send confirmation
            if self.monitor.telegram_bot:
                self.monitor.telegram_bot.bot.send_message(
                    "✅ *Content approved!* I'll help you create a draft for this opportunity."
                )
            
            return {"success": True, "action": "draft_approved"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_skip_response(self, opportunity_id: str) -> Dict:
        """Handle skip response"""
        try:
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE content_opportunities 
                        SET user_response = 'skipped', performance_score = 3.0
                        WHERE opportunity_id = %s
                    ''', (opportunity_id,))
                    conn.commit()
            
            if self.monitor.telegram_bot:
                self.monitor.telegram_bot.bot.send_message(
                    "⏭️ *Opportunity skipped.* Thanks for the feedback!"
                )
            
            return {"success": True, "action": "skipped"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_wrong_site_response(self, opportunity_id: str) -> Dict:
        """Handle wrong site feedback"""
        try:
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE content_opportunities 
                        SET user_response = 'wrong_site', performance_score = 1.0
                        WHERE opportunity_id = %s
                    ''', (opportunity_id,))
                    conn.commit()
            
            if self.monitor.telegram_bot:
                self.monitor.telegram_bot.bot.send_message(
                    "🔄 *Wrong site match noted.* This will improve future matching accuracy."
                )
            
            return {"success": True, "action": "wrong_site"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _handle_data_request(self, opportunity_id: str) -> Dict:
        """Handle request for more data"""
        try:
            # Get opportunity details
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT trending_query, confidence_score, matched_keywords, 
                               total_search_volume, metadata
                        FROM content_opportunities 
                        WHERE opportunity_id = %s
                    ''', (opportunity_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        query, confidence, keywords, volume, metadata = result
                        
                        # Format detailed data message
                        data_message = [
                            f"📊 *Detailed Analysis*",
                            f"*Query:* {query}",
                            f"*Confidence:* {confidence:.1f}%",
                            f"*Total Volume:* {volume:,} searches",
                            f"*Keywords Matched:* {len(keywords)}",
                            ""
                        ]
                        
                        # Add keyword details
                        data_message.append("*Top Keywords:*")
                        for i, kw in enumerate(keywords[:3], 1):
                            data_message.append(
                                f"{i}. \"{kw['keyword']}\" "
                                f"({kw['search_volume']:,} vol, {kw['confidence']:.1f}% match)"
                            )
                        
                        # Add metadata if available
                        if metadata and 'spike_percentage' in metadata:
                            data_message.append(f"\n*Spike:* +{metadata['spike_percentage']:.0f}%")
                        
                        if self.monitor.telegram_bot:
                            self.monitor.telegram_bot.bot.send_message(
                                "\n".join(data_message),
                                parse_mode='Markdown'
                            )
                        
                        return {"success": True, "action": "data_sent"}
            
            return {"success": False, "error": "Opportunity not found"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Main monitoring function
def run_trends_monitoring_cycle():
    """Run a complete trends monitoring cycle"""
    monitor = GoogleTrendsMonitor()
    
    if not monitor.pytrends:
        monitor._log_error("PyTrends not available, cannot run monitoring")
        return {"success": False, "error": "PyTrends not available"}
    
    try:
        monitor._log_info("Starting trends monitoring cycle")
        
        # Collect trending data
        collector = TrendsDataCollector(monitor)
        trending_topics = collector.get_trending_searches()
        
        monitor._log_info(f"Found {len(trending_topics)} trending topics")
        
        # Detect opportunities
        detector = ContentOpportunityDetector(monitor)
        opportunities = detector.analyze_trending_topics(trending_topics)
        
        monitor._log_info(f"Detected {len(opportunities)} content opportunities")
        
        # Send alerts for high-confidence opportunities
        alert_system = TelegramAlertSystem(monitor)
        alerts_sent = 0
        
        # Sort by confidence and send top opportunities
        opportunities.sort(key=lambda x: x.confidence_score, reverse=True)
        
        for opportunity in opportunities[:3]:  # Max 3 per cycle
            if opportunity.confidence_score >= 20.0:  # Higher threshold for auto-alerts
                if alert_system.send_opportunity_alert(opportunity):
                    alerts_sent += 1
        
        # Store trends in database
        _store_trending_topics(trending_topics)
        
        # Update monitoring stats
        _update_monitoring_stats(len(trending_topics), len(opportunities), alerts_sent)
        
        monitor._log_info(f"Monitoring cycle complete: {alerts_sent} alerts sent")
        
        return {
            "success": True,
            "trends_found": len(trending_topics),
            "opportunities": len(opportunities),
            "alerts_sent": alerts_sent
        }
        
    except Exception as e:
        monitor._log_error(f"Monitoring cycle failed: {e}")
        return {"success": False, "error": str(e)}

def _store_trending_topics(topics: List[TrendingTopic]):
    """Store trending topics in database"""
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                
                for topic in topics:
                    cursor.execute('''
                        INSERT INTO trending_topics 
                        (query, search_volume, spike_percentage, category, geo, related_queries)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    ''', (
                        topic.query,
                        topic.search_volume,
                        topic.spike_percentage,
                        topic.category,
                        topic.geo,
                        psycopg2.extras.Json(topic.related_queries)
                    ))
                
                conn.commit()
                
    except Exception as e:
        print(f"Failed to store trending topics: {e}")

def _update_monitoring_stats(trends_count: int, opportunities_count: int, alerts_sent: int):
    """Update monitoring statistics"""
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trends_monitor_stats 
                    (trends_checked, opportunities_found, alerts_sent, api_requests)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (check_date) DO UPDATE SET
                    trends_checked = trends_monitor_stats.trends_checked + %s,
                    opportunities_found = trends_monitor_stats.opportunities_found + %s,
                    alerts_sent = trends_monitor_stats.alerts_sent + %s,
                    api_requests = trends_monitor_stats.api_requests + 1
                ''', (
                    trends_count, opportunities_count, alerts_sent, 1,
                    trends_count, opportunities_count, alerts_sent
                ))
                
                conn.commit()
                
    except Exception as e:
        print(f"Failed to update monitoring stats: {e}")

# Global monitor instance
_trends_monitor = None

def get_trends_monitor():
    """Get global trends monitor instance"""
    global _trends_monitor
    if _trends_monitor is None:
        _trends_monitor = GoogleTrendsMonitor()
    return _trends_monitor

def is_trends_monitoring_configured() -> bool:
    """Check if trends monitoring is properly configured"""
    return (
        PYTRENDS_AVAILABLE and
        bool(os.getenv('TELEGRAM_BOT_TOKEN')) and
        bool(get_db_connection())
    )

def start_trends_monitoring() -> bool:
    """Start background trends monitoring"""
    if not is_trends_monitoring_configured():
        return False
    
    monitor = get_trends_monitor()
    
    if monitor.running:
        return False
    
    def monitoring_loop():
        """Background monitoring loop"""
        monitor.running = True
        
        while monitor.running:
            try:
                # Run monitoring cycle every 4 hours
                run_trends_monitoring_cycle()
                
                # Sleep for 4 hours (14400 seconds)
                sleep_duration = 4 * 60 * 60
                sleep_start = time.time()
                
                while time.time() - sleep_start < sleep_duration and monitor.running:
                    time.sleep(60)  # Check every minute if we should stop
                    
            except Exception as e:
                print(f"Trends monitoring loop error: {e}")
                time.sleep(300)  # 5 minute delay on error
    
    monitor.monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitor.monitor_thread.start()
    
    return True

def stop_trends_monitoring() -> bool:
    """Stop background trends monitoring"""
    monitor = get_trends_monitor()
    
    if not monitor.running:
        return False
    
    monitor.running = False
    
    if monitor.monitor_thread and monitor.monitor_thread.is_alive():
        monitor.monitor_thread.join(timeout=5)
    
    return True

def get_trends_monitoring_status() -> Dict:
    """Get current monitoring status"""
    monitor = get_trends_monitor()
    
    status = {
        'running': monitor.running,
        'configured': is_trends_monitoring_configured(),
        'pytrends_available': PYTRENDS_AVAILABLE,
        'last_check': monitor.last_check,
        'requests_today': monitor.requests_today,
        'alerts_sent_today': monitor.alerts_sent_today,
        'max_requests_per_day': monitor.max_requests_per_day,
        'max_alerts_per_day': monitor.max_alerts_per_day
    }
    
    # Get database stats
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                
                # Get today's stats
                cursor.execute('''
                    SELECT trends_checked, opportunities_found, alerts_sent 
                    FROM trends_monitor_stats 
                    WHERE check_date = CURRENT_DATE
                ''')
                
                result = cursor.fetchone()
                if result:
                    status['today_trends'] = result[0]
                    status['today_opportunities'] = result[1]
                    status['today_alerts'] = result[2]
                else:
                    status['today_trends'] = 0
                    status['today_opportunities'] = 0
                    status['today_alerts'] = 0
                
                # Get total opportunities count
                cursor.execute('SELECT COUNT(*) FROM content_opportunities')
                status['total_opportunities'] = cursor.fetchone()[0]
                
    except Exception as e:
        status['database_error'] = str(e)
    
    return status