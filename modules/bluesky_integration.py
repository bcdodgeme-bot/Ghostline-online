# modules/bluesky_integration.py
# BlueSky Social Integration for Ghostline
# Analyzes BlueSky feed against your 25K conversation database to suggest engagement opportunities

import os
import json
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from modules.database import get_db_connection, enhanced_retrieve
from psycopg2.extras import RealDictCursor

class BlueSkyIntegration:
    """BlueSky Social integration for intelligent feed analysis"""
    
    def __init__(self):
        self.api_base = "https://bsky.social"
        self.session = None
        self.access_jwt = None
        self.refresh_jwt = None
        self.handle = "bcdodgeme.bsky.social"
        self.app_password = "6oth-ty5z-fins-32f3"
        self.authenticated = False
        
    def authenticate(self) -> bool:
        """Authenticate with BlueSky using app password"""
        try:
            auth_url = f"{self.api_base}/xrpc/com.atproto.server.createSession"
            
            auth_data = {
                "identifier": self.handle,
                "password": self.app_password
            }
            
            response = requests.post(auth_url, json=auth_data)
            response.raise_for_status()
            
            session_data = response.json()
            
            self.access_jwt = session_data.get('accessJwt')
            self.refresh_jwt = session_data.get('refreshJwt')
            self.authenticated = True
            
            print(f"✅ BlueSky authentication successful for {self.handle}")
            return True
            
        except Exception as e:
            print(f"❌ BlueSky authentication failed: {e}")
            self.authenticated = False
            return False
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        if not self.access_jwt:
            return {}
        
        return {
            "Authorization": f"Bearer {self.access_jwt}",
            "Content-Type": "application/json"
        }
    
    def get_timeline(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch user's timeline from BlueSky"""
        if not self.authenticated:
            if not self.authenticate():
                return []
        
        try:
            timeline_url = f"{self.api_base}/xrpc/app.bsky.feed.getTimeline"
            params = {"limit": limit}
            
            response = requests.get(
                timeline_url, 
                headers=self.get_auth_headers(),
                params=params
            )
            response.raise_for_status()
            
            timeline_data = response.json()
            posts = timeline_data.get('feed', [])
            
            print(f"📱 Fetched {len(posts)} posts from BlueSky timeline")
            return posts
            
        except Exception as e:
            print(f"❌ Failed to fetch timeline: {e}")
            return []
    
    def analyze_post_against_conversations(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a BlueSky post against conversation database for engagement potential"""
        try:
            # Extract post content
            post_record = post.get('post', {}).get('record', {})
            post_text = post_record.get('text', '')
            
            if not post_text or len(post_text.strip()) < 10:
                return {"relevance_score": 0, "reason": "Post too short or empty"}
            
            # Get post metadata
            author = post.get('post', {}).get('author', {})
            author_handle = author.get('handle', 'unknown')
            author_display = author.get('displayName', author_handle)
            
            created_at = post.get('post', {}).get('record', {}).get('createdAt', '')
            
            # Search conversation database for similar content
            search_results = enhanced_retrieve(post_text, k=5)
            
            if not search_results:
                return {
                    "relevance_score": 0.1,
                    "reason": "No matching conversation history found",
                    "post_preview": post_text[:100] + "..." if len(post_text) > 100 else post_text,
                    "author": f"{author_display} (@{author_handle})"
                }
            
            # Calculate relevance score based on conversation matches
            total_similarity = sum(result.get('similarity', 0.5) for result in search_results)
            avg_similarity = total_similarity / len(search_results)
            
            # Boost score for conversation-type content
            conversation_indicators = [
                'what do you think', 'thoughts on', 'anyone else', 'does anyone',
                'question for', 'hot take', 'unpopular opinion', 'change my mind',
                'looking for', 'need advice', 'recommend', 'suggestions'
            ]
            
            conversation_boost = 0
            for indicator in conversation_indicators:
                if indicator in post_text.lower():
                    conversation_boost += 0.2
            
            final_score = min(avg_similarity + conversation_boost, 1.0)
            
            # Generate engagement reasoning
            reasons = []
            if avg_similarity > 0.7:
                reasons.append("High similarity to your conversation topics")
            if conversation_boost > 0:
                reasons.append("Contains conversation starters you typically engage with")
            if any('question' in result.get('text', '').lower() for result in search_results):
                reasons.append("Similar to questions you've discussed before")
            
            return {
                "relevance_score": final_score,
                "reason": "; ".join(reasons) if reasons else "Moderate relevance to your interests",
                "post_preview": post_text[:200] + "..." if len(post_text) > 200 else post_text,
                "author": f"{author_display} (@{author_handle})",
                "created_at": created_at,
                "matching_conversations": len(search_results),
                "conversation_boost": conversation_boost > 0,
                "suggested_engagement": self._suggest_engagement_type(post_text, search_results)
            }
            
        except Exception as e:
            print(f"❌ Error analyzing post: {e}")
            return {
                "relevance_score": 0,
                "reason": f"Analysis error: {str(e)}",
                "post_preview": "Error processing post"
            }
    
    def _suggest_engagement_type(self, post_text: str, search_results: List[Dict]) -> str:
        """Suggest type of engagement based on post content and conversation history"""
        post_lower = post_text.lower()
        
        # Question posts - suggest thoughtful reply
        if any(word in post_lower for word in ['?', 'what do you', 'how do you', 'why do you']):
            return "reply_with_insight"
        
        # Opinion posts - suggest like or thoughtful addition
        if any(word in post_lower for word in ['think', 'believe', 'opinion', 'take']):
            return "like_or_add_perspective"
        
        # Sharing posts - suggest like and possible reshare
        if any(word in post_lower for word in ['sharing', 'found this', 'check out', 'interesting']):
            return "like_and_consider_reshare"
        
        # Personal updates - suggest supportive engagement
        if any(word in post_lower for word in ['just', 'today', 'excited', 'proud', 'happy']):
            return "supportive_like_or_comment"
        
        return "like_or_light_engagement"
    
    def get_engagement_suggestions(self, limit: int = 20, min_score: float = 0.3) -> List[Dict[str, Any]]:
        """Get posts from timeline with engagement suggestions"""
        posts = self.get_timeline(limit)
        if not posts:
            return []
        
        suggestions = []
        
        for post in posts:
            analysis = self.analyze_post_against_conversations(post)
            
            if analysis['relevance_score'] >= min_score:
                # Add post URI for potential actions
                post_uri = post.get('post', {}).get('uri', '')
                analysis['post_uri'] = post_uri
                analysis['full_post'] = post  # Keep full post data for actions
                
                suggestions.append(analysis)
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"🎯 Found {len(suggestions)} posts worth engaging with (score >= {min_score})")
        return suggestions
    
    def format_suggestions_for_display(self, suggestions: List[Dict[str, Any]]) -> str:
        """Format engagement suggestions for display"""
        if not suggestions:
            return "📱 No high-relevance posts found in your current timeline. Check back in an hour!"
        
        output = ["🎯 **BlueSky Engagement Suggestions**\n"]
        output.append(f"Found {len(suggestions)} posts worth your attention:\n")
        
        for i, suggestion in enumerate(suggestions[:10], 1):  # Show top 10
            score = suggestion['relevance_score']
            author = suggestion.get('author', 'Unknown author')
            preview = suggestion['post_preview']
            reason = suggestion['reason']
            engagement_type = suggestion.get('suggested_engagement', 'general_engagement')
            
            # Format score as percentage
            score_pct = int(score * 100)
            
            output.append(f"**{i}. {author}** (Relevance: {score_pct}%)")
            output.append(f"   *{preview}*")
            output.append(f"   **Why:** {reason}")
            output.append(f"   **Suggested action:** {engagement_type.replace('_', ' ').title()}")
            output.append("")
        
        if len(suggestions) > 10:
            output.append(f"... and {len(suggestions) - 10} more posts worth checking out!")
        
        return "\n".join(output)

# Integration command processing
def process_bluesky_command(user_input: str) -> str:
    """Process BlueSky-related commands"""
    bluesky = BlueSkyIntegration()
    
    user_lower = user_input.lower()
    
    if any(phrase in user_lower for phrase in ['bluesky feed', 'bluesky timeline', 'analyze my bluesky']):
        suggestions = bluesky.get_engagement_suggestions(limit=50, min_score=0.3)
        return bluesky.format_suggestions_for_display(suggestions)
    
    elif any(phrase in user_lower for phrase in ['bluesky high priority', 'best bluesky posts']):
        suggestions = bluesky.get_engagement_suggestions(limit=100, min_score=0.6)
        return bluesky.format_suggestions_for_display(suggestions)
    
    elif 'bluesky test' in user_lower:
        if bluesky.authenticate():
            posts = bluesky.get_timeline(limit=5)
            return f"✅ BlueSky integration working! Fetched {len(posts)} test posts from your timeline."
        else:
            return "❌ BlueSky authentication failed. Check credentials."
    
    return "Available BlueSky commands:\n• 'analyze my bluesky feed' - Get engagement suggestions\n• 'bluesky high priority' - Show only high-relevance posts\n• 'bluesky test' - Test connection"

def is_bluesky_configured() -> bool:
    """Check if BlueSky integration is configured"""
    # Since credentials are hardcoded for now, always return True
    # In production, you'd check environment variables
    return True