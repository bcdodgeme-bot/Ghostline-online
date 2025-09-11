# modules/bluesky_integration.py
# BlueSky Social Integration for Ghostline with POSTING CAPABILITIES
# Analyzes BlueSky feed AND allows posting content

import os
import json
import requests
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from modules.database import get_db_connection
from modules.brain import enhanced_retrieve
from psycopg2.extras import RealDictCursor

class BlueSkyIntegration:
    """BlueSky Social integration for intelligent feed analysis and posting"""
    
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
    
    # ========================================================================
    # POSTING FUNCTIONALITY - NEW!
    # ========================================================================
    
    def create_post(self, text: str, reply_to: Optional[Dict] = None) -> Dict[str, Any]:
        """Create a new post on BlueSky"""
        if not self.authenticated:
            if not self.authenticate():
                return {"success": False, "error": "Authentication failed"}
        
        if len(text) > 300:  # BlueSky character limit
            return {"success": False, "error": f"Post too long ({len(text)}/300 characters)"}
        
        try:
            # Prepare post record
            record = {
                "$type": "app.bsky.feed.post",
                "text": text,
                "createdAt": datetime.utcnow().isoformat() + "Z"
            }
            
            # Add reply reference if this is a reply
            if reply_to:
                record["reply"] = reply_to
            
            # Create the post
            create_url = f"{self.api_base}/xrpc/com.atproto.repo.createRecord"
            post_data = {
                "repo": self.handle,
                "collection": "app.bsky.feed.post",
                "record": record
            }
            
            response = requests.post(
                create_url,
                headers=self.get_auth_headers(),
                json=post_data
            )
            response.raise_for_status()
            
            result = response.json()
            
            print(f"✅ Posted to BlueSky: '{text[:50]}...'")
            
            return {
                "success": True,
                "uri": result.get("uri"),
                "cid": result.get("cid"),
                "text": text,
                "char_count": len(text)
            }
            
        except Exception as e:
            print(f"❌ Failed to create post: {e}")
            return {"success": False, "error": str(e)}
    
    def create_thread(self, messages: List[str]) -> Dict[str, Any]:
        """Create a thread of posts on BlueSky"""
        if not messages:
            return {"success": False, "error": "No messages provided"}
        
        results = []
        reply_to = None
        
        for i, message in enumerate(messages):
            result = self.create_post(message, reply_to)
            
            if not result["success"]:
                return {
                    "success": False,
                    "error": f"Failed on post {i+1}: {result['error']}",
                    "partial_results": results
                }
            
            results.append(result)
            
            # Set up reply reference for next post
            if i == 0:  # First post becomes the root
                reply_to = {
                    "root": {
                        "uri": result["uri"],
                        "cid": result["cid"]
                    },
                    "parent": {
                        "uri": result["uri"],
                        "cid": result["cid"]
                    }
                }
            else:  # Subsequent posts reply to previous
                reply_to["parent"] = {
                    "uri": result["uri"],
                    "cid": result["cid"]
                }
        
        return {
            "success": True,
            "thread_length": len(results),
            "posts": results
        }
    
    def smart_compose(self, topic: str, style: str = "professional") -> str:
        """Use AI to compose a BlueSky post about a topic"""
        try:
            from utils.ghostline_engine import generate_response
            
            style_prompts = {
                "professional": "Write a professional, informative post",
                "casual": "Write a casual, conversational post",
                "funny": "Write a humorous, entertaining post",
                "thoughtful": "Write a thoughtful, reflective post",
                "question": "Write a post that asks an engaging question"
            }
            
            style_instruction = style_prompts.get(style, style_prompts["professional"])
            
            prompt = f"""
{style_instruction} for BlueSky about: {topic}

Requirements:
- Maximum 280 characters (BlueSky limit with buffer)
- Engaging and authentic tone
- Include relevant hashtags if appropriate
- No quotes around the final post
- Write in Carl's voice (the user)

Just return the post text, nothing else.
"""
            
            response = generate_response(
                prompt,
                ["SyntaxPrime"],
                False,
                project="BlueSky Content",
                retrieval_context=[]
            )
            
            # Extract the post text from the response
            post_text = response.get("SyntaxPrime", "").strip()
            
            # Remove quotes if AI added them
            if post_text.startswith('"') and post_text.endswith('"'):
                post_text = post_text[1:-1]
            
            # Truncate if too long
            if len(post_text) > 280:
                post_text = post_text[:277] + "..."
            
            return post_text
            
        except Exception as e:
            print(f"❌ Smart compose failed: {e}")
            return f"Update: {topic}"  # Fallback
    
    def break_into_thread(self, long_text: str) -> List[str]:
        """Break long text into thread-appropriate chunks"""
        if len(long_text) <= 280:
            return [long_text]
        
        # Split by sentences and paragraphs
        sentences = re.split(r'[.!?]\s+', long_text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add proper punctuation if missing
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            # Check if adding this sentence would exceed limit
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(test_chunk) <= 280:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add thread indicators
        if len(chunks) > 1:
            for i, chunk in enumerate(chunks):
                chunks[i] = f"{chunk} ({i+1}/{len(chunks)})"
        
        return chunks

    # ========================================================================
    # EXISTING ANALYSIS FUNCTIONALITY (UNCHANGED)
    # ========================================================================
    
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
                    "post_preview": post_text[:200] + "..." if len(post_text) > 200 else post_text,
                    "author": f"{author_display} (@{author_handle})",
                    "full_post": post
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
                "post_preview": post_text,
                "author": f"{author_display} (@{author_handle})",
                "created_at": created_at,
                "matching_conversations": len(search_results),
                "conversation_boost": conversation_boost > 0,
                "suggested_engagement": self._suggest_engagement_type(post_text, search_results),
                "full_post": post
            }
            
        except Exception as e:
            print(f"❌ Error analyzing post: {e}")
            return {
                "relevance_score": 0,
                "reason": f"Analysis error: {str(e)}",
                "post_preview": "Error processing post",
                "full_post": post
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
                
                suggestions.append(analysis)
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"🎯 Found {len(suggestions)} posts worth engaging with (score >= {min_score})")
        return suggestions
    
    def format_suggestions_for_display(self, suggestions: List[Dict[str, Any]]) -> str:
        """Format engagement suggestions for display with full post content"""
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
            created_at = suggestion.get('created_at', '')
            
            # Format score as percentage
            score_pct = int(score * 100)
            
            # Format timestamp if available
            time_display = ""
            if created_at:
                try:
                    # Parse BlueSky timestamp format
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    time_display = f" • {dt.strftime('%H:%M %m/%d')}"
                except:
                    pass
            
            output.append(f"**{i}. {author}** (Relevance: {score_pct}%){time_display}")
            
            # Show the FULL post content with better formatting
            full_post = preview
            
            # Format the post content nicely
            if full_post:
                lines = full_post.split('\n')
                formatted_lines = []
                for line in lines:
                    if line.strip():
                        if len(line) > 80:
                            # Break long lines at word boundaries
                            words = line.split()
                            current_line = ""
                            for word in words:
                                if len(current_line + " " + word) <= 80:
                                    current_line += (" " if current_line else "") + word
                                else:
                                    if current_line:
                                        formatted_lines.append(f"   {current_line}")
                                    current_line = word
                            if current_line:
                                formatted_lines.append(f"   {current_line}")
                        else:
                            formatted_lines.append(f"   {line}")
                    else:
                        formatted_lines.append("")  # Preserve blank lines
                
                if formatted_lines:
                    output.append("   📄 **Post:**")
                    output.extend(formatted_lines)
                else:
                    output.append(f"   📄 **Post:** {full_post}")
            
            output.append("")
            output.append(f"   **💡 Why relevant:** {reason}")
            output.append(f"   **🎯 Suggested action:** {engagement_type.replace('_', ' ').title()}")
            
            # Add engagement stats if available
            post_data = suggestion.get('full_post', {})
            if post_data:
                post_info = post_data.get('post', {})
                like_count = post_info.get('likeCount', 0)
                reply_count = post_info.get('replyCount', 0)
                repost_count = post_info.get('repostCount', 0)
                
                if like_count > 0 or reply_count > 0 or repost_count > 0:
                    stats = []
                    if like_count > 0: stats.append(f"❤️ {like_count}")
                    if reply_count > 0: stats.append(f"💬 {reply_count}")
                    if repost_count > 0: stats.append(f"🔄 {repost_count}")
                    output.append(f"   **📊 Engagement:** {' • '.join(stats)}")
            
            output.append("")
            output.append("---")
            output.append("")
        
        if len(suggestions) > 10:
            output.append(f"... and {len(suggestions) - 10} more posts worth checking out!")
            output.append("")
        
        # Add quick commands including posting
        output.append("**Quick Commands:**")
        output.append("• `bluesky high priority` - Show only high-relevance posts (60%+)")
        output.append("• `bluesky posts` - Show recent posts without analysis")
        output.append("• `post to bluesky \"Your message\"` - Create a new post")
        output.append("• `bluesky compose about [topic]` - AI-generated post")
        
        return "\n".join(output)

# Integration command processing with POSTING SUPPORT
def process_bluesky_command(user_input: str) -> str:
    """Process BlueSky-related commands with enhanced post display and posting"""
    bluesky = BlueSkyIntegration()
    user_lower = user_input.lower()
    
    # ========================================================================
    # POSTING COMMANDS - NEW!
    # ========================================================================
    
    # Direct posting: post to bluesky "message"
    if user_lower.startswith('post to bluesky'):
        # Extract message from quotes
        match = re.search(r'post to bluesky[:\s]*["\']([^"\']+)["\']', user_input, re.IGNORECASE)
        if not match:
            # Try without quotes
            match = re.search(r'post to bluesky[:\s]+(.+)', user_input, re.IGNORECASE)
        
        if match:
            message = match.group(1).strip()
            result = bluesky.create_post(message)
            
            if result["success"]:
                return f"✅ **Posted to BlueSky!**\n\n📝 **Message:** {message}\n📊 **Characters:** {result['char_count']}/300\n🔗 **Post URI:** {result['uri']}"
            else:
                return f"❌ **Failed to post:** {result['error']}\n\n💡 **Tip:** Make sure your message is under 300 characters."
        else:
            return """❌ **Invalid post format**

**Correct usage:**
• `post to bluesky "Your message here"`
• `post to bluesky: Your message here`

**Example:**
• `post to bluesky "Just integrated AI with BlueSky! 🚀"`"""
    
    # AI compose: bluesky compose about [topic]
    elif 'bluesky compose' in user_lower:
        # Extract topic
        match = re.search(r'bluesky compose about (.+)', user_input, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            
            # Extract style if specified
            style = "professional"
            if "casual" in user_lower: style = "casual"
            elif "funny" in user_lower: style = "funny"
            elif "thoughtful" in user_lower: style = "thoughtful"
            elif "question" in user_lower: style = "question"
            
            composed_text = bluesky.smart_compose(topic, style)
            
            return f"""🤖 **AI Composed Post**

📝 **Topic:** {topic}
🎨 **Style:** {style.title()}
📊 **Characters:** {len(composed_text)}/300

**Generated Post:**
"{composed_text}"

**To post this:**
`post to bluesky "{composed_text}"`

**To regenerate:**
`bluesky compose about {topic} funny` (or casual/thoughtful/question)"""
        else:
            return """❌ **Missing topic**

**Usage:** `bluesky compose about [topic]`
**Styles:** casual, funny, thoughtful, question

**Examples:**
• `bluesky compose about AI development`
• `bluesky compose about productivity funny`
• `bluesky compose about entrepreneurship thoughtful`"""
    
    # Thread creation: bluesky thread about [topic]
    elif 'bluesky thread' in user_lower:
        match = re.search(r'bluesky thread about (.+)', user_input, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            
            # Generate longer content for thread
            long_content = bluesky.smart_compose(f"Write a detailed thread about {topic}", "thoughtful")
            
            # If content is short, expand it
            if len(long_content) < 400:
                expanded_prompt = f"Write a comprehensive 3-part explanation about {topic} for a BlueSky thread. Include insights, examples, and actionable takeaways."
                try:
                    from utils.ghostline_engine import generate_response
                    response = generate_response(expanded_prompt, ["SyntaxPrime"], False, project="BlueSky Thread")
                    long_content = response.get("SyntaxPrime", long_content)
                except:
                    pass
            
            # Break into thread chunks
            thread_chunks = bluesky.break_into_thread(long_content)
            
            if len(thread_chunks) == 1:
                return f"""📝 **Thread Preview** (Single Post)

**Topic:** {topic}

**Post:**
"{thread_chunks[0]}"

**To post:**
`post to bluesky "{thread_chunks[0]}"`"""
            else:
                preview = "📝 **Thread Preview**\n\n"
                preview += f"**Topic:** {topic}\n"
                preview += f"**Thread Length:** {len(thread_chunks)} posts\n\n"
                
                for i, chunk in enumerate(thread_chunks, 1):
                    preview += f"**Post {i}:**\n\"{chunk}\"\n\n"
                
                preview += "**To post this thread:**\n"
                preview += "`bluesky post thread` (posts the generated thread above)"
                
                # Store thread for posting
                # You could implement session storage here
                
                return preview
        else:
            return """**Usage:** `bluesky thread about [topic]`

**Examples:**
• `bluesky thread about building AI systems`
• `bluesky thread about startup lessons`
• `bluesky thread about remote work productivity`"""
    
    # ========================================================================
    # EXISTING READ COMMANDS (ENHANCED)
    # ========================================================================
    
    elif any(phrase in user_lower for phrase in ['bluesky feed', 'bluesky timeline', 'analyze my bluesky']):
        suggestions = bluesky.get_engagement_suggestions(limit=50, min_score=0.3)
        return bluesky.format_suggestions_for_display(suggestions)
    
    elif any(phrase in user_lower for phrase in ['bluesky high priority', 'best bluesky posts']):
        suggestions = bluesky.get_engagement_suggestions(limit=100, min_score=0.6)
        return bluesky.format_suggestions_for_display(suggestions)
    
    elif any(phrase in user_lower for phrase in ['bluesky raw', 'bluesky posts', 'show bluesky posts']):
        # Show recent posts without analysis
        posts = bluesky.get_timeline(limit=10)
        if not posts:
            return "❌ Could not fetch BlueSky timeline. Check your connection."
        
        output = ["📱 **Recent BlueSky Posts**\n"]
        
        for i, post in enumerate(posts[:10], 1):
            post_record = post.get('post', {}).get('record', {})
            author = post.get('post', {}).get('author', {})
            
            post_text = post_record.get('text', 'No content')
            author_display = author.get('displayName', author.get('handle', 'Unknown'))
            author_handle = author.get('handle', 'unknown')
            created_at = post_record.get('createdAt', '')
            
            # Format timestamp
            time_display = ""
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    time_display = f" • {dt.strftime('%H:%M %m/%d')}"
                except:
                    pass
            
            output.append(f"**{i}. {author_display} (@{author_handle})**{time_display}")
            
            # Format post text
            lines = post_text.split('\n')
            for line in lines:
                if line.strip():
                    output.append(f"   {line}")
                else:
                    output.append("")
            
            # Add engagement stats
            post_info = post.get('post', {})
            like_count = post_info.get('likeCount', 0)
            reply_count = post_info.get('replyCount', 0)
            repost_count = post_info.get('repostCount', 0)
            
            if like_count > 0 or reply_count > 0 or repost_count > 0:
                stats = []
                if like_count > 0: stats.append(f"❤️ {like_count}")
                if reply_count > 0: stats.append(f"💬 {reply_count}")
                if repost_count > 0: stats.append(f"🔄 {repost_count}")
                output.append(f"   📊 {' • '.join(stats)}")
            
            output.append("")
            output.append("---")
            output.append("")
        
        return "\n".join(output)
    
    elif 'bluesky test' in user_lower:
        if bluesky.authenticate():
            posts = bluesky.get_timeline(limit=5)
            return f"✅ BlueSky integration working! Fetched {len(posts)} test posts from your timeline."
        else:
            return "❌ BlueSky authentication failed. Check credentials."
    
    return """**Available BlueSky Commands:**

**📖 Reading:**
• `bluesky timeline` - Get engagement suggestions with analysis
• `bluesky high priority` - Show only high-relevance posts (60%+)
• `bluesky posts` - Show recent posts with full content, no analysis
• `bluesky test` - Test connection

**✍️ Posting:**
• `post to bluesky "Your message"` - Create a new post
• `bluesky compose about [topic]` - AI-generated post about a topic
• `bluesky thread about [topic]` - Generate a thread about a topic

**✨ Examples:**
• `post to bluesky "Just shipped a new feature! 🚀"`
• `bluesky compose about AI development casual`
• `bluesky thread about startup lessons`"""

def is_bluesky_configured() -> bool:
    """Check if BlueSky integration is configured"""
    # Since credentials are hardcoded for now, always return True
    # In production, you'd check environment variables
    return True
