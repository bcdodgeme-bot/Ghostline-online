# modules/unified_conversation_context.py
# UPDATED: Simplified database-only context management
# Removed in-memory caching and expiration logic - uses database queries only

import datetime
import json
import re
import os
from typing import Dict, Any, Optional, List, Tuple
from utils.ghostline_engine import generate_response

class UnifiedConversationContext:
    """Simplified database-only context management for all integrations"""
    
    def __init__(self):
        self.integration_handlers = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default context types and their follow-up patterns"""
        self.integration_handlers = {
            'gmail_search': {
                'follow_up_patterns': [
                    r'\bthose emails?\b', r'\bthe messages?\b', r'\bfrom (\w+)\b',
                    r'\bwho sent\b', r'\bwhich email\b', r'\bany urgent\b'
                ]
            },
            'gmail_overnight': {
                'follow_up_patterns': [
                    r'\bthose emails?\b', r'\bthe overnight\b', r'\bfrom (\w+)\b',
                    r'\bmost important\b', r'\bany urgent\b', r'\bwho emailed\b'
                ]
            },
            'calendar_today': {
                'follow_up_patterns': [
                    r'\bthose meetings?\b', r'\bthe calendar\b', r'\bwhen is\b',
                    r'\bwho am i meeting\b', r'\bnext meeting\b', r'\bfree time\b'
                ]
            },
            'clickup_tasks': {
                'follow_up_patterns': [
                    r'\bthose tasks?\b', r'\bthe tasks?\b', r'\bwhich task\b',
                    r'\bdue today\b', r'\bhigh priority\b', r'\boverdue\b'
                ]
            },
            'cloze_pipeline': {
                'follow_up_patterns': [
                    r'\bthose contacts?\b', r'\bthe pipeline\b', r'\bwho should\b',
                    r'\bfollow up\b', r'\bhot leads?\b', r'\bpriority contacts?\b'
                ]
            },
            'analytics_report': {
                'follow_up_patterns': [
                    r'\bthat traffic\b', r'\bthe analytics\b', r'\bwhat does\b',
                    r'\bis that good\b', r'\bcompared to\b', r'\bany insights?\b'
                ]
            },
            'search_console_report': {
                'follow_up_patterns': [
                    r'\bthat seo\b', r'\bthe search\b', r'\brank\w*\b',
                    r'\bclicks?\b', r'\bimpressions?\b', r'\bwhat should\b'
                ]
            },
            'marketing_generation': {
                'follow_up_patterns': [
                    r'\bthat image\b', r'\bthe mockup\b', r'\bchange it\b',
                    r'\bmake it\b', r'\bdifferent\b', r'\bmodify\b'
                ]
            },
            'telegram_reminders': {
                'follow_up_patterns': [
                    r'\bthat reminder\b', r'\bthe notification\b', r'\bwhen will\b',
                    r'\bcancel it\b', r'\bchange time\b', r'\bremind me\b'
                ]
            },
            'calendar_telegram': {
                'follow_up_patterns': [
                    r'\bthose alerts\b', r'\bthe monitoring\b', r'\bcalendar notifications\b',
                    r'\bturn off\b', r'\bdisable alerts\b', r'\bstop notifications\b'
                ]
            },
            'web_scraping': {
                'follow_up_patterns': [
                    r'\bthat website\b', r'\bthe content\b', r'\bfrom the page\b',
                    r'\bscrape again\b', r'\bwhat does\b', r'\bsummarize\b'
                ]
            },
            'general_conversation': {
                'follow_up_patterns': [
                    r'\bthat answer\b', r'\bthe response\b', r'\bwhat you said\b',
                    r'\bexplain more\b', r'\btell me more\b', r'\bgo deeper\b'
                ]
            }
        }
    
    def detect_follow_up_question(self, user_input: str, session_id: str = None) -> bool:
        """Detect if this is a follow-up question - simplified to pattern matching only"""
        user_lower = user_input.lower().strip()
        
        # Generic follow-up indicators that work for all types
        generic_patterns = [
            r'\bwhat do you think\b', r'\bwhat\'?s your opinion\b', r'\byour thoughts?\b',
            r'\banalyze (?:this|that)\b', r'\bwhat does (?:this|that) mean\b',
            r'\bis (?:this|that) good\b', r'\bis (?:this|that) bad\b',
            r'\bhow (?:is|are) (?:this|that|these|those)\b',
            r'\bexplain (?:this|that)\b', r'\btell me about (?:this|that)\b',
            r'\bany insights?\b', r'\bany recommendations?\b', r'\bwhat should\b'
        ]
        
        # Check generic patterns first
        if any(re.search(pattern, user_lower) for pattern in generic_patterns):
            return True
        
        # Check type-specific patterns from all handlers
        for handler_config in self.integration_handlers.values():
            follow_up_patterns = handler_config.get('follow_up_patterns', [])
            if any(re.search(pattern, user_lower) for pattern in follow_up_patterns):
                return True
        
        return False
    
    def generate_contextual_response(self, user_input: str, session_id: str,
                                   project: str, use_voices: List[str],
                                   random_toggle: bool) -> Dict[str, Any]:
        """Generate response using database context via enhanced_retrieve"""
        
        # Build context prompt for database query
        context_prompt = self._build_context_prompt_for_retrieval(user_input)
        
        # Get relevant context from database
        try:
            from modules.brain import enhanced_retrieve
            from utils.rag_basic import is_ready
            retrieval_ctx = enhanced_retrieve(context_prompt, k=10, project=project) if is_ready() else []
        except ImportError:
            retrieval_ctx = []
        
        # Generate response with database context
        response_data = generate_response(
            context_prompt, use_voices, random_toggle,
            project=project,
            model=os.getenv("CHAT_MODEL", "openrouter/auto"),
            retrieval_context=retrieval_ctx
        )
        
        return response_data
    
    def _build_context_prompt_for_retrieval(self, user_input: str) -> str:
        """Build prompt optimized for database retrieval"""
        prompt_parts = []
        
        prompt_parts.append(f"User follow-up question: '{user_input}'")
        prompt_parts.append("")
        prompt_parts.append("This appears to be a follow-up question about a recent report or interaction.")
        prompt_parts.append("Please provide relevant context from recent:")
        prompt_parts.append("- Email reports and searches")
        prompt_parts.append("- Calendar events and meetings")
        prompt_parts.append("- Task management updates")
        prompt_parts.append("- Analytics and SEO reports")
        prompt_parts.append("- Marketing generation activities")
        prompt_parts.append("- Reminder and notification settings")
        prompt_parts.append("- Web scraping results")
        prompt_parts.append("- General conversations")
        prompt_parts.append("")
        prompt_parts.append("Based on the available context, please provide insights, analysis, or recommendations.")
        
        return "\n".join(prompt_parts)

# Global instance to be imported by all integrations
unified_context = UnifiedConversationContext()

# INTEGRATION HELPER FUNCTIONS - simplified for database-only approach

def store_integration_context(integration_name: str, session_id: str, data: Dict[str, Any],
                            user_query: str = None, **kwargs):
    """Helper function - context now stored in database via regular RAG flow"""
    # Context is automatically stored in database through normal conversation flow
    # This function remains for backward compatibility but doesn't need to do anything
    print(f"Context for {integration_name} will be stored in database via regular conversation flow")
    return True

def check_for_follow_up(user_input: str, session_id: str, project: str,
                       use_voices: List[str], random_toggle: bool) -> Tuple[Dict[str, Any], bool]:
    """Simplified helper function using database retrieval for follow-ups"""
    if unified_context.detect_follow_up_question(user_input, session_id):
        response_data = unified_context.generate_contextual_response(
            user_input, session_id, project, use_voices, random_toggle
        )
        return response_data, True
    
    return {}, False
