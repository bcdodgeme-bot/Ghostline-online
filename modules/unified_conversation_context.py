# modules/unified_conversation_context.py
# NEW FILE: Create unified context management for all integrations
# This replaces the scattered context storage across different modules

import datetime
import json
import re
import os
from typing import Dict, Any, Optional, List, Tuple
from utils.ghostline_engine import generate_response

class UnifiedConversationContext:
    """Unified context management for all integrations to ensure consistent follow-up questions"""
    
    def __init__(self):
        self.context_cache = {}
        self.integration_handlers = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default context types and their follow-up patterns"""
        self.integration_handlers = {
            'gmail_search': {
                'follow_up_patterns': [
                    r'\bthose emails?\b', r'\bthe messages?\b', r'\bfrom (\w+)\b',
                    r'\bwho sent\b', r'\bwhich email\b', r'\bany urgent\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'gmail_overnight': {
                'follow_up_patterns': [
                    r'\bthose emails?\b', r'\bthe overnight\b', r'\bfrom (\w+)\b',
                    r'\bmost important\b', r'\bany urgent\b', r'\bwho emailed\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'calendar_today': {
                'follow_up_patterns': [
                    r'\bthose meetings?\b', r'\bthe calendar\b', r'\bwhen is\b',
                    r'\bwho am i meeting\b', r'\bnext meeting\b', r'\bfree time\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'clickup_tasks': {
                'follow_up_patterns': [
                    r'\bthose tasks?\b', r'\bthe tasks?\b', r'\bwhich task\b',
                    r'\bdue today\b', r'\bhigh priority\b', r'\boverdue\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'cloze_pipeline': {
                'follow_up_patterns': [
                    r'\bthose contacts?\b', r'\bthe pipeline\b', r'\bwho should\b',
                    r'\bfollow up\b', r'\bhot leads?\b', r'\bpriority contacts?\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'analytics_report': {
                'follow_up_patterns': [
                    r'\bthat traffic\b', r'\bthe analytics\b', r'\bwhat does\b',
                    r'\bis that good\b', r'\bcompared to\b', r'\bany insights?\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'search_console_report': {
                'follow_up_patterns': [
                    r'\bthat seo\b', r'\bthe search\b', r'\brank\w*\b',
                    r'\bclicks?\b', r'\bimpressions?\b', r'\bwhat should\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'marketing_generation': {
                'follow_up_patterns': [
                    r'\bthat image\b', r'\bthe mockup\b', r'\bchange it\b',
                    r'\bmake it\b', r'\bdifferent\b', r'\bmodify\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'telegram_reminders': {
                'follow_up_patterns': [
                    r'\bthat reminder\b', r'\bthe notification\b', r'\bwhen will\b',
                    r'\bcancel it\b', r'\bchange time\b', r'\bremind me\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'calendar_telegram': {
                'follow_up_patterns': [
                    r'\bthose alerts\b', r'\bthe monitoring\b', r'\bcalendar notifications\b',
                    r'\bturn off\b', r'\bdisable alerts\b', r'\bstop notifications\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'web_scraping': {
                'follow_up_patterns': [
                    r'\bthat website\b', r'\bthe content\b', r'\bfrom the page\b',
                    r'\bscrape again\b', r'\bwhat does\b', r'\bsummarize\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            },
            'general_conversation': {
                'follow_up_patterns': [
                    r'\bthat answer\b', r'\bthe response\b', r'\bwhat you said\b',
                    r'\bexplain more\b', r'\btell me more\b', r'\bgo deeper\b'
                ],
                'context_expires_minutes': 120  # Updated to 120 minutes
            }
        }
    
    def store_context(self, session_id: str, context_type: str, data: Dict[str, Any],
                     user_query: str = None, integration_specific: Dict = None):
        """Store context for any integration with unified format"""
        
        # Validate context type
        if context_type not in self.integration_handlers:
            print(f"Warning: Unknown context type '{context_type}', storing anyway")
        
        context_entry = {
            'type': context_type,
            'timestamp': datetime.datetime.now(),
            'data': data,
            'original_query': user_query,
            'integration_specific': integration_specific or {},
            'summary': self._generate_summary(context_type, data)
        }
        
        self.context_cache[session_id] = context_entry
        print(f"Stored {context_type} context for session {session_id}")
        
        return True
    
    def get_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get context with automatic expiration"""
        context = self.context_cache.get(session_id)
        if not context:
            return None
        
        # Check expiration
        context_type = context['type']
        handler_config = self.integration_handlers.get(context_type, {})
        expire_minutes = handler_config.get('context_expires_minutes', 120)  # Default 120 minutes
        
        time_diff = datetime.datetime.now() - context['timestamp']
        if time_diff.total_seconds() > (expire_minutes * 60):
            del self.context_cache[session_id]
            print(f"Expired {context_type} context for session {session_id}")
            return None
        
        return context
    
    def detect_follow_up_question(self, user_input: str, session_id: str) -> bool:
        """Detect if this is a follow-up question about recent context"""
        context = self.get_context(session_id)
        if not context:
            return False
        
        user_lower = user_input.lower().strip()
        context_type = context['type']
        
        # Get handler configuration
        handler_config = self.integration_handlers.get(context_type, {})
        follow_up_patterns = handler_config.get('follow_up_patterns', [])
        
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
        
        # Check type-specific patterns
        if any(re.search(pattern, user_lower) for pattern in follow_up_patterns):
            return True
        
        return False
    
    def generate_contextual_response(self, user_input: str, session_id: str,
                                   project: str, use_voices: List[str],
                                   random_toggle: bool) -> Dict[str, Any]:
        """Generate response with full context awareness"""
        context = self.get_context(session_id)
        if not context:
            return {"SyntaxPrime": "I don't have recent context to reference. What would you like to know?"}
        
        # Build comprehensive context prompt
        context_prompt = self._build_context_prompt(user_input, context)
        
        # Get relevant background information
        try:
            from modules.brain import enhanced_retrieve
            from utils.rag_basic import is_ready
            retrieval_ctx = enhanced_retrieve(context_prompt, k=5, project=project) if is_ready() else []
        except ImportError:
            retrieval_ctx = []
        
        # Generate response
        response_data = generate_response(
            context_prompt, use_voices, random_toggle,
            project=project,
            model=os.getenv("CHAT_MODEL", "openrouter/auto"),
            retrieval_context=retrieval_ctx
        )
        
        return response_data
    
    def _build_context_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Build comprehensive context prompt for AI response"""
        prompt_parts = []
        
        prompt_parts.append(f"User just asked: '{user_input}'")
        prompt_parts.append("")
        prompt_parts.append("This is a follow-up question about a recent report I provided.")
        prompt_parts.append("")
        
        context_type = context['type']
        data = context['data']
        original_query = context.get('original_query', 'Unknown')
        summary = context.get('summary', 'No summary available')
        
        # Add context-specific information
        prompt_parts.append(f"**Recent {context_type.replace('_', ' ').title()} Report:**")
        prompt_parts.append(f"Original query: {original_query}")
        prompt_parts.append(f"Summary: {summary}")
        prompt_parts.append("")
        
        # Add detailed data based on context type
        if context_type in ['gmail_search', 'gmail_overnight']:
            response_data = data.get('response', {})
            # Look for email data in various possible locations
            emails = []
            for voice_content in response_data.values():
                if isinstance(voice_content, str) and 'From:' in voice_content:
                    # Parse email info from response text
                    lines = voice_content.split('\n')
                    for i, line in enumerate(lines):
                        if 'From:' in line and 'Subject:' in lines[i+1] if i+1 < len(lines) else False:
                            emails.append({
                                'sender': line.replace('From:', '').strip(),
                                'subject': lines[i+1].replace('Subject:', '').strip() if i+1 < len(lines) else 'No Subject'
                            })
            
            if emails:
                prompt_parts.append("**Email Details:**")
                for i, email in enumerate(emails[:10], 1):
                    sender = email.get('sender', 'Unknown')
                    subject = email.get('subject', 'No Subject')
                    prompt_parts.append(f"{i}. From: {sender} - Subject: {subject}")
                prompt_parts.append("")
        
        elif context_type == 'calendar_today':
            response_data = data.get('response', {})
            # Look for calendar data in response text
            for voice_content in response_data.values():
                if isinstance(voice_content, str) and ('meeting' in voice_content.lower() or 'event' in voice_content.lower()):
                    prompt_parts.append("**Calendar Information:**")
                    prompt_parts.append(voice_content[:500] + "..." if len(voice_content) > 500 else voice_content)
                    prompt_parts.append("")
                    break
        
        elif context_type == 'analytics_report':
            prompt_parts.append("**Analytics Information:**")
            response_data = data.get('response', {})
            for voice_content in response_data.values():
                if isinstance(voice_content, str):
                    prompt_parts.append(voice_content[:500] + "..." if len(voice_content) > 500 else voice_content)
                    break
            prompt_parts.append("")
        
        elif context_type == 'search_console_report':
            prompt_parts.append("**Search Console Information:**")
            response_data = data.get('response', {})
            for voice_content in response_data.values():
                if isinstance(voice_content, str):
                    prompt_parts.append(voice_content[:500] + "..." if len(voice_content) > 500 else voice_content)
                    break
            prompt_parts.append("")
        
        elif context_type == 'marketing_generation':
            prompt_parts.append("**Marketing Generation:**")
            concept = data.get('concept', 'Unknown concept')
            success = data.get('success', False)
            prompt_parts.append(f"Generated marketing asset for: {concept}")
            prompt_parts.append(f"Success: {'Yes' if success else 'No'}")
            prompt_parts.append("")
        
        elif context_type in ['clickup_tasks', 'cloze_pipeline', 'telegram_reminders', 'calendar_telegram', 'web_scraping', 'general_conversation']:
            prompt_parts.append(f"**{context_type.replace('_', ' ').title()} Information:**")
            response_data = data.get('response', {})
            for voice_content in response_data.values():
                if isinstance(voice_content, str):
                    prompt_parts.append(voice_content[:500] + "..." if len(voice_content) > 500 else voice_content)
                    break
            prompt_parts.append("")
        
        prompt_parts.append("Please provide insights, analysis, or recommendations based on this data and the user's follow-up question.")
        
        return "\n".join(prompt_parts)
    
    def _generate_summary(self, context_type: str, data: Dict[str, Any]) -> str:
        """Generate a brief summary of the context data"""
        if context_type in ['gmail_search', 'gmail_overnight']:
            response_data = data.get('response', {})
            email_count = 0
            for voice_content in response_data.values():
                if isinstance(voice_content, str):
                    email_count = voice_content.count('From:')
                    break
            return f"Found {email_count} emails" if email_count > 0 else "Email report generated"
        
        elif context_type == 'calendar_today':
            return "Calendar events retrieved"
        
        elif context_type in ['clickup_tasks', 'cloze_pipeline']:
            return f"{context_type.replace('_', ' ').title()} data retrieved"
        
        elif context_type == 'analytics_report':
            return "Analytics report generated"
        
        elif context_type == 'search_console_report':
            return "Search Console report generated"
        
        elif context_type == 'marketing_generation':
            concept = data.get('concept', 'Unknown')
            success = data.get('success', False)
            return f"Marketing: Generated '{concept}' ({'Success' if success else 'Failed'})"
        
        elif context_type == 'telegram_reminders':
            return "Reminder created"
        
        elif context_type == 'calendar_telegram':
            return "Calendar alerts configured"
        
        elif context_type == 'web_scraping':
            url = data.get('url', 'Unknown URL')
            success = data.get('success', False)
            return f"Scraped: {url} ({'Success' if success else 'Failed'})"
        
        elif context_type == 'general_conversation':
            retrieval_used = data.get('retrieval_used', False)
            return f"General response ({'with context' if retrieval_used else 'without context'})"
        
        return "Context data available"
    
    def clear_context(self, session_id: str) -> bool:
        """Manually clear context for a session"""
        if session_id in self.context_cache:
            del self.context_cache[session_id]
            print(f"Cleared context for session {session_id}")
            return True
        return False
    
    def get_all_active_contexts(self) -> Dict[str, Dict]:
        """Get all active contexts (for debugging)"""
        active_contexts = {}
        now = datetime.datetime.now()
        
        for session_id, context in self.context_cache.items():
            time_diff = now - context['timestamp']
            active_contexts[session_id] = {
                'type': context['type'],
                'summary': context['summary'],
                'age_minutes': time_diff.total_seconds() / 60,
                'original_query': context.get('original_query', 'Unknown')
            }
        
        return active_contexts

# Global instance to be imported by all integrations
unified_context = UnifiedConversationContext()

# INTEGRATION HELPER FUNCTIONS - to be used by existing modules

def store_integration_context(integration_name: str, session_id: str, data: Dict[str, Any],
                            user_query: str = None, **kwargs):
    """Helper function for integrations to store context"""
    return unified_context.store_context(
        session_id=session_id,
        context_type=integration_name,
        data=data,
        user_query=user_query,
        integration_specific=kwargs
    )

def check_for_follow_up(user_input: str, session_id: str, project: str,
                       use_voices: List[str], random_toggle: bool) -> Tuple[Dict[str, Any], bool]:
    """Helper function for integrations to check and handle follow-ups"""
    if unified_context.detect_follow_up_question(user_input, session_id):
        response_data = unified_context.generate_contextual_response(
            user_input, session_id, project, use_voices, random_toggle
        )
        return response_data, True
    
    return {}, False
