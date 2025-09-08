"""
Conversation Context Handler for Marketing Commands
Solves the continuity problem where users reference previous concepts
"""
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

class MarketingContextManager:
    def __init__(self):
        self.recent_concepts = []  # Store recent marketing concepts
        self.conversation_memory = {}  # Store conversation context
        self.max_memory_items = 10  # Keep last 10 marketing generations
        
    def store_marketing_context(self, user_input: str, concept: str, result: Dict, project: str = None):
        """Store context from a successful marketing generation"""
        context_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'extracted_concept': concept,
            'result': {
                'success': result.get('success'),
                'image_url': result.get('image_url'),
                'format': result.get('format'),
                'platform': result.get('platform'),
                'style': result.get('style')
            },
            'project': project
        }
        
        self.recent_concepts.append(context_entry)
        
        # Keep only recent items
        if len(self.recent_concepts) > self.max_memory_items:
            self.recent_concepts = self.recent_concepts[-self.max_memory_items:]
        
        print(f"Stored marketing context: '{concept}' from '{user_input}'")
    
    def resolve_follow_up_request(self, user_input: str, project: str = None) -> Optional[str]:
        """Resolve follow-up requests that reference previous concepts"""
        
        lower_input = user_input.lower().strip()
        
        # Patterns that indicate user is referencing something previous
        reference_patterns = [
            # Direct references to previous content
            r'\bmockup\s+(that|the\s+)?(exact\s+)?(same\s+)?(text|concept|idea|thing)\b',
            r'\bmake\s+(that|the\s+)?(exact\s+)?(same\s+)?(thing|image|one)\b',
            r'\buse\s+(that|the\s+)?(exact\s+)?(same\s+)?(concept|prompt|idea)\b',
            r'\b(exactly|same)\s+as\s+(before|that|the\s+last\s+one|previous)\b',
            r'\bagain\s+but\b',
            r'\bthe\s+one\s+(you\s+)?(just\s+)?(made|created|generated)\b',
            
            # Vague references
            r'\bmockup\s+(it|this)\b',
            r'\bcreate\s+(it|this|that)\b',
            r'\bmake\s+(it|this|that)\b',
            
            # Context-dependent phrases
            r'\bexact\s+text\b',
            r'\bwhat\s+(i|you)\s+just\s+said\b',
            r'\bthat\s+suggestion\b'
        ]
        
        # Check if this looks like a follow-up request
        is_follow_up = any(re.search(pattern, lower_input) for pattern in reference_patterns)
        
        if not is_follow_up:
            return None
        
        print(f"Detected follow-up request: '{user_input}'")
        
        # Try to resolve the reference
        resolved_concept = self._resolve_reference(user_input, lower_input, project)
        
        if resolved_concept:
            print(f"Resolved reference to: '{resolved_concept}'")
            return resolved_concept
        
        return None
    
    def _resolve_reference(self, user_input: str, lower_input: str, project: str = None) -> Optional[str]:
        """Resolve what the user is referring to"""
        
        if not self.recent_concepts:
            return None
        
        # Look for the most recent relevant concept
        
        # Filter by project if specified
        relevant_concepts = []
        if project:
            relevant_concepts = [c for c in self.recent_concepts if c.get('project') == project]
        
        if not relevant_concepts:
            relevant_concepts = self.recent_concepts
        
        # Get the most recent successful generation
        for concept_entry in reversed(relevant_concepts):
            if concept_entry['result']['success']:
                # Check if the user is asking for the exact same thing
                if self._is_exact_reference(lower_input):
                    return concept_entry['extracted_concept']
                
                # Check if user is asking for a variation
                variation = self._extract_variation_request(user_input, concept_entry)
                if variation:
                    return variation
        
        # Fallback: return the most recent concept
        if relevant_concepts:
            return relevant_concepts[-1]['extracted_concept']
        
        return None
    
    def _is_exact_reference(self, lower_input: str) -> bool:
        """Check if user wants exactly the same thing"""
        exact_patterns = [
            r'\bexact(ly)?\s+(same|text|concept|thing)\b',
            r'\bsame\s+(exact|thing|concept)\b',
            r'\bthat\s+exact\s+text\b',
            r'\bjust\s+like\s+that\b'
        ]
        
        return any(re.search(pattern, lower_input) for pattern in exact_patterns)
    
    def _extract_variation_request(self, user_input: str, concept_entry: Dict) -> Optional[str]:
        """Extract variation requests (same concept, different platform/style)"""
        
        lower_input = user_input.lower()
        base_concept = concept_entry['extracted_concept']
        
        # Platform change requests
        platform_changes = {
            'instagram': ['for instagram', 'insta version', 'ig post'],
            'facebook': ['for facebook', 'fb version', 'facebook post'],
            'linkedin': ['for linkedin', 'linkedin version', 'professional version'],
            'twitter': ['for twitter', 'tweet version', 'twitter post'],
            'email': ['email version', 'newsletter version', 'email header'],
            'blog': ['blog version', 'article header', 'blog header']
        }
        
        for platform, triggers in platform_changes.items():
            if any(trigger in lower_input for trigger in triggers):
                return f"{base_concept} for {platform}"
        
        # Style change requests
        style_changes = {
            'luxury': ['luxury style', 'premium version', 'high-end version'],
            'startup': ['startup style', 'modern version', 'tech version'],
            'bold': ['bold version', 'striking version', 'vibrant version'],
            'minimalist': ['minimal version', 'clean version', 'simple version']
        }
        
        for style, triggers in style_changes.items():
            if any(trigger in lower_input for trigger in triggers):
                return f"{base_concept} in {style} style"
        
        # Size/format changes
        if any(word in lower_input for word in ['banner', 'header', 'cover', 'story']):
            format_word = next(word for word in ['banner', 'header', 'cover', 'story'] if word in lower_input)
            return f"{base_concept} {format_word}"
        
        return None
    
    def get_recent_context_summary(self, project: str = None) -> str:
        """Get a summary of recent marketing context for debugging"""
        
        relevant_concepts = self.recent_concepts
        if project:
            relevant_concepts = [c for c in self.recent_concepts if c.get('project') == project]
        
        if not relevant_concepts:
            return "No recent marketing context available"
        
        summary_parts = []
        summary_parts.append(f"Recent marketing context ({len(relevant_concepts)} items):")
        
        for i, entry in enumerate(reversed(relevant_concepts[-3:])):  # Show last 3
            timestamp = entry['timestamp'][:19]  # Remove microseconds
            concept = entry['extracted_concept']
            success = "✅" if entry['result']['success'] else "❌"
            summary_parts.append(f"{i+1}. {timestamp} {success} '{concept}'")
        
        return "\n".join(summary_parts)

def process_marketing_command_with_context(user_input, project, use_voices, random_toggle, context_manager: MarketingContextManager = None):
    """Enhanced marketing command processor with conversation context"""
    
    if context_manager is None:
        context_manager = MarketingContextManager()
    
    lower_input = user_input.lower().strip()
    
    # First, check if this is a follow-up request
    resolved_concept = context_manager.resolve_follow_up_request(user_input, project)
    
    if resolved_concept:
        # Replace the vague input with the resolved concept
        enhanced_input = f"mockup {resolved_concept}"
        print(f"Enhanced follow-up request: '{user_input}' -> '{enhanced_input}'")
        
        # Process the enhanced input
        return process_enhanced_marketing_request(enhanced_input, project, use_voices, random_toggle, context_manager, original_input=user_input)
    
    # Otherwise, process as normal marketing command
    return process_standard_marketing_request(user_input, project, use_voices, random_toggle, context_manager)

def process_enhanced_marketing_request(enhanced_input, project, use_voices, random_toggle, context_manager, original_input):
    """Process a marketing request that has been enhanced with context"""
    
    # Import the standard processing
    from modules.marketing_commands import is_marketing_configured
    from modules.marketing_flux import MarketingFluxGenerator
    import base64
    import requests
    
    if not is_marketing_configured():
        response_data = {
            "SyntaxPrime": "Marketing image generation not configured. Need REPLICATE_API_TOKEN environment variable."
        }
        return response_data, True
    
    try:
        # Extract concept from enhanced input
        concept = enhanced_input.replace('mockup ', '').strip()
        
        # Generate the image
        generator = MarketingFluxGenerator()
        result = generator.create_and_wait(
            prompt=concept,
            style='corporate',
            platform=None,
            quality='standard'
        )
        
        # Store context for future follow-ups
        context_manager.store_marketing_context(original_input, concept, result, project)
        
        if result['success']:
            response_text = f"✅ **Image Created!** (understood: \"{original_input}\")\n\n"
            response_text += f"**Concept**: {concept}\n"
            response_text += f"**Format**: {result.get('format', 'Standard')}\n"
            response_text += f"**Generated in**: {result.get('generation_time', 0):.1f}s\n\n"
            response_text += "💡 **Try specifying platforms**: 'for Instagram', 'for LinkedIn', 'for Facebook'"
            
            response_data = {"SyntaxPrime": response_text}
            
            # Add image data if available
            image_url = result.get('image_url')
            if image_url:
                # Add image data for inline display
                try:
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        image_base64 = base64.b64encode(response.content).decode('utf-8')
                        content_type = response.headers.get('content-type', 'image/webp')
                        
                        response_data["image_data"] = {
                            'data': image_base64,
                            'content_type': content_type,
                            'size_bytes': len(response.content)
                        }
                        response_data["image_url"] = image_url
                except Exception as e:
                    print(f"Failed to fetch image for inline display: {e}")
            
        else:
            response_text = f"❌ **Image Generation Failed**\n\n"
            response_text += f"**Error**: {result.get('error', 'Unknown error')}\n\n"
            response_text += "🔄 **Try**: Simplifying your description or checking /marketing dashboard"
            response_data = {"SyntaxPrime": response_text}
        
        return response_data, True
        
    except Exception as e:
        print(f"Enhanced marketing command failed: {e}")
        response_data = {"SyntaxPrime": f"🚨 **System Error**: {str(e)}"}
        return response_data, True

def process_standard_marketing_request(user_input, project, use_voices, random_toggle, context_manager):
    """Process a standard marketing request (not a follow-up)"""
    
    # Use the existing marketing command processor
    from modules.marketing_commands import process_marketing_command
    
    response_data, handled = process_marketing_command(user_input, project, use_voices, random_toggle)
    
    if handled and response_data:
        # Extract concept for future context
        # This is a simplified extraction - you might want to make this more sophisticated
        concept = extract_concept_from_input(user_input)
        
        if concept and "✅" in str(response_data):
            # Store successful generation in context
            mock_result = {"success": True}  # Simplified for now
            context_manager.store_marketing_context(user_input, concept, mock_result, project)
    
    return response_data, handled

def extract_concept_from_input(user_input):
    """Extract concept from user input for context storage"""
    concept = user_input
    
    # Remove common trigger words
    remove_patterns = [
        r'\b(create|make|generate|design)\s+(an?\s+)?(image|mockup|visual|graphic)\s+(of|for|showing)?\s*',
        r'\bmockup\s+',
        r'\bimage\s+for\s+',
    ]
    
    for pattern in remove_patterns:
        concept = re.sub(pattern, '', concept, flags=re.IGNORECASE).strip()
    
    return concept if len(concept) > 2 else None

# Global context manager instance
marketing_context = MarketingContextManager()