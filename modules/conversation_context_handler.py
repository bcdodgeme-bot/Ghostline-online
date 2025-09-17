"""
Conversation Context Handler for Marketing Commands
Solves the continuity problem where users reference previous concepts
FIXED: Enhanced image handling and proper response structure
"""
import re
import json
import base64
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Import marketing modules
from modules.marketing_flux import MarketingFluxGenerator
from modules.marketing_commands import is_marketing_configured, build_configuration_error, store_marketing_context, get_recent_marketing_context


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
            r'\bthe\s+one\s+(you\s+)?(just\s+)?made\b',
            r'\bthat\s+(image|concept|idea)\b',
            
            # Modification requests that imply previous context
            r'\bmake\s+it\s+\w+',  # "make it brighter"
            r'\bchange\s+the\s+\w+',  # "change the color"
            r'\b(more|less)\s+\w+',  # "more vibrant"
            r'\bdifferent\s+\w+',  # "different background"
            r'\badd\s+\w+',  # "add text"
            r'\bremove\s+\w+',  # "remove logo"
            r'\bbut\s+with\b',  # "but with different text"
        ]
        
        # Check if this looks like a reference to previous work
        is_reference = any(re.search(pattern, lower_input) for pattern in reference_patterns)
        
        if not is_reference:
            return None
        
        # Find the most recent relevant concept
        recent_concepts = [c for c in self.recent_concepts if c.get('project') == project] if project else self.recent_concepts
        
        if not recent_concepts:
            return None
        
        # Get the most recent successful concept
        for concept_entry in reversed(recent_concepts):
            if concept_entry['result'].get('success'):
                base_concept = concept_entry['extracted_concept']
                return self._apply_modification_to_concept(base_concept, user_input)
        
        return None
    
    def _apply_modification_to_concept(self, base_concept: str, modification_request: str) -> str:
        """Apply modification request to a base concept"""
        
        lower_input = modification_request.lower().strip()
        
        # Direct replacement requests
        if 'mockup' in lower_input and 'same' in lower_input:
            if 'text' in lower_input:
                return f"{base_concept} mockup with same text"
            else:
                return f"{base_concept} exact mockup"
        
        # Platform changes
        platform_changes = {
            'instagram': ['instagram', 'ig', 'insta'],
            'facebook': ['facebook', 'fb', 'facebook post'],
            'linkedin': ['linkedin', 'professional'],
            'twitter': ['twitter', 'tweet'],
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


# Global instance
marketing_context = MarketingContextManager()

def process_marketing_command_with_context(user_input: str, project: str, use_voices: List[str], random_toggle: bool) -> tuple[Dict[str, Any], bool]:
    """
    Process marketing commands with contextual understanding
    Returns: (response_data, was_handled)
    """
    
    # First, check if this is a follow-up to previous marketing work
    resolved_concept = marketing_context.resolve_follow_up_request(user_input, project)
    
    if resolved_concept:
        print(f"Resolved follow-up request: '{user_input}' -> '{resolved_concept}'")
        # Process as a marketing command with the resolved concept
        return process_enhanced_marketing_request(resolved_concept, project, use_voices, random_toggle, marketing_context)
    
    # Check if this is a new marketing command
    marketing_patterns = [
        r'social media image',
        r'marketing (image|graphic|asset)',
        r'create (an? )?(image|graphic|post)',
        r'generate (an? )?(image|visual|graphic)',
        r'(facebook|instagram|linkedin|twitter) (post|image)',
        r'blog (header|banner)',
        r'email (banner|header)',
        r'make (an? )?(image|graphic)',
        r'design (an? )?(image|poster|flyer)'
    ]
    
    lower_input = user_input.lower()
    is_marketing_command = any(re.search(pattern, lower_input) for pattern in marketing_patterns)
    
    if is_marketing_command:
        return process_enhanced_marketing_request(user_input, project, use_voices, random_toggle, marketing_context)
    
    return {}, False


def download_and_encode_image(image_url):
    """Download image and encode as base64 for inline display - ENHANCED"""
    try:
        print(f"Downloading image from: {image_url}")
        response = requests.get(image_url, timeout=15)  # Increased timeout
        response.raise_for_status()
        
        # Encode as base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # Determine content type from response headers
        content_type = response.headers.get('content-type')
        if not content_type:
            # Fallback based on image content
            if response.content.startswith(b'\x89PNG'):
                content_type = 'image/png'
            elif response.content.startswith(b'\xff\xd8\xff'):
                content_type = 'image/jpeg'
            elif response.content.startswith(b'RIFF') and b'WEBP' in response.content[:20]:
                content_type = 'image/webp'
            else:
                content_type = 'image/webp'  # Default for FLUX
        
        print(f"✅ Image downloaded successfully: {len(response.content)} bytes, {content_type}")
        
        return {
            'data': image_base64,
            'content_type': content_type,
            'size_bytes': len(response.content)
        }
        
    except requests.exceptions.Timeout:
        print(f"❌ Image download timeout for {image_url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Image download failed for {image_url}: {e}")
        return None
    except Exception as e:
        print(f"❌ Image encoding failed: {e}")
        return None


def process_enhanced_marketing_request(user_input, project, use_voices, random_toggle, context_manager):
    """Process enhanced marketing request with FLUX integration - FIXED"""
    try:
        if not is_marketing_configured():
            error_response = build_configuration_error()
            return {"SyntaxPrime": error_response}, True
        
        # Extract concept from user input
        concept = user_input.lower().replace("social media image", "").replace("for facebook", "").strip()
        if not concept:
            concept = "professional marketing image"
        
        print(f"🎨 Generating marketing asset for concept: '{concept}'")
        
        generator = MarketingFluxGenerator()
        
        # Generate with Facebook optimization (or detect platform from input)
        platform = 'facebook'  # Default
        if 'instagram' in user_input.lower():
            platform = 'instagram'
        elif 'linkedin' in user_input.lower():
            platform = 'linkedin'
        elif 'twitter' in user_input.lower():
            platform = 'twitter'
        
        result = generator.generate_marketing_asset(
            prompt=concept,
            style='corporate',
            quality='standard',
            platform=platform
        )
        
        # Store marketing context
        store_marketing_context(project, concept, result)
        context_manager.store_marketing_context(user_input, concept, result, project)
        
        if result.get('success'):
            response_text = f"✅ **Marketing Image Generated** (understood: \"{user_input}\")\n\n"
            response_text += f"**Concept**: {concept}\n"
            response_text += f"**Format**: {result.get('format', f'{platform.title()} Post')}\n"
            response_text += f"**Generated in**: {result.get('generation_time', 0):.1f}s\n\n"
            response_text += "💡 **Try specifying platforms**: 'for Instagram', 'for LinkedIn', 'for Facebook'"
            
            # FIXED: Create proper response structure for streaming
            image_url = result.get('image_url')
            if image_url:
                try:
                    # Download and encode image
                    image_data = download_and_encode_image(image_url)
                    if image_data:
                        # Return structured response with both text and image data
                        return {
                            "SyntaxPrime": {
                                "SyntaxPrime": response_text,  # Text content
                                "image_data": image_data,
                                "image_url": image_url
                            }
                        }, True
                    else:
                        # Image download failed, but generation succeeded
                        return {"SyntaxPrime": response_text + "\n\n⚠️ Image preview failed, but generation succeeded. Check the link above."}, True
                        
                except Exception as e:
                    print(f"❌ Failed to fetch image for inline display: {e}")
                    # Return text-only response if image fetch fails
                    return {"SyntaxPrime": response_text + f"\n\n⚠️ Image preview failed: {str(e)}"}, True
            
            # Return text-only response if no image URL
            return {"SyntaxPrime": response_text}, True
            
        else:
            response_text = f"❌ **Image Generation Failed**\n\n"
            response_text += f"**Error**: {result.get('error', 'Unknown error')}\n\n"
            response_text += "🔄 **Try**: Simplifying your description or checking /marketing dashboard"
            return {"SyntaxPrime": response_text}, True
        
    except Exception as e:
        print(f"❌ Enhanced marketing command failed: {e}")
        return {"SyntaxPrime": f"🚨 **System Error**: {str(e)}"}, True


def process_standard_marketing_request(user_input, project, use_voices, random_toggle, context_manager):
    """Process a standard marketing request without FLUX integration"""
    try:
        from utils.ghostline_engine import generate_response
        
        # Enhanced marketing-focused prompt
        marketing_prompt = f"""
        Create a detailed marketing concept for: "{user_input}"
        
        Include:
        1. Visual concept description
        2. Target audience
        3. Key messaging
        4. Platform-specific recommendations
        5. Color scheme suggestions
        6. Typography recommendations
        
        Make this actionable and professional.
        """
        
        # Generate enhanced marketing advice
        response_data = generate_response(
            marketing_prompt, use_voices, random_toggle,
            project=project, model="openai/gpt-4o-mini"
        )
        
        # Store context for potential follow-ups
        context_manager.store_marketing_context(user_input, user_input, {'success': False}, project)
        
        return response_data, True
        
    except Exception as e:
        print(f"Standard marketing request failed: {e}")
        return {"SyntaxPrime": f"Marketing guidance error: {str(e)}"}, True


def extract_modification_request(user_input):
    """Extract what the user wants to modify"""
    user_lower = user_input.lower().strip()
    
    # Pattern-based extraction
    modification_patterns = [
        (r'make it\s+(.+)', r'\1'),
        (r'change\s+(?:the\s+)?(.+)', r'\1'),
        (r'(?:more|less)\s+(.+)', r'more \1'),
        (r'different\s+(.+)', r'different \1'),
        (r'add\s+(.+)', r'with \1'),
        (r'remove\s+(.+)', r'without \1'),
        (r'but\s+with\s+(.+)', r'with \1'),
    ]
    
    for pattern, replacement in modification_patterns:
        match = re.search(pattern, user_lower)
        if match:
            modification = re.sub(pattern, replacement, user_lower)
            return modification.strip()
    
    # If no specific pattern found, return the whole input as modification
    return user_input.strip()


def handle_marketing_followup(user_input, project, use_voices, random_toggle, context_manager):
    """Handle marketing follow-up modifications with proper image display - ENHANCED"""
    try:
        if not is_marketing_configured():
            return {"SyntaxPrime": build_configuration_error()}, True
        
        # Get recent marketing context
        recent_context = get_recent_marketing_context(project, limit=1)
        if not recent_context:
            return {"SyntaxPrime": "No recent marketing images to modify. Generate a new image first!"}, True
        
        last_concept = recent_context[0].get('concept', 'previous concept')
        modification = extract_modification_request(user_input)
        
        if not modification:
            return {"SyntaxPrime": f"I couldn't understand the modification request. Try: 'make it brighter' or 'change the background color'"}, True
        
        # Create modified concept
        modified_concept = f"{last_concept}, {modification}"
        
        print(f"🔄 Modifying marketing asset: '{last_concept}' + '{modification}' = '{modified_concept}'")
        
        generator = MarketingFluxGenerator()
        result = generator.generate_marketing_asset(
            prompt=modified_concept,
            style='corporate',
            quality='standard',
            platform='facebook'
        )
        
        # Store new context
        store_marketing_context(project, modified_concept, result)
        context_manager.store_marketing_context(user_input, modified_concept, result, project)
        
        if result.get('success'):
            response_text = f"🔄 **Image Modified Successfully**\n\n"
            response_text += f"**Original**: {last_concept}\n"
            response_text += f"**Modification**: {modification}\n"
            response_text += f"**New Concept**: {modified_concept}\n\n"
            
            if result.get('generation_time'):
                response_text += f"**Generated in**: {result['generation_time']:.1f}s"
            
            # FIXED: Handle image data properly
            image_url = result.get('image_url')
            if image_url:
                image_data = download_and_encode_image(image_url)
                if image_data:
                    return {
                        "SyntaxPrime": {
                            "SyntaxPrime": response_text,
                            "image_data": image_data,
                            "image_url": image_url
                        }
                    }, True
                else:
                    return {"SyntaxPrime": response_text + "\n\n⚠️ Image preview failed, but generation succeeded."}, True
            
            return {"SyntaxPrime": response_text}, True
            
        else:
            error_text = f"❌ Failed to modify image: {result.get('error', 'Unknown error')}\n\n"
            error_text += f"Original concept: {last_concept}\n"
            error_text += f"Requested modification: {modification}"
            return {"SyntaxPrime": error_text}, True
            
    except Exception as e:
        print(f"❌ Marketing follow-up failed: {e}")
        return {"SyntaxPrime": f"Failed to process modification: {str(e)}"}, True
