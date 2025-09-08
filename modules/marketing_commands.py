"""
Enhanced Marketing FLUX commands with inline image display, better UX, and persistent context storage
"""
from modules.marketing_flux import MarketingFluxGenerator, quick_social_post, test_campaign_ideas
from utils.ghostline_engine import generate_response
import re
import os
import base64
import requests
from io import BytesIO
import datetime
import json

def store_marketing_context(project, concept, result):
    """Store marketing context in database for persistence across sessions"""
    try:
        from modules.database import save_conversation_enhanced
        
        context_data = {
            'marketing_context': {
                'concept': concept,
                'timestamp': datetime.datetime.now().isoformat(),
                'result': result,
                'type': 'marketing_generation',
                'success': result.get('success', False),
                'image_url': result.get('image_url'),
                'generation_time': result.get('generation_time', 0)
            }
        }
        
        # Save with special marketing context identifier
        save_conversation_enhanced(
            f"{project}_marketing_context",
            f"Generated: {concept}",
            context_data
        )
        
        print(f"Stored marketing context for project {project}: {concept}")
        return True
        
    except Exception as e:
        print(f"Failed to store marketing context: {e}")
        return False

def get_recent_marketing_context(project, limit=5):
    """Retrieve recent marketing context for follow-up questions"""
    try:
        from modules.database import get_db_connection
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get recent marketing generations for this project
            cursor.execute("""
                SELECT user_input, ai_responses, created_at 
                FROM conversations 
                WHERE project_name = %s 
                AND ai_responses::text LIKE '%marketing_context%'
                ORDER BY created_at DESC 
                LIMIT %s
            """, (f"{project}_marketing_context", limit))
            
            rows = cursor.fetchall()
            
            contexts = []
            for row in rows:
                try:
                    ai_responses = row[1]
                    if isinstance(ai_responses, str):
                        ai_responses = json.loads(ai_responses)
                    
                    marketing_data = ai_responses.get('marketing_context')
                    if marketing_data:
                        contexts.append({
                            'concept': marketing_data.get('concept'),
                            'timestamp': marketing_data.get('timestamp'),
                            'success': marketing_data.get('success'),
                            'image_url': marketing_data.get('image_url'),
                            'created_at': row[2]
                        })
                except Exception as parse_error:
                    print(f"Failed to parse marketing context: {parse_error}")
                    continue
            
            return contexts
            
    except Exception as e:
        print(f"Failed to retrieve marketing context: {e}")
        return []

def detect_marketing_follow_up(user_input, project):
    """Detect if this is a follow-up question about recent marketing generation"""
    user_lower = user_input.lower().strip()
    
    # Follow-up indicators
    follow_up_patterns = [
        r'\bthat image\b', r'\bthe image\b', r'\bit\b',
        r'\bthat mockup\b', r'\bthe mockup\b',
        r'\bthat banner\b', r'\bthe banner\b',
        r'\bcan you\s+(change|modify|update|edit)\b',
        r'\bmake it\s+(bigger|smaller|different)\b',
        r'\bchange the\s+(color|text|style)\b',
        r'\bwith different\b', r'\binstead of\b',
        r'\bnow try\b', r'\bhow about\b',
        r'\bwhat about\b', r'\bcan we\b'
    ]
    
    if any(re.search(pattern, user_lower) for pattern in follow_up_patterns):
        # Get recent context to confirm
        recent_context = get_recent_marketing_context(project, limit=2)
        if recent_context:
            # Check if last generation was within 30 minutes
            last_context = recent_context[0]
            try:
                last_time = datetime.datetime.fromisoformat(last_context['timestamp'])
                time_diff = datetime.datetime.now() - last_time
                if time_diff.total_seconds() < 1800:  # 30 minutes
                    return True, last_context
            except Exception:
                pass
    
    return False, None

def handle_marketing_follow_up(user_input, context_info, project, use_voices, random_toggle):
    """Handle follow-up questions about recent marketing generations"""
    try:
        # Extract modification request
        modification = extract_modification_request(user_input)
        
        if not modification:
            response_text = f"I can help modify your recent image: '{context_info['concept']}'\n\n"
            response_text += "Try specific requests like:\n"
            response_text += "• 'make it blue instead of red'\n"
            response_text += "• 'change the text to say...'\n"
            response_text += "• 'make it for LinkedIn instead'\n"
            response_text += "• 'with a different style'"
            
            return {"SyntaxPrime": response_text}, True
        
        # Generate new image with modification
        original_concept = context_info['concept']
        modified_concept = f"{original_concept}, {modification}"
        
        print(f"Marketing follow-up: '{original_concept}' + '{modification}' = '{modified_concept}'")
        
        generator = MarketingFluxGenerator()
        result = generator.create_and_wait(
            prompt=modified_concept,
            style='corporate',
            platform=None,
            quality='standard'
        )
        
        if result['success']:
            # Store new context
            store_marketing_context(project, modified_concept, result)
            
            response_text = f"**Modified Image Created!**\n\n"
            response_text += f"**Original**: {original_concept}\n"
            response_text += f"**Modification**: {modification}\n"
            response_text += f"**New Concept**: {modified_concept}\n\n"
            
            if result.get('generation_time'):
                response_text += f"**Generated in**: {result['generation_time']:.1f}s"
            
            response_data = {"SyntaxPrime": response_text}
            
            # Add image data for inline display
            image_url = result.get('image_url')
            if image_url:
                image_data = download_and_encode_image(image_url)
                if image_data:
                    response_data["image_data"] = image_data
                    response_data["image_url"] = image_url
            
            return response_data, True
        else:
            error_text = f"Failed to modify image: {result.get('error', 'Unknown error')}\n\n"
            error_text += f"Original concept: {original_concept}\n"
            error_text += f"Requested modification: {modification}"
            return {"SyntaxPrime": error_text}, True
            
    except Exception as e:
        print(f"Marketing follow-up failed: {e}")
        return {"SyntaxPrime": f"Failed to process modification: {str(e)}"}, True

def extract_modification_request(user_input):
    """Extract what the user wants to modify"""
    user_lower = user_input.lower().strip()
    
    # Pattern-based extraction
    modification_patterns = [
        (r'make it\s+(.+)', r'\1'),
        (r'change\s+(?:the\s+)?(.+)', r'\1'),
        (r'with\s+(.+)\s+instead', r'\1'),
        (r'but\s+(.+)', r'\1'),
        (r'now\s+(.+)', r'\1'),
        (r'try\s+(.+)', r'\1')
    ]
    
    for pattern, replacement in modification_patterns:
        match = re.search(pattern, user_lower)
        if match:
            modification = match.group(1).strip()
            # Clean up common words
            modification = re.sub(r'\b(please|can you|could you)\b', '', modification).strip()
            if modification:
                return modification
    
    # If no pattern matches, use the whole input as modification
    # Remove common follow-up words
    cleaned = re.sub(r'\b(that image|the image|it|that|the)\b', '', user_lower).strip()
    cleaned = re.sub(r'\b(can you|could you|please|now|but)\b', '', cleaned).strip()
    
    return cleaned if len(cleaned) > 2 else None

def process_marketing_command(user_input, project, use_voices, random_toggle):
    """Process marketing-related commands with enhanced context and follow-up support"""
    
    lower_input = user_input.lower().strip()
    
    # Check for follow-up questions first
    is_follow_up, context_info = detect_marketing_follow_up(user_input, project)
    if is_follow_up and context_info:
        return handle_marketing_follow_up(user_input, context_info, project, use_voices, random_toggle)
    
    # Enhanced trigger patterns with better context understanding
    marketing_triggers = [
        r'\bmockup\b',
        r'\bimage\s+for\b',
        r'\bcreate\s+image\b',
        r'\bgenerate\s+image\b',
        r'\bmake\s+image\b',
        r'\bdesign\s+image\b',
        r'\bmarketing\s+image\b',
        r'\bsocial\s+media\s+image\b',
        r'\bbanner\b',
        r'\blogo\b',
        r'\bflyer\b',
        r'\bposter\b',
        r'\bad\s+image\b',
        r'\bpromotional\s+image\b',
        r'\bvisual\s+for\b',
        r'\bgraphic\s+for\b'
    ]
    
    # Platform-specific triggers for better context
    platform_triggers = {
        'instagram': [r'\bfor\s+instagram\b', r'\binsta\s+post\b', r'\big\s+post\b'],
        'facebook': [r'\bfor\s+facebook\b', r'\bfb\s+post\b'],
        'linkedin': [r'\bfor\s+linkedin\b', r'\blinkedin\s+post\b'],
        'twitter': [r'\bfor\s+twitter\b', r'\btweet\s+image\b'],
        'email': [r'\bemail\s+header\b', r'\bnewsletter\s+image\b'],
        'blog': [r'\bblog\s+header\b', r'\barticle\s+image\b']
    }
    
    # Check if any trigger pattern matches
    triggered = any(re.search(pattern, lower_input) for pattern in marketing_triggers)
    
    if not triggered:
        return {}, False
    
    print(f"Marketing command triggered by: '{user_input}'")
    
    # Check if marketing is configured
    if not is_marketing_configured():
        response_data = {
            "SyntaxPrime": "Marketing image generation not configured. Need REPLICATE_API_TOKEN environment variable. Visit https://replicate.com to get your API key."
        }
        return response_data, True
    
    try:
        # Enhanced concept extraction with context preservation
        concept = extract_concept_with_context(user_input, lower_input)
        
        if not concept or len(concept.strip()) < 3:
            response_data = {
                "SyntaxPrime": "I need a description for the image. Try:\n• 'mockup summer sale banner'\n• 'create image of tuxedo cat logo'\n• 'marketing image for new product launch for Instagram'"
            }
            return response_data, True
        
        # Detect platform from context
        detected_platform = detect_platform_from_input(lower_input, platform_triggers)
        
        print(f"Extracted concept: '{concept}', Platform: {detected_platform}")
        
        # Generate the image with enhanced parameters
        generator = MarketingFluxGenerator()
        result = generator.create_and_wait(
            prompt=concept,
            style='corporate',
            platform=detected_platform,
            quality='standard'
        )
        
        print(f"Generation result: {result}")
        
        # Store context for follow-ups
        store_marketing_context(project, concept, result)
        
        if result['success']:
            # Create enhanced response with inline image
            response_text = create_success_response(result, concept, detected_platform)
            
            # Add follow-up suggestions
            response_text += "\n\n**Follow-up options**:\n"
            response_text += "• 'make it blue instead'\n"
            response_text += "• 'change the text to say...'\n"
            response_text += "• 'try a different style'\n"
            response_text += "• 'make it for LinkedIn instead'"
            
            # Try to include inline image data
            image_data = None
            image_url = result.get('image_url')
            if image_url:
                image_data = download_and_encode_image(image_url)
            
            response_data = {
                "SyntaxPrime": response_text
            }
            
            # Add image data if available (for frontend to display inline)
            if image_data:
                response_data["image_data"] = image_data
                response_data["image_url"] = image_url
            
        else:
            response_text = create_error_response(result)
            response_data = {"SyntaxPrime": response_text}
        
        return response_data, True
        
    except Exception as e:
        print(f"Marketing command exception: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_msg = create_exception_response(str(e))
        return {"SyntaxPrime": error_msg}, True

def extract_concept_with_context(user_input, lower_input):
    """Enhanced concept extraction that preserves important context"""
    concept = user_input
    
    # Remove trigger words but preserve meaningful context
    remove_patterns = [
        r'\b(create|make|generate|design)\s+(an?\s+)?(image|mockup|visual|graphic)\s+(of|for|showing)?\s*',
        r'\bmockup\s+',
        r'\bimage\s+for\s+',
        r'\bmarketing\s+image\s+(for\s+)?',
        r'\bsocial\s+media\s+image\s+(for\s+)?'
    ]
    
    for pattern in remove_patterns:
        concept = re.sub(pattern, '', concept, flags=re.IGNORECASE).strip()
    
    # Clean up extra whitespace
    concept = re.sub(r'\s+', ' ', concept).strip()
    
    return concept

def detect_platform_from_input(lower_input, platform_triggers):
    """Detect target platform from user input"""
    for platform, patterns in platform_triggers.items():
        if any(re.search(pattern, lower_input) for pattern in patterns):
            return platform
    return None

def create_success_response(result, concept, platform):
    """Create a clean, user-friendly success response"""
    response_parts = []
    
    # Success header
    response_parts.append("✅ **Image Created!**")
    response_parts.append("")
    
    # Key details without overwhelming info
    response_parts.append(f"**Concept**: {concept}")
    
    if platform:
        platform_name = platform.title()
        response_parts.append(f"**Platform**: {platform_name}")
    
    format_info = result.get('format', 'Standard')
    if format_info and format_info != 'Standard':
        response_parts.append(f"**Format**: {format_info}")
    
    # Generation time (users like to see speed)
    gen_time = result.get('generation_time', 0)
    if gen_time:
        response_parts.append(f"**Generated in**: {gen_time:.1f}s")
    
    response_parts.append("")
    
    # Platform suggestions for next time
    response_parts.append("💡 **Try specifying platforms**: 'for Instagram', 'for LinkedIn', 'for Facebook', 'for email header'")
    
    return "\n".join(response_parts)

def create_error_response(result):
    """Create helpful error response"""
    error_message = result.get('error', 'Unknown error')
    
    response_parts = []
    response_parts.append("❌ **Image Generation Failed**")
    response_parts.append("")
    response_parts.append(f"**Error**: {error_message}")
    response_parts.append("")
    
    # Provide helpful troubleshooting
    if 'token' in error_message.lower() or 'auth' in error_message.lower():
        response_parts.append("🔧 **Fix**: Check your REPLICATE_API_TOKEN environment variable")
        response_parts.append("Get your token at: https://replicate.com/account/api-tokens")
    elif 'rate limit' in error_message.lower():
        response_parts.append("⏱️ **Fix**: Rate limit hit, wait a moment and try again")
    elif 'quota' in error_message.lower():
        response_parts.append("💳 **Fix**: Account quota exceeded, check your Replicate billing")
    else:
        response_parts.append("🔄 **Try**: Simplifying your description or checking /marketing dashboard")
    
    return "\n".join(response_parts)

def create_exception_response(error_str):
    """Create response for unexpected exceptions"""
    error_parts = []
    error_parts.append("🚨 **System Error**")
    error_parts.append("")
    error_parts.append(f"**Details**: {error_str}")
    error_parts.append("")
    
    # Check for common issues
    if 'MarketingFluxGenerator' in error_str:
        error_parts.append("🔧 **Likely Fix**: Missing REPLICATE_API_TOKEN environment variable")
        error_parts.append("1. Go to https://replicate.com/account/api-tokens")
        error_parts.append("2. Copy your API token")
        error_parts.append("3. Set REPLICATE_API_TOKEN=your_token in environment")
        error_parts.append("4. Restart the application")
    else:
        error_parts.append("🔄 **Try**: /marketing dashboard for manual image generation")
    
    return "\n".join(error_parts)

def download_and_encode_image(image_url):
    """Download image and encode as base64 for inline display"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        # Encode as base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # Determine content type
        content_type = response.headers.get('content-type', 'image/webp')
        
        return {
            'data': image_base64,
            'content_type': content_type,
            'size_bytes': len(response.content)
        }
        
    except Exception as e:
        print(f"Failed to download image for inline display: {e}")
        return None

def is_marketing_configured():
    """Check if marketing/FLUX is configured"""
    try:
        # Check for API token
        api_token = os.getenv('REPLICATE_API_TOKEN')
        if not api_token:
            print("Marketing not configured: Missing REPLICATE_API_TOKEN")
            return False
        
        # Try to create generator
        generator = MarketingFluxGenerator()
        print("Marketing configured successfully")
        return True
        
    except Exception as e:
        print(f"Marketing configuration check failed: {e}")
        return False
