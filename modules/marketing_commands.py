"""
Fixed Marketing FLUX commands for chat interface
"""
from modules.marketing_flux import MarketingFluxGenerator, quick_social_post, test_campaign_ideas
from utils.ghostline_engine import generate_response
import re
import os

def process_marketing_command(user_input, project, use_voices, random_toggle):
    """Process marketing-related commands in chat - FIXED VERSION"""
    
    lower_input = user_input.lower().strip()
    
    # EXPANDED trigger patterns - much more flexible
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
        r'\bpromotional\s+image\b'
    ]
    
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
        # Extract concept - be more flexible about parsing
        concept = user_input
        
        # Remove common trigger words to get the concept
        for pattern in marketing_triggers:
            concept = re.sub(pattern, '', concept, flags=re.IGNORECASE).strip()
        
        # Clean up common phrases
        concept = re.sub(r'\b(for|of|about|showing)\b', '', concept, flags=re.IGNORECASE).strip()
        concept = re.sub(r'\s+', ' ', concept)  # Normalize whitespace
        
        if not concept or len(concept.strip()) < 3:
            response_data = {
                "SyntaxPrime": "I need a description for the image. Try: 'create image of summer sale banner' or 'mockup tuxedo cat logo' or 'marketing image for new product launch'"
            }
            return response_data, True
        
        print(f"Extracted concept: '{concept}'")
        
        # Generate the image with better error handling
        generator = MarketingFluxGenerator()
        result = generator.create_and_wait(
            prompt=concept,
            style='corporate',
            platform='instagram',
            quality='standard'
        )
        
        print(f"Generation result: {result}")
        
        if result['success']:
            response_text = f"✅ **Marketing Image Created!**\n\n"
            response_text += f"**Concept**: {concept}\n"
            response_text += f"**Format**: {result.get('format', 'Instagram Post')}\n"
            response_text += f"**Style**: Corporate\n"
            response_text += f"**Cost**: ${result.get('estimated_cost', 0.030):.3f}\n"
            response_text += f"**Generation Time**: {result.get('generation_time', 0):.1f}s\n\n"
            
            image_url = result.get('image_url')
            if image_url:
                response_text += f"**Image URL**: {image_url}\n\n"
                response_text += "💡 **Next Steps:**\n"
                response_text += "• Right-click the URL above to copy/save the image\n"
                response_text += "• Visit /marketing dashboard for more options\n"
                response_text += "• Try different styles: 'luxury', 'startup', 'bold', 'minimalist'\n"
                response_text += "• Specify platforms: 'for Instagram', 'for LinkedIn', 'for Facebook'"
            else:
                response_text += "⚠️ Image generated but URL not available. Check the Marketing Dashboard at /marketing"
        else:
            error_message = result.get('error', 'Unknown error')
            response_text = f"❌ **Image Generation Failed**\n\n"
            response_text += f"**Error**: {error_message}\n\n"
            
            # Provide helpful troubleshooting
            if 'token' in error_message.lower() or 'auth' in error_message.lower():
                response_text += "🔧 **Fix**: Check your REPLICATE_API_TOKEN environment variable\n"
                response_text += "Get your token at: https://replicate.com/account/api-tokens\n\n"
            elif 'rate limit' in error_message.lower():
                response_text += "⏱️ **Fix**: Rate limit hit, wait a moment and try again\n\n"
            elif 'quota' in error_message.lower():
                response_text += "💳 **Fix**: Account quota exceeded, check your Replicate billing\n\n"
            
            response_text += "You can still use the Marketing Dashboard at /marketing for manual image generation."
        
        return {"SyntaxPrime": response_text}, True
        
    except Exception as e:
        print(f"Marketing command exception: {str(e)}")
        import traceback
        traceback.print_exc()
        
        error_msg = f"🚨 **Marketing Command Failed**\n\n"
        error_msg += f"**Error**: {str(e)}\n\n"
        
        # Check for common issues
        if 'MarketingFluxGenerator' in str(e):
            error_msg += "🔧 **Likely Fix**: Missing REPLICATE_API_TOKEN environment variable\n"
            error_msg += "1. Go to https://replicate.com/account/api-tokens\n"
            error_msg += "2. Copy your API token\n"
            error_msg += "3. Set REPLICATE_API_TOKEN=your_token in Railway environment\n"
            error_msg += "4. Restart the application\n\n"
        
        error_msg += "You can still access the Marketing Dashboard at /marketing for image generation."
        return {"SyntaxPrime": error_msg}, True

def is_marketing_configured():
    """Check if marketing/FLUX is configured - ENHANCED VERSION"""
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
