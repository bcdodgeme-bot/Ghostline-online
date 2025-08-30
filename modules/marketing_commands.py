"""
Marketing FLUX commands for chat interface
"""
from modules.marketing_flux import MarketingFluxGenerator, quick_social_post, test_campaign_ideas
from utils.ghostline_engine import generate_response
import re

def process_marketing_command(user_input, project, use_voices, random_toggle):
    """Process marketing-related commands in chat"""
    
    lower_input = user_input.lower().strip()
    
    # Check if it's a marketing command
    marketing_triggers = [
        'generate image', 'create image', 'make image',
        'marketing asset', 'social post', 'campaign image',
        'flux generate', 'flux create', 'flux make'
    ]
    
    is_marketing_command = any(trigger in lower_input for trigger in marketing_triggers)
    
    if not is_marketing_command:
        return {}, False
    
    try:
        # Extract concept from command
        concept = None
        
        # Pattern matching for different command formats
        if 'generate image' in lower_input:
            concept = re.sub(r'generate image (for |of |about )?', '', lower_input, flags=re.IGNORECASE).strip()
        elif 'create image' in lower_input:
            concept = re.sub(r'create image (for |of |about )?', '', lower_input, flags=re.IGNORECASE).strip()
        elif 'marketing asset' in lower_input:
            concept = re.sub(r'(create |make |generate )?(marketing asset (for |of |about )?)?', '', lower_input, flags=re.IGNORECASE).strip()
        elif 'social post' in lower_input:
            concept = re.sub(r'(create |make |generate )?(social post (for |about )?)?', '', lower_input, flags=re.IGNORECASE).strip()
        elif 'flux' in lower_input:
            concept = re.sub(r'flux (generate|create|make) ', '', lower_input, flags=re.IGNORECASE).strip()
        
        if not concept or len(concept.strip()) < 5:
            return {
                "SyntaxPrime": "I need more details about what image you want me to create. Try: 'generate image for summer sale announcement' or 'create social post about new product launch'"
            }, True
        
        # Generate the image
        generator = MarketingFluxGenerator()
        result = generator.create_and_wait(
            prompt=concept,
            style='corporate',
            platform='instagram', 
            quality='standard'
        )
        
        if result['success']:
            response_text = f"Marketing asset created successfully!\n\n**Concept**: {concept}\n**Format**: {result.get('format', 'Instagram Post')}\n**Cost**: ${result.get('estimated_cost', 0.030):.3f}\n**Generation Time**: {result.get('generation_time', 0):.1f}s\n\n**Image URL**: {result.get('image_url', 'Not available')}\n\nYou can download this from the Marketing Dashboard at /marketing"
        else:
            response_text = f"Image generation failed: {result.get('error', 'Unknown error')}\n\nTry rephrasing your request or check the Marketing Dashboard at /marketing"
        
        return {"SyntaxPrime": response_text}, True
        
    except Exception as e:
        error_msg = f"Marketing command failed: {str(e)}\n\nYou can still use the Marketing Dashboard at /marketing for image generation."
        return {"SyntaxPrime": error_msg}, True

def is_marketing_configured():
    """Check if marketing/FLUX is configured"""
    try:
        generator = MarketingFluxGenerator()
        return True
    except:
        return False