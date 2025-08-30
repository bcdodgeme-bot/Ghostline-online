"""
Marketing FLUX commands for chat interface
"""
from modules.marketing_flux import MarketingFluxGenerator, quick_social_post, test_campaign_ideas
from utils.ghostline_engine import generate_response
import re

def process_marketing_command(user_input, project, use_voices, random_toggle):
    """Process marketing-related commands in chat"""
    
    lower_input = user_input.lower().strip()
    
    # Single command trigger - only "mockup" activates image generation
    if not lower_input.startswith('mockup'):
        return {}, False
    
    try:
        # Extract concept after "mockup"
        concept = lower_input.replace('mockup', '', 1).strip()
        
        if not concept or len(concept.strip()) < 3:
            return {
                "SyntaxPrime": "I need a description for the mockup. Try: 'mockup summer sale banner' or 'mockup tuxedo cat logo'"
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