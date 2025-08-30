"""
Marketing-Focused FLUX Integration
Built for solo marketing departments who need reliable, professional results
"""
import os
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import json
import re

class MarketingFluxGenerator:
    def __init__(self, api_token: str = None):
        """Initialize marketing-focused FLUX generator"""
        self.api_token = api_token or os.getenv('REPLICATE_API_TOKEN')
        self.base_url = 'https://api.replicate.com/v1'
        
        # Marketing-optimized models
        self.models = {
            'professional': {
                'id': 'black-forest-labs/flux-pro',
                'cost': 0.055,
                'description': 'FLUX Pro - Best for client-facing materials',
                'use_cases': ['presentations', 'proposals', 'premium content', 'executive materials']
            },
            'standard': {
                'id': 'black-forest-labs/flux-dev',
                'cost': 0.030,
                'description': 'FLUX Dev - Perfect balance for most marketing',
                'use_cases': ['social media', 'blog headers', 'marketing materials', 'campaigns']
            },
            'rapid': {
                'id': 'black-forest-labs/flux-schnell',
                'cost': 0.003,
                'description': 'FLUX Schnell - Ultra-fast for iterations and testing',
                'use_cases': ['concept testing', 'quick iterations', 'internal drafts', 'bulk content']
            },
            'text_specialist': {
                'id': 'ideogram-ai/ideogram-v2',
                'cost': 0.080,
                'description': 'Ideogram - Specialist for text-heavy designs',
                'use_cases': ['signage', 'logos with text', 'infographics', 'typography-focused']
            }
        }
        
        # Marketing-specific templates and styles
        self.marketing_styles = {
            'corporate': "professional, clean, corporate identity, business-appropriate, polished",
            'startup': "modern, innovative, dynamic, tech-forward, energetic, disruptive",
            'luxury': "premium, elegant, sophisticated, high-end, exclusive, refined",
            'friendly': "approachable, warm, welcoming, human-centered, relatable",
            'bold': "striking, attention-grabbing, vibrant, confident, impactful",
            'minimalist': "clean, simple, uncluttered, focused, contemporary, spacious",
            'creative': "artistic, imaginative, unique, expressive, unconventional",
            'trustworthy': "reliable, established, credible, stable, professional"
        }
        
        # Platform-specific dimensions for social media
        self.social_specs = {
            'instagram_post': {'width': 1080, 'height': 1080, 'name': 'Instagram Square'},
            'instagram_story': {'width': 1080, 'height': 1920, 'name': 'Instagram Story'},
            'facebook_post': {'width': 1200, 'height': 630, 'name': 'Facebook Post'},
            'facebook_cover': {'width': 820, 'height': 312, 'name': 'Facebook Cover'},
            'twitter_post': {'width': 1200, 'height': 675, 'name': 'Twitter Post'},
            'twitter_header': {'width': 1500, 'height': 500, 'name': 'Twitter Header'},
            'linkedin_post': {'width': 1200, 'height': 627, 'name': 'LinkedIn Post'},
            'linkedin_banner': {'width': 1584, 'height': 396, 'name': 'LinkedIn Banner'},
            'youtube_thumbnail': {'width': 1280, 'height': 720, 'name': 'YouTube Thumbnail'},
            'pinterest_pin': {'width': 1000, 'height': 1500, 'name': 'Pinterest Pin'},
            
            # Print and presentation formats
            'blog_header': {'width': 1200, 'height': 600, 'name': 'Blog Header'},
            'presentation_slide': {'width': 1920, 'height': 1080, 'name': 'Presentation 16:9'},
            'email_banner': {'width': 600, 'height': 200, 'name': 'Email Header'},
            'website_hero': {'width': 1920, 'height': 800, 'name': 'Website Hero'}
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not self.api_token:
            raise ValueError(
                "Replicate API token required! Get one at:\n"
                "1. Go to replicate.com\n"
                "2. Sign up (free)\n"
                "3. Get API token from account settings\n"
                "4. Set environment variable: REPLICATE_API_TOKEN=your_token"
            )

    def _enhance_marketing_prompt(self, prompt: str, style: str = 'corporate', 
                                content_type: str = 'marketing', platform: str = None) -> str:
        """Enhance prompts with marketing-specific improvements"""
        
        # Add style guidance
        style_guidance = self.marketing_styles.get(style, self.marketing_styles['corporate'])
        
        # Add content type specifics
        content_specifics = {
            'social_media': "social media ready, engaging, thumb-stopping, shareable",
            'presentation': "presentation-ready, clear, professional, slide-appropriate",
            'blog': "blog-worthy, header-appropriate, article-focused, readable",
            'email': "email-friendly, attention-grabbing, newsletter-appropriate",
            'advertisement': "advertising-focused, persuasive, call-to-action ready",
            'branding': "brand-consistent, identity-focused, memorable, recognizable",
            'product': "product-focused, commercial, sales-oriented, feature-highlighting"
        }
        
        content_guide = content_specifics.get(content_type, "marketing-appropriate, professional")
        
        # Platform-specific enhancements
        platform_guide = ""
        if platform:
            platform_specs = {
                'instagram': "Instagram-optimized, mobile-friendly, visually striking",
                'facebook': "Facebook-appropriate, community-focused, engaging",
                'linkedin': "LinkedIn-professional, B2B-focused, business-appropriate",
                'twitter': "Twitter-ready, concise visual message, trending-worthy",
                'email': "email-optimized, newsletter-ready, inbox-friendly",
                'web': "web-optimized, responsive-design-ready, digital-first"
            }
            platform_guide = platform_specs.get(platform, "")
        
        # Combine everything
        enhanced_parts = [prompt, style_guidance, content_guide]
        if platform_guide:
            enhanced_parts.append(platform_guide)
        
        enhanced_parts.append("high quality, professional photography, marketing material")
        
        return ", ".join(enhanced_parts)

    def _make_request(self, endpoint: str, method: str = 'POST', data: dict = None) -> dict:
        """Make authenticated request to Replicate API"""
        headers = {
            'Authorization': f'Token {self.api_token}',
            'Content-Type': 'application/json'
        }
        
        try:
            if method == 'POST':
                response = requests.post(f"{self.base_url}{endpoint}", headers=headers, json=data)
            else:
                response = requests.get(f"{self.base_url}{endpoint}", headers=headers)
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Response: {e.response.text}")
            raise

    def generate_marketing_asset(self, 
                               prompt: str,
                               style: str = 'corporate',
                               quality: str = 'standard',
                               platform: str = None,
                               format_name: str = None,
                               custom_size: dict = None,
                               seed: Optional[int] = None) -> Dict:
        """
        Generate marketing asset with optimal settings
        
        Args:
            prompt: What you want to create
            style: Brand style (corporate, startup, luxury, etc.)
            quality: professional, standard, or rapid
            platform: Target platform (instagram, facebook, linkedin, etc.)
            format_name: Specific format (instagram_post, blog_header, etc.)
            custom_size: Custom dimensions {"width": 1200, "height": 800}
            seed: For reproducible results
            
        Returns:
            Generation result with prediction info
        """
        
        # Choose model based on quality setting
        model_key = quality if quality in self.models else 'standard'
        model_info = self.models[model_key]
        
        # Determine dimensions
        if custom_size:
            width, height = custom_size['width'], custom_size['height']
            format_display = f"Custom {width}x{height}"
        elif format_name and format_name in self.social_specs:
            spec = self.social_specs[format_name]
            width, height = spec['width'], spec['height']
            format_display = spec['name']
        elif platform:
            # Auto-select best format for platform
            platform_defaults = {
                'instagram': 'instagram_post',
                'facebook': 'facebook_post',
                'linkedin': 'linkedin_post',
                'twitter': 'twitter_post',
                'youtube': 'youtube_thumbnail',
                'email': 'email_banner',
                'blog': 'blog_header',
                'web': 'website_hero'
            }
            default_format = platform_defaults.get(platform, 'instagram_post')
            spec = self.social_specs[default_format]
            width, height = spec['width'], spec['height']
            format_display = spec['name']
        else:
            # Default to versatile square format
            width, height = 1024, 1024
            format_display = "Square 1024x1024"
        
        # Enhance prompt for marketing context
        enhanced_prompt = self._enhance_marketing_prompt(
            prompt, style, 'marketing', platform
        )
        
        # Prepare API request
        input_data = {
            "prompt": enhanced_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": 4 if quality == 'rapid' else 28,
            "guidance_scale": 7.5
        }
        
        if seed:
            input_data["seed"] = seed
        
        prediction_data = {
            "version": model_info['id'],
            "input": input_data
        }
        
        self.logger.info(f"Generating {format_display} asset with {model_key} quality: '{prompt[:50]}...'")
        
        try:
            prediction = self._make_request('/predictions', 'POST', prediction_data)
            
            return {
                'success': True,
                'prediction_id': prediction['id'],
                'status': prediction['status'],
                'model': model_key,
                'quality': quality,
                'estimated_cost': model_info['cost'],
                'format': format_display,
                'dimensions': f"{width}x{height}",
                'prompt': prompt,
                'enhanced_prompt': enhanced_prompt,
                'style': style,
                'platform': platform,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Asset generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model_key
            }

    def wait_for_completion(self, prediction_id: str, max_wait: int = 300) -> Dict:
        """Wait for generation to complete"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                result = self._make_request(f'/predictions/{prediction_id}', 'GET')
                
                status = result['status']
                
                if status == 'succeeded':
                    elapsed = time.time() - start_time
                    self.logger.info(f"✅ Generation completed in {elapsed:.1f}s")
                    return {
                        'success': True,
                        'status': status,
                        'output': result.get('output'),
                        'completed_at': result.get('completed_at'),
                        'generation_time': elapsed
                    }
                elif status == 'failed':
                    self.logger.error(f"❌ Generation failed: {result.get('error')}")
                    return {
                        'success': False,
                        'status': status,
                        'error': result.get('error')
                    }
                elif status in ['starting', 'processing']:
                    self.logger.info(f"⏳ Status: {status}...")
                    time.sleep(2)
                else:
                    time.sleep(2)
                    
            except Exception as e:
                self.logger.error(f"Error checking status: {e}")
                time.sleep(2)
        
        return {
            'success': False,
            'error': 'Generation timeout',
            'status': 'timeout'
        }

    def create_and_wait(self, prompt: str, **kwargs) -> Dict:
        """Create marketing asset and wait for completion"""
        generation = self.generate_marketing_asset(prompt, **kwargs)
        
        if not generation['success']:
            return generation
        
        completion = self.wait_for_completion(generation['prediction_id'])
        
        # Merge results
        result = {**generation, **completion}
        
        if result['success'] and result.get('output'):
            result['image_url'] = result['output'][0] if isinstance(result['output'], list) else result['output']
        
        return result

    def create_campaign_assets(self, 
                             campaign_name: str,
                             base_prompt: str,
                             platforms: List[str] = None,
                             style: str = 'corporate',
                             quality: str = 'standard') -> Dict:
        """
        Create a full set of assets for a marketing campaign
        
        Args:
            campaign_name: Name for organizing the campaign
            base_prompt: Core message/visual concept
            platforms: List of platforms to create for
            style: Visual style
            quality: Generation quality level
            
        Returns:
            Complete campaign asset collection
        """
        
        if not platforms:
            platforms = ['instagram', 'facebook', 'linkedin', 'twitter', 'blog']
        
        self.logger.info(f"🚀 Creating '{campaign_name}' campaign assets for {len(platforms)} platforms")
        
        assets = {}
        total_cost = 0
        
        # Platform-specific format mapping
        platform_formats = {
            'instagram': ['instagram_post', 'instagram_story'],
            'facebook': ['facebook_post', 'facebook_cover'],
            'linkedin': ['linkedin_post', 'linkedin_banner'],
            'twitter': ['twitter_post', 'twitter_header'],
            'youtube': ['youtube_thumbnail'],
            'blog': ['blog_header'],
            'email': ['email_banner'],
            'web': ['website_hero']
        }
        
        for platform in platforms:
            platform_assets = {}
            formats = platform_formats.get(platform, [f"{platform}_post"])
            
            for format_name in formats:
                if format_name in self.social_specs:
                    self.logger.info(f"📱 Creating {self.social_specs[format_name]['name']}...")
                    
                    result = self.create_and_wait(
                        prompt=base_prompt,
                        style=style,
                        quality=quality,
                        format_name=format_name,
                        platform=platform
                    )
                    
                    platform_assets[format_name] = result
                    
                    if result['success']:
                        total_cost += result['estimated_cost']
                        self.logger.info(f"✅ Created {format_name}")
                    else:
                        self.logger.error(f"❌ Failed {format_name}: {result.get('error')}")
                    
                    # Brief pause between generations
                    time.sleep(1)
            
            assets[platform] = platform_assets
        
        successful_assets = sum(
            1 for platform in assets.values() 
            for asset in platform.values() 
            if asset.get('success')
        )
        
        return {
            'success': True,
            'campaign_name': campaign_name,
            'base_prompt': base_prompt,
            'style': style,
            'quality': quality,
            'platforms': list(platforms),
            'total_assets': successful_assets,
            'estimated_total_cost': total_cost,
            'assets': assets,
            'created_at': datetime.now().isoformat()
        }

    def create_social_media_set(self, post_concept: str, platforms: List[str] = None, style: str = 'corporate') -> Dict:
        """Quick social media asset generation"""
        if not platforms:
            platforms = ['instagram', 'facebook', 'linkedin', 'twitter']
        
        return self.create_campaign_assets(
            campaign_name=f"Social Media - {post_concept[:30]}",
            base_prompt=post_concept,
            platforms=platforms,
            style=style,
            quality='standard'
        )

    def test_concepts(self, concepts: List[str], style: str = 'corporate') -> Dict:
        """Rapidly test multiple creative concepts"""
        results = {}
        total_cost = 0
        
        self.logger.info(f"🧪 Testing {len(concepts)} concepts with rapid generation...")
        
        for i, concept in enumerate(concepts):
            self.logger.info(f"Testing concept {i+1}/{len(concepts)}: {concept[:40]}...")
            
            result = self.create_and_wait(
                prompt=concept,
                style=style,
                quality='rapid',  # Use fastest/cheapest for testing
                format_name='instagram_post'
            )
            
            results[f"concept_{i+1}"] = result
            
            if result['success']:
                total_cost += result['estimated_cost']
            
            time.sleep(0.5)  # Brief pause
        
        successful_tests = sum(1 for r in results.values() if r.get('success'))
        
        return {
            'success': True,
            'total_concepts': len(concepts),
            'successful_tests': successful_tests,
            'estimated_total_cost': total_cost,
            'cost_per_concept': total_cost / len(concepts) if concepts else 0,
            'results': results,
            'tested_at': datetime.now().isoformat()
        }

    def get_marketing_recommendations(self) -> Dict:
        """Get recommendations for marketing use"""
        return {
            'quality_guide': {
                'professional': 'Use for client presentations, proposals, premium content',
                'standard': 'Perfect for most marketing needs - social, blogs, campaigns',
                'rapid': 'Great for concept testing, iterations, internal drafts'
            },
            'style_guide': self.marketing_styles,
            'platform_specs': self.social_specs,
            'cost_optimization': {
                'concept_testing': 'Use "rapid" quality with FLUX Schnell ($0.003/image)',
                'final_assets': 'Use "standard" quality with FLUX Dev ($0.030/image)',
                'premium_work': 'Use "professional" quality with FLUX Pro ($0.055/image)',
                'text_heavy': 'Use Ideogram for logos, signage, text-focused designs'
            },
            'workflow_tips': [
                "Test concepts with 'rapid' quality first",
                "Create final assets with 'standard' quality",
                "Use consistent seeds for brand consistency",
                "Batch similar requests to save time",
                "Always specify your target platform"
            ]
        }

# Quick utility functions
def quick_social_post(concept: str, platform: str = 'instagram', style: str = 'corporate') -> Dict:
    """Generate a single social media post quickly"""
    generator = MarketingFluxGenerator()
    return generator.create_and_wait(
        prompt=concept,
        style=style,
        platform=platform,
        quality='standard'
    )

def test_campaign_ideas(ideas: List[str], style: str = 'corporate') -> Dict:
    """Quickly test multiple campaign concepts"""
    generator = MarketingFluxGenerator()
    return generator.test_concepts(ideas, style)

def create_full_campaign(name: str, concept: str, platforms: List[str] = None, style: str = 'corporate') -> Dict:
    """Create complete campaign asset set"""
    generator = MarketingFluxGenerator()
    return generator.create_campaign_assets(name, concept, platforms, style)

if __name__ == "__main__":
    # Demo for marketing department
    print("🎯 Marketing FLUX Generator Ready!")
    
    try:
        generator = MarketingFluxGenerator()
        recommendations = generator.get_marketing_recommendations()
        
        print("\n💡 Quick Start Recommendations:")
        for tip in recommendations['workflow_tips']:
            print(f"  • {tip}")
        
        print("\n💰 Cost Guide:")
        for use_case, guidance in recommendations['cost_optimization'].items():
            print(f"  • {use_case.title()}: {guidance}")
            
    except ValueError as e:
        print(f"\n❌ Setup needed: {e}")