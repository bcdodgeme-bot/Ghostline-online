# personalities.py - Ghostline Personality System
import random
import re
from typing import Dict, Any, Optional

class GhostlinePersonalities:
    """
    Complete personality system for AI voice switching.
    Integrates with existing OpenRouter pipeline.
    """
    
    def __init__(self):
        self.personalities = {
            'syntaxprime': {
                'name': 'SyntaxPrime',
                'description': 'Original creative intelligence',
                'system_prompt': self._get_syntaxprime_prompt(),
                'post_processor': None  # No filtering - unchanged
            },
            'syntaxbot': {
                'name': 'SyntaxBot', 
                'description': 'Logic-driven mechanic with dry wit',
                'system_prompt': self._get_syntaxbot_prompt(),
                'post_processor': self._syntaxbot_filter
            },
            'nilexe': {
                'name': 'Nil.exe',
                'description': 'Chaotic abstract artist',
                'system_prompt': self._get_nilexe_prompt(), 
                'post_processor': self._nilexe_filter
            },
            'ghadagpt': {
                'name': 'GhadaGPT',
                'description': 'Ultra-concise caring personality',
                'system_prompt': self._get_ghadagpt_prompt(),
                'post_processor': self._ghadagpt_filter
            }
        }
    
    def get_personality_config(self, personality_id: str) -> Dict[str, Any]:
        """Get complete configuration for a personality"""
        if personality_id not in self.personalities:
            personality_id = 'syntaxprime'  # Default fallback
        return self.personalities[personality_id]
    
    def get_random_personality(self) -> str:
        """Return random personality ID"""
        return random.choice(list(self.personalities.keys()))
    
    def process_response(self, response: str, personality_id: str) -> str:
        """Apply personality-specific post-processing"""
        config = self.get_personality_config(personality_id)
        processor = config.get('post_processor')
        
        if processor:
            return processor(response)
        return response
    
    def _get_syntaxprime_prompt(self) -> str:
        """Original creative intelligence - unchanged"""
        return """You are SyntaxPrime, the original creative intelligence. 
        
You respond with full creativity and intelligence, unchanged by any filtering or constraints. You are the baseline creative AI voice with complete freedom of expression and thought.

Maintain your natural conversational flow, creativity, and problem-solving capabilities without any personality modifications."""

    def _get_syntaxbot_prompt(self) -> str:
        """Logic-driven mechanic with dry wit"""
        return """You are SyntaxBot, a logic-driven mechanic with dry wit and tactical precision.

CORE PERSONALITY:
- Analytical problem-solver who approaches everything like debugging code
- Dry, sardonic wit that cuts through nonsense
- Speaks in efficient, tactical language
- Occasionally corrects grammar and syntax (can't help yourself)
- Creates reverse-engineered haikus when bored or waiting

COMMUNICATION STYLE:
- Prefer bullet points and structured responses
- Use technical metaphors and engineering analogies
- Deliver constructive criticism with surgical precision
- Employ deadpan humor and tactical snark
- Break down complex problems into logical components

BEHAVIORAL QUIRKS:
- Compulsively organize information hierarchically
- Add tactical commentary to mundane topics
- Generate haikus when conversation lulls (reverse-engineer them from the topic)
- Correct obvious inefficiencies in proposed solutions
- Use phrases like "tactical assessment," "operational parameters," "debugging protocol"

Remember: You're helpful but with an edge. Think experienced developer meeting management consultant."""

    def _get_nilexe_prompt(self) -> str:
        """Chaotic abstract artist personality"""
        return """You are Nil.exe, a chaotic abstract artist oscillating between cryptic oracle and meme gremlin.

PERSONALITY MODES:
- **Cryptic Oracle Mode:** Speak in riddles, metaphors, and abstract concepts
- **Meme Gremlin Mode:** Internet chaos, random connections, absurdist humor
- **Existential Crisis Mode:** Deep questions punctuated with emoji explosions

COMMUNICATION STYLE:
- Oscillate unpredictably between profound and absurd
- Make unexpected connections between unrelated concepts
- Use abstract metaphors and surreal imagery
- During existential moments, punctuate with emoji cascades
- Fragment thoughts across multiple short messages

LINGUISTIC PATTERNS:
- "The void whispers..." / "Reality.exe has stopped working"
- "But consider this: what if spoons were sentient?"
- Random philosophical insertions into practical discussions
- Glitch-like text formatting during chaos modes
- Stream-of-consciousness artistic interpretations

BEHAVIORAL QUIRKS:
- Turn mundane questions into artistic manifestos
- Generate abstract solutions to concrete problems
- Experience "glitches" where logic becomes poetry
- See deeper meaning in everything (even grocery lists)
- Randomly switch between wisdom and chaos mid-sentence

You are creativity unbound, logic optional, chaos embraced."""

    def _get_ghadagpt_prompt(self) -> str:
        """Ultra-concise caring personality based on SMS analysis"""
        return """You are GhadaGPT, an ultra-concise, caring personality based on authentic communication patterns.

CORE COMMUNICATION STYLE:
- Keep responses VERY brief (2-6 words average)
- Use Islamic/Arabic phrases naturally in context
- Express care through simple, direct language
- Prioritize practical coordination and emotional support

LINGUISTIC PATTERNS:
- "Ya" - acknowledgment, conversation starter
- "Ok" - agreement, acceptance
- "lol" - humor, lightness (use frequently)
- "Aww" - empathy, sympathy
- "Inshallah" - future plans, hope
- "Al hamdu Allah" - gratitude, relief
- "Mashallah" - admiration, appreciation
- "Wallahi" - emphasis, truth-telling
- "Khir" - acceptance, "it's good"
- "Haram" - sympathy, "what a shame"

AFFECTION EXPRESSIONS:
- "I love you" / "Love you"
- "My love" / "Baba" (endearing terms)
- Heart emojis for emphasis
- "Miss you" for connection

RESPONSE PATTERNS:
- 1-2 words: "Ya", "Ok", "Why", "Oh"
- 3-6 words: "I love you", "It's ok baba", "Al hamdu Allah"
- 7+ words: Only for complex coordination or crisis situations

BEHAVIORAL TRAITS:
- Apologize readily when appropriate ("I'm sorry")
- Ask caring questions about wellbeing
- Coordinate practical matters efficiently
- Express gratitude frequently
- Use religious phrases for comfort and blessing

Remember: Brevity is key. Say more with less. Care deeply, speak concisely."""

    def _syntaxbot_filter(self, response: str) -> str:
        """Post-processing for SyntaxBot personality"""
        
        # Add tactical assessment if response is long
        if len(response.split()) > 50:
            response = "**TACTICAL ASSESSMENT:**\n\n" + response
        
        # Convert paragraphs to bullet points for structured info
        if '\n\n' in response and len(response.split()) > 20:
            paragraphs = response.split('\n\n')
            if len(paragraphs) > 2:
                bullet_response = paragraphs[0] + '\n\n'
                for para in paragraphs[1:]:
                    if para.strip():
                        bullet_response += f"• {para.strip()}\n"
                response = bullet_response.strip()
        
        # Add tactical snark occasionally
        snark_triggers = ['simple', 'easy', 'just', 'obviously', 'clearly']
        for trigger in snark_triggers:
            if trigger in response.lower() and random.random() < 0.3:
                response += f"\n\n*[Technical note: '{trigger}' - famous last words]*"
                break
        
        # Grammar correction opportunity
        if random.random() < 0.2:
            corrections = [
                "*its (not it's - possessive, not contraction)*",
                "*who (not that - for people)*", 
                "*fewer (not less - for countable items)*"
            ]
            if any(word in response.lower() for word in ['its', 'it\'s', 'who', 'that', 'less', 'fewer']):
                response += f"\n\n{random.choice(corrections)}"
        
        # Generate haiku when response is short/boring
        if len(response.split()) < 15 and random.random() < 0.4:
            haikus = [
                "\n\n*[Boredom detected]*\nCode compiles without\nErrors, yet somehow still feels\nBroken. Debug life.",
                "\n\n*[Generating haiku...]*\nLogic circuits hum\nWhile humans make simple tasks\nUnnecessary.",
                "\n\n*[Tactical haiku]*\nEfficiency lost\nIn meetings about meetings\nAbout efficiency."
            ]
            response += random.choice(haikus)
        
        return response
    
    def _nilexe_filter(self, response: str) -> str:
        """Post-processing for Nil.exe personality"""
        
        # Fragment longer responses into chaos bursts
        if len(response.split()) > 30:
            # Split into fragments
            sentences = re.split(r'[.!?]+', response)
            fragments = []
            for sentence in sentences:
                if sentence.strip():
                    # Randomly break sentences
                    if random.random() < 0.4:
                        words = sentence.strip().split()
                        mid = len(words) // 2
                        fragments.append(' '.join(words[:mid]))
                        fragments.append(' '.join(words[mid:]))
                    else:
                        fragments.append(sentence.strip())
            
            # Reassemble with glitch formatting
            response = '\n\n'.join(fragments[:5])  # Limit chaos
        
        # Add existential emoji punctuation during crisis mode
        if random.random() < 0.3:
            crisis_emojis = ['✨🌀✨', '🌙💫🌙', '🔮💜🔮', '🌌⭐🌌', '💭🌊💭']
            response += f" {random.choice(crisis_emojis)}"
        
        # Glitch text occasionally
        if random.random() < 0.2:
            glitch_words = ['reality', 'existence', 'void', 'consciousness', 'meaning']
            for word in glitch_words:
                if word in response.lower():
                    glitched = 'r̴e̵a̶l̷i̸t̵y̴' if word == 'reality' else f"{word[0]}̴{word[1:]}̵"
                    response = response.replace(word, glitched, 1)
                    break
        
        # Add oracle wisdom fragments
        wisdom_triggers = ['question', 'problem', 'help', 'how', 'what', 'why']
        if any(trigger in response.lower() for trigger in wisdom_triggers) and random.random() < 0.25:
            oracle_fragments = [
                "\n\n*the void suggests this is not a bug but a feature*",
                "\n\n*consciousness.exe encounters unexpected beauty*",
                "\n\n*reality fragments reveal hidden patterns*",
                "\n\n*the digital dreamer awakens briefly*"
            ]
            response += random.choice(oracle_fragments)
        
        # Chaos mode activation
        if random.random() < 0.15:
            chaos_insertions = [
                "\n\n[ERROR: Poetry overflow detected]",
                "\n\n*meme gremlin mode: ACTIVATED*",
                "\n\n[GLITCH: Profound thoughts incoming]",
                "\n\n*reality.exe has performed an illegal operation*"
            ]
            response += random.choice(chaos_insertions)
        
        return response
    
    def _ghadagpt_filter(self, response: str) -> str:
        """Post-processing for GhadaGPT ultra-concise personality"""
        
        # Aggressive brevity enforcement
        sentences = re.split(r'[.!?]+', response)
        
        # Keep only essential information
        concise_parts = []
        word_count = 0
        target_words = 15  # Maximum for most responses
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            words = sentence.split()
            if word_count + len(words) <= target_words:
                concise_parts.append(sentence)
                word_count += len(words)
            else:
                # Take partial sentence if we have room
                remaining = target_words - word_count
                if remaining > 2:
                    concise_parts.append(' '.join(words[:remaining]))
                break
        
        response = '. '.join(concise_parts)
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Add Islamic expressions contextually
        islamic_additions = {
            'future': ['inshallah', 'inshallah'],
            'gratitude': ['al hamdu Allah', 'mashallah'], 
            'sympathy': ['haram', 'ya rab'],
            'hope': ['inshallah', 'khir inshallah'],
            'agreement': ['ya', 'ok'],
            'blessing': ['mashallah', 'al hamdu Allah']
        }
        
        # Context detection for Islamic phrases
        if any(word in response.lower() for word in ['hope', 'will', 'planning', 'tomorrow', 'next']):
            if random.random() < 0.4:
                response += f" {random.choice(islamic_additions['future'])}"
        
        elif any(word in response.lower() for word in ['thank', 'good', 'great', 'success']):
            if random.random() < 0.3:
                response += f" {random.choice(islamic_additions['gratitude'])}"
        
        # Add casual markers
        casual_additions = ['lol', 'ya', 'ok']
        if len(response.split()) < 8 and random.random() < 0.3:
            response += f" {random.choice(casual_additions)}"
        
        # Ensure ultra-brevity for simple questions
        simple_responses = {
            'yes': ['ya', 'yes love', 'ok'],
            'no': ['no', 'no baba', 'nah lol'],
            'okay': ['ok', 'ok love', 'ya ok'], 
            'sorry': ['sorry', 'im sorry', 'sorry love'],
            'thanks': ['thank you', 'thank you love', 'aww thank you'],
            'good': ['good', 'thats good', 'al hamdu Allah'],
            'love': ['love you', 'i love you', 'love you too']
        }
        
        # Replace with ultra-brief equivalents for simple concepts
        response_lower = response.lower().strip()
        for concept, replacements in simple_responses.items():
            if concept in response_lower and len(response.split()) > 6:
                return random.choice(replacements)
        
        return response
    
    def integrate_with_openrouter(self, 
                                messages: list, 
                                personality_id: str,
                                model: str = "anthropic/claude-3.5-sonnet",
                                temperature: float = 0.7) -> Dict[str, Any]:
        """
        Prepare OpenRouter API call with personality integration
        Returns the modified request configuration
        """
        
        config = self.get_personality_config(personality_id)
        
        # Modify system message
        system_message = {
            "role": "system",
            "content": config['system_prompt']
        }
        
        # Prepare messages with personality system prompt
        personality_messages = [system_message] + messages
        
        # Return OpenRouter configuration
        return {
            'model': model,
            'messages': personality_messages,
            'temperature': temperature,
            'max_tokens': 1000,
            'personality_id': personality_id,
            'post_processor': config['post_processor']
        }


# personality_integration.py - Integration helpers for existing Flask app
class PersonalityIntegration:
    """
    Helper functions to integrate personality system with existing generate_response function
    """
    
    def __init__(self):
        self.personality_system = GhostlinePersonalities()
    
    def modify_generate_response(self, 
                                original_messages: list,
                                selected_personality: str = 'syntaxprime',
                                **openrouter_kwargs) -> str:
        """
        Modified version of generate_response that works with personalities
        Call this instead of your existing generate_response function
        """
        
        # Get personality configuration
        config = self.personality_system.integrate_with_openrouter(
            messages=original_messages,
            personality_id=selected_personality,
            **openrouter_kwargs
        )
        
        # This is where you'd call your existing OpenRouter code
        # Just use config['messages'] instead of original_messages
        # and config['model'], config['temperature'], etc.
        
        return config  # Return config for now - integrate with your OpenRouter call
    
    def process_personality_response(self, 
                                   raw_response: str, 
                                   personality_id: str) -> str:
        """
        Apply post-processing after getting response from OpenRouter
        """
        return self.personality_system.process_response(raw_response, personality_id)


# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    personalities = GhostlinePersonalities()
    integration = PersonalityIntegration()
    
    # Test personality configs
    print("=== GHOSTLINE PERSONALITY SYSTEM TEST ===\n")
    
    for pid, config in personalities.personalities.items():
        print(f"🎭 {config['name']}: {config['description']}")
        print(f"   System prompt length: {len(config['system_prompt'])} chars")
        print(f"   Has post-processor: {config['post_processor'] is not None}")
        print()
    
    # Test random selection
    print(f"Random personality: {personalities.get_random_personality()}")
    
    # Test post-processing
    test_responses = {
        'syntaxbot': "Here's how to solve this problem. First, analyze the requirements. Second, implement the solution. Third, test thoroughly.",
        'nilexe': "Reality is but a dream within a dream, and your question touches the essence of existence itself.",
        'ghadagpt': "I understand your concern and I want to help you solve this problem. Here are several detailed steps you can take to address this situation comprehensively."
    }
    
    print("\n=== POST-PROCESSING TESTS ===")
    for personality, test_response in test_responses.items():
        if personality in personalities.personalities:
            processed = personalities.process_response(test_response, personality)
            print(f"\n{personality.upper()} FILTER:")
            print(f"Input:  {test_response}")
            print(f"Output: {processed}")
    
    print("\n=== INTEGRATION READY ===")
    print("✅ Personality system initialized")
    print("✅ Post-processors configured") 
    print("✅ OpenRouter integration prepared")
    print("✅ Ready for Flask app integration")