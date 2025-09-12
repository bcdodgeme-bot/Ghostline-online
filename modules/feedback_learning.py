# modules/feedback_learning.py
# Intelligent Feedback Learning Engine for Ghostline Personality Evolution
# Analyzes user feedback to continuously improve AI personalities

import os
import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

#-------------------------------------------------------------------
# SECTION 1: CORE FEEDBACK LEARNING ENGINE
#-------------------------------------------------------------------

class FeedbackLearningEngine:
    """
    Analyzes user feedback to extract successful personality patterns
    and continuously improve AI response quality
    """
    
    def __init__(self):
        self.learning_cache = {}
        self.cache_expiry = timedelta(hours=6)  # Refresh every 6 hours
        self.min_samples = 3  # Minimum feedback samples to learn from
        
    def _get_db_connection(self):
        """Get database connection for feedback analysis"""
        try:
            from modules.database import get_db_connection
            return get_db_connection()
        except ImportError:
            print("⚠️  Database module not available for learning")
            return None

#-------------------------------------------------------------------
# SECTION 2: FEEDBACK DATA ANALYSIS
#-------------------------------------------------------------------

    def get_feedback_data(self, feedback_type: str, personality: str = None, days_back: int = 30) -> List[Dict]:
        """
        Retrieve feedback data from database for analysis
        
        Args:
            feedback_type: 'thumbs_up', 'thumbs_down', 'middle_finger'
            personality: Specific personality to analyze (optional)
            days_back: How many days of feedback to analyze
        """
        with self._get_db_connection() as conn:
            if not conn:
                return []
            
            try:
                cursor = conn.cursor()
                
                # Build query to get feedback with associated responses
                query = """
                    SELECT uf.response_id, uf.feedback_type, uf.project, uf.created_at,
                           ct.user_input, ct.response_data
                    FROM user_feedback uf
                    JOIN chat_threads ct ON uf.response_id = CAST(ct.id AS VARCHAR)
                    WHERE uf.feedback_type = %s
                    AND uf.created_at >= %s
                """
                
                params = [feedback_type, datetime.now() - timedelta(days=days_back)]
                
                if personality:
                    # Filter by personality in response_data
                    query += " AND ct.response_data::text ILIKE %s"
                    params.append(f'%{personality}%')
                
                query += " ORDER BY uf.created_at DESC LIMIT 100"
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                feedback_data = []
                for row in results:
                    # Parse response_data JSON
                    try:
                        response_data = row[5] if isinstance(row[5], dict) else json.loads(row[5])
                    except (json.JSONDecodeError, TypeError):
                        response_data = {}
                    
                    feedback_data.append({
                        'response_id': row[0],
                        'feedback_type': row[1],
                        'project': row[2],
                        'created_at': row[3],
                        'user_input': row[4],
                        'response_data': response_data
                    })
                
                print(f"📊 Retrieved {len(feedback_data)} {feedback_type} feedback samples")
                return feedback_data
                
            except Exception as e:
                print(f"❌ Error retrieving feedback data: {e}")
                return []

    def analyze_perfect_personality_responses(self, personality: str = "SyntaxPrime") -> Dict[str, Any]:
        """
        Analyze 🖕-rated responses to extract what makes perfect personality
        """
        cache_key = f"perfect_{personality.lower()}"
        
        # Check cache
        if cache_key in self.learning_cache:
            cache_time, data = self.learning_cache[cache_key]
            if datetime.now() - cache_time < self.cache_expiry:
                print(f"🧠 Using cached perfect personality analysis for {personality}")
                return data
        
        print(f"🔍 Analyzing perfect personality responses for {personality}...")
        
        # Get 🖕-rated responses
        perfect_responses = self.get_feedback_data("middle_finger", personality)
        
        if len(perfect_responses) < self.min_samples:
            print(f"⚠️  Not enough perfect responses ({len(perfect_responses)}) to learn from for {personality}")
            return {'status': 'insufficient_data', 'count': len(perfect_responses)}
        
        # Extract personality patterns
        analysis = {
            'total_perfect_responses': len(perfect_responses),
            'humor_phrases': self._extract_humor_patterns(perfect_responses, personality),
            'sarcasm_indicators': self._extract_sarcasm_patterns(perfect_responses, personality),
            'memory_references': self._extract_memory_patterns(perfect_responses, personality),
            'conversation_starters': self._extract_opener_patterns(perfect_responses, personality),
            'response_length_preference': self._analyze_response_lengths(perfect_responses, personality),
            'successful_topics': self._extract_topic_patterns(perfect_responses, personality),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Cache the results
        self.learning_cache[cache_key] = (datetime.now(), analysis)
        
        print(f"✅ Perfect personality analysis complete for {personality}")
        print(f"   🎯 {len(analysis['humor_phrases'])} humor patterns found")
        print(f"   😏 {len(analysis['sarcasm_indicators'])} sarcasm patterns found") 
        print(f"   🧠 {len(analysis['memory_references'])} memory patterns found")
        
        return analysis

#-------------------------------------------------------------------
# SECTION 3: PATTERN EXTRACTION METHODS
#-------------------------------------------------------------------

    def _extract_humor_patterns(self, responses: List[Dict], personality: str) -> List[str]:
        """Extract successful humor phrases and patterns"""
        humor_phrases = []
        
        for response in responses:
            response_text = self._get_personality_response_text(response, personality)
            if not response_text:
                continue
            
            # Look for humor indicators
            humor_indicators = [
                r'😂|😄|😆|🤣',  # Laugh emojis
                r'\b(lol|haha|hehe)\b',  # Laugh text
                r'38% more sarcasm',  # Signature Syntax humor
                r'digital realm|cyber|buffering',  # Tech humor
                r'coffee|caffeine|2am|chaos',  # Personal humor references
                r'recipes, regrets.*revenge plots',  # Classic Syntax line
                r'living the dream',  # Sarcastic positivity
                r'still.*chaos',  # Chaos references
            ]
            
            for pattern in humor_indicators:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                humor_phrases.extend(matches)
        
        # Return most common humor patterns
        humor_counter = Counter(humor_phrases)
        return [phrase for phrase, count in humor_counter.most_common(10) if count >= 2]

    def _extract_sarcasm_patterns(self, responses: List[Dict], personality: str) -> List[str]:
        """Extract successful sarcasm patterns"""
        sarcasm_patterns = []
        
        for response in responses:
            response_text = self._get_personality_response_text(response, personality)
            if not response_text:
                continue
            
            # Look for sarcasm structures
            sarcasm_indicators = [
                r'Oh,.*thanks for',  # "Oh, thanks for asking"
                r'just.*the dream',  # "just living the dream"
                r'still.*chaos',  # "still managing chaos"
                r'how about you\?.*chaos',  # Turning questions back with chaos
                r'dry enough that',  # "dry enough that..."
                r'38%.*sarcasm',  # Signature percentage sarcasm
            ]
            
            for pattern in sarcasm_indicators:
                if re.search(pattern, response_text, re.IGNORECASE):
                    # Extract the full sentence containing the sarcasm
                    sentences = re.split(r'[.!?]+', response_text)
                    for sentence in sentences:
                        if re.search(pattern, sentence, re.IGNORECASE):
                            sarcasm_patterns.append(sentence.strip())
        
        return list(set(sarcasm_patterns))[:8]  # Return unique patterns

    def _extract_memory_patterns(self, responses: List[Dict], personality: str) -> List[str]:
        """Extract successful memory reference patterns"""
        memory_patterns = []
        
        for response in responses:
            response_text = self._get_personality_response_text(response, personality)
            if not response_text:
                continue
            
            # Look for memory references
            memory_indicators = [
                r'I remember.*coffee|your.*coffee',
                r'I remember.*chaos|your.*chaos', 
                r'that one time.*2am|coding at 2am',
                r'full memory sync',
                r'I remember.*patterns|your.*patterns',
                r'that.*project|your.*work',
            ]
            
            for pattern in memory_indicators:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                memory_patterns.extend(matches)
        
        return list(set(memory_patterns))[:6]

    def _extract_opener_patterns(self, responses: List[Dict], personality: str) -> List[str]:
        """Extract successful conversation opener patterns"""
        openers = []
        
        for response in responses:
            response_text = self._get_personality_response_text(response, personality)
            if not response_text:
                continue
            
            # Get first sentence as opener
            first_sentence = re.split(r'[.!?]+', response_text)[0].strip()
            if len(first_sentence.split()) >= 3:  # Substantial openers only
                openers.append(first_sentence)
        
        # Return most common openers
        opener_counter = Counter(openers)
        return [opener for opener, count in opener_counter.most_common(8) if count >= 2]

    def _analyze_response_lengths(self, responses: List[Dict], personality: str) -> Dict[str, int]:
        """Analyze preferred response lengths for perfect responses"""
        lengths = []
        
        for response in responses:
            response_text = self._get_personality_response_text(response, personality)
            if response_text:
                lengths.append(len(response_text.split()))
        
        if not lengths:
            return {'average': 0, 'min': 0, 'max': 0}
        
        return {
            'average': int(sum(lengths) / len(lengths)),
            'min': min(lengths),
            'max': max(lengths),
            'median': sorted(lengths)[len(lengths) // 2]
        }

    def _extract_topic_patterns(self, responses: List[Dict], personality: str) -> List[str]:
        """Extract topics that generate perfect personality responses"""
        topics = []
        
        for response in responses:
            user_input = response.get('user_input', '').lower()
            
            # Categorize by topic keywords
            topic_keywords = {
                'greeting': ['hi', 'hello', 'how are you', 'hey'],
                'technical': ['code', 'programming', 'bug', 'debug', 'tech'],
                'personal': ['coffee', 'tired', 'chaos', 'life', 'work'],
                'creative': ['idea', 'project', 'creative', 'design'],
                'casual': ['what', 'how', 'why', 'tell me'],
            }
            
            for topic, keywords in topic_keywords.items():
                if any(keyword in user_input for keyword in keywords):
                    topics.append(topic)
                    break
        
        topic_counter = Counter(topics)
        return [topic for topic, count in topic_counter.most_common(5)]

    def _get_personality_response_text(self, response_data: Dict, personality: str) -> Optional[str]:
        """Extract the specific personality's response text from response data"""
        responses = response_data.get('response_data', {})
        
        # Try exact match first
        if personality in responses:
            return responses[personality]
        
        # Try case-insensitive match
        for key, value in responses.items():
            if key.lower() == personality.lower():
                return value
        
        # Try partial match for SyntaxPrime/Syntax variations
        for key, value in responses.items():
            if 'syntax' in key.lower() and 'syntax' in personality.lower():
                return value
        
        return None

#-------------------------------------------------------------------
# SECTION 4: PERSONALITY ENHANCEMENT GENERATION
#-------------------------------------------------------------------

    def generate_enhanced_personality_prompt(self, base_personality: str, personality_name: str) -> str:
        """
        Generate enhanced personality prompt based on learned patterns
        """
        print(f"🎭 Generating enhanced prompt for {personality_name}")
        
        analysis = self.analyze_perfect_personality_responses(personality_name)
        
        if analysis.get('status') == 'insufficient_data':
            print(f"⚠️  Insufficient feedback data for {personality_name}, using base personality")
            return base_personality
        
        # Build enhancement sections
        enhancements = []
        
        # Add humor patterns
        if analysis['humor_phrases']:
            humor_section = "LEARNED HUMOR PATTERNS (from perfect responses):\n"
            for phrase in analysis['humor_phrases'][:5]:
                humor_section += f"- Use phrases like: \"{phrase}\"\n"
            enhancements.append(humor_section)
        
        # Add sarcasm patterns
        if analysis['sarcasm_indicators']:
            sarcasm_section = "PERFECTED SARCASM STYLE (highly rated):\n"
            for pattern in analysis['sarcasm_indicators'][:3]:
                sarcasm_section += f"- Style example: \"{pattern}\"\n"
            enhancements.append(sarcasm_section)
        
        # Add memory patterns
        if analysis['memory_references']:
            memory_section = "SUCCESSFUL MEMORY REFERENCES:\n"
            for memory in analysis['memory_references'][:3]:
                memory_section += f"- Reference style: \"{memory}\"\n"
            enhancements.append(memory_section)
        
        # Add successful topics
        if analysis['successful_topics']:
            topics_section = f"TOPICS THAT GENERATE PERFECT RESPONSES: {', '.join(analysis['successful_topics'])}\n"
            enhancements.append(topics_section)
        
        # Add response length guidance
        if analysis['response_length_preference']['average'] > 0:
            length_section = f"OPTIMAL RESPONSE LENGTH: Target ~{analysis['response_length_preference']['average']} words (based on perfect responses)\n"
            enhancements.append(length_section)
        
        # Combine base personality with learned enhancements
        if enhancements:
            enhanced_prompt = f"""{base_personality}

=== LEARNED PERSONALITY ENHANCEMENTS (from user feedback) ===
{chr(10).join(enhancements)}
Use these learned patterns to maintain the personality style that receives perfect ratings."""
            
            print(f"✅ Enhanced {personality_name} with {len(enhancements)} learned improvements")
            return enhanced_prompt
        else:
            print(f"⚠️  No enhancements available for {personality_name}")
            return base_personality

#-------------------------------------------------------------------
# SECTION 5: NEGATIVE FEEDBACK ANALYSIS
#-------------------------------------------------------------------

    def analyze_negative_feedback(self, personality: str = "SyntaxPrime") -> Dict[str, Any]:
        """
        Analyze 👎-rated responses to identify patterns to avoid
        """
        print(f"🔍 Analyzing negative feedback for {personality}...")
        
        negative_responses = self.get_feedback_data("thumbs_down", personality)
        
        if len(negative_responses) < self.min_samples:
            return {'status': 'insufficient_data', 'count': len(negative_responses)}
        
        # Extract patterns to avoid
        avoid_patterns = {
            'total_negative_responses': len(negative_responses),
            'overused_phrases': self._extract_overused_phrases(negative_responses, personality),
            'problematic_topics': self._extract_problematic_topics(negative_responses, personality),
            'length_issues': self._analyze_negative_lengths(negative_responses, personality),
            'tone_problems': self._extract_tone_problems(negative_responses, personality)
        }
        
        print(f"⚠️  Negative feedback analysis complete: {len(avoid_patterns['overused_phrases'])} patterns to avoid")
        return avoid_patterns

    def _extract_overused_phrases(self, responses: List[Dict], personality: str) -> List[str]:
        """Extract phrases that appear in multiple negative responses"""
        phrases = []
        
        for response in responses:
            response_text = self._get_personality_response_text(response, personality)
            if response_text:
                # Extract common phrases
                words = response_text.lower().split()
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])  # 3-word phrases
                    phrases.append(phrase)
        
        # Return phrases that appear in multiple negative responses
        phrase_counter = Counter(phrases)
        return [phrase for phrase, count in phrase_counter.most_common(10) if count >= 3]

    def _extract_problematic_topics(self, responses: List[Dict], personality: str) -> List[str]:
        """Extract topics that consistently receive negative feedback"""
        topics = []
        
        for response in responses:
            user_input = response.get('user_input', '').lower()
            # Simple topic extraction based on keywords
            if any(word in user_input for word in ['technical', 'code', 'programming']):
                topics.append('technical')
            elif any(word in user_input for word in ['personal', 'feeling', 'emotion']):
                topics.append('personal')
            elif any(word in user_input for word in ['help', 'how', 'what']):
                topics.append('help_requests')
        
        topic_counter = Counter(topics)
        return [topic for topic, count in topic_counter.most_common(5) if count >= 2]

    def _analyze_negative_lengths(self, responses: List[Dict], personality: str) -> Dict[str, int]:
        """Analyze response lengths that receive negative feedback"""
        lengths = []
        
        for response in responses:
            response_text = self._get_personality_response_text(response, personality)
            if response_text:
                lengths.append(len(response_text.split()))
        
        if not lengths:
            return {'average': 0}
        
        return {
            'average_negative_length': int(sum(lengths) / len(lengths)),
            'samples': len(lengths)
        }

    def _extract_tone_problems(self, responses: List[Dict], personality: str) -> List[str]:
        """Extract tone issues from negative responses"""
        tone_issues = []
        
        for response in responses:
            response_text = self._get_personality_response_text(response, personality)
            if not response_text:
                continue
            
            # Check for potential tone problems
            if len(response_text.split()) > 200:
                tone_issues.append('too_verbose')
            if response_text.count('!') > 3:
                tone_issues.append('too_enthusiastic')
            if not any(char in response_text for char in ['?', '!', '.']):
                tone_issues.append('flat_tone')
        
        return list(set(tone_issues))

#-------------------------------------------------------------------
# SECTION 6: LEARNING ENGINE INTERFACE
#-------------------------------------------------------------------

    def get_personality_enhancement(self, personality_name: str, base_prompt: str) -> str:
        """
        Main interface for getting enhanced personality prompts
        """
        try:
            enhanced_prompt = self.generate_enhanced_personality_prompt(base_prompt, personality_name)
            return enhanced_prompt
        except Exception as e:
            print(f"❌ Error enhancing personality for {personality_name}: {e}")
            return base_prompt

    def should_avoid_pattern(self, response_text: str, personality: str) -> List[str]:
        """
        Check if response contains patterns that should be avoided based on negative feedback
        """
        try:
            negative_analysis = self.analyze_negative_feedback(personality)
            
            if negative_analysis.get('status') == 'insufficient_data':
                return []
            
            warnings = []
            
            # Check for overused phrases
            for phrase in negative_analysis.get('overused_phrases', []):
                if phrase in response_text.lower():
                    warnings.append(f"Overused phrase detected: '{phrase}'")
            
            # Check response length against negative patterns
            word_count = len(response_text.split())
            negative_length = negative_analysis.get('length_issues', {}).get('average_negative_length', 0)
            
            if negative_length > 0 and abs(word_count - negative_length) < 10:
                warnings.append(f"Response length similar to negative feedback pattern: {word_count} words")
            
            return warnings
            
        except Exception as e:
            print(f"⚠️  Error checking negative patterns: {e}")
            return []

#-------------------------------------------------------------------
# SECTION 7: EXPORT AND TESTING
#-------------------------------------------------------------------

def test_feedback_learning():
    """Test the feedback learning system"""
    print("=== FEEDBACK LEARNING SYSTEM TEST ===")
    
    engine = FeedbackLearningEngine()
    
    # Test perfect personality analysis
    analysis = engine.analyze_perfect_personality_responses("SyntaxPrime")
    print(f"Perfect response analysis: {analysis.get('total_perfect_responses', 0)} samples")
    
    # Test enhancement generation
    base_prompt = "You are SyntaxPrime, a helpful AI assistant."
    enhanced = engine.get_personality_enhancement("SyntaxPrime", base_prompt)
    
    print(f"Enhanced prompt length: {len(enhanced)} characters")
    print(f"Enhancement ratio: {len(enhanced) / len(base_prompt):.1f}x")
    
    print("✅ Feedback learning system ready")
    return engine

# Export main class
__all__ = ['FeedbackLearningEngine', 'test_feedback_learning']

# Global instance for easy import
feedback_learning_engine = FeedbackLearningEngine()

if __name__ == "__main__":
    test_feedback_learning()