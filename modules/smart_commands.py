#!/usr/bin/env python3
"""
Smart Commands Module - FIXED VERSION
====================================
This module handles smart command detection and routing to prevent conflicts
between content creation and integration commands.

Key Fix: "draft email about [subject]" now routes to content creation 
instead of Gmail integration.
"""

import re
import datetime
from typing import Dict, Any, Tuple, Optional, List

# =============================================================================
# EMAIL COMMAND CLASSIFICATION
# =============================================================================

def classify_email_command(user_input: str) -> str:
    """
    Classify email commands into different routing categories
    
    Returns:
        - "content_creation": Should generate draft content for review
        - "gmail_integration": Should use Gmail API to send actual emails  
        - "cloze_integration": Should use Cloze contacts for email drafting
        - "unknown": Not an email command
    """
    user_lower = user_input.lower().strip()
    
    # CONTENT CREATION patterns - should create drafts for review
    content_patterns = [
        r'\bdraft\s+email\s+about\b',           # "draft email about quarterly results"
        r'\bwrite\s+email\s+about\b',           # "write email about project update"
        r'\bcompose\s+email\s+about\b',         # "compose email about meeting"
        r'\bcreate\s+email\s+(for|about)\b',    # "create email for newsletter"
        r'\bemail\s+draft\s+(for|about)\b',     # "email draft for clients"
        r'\bemail\s+template\s+(for|about)\b',  # "email template for onboarding"
        r'\bdraft\s+.+\s+email\b',              # "draft quarterly email"
        r'\bwrite\s+.+\s+email\b',              # "write marketing email"
        r'\bemail\s+content\s+(for|about)\b',   # "email content for campaign"
        r'\bcompose\s+.+\s+email\b',            # "compose newsletter email"
    ]
    
    for pattern in content_patterns:
        if re.search(pattern, user_lower):
            return "content_creation"
    
    # GMAIL INTEGRATION patterns - should send actual emails
    integration_patterns = [
        r'\bsend\s+email\s+to\s+\S+@\S+',       # "send email to john@company.com"
        r'\bemail\s+\S+@\S+',                   # "email sarah@example.org"
        r'\bcompose\s+email\s+to\s+\S+@\S+',    # "compose email to team@startup.com"
        r'\bdraft\s+email\s+to\s+\S+@\S+',      # "draft email to client@company.com"
        r'\bwrite\s+email\s+to\s+\S+@\S+',      # "write email to boss@work.com"
        r'\bsend\s+to\s+\S+@\S+',               # "send to contact@business.com"
        r'\bmail\s+\S+@\S+',                    # "mail info@company.com"
    ]
    
    for pattern in integration_patterns:
        if re.search(pattern, user_lower):
            return "gmail_integration"
    
    # CLOZE INTEGRATION patterns - contact names (capitalized words)
    cloze_patterns = [
        r'\bdraft\s+email\s+to\s+[A-Z][a-z]+',  # "draft email to John"
        r'\bwrite\s+email\s+to\s+[A-Z][a-z]+',  # "write email to Sarah"
        r'\bemail\s+[A-Z][a-z]+\s+about',       # "email John about meeting"
        r'\bcompose\s+email\s+to\s+[A-Z][a-z]+', # "compose email to Mary"
        r'\bsend\s+email\s+to\s+[A-Z][a-z]+',   # "send email to Bob"
    ]
    
    for pattern in cloze_patterns:
        if re.search(pattern, user_input):  # Use original case for name detection
            return "cloze_integration"
    
    return "unknown"

# =============================================================================
# SMART COMMANDS PROCESSOR
# =============================================================================

def process_smart_commands(user_input: str, project: str, use_voices: List[str], random_toggle: bool) -> Tuple[Dict[str, Any], bool]:
    """
    Process smart commands for content creation
    
    This function should be called BEFORE Gmail integration to handle
    content creation requests like "draft email about [topic]"
    """
    
    command_type = classify_email_command(user_input)
    
    if command_type != "content_creation":
        return {}, False
    
    print(f"🎯 Smart Commands: Processing content creation request: '{user_input}'")
    
    try:
        # Import AI response generator
        from utils.ghostline_engine import generate_response
        from utils.rag_basic import enhanced_retrieve, is_ready
        
        # Extract the topic/subject from the command
        topic = extract_email_topic(user_input)
        
        # Create a specialized prompt for email draft creation
        draft_prompt = create_email_draft_prompt(user_input, topic)
        
        # Get contextual information if available
        retrieval_ctx = enhanced_retrieve(draft_prompt, k=5, project=project) if is_ready() else []
        
        # Generate the email draft content
        response_data = generate_response(
            draft_prompt, use_voices, random_toggle,
            project=project, retrieval_context=retrieval_ctx
        )
        
        # Add metadata to indicate this is a draft for review
        for voice in response_data:
            if isinstance(response_data[voice], str):
                response_data[voice] = format_draft_response(response_data[voice], topic)
        
        print(f"✅ Smart Commands: Email draft created successfully")
        return response_data, True
        
    except ImportError as e:
        print(f"❌ Smart Commands: Import error - {e}")
        error_response = {
            "SyntaxPrime": f"**Email Draft Creation Failed**\n\nMissing dependency: {str(e)}\n\nPlease ensure the AI response system is properly configured."
        }
        return error_response, True
        
    except Exception as e:
        print(f"❌ Smart Commands: Content creation failed: {e}")
        error_response = {
            "SyntaxPrime": f"**Email Draft Creation Failed**\n\nError: {str(e)}\n\nPlease try rephrasing your request or use a simpler format like 'draft email about [topic]'"
        }
        return error_response, True

def extract_email_topic(user_input: str) -> Optional[str]:
    """Extract the topic/subject from an email command"""
    user_lower = user_input.lower()
    
    # Look for topic indicators
    topic_patterns = [
        r'about\s+(.+)$',           # "draft email about quarterly results"
        r'regarding\s+(.+)$',       # "write email regarding project status" 
        r'for\s+(.+)$',             # "email template for new clients"
        r'on\s+(.+)$',              # "compose email on budget review"
        r'concerning\s+(.+)$',      # "draft email concerning policy changes"
    ]
    
    for pattern in topic_patterns:
        match = re.search(pattern, user_lower)
        if match:
            # Return original case from user input
            original_match = re.search(pattern, user_input, re.IGNORECASE)
            if original_match:
                return original_match.group(1).strip()
            return match.group(1).strip()
    
    return None

def create_email_draft_prompt(user_input: str, topic: Optional[str]) -> str:
    """Create a specialized prompt for email draft generation"""
    
    base_prompt = "Create a professional email draft based on the following request:\n\n"
    base_prompt += f"User Request: {user_input}\n\n"
    
    if topic:
        base_prompt += f"Email Topic: {topic}\n\n"
    
    base_prompt += """Instructions:
• Write a complete, professional email ready for review and editing
• Include an appropriate subject line
• Use a professional but friendly tone
• Make the content clear and actionable
• Do not include recipient email addresses (use placeholders like [Recipient Name])
• Structure with proper greeting, body, and closing
• Format as a complete email draft that can be copied and pasted

Important: This is a DRAFT for review - do NOT send this email automatically.
The user will review, edit, and send it themselves when ready.

Email Draft:"""
    
    return base_prompt

def format_draft_response(response_content: str, topic: Optional[str]) -> str:
    """Format the draft response with helpful metadata"""
    
    formatted_response = f"📧 **Email Draft Created**"
    
    if topic:
        formatted_response += f" - {topic}"
    
    formatted_response += f"\n\n{response_content}\n\n"
    formatted_response += "---\n\n"
    formatted_response += "💡 **Next Steps:**\n"
    formatted_response += "• Review and edit the content above\n"
    formatted_response += "• Copy to your email client when ready\n"
    formatted_response += "• Use `send email to [address]` to send via Gmail integration\n"
    formatted_response += "• Use `draft email to [name]` to create contact-specific drafts\n\n"
    formatted_response += "*This draft was created for review - no emails were sent automatically.*"
    
    return formatted_response

# =============================================================================
# INTENT DETECTION (Legacy Support)
# =============================================================================

def detect_intent(user_input: str) -> str:
    """
    Legacy function for backward compatibility
    Detects the general intent of user input
    """
    user_lower = user_input.lower().strip()
    
    # Email content creation
    email_type = classify_email_command(user_input)
    if email_type == "content_creation":
        return "content_creation"
    
    # Casual conversation patterns
    casual_patterns = [
        r'^\b(hello|hi|hey|good\s+(morning|afternoon|evening))\b',
        r'^\b(thanks?|thank\s+you|goodbye|bye)\b',
        r'\bhow\s+are\s+you\b',
        r'\bwhat\'?s\s+up\b',
    ]
    
    if any(re.search(pattern, user_lower) for pattern in casual_patterns):
        return "casual"
    
    # Analysis mode patterns
    analysis_patterns = [
        r'\b(marketing|business)\s+(plan|strategy)\b',
        r'\bboard\s+report\b',
        r'\bexecutive\s+(summary|report)\b',
        r'\banalys[ie]s\s+of\b',
        r'\bcreate\s+(plan|strategy|analysis)\b',
    ]
    
    if any(re.search(pattern, user_lower) for pattern in analysis_patterns):
        return "analysis_mode"
    
    # Morning briefing patterns
    morning_patterns = [
        r'\b(good\s+morning|daily\s+briefing|morning\s+briefing)\b',
        r'\bbrief\s+me\b',
        r'\bwhat\'?s\s+(on\s+)?my\s+agenda\b',
    ]
    
    if any(re.search(pattern, user_lower) for pattern in morning_patterns):
        return "morning_briefing"
    
    return "general"

def matches_pattern(text: str, patterns: List[str]) -> bool:
    """Check if text matches any of the regex patterns (case insensitive)"""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False

# =============================================================================
# TESTING AND DEBUGGING
# =============================================================================

def test_email_classification():
    """Test the email command classification system"""
    
    test_cases = [
        # Should route to CONTENT CREATION
        ("draft email about quarterly results", "content_creation"),
        ("write email about project update", "content_creation"), 
        ("compose email about meeting follow-up", "content_creation"),
        ("create email for newsletter", "content_creation"),
        ("email template for onboarding", "content_creation"),
        ("draft marketing email", "content_creation"),
        ("write quarterly email", "content_creation"),
        ("email content for campaign", "content_creation"),
        
        # Should route to GMAIL INTEGRATION  
        ("send email to john@company.com", "gmail_integration"),
        ("email sarah@example.org about meeting", "gmail_integration"),
        ("compose email to team@startup.com", "gmail_integration"),
        ("draft email to client@business.com", "gmail_integration"),
        ("mail info@company.com", "gmail_integration"),
        
        # Should route to CLOZE INTEGRATION
        ("draft email to John Smith", "cloze_integration"),
        ("write email to Sarah about project", "cloze_integration"),
        ("email John about meeting", "cloze_integration"),
        ("compose email to Mary", "cloze_integration"),
        
        # Should be UNKNOWN
        ("check my email", "unknown"),
        ("what's in my inbox", "unknown"),
        ("good morning", "unknown"),
        ("draft presentation", "unknown"),
    ]
    
    print("🧪 Testing Email Command Classification\n")
    
    passed = 0
    failed = 0
    
    for user_input, expected in test_cases:
        result = classify_email_command(user_input)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        
        print(f"{status} | '{user_input}' → {result} (expected: {expected})")
        
        if result == expected:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    return failed == 0

def log_command_routing(user_input: str, classification: str, handler: str, success: bool):
    """Log command routing for debugging"""
    
    timestamp = datetime.datetime.now().isoformat()
    status = "✅" if success else "❌"
    
    print(f"📋 COMMAND ROUTING LOG [{timestamp}]:")
    print(f"   Input: '{user_input}'")
    print(f"   Classification: {classification}")
    print(f"   Handler: {handler}")
    print(f"   Success: {status}")
    print()

def verify_smart_commands_setup():
    """Verify smart commands are properly configured"""
    
    print("🔍 Verifying Smart Commands Setup...")
    
    checks = {
        "Email Classification": test_email_classification,
        "Pattern Matching": lambda: matches_pattern("test email", [r'\bemail\b']),
        "Topic Extraction": lambda: extract_email_topic("draft email about testing") == "testing",
    }
    
    results = {}
    for check_name, check_func in checks.items():
        try:
            result = check_func()
            results[check_name] = "✅ PASS" if result else "❌ FAIL"
        except Exception as e:
            results[check_name] = f"❌ ERROR: {e}"
    
    print("\n📊 Setup Verification Results:")
    for check_name, result in results.items():
        print(f"   {check_name}: {result}")
    
    return all("✅" in result for result in results.values())

# =============================================================================
# LEGACY COMPATIBILITY 
# =============================================================================

# Maintain compatibility with existing test file
TEST_CASES = [
    # Cases that SHOULD NOT trigger smart commands
    {
        "input": "Praying fajr and cup two!",
        "expected": "casual",
        "should_not_trigger": ["clickup", "content_mode", "analysis_mode"]
    },
    {
        "input": "I had my first cup of coffee",
        "expected": "casual", 
        "should_not_trigger": ["clickup"]
    },
    {
        "input": "Let's plan our weekend",
        "expected": "general",
        "should_not_trigger": ["marketing_plan", "analysis_mode"]
    },
    {
        "input": "Thanks for the help!",
        "expected": "casual",
        "should_not_trigger": ["content_mode", "analysis_mode"]
    },
    
    # Cases that SHOULD trigger smart commands
    {
        "input": "write email about project status",
        "expected": "content_creation",
        "should_trigger": ["email_mode"]
    },
    {
        "input": "draft email about quarterly results",
        "expected": "content_creation",
        "should_trigger": ["email_mode"]
    },
    {
        "input": "create marketing plan for Q4",
        "expected": "analysis_mode", 
        "should_trigger": ["marketing_plan"]
    },
    {
        "input": "daily briefing please",
        "expected": "morning_briefing",
        "should_trigger": ["briefing"]
    },
]

def test_intent_detection(user_input: str) -> str:
    """Legacy function for existing test compatibility"""
    return detect_intent(user_input)

def run_tests():
    """Run legacy test cases for backward compatibility"""
    print("🧪 Running Legacy Compatibility Tests\n")
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(TEST_CASES, 1):
        user_input = test["input"]
        expected = test["expected"] 
        
        result = test_intent_detection(user_input)
        
        print(f"Test {i}: '{user_input}'")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        
        if result == expected:
            print(f"  ✅ PASS\n")
            passed += 1
        else:
            print(f"  ❌ FAIL\n")
            failed += 1
    
    print(f"📊 Legacy Test Results: {passed} passed, {failed} failed")
    
    # Also run new email classification tests
    print("\n" + "="*50)
    email_test_passed = test_email_classification()
    
    return failed == 0 and email_test_passed

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 Smart Commands Module - Testing Suite")
    print("="*50)
    
    # Run all tests
    legacy_passed = run_tests()
    print("\n" + "="*50)
    setup_verified = verify_smart_commands_setup()
    
    print("\n🎯 FINAL RESULTS:")
    print(f"   Legacy Tests: {'✅ PASS' if legacy_passed else '❌ FAIL'}")
    print(f"   Setup Verification: {'✅ PASS' if setup_verified else '❌ FAIL'}")
    
    if legacy_passed and setup_verified:
        print("\n🎉 Smart Commands module is ready for deployment!")
        print("\n💡 To fix the email routing issue, update app.py with:")
        print("   1. Import: from modules.smart_commands import process_smart_commands, classify_email_command")
        print("   2. Add smart commands processing BEFORE Gmail integration")
        print("   3. Test with: 'draft email about quarterly results'")
    else:
        print("\n⚠️  Issues detected - review test results above")
    
    print("\n" + "="*50)