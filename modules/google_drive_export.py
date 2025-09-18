# =============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# =============================================================================

# modules/google_drive_export.py - Enhanced Google Drive Export for Chat Content

import os
import datetime
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# =============================================================================
# SECTION 2: MAIN GOOGLE DRIVE EXPORTER CLASS
# =============================================================================

class GoogleDriveExporter:
    """Enhanced Google Drive integration for exporting chat content with proper formatting"""
    
    def __init__(self, credentials=None):
        """Initialize with Google API credentials"""
        self.credentials = credentials
        self.docs_service = None
        self.drive_service = None
        
        if credentials:
            self._initialize_services()
    
    def _initialize_services(self):
        """Initialize Google API services"""
        try:
            self.docs_service = build('docs', 'v1', credentials=self.credentials, cache_discovery=False)
            self.drive_service = build('drive', 'v3', credentials=self.credentials, cache_discovery=False)
            print("Google Drive export services initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Google services: {e}")
            raise

# =============================================================================
# SECTION 3: MARKDOWN PROCESSING AND CONTENT CLEANING
# =============================================================================

    def clean_markdown_for_docs(self, markdown_content: str) -> str:
        """Clean markdown content for Google Docs insertion - SAFE VERSION
        
        Args:
            markdown_content: Raw markdown content with markdown formatting
            
        Returns:
            Cleaned content suitable for Google Docs (plain text)
        """
        if not markdown_content:
            return ""
        
        content = str(markdown_content)
        
        # Clean up headings - convert to plain text with visual hierarchy
        content = re.sub(r'^# (.+)$', r'\1', content, flags=re.MULTILINE)      # H1
        content = re.sub(r'^## (.+)$', r'\1', content, flags=re.MULTILINE)     # H2
        content = re.sub(r'^### (.+)$', r'\1', content, flags=re.MULTILINE)    # H3
        content = re.sub(r'^#### (.+)$', r'\1', content, flags=re.MULTILINE)   # H4
        content = re.sub(r'^##### (.+)$', r'\1', content, flags=re.MULTILINE)  # H5
        content = re.sub(r'^###### (.+)$', r'\1', content, flags=re.MULTILINE) # H6
        
        # Clean up formatting markers but preserve the text content
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Bold: **text** -> text
        content = re.sub(r'__(.*?)__', r'\1', content)      # Bold alt: __text__ -> text
        content = re.sub(r'(?<!\*)\*([^*\n]+)\*(?!\*)', r'\1', content)  # Italic: *text* -> text (not part of bold)
        content = re.sub(r'(?<!_)_([^_\n]+)_(?!_)', r'\1', content)      # Italic alt: _text_ -> text (not part of bold)
        content = re.sub(r'`([^`\n]+)`', r'\1', content)    # Inline code: `text` -> text
        
        # Clean up code blocks
        content = re.sub(r'^```[^\n]*\n', '', content, flags=re.MULTILINE)  # Opening code block
        content = re.sub(r'^```\s*$', '', content, flags=re.MULTILINE)      # Closing code block
        
        # Clean up links - keep just the text part
        content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # [text](url) -> text
        content = re.sub(r'<([^>]+)>', r'\1', content)              # <url> -> url
        
        # Convert lists to simple bullets
        content = re.sub(r'^\s*[-*+]\s+', '• ', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\d+\.\s+', '• ', content, flags=re.MULTILINE)
        
        # Clean up blockquotes
        content = re.sub(r'^>\s+', '', content, flags=re.MULTILINE)
        
        # Remove horizontal rules
        content = re.sub(r'^---+\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\*\*\*+\s*$', '', content, flags=re.MULTILINE)
        
        # Normalize whitespace - remove excessive newlines but preserve paragraph structure
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces/tabs to single space
        content = re.sub(r'^[ \t]+', '', content, flags=re.MULTILINE)  # Leading whitespace
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)  # Trailing whitespace
        
        return content.strip()

    def prepare_content_for_export(self, raw_content: str, content_type: str = "chat") -> str:
        """Prepare content for export based on type
        
        Args:
            raw_content: Raw content to prepare
            content_type: Type of content ('chat', 'blog', 'conversation', etc.)
            
        Returns:
            Prepared content string
        """
        if not raw_content:
            return "No content available"
        
        # Clean the content first
        clean_content = self.clean_markdown_for_docs(raw_content)
        
        # Add content-specific formatting
        if content_type == "blog":
            # Add blog-specific structure
            if not clean_content.startswith("BLOG POST"):
                clean_content = f"BLOG POST\n\n{clean_content}"
        elif content_type == "chat":
            # Add chat-specific structure if needed
            pass
        
        return clean_content

# =============================================================================
# SECTION 4: GOOGLE DOCS CREATION AND MANAGEMENT
# =============================================================================

    def create_google_doc(self, title: str, content: str, apply_formatting: bool = False) -> Dict[str, Any]:
        """Create a new Google Doc with content - SAFE VERSION
        
        Args:
            title: Document title
            content: Document content (can contain markdown)
            apply_formatting: Whether to apply formatting (disabled for stability)
            
        Returns:
            Dictionary with success status and document info
        """
        if not self.docs_service:
            return {
                'success': False,
                'error': 'Google Docs service not initialized. Check OAuth credentials.'
            }
        
        try:
            # Create the document with title
            document = {'title': title}
            doc = self.docs_service.documents().create(body=document).execute()
            document_id = doc.get('documentId')
            
            print(f"Created Google Doc: {title} (ID: {document_id})")
            
            # Clean content for safe insertion
            clean_content = self.clean_markdown_for_docs(content)
            
            if not clean_content:
                clean_content = "Document created successfully but no content was provided."
            
            # Insert the content at the beginning of the document
            insert_request = {
                'insertText': {
                    'location': {'index': 1},  # Insert at beginning (after document title)
                    'text': clean_content
                }
            }
            
            # Execute the text insertion
            self.docs_service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': [insert_request]}
            ).execute()
            
            print(f"Successfully inserted {len(clean_content)} characters of content")
            
            # Make document shareable (optional)
            try:
                if self.drive_service:
                    self.drive_service.permissions().create(
                        fileId=document_id,
                        body={'role': 'reader', 'type': 'anyone'}
                    ).execute()
                    print(f"Document made publicly readable")
            except Exception as e:
                print(f"Warning: Could not make document shareable: {e}")
            
            return {
                'success': True,
                'document_id': document_id,
                'document_url': f"https://docs.google.com/document/d/{document_id}",
                'title': title,
                'content_length': len(clean_content),
                'formatting_applied': False,
                'shareable': True
            }
            
        except HttpError as e:
            error_msg = f"Google Docs API error: {e}"
            print(f"HttpError in document creation: {error_msg}")
            return {'success': False, 'error': error_msg}
        except Exception as e:
            error_msg = f"Document creation failed: {str(e)}"
            print(f"General error in document creation: {error_msg}")
            return {'success': False, 'error': error_msg}
    
    def append_to_document(self, document_id: str, content: str) -> Dict[str, Any]:
        """Append content to an existing Google Doc
        
        Args:
            document_id: Google Docs document ID
            content: Content to append
            
        Returns:
            Operation result dictionary
        """
        if not self.docs_service:
            return {'success': False, 'error': 'Google Docs API not available'}
        
        try:
            # Get current document to find end index
            document = self.docs_service.documents().get(documentId=document_id).execute()
            content_list = document.get('body', {}).get('content', [])
            
            if not content_list:
                return {'success': False, 'error': 'Could not determine document structure'}
            
            # Find the end index (last element's endIndex)
            end_index = content_list[-1].get('endIndex', 1)
            
            # Clean content for insertion
            clean_content = self.clean_markdown_for_docs(content)
            
            # Insert content at the end
            append_request = {
                'insertText': {
                    'location': {'index': end_index - 1},  # Insert before the very last character
                    'text': f"\n\n{clean_content}"
                }
            }
            
            self.docs_service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': [append_request]}
            ).execute()
            
            return {
                'success': True,
                'message': f'Content appended to document',
                'document_url': f"https://docs.google.com/document/d/{document_id}",
                'content_length': len(clean_content)
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Append failed: {str(e)}"}

# =============================================================================
# SECTION 5: CHAT CONVERSATION EXPORT METHODS
# =============================================================================

    def export_chat_conversation(self, conversation_data: Dict, include_responses: bool = True) -> Dict[str, Any]:
        """Export a single chat conversation to Google Docs
        
        Args:
            conversation_data: Dictionary containing conversation details
            include_responses: Whether to include AI responses
            
        Returns:
            Export result dictionary
        """
        try:
            # Generate document title with timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            user_input_preview = str(conversation_data.get('user_input', 'Chat'))[:50]
            # Clean the preview for use in filename
            user_input_preview = re.sub(r'[^\w\s-]', '', user_input_preview).strip()
            title = f"Ghostline Chat - {user_input_preview} - {timestamp}"
            
            # Build content with clear structure
            content_lines = []
            content_lines.append(f"GHOSTLINE AI CHAT EXPORT")
            content_lines.append("")
            content_lines.append(f"Export Date: {timestamp}")
            content_lines.append(f"Project: {conversation_data.get('project', 'Unknown')}")
            content_lines.append("")
            content_lines.append("=" * 50)
            content_lines.append("")
            
            # Add user input
            user_input = conversation_data.get('user_input', 'No input recorded')
            content_lines.append("USER INPUT:")
            content_lines.append(user_input)
            content_lines.append("")
            
            # Add AI response if requested
            if include_responses and conversation_data.get('response_data'):
                content_lines.append("AI RESPONSE:")
                content_lines.append("")
                
                response_data = conversation_data['response_data']
                if isinstance(response_data, dict):
                    for voice, response in response_data.items():
                        if response and isinstance(response, str):
                            content_lines.append(f"{voice.upper()}:")
                            content_lines.append(response)
                            content_lines.append("")
                else:
                    content_lines.append(str(response_data))
                    content_lines.append("")
            
            content_lines.append("")
            content_lines.append("=" * 50)
            content_lines.append("Exported from Ghostline AI")
            
            content = "\n".join(content_lines)
            
            # Create the Google Doc
            result = self.create_google_doc(title, content)
            
            if result['success']:
                result['export_type'] = 'single_conversation'
                result['included_responses'] = include_responses
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Chat export failed: {str(e)}"
            }
    
    def export_bookmarked_conversations(self, bookmarks: List[Dict], include_responses: bool = True) -> Dict[str, Any]:
        """Export multiple bookmarked conversations to a single Google Doc
        
        Args:
            bookmarks: List of bookmark dictionaries with conversation data
            include_responses: Whether to include AI responses
            
        Returns:
            Export result dictionary
        """
        try:
            if not bookmarks:
                return {'success': False, 'error': 'No bookmarks provided for export'}
            
            # Generate document title
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            title = f"Ghostline Bookmarks Export - {timestamp}"
            
            # Build comprehensive content
            content_lines = []
            content_lines.append("GHOSTLINE AI BOOKMARKS EXPORT")
            content_lines.append("")
            content_lines.append(f"Export Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
            content_lines.append(f"Total Bookmarks: {len(bookmarks)}")
            content_lines.append("")
            content_lines.append("=" * 60)
            content_lines.append("")
            
            # Table of Contents
            content_lines.append("TABLE OF CONTENTS:")
            for i, bookmark in enumerate(bookmarks, 1):
                bookmark_title = bookmark.get('title', f'Bookmark {i}')
                content_lines.append(f"{i}. {bookmark_title}")
            content_lines.append("")
            content_lines.append("=" * 60)
            content_lines.append("")
            
            # Add each bookmarked conversation
            for i, bookmark in enumerate(bookmarks, 1):
                bookmark_title = bookmark.get('title', f'Bookmark {i}')
                content_lines.append(f"BOOKMARK {i}: {bookmark_title}")
                content_lines.append("")
                
                # Add metadata
                if bookmark.get('created_at'):
                    content_lines.append(f"Bookmarked: {bookmark['created_at']}")
                if bookmark.get('project'):
                    content_lines.append(f"Project: {bookmark['project']}")
                if bookmark.get('notes'):
                    content_lines.append(f"Notes: {bookmark['notes']}")
                content_lines.append("")
                
                # Add conversation content
                if bookmark.get('user_input'):
                    content_lines.append("User Input:")
                    content_lines.append(bookmark['user_input'])
                    content_lines.append("")
                
                if include_responses and bookmark.get('ai_response'):
                    content_lines.append("AI Response:")
                    content_lines.append(bookmark['ai_response'])
                    content_lines.append("")
                
                content_lines.append("-" * 40)
                content_lines.append("")
            
            content_lines.append(f"End of export - {len(bookmarks)} bookmarked conversations")
            
            content = "\n".join(content_lines)
            
            # Create the Google Doc
            result = self.create_google_doc(title, content)
            
            if result['success']:
                result['export_type'] = 'bookmarked_conversations'
                result['bookmark_count'] = len(bookmarks)
                result['included_responses'] = include_responses
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Bookmark export failed: {str(e)}"
            }

# =============================================================================
# SECTION 6: THREAD EXPORT METHODS
# =============================================================================

    def export_thread_conversations(self, thread_data: Dict, include_responses: bool = True) -> Dict[str, Any]:
        """Export an entire conversation thread to Google Docs
        
        Args:
            thread_data: Dictionary containing thread metadata and conversations
            include_responses: Whether to include AI responses
            
        Returns:
            Export result dictionary
        """
        try:
            thread_metadata = thread_data.get('metadata', {})
            conversations = thread_data.get('conversations', [])
            bookmarks = thread_data.get('bookmarks', [])
            
            if not conversations:
                return {'success': False, 'error': 'No conversations found in thread'}
            
            # Generate document title
            thread_title = thread_metadata.get('title', 'Untitled Thread')
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            title = f"{thread_title} - Full Thread Export - {timestamp}"
            
            # Build comprehensive content
            content_lines = []
            content_lines.append("GHOSTLINE AI THREAD EXPORT")
            content_lines.append("")
            content_lines.append(f"Thread Title: {thread_title}")
            content_lines.append(f"Project: {thread_metadata.get('project', 'Unknown')}")
            content_lines.append(f"Created: {thread_metadata.get('created_at', 'Unknown')}")
            content_lines.append(f"Total Messages: {len(conversations)}")
            content_lines.append(f"Bookmarks: {len(bookmarks)}")
            content_lines.append(f"Export Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
            content_lines.append("")
            content_lines.append("=" * 70)
            content_lines.append("")
            
            # Add bookmarks overview if any
            if bookmarks:
                content_lines.append("BOOKMARKS IN THIS THREAD:")
                for i, bookmark in enumerate(bookmarks, 1):
                    bookmark_title = bookmark.get('title', 'Untitled')
                    notes = f" - {bookmark['notes']}" if bookmark.get('notes') else ""
                    content_lines.append(f"{i}. {bookmark_title}{notes}")
                content_lines.append("")
                content_lines.append("=" * 70)
                content_lines.append("")
            
            # Add conversation thread
            content_lines.append("CONVERSATION THREAD:")
            content_lines.append("")
            
            for i, conversation in enumerate(conversations, 1):
                conv_time = conversation.get('created_at', 'Unknown time')
                content_lines.append(f"MESSAGE {i} - {conv_time}")
                content_lines.append("")
                
                # Check if this conversation is bookmarked
                is_bookmarked = any(
                    bookmark.get('chat_id') == conversation.get('id')
                    for bookmark in bookmarks
                )
                
                if is_bookmarked:
                    bookmark_info = next(
                        (b for b in bookmarks if b.get('chat_id') == conversation.get('id')),
                        {}
                    )
                    content_lines.append(f"[BOOKMARKED: {bookmark_info.get('title', 'Untitled')}]")
                    content_lines.append("")
                
                # User input
                user_input = conversation.get('user_input', 'No input recorded')
                content_lines.append("User:")
                content_lines.append(user_input)
                content_lines.append("")
                
                # AI response
                if include_responses and conversation.get('response_data'):
                    response_data = conversation['response_data']
                    if isinstance(response_data, dict):
                        for voice, response in response_data.items():
                            if response and isinstance(response, str):
                                content_lines.append(f"{voice}:")
                                content_lines.append(response)
                                content_lines.append("")
                    else:
                        content_lines.append("AI:")
                        content_lines.append(str(response_data))
                        content_lines.append("")
                
                content_lines.append("-" * 50)
                content_lines.append("")
            
            content_lines.append(f"End of thread export - {len(conversations)} messages")
            
            content = "\n".join(content_lines)
            
            # Create the Google Doc
            result = self.create_google_doc(title, content)
            
            if result['success']:
                result['export_type'] = 'full_thread'
                result['message_count'] = len(conversations)
                result['bookmark_count'] = len(bookmarks)
                result['included_responses'] = include_responses
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Thread export failed: {str(e)}"
            }

# =============================================================================
# SECTION 7: FACTORY FUNCTION AND UTILITIES
# =============================================================================

def get_google_drive_exporter():
    """Factory function to create GoogleDriveExporter with current credentials"""
    try:
        # Try to use the enhanced integration credentials
        from modules.enhanced_google_integration import GoogleIntegration
        google_integration = GoogleIntegration()
        
        if google_integration.credentials:
            return GoogleDriveExporter(google_integration.credentials)
        else:
            print("No Google credentials available for drive export")
            return None
            
    except ImportError:
        print("Enhanced Google integration not available")
        return None
    except Exception as e:
        print(f"Failed to create Google Drive exporter: {e}")
        return None

def test_export_functionality():
    """Test function to verify export functionality"""
    exporter = get_google_drive_exporter()
    if not exporter:
        return {"success": False, "error": "Could not initialize exporter"}
    
    # Test with simple content
    test_content = """# Test Blog Post

This is a **bold** statement with *italic* text.

## Section 2

Here's some `code` and a list:

- Item 1
- Item 2
- Item 3

### Conclusion

This is the end of the test."""
    
    result = exporter.create_google_doc("Test Export", test_content)
    return result
