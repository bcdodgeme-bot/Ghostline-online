# modules/google_drive_export.py - Enhanced Google Drive Export for Chat Content

import os
import datetime
import json
import re
from typing import Dict, List, Optional, Any
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class GoogleDriveExporter:
    """Enhanced Google Drive integration for exporting chat content"""
    
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
            self.docs_service = build('docs', 'v1', credentials=self.credentials)
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            print("Google Drive export services initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Google services: {e}")
            raise
    
    def markdown_to_google_docs_requests(self, markdown_content: str) -> List[Dict]:
        """Convert markdown content to Google Docs API requests
        
        Args:
            markdown_content: String content with markdown formatting
            
        Returns:
            List of Google Docs API requests to format the document
        """
        requests = []
        
        # Split content into lines for processing
        lines = markdown_content.split('\n')
        current_index = 1  # Start after the document title
        
        for line in lines:
            line_length = len(line)
            if line_length == 0:
                current_index += 1  # Account for newline
                continue
            
            # Process different markdown elements
            if line.startswith('# '):
                # H1 - Main heading
                requests.append({
                    'updateParagraphStyle': {
                        'range': {
                            'startIndex': current_index,
                            'endIndex': current_index + line_length
                        },
                        'paragraphStyle': {
                            'namedStyleType': 'HEADING_1'
                        },
                        'fields': 'namedStyleType'
                    }
                })
                
            elif line.startswith('## '):
                # H2 - Section heading
                requests.append({
                    'updateParagraphStyle': {
                        'range': {
                            'startIndex': current_index,
                            'endIndex': current_index + line_length
                        },
                        'paragraphStyle': {
                            'namedStyleType': 'HEADING_2'
                        },
                        'fields': 'namedStyleType'
                    }
                })
                
            elif line.startswith('### '):
                # H3 - Subsection heading
                requests.append({
                    'updateParagraphStyle': {
                        'range': {
                            'startIndex': current_index,
                            'endIndex': current_index + line_length
                        },
                        'paragraphStyle': {
                            'namedStyleType': 'HEADING_3'
                        },
                        'fields': 'namedStyleType'
                    }
                })
            
            # Process inline formatting within the line
            inline_requests = self._process_inline_formatting(line, current_index)
            requests.extend(inline_requests)
            
            # Handle code blocks
            if line.startswith('```'):
                requests.append({
                    'updateTextStyle': {
                        'range': {
                            'startIndex': current_index,
                            'endIndex': current_index + line_length
                        },
                        'textStyle': {
                            'fontFamily': 'Courier New',
                            'backgroundColor': {
                                'color': {
                                    'rgbColor': {
                                        'red': 0.95,
                                        'green': 0.95,
                                        'blue': 0.95
                                    }
                                }
                            }
                        },
                        'fields': 'fontFamily,backgroundColor'
                    }
                })
            
            current_index += line_length + 1  # +1 for newline
        
        return requests
    
    def _process_inline_formatting(self, text: str, start_index: int) -> List[Dict]:
        """Process bold, italic, and inline code formatting
        
        Args:
            text: Text line to process
            start_index: Starting character index in the document
            
        Returns:
            List of formatting requests
        """
        requests = []
        
        # Bold text (**text** or __text__)
        bold_pattern = r'\*\*(.*?)\*\*|__(.*?)__'
        for match in re.finditer(bold_pattern, text):
            content = match.group(1) or match.group(2)
            requests.append({
                'updateTextStyle': {
                    'range': {
                        'startIndex': start_index + match.start(),
                        'endIndex': start_index + match.end()
                    },
                    'textStyle': {
                        'bold': True
                    },
                    'fields': 'bold'
                }
            })
        
        # Italic text (*text* or _text_)
        italic_pattern = r'\*(.*?)\*|_(.*?)_'
        for match in re.finditer(italic_pattern, text):
            # Skip if it's part of a bold pattern
            if not any(bold_match.start() <= match.start() < bold_match.end() 
                      for bold_match in re.finditer(r'\*\*(.*?)\*\*|__(.*?)__', text)):
                content = match.group(1) or match.group(2)
                requests.append({
                    'updateTextStyle': {
                        'range': {
                            'startIndex': start_index + match.start(),
                            'endIndex': start_index + match.end()
                        },
                        'textStyle': {
                            'italic': True
                        },
                        'fields': 'italic'
                    }
                })
        
        # Inline code (`text`)
        code_pattern = r'`([^`]+)`'
        for match in re.finditer(code_pattern, text):
            requests.append({
                'updateTextStyle': {
                    'range': {
                        'startIndex': start_index + match.start(),
                        'endIndex': start_index + match.end()
                    },
                    'textStyle': {
                        'fontFamily': 'Courier New',
                        'backgroundColor': {
                            'color': {
                                'rgbColor': {
                                    'red': 0.95,
                                    'green': 0.95,
                                    'blue': 0.95
                                }
                            }
                        }
                    },
                    'fields': 'fontFamily,backgroundColor'
                }
            })
        
        return requests
    
    def clean_markdown_for_docs(self, markdown_content: str) -> str:
        """Clean markdown content for Google Docs insertion
        
        Args:
            markdown_content: Raw markdown content
            
        Returns:
            Cleaned content suitable for Google Docs
        """
        # Remove markdown syntax but keep the text
        content = markdown_content
        
        # Clean up headings - remove # symbols
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        
        # Clean up bold/italic markers but keep content
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Bold
        content = re.sub(r'__(.*?)__', r'\1', content)      # Bold alt
        content = re.sub(r'\*(.*?)\*', r'\1', content)      # Italic
        content = re.sub(r'_(.*?)_', r'\1', content)        # Italic alt
        content = re.sub(r'`([^`]+)`', r'\1', content)      # Inline code
        
        # Clean up code blocks
        content = re.sub(r'^```.*$', '', content, flags=re.MULTILINE)
        
        # Remove excessive newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def create_google_doc(self, title: str, content: str, apply_formatting: bool = True) -> Dict[str, Any]:
        """Create a new Google Doc with formatted content
        
        Args:
            title: Document title
            content: Document content (markdown formatted)
            apply_formatting: Whether to apply markdown formatting
            
        Returns:
            Dictionary with success status and document info
        """
        if not self.docs_service:
            return {
                'success': False,
                'error': 'Google Docs service not initialized'
            }
        
        try:
            # Create the document
            document = {
                'title': title
            }
            
            doc = self.docs_service.documents().create(body=document).execute()
            document_id = doc.get('documentId')
            
            print(f"Created Google Doc: {title} (ID: {document_id})")
            
            # Clean content for insertion
            clean_content = self.clean_markdown_for_docs(content)
            
            # Insert the content
            requests = [{
                'insertText': {
                    'location': {
                        'index': 1  # Insert at the beginning
                    },
                    'text': clean_content
                }
            }]
            
            # Apply formatting if requested
            if apply_formatting:
                formatting_requests = self.markdown_to_google_docs_requests(content)
                requests.extend(formatting_requests)
            
            # Execute all requests
            if requests:
                self.docs_service.documents().batchUpdate(
                    documentId=document_id,
                    body={'requests': requests}
                ).execute()
            
            # Make the document shareable (optional)
            try:
                self.drive_service.permissions().create(
                    fileId=document_id,
                    body={
                        'role': 'reader',
                        'type': 'anyone'
                    }
                ).execute()
            except Exception as e:
                print(f"Warning: Could not make document shareable: {e}")
            
            return {
                'success': True,
                'document_id': document_id,
                'document_url': f"https://docs.google.com/document/d/{document_id}",
                'title': title,
                'content_length': len(clean_content)
            }
            
        except HttpError as e:
            error_msg = f"Google Docs API error: {e}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Document creation failed: {str(e)}"
            print(error_msg)
            return {
                'success': False,
                'error': error_msg
            }
    
    def export_chat_conversation(self, conversation_data: Dict, include_responses: bool = True) -> Dict[str, Any]:
        """Export a single chat conversation to Google Docs
        
        Args:
            conversation_data: Dictionary containing conversation details
            include_responses: Whether to include AI responses
            
        Returns:
            Export result dictionary
        """
        try:
            # Generate document title
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            user_input_preview = conversation_data.get('user_input', 'Chat')[:50]
            title = f"Ghostline Chat - {user_input_preview} - {timestamp}"
            
            # Build markdown content
            content = f"# {title}\n\n"
            content += f"**Date:** {timestamp}\n"
            content += f"**Project:** {conversation_data.get('project', 'Unknown')}\n\n"
            
            content += f"## User Input\n{conversation_data.get('user_input', 'No input recorded')}\n\n"
            
            if include_responses and conversation_data.get('response_data'):
                content += f"## AI Response\n"
                
                response_data = conversation_data['response_data']
                if isinstance(response_data, dict):
                    for voice, response in response_data.items():
                        if response and isinstance(response, str):
                            content += f"**{voice}:** {response}\n\n"
                else:
                    content += f"{str(response_data)}\n\n"
            
            content += f"\n---\n*Exported from Ghostline AI*"
            
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
                return {
                    'success': False,
                    'error': 'No bookmarks provided for export'
                }
            
            # Generate document title
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            title = f"Ghostline Bookmarks Export - {timestamp}"
            
            # Build comprehensive markdown content
            content = f"# {title}\n\n"
            content += f"**Export Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            content += f"**Total Bookmarks:** {len(bookmarks)}\n\n"
            
            content += f"## Table of Contents\n"
            for i, bookmark in enumerate(bookmarks, 1):
                bookmark_title = bookmark.get('title', f'Bookmark {i}')
                content += f"{i}. [{bookmark_title}](#bookmark-{i})\n"
            content += "\n---\n\n"
            
            # Add each bookmarked conversation
            for i, bookmark in enumerate(bookmarks, 1):
                bookmark_title = bookmark.get('title', f'Bookmark {i}')
                content += f"## {i}. {bookmark_title} {{#bookmark-{i}}}\n\n"
                
                # Add metadata
                if bookmark.get('created_at'):
                    content += f"**Bookmarked:** {bookmark['created_at']}\n"
                if bookmark.get('project'):
                    content += f"**Project:** {bookmark['project']}\n"
                if bookmark.get('notes'):
                    content += f"**Notes:** {bookmark['notes']}\n"
                content += "\n"
                
                # Add conversation content
                if bookmark.get('user_input'):
                    content += f"### User Input\n```\n{bookmark['user_input']}\n```\n\n"
                
                if include_responses and bookmark.get('ai_response'):
                    content += f"### AI Response\n{bookmark['ai_response']}\n\n"
                
                content += "---\n\n"
            
            content += f"\n*Exported from Ghostline AI - {len(bookmarks)} bookmarked conversations*"
            
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
                return {
                    'success': False,
                    'error': 'No conversations found in thread'
                }
            
            # Generate document title
            thread_title = thread_metadata.get('title', 'Untitled Thread')
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
            title = f"{thread_title} - Full Thread Export - {timestamp}"
            
            # Build comprehensive markdown content
            content = f"# {title}\n\n"
            
            # Thread metadata
            content += f"**Thread Title:** {thread_title}\n"
            content += f"**Project:** {thread_metadata.get('project', 'Unknown')}\n"
            content += f"**Created:** {thread_metadata.get('created_at', 'Unknown')}\n"
            content += f"**Total Messages:** {len(conversations)}\n"
            content += f"**Bookmarks:** {len(bookmarks)}\n"
            content += f"**Export Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            
            # Add bookmarks overview if any
            if bookmarks:
                content += f"## Bookmarks in This Thread\n"
                for i, bookmark in enumerate(bookmarks, 1):
                    content += f"{i}. **{bookmark.get('title', 'Untitled')}**"
                    if bookmark.get('notes'):
                        content += f" - {bookmark['notes']}"
                    content += "\n"
                content += "\n---\n\n"
            
            # Add conversation thread
            content += f"## Conversation Thread\n\n"
            
            for i, conversation in enumerate(conversations, 1):
                conv_time = conversation.get('created_at', 'Unknown time')
                content += f"### Message {i} - {conv_time}\n\n"
                
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
                    content += f"📌 **BOOKMARKED:** {bookmark_info.get('title', 'Untitled')}\n\n"
                
                # User input
                user_input = conversation.get('user_input', 'No input recorded')
                content += f"**User:** {user_input}\n\n"
                
                # AI response
                if include_responses and conversation.get('response_data'):
                    response_data = conversation['response_data']
                    if isinstance(response_data, dict):
                        for voice, response in response_data.items():
                            if response and isinstance(response, str):
                                content += f"**{voice}:** {response}\n\n"
                    else:
                        content += f"**AI:** {str(response_data)}\n\n"
                
                content += "---\n\n"
            
            content += f"\n*Full thread export from Ghostline AI - {len(conversations)} messages*"
            
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