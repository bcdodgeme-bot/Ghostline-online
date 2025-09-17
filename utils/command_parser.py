# utils/command_parser.py
# Command parsing for bookmark and Google Docs export functionality

import re
import datetime
from typing import Dict, Tuple, Optional

def detect_bookmark_command(user_input: str) -> bool:
    """Detect if user wants to create a bookmark"""
    bookmark_patterns = [
        r'\bbookmark\b',
        r'\bsave this\b',
        r'\bmark this\b',
        r'\bremember this\b',
        r'\bsave conversation\b'
    ]
    
    user_lower = user_input.lower().strip()
    return any(re.search(pattern, user_lower) for pattern in bookmark_patterns)

def detect_export_command(user_input: str) -> bool:
    """Detect if user wants to export to Google Docs"""
    export_patterns = [
        r'\bcopy to google docs?\b',
        r'\bexport to google\b',
        r'\bgoogle docs?\b',
        r'\bsend to drive\b',
        r'\bexport conversation\b',
        r'\bcreate doc\b'
    ]
    
    user_lower = user_input.lower().strip()
    return any(re.search(pattern, user_lower) for pattern in export_patterns)

def extract_bookmark_title(user_input: str) -> Optional[str]:
    """Extract a custom title from bookmark command"""
    user_input = user_input.strip()
    
    # Look for patterns like "bookmark this as X" or "bookmark: title here"
    title_patterns = [
        r'bookmark(?:\s+this)?\s+as\s+(.+)',
        r'bookmark:\s*(.+)',
        r'save this as\s+(.+)',
        r'mark this as\s+(.+)'
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Clean up the title
            title = re.sub(r'[^\w\s\-\(\)]+', '', title)  # Remove special chars
            return title[:100]  # Limit length
    
    return None

def process_bookmark_command(user_input: str, project: str, last_response_data: Dict = None) -> Tuple[Dict[str, str], bool]:
    """Process bookmark creation command"""
    try:
        from modules.database import get_db_connection, create_bookmark
        
        # Extract custom title if provided
        custom_title = extract_bookmark_title(user_input)
        
        # Generate default title if none provided
        if not custom_title:
            timestamp = datetime.datetime.now().strftime("%m/%d %H:%M")
            custom_title = f"Bookmark - {project} - {timestamp}"
        
        # For now, we'll create a placeholder bookmark since we need the conversation to be saved first
        # This would normally be called after save_conversation_enhanced returns a chat_id
        
        response_data = {
            "SyntaxPrime": f"📑 Bookmark created: \"{custom_title}\"\n\nThis conversation point has been saved for easy reference. Use 'copy to google docs' to export it later."
        }
        
        return response_data, True
        
    except Exception as e:
        response_data = {
            "SyntaxPrime": f"Failed to create bookmark: {str(e)}\n\nThe conversation is still saved in your history, but the bookmark wasn't created."
        }
        return response_data, True

def process_export_command(user_input: str, project: str) -> Tuple[Dict[str, str], bool]:
    """Process Google Docs export command"""
    try:
        # Check if Google integration is available
        try:
            from modules.enhanced_google_integration import EnhancedGoogleIntegration
            google_integration = EnhancedGoogleIntegration()
            
            if not google_integration.is_configured():
                response_data = {
                    "SyntaxPrime": "Google Docs export requires Google OAuth setup. Visit /integrations to configure Google Drive access first."
                }
                return response_data, True
                
        except ImportError:
            response_data = {
                "SyntaxPrime": "Google Docs integration not available. The enhanced Google integration module needs to be configured."
            }
            return response_data, True
        
        # Get recent bookmarks for this project
        from modules.database import get_bookmarks
        bookmarks = get_bookmarks(project=project, limit=5)
        
        if not bookmarks:
            response_data = {
                "SyntaxPrime": "No bookmarks found to export. Create a bookmark first using 'bookmark this' or 'save this'."
            }
            return response_data, True
        
        # Use the most recent bookmark
        latest_bookmark = bookmarks[0]
        
        # Export to Google Docs
        doc_title = f"Ghostline Export - {latest_bookmark['title']} - {datetime.datetime.now().strftime('%Y-%m-%d')}"
        
        # Get the conversation content
        from modules.database import get_db_connection
        
        with get_db_connection() as conn:
            if not conn:
                response_data = {
                    "SyntaxPrime": "Database connection failed. Cannot retrieve conversation for export."
                }
                return response_data, True
            
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_input, response_data, created_at 
                FROM chat_threads 
                WHERE id = %s
            ''', (latest_bookmark['chat_id'],))
            
            conversation = cursor.fetchone()
            
            if not conversation:
                response_data = {
                    "SyntaxPrime": "Conversation not found. The bookmark may reference a conversation that was deleted."
                }
                return response_data, True
            
            # Format content for Google Docs
            user_input, response_data_json, created_at = conversation
            ai_response = response_data_json.get('SyntaxPrime', '') if response_data_json else ''
            
            document_content = f"""# {doc_title}

**Project:** {project}
**Date:** {created_at.strftime('%Y-%m-%d %H:%M')}
**Bookmark:** {latest_bookmark['title']}

## User Input
{user_input}

## AI Response
{ai_response}

---
*Exported from Ghostline AI*
"""
            
            # Create Google Doc
            doc_result = google_integration.create_google_doc(
                title=doc_title,
                content=document_content
            )
            
            if doc_result.get('success'):
                response_data = {
                    "SyntaxPrime": f"📄 Successfully exported to Google Docs!\n\n**Document:** {doc_title}\n**URL:** {doc_result['document_url']}\n\nThe bookmark \"{latest_bookmark['title']}\" has been exported with full conversation context."
                }
            else:
                response_data = {
                    "SyntaxPrime": f"Google Docs export failed: {doc_result.get('error', 'Unknown error')}\n\nCheck your Google integration setup in /integrations."
                }
        
        return response_data, True
        
    except Exception as e:
        response_data = {
            "SyntaxPrime": f"Export failed: {str(e)}\n\nTry checking your Google integration setup or create a bookmark first."
        }
        return response_data, True

# Integration with existing generate_response function
def process_chat_commands(user_input: str, project: str, last_response_data: Dict = None) -> Tuple[Optional[Dict[str, str]], bool]:
    """
    Process chat commands before normal AI response generation
    
    Returns:
        (response_data, handled) - response_data if command was handled, None otherwise
    """
    
    # Check for bookmark command
    if detect_bookmark_command(user_input):
        return process_bookmark_command(user_input, project, last_response_data)
    
    # Check for export command
    if detect_export_command(user_input):
        return process_export_command(user_input, project)
    
    # No commands detected
    return None, False


# Modified generate_response function integration
def generate_response_with_commands(
    user_input: str,
    use_voices: list,
    random_toggle: bool,
    project: str = "default",
    model: str = None,
    retrieval_context: list = None,
    last_response_data: Dict = None,
    **kwargs
) -> Dict[str, str]:
    """
    Enhanced generate_response with command parsing
    
    This function should replace or be called by your existing generate_response
    """
    
    # First, check for commands
    command_response, command_handled = process_chat_commands(
        user_input, project, last_response_data
    )
    
    if command_handled:
        # Command was processed, return command response
        return command_response
    
    # No command detected, proceed with normal AI response
    # Import your existing generate_response function
    from utils.ghostline_engine import generate_response
    
    return generate_response(
        user_input=user_input,
        use_voices=use_voices,
        random_toggle=random_toggle,
        project=project,
        model=model,
        retrieval_context=retrieval_context,
        **kwargs
    )


# Helper function to create bookmark after conversation is saved
def create_bookmark_after_save(chat_id: int, title: str, project: str) -> bool:
    """
    Create bookmark after conversation is saved to database
    This should be called after save_conversation_enhanced returns a chat_id
    """
    try:
        from modules.database import create_bookmark
        
        bookmark_id = create_bookmark(
            chat_id=chat_id,
            title=title,
            notes=f"Auto-created bookmark for {project}",
            bookmark_type='user_command'
        )
        
        return bookmark_id is not None
        
    except Exception as e:
        print(f"Failed to create bookmark after save: {e}")
        return False