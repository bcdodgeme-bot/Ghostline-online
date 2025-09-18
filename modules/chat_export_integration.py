# modules/chat_export_integration.py - Integration with existing Ghostline system

# Section 1: Imports and Core Setup
from typing import Dict, List, Optional, Any
import datetime
import re

# Section 2: Main Export Command Handler
def handle_export_command(user_input: str, project: str, use_voices: list, random_toggle: bool) -> tuple[Dict, bool]:
    """Handle export commands in the main chat flow
    
    Args:
        user_input: User's command input
        project: Current project context
        use_voices: Voice settings
        random_toggle: Random toggle setting
        
    Returns:
        Tuple of (response_data, handled_flag)
    """
    user_lower = user_input.lower().strip()
    
    # Export command patterns
    export_patterns = [
        'export to google docs', 'export to drive', 'create google doc',
        'save to google docs', 'copy to google docs', 'export conversation',
        'export bookmarks', 'export thread'
    ]
    
    # Check if this is an export command
    is_export_command = any(pattern in user_lower for pattern in export_patterns)
    
    if not is_export_command:
        return {}, False
    
    try:
        from modules.google_drive_export import get_google_drive_exporter
        from modules.database import get_bookmarks, get_db_connection
        
        # Get the Google Drive exporter
        exporter = get_google_drive_exporter()
        
        if not exporter:
            return {
                "SyntaxPrime": """Google Drive export is not available. This could be due to:
                
1. **Google OAuth not configured** - Set up Google integration first
2. **Missing Google Docs API permissions** - Re-authenticate with document creation scope
3. **Service initialization failed** - Check your Google API credentials

Visit `/integrations` to configure Google Drive access, then try the export command again."""
            }, True
        
        # Determine export type and execute
        if 'bookmark' in user_lower:
            return handle_bookmark_export(exporter, project, user_input)
        elif 'thread' in user_lower:
            return handle_thread_export(exporter, project, user_input)
        else:
            return handle_recent_conversation_export(exporter, project, user_input)
            
    except Exception as e:
        return {
            "SyntaxPrime": f"Export command failed: {str(e)}\n\nPlease ensure Google Drive integration is properly configured in `/integrations`."
        }, True

# Section 3: Title Extraction Utility Functions
def extract_document_title(user_input: str, default_prefix: str = "New Document") -> str:
    """Extract document title from user commands like 'create google doc test15'"""
    user_input = user_input.strip()
    
    # Patterns to extract title from various command formats
    title_patterns = [
        r'create google doc(?:ument)?\s+(.+)',
        r'export to (?:google )?docs?\s+(.+)',
        r'save (?:to|as) (?:google )?docs?\s+(.+)',
        r'copy to (?:google )?docs?\s+(.+)',
        r'export bookmarks?\s+(.+)',
        r'export thread\s+(.+)'
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            # Clean up the title - remove common trailing words
            cleanup_words = ['with responses', 'without responses', 'responses', 'ai', 'formatted']
            for cleanup in cleanup_words:
                if title.lower().endswith(' ' + cleanup):
                    title = title[:-len(' ' + cleanup)].strip()
            
            if title and len(title) > 0:
                return title
    
    # If no title found, generate default with timestamp
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    return f"{default_prefix} {timestamp}"

# Section 4: Bookmark Export Handler
def handle_bookmark_export(exporter, project: str, user_input: str) -> tuple[Dict, bool]:
    """Handle bookmark export commands"""
    try:
        from modules.database import get_bookmarks
        
        # Get recent bookmarks for this project
        bookmarks = get_bookmarks(project=project, limit=10)
        
        if not bookmarks:
            return {
                "SyntaxPrime": f"""No bookmarks found for project "{project}".
                
Create bookmarks first using:
- `bookmark this` - Mark current conversation
- `bookmark this as "Meeting Notes"` - Mark with custom title

Then try the export command again."""
            }, True
        
        # Extract custom title if provided
        document_title = extract_document_title(user_input, f"{project} Bookmarks")
        
        # Convert bookmarks to export format
        export_bookmarks = []
        for bookmark in bookmarks:
            export_bookmarks.append({
                'title': bookmark.get('title', 'Untitled'),
                'notes': bookmark.get('notes', ''),
                'created_at': bookmark.get('created_at', '').strftime('%Y-%m-%d %H:%M') if bookmark.get('created_at') else 'Unknown',
                'project': project,
                'user_input': bookmark.get('conversation_preview', 'No preview available'),
                'ai_response': bookmark.get('ai_response_preview', '') if 'include responses' in user_input.lower() else ''
            })
        
        # Export to Google Docs with extracted title
        include_responses = 'responses' in user_input.lower() or 'ai' in user_input.lower()
        result = exporter.export_bookmarked_conversations(
            export_bookmarks,
            include_responses,
            document_title=document_title
        )
        
        if result['success']:
            return {
                "SyntaxPrime": f"""✅ **Bookmark Export Successful!**

**Document Created:** {result['document_title']}
**Google Docs URL:** {result['document_url']}
**Bookmarks Exported:** {result['bookmark_count']}
**Responses Included:** {'Yes' if result['included_responses'] else 'No'}

Your bookmarked conversations have been exported to a formatted Google Doc. The document is now available in your Google Drive and can be shared or edited as needed."""
            }, True
        else:
            return {
                "SyntaxPrime": f"""❌ **Bookmark Export Failed**

Error: {result.get('error', 'Unknown error')}

This could be due to:
- Google API rate limits or permissions
- Network connectivity issues
- Document formatting problems

Try again in a few minutes or check your Google integration status."""
            }, True
            
    except Exception as e:
        return {
            "SyntaxPrime": f"Bookmark export error: {str(e)}"
        }, True

# Section 5: Thread Export Handler
def handle_thread_export(exporter, project: str, user_input: str) -> tuple[Dict, bool]:
    """Handle conversation thread export commands"""
    try:
        from modules.database import load_conversation_enhanced
        
        # Get recent conversations for this project
        conversations = load_conversation_enhanced(project, limit=20)
        
        if not conversations:
            return {
                "SyntaxPrime": f"No conversation history found for project '{project}' to export as a thread."
            }, True
        
        # Extract custom title if provided
        document_title = extract_document_title(user_input, f"{project} Conversation Thread")
        
        # Convert to thread format
        thread_data = {
            'metadata': {
                'title': document_title,
                'project': project,
                'created_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                'message_count': len(conversations)
            },
            'conversations': [
                {
                    'id': conv.get('id'),
                    'user_input': conv.get('user_input', ''),
                    'response_data': conv.get('response_data', {}),
                    'created_at': conv.get('created_at', '').strftime('%Y-%m-%d %H:%M') if conv.get('created_at') else 'Unknown'
                }
                for conv in conversations
            ],
            'bookmarks': []  # Could be populated from bookmark system
        }
        
        # Export to Google Docs with extracted title
        include_responses = 'responses' in user_input.lower() or 'ai' in user_input.lower()
        result = exporter.export_thread_conversations(
            thread_data,
            include_responses,
            document_title=document_title
        )
        
        if result['success']:
            return {
                "SyntaxPrime": f"""✅ **Thread Export Successful!**

**Document Created:** {result['document_title']}
**Google Docs URL:** {result['document_url']}
**Messages Exported:** {result['message_count']}
**Responses Included:** {'Yes' if result['included_responses'] else 'No'}

Your conversation thread has been exported to a formatted Google Doc with full conversation history."""
            }, True
        else:
            return {
                "SyntaxPrime": f"""❌ **Thread Export Failed**

Error: {result.get('error', 'Unknown error')}

Check your Google Drive integration and try again."""
            }, True
            
    except Exception as e:
        return {
            "SyntaxPrime": f"Thread export error: {str(e)}"
        }, True

# Section 6: Single Conversation Export Handler (MAIN FIX)
def handle_recent_conversation_export(exporter, project: str, user_input: str) -> tuple[Dict, bool]:
    """Handle single conversation export commands - FIXED VERSION"""
    try:
        from modules.database import load_conversation_enhanced
        
        # Get the most recent conversation
        conversations = load_conversation_enhanced(project, limit=1)
        
        if not conversations:
            return {
                "SyntaxPrime": "No recent conversation found to export."
            }, True
        
        recent_conversation = conversations[0]
        
        # CRITICAL FIX: Extract custom title from user input
        document_title = extract_document_title(user_input, f"{project} Conversation")
        
        # Export to Google Docs with extracted title
        include_responses = 'responses' in user_input.lower() or 'ai' in user_input.lower()
        
        # FIXED: Pass the extracted title as document_title parameter
        result = exporter.export_chat_conversation(
            recent_conversation,
            include_responses,
            document_title=document_title  # This was missing before!
        )
        
        if result['success']:
            return {
                "SyntaxPrime": f"""✅ **Conversation Export Successful!**

**Document Created:** {result['document_title']}
**Google Docs URL:** {result['document_url']}
**Content Length:** {result.get('content_length', 0)} characters
**Responses Included:** {'Yes' if result['included_responses'] else 'No'}

Your recent conversation has been exported to Google Docs and is ready for sharing or editing."""
            }, True
        else:
            return {
                "SyntaxPrime": f"""❌ **Conversation Export Failed**

Error: {result.get('error', 'Unknown error')}

Check your Google Drive integration and try again."""
            }, True
            
    except Exception as e:
        return {
            "SyntaxPrime": f"Conversation export error: {str(e)}"
        }, True

# Section 7: Bookmark Enhancement Integration
def enhance_bookmark_command(user_input: str, project: str) -> tuple[Dict, bool]:
    """Enhanced bookmark command that offers export option
    
    This can be integrated into your existing bookmark handler
    """
    user_lower = user_input.lower().strip()
    
    # Check for bookmark creation with export intent
    export_bookmark_patterns = [
        'bookmark and export', 'bookmark for export',
        'save and export', 'bookmark to drive'
    ]
    
    if any(pattern in user_lower for pattern in export_bookmark_patterns):
        return {
            "SyntaxPrime": """📌 **Bookmark Created with Export Intent**

Your conversation has been bookmarked. To export your bookmarks to Google Docs, use:

- `export bookmarks` - Export all bookmarks for this project
- `export bookmarks with responses` - Include AI responses in export
- `export to google docs` - Export current conversation

The export will create a formatted Google Doc with all your bookmarked conversations."""
        }, True
    
    return {}, False

# Section 8: Help and Documentation
def get_export_help() -> str:
    """Return help text for export commands"""
    return """📄 **Google Drive Export Commands**

**Single Conversation:**
- `export to google docs` - Export current conversation
- `create google doc MyTitle` - Export with custom title
- `copy to google docs ProjectNotes` - Export with custom name

**Bookmark Export:**
- `export bookmarks` - Export all project bookmarks
- `export bookmarks MyBookmarks` - Export with custom title
- `export bookmarks with responses` - Include AI responses

**Thread Export:**
- `export thread` - Export conversation history
- `export thread MyThread` - Export with custom title
- `export conversation thread` - Full project history

**Title Examples:**
- `create google doc test15` - Creates "test15"
- `export bookmarks Meeting Notes` - Creates "Meeting Notes"
- `export thread Project Summary` - Creates "Project Summary"

**Options:**
- Add "with responses" to include AI replies
- Add custom titles after the command
- Titles are automatically cleaned of command words

**Requirements:**
- Google OAuth configured (`/integrations`)
- Google Docs API access enabled
- Valid authentication token

**Command Processing:**
When you type "create google doc test15", the system:
1. Detects it's an export command
2. Extracts "test15" as the document title
3. Gets your recent conversation
4. Creates a Google Doc named "test15"
5. Returns the document URL

Example: `create google doc test15`"""
