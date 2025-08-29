# modules/utils.py
# Final utility module for The Refactory project
# Extracted from Section 5: Utility Functions

import os
import json
import datetime
from flask import render_template
from modules.database import load_conversation_enhanced

def load_conversation(project: str, limit: int = 50):
    """Load conversation history for a project from file-based storage
    
    This function provides file-based fallback for conversation loading.
    The enhanced database version is preferred when available.
    """
    path = f"sessions/{project.lower().replace(' ', '_')}.json"
    if not os.path.exists(path):
        return []
    
    turns = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        
        for line in lines[-limit:]:
            try:
                row = json.loads(line)
                turns.append({
                    "user": row.get("prompt", ""), 
                    "responses": row.get("response", {})
                })
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        print(f"Error loading conversation for {project}: {e}")
        return []
    
    return turns

def _append_session(project: str, user_input: str, response_data: dict):
    """Append conversation to session file for backup storage
    
    This provides file-based backup alongside database storage.
    Used as fallback when database operations fail.
    """
    try:
        # Ensure sessions directory exists
        os.makedirs("sessions", exist_ok=True)
        
        path = f"sessions/{project.lower().replace(' ', '_')}.json"
        
        with open(path, 'a', encoding='utf-8') as f:
            json.dump({
                'prompt': user_input, 
                'response': response_data,
                'timestamp': datetime.datetime.now().isoformat()
            }, f)
            f.write('\n')
            
    except Exception as e:
        print(f"Warning: Failed to save session backup for {project}: {e}")

def _save_daily_log(sync_type: str, content: str):
    """Save daily sync results to log file
    
    Creates markdown-formatted daily logs for morning/evening briefings
    and other automated sync operations.
    """
    try:
        # Ensure daily_logs directory exists
        os.makedirs("daily_logs", exist_ok=True)
        
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_path = f"daily_logs/{today}.md"
        
        timestamp = datetime.datetime.now().strftime("%I:%M %p")
        
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n## {sync_type.title()} Sync - {timestamp}\n\n")
            f.write(content)
            f.write("\n\n---\n")
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"DEBUG: Daily log save failed: {error_details}")

def _render_enhanced(project: str, response_data: dict):
    """Render the main template with enhanced conversation data
    
    This function centralizes template rendering logic and handles
    both database and file-based conversation loading with graceful fallback.
    """
    try:
        # Try enhanced database loading first
        conversation = load_conversation_enhanced(project, limit=50)
    except Exception as e:
        print(f"Database conversation loading failed for {project}: {e}")
        # Fallback to file-based loading
        conversation = load_conversation(project, limit=50)
    
    # Get projects list (imported from main app context)
    from app import PROJECTS
    
    return render_template(
        'index.html',
        projects=PROJECTS,
        response_data=response_data,
        conversation=conversation,
        current_project=project
    )

def ensure_directories():
    """Ensure all required directories exist
    
    Creates necessary directories for file-based operations
    that serve as backups to database storage.
    """
    directories = [
        "sessions",
        "daily_logs", 
        "uploads",
        "data",
        "data/cleaned"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {directory}: {e}")

def format_timestamp(dt: datetime.datetime = None) -> str:
    """Format timestamp for consistent logging across the application"""
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y-%m-%d %I:%M:%S %p")

def safe_filename(filename: str) -> str:
    """Convert filename to safe format for file operations"""
    import re
    # Remove or replace unsafe characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove multiple consecutive underscores
    safe = re.sub(r'_+', '_', safe)
    # Remove leading/trailing underscores and dots
    safe = safe.strip('_.')
    return safe or 'unnamed_file'

def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase"""
    return os.path.splitext(filename.lower())[1]

def is_supported_file_type(filename: str) -> bool:
    """Check if file type is supported for processing"""
    supported_extensions = {
        '.txt', '.md', '.pdf', '.docx', '.doc',
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'
    }
    return get_file_extension(filename) in supported_extensions

# Initialize directories when module is imported
ensure_directories()