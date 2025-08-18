#!/usr/bin/env python3
"""
Fixed ChatGPT conversation processor
"""
import json
import gzip
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Project mapping based on keywords
PROJECT_MAPPING = {
    'AMCF': ['amcf', 'giving circle', 'nonprofit', 'fundraising', 'charity'],
    'Business': ['marketing', 'business', 'strategy', 'social media', 'seo', 'blog', 'content', 'email'],
    'Personal Development': ['productivity', 'habits', 'goals', 'mindset', 'personal growth'],
    'Health': ['health', 'medical', 'fitness', 'nutrition', 'wellness', 'exercise'],
    'Kitchen': ['cooking', 'recipe', 'food', 'kitchen', 'meal'],
    'Personal Operating Manual': ['workflow', 'system', 'process', 'automation', 'efficiency'],
    'General': []  # Fallback for everything else
}

# Raw data folder mapping
RAW_FOLDER_MAPPING = {
    'raw_business': 'Business',
    'raw_personal_dev': 'Personal Development', 
    'raw_health': 'Health',
    'raw_personal': 'Personal Operating Manual',
    'raw_chatgpt': None  # Special processing
}

def classify_conversation(title: str, content: str) -> str:
    """Classify conversation into project based on title and content"""
    title_lower = title.lower()
    content_lower = content.lower()
    
    for project, keywords in PROJECT_MAPPING.items():
        for keyword in keywords:
            if keyword in title_lower or keyword in content_lower:
                return project
    
    return 'General'

def extract_conversation_content(mapping: Dict) -> str:
    """Extract conversation text from ChatGPT mapping - FIXED VERSION"""
    messages = []
    
    # Sort nodes by create_time to get chronological order
    nodes_with_time = []
    for node_id, node in mapping.items():
        message = node.get('message')
        if message and message.get('content'):
            create_time = message.get('create_time', 0)
            # Handle None create_time values
            if create_time is None:
                create_time = 0
            nodes_with_time.append((create_time, message))
    
    # Sort by time (None-safe)
    nodes_with_time.sort(key=lambda x: x[0] or 0)
    
    # Extract messages in order
    for create_time, message in nodes_with_time:
        author = message.get('author', {}).get('role', 'unknown')
        content = message.get('content', {})
        
        # Handle different content structures
        text = ""
        if isinstance(content, dict):
            if 'parts' in content and content['parts']:
                # Standard text content
                text = ' '.join(str(part) for part in content['parts'] if part)
            elif 'text' in content:
                text = content['text']
        elif isinstance(content, str):
            text = content
        
        if text.strip():
            # Clean up the text
            text = text.strip()
            # Limit very long messages
            if len(text) > 2000:
                text = text[:2000] + "..."
            messages.append(f"{author.title()}: {text}")
    
    return '\n\n'.join(messages)

def process_chatgpt_conversations(chatgpt_file: str) -> List[Dict]:
    """Process ChatGPT conversations.json file"""
    print(f"Processing ChatGPT conversations from {chatgpt_file}")
    
    try:
        with open(chatgpt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} conversations from file")
    except Exception as e:
        print(f"Error loading conversations: {e}")
        return []
    
    processed = []
    
    for i, conversation in enumerate(data):
        try:
            title = conversation.get('title', f'Conversation {i+1}')
            mapping = conversation.get('mapping', {})
            
            if not mapping:
                continue
                
            content = extract_conversation_content(mapping)
            if not content.strip():
                continue
                
            project = classify_conversation(title, content)
            
            # Create knowledge entry
            entry = {
                'source': 'ChatGPT Conversation',
                'title': title,
                'project': project,
                'conversation_id': conversation.get('conversation_id', f'conv_{i}'),
                'create_time': conversation.get('create_time', 0),
                'content': content,
                'content_type': 'conversation',
                'sha1': hashlib.sha1(content.encode()).hexdigest()
            }
            
            processed.append(entry)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(data)} conversations...")
            
        except Exception as e:
            print(f"  Error processing conversation {i}: {e}")
            continue
    
    print(f"Successfully processed {len(processed)} conversations")
    return processed

def process_raw_folder(folder_path: Path, project: str) -> List[Dict]:
    """Process files in a raw data folder"""
    print(f"Processing {folder_path} → {project}")
    
    processed = []
    
    for file_path in folder_path.rglob("*.html"):
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Basic HTML to text
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            if len(text.strip()) < 100:
                continue
                
            entry = {
                'source': f'Raw Data - {folder_path.name}',
                'title': file_path.stem.replace('_', ' ').title(),
                'project': project,
                'filename': file_path.name,
                'content': text,
                'content_type': 'article',
                'sha1': hashlib.sha1(text.encode()).hexdigest()
            }
            
            processed.append(entry)
            
        except Exception as e:
            print(f"  ⚠️  Error processing {file_path}: {e}")
            continue
    
    return processed

def build_new_brain():
    """Build new project-aware knowledge base"""
    print("🧠 Building new Ghostline brain...")
    
    all_entries = []
    
    # Process ChatGPT conversations with improved parser
    chatgpt_file = Path('data/raw_chatgpt/conversations.json')
    if chatgpt_file.exists():
        chatgpt_entries = process_chatgpt_conversations(chatgpt_file)
        all_entries.extend(chatgpt_entries)
        print(f"✅ Processed {len(chatgpt_entries)} ChatGPT conversations")
    
    # Process organized raw folders
    data_dir = Path('data')
    for folder_name, project in RAW_FOLDER_MAPPING.items():
        if project is None:
            continue
            
        folder_path = data_dir / folder_name
        if folder_path.exists():
            entries = process_raw_folder(folder_path, project)
            all_entries.extend(entries)
            print(f"✅ Processed {len(entries)} entries from {folder_name}")
    
    # Save new brain
    output_path = Path('data/cleaned/ghostline_sources_new.jsonl.gz')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with gzip.open(output_path, 'wt', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Project breakdown
    project_counts = {}
    for entry in all_entries:
        proj = entry['project']
        project_counts[proj] = project_counts.get(proj, 0) + 1
    
    print(f"\n🎉 New brain created: {output_path}")
    print(f"📊 Total entries: {len(all_entries)}")
    print("📂 Project breakdown:")
    for project, count in sorted(project_counts.items()):
        print(f"   {project}: {count} entries")
    
    return output_path

if __name__ == '__main__':
    build_new_brain()