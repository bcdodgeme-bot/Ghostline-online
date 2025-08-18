#!/usr/bin/env python3
"""
Process ChatGPT conversations and organized knowledge into project-specific brain
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
    'Business': ['marketing', 'business', 'strategy', 'social media', 'seo', 'blog', 'content'],
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
    """Extract the actual conversation text from ChatGPT mapping structure"""
    content_parts = []
    
    def traverse_node(node_id: str, visited: set):
        if node_id in visited or node_id not in mapping:
            return
        visited.add(node_id)
        
        node = mapping[node_id]
        message = node.get('message')
        if message:
            author = message.get('author', {}).get('role', '')
            content = message.get('content', {})
            
            if isinstance(content, dict) and 'parts' in content:
                text = ' '.join(content['parts'])
                if text.strip():
                    content_parts.append(f"{author}: {text}")
            
            # Follow children
            children = node.get('children', [])
            for child_id in children:
                traverse_node(child_id, visited)
    
    # Start from root nodes
    for node_id in mapping:
        if mapping[node_id].get('parent') is None:
            traverse_node(node_id, set())
    
    return '\n\n'.join(content_parts)

def process_chatgpt_conversations(chatgpt_file: str) -> List[Dict]:
    """Process ChatGPT conversations.json file"""
    print(f"Processing ChatGPT conversations from {chatgpt_file}")
    
    with open(chatgpt_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed = []
    
    for conversation in data:
        title = conversation.get('title', 'Untitled')
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
            'conversation_id': conversation.get('conversation_id', ''),
            'create_time': conversation.get('create_time', 0),
            'content': content,
            'content_type': 'conversation',
            'sha1': hashlib.sha1(content.encode()).hexdigest()
        }
        
        processed.append(entry)
        print(f"  → {title} → {project}")
    
    return processed

def process_raw_folder(folder_path: Path, project: str) -> List[Dict]:
    """Process files in a raw data folder"""
    print(f"Processing {folder_path} → {project}")
    
    processed = []
    
    for file_path in folder_path.rglob("*.html"):
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            # Basic HTML to text (you could improve this)
            import re
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive newlines
            
            if len(text.strip()) < 100:  # Skip very short content
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
    
    # Process ChatGPT conversations
    chatgpt_file = Path('data/raw_chatgpt/conversations.json')
    if chatgpt_file.exists():
        chatgpt_entries = process_chatgpt_conversations(chatgpt_file)
        all_entries.extend(chatgpt_entries)
        print(f"✅ Processed {len(chatgpt_entries)} ChatGPT conversations")
    
    # Process organized raw folders
    data_dir = Path('data')
    for folder_name, project in RAW_FOLDER_MAPPING.items():
        if project is None:  # Skip special folders
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