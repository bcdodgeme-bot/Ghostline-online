# modules/database_fixes.py
# Fixed database functions that actually use the new search indexes

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import List, Dict, Any
import datetime

DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

@contextmanager
def get_reliable_db_connection():
    """Reliable database connection with retry logic"""
    conn = None
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            if DATABASE_URL:
                conn = psycopg2.connect(
                    DATABASE_URL,
                    connect_timeout=10,
                    keepalives=1,
                    keepalives_idle=30,
                    keepalives_interval=5,
                    keepalives_count=3
                )
                yield conn
                break
            else:
                print("No DATABASE_URL configured")
                yield None
                break
        except Exception as e:
            retry_count += 1
            print(f"Database connection attempt {retry_count} failed: {e}")
            if conn:
                try:
                    conn.close()
                except:
                    pass
            conn = None
            
            if retry_count >= max_retries:
                print("Max retries exceeded, yielding None")
                yield None
            else:
                import time
                time.sleep(1)  # Wait before retry
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def search_conversations_with_fts(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search using the new full-text search indexes"""
    
    with get_reliable_db_connection() as conn:
        if not conn:
            print("No database connection for FTS search")
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Use the new full-text search index
            search_sql = '''
                SELECT id, project, user_input, response_data, created_at,
                       ts_rank(to_tsvector('english', 
                               user_input || ' ' || 
                               COALESCE(response_data->>'SyntaxPrime', '') || ' ' ||
                               COALESCE(project, '')), 
                               plainto_tsquery('english', %s)) as rank
                FROM chat_threads 
                WHERE to_tsvector('english', 
                      user_input || ' ' || 
                      COALESCE(response_data->>'SyntaxPrime', '') || ' ' ||
                      COALESCE(project, '')) 
                      @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC, created_at DESC
                LIMIT %s
            '''
            
            cursor.execute(search_sql, (query_text, query_text, k * 2))
            rows = cursor.fetchall()
            
            print(f"FTS search for '{query_text}' found {len(rows)} results")
            
            # Convert to RAG format
            results = []
            for row in rows:
                response_content = row['response_data'].get('SyntaxPrime', '') if row['response_data'] else ''
                combined_text = f"User: {row['user_input']}\nResponse: {response_content}"
                
                results.append({
                    'text': combined_text[:1500],
                    'source': f"Conversation - {row['project']} ({row['created_at'].strftime('%m/%d/%Y')})",
                    'id': f"conversation_{row['id']}",
                    'score': float(row['rank']),
                    'metadata': {
                        'type': 'conversation',
                        'project': row['project'],
                        'date': row['created_at'].isoformat(),
                        'chat_id': row['id']
                    }
                })
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            print(f"FTS search failed: {e}")
            return []

def search_personal_context(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search personal context using family names and personal keywords"""
    
    # Personal context keywords
    personal_keywords = ['miller', 'ghada', 'shazeen', 'mom', 'family', 'daughter', 'wife', 'cat']
    
    # Check if query contains personal keywords
    query_lower = query_text.lower()
    is_personal = any(keyword in query_lower for keyword in personal_keywords)
    
    if not is_personal:
        return []
    
    with get_reliable_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search recent personal conversations first
            personal_sql = '''
                SELECT id, project, user_input, response_data, created_at,
                       ts_rank(to_tsvector('english', 
                               LOWER(user_input) || ' ' || 
                               LOWER(COALESCE(response_data->>'SyntaxPrime', ''))), 
                               plainto_tsquery('english', %s)) as rank
                FROM chat_threads 
                WHERE (LOWER(user_input) LIKE ANY(%s) 
                       OR LOWER(response_data->>'SyntaxPrime') LIKE ANY(%s))
                   AND to_tsvector('english', 
                       LOWER(user_input) || ' ' || 
                       LOWER(COALESCE(response_data->>'SyntaxPrime', ''))) 
                       @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC, created_at DESC
                LIMIT %s
            '''
            
            like_patterns = [f'%{kw}%' for kw in personal_keywords]
            
            cursor.execute(personal_sql, (
                query_text, like_patterns, like_patterns, query_text, k * 2
            ))
            rows = cursor.fetchall()
            
            print(f"Personal context search for '{query_text}' found {len(rows)} results")
            
            # Convert to RAG format - prioritize these results
            results = []
            for row in rows:
                response_content = row['response_data'].get('SyntaxPrime', '') if row['response_data'] else ''
                combined_text = f"User: {row['user_input']}\nResponse: {response_content}"
                
                results.append({
                    'text': combined_text[:1500],
                    'source': f"Personal Context - {row['project']} ({row['created_at'].strftime('%m/%d/%Y')})",
                    'id': f"personal_{row['id']}",
                    'score': float(row['rank']) + 1.0,  # Boost personal context
                    'metadata': {
                        'type': 'personal_context',
                        'project': row['project'],
                        'date': row['created_at'].isoformat(),
                        'chat_id': row['id'],
                        'priority': 'high'
                    }
                })
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            print(f"Personal context search failed: {e}")
            return []

def enhanced_context_search(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """Enhanced search that combines personal context and general FTS"""
    
    print(f"Enhanced search for: '{query_text}'")
    
    # Try personal context first
    personal_results = search_personal_context(query_text, k=3)
    
    # Then get general results
    general_results = search_conversations_with_fts(query_text, k=k)
    
    # Combine and deduplicate
    all_results = personal_results + general_results
    seen_ids = set()
    unique_results = []
    
    for result in all_results:
        result_id = result['id']
        if result_id not in seen_ids:
            unique_results.append(result)
            seen_ids.add(result_id)
        
        if len(unique_results) >= k:
            break
    
    # Sort by score (personal context gets boosted scores)
    unique_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"Enhanced search returning {len(unique_results)} results")
    for i, result in enumerate(unique_results[:3]):
        print(f"  {i+1}. {result['source']} (score: {result['score']:.2f})")
    
    return unique_results[:k]

# Thread and Bookmark Functions
def create_thread(title: str, project: str, tags: List[str] = None) -> str:
    """Create a new conversation thread"""
    with get_reliable_db_connection() as conn:
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO chat_thread_metadata (title, project, tags)
                VALUES (%s, %s, %s)
                RETURNING thread_id
            ''', (title, project, tags or []))
            
            thread_id = cursor.fetchone()[0]
            conn.commit()
            return str(thread_id)
            
        except Exception as e:
            print(f"Failed to create thread: {e}")
            conn.rollback()
            return None

def add_conversation_to_thread(chat_id: int, thread_id: str):
    """Add a conversation to a thread"""
    with get_reliable_db_connection() as conn:
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE chat_threads 
                SET thread_id = %s 
                WHERE id = %s
            ''', (thread_id, chat_id))
            
            # Update message count
            cursor.execute('''
                UPDATE chat_thread_metadata 
                SET message_count = (
                    SELECT COUNT(*) FROM chat_threads WHERE thread_id = %s
                ),
                updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = %s
            ''', (thread_id, thread_id))
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Failed to add conversation to thread: {e}")
            conn.rollback()
            return False

def create_bookmark(chat_id: int, title: str, notes: str = None, bookmark_type: str = 'manual') -> str:
    """Create a bookmark for a specific conversation"""
    with get_reliable_db_connection() as conn:
        if not conn:
            return None
        
        try:
            cursor = conn.cursor()
            
            # Get thread_id if exists
            cursor.execute('SELECT thread_id FROM chat_threads WHERE id = %s', (chat_id,))
            result = cursor.fetchone()
            thread_id = result[0] if result else None
            
            cursor.execute('''
                INSERT INTO conversation_bookmarks (chat_id, thread_id, title, notes, bookmark_type)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING bookmark_id
            ''', (chat_id, thread_id, title, notes, bookmark_type))
            
            bookmark_id = cursor.fetchone()[0]
            conn.commit()
            return str(bookmark_id)
            
        except Exception as e:
            print(f"Failed to create bookmark: {e}")
            conn.rollback()
            return None

def test_miller_search():
    """Test function to verify Miller search is working"""
    print("Testing Miller search...")
    
    results = enhanced_context_search("Miller cat", k=5)
    
    print(f"Found {len(results)} results for 'Miller cat':")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['source']} (score: {result['score']:.2f})")
        print(f"   Content: {result['text'][:200]}...")
    
    return len(results) > 0

if __name__ == '__main__':
    # Test the search
    test_miller_search()