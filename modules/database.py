# modules/database.py - COMPLETE REWRITE with Enhanced Search and Thread Support
# Fixed database operations with proper full-text search indexes and thread/bookmark functionality

# ==========================================
# Section 1: Imports and Configuration
# ==========================================
import os
import datetime
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import time

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    # Railway provides postgres:// but psycopg2 needs postgresql://
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

# ==========================================
# Section 2: Enhanced Database Connection Management
# ==========================================
# ==========================================
# Section 2: Enhanced Database Connection Management 9/15/25
# ==========================================
# ==========================================
# Section 2: Enhanced Database Connection Management 9/15/25
# ==========================================
# ==========================================
# Section 2: Enhanced Database Connection Management 9/15/25
# ==========================================

@contextmanager
def get_db_connection():
    """TUPLE-SAFE: Database connection that prevents generator/tuple errors"""
    conn = None
    try:
        if not DATABASE_URL:
            print("No DATABASE_URL configured - using file storage only")
            yield None
            return
        
        # Single connection attempt without problematic retry loops
        conn = psycopg2.connect(
            DATABASE_URL,
            connect_timeout=15,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=5,
            keepalives_count=3
        )
        yield conn
        
    except psycopg2.Error as db_error:
        print(f"PostgreSQL connection error: {db_error}")
        yield None
    except Exception as general_error:
        print(f"Database connection error: {general_error}")
        yield None
    finally:
        # Safe cleanup that won't cause generator errors
        if conn is not None:
            try:
                conn.close()
            except:
                pass

def test_database_connection():
    """Test database connectivity and return status"""
    with get_db_connection() as conn:
        if not conn:
            return False, "No database connection available"
        
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT 1')
            result = cursor.fetchone()
            if result and len(result) > 0:
                return True, "Database connection successful"
            else:
                return False, "Database query returned no results"
        except Exception as e:
            return False, f"Database test failed: {e}"

def get_database_connection_info():
    """Get database connection information for diagnostics"""
    info = {
        'database_url_configured': bool(DATABASE_URL),
        'database_url_length': len(DATABASE_URL) if DATABASE_URL else 0,
        'connection_working': False,
        'connection_test_time': None,
        'error_message': None
    }
    
    start_time = time.time()
    
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute('SELECT version()')
                version_result = cursor.fetchone()
                
                info['connection_working'] = True
                info['connection_test_time'] = time.time() - start_time
                
                # Safe version string extraction
                if version_result and len(version_result) > 0:
                    version_str = str(version_result[0])
                    info['database_version'] = version_str[:100]  # Truncate long version strings
                
                # Test table existence with tuple-safe handling
                cursor.execute('''
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('chat_threads', 'brain_documents')
                ''')
                table_results = cursor.fetchall()
                
                # Safe table name extraction
                tables = []
                for table_result in table_results:
                    if table_result and len(table_result) > 0:
                        tables.append(str(table_result[0]))
                
                info['essential_tables'] = tables
                info['tables_exist'] = len(tables) >= 2
                
            else:
                info['error_message'] = "Database connection returned None"
                
    except Exception as e:
        info['error_message'] = str(e)
        info['connection_test_time'] = time.time() - start_time
    
    return info

def get_reliable_db_connection():
    """Alternative connection function with simpler retry logic"""
    if not DATABASE_URL:
        return None
    
    try:
        conn = psycopg2.connect(
            DATABASE_URL,
            connect_timeout=15,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=3
        )
        return conn
    except Exception as e:
        print(f"Reliable connection failed: {e}")
        return None

@contextmanager
def get_simple_db_connection():
    """Simplified database connection without complex retry logic"""
    conn = None
    
    try:
        if DATABASE_URL:
            conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
            yield conn
        else:
            yield None
    except Exception as e:
        print(f"Simple database connection failed: {e}")
        yield None
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def execute_safe_query(query, params=None, fetch_one=False, fetch_all=False):
    """Execute a database query with automatic connection management and tuple-safe error handling"""
    try:
        with get_db_connection() as conn:
            if not conn:
                print("No database connection available for query")
                return None
            
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch_one:
                result = cursor.fetchone()
                return result
            elif fetch_all:
                results = cursor.fetchall()
                return results
            else:
                conn.commit()
                return cursor.rowcount
                
    except psycopg2.Error as db_error:
        print(f"PostgreSQL query error: {db_error}")
        return None
    except Exception as general_error:
        print(f"Safe query execution failed: {general_error}")
        return None

def get_database_health_check():
    """Comprehensive database health check with tuple-safe operations"""
    health = {
        'timestamp': datetime.datetime.now().isoformat(),
        'connection_info': get_database_connection_info(),
        'table_counts': {},
        'recent_activity': {},
        'health_status': 'unknown'
    }
    
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                
                # Get table counts with safe result handling
                tables_to_check = ['chat_threads', 'brain_documents', 'brain_health']
                for table in tables_to_check:
                    try:
                        cursor.execute(f'SELECT COUNT(*) FROM {table}')
                        count_result = cursor.fetchone()
                        if count_result and len(count_result) > 0:
                            health['table_counts'][table] = int(count_result[0])
                        else:
                            health['table_counts'][table] = 0
                    except Exception as e:
                        health['table_counts'][table] = f'Error: {e}'
                
                # Get recent activity with tuple-safe handling
                try:
                    cursor.execute('''
                        SELECT COUNT(*) FROM chat_threads 
                        WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                    ''')
                    activity_result = cursor.fetchone()
                    if activity_result and len(activity_result) > 0:
                        health['recent_activity']['conversations_last_7_days'] = int(activity_result[0])
                    else:
                        health['recent_activity']['conversations_last_7_days'] = 0
                except Exception as e:
                    health['recent_activity']['conversations_last_7_days'] = f'Error: {e}'
                
                # Test a simple Miller search with safe result extraction
                try:
                    cursor.execute('''
                        SELECT COUNT(*) FROM chat_threads 
                        WHERE LOWER(user_input) LIKE '%miller%' 
                           OR LOWER(response_data->>'SyntaxPrime') LIKE '%miller%'
                    ''')
                    miller_result = cursor.fetchone()
                    if miller_result and len(miller_result) > 0:
                        miller_count = int(miller_result[0])
                        health['miller_conversations'] = miller_count
                        
                        if miller_count > 0:
                            health['health_status'] = 'healthy'
                        else:
                            health['health_status'] = 'no_data'
                    else:
                        health['miller_conversations'] = 0
                        health['health_status'] = 'no_data'
                        
                except Exception as e:
                    health['miller_conversations'] = f'Error: {e}'
                    health['health_status'] = 'error'
            else:
                health['health_status'] = 'no_connection'
                
    except Exception as e:
        health['health_status'] = 'error'
        health['error'] = str(e)
    
    return health

def debug_tuple_error():
    """Diagnostic function to identify the exact source of tuple index errors"""
    
    print("\n" + "="*60)
    print("DEBUGGING TUPLE INDEX ERROR - COMPREHENSIVE TEST")
    print("="*60)
    
    # Test 1: Database URL and basic connection
    print("\n1. Testing database configuration...")
    if not DATABASE_URL:
        print("❌ No DATABASE_URL configured")
        return
    else:
        print(f"✅ DATABASE_URL configured (length: {len(DATABASE_URL)})")
    
    # Test 2: Basic connection without context manager
    print("\n2. Testing direct connection...")
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
        print("✅ Direct connection successful")
        
        # Test 3: Simple query
        print("\n3. Testing simple query...")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM chat_threads WHERE LOWER(user_input) LIKE '%miller%'")
        count_result = cursor.fetchone()
        
        if count_result and len(count_result) > 0:
            print(f"✅ Simple query works: {count_result[0]} Miller conversations")
        else:
            print("❌ Simple query returned empty result")
            
        # Test 4: RealDictCursor
        print("\n4. Testing RealDictCursor...")
        dict_cursor = conn.cursor(cursor_factory=RealDictCursor)
        dict_cursor.execute("""
            SELECT id, user_input, response_data, project, created_at
            FROM chat_threads 
            WHERE LOWER(user_input) LIKE '%miller%'
            LIMIT 1
        """)
        
        row = dict_cursor.fetchone()
        if row:
            print(f"✅ RealDictCursor works")
            print(f"   Row type: {type(row)}")
            
            # Test 5: Different access methods
            print("\n5. Testing data access methods...")
            
            # Method 1: Dictionary access
            try:
                if hasattr(row, 'get'):
                    test_id = row.get('id')
                    test_input = row.get('user_input', '')
                    print(f"✅ Dictionary access: ID={test_id}")
                    print(f"   Input preview: {test_input[:50]}...")
                else:
                    print("❌ Row doesn't support dictionary access")
            except Exception as dict_error:
                print(f"❌ Dictionary access failed: {dict_error}")
            
            # Method 2: Tuple access
            try:
                if hasattr(row, '__len__'):
                    print(f"   Row length: {len(row)}")
                    if len(row) >= 5:
                        test_id_tuple = row[0]
                        test_input_tuple = row[1] if row[1] else ''
                        print(f"✅ Tuple access: ID={test_id_tuple}")
                        print(f"   Input preview: {test_input_tuple[:50]}...")
                    else:
                        print(f"❌ Row too short for tuple access: length {len(row)}")
                else:
                    print("❌ Row doesn't support length/indexing")
            except IndexError as idx_error:
                print(f"❌ Tuple access failed with IndexError: {idx_error}")
            except Exception as tuple_error:
                print(f"❌ Tuple access failed: {tuple_error}")
            
        else:
            print("❌ No Miller rows found in database")
        
        # Test 6: Context manager
        print("\n6. Testing context manager...")
        try:
            with get_db_connection() as test_conn:
                if test_conn:
                    print("✅ Context manager connection works")
                    test_cursor = test_conn.cursor()
                    test_cursor.execute("SELECT 1")
                    test_result = test_cursor.fetchone()
                    if test_result and len(test_result) > 0:
                        print("✅ Context manager query works")
                    else:
                        print("❌ Context manager query returned empty")
                else:
                    print("❌ Context manager returned None")
        except Exception as cm_error:
            print(f"❌ Context manager failed: {cm_error}")
            import traceback
            traceback.print_exc()
        
        conn.close()
        
    except psycopg2.Error as db_error:
        print(f"❌ PostgreSQL error: {db_error}")
    except Exception as general_error:
        print(f"❌ General connection error: {general_error}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TUPLE ERROR DEBUG COMPLETE")
    print("="*60)
# ==========================================
# Section 3: Content Classification and Filtering
# ==========================================

def get_content_tier(content: str) -> str:
    """Classify content by length and substance into tiers for smart filtering"""
    if not content or not content.strip():
        return "minimal"
    
    content = content.strip()
    length = len(content)
    word_count = len(content.split())
    sentence_count = content.count('.') + content.count('!') + content.count('?')
    
    # Minimal: Very short responses, greetings, confirmations
    if (length < 50 or word_count < 8 or
        any(greeting in content.lower() for greeting in ['hi', 'hello', 'thanks', 'ok', 'yes', 'no'])):
        return "minimal"
    
    # Basic: Short but substantial responses
    elif length < 200 or word_count < 30 or sentence_count <= 2:
        return "basic"
    
    # Substantial: Medium-length informative content
    elif length < 800 or word_count < 120 or sentence_count <= 5:
        return "substantial"
    
    # Comprehensive: Long, detailed responses
    else:
        return "comprehensive"

def should_include_content(content: str, context_type: str = "mixed") -> bool:
    """Smart filter to determine if content should be included based on tier and context"""
    tier = get_content_tier(content)
    
    # Context-aware filtering
    if context_type == "personal_context":
        # For personal context, include more content types
        return tier in ["basic", "substantial", "comprehensive"]
    elif context_type == "knowledge_base":
        # For knowledge base, prioritize substantial content
        return tier in ["substantial", "comprehensive"]
    elif context_type == "recent_priority":
        # For recent high-priority searches, include most content
        return tier in ["basic", "substantial", "comprehensive"]
    else:  # mixed or default
        # Standard filtering - exclude only minimal content
        return tier != "minimal"

def classify_search_intent(query_text: str, conversation_context: List[str] = None) -> str:
    """Classify whether a search needs personal context vs knowledge base - ENHANCED VERSION"""
    
    query_lower = query_text.lower().strip()
    
    # Personal context indicators (people, family, ongoing situations) - EXPANDED
    personal_patterns = [
        # Family and personal relationships - COMPREHENSIVE
        "miller", "ghada", "shazeen", "mom", "mother", "wife", "daughter", "cat", "tux cat",
        "my family", "my daughter", "my wife", "my mom", "my cat", "my child", "fajr",
        "muhi", "inner circle", "family", "hysterectomy", "surgery", "recovery", "tired",
        
        # Work and personal projects - EXPANDED
        "amcf", "my company", "my work", "my projects", "my sites", "meals n feelz",
        "mealsnfeelz", "rose and angel", "bcdodgeme", "tv signals", "damn it carl",
        "halalbot", "kitchen", "health", "side quests", "nonprofit", "giving circle",
        
        # Personal situations and locations
        "nh", "new hampshire", "estate", "mom's estate", "courts", "apartment",
        "hubspot", "conference", "summit", "ticket sales", "my marketing",
        
        # Daily routines and personal contexts
        "cup one", "coffee", "breakfast", "morning routine", "5am", "5:00", "9 to 16:30",
        "work windows", "my schedule", "goals", "weekly goals", "deliverables",
        
        # Personal tools and AI relationship
        "ghostline", "syntax", "syntaxprime", "you", "we", "our conversation",
        "reminder", "reminders", "telegram", "my workflow", "my system",
        
        # Conversational continuity - EXPANDED
        "we were talking about", "you mentioned", "earlier you said", "as we discussed",
        "my situation", "my project", "our conversation", "what i told you",
        "catch me up", "update me", "where are we", "current status", "latest on",
        "how are things", "what's happening with", "progress", "status update",
        
        # Time-based personal references - EXPANDED
        "today", "yesterday", "this week", "recently", "just now", "right now",
        "currently", "at the moment", "these days", "this morning", "tonight",
        "earlier today", "last night", "this weekend", "past few days"
    ]
    
    # Knowledge base indicators (factual/reference information)
    knowledge_patterns = [
        # Factual queries
        "what is", "what does", "tell me about", "explain", "describe", "define",
        "how does", "why does", "when was", "where is", "who is", "who was",
        
        # TV shows, entertainment, general knowledge
        "dead like me", "happy time", "tv show", "television", "movie", "book",
        "actor", "character", "episode", "season", "plot",
        
        # Technical/procedural
        "how to", "tutorial", "instructions", "guide", "documentation",
        "best practices", "examples of", "tips for",
        
        # General topics (not personal)
        "marketing strategy", "seo", "algorithm", "technology", "history of"
    ]
    
    # Check patterns in order of specificity
    if any(pattern in query_lower for pattern in personal_patterns):
        return "personal_context"
    elif any(pattern in query_lower for pattern in knowledge_patterns):
        return "knowledge_base"
    
    # Context-based classification
    if conversation_context:
        recent_topics = " ".join(conversation_context[-3:]).lower()  # Last 3 exchanges
        
        # If recent conversation mentioned personal topics, lean personal
        if any(term in recent_topics for term in ["miller", "ghada", "shazeen", "amcf", "daughter", "project"]):
            return "personal_context"
    
    # Default based on query length and structure
    if len(query_text.split()) <= 3:
        return "personal_context"  # Short queries often reference ongoing conversation
    else:
        return "knowledge_base"   # Longer queries often seek information

# ==========================================
# Section 4: Enhanced Full-Text Search Functions
# ==========================================

def search_personal_context(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search personal context using family names and personal keywords - ENHANCED FOR MILLER"""
    
    # Enhanced personal context keywords
    personal_keywords = [
        'miller', 'ghada', 'shazeen', 'mom', 'mother', 'family', 'daughter', 'wife', 'cat', 'tux',
        'amcf', 'my daughter', 'my wife', 'my mom', 'my cat', 'tux cat', 'tuxedo cat'
    ]
    
    # Check if query contains personal keywords
    query_lower = query_text.lower()
    is_personal = any(keyword in query_lower for keyword in personal_keywords)
    
    if not is_personal:
        return []
    
    with get_db_connection() as conn:
        if not conn:
            print("No database connection for personal context search")
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Enhanced personal context search with full-text search
            personal_sql = '''
                SELECT id, project, user_input, response_data, created_at,
                       ts_rank(to_tsvector('english', 
                               user_input || ' ' || 
                               COALESCE(response_data->>'SyntaxPrime', '')), 
                               plainto_tsquery('english', %s)) as rank
                FROM chat_threads 
                WHERE to_tsvector('english', 
                      user_input || ' ' || 
                      COALESCE(response_data->>'SyntaxPrime', '')) 
                      @@ plainto_tsquery('english', %s)
                   AND (LOWER(user_input) LIKE ANY(%s) 
                        OR LOWER(COALESCE(response_data->>'SyntaxPrime', '')) LIKE ANY(%s))
                ORDER BY rank DESC, created_at DESC
                LIMIT %s
            '''
            
            like_patterns = [f'%{kw}%' for kw in personal_keywords]
            
            cursor.execute(personal_sql, (
                query_text, query_text, like_patterns, like_patterns, k * 2
            ))
            rows = cursor.fetchall()
            
            print(f"Personal context search for '{query_text}' found {len(rows)} results")
            
            # Convert to RAG format with high priority scoring
            results = []
            for row in rows:
                response_content = row['response_data'].get('SyntaxPrime', '') if row['response_data'] else ''
                
                # Apply smart filtering for personal context
                if should_include_content(response_content, "personal_context"):
                    combined_text = f"User: {row['user_input']}\nResponse: {response_content}"
                    
                    results.append({
                        'text': combined_text[:1500],
                        'source': f"Personal Memory - {row['project']} ({row['created_at'].strftime('%m/%d/%Y')})",
                        'id': f"personal_{row['id']}",
                        'score': float(row['rank']) + 2.0,  # Boost personal context significantly
                        'metadata': {
                            'type': 'personal_context',
                            'project': row['project'],
                            'date': row['created_at'].isoformat(),
                            'chat_id': row['id'],
                            'priority': 'high',
                            'content_tier': get_content_tier(response_content)
                        }
                    })
                    
                    if len(results) >= k:
                        break
            
            print(f"Personal context search returning {len(results)} quality results")
            return results
            
        except Exception as e:
            print(f"Personal context search failed: {e}")
            return []

def search_recent_conversations(query_text: str, k: int = 3, days: int = 7) -> List[Dict[str, Any]]:
    """Search recent conversation history with enhanced full-text search"""
    
    with get_db_connection() as conn:
        if not conn:
            print("No database connection for recent conversation search")
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search conversations from last N days with full-text search
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            search_sql = '''
                SELECT id, project, user_input, response_data, created_at,
                       ts_rank(to_tsvector('english', 
                               user_input || ' ' || 
                               COALESCE(response_data->>'SyntaxPrime', '')), 
                               plainto_tsquery('english', %s)) as rank
                FROM chat_threads 
                WHERE created_at >= %s
                   AND to_tsvector('english', 
                       user_input || ' ' || 
                       COALESCE(response_data->>'SyntaxPrime', '')) 
                       @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC, created_at DESC
                LIMIT %s
            '''
            
            cursor.execute(search_sql, (query_text, cutoff_date, query_text, k * 2))
            rows = cursor.fetchall()
            
            print(f"Recent conversation search found {len(rows)} results from last {days} days")
            
            # Convert to RAG format with smart filtering
            results = []
            for row in rows:
                response_content = row['response_data'].get('SyntaxPrime', '') if row['response_data'] else ''
                
                # Apply smart filtering for recent conversations
                if should_include_content(response_content, "recent_priority"):
                    combined_text = f"User: {row['user_input']}\nResponse: {response_content}"
                    
                    results.append({
                        'text': combined_text[:1200],
                        'source': f"Recent - {row['project']} ({row['created_at'].strftime('%m/%d')})",
                        'id': f"recent_{row['id']}",
                        'score': float(row['rank']) + 1.0,  # Boost recent conversations
                        'metadata': {
                            'type': 'recent_conversation',
                            'project': row['project'],
                            'date': row['created_at'].isoformat(),
                            'chat_id': row['id'],
                            'content_tier': get_content_tier(response_content)
                        }
                    })
                    
                    if len(results) >= k:
                        break
            
            print(f"Recent conversation search returning {len(results)} quality results")
            return results
            
        except Exception as e:
            print(f"Recent conversation search failed: {e}")
            return []

def search_knowledge_base_only(query_text: str, k: int = 5) -> List[Dict[str, Any]]:
    """Search only knowledge base documents with enhanced filtering"""
    
    with get_db_connection() as conn:
        if not conn:
            print("No database connection for knowledge base search")
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Search brain documents with full-text search
            search_sql = '''
                SELECT document_id, title, content, metadata,
                       ts_rank(to_tsvector('english', content || ' ' || COALESCE(title, '')), 
                               plainto_tsquery('english', %s)) as rank
                FROM brain_documents 
                WHERE to_tsvector('english', content || ' ' || COALESCE(title, '')) 
                      @@ plainto_tsquery('english', %s)
                -- Exclude conversation-like content
                AND NOT (content LIKE '%User:%' OR content LIKE '%Assistant:%' OR content LIKE '%Response:%')
                ORDER BY rank DESC
                LIMIT %s
            '''
            
            cursor.execute(search_sql, (query_text, query_text, k * 2))
            rows = cursor.fetchall()
            
            if not rows:
                # Fallback: try simpler search without conversation filtering
                fallback_sql = '''
                    SELECT document_id, title, content, metadata, 1.0 as rank
                    FROM brain_documents 
                    WHERE LOWER(content) LIKE %s OR LOWER(COALESCE(title, '')) LIKE %s
                    ORDER BY LENGTH(content) DESC
                    LIMIT %s
                '''
                
                like_pattern = f'%{query_text.lower()}%'
                cursor.execute(fallback_sql, (like_pattern, like_pattern, k * 2))
                rows = cursor.fetchall()
                print(f"Knowledge base fallback search found {len(rows)} results")
            else:
                print(f"Knowledge base search found {len(rows)} results")
            
            # Convert to RAG format with smart filtering
            results = []
            for row in rows:
                content = row['content']
                
                # Apply smart filtering for knowledge base content
                if should_include_content(content, "knowledge_base"):
                    results.append({
                        'text': content[:1500],
                        'source': row['title'] or f"Document {row['document_id']}",
                        'id': f"kb_{row['document_id']}",
                        'score': float(row['rank']),
                        'metadata': {
                            'type': 'knowledge_base',
                            'content_tier': get_content_tier(content),
                            **(row['metadata'] or {})
                        }
                    })
                    
                    if len(results) >= k:
                        break
            
            print(f"Knowledge base search returning {len(results)} quality results")
            return results
            
        except Exception as e:
            print(f"Knowledge base search failed: {e}")
            return []

def enhanced_context_search(query_text: str, k: int = 5, conversation_context: List[str] = None) -> List[Dict[str, Any]]:
    """MAIN SEARCH FUNCTION - Intelligent search routing with Miller-aware personal context"""
    
    # Classify the search intent
    intent = classify_search_intent(query_text, conversation_context)
    print(f"Enhanced search classified query '{query_text}' as: {intent}")
    
    if intent == "personal_context":
        print("Prioritizing personal context search (family, Miller, ongoing projects)")
        
        # Search personal context first with high priority
        personal_results = search_personal_context(query_text, k=max(3, k//2))
        
        # Add recent conversations if we need more context
        if len(personal_results) < 2:
            recent_results = search_recent_conversations(query_text, k=2, days=7)
            personal_results.extend(recent_results)
            print(f"Personal context search: {len(personal_results)} total results (personal + recent)")
        
        # Add minimal knowledge base if still not enough
        if len(personal_results) < k//2:
            kb_results = search_knowledge_base_only(query_text, k=1)
            personal_results.extend(kb_results)
        
        return personal_results[:k]
    
    elif intent == "knowledge_base":
        print("Prioritizing knowledge base search")
        
        # Search knowledge base primarily
        kb_results = search_knowledge_base_only(query_text, k=k)
        
        # Add minimal recent context for continuity
        recent_results = search_recent_conversations(query_text, k=1, days=3)
        if recent_results:
            kb_results.insert(0, recent_results[0])  # Put recent context first
        
        print(f"Knowledge base search: {len(kb_results)} results")
        return kb_results[:k]
    
    else:
        print("Using balanced search approach")
        
        # Balanced approach for unclear intent
        recent_results = search_recent_conversations(query_text, k=2, days=7)
        kb_results = search_knowledge_base_only(query_text, k=3)
        
        combined = recent_results + kb_results
        print(f"Balanced search: {len(combined)} results (recent + knowledge)")
        return combined[:k]

# ==========================================
# Section 5: Legacy Search Function (for compatibility)
# ==========================================

def search_brain_database(query_text, k=5):
    """Main search function - uses enhanced context search with full-text indexes"""
    try:
        print(f"Search request: '{query_text}' (limit: {k})")
        
        # Use the enhanced search system
        results = enhanced_context_search(query_text, k=k)
        
        if results:
            print(f"Enhanced search found {len(results)} results")
            
            # Log the search for monitoring
            update_brain_health(query_text, len(results))
            
            return results
        else:
            print(f"No results found for '{query_text}'")
            update_brain_health(query_text, 0)
            return []
            
    except Exception as e:
        print(f"Search failed: {e}")
        update_brain_health(query_text, 0, error=str(e))
        return []

# ==========================================
# Section 6: Brain Health Monitoring
# ==========================================

def update_brain_health(query=None, results_count=0, error=None):
    """Update brain health monitoring with search statistics"""
    with get_db_connection() as conn:
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            # Get current document count
            cursor.execute('SELECT COUNT(*) FROM brain_documents')
            doc_count = cursor.fetchone()[0]
            
            # Get current conversation count
            cursor.execute('SELECT COUNT(*) FROM chat_threads')
            chat_count = cursor.fetchone()[0]
            
            # Update or insert health record
            cursor.execute('''
                INSERT INTO brain_health (last_refresh, total_documents, last_search_query, 
                                        last_search_results, health_status, error_log)
                VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            ''', (doc_count, query, results_count, 'healthy' if not error else 'error', error))
            
            # Also update the most recent record if it exists
            cursor.execute('''
                UPDATE brain_health 
                SET last_refresh = CURRENT_TIMESTAMP,
                    total_documents = %s,
                    last_search_query = %s,
                    last_search_results = %s,
                    health_status = %s,
                    error_log = %s
                WHERE id = (SELECT MAX(id) FROM brain_health)
            ''', (doc_count, query, results_count, 'healthy' if not error else 'error', error))
            
            conn.commit()
            
        except Exception as e:
            print(f"Failed to update brain health: {e}")

def get_brain_health_status():
    """Get current brain health status with enhanced monitoring"""
    with get_db_connection() as conn:
        if not conn:
            return {"status": "no_database", "message": "No database connection"}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get latest health record
            cursor.execute('''
                SELECT * FROM brain_health 
                ORDER BY last_refresh DESC 
                LIMIT 1
            ''')
            
            health_record = cursor.fetchone()
            
            # Get conversation count
            cursor.execute('SELECT COUNT(*) FROM chat_threads')
            chat_count = cursor.fetchone()[0]
            
            # Test search functionality
            test_query = "miller"
            test_results = enhanced_context_search(test_query, k=1)
            search_working = len(test_results) > 0
            
            if not health_record:
                return {
                    "status": "unknown",
                    "message": "No health records found",
                    "chat_threads_count": chat_count,
                    "search_working": search_working
                }
            
            return {
                "status": health_record['health_status'],
                "last_refresh": health_record['last_refresh'].isoformat(),
                "total_documents": health_record['total_documents'],
                "chat_threads_count": chat_count,
                "last_search_query": health_record['last_search_query'],
                "last_search_results": health_record['last_search_results'],
                "search_working": search_working,
                "error_log": health_record['error_log']
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}

# ==========================================
# Section 7: Conversation Storage and Retrieval
# ==========================================

def load_conversation_enhanced(project: str, limit: int = 50):
    """Load conversation history from database with enhanced context"""
    conversations = []
    
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute('''
                    SELECT id, user_input, response_data, created_at, context_project, context_data, thread_id
                    FROM chat_threads 
                    WHERE project = %s 
                    ORDER BY created_at DESC 
                    LIMIT %s
                ''', (project, limit))
                
                rows = cursor.fetchall()
                for row in rows:
                    conversations.append({
                        "id": f"db_{row['id']}",
                        "chat_id": row['id'],
                        "user": row['user_input'],
                        "responses": row['response_data'],
                        "timestamp": row['created_at'].isoformat(),
                        "context_project": row.get('context_project'),
                        "context_data": row.get('context_data', {}),
                        "thread_id": str(row['thread_id']) if row.get('thread_id') else None
                    })
                
                # Reverse to get chronological order
                conversations.reverse()
                print(f"Loaded {len(conversations)} conversations from database for {project}")
                return conversations
                
            except Exception as e:
                print(f"Failed to load conversations from database: {e}")
    
    return []

def save_conversation_enhanced(project: str, user_input: str, response_data: dict,
                             context_project: str = None, context_data: dict = None):
    """Enhanced conversation saving with project context and thread support"""
    with get_db_connection() as conn:
        if not conn:
            print("No database connection - conversation not saved")
            return None
        
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO chat_threads 
                (project, user_input, response_data, context_project, context_data)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            ''', (
                project,
                user_input,
                json.dumps(response_data, ensure_ascii=False),
                context_project,
                json.dumps(context_data or {}, ensure_ascii=False)
            ))
            
            chat_id = cursor.fetchone()[0]
            conn.commit()
            
            print(f"Conversation saved to database for {project} (ID: {chat_id}, context: {context_project})")
            return chat_id
            
        except Exception as e:
            print(f"Error saving conversation: {e}")
            conn.rollback()
            return None

def get_conversation_context(project: str, limit: int = 5) -> List[str]:
    """Get recent conversation context for intent classification"""
    
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_input, response_data 
                FROM chat_threads 
                WHERE project = %s 
                ORDER BY created_at DESC 
                LIMIT %s
            ''', (project, limit))
            
            rows = cursor.fetchall()
            context = []
            
            for row in rows:
                context.append(row[0])  # user input
                if row[1] and 'SyntaxPrime' in row[1]:
                    response_content = row[1]['SyntaxPrime']
                    # Only include substantial responses in context
                    if should_include_content(response_content, "recent_priority"):
                        context.append(response_content[:200])  # truncated response
            
            return context
            
        except Exception as e:
            print(f"Failed to get conversation context: {e}")
            return []

# ==========================================
# Section 8: Thread Management System
# ==========================================

def create_thread(title: str, project: str, tags: List[str] = None) -> Optional[str]:
    """Create a new conversation thread"""
    with get_db_connection() as conn:
        if not conn:
            print("No database connection for thread creation")
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
            
            print(f"Created thread '{title}' with ID: {thread_id}")
            return str(thread_id)
            
        except Exception as e:
            print(f"Failed to create thread: {e}")
            conn.rollback()
            return None

def add_conversation_to_thread(chat_id: int, thread_id: str) -> bool:
    """Add a conversation to an existing thread"""
    with get_db_connection() as conn:
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Update the conversation with thread_id
            cursor.execute('''
                UPDATE chat_threads 
                SET thread_id = %s 
                WHERE id = %s
            ''', (thread_id, chat_id))
            
            # Update thread metadata
            cursor.execute('''
                UPDATE chat_thread_metadata 
                SET message_count = (
                    SELECT COUNT(*) FROM chat_threads WHERE thread_id = %s
                ),
                updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = %s
            ''', (thread_id, thread_id))
            
            conn.commit()
            
            print(f"Added conversation {chat_id} to thread {thread_id}")
            return True
            
        except Exception as e:
            print(f"Failed to add conversation to thread: {e}")
            conn.rollback()
            return False

def get_thread_conversations(thread_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Get all conversations in a thread"""
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('''
                SELECT ct.id, ct.user_input, ct.response_data, ct.created_at, ct.project,
                       tm.title as thread_title
                FROM chat_threads ct
                JOIN chat_thread_metadata tm ON ct.thread_id = tm.thread_id
                WHERE ct.thread_id = %s
                ORDER BY ct.created_at ASC
                LIMIT %s
            ''', (thread_id, limit))
            
            rows = cursor.fetchall()
            conversations = []
            
            for row in rows:
                conversations.append({
                    'id': row['id'],
                    'user_input': row['user_input'],
                    'response_data': row['response_data'],
                    'created_at': row['created_at'].isoformat(),
                    'project': row['project'],
                    'thread_title': row['thread_title']
                })
            
            return conversations
            
        except Exception as e:
            print(f"Failed to get thread conversations: {e}")
            return []

def list_threads(project: str = None, limit: int = 20) -> List[Dict[str, Any]]:
    """List available threads, optionally filtered by project"""
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if project:
                cursor.execute('''
                    SELECT thread_id, title, project, created_at, updated_at, 
                           message_count, tags, is_archived
                    FROM chat_thread_metadata
                    WHERE project = %s AND NOT is_archived
                    ORDER BY updated_at DESC
                    LIMIT %s
                ''', (project, limit))
            else:
                cursor.execute('''
                    SELECT thread_id, title, project, created_at, updated_at, 
                           message_count, tags, is_archived
                    FROM chat_thread_metadata
                    WHERE NOT is_archived
                    ORDER BY updated_at DESC
                    LIMIT %s
                ''', (limit,))
            
            return cursor.fetchall()
            
        except Exception as e:
            print(f"Failed to list threads: {e}")
            return []

# ==========================================
# Section 9: Bookmark Management System
# ==========================================
# ==========================================
# Section 9: Bookmark Management System (UPDATED WITH FIXES) 9/17/25
# ==========================================

def create_bookmark(chat_id: int, title: str, notes: str = None, bookmark_type: str = 'manual') -> str:
    """Create a bookmark for a conversation with better error handling"""
    with get_db_connection() as conn:
        if not conn:
            print("No database connection available")
            return None
            
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # First, verify the conversation exists and get thread_id
            cursor.execute('''
                SELECT id, thread_id, project, user_input
                FROM chat_threads
                WHERE id = %s
            ''', (chat_id,))
            
            conversation = cursor.fetchone()
            if not conversation:
                print(f"Conversation {chat_id} not found")
                return None
            
            thread_id = conversation['thread_id']
            
            # Check if bookmark already exists for this conversation
            cursor.execute('''
                SELECT bookmark_id FROM conversation_bookmarks 
                WHERE chat_id = %s
            ''', (chat_id,))
            
            existing = cursor.fetchone()
            if existing:
                print(f"Bookmark already exists for conversation {chat_id}: {existing['bookmark_id']}")
                return str(existing['bookmark_id'])
            
            # Ensure thread_id is set (create if missing)
            if not thread_id:
                print(f"Creating thread_id for conversation {chat_id}")
                cursor.execute('''
                    UPDATE chat_threads 
                    SET thread_id = gen_random_uuid()
                    WHERE id = %s
                    RETURNING thread_id
                ''', (chat_id,))
                result = cursor.fetchone()
                thread_id = result['thread_id'] if result else None
                
                if thread_id:
                    # Also create thread metadata
                    cursor.execute('''
                        INSERT INTO chat_thread_metadata (thread_id, title, project)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (thread_id) DO NOTHING
                    ''', (thread_id, title[:500], conversation['project']))
            
            # Create the bookmark
            cursor.execute('''
                INSERT INTO conversation_bookmarks (chat_id, thread_id, title, notes, bookmark_type)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING bookmark_id
            ''', (chat_id, thread_id, title, notes, bookmark_type))
            
            result = cursor.fetchone()
            bookmark_id = result['bookmark_id']
            conn.commit()
            
            print(f"✅ Successfully created bookmark '{title}' with ID {bookmark_id} for conversation {chat_id}")
            return str(bookmark_id)
            
        except Exception as e:
            print(f"❌ Failed to create bookmark: {e}")
            conn.rollback()
            return None

def get_bookmarks(thread_id: str = None, project: str = None, limit: int = 20, bookmark_ids: list = None) -> List[Dict[str, Any]]:
    """Get bookmarks with enhanced data, optionally filtered by thread, project, or specific IDs"""
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            if bookmark_ids:
                # Get specific bookmarks by ID
                placeholders = ','.join(['%s'] * len(bookmark_ids))
                cursor.execute(f'''
                    SELECT cb.bookmark_id, cb.title, cb.notes, cb.bookmark_type, cb.created_at,
                           cb.chat_id, ct.user_input, ct.project, ct.response_data
                    FROM conversation_bookmarks cb
                    JOIN chat_threads ct ON cb.chat_id = ct.id
                    WHERE cb.bookmark_id IN ({placeholders})
                    ORDER BY cb.created_at DESC
                ''', bookmark_ids)
                
            elif thread_id:
                cursor.execute('''
                    SELECT cb.bookmark_id, cb.title, cb.notes, cb.bookmark_type, cb.created_at,
                           cb.chat_id, ct.user_input, ct.project, ct.response_data
                    FROM conversation_bookmarks cb
                    JOIN chat_threads ct ON cb.chat_id = ct.id
                    WHERE cb.thread_id = %s
                    ORDER BY cb.created_at DESC
                    LIMIT %s
                ''', (thread_id, limit))
                
            elif project:
                cursor.execute('''
                    SELECT cb.bookmark_id, cb.title, cb.notes, cb.bookmark_type, cb.created_at,
                           cb.chat_id, ct.user_input, ct.project, ct.response_data
                    FROM conversation_bookmarks cb
                    JOIN chat_threads ct ON cb.chat_id = ct.id
                    WHERE ct.project = %s
                    ORDER BY cb.created_at DESC
                    LIMIT %s
                ''', (project, limit))
                
            else:
                cursor.execute('''
                    SELECT cb.bookmark_id, cb.title, cb.notes, cb.bookmark_type, cb.created_at,
                           cb.chat_id, ct.user_input, ct.project, ct.response_data
                    FROM conversation_bookmarks cb
                    JOIN chat_threads ct ON cb.chat_id = ct.id
                    ORDER BY cb.created_at DESC
                    LIMIT %s
                ''', (limit,))
            
            bookmarks = cursor.fetchall()
            print(f"✅ Retrieved {len(bookmarks)} bookmarks from database")
            return bookmarks
            
        except Exception as e:
            print(f"❌ Failed to get bookmarks: {e}")
            return []

def delete_bookmark(bookmark_id: str) -> bool:
    """Delete a bookmark by ID"""
    with get_db_connection() as conn:
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM conversation_bookmarks 
                WHERE bookmark_id = %s
                RETURNING bookmark_id
            ''', (bookmark_id,))
            
            deleted = cursor.fetchone()
            conn.commit()
            
            if deleted:
                print(f"✅ Successfully deleted bookmark {bookmark_id}")
                return True
            else:
                print(f"❌ Bookmark {bookmark_id} not found")
                return False
                
        except Exception as e:
            print(f"❌ Failed to delete bookmark: {e}")
            conn.rollback()
            return False

def update_bookmark(bookmark_id: str, title: str = None, notes: str = None) -> bool:
    """Update bookmark title and/or notes"""
    with get_db_connection() as conn:
        if not conn:
            return False
        
        try:
            cursor = conn.cursor()
            
            # Build update query dynamically
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = %s")
                params.append(title)
            
            if notes is not None:
                updates.append("notes = %s")
                params.append(notes)
            
            if not updates:
                print("No updates provided")
                return False
            
            # Add bookmark_id as last parameter
            params.append(bookmark_id)
            
            cursor.execute(f'''
                UPDATE conversation_bookmarks 
                SET {', '.join(updates)}
                WHERE bookmark_id = %s
                RETURNING bookmark_id
            ''', params)
            
            updated = cursor.fetchone()
            conn.commit()
            
            if updated:
                print(f"✅ Successfully updated bookmark {bookmark_id}")
                return True
            else:
                print(f"❌ Bookmark {bookmark_id} not found")
                return False
                
        except Exception as e:
            print(f"❌ Failed to update bookmark: {e}")
            conn.rollback()
            return False

def get_bookmark_by_id(bookmark_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific bookmark by ID with full details"""
    with get_db_connection() as conn:
        if not conn:
            return None
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('''
                SELECT cb.bookmark_id, cb.title, cb.notes, cb.bookmark_type, cb.created_at,
                       cb.chat_id, cb.thread_id, ct.user_input, ct.response_data, ct.project,
                       ctm.title as thread_title
                FROM conversation_bookmarks cb
                JOIN chat_threads ct ON cb.chat_id = ct.id
                LEFT JOIN chat_thread_metadata ctm ON cb.thread_id = ctm.thread_id
                WHERE cb.bookmark_id = %s
            ''', (bookmark_id,))
            
            bookmark = cursor.fetchone()
            
            if bookmark:
                print(f"✅ Retrieved bookmark {bookmark_id}: {bookmark['title']}")
                return bookmark
            else:
                print(f"❌ Bookmark {bookmark_id} not found")
                return None
                
        except Exception as e:
            print(f"❌ Failed to get bookmark by ID: {e}")
            return None

def get_bookmark_stats(project: str = None) -> Dict[str, int]:
    """Get bookmark statistics"""
    with get_db_connection() as conn:
        if not conn:
            return {'total': 0, 'by_type': {}}
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Base query for total count
            if project:
                cursor.execute('''
                    SELECT COUNT(*) as total
                    FROM conversation_bookmarks cb
                    JOIN chat_threads ct ON cb.chat_id = ct.id
                    WHERE ct.project = %s
                ''', (project,))
            else:
                cursor.execute('SELECT COUNT(*) as total FROM conversation_bookmarks')
            
            total_count = cursor.fetchone()['total']
            
            # Get counts by type
            if project:
                cursor.execute('''
                    SELECT cb.bookmark_type, COUNT(*) as count
                    FROM conversation_bookmarks cb
                    JOIN chat_threads ct ON cb.chat_id = ct.id
                    WHERE ct.project = %s
                    GROUP BY cb.bookmark_type
                ''', (project,))
            else:
                cursor.execute('''
                    SELECT bookmark_type, COUNT(*) as count
                    FROM conversation_bookmarks
                    GROUP BY bookmark_type
                ''')
            
            by_type = {row['bookmark_type']: row['count'] for row in cursor.fetchall()}
            
            return {
                'total': total_count,
                'by_type': by_type,
                'project': project
            }
            
        except Exception as e:
            print(f"❌ Failed to get bookmark stats: {e}")
            return {'total': 0, 'by_type': {}}

def auto_create_bookmarks():
    """Automatically create bookmarks for important conversations"""
    important_patterns = [
        "error", "fix", "solution", "resolved", "completed", "milestone",
        "decision", "important", "reminder", "deadline", "meeting"
    ]
    
    with get_db_connection() as conn:
        if not conn:
            return
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Find recent conversations that might be important
            cursor.execute('''
                SELECT id, user_input, response_data, project, created_at
                FROM chat_threads
                WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
                  AND id NOT IN (SELECT chat_id FROM conversation_bookmarks WHERE chat_id IS NOT NULL)
                ORDER BY created_at DESC
                LIMIT 50
            ''')
            
            rows = cursor.fetchall()
            bookmarks_created = 0
            
            for row in rows:
                text_content = (row['user_input'] + ' ' +
                              (row['response_data'].get('SyntaxPrime', '') if row['response_data'] else '')).lower()
                
                for pattern in important_patterns:
                    if pattern in text_content:
                        # Create auto-bookmark
                        bookmark_title = f"Auto: {pattern.title()} - {row['user_input'][:50]}..."
                        
                        result = create_bookmark(
                            chat_id=row['id'],
                            title=bookmark_title,
                            notes=f"Auto-created for keyword: {pattern}",
                            bookmark_type='auto'
                        )
                        
                        if result:
                            bookmarks_created += 1
                        break  # Only create one bookmark per conversation
            
            print(f"✅ Auto-created {bookmarks_created} bookmarks")
            
        except Exception as e:
            print(f"❌ Auto-bookmark creation failed: {e}")

def search_bookmarks(query: str, project: str = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Search bookmarks by title, notes, or conversation content"""
    with get_db_connection() as conn:
        if not conn:
            return []
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Use ILIKE for case-insensitive search
            search_pattern = f'%{query.lower()}%'
            
            if project:
                cursor.execute('''
                    SELECT cb.bookmark_id, cb.title, cb.notes, cb.bookmark_type, cb.created_at,
                           cb.chat_id, ct.user_input, ct.project, ct.response_data
                    FROM conversation_bookmarks cb
                    JOIN chat_threads ct ON cb.chat_id = ct.id
                    WHERE ct.project = %s
                      AND (LOWER(cb.title) LIKE %s 
                           OR LOWER(cb.notes) LIKE %s 
                           OR LOWER(ct.user_input) LIKE %s)
                    ORDER BY cb.created_at DESC
                    LIMIT %s
                ''', (project, search_pattern, search_pattern, search_pattern, limit))
            else:
                cursor.execute('''
                    SELECT cb.bookmark_id, cb.title, cb.notes, cb.bookmark_type, cb.created_at,
                           cb.chat_id, ct.user_input, ct.project, ct.response_data
                    FROM conversation_bookmarks cb
                    JOIN chat_threads ct ON cb.chat_id = ct.id
                    WHERE LOWER(cb.title) LIKE %s 
                       OR LOWER(cb.notes) LIKE %s 
                       OR LOWER(ct.user_input) LIKE %s
                    ORDER BY cb.created_at DESC
                    LIMIT %s
                ''', (search_pattern, search_pattern, search_pattern, limit))
            
            results = cursor.fetchall()
            print(f"✅ Found {len(results)} bookmarks matching '{query}'")
            return results
            
        except Exception as e:
            print(f"❌ Failed to search bookmarks: {e}")
            return []

# ==========================================
# Section 10: Database Maintenance and Migration
# ==========================================

def save_daily_log_enhanced(sync_type: str, content: str):
    """Save daily log to database with enhanced metadata"""
    today = datetime.datetime.now().date()
    
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO daily_logs (log_date, log_type, content)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (log_date, log_type) 
                    DO UPDATE SET content = EXCLUDED.content, created_at = CURRENT_TIMESTAMP
                ''', (today, sync_type, content))
                
                conn.commit()
                print(f"Daily log saved to database: {sync_type}")
                
            except Exception as e:
                print(f"Failed to save daily log to database: {e}")
                conn.rollback()

def track_uploaded_file(filename: str, file_type: str, project: str, content_preview: str = ""):
    """Track uploaded files in database with enhanced metadata"""
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO uploaded_files (filename, file_type, project, content_preview)
                    VALUES (%s, %s, %s, %s)
                ''', (filename, file_type, project, content_preview[:500]))
                
                conn.commit()
                print(f"File upload tracked: {filename}")
                
            except Exception as e:
                print(f"Failed to track uploaded file: {e}")
                conn.rollback()

def save_brain_to_database(corpus_data):
    """Save processed brain corpus to database with smart filtering"""
    with get_db_connection() as conn:
        if not conn:
            print("No database connection - brain will only be saved to file")
            return False
            
        try:
            cursor = conn.cursor()
            
            # Clear existing brain data
            cursor.execute('DELETE FROM brain_documents')
            print("Cleared existing brain documents from database")
            
            # Insert new brain data in batches with smart filtering
            saved_count = 0
            filtered_count = 0
            batch_size = 100
            
            for i in range(0, len(corpus_data), batch_size):
                batch = corpus_data[i:i + batch_size]
                
                for item in batch:
                    content = item.get('content', '')
                    
                    # Apply smart filtering during save process
                    if should_include_content(content, "knowledge_base"):
                        cursor.execute('''
                            INSERT INTO brain_documents (document_id, title, content, chunk_index, metadata)
                            VALUES (%s, %s, %s, %s, %s)
                        ''', (
                            item.get('id', 'unknown'),
                            item.get('title', '')[:500],
                            content,
                            item.get('chunk_index', 0),
                            psycopg2.extras.Json({
                                **item.get('metadata', {}),
                                'content_tier': get_content_tier(content)
                            })
                        ))
                        saved_count += 1
                    else:
                        filtered_count += 1
                
                # Commit each batch
                conn.commit()
                print(f"Saved batch {i//batch_size + 1}: {saved_count} saved, {filtered_count} filtered")
            
            # Update brain health
            update_brain_health(results_count=saved_count)
            
            print(f"Successfully saved {saved_count} brain documents to database ({filtered_count} filtered out)")
            return True
            
        except Exception as e:
            print(f"Failed to save brain to database: {e}")
            conn.rollback()
            update_brain_health(error=str(e))
            return False

def load_brain_from_database():
    """Load brain corpus from database"""
    with get_db_connection() as conn:
        if not conn:
            return None
            
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute('''
                SELECT document_id, title, content, chunk_index, metadata 
                FROM brain_documents 
                ORDER BY document_id, chunk_index
            ''')
            
            rows = cursor.fetchall()
            corpus_data = []
            
            for row in rows:
                corpus_data.append({
                    'id': row['document_id'],
                    'title': row['title'],
                    'content': row['content'],
                    'chunk_index': row['chunk_index'],
                    'metadata': row['metadata'] or {}
                })
            
            print(f"Loaded {len(corpus_data)} brain documents from database")
            return corpus_data
            
        except Exception as e:
            print(f"Failed to load brain from database: {e}")
            return None

def init_database():
    """Create necessary database tables with enhanced thread and bookmark support"""
    if not DATABASE_URL:
        print("No database URL - running in file-only mode")
        return
    
    with get_db_connection() as conn:
        if not conn:
            return
            
        cursor = conn.cursor()
        
        try:
            print("Initializing database tables...")
            
            # Create chat_threads table (should already exist)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_threads (
                    id SERIAL PRIMARY KEY,
                    project VARCHAR(100) NOT NULL,
                    user_input TEXT NOT NULL,
                    response_data JSONB NOT NULL,
                    context_project VARCHAR(100),
                    context_data JSONB DEFAULT '{}',
                    thread_id UUID,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create thread metadata table (should already exist)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_thread_metadata (
                    thread_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title VARCHAR(500),
                    project VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    tags TEXT[],
                    is_archived BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create bookmarks table (should already exist)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversation_bookmarks (
                    bookmark_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chat_id INTEGER REFERENCES chat_threads(id),
                    thread_id UUID REFERENCES chat_thread_metadata(thread_id),
                    title VARCHAR(300) NOT NULL,
                    notes TEXT,
                    bookmark_type VARCHAR(50) DEFAULT 'manual',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create other essential tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(50) NOT NULL,
                    content_preview TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    project VARCHAR(100) NOT NULL,
                    processing_status VARCHAR(50) DEFAULT 'completed'
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_logs (
                    id SERIAL PRIMARY KEY,
                    log_date DATE NOT NULL,
                    log_type VARCHAR(50) NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(log_date, log_type)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS brain_documents (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) NOT NULL,
                    title VARCHAR(500),
                    content TEXT NOT NULL,
                    embedding_vector FLOAT8[] NULL,
                    chunk_index INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS brain_health (
                    id SERIAL PRIMARY KEY,
                    last_refresh TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_documents INTEGER DEFAULT 0,
                    last_search_query TEXT,
                    last_search_results INTEGER DEFAULT 0,
                    health_status VARCHAR(50) DEFAULT 'healthy',
                    error_log TEXT
                )
            ''')
            
            # Ensure all indexes exist (should already be created)
            indexes = [
                'CREATE INDEX IF NOT EXISTS idx_chat_threads_project_date ON chat_threads (project, created_at DESC)',
                'CREATE INDEX IF NOT EXISTS idx_chat_threads_context_project ON chat_threads (context_project, created_at DESC)',
                'CREATE INDEX IF NOT EXISTS idx_chat_threads_thread_id ON chat_threads (thread_id)',
                'CREATE INDEX IF NOT EXISTS idx_thread_metadata_project ON chat_thread_metadata (project, created_at DESC)',
                'CREATE INDEX IF NOT EXISTS idx_bookmarks_thread_id ON conversation_bookmarks (thread_id)',
                'CREATE INDEX IF NOT EXISTS idx_bookmarks_chat_id ON conversation_bookmarks (chat_id)',
                'CREATE INDEX IF NOT EXISTS idx_brain_docs_content_fts ON brain_documents USING gin(to_tsvector(\'english\', content))',
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except Exception as idx_e:
                    print(f"Index creation note: {idx_e}")
            
            conn.commit()
            print("Database tables and indexes verified successfully")
            
        except Exception as e:
            conn.rollback()
            print(f"Database initialization failed: {e}")

def get_database_status():
    """Check database connection and enhanced table status"""
    status = {
        "database_url_configured": bool(DATABASE_URL),
        "connection_working": False,
        "tables_exist": False,
        "conversation_count": 0,
        "thread_count": 0,
        "bookmark_count": 0,
        "brain_documents_count": 0,
        "search_indexes_working": False,
        "brain_health": None
    }
    
    with get_db_connection() as conn:
        if conn:
            try:
                cursor = conn.cursor()
                status["connection_working"] = True
                
                # Test search functionality
                try:
                    test_results = enhanced_context_search("miller", k=1)
                    status["search_indexes_working"] = len(test_results) > 0
                except:
                    status["search_indexes_working"] = False
                
                # Check if tables exist
                cursor.execute('''
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name IN ('chat_threads', 'chat_thread_metadata', 'conversation_bookmarks', 
                                         'brain_documents', 'brain_health')
                ''')
                table_count = cursor.fetchone()[0]
                status["tables_exist"] = table_count >= 5
                
                if status["tables_exist"]:
                    # Get record counts
                    cursor.execute('SELECT COUNT(*) FROM chat_threads')
                    status["conversation_count"] = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM chat_thread_metadata')
                    status["thread_count"] = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM conversation_bookmarks')
                    status["bookmark_count"] = cursor.fetchone()[0]
                    
                    cursor.execute('SELECT COUNT(*) FROM brain_documents')
                    status["brain_documents_count"] = cursor.fetchone()[0]
                
                # Get brain health status
                status["brain_health"] = get_brain_health_status()
                
            except Exception as e:
                print(f"Database status check failed: {e}")
    
    return status

# ==========================================
# Section 11: Testing and Debugging Functions
# ==========================================

def test_miller_search():
    """Test function to verify Miller search is working properly"""
    print("\n" + "="*50)
    print("TESTING MILLER SEARCH FUNCTIONALITY")
    print("="*50)
    
    test_queries = [
        "Miller cat",
        "Miller",
        "tux cat",
        "my cat Miller",
        "Tell me about Miller"
    ]
    
    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        print("-" * 30)
        
        try:
            results = enhanced_context_search(query, k=3)
            
            if results:
                print(f"✅ Found {len(results)} results")
                for i, result in enumerate(results):
                    print(f"  {i+1}. {result['source']} (score: {result['score']:.2f})")
                    print(f"     Type: {result['metadata'].get('type', 'unknown')}")
                    if 'miller' in result['text'].lower():
                        print(f"     ✅ Contains Miller context")
                    else:
                        print(f"     ⚠️  No Miller context found")
            else:
                print(f"❌ No results found")
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
    
    print("\n" + "="*50)
    print("MILLER SEARCH TEST COMPLETE")
    print("="*50)

def debug_database_indexes():
    """Debug function to check database indexes"""
    with get_db_connection() as conn:
        if not conn:
            print("No database connection")
            return
        
        try:
            cursor = conn.cursor()
            
            print("\nChecking database indexes:")
            print("-" * 40)
            
            # Check for full-text search indexes
            cursor.execute('''
                SELECT indexname, indexdef 
                FROM pg_indexes 
                WHERE tablename = 'chat_threads' 
                AND indexdef LIKE '%gin%'
            ''')
            
            indexes = cursor.fetchall()
            
            if indexes:
                print("✅ Full-text search indexes found:")
                for idx_name, idx_def in indexes:
                    print(f"  - {idx_name}")
            else:
                print("❌ No full-text search indexes found")
            
            # Test a simple search
            cursor.execute('''
                SELECT COUNT(*) FROM chat_threads 
                WHERE to_tsvector('english', user_input || ' ' || 
                      COALESCE(response_data->>'SyntaxPrime', '')) 
                      @@ plainto_tsquery('english', 'miller')
            ''')
            
            count = cursor.fetchone()[0]
            print(f"✅ Full-text search test: {count} Miller conversations found")
            
        except Exception as e:
            print(f"❌ Index check failed: {e}")

if __name__ == '__main__':
    # Run tests when module is executed directly
    print("Testing enhanced database functionality...")
    
    # Test connection
    connected, msg = test_database_connection()
    print(f"Database connection: {msg}")
    
    if connected:
        # Debug indexes
        debug_database_indexes()
        
        # Test Miller search
        test_miller_search()
        
        # Show status
        status = get_database_status()
        print(f"\nDatabase Status Summary:")
        print(f"- Conversations: {status['conversation_count']}")
        print(f"- Threads: {status['thread_count']}")
        print(f"- Bookmarks: {status['bookmark_count']}")
        print(f"- Search working: {status['search_indexes_working']}")
