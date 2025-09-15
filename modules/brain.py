# modules/brain.py - FIXED Memory and Retrieval System
# Restored working Miller/Ghada memory with proper personality integration
# Sectioned for easy maintenance and debugging

#-------------------------------------------------------------------
# SECTION 1: IMPORTS AND CONFIGURATION
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 1: IMPORTS AND CONFIGURATION
#-------------------------------------------------------------------

import os
import datetime
import threading
import json
from flask import jsonify
from psycopg2.extras import RealDictCursor
import psycopg2
from contextlib import contextmanager

# Database connection - FIXED VERSION
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

@contextmanager
def get_db_connection():
    """FIXED: Database connection without retry loop generator issues"""
    conn = None
    try:
        if DATABASE_URL:
            conn = psycopg2.connect(
                DATABASE_URL,
                connect_timeout=15,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=5,
                keepalives_count=3
            )
            yield conn
        else:
            print("No DATABASE_URL configured")
            yield None
    except Exception as e:
        print(f"Database connection failed: {e}")
        yield None
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

# Import other database functions we need
try:
    from modules.database import (
        get_database_status,
        update_brain_health
    )
except ImportError as e:
    print(f"Database import warning: {e}")
    
    # Fallback functions if imports fail
    def get_database_status():
        return {"status": "import_failed"}
    
    def update_brain_health(*args, **kwargs):
        pass

#-------------------------------------------------------------------
# SECTION 2: GLOBAL STATE AND CONSTANTS
#-------------------------------------------------------------------

# Global brain system state
_brain_building = False
_brain_build_error = None
_last_brain_refresh = None

# File paths
CORPUS_PATH = "data/cleaned/ghostline_sources.jsonl.gz"

# Miller and Ghada are family - these should ALWAYS be found
PERSONAL_KEYWORDS = [
    'miller', 'ghada', 'shazeen', 'mom', 'mother', 'family',
    'daughter', 'wife', 'cat', 'tux cat', 'tuxedo cat'
]

#-------------------------------------------------------------------
# SECTION 3: CORE BRAIN SYSTEM CLASS
#-------------------------------------------------------------------

class SimpleBrainSystem:
    """Simple, working brain system that actually finds Miller and Ghada"""
    
    def __init__(self):
        self.ready = False
        self.conversation_count = 0
        self.document_count = 0
        self._check_system_status()
    
    def _check_system_status(self):
        """Check if we have conversations and documents available"""
        try:
            # Check database status
            db_status = get_database_status()
            self.conversation_count = db_status.get('conversation_count', 0)
            self.document_count = db_status.get('brain_documents_count', 0)
            
            # We're ready if we have conversations OR documents
            self.ready = (self.conversation_count > 0) or (self.document_count > 0)
            
            print(f"Brain system status: {self.conversation_count} conversations, {self.document_count} documents")
            
            if self.ready:
                print("Brain system ready for retrieval")
            else:
                print("Brain system not ready - no data found")
                
        except Exception as e:
            print(f"Failed to check brain system status: {e}")
            self.ready = False
    
    def get_status(self):
        """Get system status for monitoring"""
        return {
            "ready": self.ready,
            "conversation_count": self.conversation_count,
            "document_count": self.document_count,
            "status": "ready" if self.ready else "no_data",
            "method": "direct_database_simple"
        }

#-------------------------------------------------------------------
# SECTION 4: ENHANCED RETRIEVAL FUNCTIONS - THE CORE FIX
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 4: ENHANCED RETRIEVAL FUNCTIONS - THE CORE FIX 9/15/25
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 4: ENHANCED RETRIEVAL FUNCTIONS - THE CORE FIX
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 4: ENHANCED RETRIEVAL FUNCTIONS - THE CORE FIX 9/15/25
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 4: ENHANCED RETRIEVAL FUNCTIONS - THE CORE FIX 9/15/25
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 4: ENHANCED RETRIEVAL FUNCTIONS - THE CORE FIX 9/15/25
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 4: ENHANCED RETRIEVAL FUNCTIONS - THE CORE FIX 9/15/25
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 4: ENHANCED RETRIEVAL FUNCTIONS - THE CORE FIX 9/15/25 - MILLER MEMORY RESTORED
#-------------------------------------------------------------------

def enhanced_retrieve(query_text, k=5, project=None):
    """
    FIXED: Enhanced retrieval that actually finds Miller and Ghada
    This function directly queries the working database with proper SQL
    """
    print(f"Enhanced retrieve: searching for '{query_text}' (limit: {k})")
    
    all_results = []
    
    # STEP 1: Search conversation history for personal context
    try:
        conversation_results = _search_conversation_memory(query_text, k)
        all_results.extend(conversation_results)
        print(f"Found {len(conversation_results)} conversation results")
        
    except Exception as e:
        print(f"Conversation search failed: {e}")
    
    # STEP 2: Search brain documents if we need more results
    remaining_slots = k - len(all_results)
    if remaining_slots > 0:
        try:
            document_results = _search_brain_documents(query_text, remaining_slots)
            all_results.extend(document_results)
            print(f"Found {len(document_results)} document results")
            
        except Exception as e:
            print(f"Document search failed: {e}")
    
    # STEP 3: Sort results by relevance (conversation history first)
    all_results.sort(key=lambda x: (
        0 if x.get('source_type') == 'conversation' else 1,
        -x.get('relevance', 0)
    ))
    
    # Return top results
    final_results = all_results[:k]
    
    print(f"Enhanced retrieve returning {len(final_results)} total results:")
    for i, result in enumerate(final_results, 1):
        source_type = result.get('source_type', 'unknown')
        source = result.get('source', 'unknown')[:60]
        print(f"  {i}. {source_type}: {source}")
    
    # Update health tracking
    try:
        update_brain_health(
            query=query_text[:100],
            results_count=len(final_results)
        )
    except Exception as e:
        print(f"Health tracking failed: {e}")
    
    return final_results

def _search_conversation_memory(query_text, k=5):
    """FIXED: Search using proper parameterized queries to avoid SQL errors"""
    
    print(f"Searching conversation memory for: '{query_text}' (limit: {k})")
    
    # Validate inputs
    if not query_text or not isinstance(query_text, str):
        print("Invalid query input")
        return []
    
    if k <= 0:
        k = 5
    
    try:
        if not DATABASE_URL:
            print("No DATABASE_URL configured")
            return []
        
        # Use direct connection to avoid context manager issues
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
        cursor = conn.cursor()
        
        # CRITICAL FIX: Use proper parameterized queries
        search_term = f"%{query_text.lower()}%"
        
        sql = """
            SELECT user_input, response_data, project, created_at, id
            FROM chat_threads 
            WHERE LOWER(user_input) LIKE %s
               OR LOWER(response_data->>'SyntaxPrime') LIKE %s
            ORDER BY created_at DESC
            LIMIT %s
        """
        
        # Execute with proper parameters (no string formatting!)
        cursor.execute(sql, (search_term, search_term, k * 3))
        rows = cursor.fetchall()
        
        print(f"SQL executed successfully: {len(rows)} raw results")
        
        results = []
        for row in rows:
            try:
                # Safe tuple unpacking
                if len(row) >= 5:
                    user_input = row[0] or ''
                    response_data = row[1] or {}
                    project = row[2] or 'Unknown'
                    created_at = row[3]
                    chat_id = row[4]
                    
                    # Extract response safely
                    if isinstance(response_data, dict):
                        response = response_data.get('SyntaxPrime', '')
                    else:
                        response = ''
                    
                    # Skip empty conversations
                    if len(user_input.strip()) < 1:
                        continue
                    
                    # Verify content actually matches query
                    combined_text = f"{user_input} {response}".lower()
                    if query_text.lower() not in combined_text:
                        continue
                    
                    # Build result
                    conversation_content = f"CONVERSATION MEMORY:\nUser: {user_input}\nSyntax: {response}"
                    if len(conversation_content) > 2000:
                        conversation_content = conversation_content[:2000] + "..."
                    
                    # Personal content boost
                    personal_keywords = ['miller', 'ghada', 'shazeen', 'mom', 'family', 'daughter', 'wife', 'cat']
                    is_personal = any(kw in combined_text for kw in personal_keywords)
                    
                    result = {
                        'text': conversation_content,
                        'source': f"Memory - {project}",
                        'source_type': 'conversation',
                        'relevance': 0.98 if is_personal else 0.85,
                        'created_at': created_at,
                        'chat_id': chat_id
                    }
                    
                    # Add date if available
                    if created_at and hasattr(created_at, 'strftime'):
                        try:
                            date_str = created_at.strftime('%m/%d/%Y')
                            result['source'] = f"Memory - {project} ({date_str})"
                        except:
                            pass
                    
                    results.append(result)
                    
                    if len(results) >= k:
                        break
                        
            except Exception as row_error:
                print(f"Row processing error: {row_error}")
                continue
        
        conn.close()
        
        print(f"Conversation search returning {len(results)} results for '{query_text}'")
        
        # Debug output for Miller searches
        if 'miller' in query_text.lower():
            print(f"MILLER SEARCH DEBUG: Found {len(results)} results")
            for i, result in enumerate(results[:3], 1):
                preview = result['text'][:100].replace('\n', ' ')
                print(f"  {i}. {preview}...")
        
        return results
        
    except psycopg2.Error as db_error:
        print(f"PostgreSQL error in conversation search: {db_error}")
        print(f"Error code: {db_error.pgcode}")
        return []
    except Exception as e:
        print(f"Conversation search failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return []

def _search_brain_documents(query_text, k=5):
    """Search brain documents - simple version that finds actual matches"""
    
    try:
        if not DATABASE_URL:
            return []
        
        # Direct connection
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
        cursor = conn.cursor()
        
        # Search documents with parameterized query
        search_term = f"%{query_text.lower()}%"
        sql = """
            SELECT title, content, metadata
            FROM brain_documents 
            WHERE LOWER(content) LIKE %s
            AND LENGTH(TRIM(content)) > 20
            ORDER BY LENGTH(content) DESC
            LIMIT %s
        """
        
        cursor.execute(sql, (search_term, k))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            try:
                if len(row) >= 3:
                    title = row[0] or 'Untitled'
                    content = row[1] or ''
                    metadata = row[2] or {}
                    
                    # Verify content actually contains search term
                    if query_text.lower() not in content.lower():
                        continue
                    
                    if len(content.strip()) < 20:
                        continue
                    
                    # Limit content length
                    if len(content) > 1500:
                        content = content[:1500] + "..."
                    
                    result = {
                        'text': content,
                        'source': f"Knowledge - {title}",
                        'source_type': 'document',
                        'relevance': 0.7,
                        'metadata': metadata if isinstance(metadata, dict) else {}
                    }
                    
                    results.append(result)
                    
            except Exception as row_error:
                print(f"Document row error: {row_error}")
                continue
        
        conn.close()
        
        if results:
            print(f"Document search found {len(results)} results")
        
        return results
        
    except Exception as e:
        print(f"Document search failed: {e}")
        return []

#-------------------------------------------------------------------
# SECTION 5: BRAIN REFRESH AND MAINTENANCE
#-------------------------------------------------------------------

def refresh_brain_context():
    """Refresh brain context and check system health"""
    global _last_brain_refresh, _brain_system
    
    current_time = datetime.datetime.now()
    
    # Refresh every 30 minutes or on first run
    if (_last_brain_refresh is None or
        (current_time - _last_brain_refresh).total_seconds() > 1800):
        
        try:
            print("Refreshing brain context...")
            
            # Check system status
            _brain_system._check_system_status()
            _last_brain_refresh = current_time
            
            print(f"Brain refresh complete: {_brain_system.conversation_count} conversations available")
            
            # Test Miller search to ensure memory is working
            test_results = enhanced_retrieve("miller", k=2)
            if test_results:
                print(f"Miller memory test: SUCCESS ({len(test_results)} results)")
            else:
                print("Miller memory test: FAILED - no results found")
            
            # Update health status
            update_brain_health(
                query="refresh_context",
                results_count=_brain_system.conversation_count + _brain_system.document_count
            )
                
        except Exception as e:
            print(f"Brain refresh failed: {e}")
            update_brain_health(
                query="refresh_context",
                results_count=0,
                error=str(e)
            )

#-------------------------------------------------------------------
# SECTION 6: BRAIN BUILDING FUNCTIONS
#-------------------------------------------------------------------

def build_brain_from_corpus():
    """Build brain by processing corpus file and saving to database"""
    global _brain_building, _brain_build_error
    
    try:
        _brain_building = True
        _brain_build_error = None
        print("Starting brain build from corpus file...")
        
        # Check if corpus file exists
        if not os.path.exists(CORPUS_PATH):
            raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")
        
        # Import corpus processing
        import gzip
        
        print(f"Processing corpus file: {CORPUS_PATH}")
        
        corpus_data = []
        chunk_id = 0
        
        with gzip.open(CORPUS_PATH, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Extract text content
                    text_content = _extract_text_from_json(data)
                    
                    if text_content and len(text_content.strip()) > 50:
                        # Create chunks for long text
                        chunks = _chunk_text(text_content, max_words=400)
                        
                        for chunk_index, chunk_text in enumerate(chunks):
                            corpus_item = {
                                'id': f'corpus_{chunk_id}',
                                'title': f'line_{line_num + 1}_{chunk_index}',
                                'content': chunk_text,
                                'chunk_index': chunk_index,
                                'metadata': {
                                    'source': f'line_{line_num + 1}',
                                    'created_at': datetime.datetime.now().isoformat(),
                                    'chunk_index': chunk_index
                                }
                            }
                            corpus_data.append(corpus_item)
                            chunk_id += 1
                
                except json.JSONDecodeError:
                    continue
                
                # Progress update
                if line_num % 1000 == 0 and line_num > 0:
                    print(f"Processed {line_num} lines, created {len(corpus_data)} chunks")
        
        print(f"Corpus processing complete: {len(corpus_data)} total chunks")
        
        # Save to database
        if corpus_data:
            from modules.database import save_brain_to_database
            if save_brain_to_database(corpus_data):
                print("Brain successfully saved to database")
                update_brain_health(results_count=len(corpus_data))
                
                # Refresh system status
                _brain_system._check_system_status()
            else:
                raise Exception("Database save failed")
        else:
            raise Exception("No valid chunks created from corpus")
        
        _brain_building = False
        print("Brain build from corpus complete!")
        
    except Exception as e:
        _brain_building = False
        _brain_build_error = str(e)
        print(f"Brain build from corpus failed: {e}")
        update_brain_health(error=str(e))

def build_brain_from_sources():
    """Build new brain from raw sources"""
    global _brain_building, _brain_build_error
    
    try:
        _brain_building = True
        _brain_build_error = None
        print("Starting brain build from raw sources...")
        
        # Try to import the brain building module
        try:
            from build_brain_fixed2 import build_new_brain
            result_path = build_new_brain()
            print(f"New brain built at: {result_path}")
            
            # Copy to expected location if different
            if str(result_path) != CORPUS_PATH:
                import shutil
                shutil.copy(str(result_path), CORPUS_PATH)
                print(f"New brain copied to {CORPUS_PATH}")
            
        except ImportError:
            raise Exception("build_brain_fixed2 module not available")
        except Exception as e:
            raise Exception(f"Brain building from sources failed: {e}")
        
        # Process the new corpus file
        build_brain_from_corpus()
        
    except Exception as e:
        _brain_building = False
        _brain_build_error = str(e)
        print(f"Brain build from sources failed: {e}")
        update_brain_health(error=str(e))

#-------------------------------------------------------------------
# SECTION 7: TEXT PROCESSING UTILITIES
#-------------------------------------------------------------------

def _extract_text_from_json(json_obj):
    """Extract meaningful text from a JSON object"""
    texts = []
    text_fields = ['text', 'content', 'message', 'body', 'description', 'title', 'question', 'answer']
    
    def extract_recursive(obj, prefix=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key.lower() in text_fields and isinstance(value, str) and len(value.strip()) > 20:
                    context = f"[{prefix}{key}] " if prefix or key != 'text' else ""
                    texts.append(context + value.strip())
                elif isinstance(value, (dict, list)):
                    extract_recursive(value, f"{prefix}{key}.")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    extract_recursive(item, f"{prefix}[{i}].")
                elif isinstance(item, str) and len(item.strip()) > 20:
                    texts.append(item.strip())
    
    extract_recursive(json_obj)
    return " ".join(texts) if texts else ""

def _chunk_text(text, max_words=400):
    """Break text into smaller chunks for processing"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
    
    return chunks if chunks else [text]

#-------------------------------------------------------------------
# SECTION 8: DIAGNOSTIC AND TESTING FUNCTIONS
#-------------------------------------------------------------------
#-------------------------------------------------------------------
# SECTION 8: DIAGNOSTIC AND TESTING FUNCTIONS 9/15/25
#-------------------------------------------------------------------

def debug_miller_search():
    """Debug function to test Miller search step by step - CRITICAL FIX"""
    print("\n" + "="*60)
    print("DEBUGGING MILLER SEARCH - STEP BY STEP")
    print("="*60)
    
    # Test 1: Database connection
    print("\n1. Testing database connection...")
    try:
        with get_db_connection() as conn:
            if conn:
                print("✅ Database connection: SUCCESS")
                
                # Test 2: Raw SQL query
                print("\n2. Testing raw SQL query...")
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM chat_threads 
                    WHERE LOWER(user_input) LIKE '%miller%' 
                       OR LOWER(response_data->>'SyntaxPrime') LIKE '%miller%'
                """)
                count = cursor.fetchone()[0]
                print(f"✅ Raw SQL found {count} Miller conversations")
                
                # Test 3: Get sample conversation
                print("\n3. Getting sample Miller conversation...")
                cursor.execute("""
                    SELECT user_input, response_data->>'SyntaxPrime' as response, project, created_at
                    FROM chat_threads 
                    WHERE LOWER(user_input) LIKE '%miller%' 
                       OR LOWER(response_data->>'SyntaxPrime') LIKE '%miller%'
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row:
                    print(f"✅ Sample conversation found:")
                    print(f"   User: {row[0][:100]}...")
                    print(f"   Response: {row[1][:100] if row[1] else 'None'}...")
                    print(f"   Project: {row[2]}")
                    print(f"   Date: {row[3]}")
                else:
                    print("❌ No sample conversation found")
                    
            else:
                print("❌ Database connection: FAILED")
                
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Test the fixed search function directly
    print("\n4. Testing fixed search function...")
    try:
        results = _search_conversation_memory("miller", k=2)
        if results:
            print(f"✅ Fixed search function found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result['source']}")
                print(f"      Preview: {result['text'][:150]}...")
        else:
            print("❌ Fixed search function returned no results")
    except Exception as e:
        print(f"❌ Fixed search function failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Test full enhanced_retrieve
    print("\n5. Testing full enhanced_retrieve...")
    try:
        results = enhanced_retrieve("miller", k=2)
        if results:
            print(f"✅ Enhanced retrieve found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result.get('source', 'Unknown source')}")
        else:
            print("❌ Enhanced retrieve returned no results")
    except Exception as e:
        print(f"❌ Enhanced retrieve failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Test the ghostline_engine integration
    print("\n6. Testing ghostline_engine generate_response...")
    try:
        from utils.ghostline_engine import generate_response
        response_data = generate_response(
            user_input="Who is Miller?",
            use_voices=['SyntaxPrime'],
            random_toggle=False,
            project='Personal Operating Manual'
        )
        
        if response_data and 'SyntaxPrime' in response_data:
            response_text = response_data['SyntaxPrime']
            if "trouble processing" in response_text:
                print("❌ Ghostline engine returning generic error message")
                print(f"   Response: {response_text[:200]}...")
            else:
                print("✅ Ghostline engine generated real response")
                print(f"   Response: {response_text[:200]}...")
        else:
            print("❌ Ghostline engine returned no SyntaxPrime response")
            print(f"   Full response: {response_data}")
            
    except Exception as e:
        print(f"❌ Ghostline engine test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE - CHECK RESULTS ABOVE")
    print("="*60)

def get_brain_diagnostics():
    """Get comprehensive brain system diagnostics"""
    diagnostics = {
        "system_status": {},
        "database_status": {},
        "conversation_tests": {},
        "miller_test": {},
        "ghada_test": {},
        "file_system": {}
    }
    
    # System status
    try:
        global _brain_system
        diagnostics["system_status"] = _brain_system.get_status()
    except Exception as e:
        diagnostics["system_status"]["error"] = str(e)
    
    # Database status
    try:
        from modules.database import get_database_status
        diagnostics["database_status"] = get_database_status()
    except Exception as e:
        diagnostics["database_status"]["error"] = str(e)
    
    # Test Miller memory specifically
    try:
        miller_results = enhanced_retrieve("miller", k=3)
        diagnostics["miller_test"] = {
            "query": "miller",
            "results_found": len(miller_results),
            "has_conversation_results": any(r.get('source_type') == 'conversation' for r in miller_results),
            "sample_sources": [r.get('source', 'unknown')[:50] for r in miller_results[:2]]
        }
    except Exception as e:
        diagnostics["miller_test"]["error"] = str(e)
    
    # Test Ghada memory specifically
    try:
        ghada_results = enhanced_retrieve("ghada", k=3)
        diagnostics["ghada_test"] = {
            "query": "ghada",
            "results_found": len(ghada_results),
            "has_conversation_results": any(r.get('source_type') == 'conversation' for r in ghada_results),
            "sample_sources": [r.get('source', 'unknown')[:50] for r in ghada_results[:2]]
        }
    except Exception as e:
        diagnostics["ghada_test"]["error"] = str(e)
    
    # Database conversation counts
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                
                # Count conversations mentioning key people
                test_names = ['miller', 'ghada', 'shazeen']
                for name in test_names:
                    cursor.execute(
                        "SELECT COUNT(*) FROM chat_threads WHERE LOWER(user_input) LIKE %s OR LOWER(response_data->>'SyntaxPrime') LIKE %s",
                        (f'%{name}%', f'%{name}%')
                    )
                    count = cursor.fetchone()[0]
                    diagnostics["conversation_tests"][f"{name}_mentions"] = count
                    
    except Exception as e:
        diagnostics["conversation_tests"]["error"] = str(e)
    
    # File system status
    try:
        diagnostics["file_system"]["corpus_exists"] = os.path.exists(CORPUS_PATH)
        if os.path.exists(CORPUS_PATH):
            diagnostics["file_system"]["corpus_size"] = os.path.getsize(CORPUS_PATH)
    except Exception as e:
        diagnostics["file_system"]["error"] = str(e)
    
    return diagnostics

def test_miller_memory_directly():
    """Direct test function for Miller memory"""
    print("=== TESTING MILLER MEMORY DIRECTLY ===")
    
    try:
        results = enhanced_retrieve("miller", k=3)
        print(f"Miller search results: {len(results)}")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Source: {result.get('source', 'unknown')}")
                print(f"  Type: {result.get('source_type', 'unknown')}")
                print(f"  Preview: {result.get('text', '')[:150]}...")
            return True
        else:
            print("No Miller results found!")
            return False
            
    except Exception as e:
        print(f"Miller test failed: {e}")
        return False

def test_ghada_memory_directly():
    """Direct test function for Ghada memory"""
    print("=== TESTING GHADA MEMORY DIRECTLY ===")
    
    try:
        results = enhanced_retrieve("ghada", k=3)
        print(f"Ghada search results: {len(results)}")
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"  Source: {result.get('source', 'unknown')}")
                print(f"  Type: {result.get('source_type', 'unknown')}")
                print(f"  Preview: {result.get('text', '')[:150]}...")
            return True
        else:
            print("No Ghada results found!")
            return False
            
    except Exception as e:
        print(f"Ghada test failed: {e}")
        return False

def test_search_integration():
    """Test the complete search integration pipeline"""
    print("=== TESTING COMPLETE SEARCH INTEGRATION ===")
    
    test_queries = [
        ("miller", "Should find your tux cat"),
        ("ghada", "Should find your wife"),
        ("shazeen", "Should find your daughter"),
        ("coffee", "Should find morning routine references"),
        ("2am", "Should find late night coding sessions")
    ]
    
    for query, description in test_queries:
        print(f"\nTesting: '{query}' - {description}")
        try:
            results = enhanced_retrieve(query, k=2)
            if results:
                print(f"  ✅ Found {len(results)} results")
                for result in results:
                    source_type = result.get('source_type', 'unknown')
                    print(f"    - {source_type}: {result.get('source', 'unknown')[:60]}")
            else:
                print(f"  ❌ No results found")
        except Exception as e:
            print(f"  ❌ Search failed: {e}")
    
    print("\n=== SEARCH INTEGRATION TEST COMPLETE ===")

def comprehensive_memory_test():
    """Run all memory tests to verify system is working"""
    print("\n" + "="*80)
    print("COMPREHENSIVE MEMORY TEST - FULL SYSTEM CHECK")
    print("="*80)
    
    # Test 1: Database connectivity
    print("\n1. DATABASE CONNECTIVITY TEST")
    try:
        with get_db_connection() as conn:
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM chat_threads")
                total_conversations = cursor.fetchone()[0]
                print(f"✅ Database connected: {total_conversations} total conversations")
            else:
                print("❌ Database connection failed")
                return False
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False
    
    # Test 2: Personal memory searches
    print("\n2. PERSONAL MEMORY TESTS")
    personal_tests = [
        ("miller", 272),  # You confirmed 272 Miller conversations exist
        ("ghada", 50),    # Estimate
        ("shazeen", 30)   # Estimate
    ]
    
    for name, expected_min in personal_tests:
        try:
            results = enhanced_retrieve(name, k=3)
            print(f"   {name}: Found {len(results)} results (expected >= 1)")
            if len(results) > 0:
                print(f"      ✅ Memory working for {name}")
            else:
                print(f"      ❌ No memory found for {name}")
        except Exception as e:
            print(f"      ❌ Search failed for {name}: {e}")
    
    # Test 3: System integration
    print("\n3. SYSTEM INTEGRATION TEST")
    test_search_integration()
    
    # Test 4: Brain system status
    print("\n4. BRAIN SYSTEM STATUS")
    try:
        status = get_brain_status()
        print(f"   Ready: {status['ready']}")
        print(f"   Conversations: {status['conversations']}")
        print(f"   Documents: {status['documents']}")
        print(f"   Miller Memory: {status.get('miller_memory_working', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Status check failed: {e}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MEMORY TEST COMPLETE")
    print("="*80)

#-------------------------------------------------------------------
# SECTION 9: CONTROL ENDPOINTS AND STATUS FUNCTIONS
#-------------------------------------------------------------------

def handle_build_brain(session):
    """Handle brain building from corpus file"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _brain_building
    
    if _brain_building:
        return jsonify({"ok": False, "error": "Brain is already building"}), 400
    
    # Start building in background
    thread = threading.Thread(target=build_brain_from_corpus)
    thread.daemon = True
    thread.start()
    
    return jsonify({"ok": True, "message": "Brain building from corpus started"})

def handle_build_new_brain(session):
    """Handle building new brain from raw sources"""
    if not session.get('logged_in'):
        return "Unauthorized", 401
    
    global _brain_building
    
    if _brain_building:
        return jsonify({"ok": False, "error": "Brain is already building"}), 400
    
    # Start building in background
    thread = threading.Thread(target=build_brain_from_sources)
    thread.daemon = True
    thread.start()
    
    return jsonify({"ok": True, "message": "Brain building from sources started"})

def get_brain_status():
    """Get comprehensive brain status for monitoring"""
    global _brain_building, _brain_build_error, _brain_system
    
    # Get brain system status
    brain_status = _brain_system.get_status()
    
    # Test if Miller memory is working
    miller_working = False
    try:
        miller_results = enhanced_retrieve("miller", k=1)
        miller_working = len(miller_results) > 0
    except:
        pass
    
    status = {
        "ready": brain_status["ready"] and miller_working,
        "building": _brain_building,
        "progress": "Building brain..." if _brain_building else (
            f"Ready: {brain_status['conversation_count']} conversations, {brain_status['document_count']} documents" if brain_status["ready"]
            else "No data available"
        ),
        "error": _brain_build_error,
        "percentage": 100 if (brain_status["ready"] and miller_working) else (50 if _brain_building else 0),
        "conversations": brain_status.get("conversation_count", 0),
        "documents": brain_status.get("document_count", 0),
        "chunks": brain_status.get("document_count", 0),  # For compatibility
        "method": "fixed_direct_database",
        "miller_memory_working": miller_working,
        "last_refresh": _last_brain_refresh.isoformat() if _last_brain_refresh else None
    }
    
    return status

def get_brain_control_dashboard():
    """Generate brain control dashboard HTML"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fixed Brain Control - Miller Memory Restored</title>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f0f0f; color: #fff; margin: 0; padding: 20px; 
            }
            .container { max-width: 1200px; margin: 0 auto; }
            .status-box { 
                background: #1a1a1a; border: 1px solid #333; border-radius: 8px; 
                padding: 20px; margin: 20px 0; 
            }
            .btn { 
                background: #6366f1; color: white; border: none; padding: 12px 24px; 
                border-radius: 8px; cursor: pointer; font-size: 16px; margin: 10px 5px;
            }
            .btn:hover { background: #5855eb; }
            .btn.test { background: #059669; }
            .btn.diagnostic { background: #dc2626; }
            .fixed-badge {
                background: #059669; color: white; padding: 4px 8px;
                border-radius: 12px; font-size: 11px; font-weight: bold;
            }
            #status { 
                font-family: monospace; font-size: 16px; padding: 15px;
                border-radius: 6px; background: #000; border: 1px solid #333;
            }
            .success { color: #10b981; }
            .error { color: #ef4444; }
            .building { color: #f59e0b; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Brain Control - Miller Memory <span class="fixed-badge">FIXED</span></h1>
            <p>Restored direct database access with working Miller and Ghada memory.</p>
            
            <div class="status-box">
                <h3>System Status</h3>
                <div id="status">Loading status...</div>
            </div>
            
            <div class="status-box">
                <h3>Memory Tests</h3>
                <button class="btn test" onclick="testMiller()">Test Miller Memory</button>
                <button class="btn test" onclick="testGhada()">Test Ghada Memory</button>
                <button class="btn diagnostic" onclick="showDiagnostics()">Full Diagnostics</button>
            </div>
            
            <div class="status-box">
                <h3>Controls</h3>
                <button class="btn" onclick="buildBrain()">Build from Corpus</button>
                <button class="btn" onclick="buildNewBrain()">Build from Sources</button>
                <button class="btn" onclick="refreshStatus()">Refresh Status</button>
                <button class="btn" onclick="window.location.href='/'">&larr; Back to Chat</button>
            </div>
            
            <div id="diagnostics" class="status-box" style="display:none;">
                <h3>Diagnostics</h3>
                <div id="diagnostics-content">Loading...</div>
            </div>
        </div>
        
        <script>
            function refreshStatus() {
                fetch('/brain_status')
                    .then(r => r.json())
                    .then(data => {
                        const statusDiv = document.getElementById('status');
                        
                        let html = '';
                        if (data.ready) {
                            html += '<span class="success">✅ Brain System Ready</span><br>';
                            html += `<small>${data.conversations} conversations, ${data.documents} documents</small><br>`;
                            if (data.miller_memory_working) {
                                html += '<span class="success">✅ Miller Memory Working</span>';
                            } else {
                                html += '<span class="error">❌ Miller Memory Failed</span>';
                            }
                        } else if (data.building) {
                            html += '<span class="building">🔄 Building Brain...</span>';
                        } else if (data.error) {
                            html += '<span class="error">❌ Error: ' + data.error + '</span>';
                        } else {
                            html += '<span class="error">❌ Brain Not Ready</span><br>';
                            html += '<small>No conversations or documents found</small>';
                        }
                        
                        statusDiv.innerHTML = html;
                    })
                    .catch(e => {
                        document.getElementById('status').innerHTML = '<span class="error">Connection Error</span>';
                    });
            }
            
            function testMiller() {
                alert('Testing Miller memory...');
                // This would call a test endpoint
            }
            
            function testGhada() {
                alert('Testing Ghada memory...');
                // This would call a test endpoint
            }
            
            function showDiagnostics() {
                const div = document.getElementById('diagnostics');
                const content = document.getElementById('diagnostics-content');
                
                div.style.display = 'block';
                content.innerHTML = 'Loading diagnostics...';
                
                fetch('/debug/brain_diagnostics')
                    .then(r => r.json())
                    .then(data => {
                        content.innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                    })
                    .catch(e => {
                        content.innerHTML = '<span class="error">Failed to load diagnostics</span>';
                    });
            }
            
            function buildBrain() {
                if (confirm('Build brain from corpus file?')) {
                    fetch('/build_brain', { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                            if (!data.ok) alert('Build failed: ' + data.error);
                        });
                }
            }
            
            function buildNewBrain() {
                if (confirm('Build new brain from raw sources?')) {
                    fetch('/build_new_brain', { method: 'POST' })
                        .then(r => r.json())
                        .then(data => {
                            if (!data.ok) alert('Build failed: ' + data.error);
                        });
                }
            }
            
            // Auto-refresh
            refreshStatus();
            setInterval(refreshStatus, 5000);
        </script>
    </body>
    </html>
    '''

#-------------------------------------------------------------------
# SECTION 10: COMPATIBILITY FUNCTIONS FOR EXISTING CODE
#-------------------------------------------------------------------

def is_ready():
    """Check if brain system is ready - compatibility function"""
    global _brain_system
    return _brain_system.ready

def load_corpus(path):
    """Compatibility function - triggers status refresh"""
    global _brain_system
    print("Brain system: Refreshing status instead of loading corpus file")
    _brain_system._check_system_status()

def get_build_status():
    """Get build status in expected format - compatibility function"""
    global _brain_building, _brain_system
    
    brain_status = _brain_system.get_status()
    
    return {
        "status": "building" if _brain_building else brain_status["status"],
        "progress": "Building..." if _brain_building else f"Ready: {brain_status['conversation_count']} conversations",
        "percentage": 50 if _brain_building else (100 if brain_status["ready"] else 0),
        "chunks_processed": brain_status["document_count"],
        "embeddings_created": brain_status["document_count"],
        "conversations_available": brain_status["conversation_count"],
        "method": "fixed_direct_database"
    }

#-------------------------------------------------------------------
# SECTION 11: INITIALIZATION AND GLOBAL INSTANCE
#-------------------------------------------------------------------

# Initialize global brain system instance
_brain_system = SimpleBrainSystem()

# Test Miller memory on startup
def _startup_memory_test():
    """Test memory on startup to ensure system is working"""
    try:
        print("Testing Miller memory on startup...")
        results = enhanced_retrieve("miller", k=1)
        if results:
            print(f"✅ Miller memory working: {len(results)} results found")
        else:
            print("❌ Miller memory test failed: no results")
    except Exception as e:
        print(f"❌ Miller memory test error: {e}")

# Run startup test
_startup_memory_test()

print("Brain system initialized - Fixed Miller and Ghada memory")
