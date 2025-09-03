# utils/rag_basic.py - Simplified database-only retrieval
import os
import json
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("NumPy not available - falling back to simple text search")
    NUMPY_AVAILABLE = False

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
    DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

@contextmanager
def get_db_connection():
    """Get database connection with proper error handling"""
    if not DATABASE_URL:
        print("No database URL configured")
        yield None
        return
    
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
    except Exception as e:
        print(f"Database connection failed: {e}")
        yield None
    finally:
        if conn:
            conn.close()

class DatabaseRAG:
    """Simple database-only retrieval system using existing brain_documents table"""
    
    def __init__(self):
        self.ready = False
        self.chunk_count = 0
        self._check_database_status()
    
    def _check_database_status(self):
        """Check if brain_documents table has data"""
        try:
            with get_db_connection() as conn:
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM brain_documents WHERE content IS NOT NULL")
                    self.chunk_count = cursor.fetchone()[0]
                    self.ready = self.chunk_count > 0
                    print(f"Database RAG: Found {self.chunk_count} brain documents")
                else:
                    print("Database RAG: No database connection")
        except Exception as e:
            print(f"Database RAG initialization failed: {e}")
            self.ready = False
    
    def search_simple(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simple text search using PostgreSQL full-text search"""
        if not self.ready:
            return []
        
        try:
            with get_db_connection() as conn:
                if not conn:
                    return []
                
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                # Use PostgreSQL full-text search
                sql = """
                SELECT title, content, metadata,
                       ts_rank(to_tsvector('english', content), plainto_tsquery('english', %s)) as rank
                FROM brain_documents 
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
                   AND content IS NOT NULL
                ORDER BY rank DESC
                LIMIT %s
                """
                
                cursor.execute(sql, (query, query, top_k))
                results = cursor.fetchall()
                
                return [
                    {
                        "text": row['content'],
                        "title": row['title'] or 'Untitled',
                        "source": self._extract_source(row.get('metadata', {})),
                        "similarity": float(row['rank']) if row['rank'] else 0.0
                    }
                    for row in results
                ]
                
        except Exception as e:
            print(f"Database search failed: {e}")
            return []
    
    def search_vector(self, query: str, top_k: int = 5) -> List[Dict]:
        """Vector similarity search using stored embeddings (requires OpenAI for query embedding)"""
        if not self.ready or not NUMPY_AVAILABLE:
            return self.search_simple(query, top_k)  # Fallback
        
        # Get OpenAI API key for query embedding
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("No OpenAI key for vector search, using text search instead")
            return self.search_simple(query, top_k)
        
        try:
            # Create query embedding
            import openai
            client = openai.OpenAI(api_key=openai_key)
            
            response = client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_vector = response.data[0].embedding
            
            with get_db_connection() as conn:
                if not conn:
                    return []
                
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                
                # Use pgvector for similarity search if available
                # Otherwise calculate similarity in Python
                sql = """
                SELECT title, content, metadata, embedding_vector
                FROM brain_documents 
                WHERE embedding_vector IS NOT NULL
                   AND content IS NOT NULL
                LIMIT 500
                """
                
                cursor.execute(sql)
                results = cursor.fetchall()
                
                # Calculate similarities in Python
                scored_results = []
                for row in results:
                    if row['embedding_vector']:
                        try:
                            # Convert stored embedding to numpy array
                            stored_vector = np.array(row['embedding_vector'])
                            query_np = np.array(query_vector)
                            
                            # Calculate cosine similarity
                            similarity = np.dot(query_np, stored_vector) / (
                                np.linalg.norm(query_np) * np.linalg.norm(stored_vector)
                            )
                            
                            scored_results.append({
                                "text": row['content'],
                                "title": row['title'] or 'Untitled',
                                "source": self._extract_source(row.get('metadata', {})),
                                "similarity": float(similarity)
                            })
                        except Exception as e:
                            print(f"Error calculating similarity: {e}")
                            continue
                
                # Sort by similarity and return top results
                scored_results.sort(key=lambda x: x['similarity'], reverse=True)
                return scored_results[:top_k]
                
        except Exception as e:
            print(f"Vector search failed: {e}, falling back to text search")
            return self.search_simple(query, top_k)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Main search method - tries vector search first, falls back to text search"""
        # Try vector search first if we have OpenAI key
        if os.getenv('OPENAI_API_KEY') and NUMPY_AVAILABLE:
            results = self.search_vector(query, top_k)
            if results:
                return results
        
        # Fallback to simple text search
        return self.search_simple(query, top_k)
    
    def _extract_source(self, metadata: dict) -> str:
        """Extract source info from metadata"""
        if isinstance(metadata, dict):
            return metadata.get('source', 'Unknown')
        return 'Unknown'
    
    def get_status(self) -> Dict:
        """Get system status"""
        return {
            "status": "complete" if self.ready else "not_ready",
            "progress": f"Database ready with {self.chunk_count} documents" if self.ready else "Database not ready",
            "percentage": 100 if self.ready else 0,
            "chunks_processed": self.chunk_count,
            "embeddings_created": self.chunk_count if self.ready else 0,
            "method": "database"
        }

# Global instance
_database_rag = None

def _get_database_rag():
    global _database_rag
    if _database_rag is None:
        _database_rag = DatabaseRAG()
    return _database_rag

# Public API functions (compatible with existing code)
def retrieve(query: str, k: int = 5) -> List[Dict]:
    """Retrieve relevant context using database RAG"""
    rag = _get_database_rag()
    try:
        results = rag.search(query, top_k=k)
        # Convert to expected format
        return [{"text": result["text"], "source": result["source"]} for result in results]
    except Exception as e:
        print(f"Database RAG retrieval error: {e}")
        return []

def is_ready() -> bool:
    """Check if database RAG system is ready"""
    rag = _get_database_rag()
    return rag.ready

def load_corpus(path):
    """No-op for database system - data is already loaded"""
    print("Database RAG: Corpus already loaded in database, no file processing needed")
    rag = _get_database_rag()
    rag._check_database_status()  # Refresh status
    if rag.ready:
        print(f"Database RAG: Ready with {rag.chunk_count} documents")
    else:
        print("Database RAG: No documents found in brain_documents table")

def get_build_status() -> Dict:
    """Get build status for progress tracking"""
    rag = _get_database_rag()
    return rag.get_status()

# Initialize on import
_get_database_rag()

__all__ = ['retrieve', 'is_ready', 'load_corpus', 'get_build_status']
