# utils/rag_basic.py - Fixed version with proper exports
import json
import os
import pickle
import gzip
from typing import List, Dict
from datetime import datetime

try:
    import numpy as np
    import openai
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

# Global RAG instance
_rag_system = None

class SimpleRAG:
    def __init__(self, data_dir: str = "rag_data"):
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Missing required dependencies: numpy, openai")
            
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.data_dir = data_dir
        self.chunks_file = os.path.join(data_dir, "chunks.json")
        self.embeddings_file = os.path.join(data_dir, "embeddings.pkl")
        
        os.makedirs(data_dir, exist_ok=True)
        
        self.chunks = []
        self.embeddings = []
        self.load_existing_data()
    
    def chunk_text(self, text: str, max_words: int = 500) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
        
        return chunks
    
    def extract_text_from_json_line(self, json_obj: Dict) -> str:
        texts = []
        text_fields = ['text', 'content', 'message', 'body', 'description', 'title', 'question', 'answer']
        
        def extract_from_obj(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in text_fields and isinstance(value, str) and len(value.strip()) > 20:
                        context = f"[{prefix}{key}] " if prefix or key != 'text' else ""
                        texts.append(context + value.strip())
                    elif isinstance(value, (dict, list)):
                        extract_from_obj(value, f"{prefix}{key}.")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (dict, list)):
                        extract_from_obj(item, f"{prefix}[{i}].")
                    elif isinstance(item, str) and len(item.strip()) > 20:
                        texts.append(item.strip())
        
        extract_from_obj(json_obj)
        return " ".join(texts) if texts else ""
    
    def load_and_chunk_jsonl_gz(self, jsonl_gz_file_path: str) -> List[Dict]:
        print(f"Loading JSONL.GZ file: {jsonl_gz_file_path}")
        
        all_chunks = []
        chunk_id = 0
        line_count = 0
        
        try:
            with gzip.open(jsonl_gz_file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    line_count += 1
                    if line_count % 1000 == 0:
                        print(f"Processed {line_count} lines, created {len(all_chunks)} chunks")
                    
                    try:
                        data = json.loads(line)
                        text_content = self.extract_text_from_json_line(data)
                        
                        if text_content and len(text_content.strip()) > 50:
                            text_chunks = self.chunk_text(text_content, max_words=400)
                            
                            for chunk_text in text_chunks:
                                chunk = {
                                    "id": chunk_id,
                                    "text": chunk_text,
                                    "source": f"line_{line_count}",
                                    "created_at": datetime.now().isoformat()
                                }
                                all_chunks.append(chunk)
                                chunk_id += 1
                    
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error loading file: {e}")
            return []
        
        print(f"Processed {line_count} lines, created {len(all_chunks)} chunks")
        return all_chunks
    
    def create_embedding(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return [0.0] * 1536
    
    def build_index(self, jsonl_gz_file_path: str):
        print("Building RAG index...")
        
        self.chunks = self.load_and_chunk_jsonl_gz(jsonl_gz_file_path)
        
        if not self.chunks:
            print("No chunks loaded - aborting index build")
            return
        
        print("Creating embeddings...")
        self.embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            if i % 100 == 0:
                print(f"Processing chunk {i + 1}/{len(self.chunks)}")
            embedding = self.create_embedding(chunk["text"])
            self.embeddings.append(embedding)
        
        print(f"Created embeddings for {len(self.chunks)} chunks")
        self.save_data()
        print("RAG index built and saved!")
    
    def save_data(self):
        try:
            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_existing_data(self):
        if os.path.exists(self.chunks_file) and os.path.exists(self.embeddings_file):
            try:
                print("Loading existing RAG data...")
                
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    self.chunks = json.load(f)
                
                with open(self.embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                print(f"Loaded {len(self.chunks)} chunks with embeddings")
            except Exception as e:
                print(f"Error loading existing data: {e}")
                self.chunks = []
                self.embeddings = []
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        if not DEPENDENCIES_AVAILABLE:
            return 0.0
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.chunks or not self.embeddings:
            return []
        
        query_embedding = self.create_embedding(query)
        similarities = []
        
        for i, chunk_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((similarity, i))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for similarity, idx in similarities[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["similarity"] = similarity
            results.append(chunk)
        
        return results

# Initialize RAG system
def _get_rag_system():
    global _rag_system
    if _rag_system is None:
        try:
            if DEPENDENCIES_AVAILABLE:
                _rag_system = SimpleRAG()
                print("RAG system initialized")
            else:
                print("RAG system disabled - missing dependencies")
        except Exception as e:
            print(f"Failed to initialize RAG system: {e}")
            _rag_system = None
    return _rag_system

# Public API functions (these are what app.py imports)
def retrieve(query: str, k: int = 5):
    """Retrieve relevant context using the RAG system"""
    rag = _get_rag_system()
    if not rag:
        return []
    
    try:
        results = rag.search(query, top_k=k)
        return [{"text": result["text"], "source": result["source"]} for result in results]
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        return []

def is_ready():
    """Check if RAG system is ready"""
    rag = _get_rag_system()
    return rag is not None and len(rag.chunks) > 0

def load_corpus(path):
    """Load the corpus (build the RAG index)"""
    global _rag_system
    try:
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Missing required dependencies")
        
        _rag_system = SimpleRAG()
        _rag_system.build_index(path)
        print("RAG corpus loaded successfully")
    except Exception as e:
        print(f"RAG corpus load failed: {e}")
        _rag_system = None
        raise

# Test functions
def test_system():
    """Test if the system is working"""
    print("Testing RAG system...")
    print(f"Dependencies available: {DEPENDENCIES_AVAILABLE}")
    print(f"System ready: {is_ready()}")
    
    if is_ready():
        results = retrieve("test query", k=3)
        print(f"Test search returned {len(results)} results")
    
    return is_ready()

# Auto-initialize
if DEPENDENCIES_AVAILABLE:
    _get_rag_system()
else:
    print("RAG system initialization skipped - missing dependencies")

# Ensure these functions are available for import
__all__ = ['retrieve', 'is_ready', 'load_corpus', 'test_system']

