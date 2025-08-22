# utils/rag_basic.py - Memory-optimized batched RAG system
import json
import os
import pickle
import gzip
from typing import List, Dict, Tuple
from datetime import datetime
import math
import gc

try:
    import numpy as np
    import openai
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False

# Global RAG instance
_rag_system = None

class BatchedRAG:
    def __init__(self, data_dir: str = "rag_data", batch_size: int = 6000):  # Reduced batch size
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Missing required dependencies: numpy, openai")
            
        # Get API key from environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        # File paths
        self.chunks_file = os.path.join(data_dir, "chunks.json")
        self.embeddings_file = os.path.join(data_dir, "embeddings.pkl")
        self.progress_file = os.path.join(data_dir, "batch_progress.json")
        self.batch_dir = os.path.join(data_dir, "batches")
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.batch_dir, exist_ok=True)
        
        self.chunks = []
        self.embeddings = []
        self.batch_progress = self.load_batch_progress()
        
        # Load existing data
        self.load_existing_data()
    
    def load_batch_progress(self) -> Dict:
        """Load batch processing progress"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "completed_batches": 0,
            "total_batches": 0,
            "total_chunks_processed": 0,
            "total_embeddings_created": 0,
            "last_updated": None
        }
    
    def save_batch_progress(self):
        """Save batch processing progress"""
        self.batch_progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.batch_progress, f, indent=2)
    
    def count_total_lines(self, jsonl_gz_file_path: str) -> int:
        """Count total lines in the compressed file"""
        print("Counting total lines in dataset...")
        line_count = 0
        try:
            with gzip.open(jsonl_gz_file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        line_count += 1
        except Exception as e:
            print(f"Error counting lines: {e}")
            return 0
        print(f"Total lines in dataset: {line_count}")
        return line_count
    
    def chunk_text(self, text: str, max_words: int = 500) -> List[str]:
        """Break text into chunks of roughly max_words"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
        
        return chunks
    
    def extract_text_from_json_line(self, json_obj: Dict) -> str:
        """Extract meaningful text from a JSON line"""
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
    
    def process_batch_lines(self, jsonl_gz_file_path: str, batch_num: int) -> List[Dict]:
        """Process a specific batch of lines from the file"""
        start_line = batch_num * self.batch_size
        end_line = start_line + self.batch_size
        
        batch_chunks = []
        chunk_id = self.batch_progress["total_chunks_processed"]
        current_line = 0
        
        print(f"Processing batch {batch_num + 1}: lines {start_line + 1} to {end_line}")
        
        try:
            with gzip.open(jsonl_gz_file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Skip lines until we reach our batch start
                    if current_line < start_line:
                        current_line += 1
                        continue
                    
                    # Stop when we reach batch end
                    if current_line >= end_line:
                        break
                    
                    try:
                        data = json.loads(line)
                        text_content = self.extract_text_from_json_line(data)
                        
                        if text_content and len(text_content.strip()) > 50:
                            text_chunks = self.chunk_text(text_content, max_words=400)
                            
                            for chunk_text in text_chunks:
                                chunk = {
                                    "id": chunk_id,
                                    "text": chunk_text,
                                    "source": f"line_{current_line + 1}",
                                    "batch": batch_num,
                                    "created_at": datetime.now().isoformat()
                                }
                                batch_chunks.append(chunk)
                                chunk_id += 1
                    
                    except json.JSONDecodeError:
                        pass
                    
                    current_line += 1
                    
                    # Progress update within batch
                    if current_line % 100 == 0:
                        progress_in_batch = current_line - start_line
                        print(f"Batch {batch_num + 1}: processed {progress_in_batch}/{self.batch_size} lines, created {len(batch_chunks)} chunks", end='\r')
        
        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            return []
        
        print(f"\nBatch {batch_num + 1} complete: created {len(batch_chunks)} chunks")
        return batch_chunks
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return [0.0] * 1536
    
    def save_embedding_batch(self, batch_num: int, sub_batch_num: int, embeddings: List[List[float]], chunk_indices: List[int]):
        """Save a sub-batch of embeddings to disk immediately"""
        embedding_sub_file = os.path.join(self.batch_dir, f"embeddings_{batch_num}_{sub_batch_num}.pkl")
        indices_sub_file = os.path.join(self.batch_dir, f"indices_{batch_num}_{sub_batch_num}.json")
        
        # Save embeddings
        with open(embedding_sub_file, 'wb') as f:
            pickle.dump(embeddings, f)
        
        # Save chunk indices for this sub-batch
        with open(indices_sub_file, 'w') as f:
            json.dump(chunk_indices, f)
        
        print(f"Saved embedding sub-batch {sub_batch_num} to disk")
    
    def process_batch_embeddings(self, batch_chunks: List[Dict], batch_num: int) -> int:
        """Create embeddings with streaming saves to prevent memory buildup"""
        total_chunks = len(batch_chunks)
        print(f"Creating embeddings for {total_chunks} chunks with streaming saves...")
        
        # Process embeddings in smaller sub-batches and save immediately
        embedding_batch_size = 50  # Reduced sub-batch size
        sub_batch_count = 0
        total_embeddings_created = 0
        
        for i in range(0, total_chunks, embedding_batch_size):
            end_idx = min(i + embedding_batch_size, total_chunks)
            sub_batch = batch_chunks[i:end_idx]
            
            # Calculate progress percentage
            progress_pct = int((i / total_chunks) * 100)
            overall_progress = int(((self.batch_progress["total_embeddings_created"] + i) / 
                                  (self.batch_progress["total_chunks_processed"] + total_chunks)) * 100)
            
            print(f"Creating embeddings: {i + 1}-{end_idx}/{total_chunks} ({progress_pct}% batch, {overall_progress}% overall)")
            
            # Extract texts for this sub-batch
            texts = [chunk["text"] for chunk in sub_batch]
            chunk_indices = [chunk["id"] for chunk in sub_batch]
            
            try:
                # Create embeddings for multiple texts in one API call
                response = self.client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                )
                
                # Extract embeddings from response
                sub_batch_embeddings = [data.embedding for data in response.data]
                
                # Immediately save to disk and free memory
                self.save_embedding_batch(batch_num, sub_batch_count, sub_batch_embeddings, chunk_indices)
                total_embeddings_created += len(sub_batch_embeddings)
                
                # Clear memory
                del sub_batch_embeddings
                del texts
                del chunk_indices
                gc.collect()  # Force garbage collection
                
            except Exception as e:
                print(f"\nError creating embeddings for sub-batch {sub_batch_count}: {e}")
                # Fallback to individual processing for this sub-batch
                sub_batch_embeddings = []
                chunk_indices = []
                for chunk in sub_batch:
                    embedding = self.create_embedding(chunk["text"])
                    sub_batch_embeddings.append(embedding)
                    chunk_indices.append(chunk["id"])
                
                # Save fallback results
                self.save_embedding_batch(batch_num, sub_batch_count, sub_batch_embeddings, chunk_indices)
                total_embeddings_created += len(sub_batch_embeddings)
                
                # Clear memory
                del sub_batch_embeddings
                del chunk_indices
                gc.collect()
            
            sub_batch_count += 1
        
        print(f"\nCompleted all embeddings for batch {batch_num + 1}: {total_embeddings_created} embeddings created")
        return total_embeddings_created
    
    def save_batch_data(self, batch_num: int, batch_chunks: List[Dict]):
        """Save batch chunk data to disk"""
        batch_file = os.path.join(self.batch_dir, f"batch_{batch_num}.json")
        
        # Save chunks
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_chunks, f, indent=2, ensure_ascii=False)
        
        print(f"Saved batch {batch_num + 1} chunk data to disk")
    
    def load_all_batches(self):
        """Load all completed batches into memory"""
        print("Loading all completed batches...")
        
        all_chunks = []
        all_embeddings = []
        
        for batch_num in range(self.batch_progress["completed_batches"]):
            batch_file = os.path.join(self.batch_dir, f"batch_{batch_num}.json")
            
            if os.path.exists(batch_file):
                # Load chunks
                with open(batch_file, 'r', encoding='utf-8') as f:
                    batch_chunks = json.load(f)
                
                # Load all embedding sub-batches for this batch
                batch_embeddings = []
                sub_batch_num = 0
                
                while True:
                    embedding_sub_file = os.path.join(self.batch_dir, f"embeddings_{batch_num}_{sub_batch_num}.pkl")
                    indices_sub_file = os.path.join(self.batch_dir, f"indices_{batch_num}_{sub_batch_num}.json")
                    
                    if not os.path.exists(embedding_sub_file):
                        break
                    
                    # Load embeddings sub-batch
                    with open(embedding_sub_file, 'rb') as f:
                        sub_embeddings = pickle.load(f)
                    
                    # Load indices sub-batch
                    with open(indices_sub_file, 'r') as f:
                        sub_indices = json.load(f)
                    
                    batch_embeddings.extend(sub_embeddings)
                    sub_batch_num += 1
                
                all_chunks.extend(batch_chunks)
                all_embeddings.extend(batch_embeddings)
                
                print(f"Loaded batch {batch_num + 1}: {len(batch_chunks)} chunks, {len(batch_embeddings)} embeddings")
        
        self.chunks = all_chunks
        self.embeddings = all_embeddings
        
        print(f"Total loaded: {len(self.chunks)} chunks with {len(self.embeddings)} embeddings")
    
    def build_index_in_batches(self, jsonl_gz_file_path: str):
        """Build the RAG index using memory-optimized batch processing"""
        print("Starting memory-optimized batched RAG index building...")
        
        # Count total lines if not already done
        if self.batch_progress["total_batches"] == 0:
            total_lines = self.count_total_lines(jsonl_gz_file_path)
            total_batches = math.ceil(total_lines / self.batch_size)
            
            self.batch_progress["total_batches"] = total_batches
            self.save_batch_progress()
            
            print(f"Will process {total_lines} lines in {total_batches} batches of {self.batch_size}")
        
        # Process remaining batches
        start_batch = self.batch_progress["completed_batches"]
        total_batches = self.batch_progress["total_batches"]
        
        for batch_num in range(start_batch, total_batches):
            print(f"\n=== Processing Batch {batch_num + 1}/{total_batches} ===")
            
            try:
                # Process batch lines into chunks
                batch_chunks = self.process_batch_lines(jsonl_gz_file_path, batch_num)
                
                if not batch_chunks:
                    print(f"No chunks created for batch {batch_num + 1}, skipping...")
                    self.batch_progress["completed_batches"] = batch_num + 1
                    self.save_batch_progress()
                    continue
                
                # Save chunks to disk
                self.save_batch_data(batch_num, batch_chunks)
                
                # Create embeddings with streaming saves
                embeddings_created = self.process_batch_embeddings(batch_chunks, batch_num)
                
                # Update progress
                self.batch_progress["completed_batches"] = batch_num + 1
                self.batch_progress["total_chunks_processed"] += len(batch_chunks)
                self.batch_progress["total_embeddings_created"] += embeddings_created
                self.save_batch_progress()
                
                print(f"Completed batch {batch_num + 1}/{total_batches}")
                
                # Clear memory for next batch
                del batch_chunks
                gc.collect()
                
            except Exception as e:
                print(f"Error processing batch {batch_num + 1}: {e}")
                break
        
        # Load all completed batches into memory
        self.load_all_batches()
        
        print("Memory-optimized batched index building complete!")
    
    def load_existing_data(self):
        """Load existing data if available"""
        if self.batch_progress["completed_batches"] > 0:
            print(f"Found {self.batch_progress['completed_batches']} completed batches")
            self.load_all_batches()
    
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
    
    def get_build_status(self) -> Dict:
        """Get current build status for progress tracking"""
        total_batches = self.batch_progress["total_batches"]
        completed_batches = self.batch_progress["completed_batches"]
        
        if total_batches == 0:
            return {
                "status": "not_started",
                "progress": "Ready to build",
                "percentage": 0,
                "chunks_processed": 0,
                "embeddings_created": 0,
                "total_estimated_chunks": 0
            }
        
        percentage = int((completed_batches / total_batches) * 100)
        
        if completed_batches >= total_batches:
            return {
                "status": "complete",
                "progress": f"Build complete! {len(self.chunks)} chunks ready",
                "percentage": 100,
                "chunks_processed": len(self.chunks),
                "embeddings_created": len(self.embeddings),
                "total_estimated_chunks": len(self.chunks)
            }
        
        return {
            "status": "building",
            "progress": f"Processing batch {completed_batches + 1}/{total_batches}",
            "percentage": percentage,
            "chunks_processed": self.batch_progress["total_chunks_processed"],
            "embeddings_created": self.batch_progress["total_embeddings_created"],
            "batches_completed": completed_batches,
            "total_batches": total_batches
        }

# Initialize RAG system
def _get_rag_system():
    global _rag_system
    if _rag_system is None:
        try:
            if DEPENDENCIES_AVAILABLE:
                _rag_system = BatchedRAG()
                print("Batched RAG system initialized")
            else:
                print("RAG system disabled - missing dependencies")
        except Exception as e:
            print(f"Failed to initialize RAG system: {e}")
            _rag_system = None
    return _rag_system

# Public API functions
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
    """Load the corpus using batched processing"""
    global _rag_system
    try:
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("Missing required dependencies")
        
        _rag_system = BatchedRAG()
        _rag_system.build_index_in_batches(path)
        print("Batched RAG corpus loaded successfully")
    except Exception as e:
        print(f"RAG corpus load failed: {e}")
        _rag_system = None
        raise

def get_build_status():
    """Get build status for progress tracking"""
    rag = _get_rag_system()
    if rag:
        return rag.get_build_status()
    return {"status": "not_initialized", "progress": "", "percentage": 0}

# Auto-initialize
if DEPENDENCIES_AVAILABLE:
    _get_rag_system()
else:
    print("Batched RAG system initialization skipped - missing dependencies")

__all__ = ['retrieve', 'is_ready', 'load_corpus', 'get_build_status']
