import json
import os
import pickle
import numpy as np
import gzip
from typing import List, Dict, Tuple
import openai
from datetime import datetime
import re

class SimpleRAG:
    def __init__(self, data_dir: str = "rag_data"):
        """
        Simple RAG system for Ghostline project
        
        Args:
            data_dir: Directory to store embeddings and chunks
        """
        # Get API key from environment (Render sets this)
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.data_dir = data_dir
        self.chunks_file = os.path.join(data_dir, "chunks.json")
        self.embeddings_file = os.path.join(data_dir, "embeddings.pkl")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        self.chunks = []
        self.embeddings = []
        self.load_existing_data()
    
    def chunk_text(self, text: str, max_words: int = 500) -> List[str]:
        """Break text into chunks of roughly max_words"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.strip()) > 50:  # Skip tiny chunks
                chunks.append(chunk.strip())
        
        return chunks
    
    def extract_conversations_from_json(self, json_data: Dict) -> List[Dict]:
        """Extract meaningful conversations from ChatGPT export JSON"""
        conversations = []
        
        # Handle different possible JSON structures
        if isinstance(json_data, list):
            # If it's a list of conversations
            for item in json_data:
                if isinstance(item, dict):
                    conversations.extend(self.extract_conversations_from_json(item))
        
        elif isinstance(json_data, dict):
            # Check for common ChatGPT export structures
            if "conversations" in json_data:
                for conv in json_data["conversations"]:
                    conversations.append(conv)
            
            elif "messages" in json_data:
                # Single conversation format
                conversations.append(json_data)
            
            elif "mapping" in json_data:
                # Another ChatGPT export format
                messages = []
                for msg_id, msg_data in json_data["mapping"].items():
                    if msg_data and "message" in msg_data and msg_data["message"]:
                        message = msg_data["message"]
                        if "content" in message and message["content"]:
                            messages.append(message)
                
                if messages:
                    conversations.append({"messages": messages})
        
        return conversations
    
    def process_conversation(self, conversation: Dict) -> List[str]:
        """Extract text chunks from a conversation"""
        texts = []
        
        if "messages" in conversation:
            for message in conversation["messages"]:
                # Handle different message formats
                content = ""
                
                if isinstance(message, dict):
                    if "content" in message:
                        if isinstance(message["content"], dict):
                            if "parts" in message["content"]:
                                content = " ".join(message["content"]["parts"])
                            elif "text" in message["content"]:
                                content = message["content"]["text"]
                        elif isinstance(message["content"], str):
                            content = message["content"]
                    
                    elif "text" in message:
                        content = message["text"]
                
                if content and len(content.strip()) > 20:
                    # Add some context about the message
                    author = message.get("author", {}).get("role", "unknown") if isinstance(message.get("author"), dict) else "unknown"
                    timestamp = message.get("create_time", "")
                    
                    context_prefix = f"[{author}] "
                    if timestamp:
                        context_prefix += f"({timestamp[:10]}) "
                    
                    full_content = context_prefix + content.strip()
                    texts.append(full_content)
        
        return texts
    
    def load_and_chunk_jsonl_gz(self, jsonl_gz_file_path: str) -> List[Dict]:
        """Load compressed JSONL file and convert to chunks with metadata"""
        print(f"Loading JSONL.GZ file: {jsonl_gz_file_path}")
        
        all_chunks = []
        chunk_id = 0
        line_count = 0
        
        with gzip.open(jsonl_gz_file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                line_count += 1
                if line_count % 1000 == 0:
                    print(f"Processed {line_count} lines, created {len(all_chunks)} chunks", end='\r')
                
                try:
                    # Parse each line as JSON
                    data = json.loads(line)
                    
                    # Extract text content from the JSON line
                    text_content = self.extract_text_from_json_line(data)
                    
                    if text_content and len(text_content.strip()) > 50:
                        # Chunk the text if it's long
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
                
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_count}: {e}")
                    continue
        
        print(f"\nProcessed {line_count} lines, created {len(all_chunks)} chunks")
        return all_chunks
    
    def extract_text_from_json_line(self, json_obj: Dict) -> str:
        """Extract meaningful text from a JSON line"""
        texts = []
        
        # Common field names that might contain text
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
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"  # Cheaper and faster
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return [0.0] * 1536  # Default embedding size
    
    def build_index(self, jsonl_gz_file_path: str):
        """Build the RAG index from your JSONL.GZ file"""
        print("Building RAG index...")
        
        # Load and chunk the JSONL.GZ
        self.chunks = self.load_and_chunk_jsonl_gz(jsonl_gz_file_path)
        
        # Create embeddings for each chunk
        print("Creating embeddings...")
        self.embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            print(f"Processing chunk {i + 1}/{len(self.chunks)}", end='\r')
            embedding = self.create_embedding(chunk["text"])
            self.embeddings.append(embedding)
        
        print(f"\nCreated embeddings for {len(self.chunks)} chunks")
        
        # Save everything
        self.save_data()
        print("RAG index built and saved!")
    
    def save_data(self):
        """Save chunks and embeddings to disk"""
        with open(self.chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)
    
    def load_existing_data(self):
        """Load existing chunks and embeddings if they exist"""
        if os.path.exists(self.chunks_file) and os.path.exists(self.embeddings_file):
            print("Loading existing RAG data...")
            
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            
            print(f"Loaded {len(self.chunks)} chunks with embeddings")
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks given a query"""
        if not self.chunks or not self.embeddings:
            return []
        
        # Create embedding for the query
        query_embedding = self.create_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, chunk_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top_k
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for similarity, idx in similarities[:top_k]:
            chunk = self.chunks[idx].copy()
            chunk["similarity"] = similarity
            results.append(chunk)
        
        return results
    
    def get_context_for_query(self, query: str, max_context_length: int = 3000) -> str:
        """Get relevant context for a query, formatted for the AI"""
        relevant_chunks = self.search(query, top_k=10)
        
        if not relevant_chunks:
            return "No relevant information found in your chat history."
        
        context_parts = []
        current_length = 0
        
        for chunk in relevant_chunks:
            chunk_text = chunk["text"]
            if current_length + len(chunk_text) > max_context_length:
                break
            
            context_parts.append(f"From {chunk['source']}: {chunk_text}")
            current_length += len(chunk_text)
        
        if context_parts:
            context = "Based on your previous conversations:\n\n" + "\n\n".join(context_parts)
        else:
            context = "No relevant information found in your chat history."
        
        return context


# Example usage and setup function
def setup_rag_system(jsonl_gz_file_path: str):
    """Setup function to initialize the RAG system"""
    print("Setting up RAG system for Ghostline...")
    
    rag = SimpleRAG()
    
    # Check if we already have an index
    if not rag.chunks:
        print("No existing index found. Building new index...")
        rag.build_index(jsonl_gz_file_path)
    else:
        print("Existing index found! Ready to use.")
    
    return rag


# Test function
def test_rag_system(rag: SimpleRAG):
    """Test the RAG system"""
    print("\nTesting RAG system...")
    
    test_queries = [
        "recipes",
        "wife",
        "cooking",
        "family"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = rag.search(query, top_k=3)
        
        if results:
            for i, result in enumerate(results):
                print(f"  {i+1}. (similarity: {result['similarity']:.3f}) {result['text'][:200]}...")
        else:
            print("  No results found")


if __name__ == "__main__":
    # Replace with your actual file path
    JSONL_GZ_FILE = "/data/cleaned/ghostline_sources.jsonl.gz"
    
    # Setup the RAG system (API key from environment)
    rag = setup_rag_system(JSONL_GZ_FILE)
    
    # Test it
    test_rag_system(rag)