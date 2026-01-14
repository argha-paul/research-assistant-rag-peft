# src/retrieval/vector_store.py
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

class FastVectorStore:
    """
    Speed-first vector store.
    - Small embedding model (90MB)
    - Batch processing
    - No re-ranking
    """
    
    def __init__(self, persist_dir="data/embeddings", collection_name="research_papers"):
        # Use tiny embedding model for speed
        # all-MiniLM-L6-v2: 384 dims, ~90MB, ~2000 docs/sec on M1
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Fast approximate search
        )
    
    def embed_and_store(self, chunks_file):
        """Embed and store chunks in batch - FAST."""
        # Load chunks
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        
        print(f"Embedding {len(chunks)} chunks...")
        
        # Batch processing for speed (process 100 at a time)
        batch_size = 100
        for i in tqdm(range(0, len(chunks), batch_size)):
            batch = chunks[i:i + batch_size]
            
            # Extract text and metadata
            texts = [c['text'] for c in batch]
            ids = [c['chunk_id'] for c in batch]
            metadatas = [{'source': c['source'], 'chunk_num': c.get('chunk_id', '')} 
                        for c in batch]
            
            # Embed batch (fast!)
            embeddings = self.embedding_model.encode(
                texts, 
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            # Store in ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"✓ Stored {len(chunks)} chunks in vector database")
    
    def search(self, query, top_k=5):
        """
        Fast search - no re-ranking.
        Returns results in ~50ms on M1.
        """
        # Embed query
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results

if __name__ == "__main__":
    store = FastVectorStore()
    store.embed_and_store("data/processed/chunks.json")