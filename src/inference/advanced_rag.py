import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
import time
from functools import lru_cache
import hashlib

class AdvancedRAGSystem:
    """
    Production-grade RAG with advanced techniques:
    - Hybrid retrieval (Dense + BM25)
    - Cross-encoder re-ranking
    - Query enhancement (HyDE)
    - Context compression
    - Smart caching
    - Source attribution
    """
    
    def __init__(self, 
                 model_name="tinyllama",
                 embedding_model="all-MiniLM-L6-v2",
                 reranker_model="cross-encoder/ms-marco-MiniLM-L-2-v2",
                 persist_dir="data/embeddings"):
        
        print("🚀 Initializing Advanced RAG System...")
        
        # LLM
        self.model_name = model_name
        
        # Embedding model for semantic search
        print("  ├─ Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Cross-encoder for re-ranking
        print("  ├─ Loading re-ranker...")
        self.reranker = CrossEncoder(reranker_model)
        
        # Vector database
        print("  ├─ Connecting to vector database...")
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(name="research_papers")
        
        # BM25 for keyword search
        print("  ├─ Loading BM25 index...")
        self._init_bm25()
        
        # Cache for speed
        self.query_cache = {}
        self.max_cache_size = 100
        
        print("✅ Advanced RAG System ready!\n")
    
    def _init_bm25(self):
        """Initialize BM25 index from ChromaDB."""
        # Get all documents
        all_docs = self.collection.get()
        self.bm25_docs = all_docs['documents']
        self.bm25_ids = all_docs['ids']
        self.bm25_metadatas = all_docs['metadatas']
        
        # Tokenize for BM25
        tokenized_corpus = [doc.lower().split() for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)
    
    def _get_cache_key(self, question: str, top_k: int) -> str:
        """Generate cache key for query."""
        key_str = f"{question}_{top_k}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @lru_cache(maxsize=50)
    def _embed_query_cached(self, question: str) -> np.ndarray:
        """Cache query embeddings."""
        return self.embedding_model.encode(question)
    
    def enhance_query_hyde(self, question: str) -> str:
        """
        HyDE: Generate hypothetical document to improve retrieval.
        Creates a synthetic answer, then searches for similar docs.
        """
        hyde_prompt = f"""Generate a concise, technical answer to this question as if from a research paper:

Question: {question}

Technical Answer:"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=hyde_prompt,
                options={'temperature': 0.3, 'num_predict': 100}
            )
            return response['response'].strip()
        except:
            return question
    
    def query_expansion(self, question: str) -> List[str]:
        """
        Generate query variations for better retrieval coverage.
        """
        expansion_prompt = f"""Generate 2 alternative phrasings of this question (one line each):

Question: {question}

Alternative 1:"""
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=expansion_prompt,
                options={'temperature': 0.7, 'num_predict': 50}
            )
            
            # Parse alternatives
            alternatives = [question]
            lines = response['response'].strip().split('\n')
            for line in lines[:2]:
                cleaned = line.strip().replace('Alternative 2:', '').strip()
                if cleaned and len(cleaned) > 10:
                    alternatives.append(cleaned)
            
            return alternatives
        except:
            return [question]
    
    def hybrid_retrieve(self, 
                       question: str, 
                       top_k: int = 20,
                       use_hyde: bool = True,
                       use_expansion: bool = False) -> List[Dict]:
        """
        Hybrid retrieval: Dense (semantic) + Sparse (BM25).
        Combines both for better recall.
        """
        results = []
        seen_ids = set()
        
        # Query variations
        queries = [question]
        if use_hyde:
            hyde_answer = self.enhance_query_hyde(question)
            queries.append(hyde_answer)
        
        if use_expansion:
            expanded = self.query_expansion(question)
            queries.extend(expanded)
        
        # 1. Dense retrieval (semantic search) for each query
        for query in queries:
            query_embedding = self._embed_query_cached(query)
            
            dense_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k // 2
            )
            
            for i, doc_id in enumerate(dense_results['ids'][0]):
                if doc_id not in seen_ids:
                    results.append({
                        'id': doc_id,
                        'text': dense_results['documents'][0][i],
                        'metadata': dense_results['metadatas'][0][i],
                        'score': 1.0 / (i + 1),  # Reciprocal rank
                        'source': 'dense'
                    })
                    seen_ids.add(doc_id)
        
        # 2. Sparse retrieval (BM25 keyword search)
        tokenized_query = question.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k // 2]
        
        for idx in bm25_top_indices:
            doc_id = self.bm25_ids[idx]
            if doc_id not in seen_ids and bm25_scores[idx] > 0:
                results.append({
                    'id': doc_id,
                    'text': self.bm25_docs[idx],
                    'metadata': self.bm25_metadatas[idx],
                    'score': bm25_scores[idx],
                    'source': 'bm25'
                })
                seen_ids.add(doc_id)
        
        return results[:top_k]
    
    def rerank_results(self, 
                      question: str, 
                      results: List[Dict], 
                      top_k: int = 5) -> List[Dict]:
        """
        Re-rank results using cross-encoder for better precision.
        """
        if not results:
            return []
        
        # Prepare pairs for re-ranking
        pairs = [[question, r['text']] for r in results]
        
        # Get cross-encoder scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine with original scores
        for i, result in enumerate(results):
            result['rerank_score'] = float(rerank_scores[i])
            result['final_score'] = result['rerank_score']
        
        # Sort by re-rank score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return results[:top_k]
    
    def filter_and_compress_context(self, 
                                    results: List[Dict], 
                                    max_tokens: int = 1500) -> List[Dict]:
        """
        Filter low-quality results and compress context.
        """
        # Filter by relevance threshold
        threshold = 0.3
        filtered = [r for r in results if r.get('final_score', 0) > threshold]
        
        # If nothing passes threshold, take top 3
        if not filtered:
            filtered = results[:3]
        
        # Estimate tokens (rough: 4 chars per token)
        current_tokens = 0
        compressed = []
        
        for result in filtered:
            text = result['text']
            estimated_tokens = len(text) // 4
            
            if current_tokens + estimated_tokens <= max_tokens:
                compressed.append(result)
                current_tokens += estimated_tokens
            else:
                # Truncate last document to fit
                remaining_tokens = max_tokens - current_tokens
                remaining_chars = remaining_tokens * 4
                if remaining_chars > 100:
                    result['text'] = text[:remaining_chars] + "..."
                    compressed.append(result)
                break
        
        return compressed
    
    def build_enhanced_prompt(self, 
                            question: str, 
                            contexts: List[Dict]) -> str:
        """
        Build structured prompt with citations.
        """
        # Build context with source attribution
        context_parts = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx['metadata'].get('source', 'Unknown')
            text = ctx['text']
            context_parts.append(f"[Source {i}: {source}]\n{text}")
        
        context_str = "\n\n".join(context_parts)
        
        prompt = f"""You are an AI research assistant. Answer the question based on the provided research paper excerpts.

# Research Context:
{context_str}

# Question:
{question}

# Instructions:
1. Answer based ONLY on the context provided above
2. Be specific and cite sources using [Source X] notation
3. If the context doesn't contain the answer, say "Based on the provided context, I cannot answer this question"
4. Be concise but comprehensive

# Answer:"""
        
        return prompt
    
    def query(self, 
             question: str, 
             top_k: int = 5,
             use_hybrid: bool = True,
             use_rerank: bool = True,
             use_hyde: bool = True,
             stream: bool = True,
             verbose: bool = True) -> Dict:
        """
        Advanced query with all optimizations.
        
        Returns:
            Dict with 'answer', 'sources', and 'timing' info
        """
        if verbose:
            print("\n" + "="*70)
            print(f"📝 Question: {question}")
            print("="*70)
        
        timing = {}
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(question, top_k)
        if cache_key in self.query_cache:
            if verbose:
                print("⚡ [Cache hit - instant response]")
            return self.query_cache[cache_key]
        
        # Step 1: Hybrid Retrieval
        if verbose:
            print("\n Stage 1: Hybrid Retrieval (Dense + BM25)...")
        
        retrieval_start = time.time()
        if use_hybrid:
            results = self.hybrid_retrieve(
                question, 
                top_k=top_k * 4,  # Get more candidates for re-ranking
                use_hyde=use_hyde
            )
        else:
            # Simple dense retrieval
            query_embedding = self._embed_query_cached(question)
            dense_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k * 2
            )
            results = [{
                'id': dense_results['ids'][0][i],
                'text': dense_results['documents'][0][i],
                'metadata': dense_results['metadatas'][0][i],
                'score': 1.0,
                'source': 'dense'
            } for i in range(len(dense_results['ids'][0]))]
        
        timing['retrieval'] = time.time() - retrieval_start
        if verbose:
            print(f"   Retrieved {len(results)} candidates in {timing['retrieval']:.2f}s")
        
        # Step 2: Re-ranking
        if use_rerank and len(results) > top_k:
            if verbose:
                print(f"\n Stage 2: Re-ranking with Cross-Encoder...")
            
            rerank_start = time.time()
            results = self.rerank_results(question, results, top_k=top_k)
            timing['rerank'] = time.time() - rerank_start
            
            if verbose:
                print(f"   Re-ranked to top {len(results)} in {timing['rerank']:.2f}s")
        
        # Step 3: Context compression
        results = self.filter_and_compress_context(results, max_tokens=1500)
        
        if verbose:
            print(f"\n Using {len(results)} contexts for generation")
            for i, r in enumerate(results, 1):
                score = r.get('final_score', r.get('score', 0))
                source = r['metadata'].get('source', 'Unknown')
                print(f"  [{i}] {source[:40]:40} (score: {score:.3f})")
        
        # Step 4: Build enhanced prompt
        prompt = self.build_enhanced_prompt(question, results)
        
        # Step 5: Generate answer
        if verbose:
            print(f"\n Stage 3: Generating answer...")
        
        gen_start = time.time()
        
        if stream:
            if verbose:
                print("\n" + "-"*70)
                print("Answer:")
                print("-"*70 + "\n")
            
            response_text = ""
            for chunk in ollama.generate(
                model=self.model_name, 
                prompt=prompt, 
                stream=True,
                options={'temperature': 0.3, 'top_p': 0.9}
            ):
                token = chunk['response']
                if verbose:
                    print(token, end="", flush=True)
                response_text += token
            
            if verbose:
                print("\n")
        else:
            response = ollama.generate(
                model=self.model_name, 
                prompt=prompt,
                options={'temperature': 0.3, 'top_p': 0.9}
            )
            response_text = response['response']
        
        timing['generation'] = time.time() - gen_start
        timing['total'] = time.time() - start_time
        
        # Prepare result
        result = {
            'answer': response_text,
            'sources': [{
                'text': r['text'][:200] + "...",
                'source': r['metadata'].get('source', 'Unknown'),
                'score': r.get('final_score', r.get('score', 0))
            } for r in results],
            'timing': timing,
            'question': question
        }
        
        # Cache result
        if len(self.query_cache) >= self.max_cache_size:
            # Remove oldest entry
            self.query_cache.pop(next(iter(self.query_cache)))
        self.query_cache[cache_key] = result
        
        if verbose:
            print("="*70)
            print(f"  Timing: Retrieval={timing['retrieval']:.2f}s | "
                  f"Rerank={timing.get('rerank', 0):.2f}s | "
                  f"Generation={timing['generation']:.2f}s | "
                  f"Total={timing['total']:.2f}s")
            print("="*70)
        
        return result
    
    def compare_modes(self, question: str):
        """
        Compare different retrieval modes for analysis.
        """
        print("\n" + "="*70)
        print("🔬 COMPARISON MODE")
        print("="*70)
        
        modes = [
            ("Simple Dense", False, False, False),
            ("Dense + BM25", True, False, False),
            ("Dense + BM25 + Rerank", True, True, False),
            ("Full (HyDE + Hybrid + Rerank)", True, True, True),
        ]
        
        for mode_name, use_hybrid, use_rerank, use_hyde in modes:
            print(f"\n{'─'*70}")
            print(f"Mode: {mode_name}")
            print('─'*70)
            
            result = self.query(
                question,
                top_k=3,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                use_hyde=use_hyde,
                stream=False,
                verbose=False
            )
            
            print(f"Time: {result['timing']['total']:.2f}s")
            print(f"Answer: {result['answer'][:200]}...")
    
    def interactive(self):
        """Interactive query loop with advanced features."""
        print("\n" + "="*70)
        print(" Advanced AI Research Assistant")
        print("="*70)
        print("\nCommands:")
        print("  - Type your question normally")
        print("  - 'compare <question>' - Compare different retrieval modes")
        print("  - 'simple <question>' - Use simple mode (faster)")
        print("  - 'clear' - Clear cache")
        print("  - 'quit' - Exit")
        print("="*70)
        
        while True:
            user_input = input("\n❯ ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                self.query_cache.clear()
                print("✓ Cache cleared")
                continue
            
            try:
                if user_input.startswith('compare '):
                    question = user_input[8:].strip()
                    self.compare_modes(question)
                
                elif user_input.startswith('simple '):
                    question = user_input[7:].strip()
                    self.query(
                        question, 
                        use_hybrid=False, 
                        use_rerank=False, 
                        use_hyde=False,
                        stream=True
                    )
                
                else:
                    self.query(user_input, stream=True)
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    rag = AdvancedRAGSystem()
    rag.interactive()