import ollama
from src.retrieval.vector_store import FastVectorStore
import time

class FastRAGSystem:
    """
    Speed-optimized RAG system.
    - Direct Ollama integration
    - Simple prompt
    - No post-processing overhead
    """
    
    def __init__(self, model_name="llama3.2:1b"):
        self.model_name = model_name
        self.vector_store = FastVectorStore()
        print(f"✓ RAG System ready with {model_name}")
    
    def query(self, question, top_k=3, stream=True):
        """
        Fast query with minimal overhead.
        
        Args:
            question: User question
            top_k: Number of context chunks (fewer = faster)
            stream: Stream response for better UX
        """
        start_time = time.time()
        
        # Step 1: Retrieve context (fast)
        results = self.vector_store.search(question, top_k=top_k)
        
        retrieval_time = time.time() - start_time
        
        # Step 2: Build simple prompt
        context = "\n\n".join(results['documents'][0])
        
        prompt = f"""Based on the following research paper excerpts, answer the question.

Context:
{context}

Question: {question}

Answer concisely based only on the context above:"""
        
        # Step 3: Generate answer
        gen_start = time.time()
        
        if stream:
            print(f"\n[Retrieved in {retrieval_time:.2f}s]\n")
            print("Answer: ", end="", flush=True)
            
            response_text = ""
            for chunk in ollama.generate(model=self.model_name, prompt=prompt, stream=True):
                token = chunk['response']
                print(token, end="", flush=True)
                response_text += token
            
            print(f"\n\n[Generated in {time.time() - gen_start:.2f}s]")
            return response_text
        
        else:
            response = ollama.generate(model=self.model_name, prompt=prompt)
            total_time = time.time() - start_time
            
            print(f"\n[Total time: {total_time:.2f}s]")
            return response['response']
    
    def interactive(self):
        """Interactive query loop."""
        print("\n" + "="*60)
        print("AI Research Assistant (Speed Mode)")
        print("="*60)
        print("Type your question or 'quit' to exit\n")
        
        while True:
            question = input("\n❯ Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            try:
                self.query(question, top_k=3, stream=True)
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    rag = FastRAGSystem()
    rag.interactive()