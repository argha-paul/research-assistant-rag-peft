import argparse
from src.ingestion.download_papers import download_arxiv_papers
from src.ingestion.process_documents import FastDocumentProcessor
from src.retrieval.vector_store import FastVectorStore

def main():
    parser = argparse.ArgumentParser(
        description="AI Research Assistant with Advanced RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup pipeline
  python main.py download --papers 50
  python main.py process
  python main.py embed
  
  # Query (different modes)
  python main.py query                    # Advanced RAG (interactive)
  python main.py query --mode simple      # Basic RAG (faster)
  python main.py query --question "What is RAG?"
  
  # Benchmarking
  python main.py benchmark                # Compare basic vs advanced
  python main.py benchmark --modes        # Compare advanced modes
        """
    )
    
    parser.add_argument(
        'command', 
        choices=['download', 'process', 'embed', 'query', 'benchmark'],
        help='Command to execute'
    )
    
    # Download options
    parser.add_argument(
        '--papers', 
        type=int, 
        default=50, 
        help='Number of papers to download (default: 50)'
    )
    parser.add_argument(
        '--topics',
        nargs='+',
        default=["large language models", "RAG", "transformer"],
        help='Topics to search for'
    )
    
    # Query options
    parser.add_argument(
        '--question', 
        type=str, 
        help='Question to ask (non-interactive mode)'
    )
    parser.add_argument(
        '--mode',
        choices=['simple', 'advanced'],
        default='advanced',
        help='RAG mode: simple (fast) or advanced (high quality)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of documents to retrieve (default: 5)'
    )
    
    # Advanced RAG options
    parser.add_argument(
        '--no-hybrid',
        action='store_true',
        help='Disable hybrid retrieval (use dense only)'
    )
    parser.add_argument(
        '--no-rerank',
        action='store_true',
        help='Disable re-ranking'
    )
    parser.add_argument(
        '--no-hyde',
        action='store_true',
        help='Disable HyDE query enhancement'
    )
    
    # Benchmark options
    parser.add_argument(
        '--modes',
        action='store_true',
        help='Benchmark different advanced RAG modes'
    )
    
    args = parser.parse_args()
    
    # Execute command
    if args.command == 'download':
        print(f"Downloading papers on topics: {args.topics}")
        papers_per_topic = args.papers // len(args.topics)
        
        for topic in args.topics:
            print(f"\n🔍 Searching for: {topic}")
            download_arxiv_papers(topic, max_results=papers_per_topic)
        
        print(f"\nDownload complete!")
    
    elif args.command == 'process':
        print("⚙️  Processing documents...")
        processor = FastDocumentProcessor()
        processor.process_directory("data/raw/papers", "data/processed/chunks.json")
        print("Processing complete!")
    
    elif args.command == 'embed':
        print("🔢 Creating embeddings...")
        store = FastVectorStore()
        store.embed_and_store("data/processed/chunks.json")
        print("Embeddings created!")
    
    elif args.command == 'query':
        if args.mode == 'simple':
            print("Starting Basic RAG (Fast Mode)...")
            from src.inference.rag_query import FastRAGSystem
            rag = FastRAGSystem(model_name="tinyllama")
            
            if args.question:
                rag.query(args.question, top_k=args.top_k)
            else:
                rag.interactive()
        
        else:  # advanced mode
            print("🚀 Starting Advanced RAG (High Quality Mode)...")
            from src.inference.advanced_rag import AdvancedRAGSystem
            rag = AdvancedRAGSystem(model_name="tinyllama")
            
            if args.question:
                rag.query(
                    args.question,
                    top_k=args.top_k,
                    use_hybrid=not args.no_hybrid,
                    use_rerank=not args.no_rerank,
                    use_hyde=not args.no_hyde,
                    stream=True
                )
            else:
                rag.interactive()
    
    elif args.command == 'benchmark':
        if args.modes:
            print("Benchmarking Advanced RAG modes...")
            from src.benchmark_advanced import benchmark_modes
            benchmark_modes()
        else:
            print("Benchmarking Basic vs Advanced RAG...")
            from src.benchmark_advanced import benchmark_comparison
            benchmark_comparison()

if __name__ == "__main__":
    main()