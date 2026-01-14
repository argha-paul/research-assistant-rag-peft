import time
import json
from src.inference.rag_query import FastRAGSystem
from src.inference.advanced_rag import AdvancedRAGSystem

def benchmark_comparison():
    """
    Compare basic RAG vs Advanced RAG performance.
    """
    
    # Test questions
    test_questions = [
        "What is a transformer architecture?",
        "Explain how attention mechanism works in neural networks",
        "What are the main differences between BERT and GPT models?",
        "How does fine-tuning improve language model performance?",
        "What is retrieval augmented generation and why is it useful?",
        "Explain the concept of positional encoding",
        "What is the role of layer normalization in transformers?",
        "How do you prevent overfitting in deep learning?",
    ]
    
    print("\n" + "="*80)
    print("🔬 RAG SYSTEM BENCHMARK - Basic vs Advanced")
    print("="*80)
    
    # Initialize both systems
    print("\n📦 Initializing systems...")
    try:
        basic_rag = FastRAGSystem(model_name="tinyllama")
        print("✓ Basic RAG loaded")
    except Exception as e:
        print(f"❌ Could not load basic RAG: {e}")
        basic_rag = None
    
    try:
        advanced_rag = AdvancedRAGSystem(model_name="tinyllama")
        print("✓ Advanced RAG loaded")
    except Exception as e:
        print(f"❌ Could not load advanced RAG: {e}")
        advanced_rag = None
    
    if not advanced_rag:
        print("\n⚠️  Advanced RAG not available. Exiting.")
        return
    
    results = {
        'basic': {'times': [], 'answers': []},
        'advanced': {'times': [], 'answers': []}
    }
    
    # Test each question
    for i, question in enumerate(test_questions[:5], 1):  # Test on 5 questions
        print("\n" + "─"*80)
        print(f"Question {i}/5: {question}")
        print("─"*80)
        
        # Test Basic RAG
        if basic_rag:
            print("\n[Basic RAG]")
            try:
                start = time.time()
                answer = basic_rag.query(question, top_k=3, stream=False)
                elapsed = time.time() - start
                
                results['basic']['times'].append(elapsed)
                results['basic']['answers'].append(answer[:150])
                
                print(f"Time: {elapsed:.2f}s")
                print(f"Answer: {answer[:150]}...")
            except Exception as e:
                print(f"Error: {e}")
        
        # Test Advanced RAG
        print("\n[Advanced RAG]")
        try:
            result = advanced_rag.query(
                question, 
                top_k=5,
                use_hybrid=True,
                use_rerank=True,
                use_hyde=True,
                stream=False,
                verbose=False
            )
            
            results['advanced']['times'].append(result['timing']['total'])
            results['advanced']['answers'].append(result['answer'][:150])
            
            print(f"Time: {result['timing']['total']:.2f}s")
            print(f"  ├─ Retrieval: {result['timing']['retrieval']:.2f}s")
            print(f"  ├─ Rerank: {result['timing'].get('rerank', 0):.2f}s")
            print(f"  └─ Generation: {result['timing']['generation']:.2f}s")
            print(f"Answer: {result['answer'][:150]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    # Summary statistics
    print("\n\n" + "="*80)
    print("📊 BENCHMARK RESULTS")
    print("="*80)
    
    if results['basic']['times']:
        basic_avg = sum(results['basic']['times']) / len(results['basic']['times'])
        print(f"\n🔵 Basic RAG:")
        print(f"  Average time: {basic_avg:.2f}s")
        print(f"  Min time: {min(results['basic']['times']):.2f}s")
        print(f"  Max time: {max(results['basic']['times']):.2f}s")
    
    if results['advanced']['times']:
        adv_avg = sum(results['advanced']['times']) / len(results['advanced']['times'])
        print(f"\n Advanced RAG:")
        print(f"  Average time: {adv_avg:.2f}s")
        print(f"  Min time: {min(results['advanced']['times']):.2f}s")
        print(f"  Max time: {max(results['advanced']['times']):.2f}s")
        
        if results['basic']['times']:
            speedup = (basic_avg / adv_avg - 1) * 100
            if speedup > 0:
                print(f"\n⚡ Advanced RAG is {speedup:.1f}% faster")
            else:
                print(f"\n  Advanced RAG is {abs(speedup):.1f}% slower (but much higher quality)")
    
    print("\n" + "="*80)
    
    # Save results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✓ Detailed results saved to benchmark_results.json")

def benchmark_modes():
    """
    Benchmark different modes of advanced RAG.
    """
    print("\n" + "="*80)
    print("🔬 ADVANCED RAG MODE COMPARISON")
    print("="*80)
    
    advanced_rag = AdvancedRAGSystem(model_name="tinyllama")
    
    question = "What is retrieval augmented generation and how does it work?"
    
    modes = [
        ("Simple Dense Only", False, False, False),
        ("Hybrid (Dense + BM25)", True, False, False),
        ("Hybrid + Re-ranking", True, True, False),
        ("Full Pipeline (HyDE + Hybrid + Rerank)", True, True, True),
    ]
    
    results = []
    
    for mode_name, use_hybrid, use_rerank, use_hyde in modes:
        print(f"\n{'─'*80}")
        print(f"📍 Mode: {mode_name}")
        print('─'*80)
        
        try:
            result = advanced_rag.query(
                question,
                top_k=5,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                use_hyde=use_hyde,
                stream=False,
                verbose=True
            )
            
            results.append({
                'mode': mode_name,
                'time': result['timing']['total'],
                'answer_length': len(result['answer'])
            })
        except Exception as e:
            print(f"Error: {e}")
    
    # Summary
    print("\n\n" + "="*80)
    print(" MODE COMPARISON SUMMARY")
    print("="*80)
    
    for r in results:
        print(f"\n{r['mode']}:")
        print(f"  Time: {r['time']:.2f}s")
        print(f"  Answer length: {r['answer_length']} chars")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "modes":
        benchmark_modes()
    else:
        benchmark_comparison()