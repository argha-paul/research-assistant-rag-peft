import json
import ollama
from tqdm import tqdm
import random

def generate_qa_pairs(chunks_file, num_samples=500):
    """
    Generate Q&A pairs from chunks using Llama.
    """
    # Load chunks
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Sample random chunks
    sampled = random.sample(chunks, min(num_samples, len(chunks)))
    
    qa_pairs = []
    
    print(f"Generating {num_samples} Q&A pairs...")
    for chunk in tqdm(sampled[:num_samples]):
        prompt = f"""Based on this text, generate ONE clear question and answer.

Text: {chunk['text'][:500]}

Format your response as:
Q: [question]
A: [answer]"""
        
        try:
            response = ollama.generate(
                model='llama3.2:1b',
                prompt=prompt,
                options={'temperature': 0.7, 'num_predict': 150}
            )
            
            text = response['response']
            
            # Simple parsing
            if 'Q:' in text and 'A:' in text:
                parts = text.split('A:')
                question = parts[0].replace('Q:', '').strip()
                answer = parts[1].strip()
                
                qa_pairs.append({
                    'instruction': question,
                    'context': chunk['text'][:400],
                    'response': answer,
                    'source': chunk['source']
                })
        
        except Exception as e:
            continue
    
    # Save training data
    output_file = 'data/processed/training_data.json'
    with open(output_file, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"\n✓ Generated {len(qa_pairs)} Q&A pairs")
    print(f"✓ Saved to {output_file}")
    
    return qa_pairs

if __name__ == "__main__":
    generate_qa_pairs('data/processed/chunks.json', num_samples=500)