import os
import json
from pypdf import PdfReader
from pathlib import Path
from tqdm import tqdm
import tiktoken

class FastDocumentProcessor:
    """
    Speed-optimized document processor.
    No fancy parsing - just extract text fast.
    """
    
    def __init__(self, chunk_size=300, chunk_overlap=50):
        """
        Smaller chunks = faster retrieval
        300 tokens ≈ 225 words ≈ 1-2 paragraphs
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF - fast and simple."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
            return ""
    
    def simple_chunk_text(self, text):
        """
        Simple token-based chunking - no semantic analysis.
        FAST: ~1000 docs/second
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if len(chunk_text.strip()) > 50:  # Skip tiny chunks
                chunks.append(chunk_text)
        
        return chunks
    
    def process_directory(self, input_dir, output_file):
        """Process all PDFs in directory."""
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob("**/*.pdf"))
        
        all_chunks = []
        
        print(f"Processing {len(pdf_files)} PDFs...")
        for pdf_file in tqdm(pdf_files):
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text:
                continue
            
            # Chunk text
            chunks = self.simple_chunk_text(text)
            
            # Store with metadata
            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'source': str(pdf_file.name),
                    'chunk_id': f"{pdf_file.stem}_chunk_{idx}",
                    'total_chunks': len(chunks)
                })
        
        # Save processed chunks
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2)
        
        print(f"\nProcessed {len(all_chunks)} chunks from {len(pdf_files)} papers")
        return all_chunks

if __name__ == "__main__":
    processor = FastDocumentProcessor(chunk_size=300, chunk_overlap=50)
    chunks = processor.process_directory(
        "data/raw/papers",
        "data/processed/chunks.json"
    )