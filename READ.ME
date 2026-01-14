# AI Research Assistant with Advanced RAG & LoRA Fine-tuning

A production-grade toolkit featuring two powerful standalone systems: (1) An advanced RAG system for intelligent document querying with citation-backed answers, and (2) A memory-optimized LoRA fine-tuning pipeline for adapting language models. Built for researchers, developers, and AI enthusiasts.

## ✨ Features

This project provides two independent systems that can be used separately or together:

### 🔍 Advanced RAG System
Intelligent document querying with state-of-the-art retrieval techniques:

- **🔍 Hybrid Retrieval**: Combines dense semantic search (embeddings) with sparse keyword search (BM25) for superior recall
- **🎯 Cross-Encoder Re-ranking**: Precision re-ranking using cross-encoder models for the most relevant results
- **🚀 HyDE (Hypothetical Document Embeddings)**: Query enhancement technique that generates synthetic answers to improve retrieval
- **💾 Smart Caching**: LRU caching for repeated queries with instant responses
- **📝 Source Attribution**: Automatic citation tracking with confidence scores
- **⚡ Context Compression**: Intelligent filtering and truncation to maximize relevant information

### 💡 LoRA Fine-tuning Pipeline
Memory-efficient language model adaptation:

- **💡 Parameter-Efficient Training**: Fine-tune large language models with minimal memory footprint
- **🎛️ Memory-Optimized**: Gradient checkpointing, mixed precision, and aggressive memory management for M-series Macs
- **📊 Custom Dataset Support**: Easy adaptation to domain-specific knowledge
- **🔄 Adapter Architecture**: Swap LoRA adapters without reloading base models
- **📈 Training Data Generation**: Automated generation of instruction-tuning datasets

## 🏗️ Architecture

This project consists of two independent systems:

### 1️⃣ Advanced RAG System

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Query Enhancement (HyDE)   │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │    Hybrid Retrieval          │
         │  ┌──────────┐  ┌──────────┐ │
         │  │  Dense   │  │   BM25   │ │
         │  │ (Vector) │  │(Keyword) │ │
         │  └────┬─────┘  └─────┬────┘ │
         └───────┼──────────────┼──────┘
                 └──────┬───────┘
                        ▼
         ┌─────────────────────────────┐
         │  Cross-Encoder Re-ranking    │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │   Context Compression        │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │    LLM Generation            │
         │       (Ollama)               │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  Answer + Citations          │
         └─────────────────────────────┘
```

### 2️⃣ LoRA Fine-tuning Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│           Raw Training Data (JSON)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  Data Formatting & Tokenize  │
         │  (generate_training_data.py) │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  Base Model Loading          │
         │  (TinyLlama-1.1B-Chat)       │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  LoRA Adapter Injection      │
         │  (Low-Rank Matrices)         │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  Memory-Efficient Training   │
         │  (train_lora.py)             │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  Save LoRA Adapters          │
         │  (models/lora_adapters/)     │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  Inference Demo               │
         │  (demo_lora_inference.py)    │
         └─────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
python --version

# For RAG System: Ollama (for inference)
# Install from: https://ollama.ai
ollama pull tinyllama
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-research-assistant.git
cd ai-research-assistant

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw/papers data/processed data/embeddings models/lora_adapters
```

---

## 📖 Usage Guide

This project provides two independent systems. Choose what you need:

### 🔍 Option 1: Advanced RAG System

Perfect for querying research papers and documents with intelligent retrieval.

#### Setup Pipeline

```bash
# 1. Download research papers from arXiv
python main.py download --papers 50 --topics "large language models" "RAG" "transformer"

# 2. Process and chunk documents
python main.py process

# 3. Create vector embeddings
python main.py embed
```

#### Interactive Query

```bash
# Advanced RAG with all features (recommended)
python main.py query

# Simple mode (faster, basic retrieval)
python main.py query --mode simple

# Single question
python main.py query --question "What is retrieval augmented generation?"
```

#### Advanced Options

```bash
# Fine-tune retrieval settings
python main.py query \
  --top-k 10 \
  --no-hyde \
  --no-rerank

# Available flags:
#   --no-hybrid   : Disable BM25, use dense retrieval only
#   --no-rerank   : Skip cross-encoder re-ranking
#   --no-hyde     : Disable query enhancement
```

#### Benchmarking RAG Performance

```bash
# Compare basic vs advanced RAG
python main.py benchmark

# Compare different advanced RAG modes
python main.py benchmark --modes
```

---

### 💡 Option 2: LoRA Fine-tuning Pipeline

Perfect for adapting language models to your specific domain or task.

#### 1. Prepare Training Data

Create training data in JSON format (see `data/processed/training_data.json`):

```json
[
  {
    "instruction": "Explain transformers",
    "response": "Transformers are neural network architectures...",
    "context": "Optional context from papers..."
  }
]
```

Or generate training data from your documents:

```bash
python src/fine_tuning/generate_training_data.py
```

#### 2. Train LoRA Adapters

```bash
# Run memory-optimized training
python src/fine_tuning/train_lora.py
```

**Training Output:**
- Adapters saved to: `models/lora_adapters/`
- Training time: ~30-45 minutes (200 examples on M-series Mac)

#### 3. Test Fine-tuned Model

```bash
# Compare base model vs LoRA-adapted model
python src/fine_tuning/demo_lora_inference.py
```

This will show side-by-side outputs for sample queries.

### Memory Requirements

| System | Component | RAM | VRAM/Unified Memory |
|--------|-----------|-----|---------------------|
| **RAG** | Inference | 4GB | 6GB |
| **LoRA** | Training | 4GB | 8GB |
| **LoRA** | Inference | 2GB | 4GB |

## 📊 Benchmarking

### RAG System Performance

```bash
# Basic vs Advanced RAG comparison
python main.py benchmark

# Compare different advanced RAG modes
python main.py benchmark --modes

# Or run benchmark script directly
python src/benchmark_advanced.py
```

**Sample Results:**

```
Mode                                    Avg Time    Quality
────────────────────────────────────────────────────────────
Simple Dense Only                       1.2s        ⭐⭐⭐
Dense + BM25 Hybrid                     1.8s        ⭐⭐⭐⭐
Hybrid + Re-ranking                     2.3s        ⭐⭐⭐⭐⭐
Full Pipeline (HyDE + Hybrid + Rerank)  2.8s        ⭐⭐⭐⭐⭐
```

### LoRA Model Comparison

Compare base model vs fine-tuned outputs:

```bash
python src/fine_tuning/demo_lora_inference.py
```

## 🛠️ Project Structure

```
ai-research-assistant/
├── main.py                              # RAG system entry point
├── requirements.txt                     # Python dependencies
│
├── data/
│   ├── embeddings/                      # ChromaDB vector store
│   ├── processed/
│   │   ├── chunks.json                  # Processed document chunks
│   │   └── training_data.json           # LoRA training data
│   └── raw/
│       └── papers/                      # Downloaded PDFs
│
├── models/
│   └── lora_adapters/                   # Trained LoRA weights
│
├── ragft/                               # Package directory
│
└── src/
    ├── fine_tuning/                     # 💡 LoRA Fine-tuning System
    │   ├── demo_lora_inference.py       # ↳ Compare base vs LoRA models
    │   ├── generate_training_data.py    # ↳ Create training datasets
    │   └── train_lora.py                # ↳ Train LoRA adapters
    │
    ├── inference/                       # 🔍 RAG Query Systems
    │   ├── advanced_rag.py              # ↳ Advanced RAG with all features
    │   └── rag_query.py                 # ↳ Basic RAG (fast mode)
    │
    ├── ingestion/                       # 📥 Document Processing
    │   ├── download_papers.py           # ↳ arXiv paper downloader
    │   └── process_documents.py         # ↳ PDF chunking & processing
    │
    ├── retrieval/                       # 🗄️ Vector Storage
    │   └── vector_store.py              # ↳ ChromaDB interface
    │
    └── benchmark_advanced.py            # 📊 RAG benchmarking suite
```

### Key Files

#### RAG System (🔍)
- `main.py` - CLI interface for RAG operations
- `src/inference/advanced_rag.py` - Core advanced RAG logic
- `src/inference/rag_query.py` - Simple/fast RAG mode
- `src/benchmark_advanced.py` - Performance comparison

#### LoRA System (💡)
- `src/fine_tuning/train_lora.py` - Training pipeline
- `src/fine_tuning/demo_lora_inference.py` - Inference demo
- `src/fine_tuning/generate_training_data.py` - Dataset creation

## 🔧 Configuration

### RAG System Configuration

```python
# src/inference/advanced_rag.py
rag = AdvancedRAGSystem(
    model_name="tinyllama",                           # Ollama model
    embedding_model="all-MiniLM-L6-v2",              # Sentence transformer
    reranker_model="cross-encoder/ms-marco-MiniLM-L-2-v2",  # Re-ranker
    persist_dir="data/embeddings"                     # Vector DB path
)
```

### LoRA Training Configuration

```python
# src/fine_tuning/train_lora.py
lora_config = LoraConfig(
    r=4,                              # LoRA rank (lower = less memory)
    lora_alpha=8,                     # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,                # Regularization
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="models/lora_adapters",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-4,
    fp16=False,                       # Set to True for GPU
    gradient_checkpointing=True,      # Memory optimization
)
```

## 📚 Key Technologies

### RAG System
- **LLM**: Ollama (TinyLlama)
- **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
- **Re-ranking**: Cross-Encoder (ms-marco-MiniLM-L-2)
- **Vector Store**: ChromaDB
- **Sparse Retrieval**: BM25 (rank-bm25)
- **Document Processing**: PyPDF2, arxiv API

### LoRA Fine-tuning
- **Base Model**: TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning**: LoRA via PEFT (Hugging Face)
- **Training**: Transformers Trainer API
- **Optimization**: Gradient checkpointing, mixed precision

## 🎯 Use Cases

### RAG System
- **Research Literature Review**: Query thousands of papers instantly with citations
- **Technical Documentation**: Build searchable internal knowledge bases
- **Academic Research**: Automated literature surveys with source attribution
- **Competitive Intelligence**: Analyze industry research trends

### LoRA Fine-tuning
- **Domain Adaptation**: Specialize models for medical, legal, or technical domains
- **Task-Specific Tuning**: Create models for summarization, Q&A, or code generation
- **Style Transfer**: Adapt writing style for specific audiences or formats
- **Low-Resource Languages**: Fine-tune for languages with limited data

## 🔍 Example Usage

### RAG System Queries

```python
# Complex research question
"What are the main architectural differences between BERT and GPT models, 
 and how do they affect downstream task performance?"

# Technical implementation
"Explain the attention mechanism in transformers with mathematical formulation"

# Comparative analysis
"Compare retrieval augmented generation with traditional fine-tuning approaches"
```

### LoRA Training Examples

```python
# Example training data format
{
  "instruction": "Explain the transformer architecture",
  "response": "The transformer is a neural network architecture that relies on self-attention mechanisms...",
  "context": "From 'Attention is All You Need' paper..."
}

# Domain-specific example (Medical)
{
  "instruction": "What are the symptoms of condition X?",
  "response": "Common symptoms include...",
  "context": "Medical literature excerpt..."
}
```

## 📈 Performance Tips

### RAG System
1. **Retrieval Quality**: Enable all features (hybrid, rerank, HyDE) for best results
2. **Speed**: Use `--mode simple` or disable `--no-rerank` for faster responses
3. **Memory**: Reduce `top_k` parameter if running on limited resources
4. **Accuracy**: Increase `top_k` to 10-20 for comprehensive answers

### LoRA Fine-tuning
1. **Memory Issues**: Reduce `per_device_train_batch_size` or `r` (rank) parameter
2. **Training Speed**: Increase `gradient_accumulation_steps` for larger effective batch size
3. **Quality**: Train for more epochs with domain-specific data
4. **Overfitting**: Use dropout and limit training data size

## 🤝 Contributing

Contributions welcome! Areas of interest:

### RAG System
- Additional retrieval algorithms (ColBERT, SPLADE)
- More LLM backend support (GPT-4, Claude API, local models)
- Query reformulation techniques
- Evaluation metrics and datasets

### LoRA Fine-tuning
- Support for other base models (Llama 2, Mistral, etc.)
- Multi-GPU training support
- Quantization (QLoRA) implementation
- Automated hyperparameter tuning

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- [Sentence-Transformers](https://www.sbert.net/) for embedding models
- [Hugging Face](https://huggingface.co/) for PEFT and Transformers
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Ollama](https://ollama.ai/) for local LLM inference

## 📞 Contact

For questions or collaboration:
- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/ai-research-assistant/issues)
- Email: your.email@example.com

---

⭐ Star this repo if you find it useful! PRs welcome.