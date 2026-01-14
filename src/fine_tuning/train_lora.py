import torch
import os
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import json

# CRITICAL: Set memory limits for MPS
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

def format_instruction(example):
    """Format training example."""
    return {
        'text': f"""### Context:
{example['context'][:300]}

### Question:
{example['instruction']}

### Answer:
{example['response']}<|endoftext|>"""
    }

def train_lora_minimal_memory():
    """
    ULTRA LOW MEMORY training.
    - Minimal LoRA rank
    - Batch size 1
    - Gradient checkpointing
    - Short sequences
    - Aggressive memory cleanup
    """
    
    print("Ultra-low memory training mode")
    print("="*60)
    
    # Clear any existing cache
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("\n1. Loading TinyLlama model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load model with minimal memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  
        device_map="mps",           
        low_cpu_mem_usage=True,     
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing (saves memory)
    model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print("✓ Model loaded")
    
    # MINIMAL LoRA config for low memory
    print("\n2. Applying LoRA (minimal rank for memory)...")
    lora_config = LoraConfig(
        r=4,                        
        lora_alpha=8,               
        target_modules=["q_proj", "v_proj"],  
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    print("✓ LoRA applied")
    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    
    # Clear cache after model loading
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Load training data
    print("\n3. Loading training data...")
    with open('data/processed/training_data.json', 'r') as f:
        data = json.load(f)
    
    # REDUCE dataset size for memory
    data = data[:200]  
    print(f"Using {len(data)} training examples (reduced for memory)")
    
    # Save as JSONL
    with open('data/processed/train.jsonl', 'w') as f:
        for item in data:
            f.write(json.dumps(format_instruction(item)) + '\n')
    
    dataset = load_dataset('json', data_files='data/processed/train.jsonl', split='train')
    
    # Tokenize with SHORT sequences
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=256,              
            padding='max_length'
        )
    
    print("✓ Tokenizing...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=dataset.column_names,
        batch_size=10
    )
    
    # Clear cache after tokenization
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # ULTRA LOW MEMORY training arguments
    print("\n4. Configuring training...")
    training_args = TrainingArguments(
        output_dir="models/lora_adapters",
        num_train_epochs=1,                      
        per_device_train_batch_size=1,          
        gradient_accumulation_steps=16,          
        learning_rate=5e-4,                      
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        save_total_limit=1,                      
        warmup_steps=20,
        gradient_checkpointing=True,             
        max_grad_norm=0.3,
        dataloader_num_workers=0,                
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Custom Trainer with memory cleanup
    class MemoryEfficientTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            # Standard training step
            loss = super().training_step(model, inputs, num_items_in_batch)
            
            # Clear cache every step
            if self.state.global_step % 5 == 0:
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            return loss
    
    # Trainer
    trainer = MemoryEfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train!
    print("\n5. Starting training...")
    print("="*60)
    print("Estimated time: 30-45 minutes")
    print("="*60)
    
    try:
        trainer.train()
        print("\nTraining complete!")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        print("\nTrying to save checkpoint...")
    
    # Save adapters
    try:
        model.save_pretrained("models/lora_adapters")
        tokenizer.save_pretrained("models/lora_adapters")
        print("✓ LoRA adapters saved to models/lora_adapters")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Final cleanup
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

if __name__ == "__main__":
    train_lora_minimal_memory()