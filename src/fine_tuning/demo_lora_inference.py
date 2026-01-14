import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ===== CONFIG =====
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "models/lora_adapters"   # same path used in train_lora.py
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ===== LOAD TOKENIZER =====
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ===== LOAD BASE MODEL =====
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map=DEVICE
)

# ===== LOAD LORA ADAPTER =====
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# ===== GENERATION FUNCTION =====
def generate(prompt, max_new_tokens=256):
    # ---- CHAT TEMPLATE (CRITICAL) ----
    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,        
            temperature=0.3,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )

def generate_base(prompt, max_new_tokens=256):
    # ---- CHAT TEMPLATE ----
    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = base_model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,       
            temperature=0.3,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )


# def generate_base(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
#     with torch.no_grad():
#         outputs = base_model.generate(
#             **inputs,
#             max_new_tokens=256,
#             temperature=0.2
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ===== DEMO QUERIES =====
if __name__ == "__main__":
    prompts = [
        "Explain Retrieval-Augmented Generation in simple terms.",
        "Summarize the key limitations of transformer-based language models.",
        "What are open research problems in large language models?"
    ]

    for i, p in enumerate(prompts, 1):
        print("\n" + "=" * 80)
        print(f"Prompt {i}: {p}")
        print("-" * 80)
        print("\n--- BASE MODEL OUTPUT ---")
        print(generate_base(p))
        print("-" * 80)
        print("\n--- LORA MODEL OUTPUT ---")
        print(generate(p))
