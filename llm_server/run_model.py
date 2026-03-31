import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def run_model():
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"  # 9B → 3B로 교체 (속도 3배 향상)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    print("model ready")
    return model, tokenizer
