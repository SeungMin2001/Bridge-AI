import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_model():
    MODEL_NAME = "Qwen/Qwen2.5-3B"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print("Loading model (float16)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model.eval()
    print("model ready")
    return model, tokenizer
