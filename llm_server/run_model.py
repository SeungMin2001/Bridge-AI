import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

def run_model():
    MODEL_NAME = "Qwen/Qwen3.5-27B"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading tokenizer...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)


    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()
    print("model ready")
    return model, processor
