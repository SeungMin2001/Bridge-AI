import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

def run_model():
    MODEL_NAME = "google/gemma-4-26B-A4B-it"  # Gemma 4 26B A4B (MoE, 4B activated)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print("Loading tokenizer...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)


    print("Loading model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",

    )

    model.eval()
    print("model ready")
    return model, processor
