import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Rag를 통해 나온 유사한 벡터 데이터를 받고 프론트엔드에 전달 코드 구현

def run_model():
    MODEL_NAME = "Qwen/Qwen3.5-9B"

    if torch.cuda.is_available():
        device="cuda"
    elif torch.backends.mps.is_available():
        device= "mps"
    else:
        device="cpu"

    print("device: ",device)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    print("model ready")

    print("tokenizer loaded:", type(tokenizer))
    print("model loaded:", type(model))
    return model, tokenizer


