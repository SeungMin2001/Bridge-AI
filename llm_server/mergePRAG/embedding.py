from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def embedding(MODEL_NAME, text):
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    text = "This is a passage for hypernetwork input."

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Qwen의 입력 임베딩 사용
    embeddings = model.get_input_embeddings()(input_ids)   # [B, T, d]

    print(embeddings.shape)
    
embedding("Qwen/Qwen3.5-9B")