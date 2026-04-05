import torch


def embedding(model, tokenizer, text):
    """Qwen의 입력 임베딩 레이어로 텍스트를 벡터화. [B, T, d_model] 반환."""
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        embeddings = model.get_input_embeddings()(input_ids)  # [B, T, d]

    return embeddings