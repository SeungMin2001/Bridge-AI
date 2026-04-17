import torch


def token_embed(model, input_ids):
    """Frozen token embeddings from the base LLM embedding table."""
    with torch.no_grad():
        embedded = model.model.embed_tokens(input_ids)
    return embedded.to(dtype=torch.float32)


def contextualize(model, input_ids, attention_mask=None):
    """Frozen Qwen hidden states for passage encoding."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )

    return outputs.last_hidden_state.to(dtype=torch.float32)


def embedding(model, tokenizer, text):
    """Qwen token embeddings를 반환. [B, T, d_model]."""
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    input_ids = inputs["input_ids"].to(device)
    return token_embed(model, input_ids)
