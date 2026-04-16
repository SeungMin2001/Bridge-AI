import torch


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
    """Qwen의 contextual hidden state를 반환. [B, T, d_model]."""
    device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    return contextualize(model, input_ids, attention_mask)
