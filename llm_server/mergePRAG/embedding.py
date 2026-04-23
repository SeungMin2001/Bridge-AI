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


def encode_passage_states(model, input_ids, attention_mask=None, use_contextual=False):
    """논문 `KV_train.py`: `model.model.embed_tokens(input_ids)` 사용 (token embed only).

    use_contextual=True는 실험적 옵션 (Qwen 전체를 통과시켜 hidden state를 씀).
    논문 재현이 목적이라면 False가 기본.
    """
    if use_contextual:
        return contextualize(model, input_ids, attention_mask=attention_mask)
    return token_embed(model, input_ids)


def tokenize_conditioned_memory(tokenizer, question, passage, device, max_length=512):
    """Build a question-conditioned memory sequence and token masks.

    The hypernetwork should focus on passage tokens, but the question tells it
    which parts of the passage matter. To keep those roles separate we return
    masks for question tokens and passage tokens.
    """
    segments = [
        (tokenizer("Question:", add_special_tokens=False)["input_ids"], False, False),
        (tokenizer(f" {question}\n", add_special_tokens=False)["input_ids"], True, False),
        (tokenizer("Passage:", add_special_tokens=False)["input_ids"], False, False),
        (tokenizer(f" {passage}", add_special_tokens=False)["input_ids"], False, True),
    ]

    input_ids = []
    question_mask = []
    passage_mask = []

    for token_ids, is_question, is_passage in segments:
        if not token_ids:
            continue
        remaining = max_length - len(input_ids)
        if remaining <= 0:
            break
        token_ids = token_ids[:remaining]
        input_ids.extend(token_ids)
        question_mask.extend([1 if is_question else 0] * len(token_ids))
        passage_mask.extend([1 if is_passage else 0] * len(token_ids))

    if not input_ids:
        input_ids = [tokenizer.eos_token_id]
        question_mask = [0]
        passage_mask = [1]

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    question_mask = torch.tensor([question_mask], dtype=torch.long, device=device)
    passage_mask = torch.tensor([passage_mask], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "question_mask": question_mask,
        "passage_mask": passage_mask,
    }


def tokenize_passage_memory(tokenizer, passage, device, max_length=512):
    encoded = tokenizer(
        passage,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    zero_mask = torch.zeros_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "question_mask": zero_mask,
        "passage_mask": attention_mask.clone(),
    }


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
