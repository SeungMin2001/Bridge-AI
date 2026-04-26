import re

import torch

from .config import (
    QUERY_LEXICAL_FOCUS_WINDOW,
    USE_QUERY_LEXICAL_FOCUS,
    contains_hangul,
    select_memory_encoder_instruction,
)


_EN_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "said",
    "say",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    # Deadline/definition relation words tend to appear in distractor clauses too.
    # The object term ("homework", "project proposal", "process", ...) is a
    # better anchor for the local answer window.
    "due",
    "mean",
    "means",
}

_KO_QUERY_STOPWORDS = {
    "언제",
    "누가",
    "무엇",
    "뭐야",
    "뭐라고",
    "어디야",
    "제출",
    "기한",
    "설명",
    "했어",
}


def _extract_focus_terms(question: str) -> list[str]:
    """Return question terms that can anchor a passage-local answer window."""
    question = str(question or "")
    terms: list[str] = []

    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", question.lower()):
        if len(word) < 3 or word in _EN_QUERY_STOPWORDS:
            continue
        terms.append(word)

    for word in re.findall(r"[가-힣]{2,}", question):
        if word in _KO_QUERY_STOPWORDS:
            continue
        terms.append(word)

    seen = set()
    unique_terms = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        unique_terms.append(term)
    return unique_terms[:12]


def _find_term_spans(text: str, terms: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    lowered = text.lower()
    for term in terms:
        if re.fullmatch(r"[A-Za-z0-9_-]+", term):
            pattern = r"(?<![A-Za-z0-9_-])" + re.escape(term) + r"(?![A-Za-z0-9_-])"
            spans.extend(match.span() for match in re.finditer(pattern, lowered))
            continue

        start = 0
        while True:
            idx = text.find(term, start)
            if idx < 0:
                break
            spans.append((idx, idx + len(term)))
            start = idx + len(term)
    return spans


def _build_query_focus_values(
    question: str,
    passage_segment: str,
    offsets,
    token_count: int,
) -> list[int]:
    if not USE_QUERY_LEXICAL_FOCUS or offsets is None:
        return [0] * token_count

    terms = _extract_focus_terms(question)
    if not terms:
        return [0] * token_count

    spans = _find_term_spans(passage_segment, terms)
    if not spans:
        return [0] * token_count

    hit_indices: set[int] = set()
    for idx, (start, end) in enumerate(offsets[:token_count]):
        if end <= start:
            continue
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                hit_indices.add(idx)
                break

    if not hit_indices:
        return [0] * token_count

    window = max(0, int(QUERY_LEXICAL_FOCUS_WINDOW))
    focus = [0] * token_count
    for idx in hit_indices:
        left = max(0, idx - window)
        right = min(token_count, idx + window + 1)
        for focus_idx in range(left, right):
            focus[focus_idx] = 1
    return focus


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
    instruction = str(select_memory_encoder_instruction(question, passage) or "").strip()
    if contains_hangul(f"{question}\n{passage}"):
        question_label = "질문:"
        passage_label = "본문:"
    else:
        question_label = "Question:"
        passage_label = "Passage:"
    passage_segment = f" {passage}"
    try:
        passage_encoded = tokenizer(
            passage_segment,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        passage_offsets = passage_encoded.get("offset_mapping")
    except (NotImplementedError, TypeError):
        passage_encoded = tokenizer(passage_segment, add_special_tokens=False)
        passage_offsets = None
    passage_token_ids = passage_encoded["input_ids"]
    passage_focus_values = _build_query_focus_values(
        question,
        passage_segment,
        passage_offsets,
        len(passage_token_ids),
    )

    segments = []
    if instruction:
        segments.append(
            (
                tokenizer(f"{instruction}\n", add_special_tokens=False)["input_ids"],
                False,
                False,
                None,
            )
        )
    segments.extend([
        (tokenizer(question_label, add_special_tokens=False)["input_ids"], False, False, None),
        (tokenizer(f" {question}\n", add_special_tokens=False)["input_ids"], True, False, None),
        (tokenizer(passage_label, add_special_tokens=False)["input_ids"], False, False, None),
        (passage_token_ids, False, True, passage_focus_values),
    ])

    input_ids = []
    question_mask = []
    passage_mask = []
    query_focus_mask = []

    for token_ids, is_question, is_passage, focus_values in segments:
        if not token_ids:
            continue
        remaining = max_length - len(input_ids)
        if remaining <= 0:
            break
        token_ids = token_ids[:remaining]
        if focus_values is None:
            focus_values = [0] * len(token_ids)
        else:
            focus_values = focus_values[:remaining]
        input_ids.extend(token_ids)
        question_mask.extend([1 if is_question else 0] * len(token_ids))
        passage_mask.extend([1 if is_passage else 0] * len(token_ids))
        query_focus_mask.extend([1 if is_passage and v else 0 for v in focus_values])

    if not input_ids:
        input_ids = [tokenizer.eos_token_id]
        question_mask = [0]
        passage_mask = [1]
        query_focus_mask = [0]

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    question_mask = torch.tensor([question_mask], dtype=torch.long, device=device)
    passage_mask = torch.tensor([passage_mask], dtype=torch.long, device=device)
    query_focus_mask = torch.tensor([query_focus_mask], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "question_mask": question_mask,
        "passage_mask": passage_mask,
        "query_focus_mask": query_focus_mask,
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
        "query_focus_mask": zero_mask.clone(),
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
