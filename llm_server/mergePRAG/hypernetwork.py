import torch
import torch.nn as nn
from .embedding import encode_passage_states
from .pooling import AttentivePooling
from .mlp import MLP
from .linearProjection import LinearProjection


class HyperNetwork(nn.Module):
    """
    passage text → Qwen embedding → pooling → MLP → LinearProjection → (K, V)

    학습 대상: pooling, mlp, lp의 가중치
    Qwen 모델은 외부에서 주입받으며 freeze 상태로 사용.
    """
    def __init__(self, d_model, k=1, hidden_dim=1024):
        super().__init__()
        self.pooling = AttentivePooling(d_model)
        self.mlp = MLP(d_model, hidden_dim=hidden_dim)
        self.lp = LinearProjection(hidden_dim, d_model, k)

    def encode_embedded(self, embedded, attention_mask=None, query=None, focus_mask=None):
        h = self.pooling(embedded, mask=attention_mask, query=query, focus_mask=focus_mask)
        projected = self.mlp(h)
        K, V = self.lp(projected)
        return h, projected, K, V

    def normalize_kv(self, K, V):
        return K, V

    def forward(self, embedded, attention_mask=None, query=None, focus_mask=None):
        _, _, K, V = self.encode_embedded(
            embedded,
            attention_mask=attention_mask,
            query=query,
            focus_mask=focus_mask,
        )
        return K, V

    @torch.no_grad()
    def encode_passage(self, model, tokenizer, text, use_contextual=False):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        embedded = encode_passage_states(
            model,
            input_ids,
            attention_mask=attention_mask,
            use_contextual=use_contextual,
        )
        embedded = embedded.to(dtype=next(self.parameters()).dtype)
        return self.forward(embedded, attention_mask=attention_mask)
