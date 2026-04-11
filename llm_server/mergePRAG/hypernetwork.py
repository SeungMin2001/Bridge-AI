import torch
import torch.nn as nn
from .embedding import embedding
from .pooling import AttentivePooling
from .mlp import MLP
from .linearProjection import LinearProjection


class HyperNetwork(nn.Module):
    """
    passage text → Qwen embedding → pooling → MLP → LinearProjection → (K, V)

    학습 대상: pooling, mlp, lp의 가중치
    Qwen 모델은 외부에서 주입받으며 freeze 상태로 사용.
    """
    def __init__(self, d_model, k=16):
        super().__init__()
        self.pooling = AttentivePooling(d_model)
        self.mlp = MLP(d_model, hidden_dim=256)
        self.lp = LinearProjection(256, d_model, k)  # MLP 출력 256 → K,V는 d_model

    def forward(self, embedded):
        """
        Args:
            embedded: Qwen 임베딩 출력 [B, T, d_model]
        Returns:
            K: [B, k, d_model]
            V: [B, k, d_model]
        """
        h = self.pooling(embedded)   # [B, T, d] → [B, d]
        h = self.mlp(h)              # [B, d] → [B, d]
        K, V = self.lp(h)            # [B, d] → [B, k, d], [B, k, d]
        return K, V

    @torch.no_grad()
    def encode_passage(self, model, tokenizer, text):
        """
        텍스트를 받아서 Qwen 임베딩 → K, V 생성 (추론용).
        model: Qwen 모델 (freeze)
        """
        embedded = embedding(model, tokenizer, text)
        embedded = embedded.to(dtype=next(self.parameters()).dtype)
        return self.forward(embedded)
