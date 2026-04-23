import torch
import torch.nn as nn


class AttentivePooling(nn.Module):
    """논문 `HyperKVGeneratorFixed`의 att_pool을 충실히 재현.

    c_emb [B, T, d] → att_weights softmax → weighted sum → pooled [B, d]
    """

    def __init__(self, d_model):
        super().__init__()
        self.att_pool = nn.Linear(d_model, 1)

    def forward(self, c_emb, mask=None, focus_mask=None, **_ignored):
        att_scores = self.att_pool(c_emb)  # [B, T, 1]
        # focus_mask가 주어지면 passage 영역에만 attention.
        # 서비스 요구사항(question과 passage를 같이 인코딩)을 위한 minimal 확장.
        if focus_mask is not None and focus_mask.sum() > 0:
            att_scores = att_scores.masked_fill(focus_mask.unsqueeze(-1) == 0, float("-inf"))
        elif mask is not None:
            att_scores = att_scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        att_weights = torch.softmax(att_scores, dim=1)
        pooled = (c_emb * att_weights).sum(dim=1)  # [B, d]
        return pooled
