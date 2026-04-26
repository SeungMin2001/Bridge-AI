import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import QUERY_LEXICAL_FOCUS_SCALE, QUERY_POOL_SCALE, USE_QUERY_LEXICAL_FOCUS


class AttentivePooling(nn.Module):
    """논문 `HyperKVGeneratorFixed`의 att_pool을 충실히 재현.

    c_emb [B, T, d] → att_weights softmax → weighted sum.
    num_slots=1이면 pooled [B, d], num_slots>1이면 pooled [B, S, d].
    """

    def __init__(self, d_model, num_slots=1):
        super().__init__()
        self.num_slots = int(num_slots)
        self.att_pool = nn.Linear(d_model, self.num_slots)

    def forward(
        self,
        c_emb,
        mask=None,
        focus_mask=None,
        query=None,
        query_focus_mask=None,
        **_ignored,
    ):
        att_scores = self.att_pool(c_emb)  # [B, T, S]
        if query is not None:
            # 질문 평균 query와 passage token의 방향 유사도를 더해
            # question-conditioned mode가 실제 token selection에 반영되게 한다.
            norm_tokens = F.normalize(c_emb, dim=-1)
            norm_query = F.normalize(query, dim=-1).unsqueeze(1)
            query_scores = (norm_tokens * norm_query).sum(dim=-1, keepdim=True)
            att_scores = att_scores + QUERY_POOL_SCALE * query_scores
        if (
            USE_QUERY_LEXICAL_FOCUS
            and query_focus_mask is not None
            and query_focus_mask.sum() > 0
        ):
            # 질문 핵심 단어 주변 window를 추가로 살려 swapped-role passage에서
            # answer 근처 token이 평균 pooling에 묻히지 않게 한다.
            focus_scores = query_focus_mask.unsqueeze(-1).to(att_scores.dtype)
            att_scores = att_scores + QUERY_LEXICAL_FOCUS_SCALE * focus_scores
        # focus_mask가 주어지면 passage 영역에만 attention.
        # 서비스 요구사항(question과 passage를 같이 인코딩)을 위한 minimal 확장.
        if focus_mask is not None and focus_mask.sum() > 0:
            att_scores = att_scores.masked_fill(focus_mask.unsqueeze(-1) == 0, float("-inf"))
        elif mask is not None:
            att_scores = att_scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        att_weights = torch.softmax(att_scores, dim=1)
        pooled = torch.einsum("btd,bts->bsd", c_emb, att_weights)
        if self.num_slots == 1:
            pooled = pooled[:, 0, :]
        return pooled
