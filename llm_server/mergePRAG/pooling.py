import torch
import torch.nn as nn


class MultiSlotCrossAttention(nn.Module):
    """k개의 learnable slot query가 passage token에 각자 독립적으로 attend.

    기존 AttentivePooling이 [B, T, d] → [B, d] 단일 벡터로 축약하면서
    "Monday"/"Friday" 같은 한 단어 차이를 지워버리는 문제를 해결한다.

    출력 [B, k, d]: slot마다 서로 다른 attention pattern으로 passage를 읽어
    token 수준 정보를 유지.
    """

    def __init__(self, d_model, num_slots, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_slots = num_slots
        self.d_model = d_model

        self.slot_queries = nn.Parameter(
            torch.randn(num_slots, d_model) * (d_model ** -0.5)
        )

        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, H, mask=None, focus_mask=None, query=None, **_ignored):
        """H: [B, T, d].

        mask: [B, T] attention mask (1 = 유효).
        focus_mask: [B, T] passage 영역 (1 = passage). 주어지면 passage에만 attend.
        query: optional [B, d] question 임베딩. slot query에 조건 부여.
        """
        B = H.size(0)

        slot_q = self.slot_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        if query is not None:
            slot_q = slot_q + query.unsqueeze(1).to(dtype=slot_q.dtype)
        slot_q = self.q_norm(slot_q)

        kv = self.kv_norm(H)

        key_padding_mask = None
        if focus_mask is not None and focus_mask.sum() > 0:
            key_padding_mask = focus_mask == 0
        elif mask is not None:
            key_padding_mask = mask == 0

        slots, _ = self.attn(
            slot_q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.out_norm(slots)


AttentivePooling = MultiSlotCrossAttention
