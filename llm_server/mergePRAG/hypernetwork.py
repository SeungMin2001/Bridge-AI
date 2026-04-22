import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import encode_passage_states
from .pooling import MultiSlotCrossAttention


class SlotFFN(nn.Module):
    """Slot별 FFN + slot-identity bias.

    [B, k, d] → [B, k, d]. residual + slot bias로 slot 간 collapse 방지.
    """

    def __init__(self, d_model, num_slots, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or d_model * 2
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.slot_bias = nn.Parameter(
            torch.randn(num_slots, d_model) * (d_model ** -0.5)
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, slots):
        x = self.norm(slots)
        out = self.ff(x) + slots
        out = out + self.slot_bias.unsqueeze(0)
        return self.out_norm(out)


class HyperNetwork(nn.Module):
    """passage → MultiSlotCrossAttention → SlotFFN_K / SlotFFN_V → (K, V)

    핵심 설계:
    - learnable slot query가 passage token에 각자 attend (pool collapse 방지)
    - K/V 각자 독립 FFN + slot bias (slot collapse 방지)
    - LayerNorm 기반 scaling (unit-norm collapse 방지, magnitude 정보 보존)
    """

    def __init__(self, d_model, k=5, hidden_dim=None, num_heads=8):
        super().__init__()
        self.k = k
        self.d_model = d_model

        self.pooling = MultiSlotCrossAttention(
            d_model, num_slots=k, num_heads=num_heads
        )
        self.ffn_k = SlotFFN(d_model, num_slots=k, hidden_dim=hidden_dim)
        self.ffn_v = SlotFFN(d_model, num_slots=k, hidden_dim=hidden_dim)

        # hook 주입 크기 조정용 learnable scalar — F.normalize 대신
        self.k_scale = nn.Parameter(torch.ones(1))
        self.v_scale = nn.Parameter(torch.ones(1))

    def encode_embedded(self, embedded, attention_mask=None, query=None, focus_mask=None):
        """debug/train 호환용 — stage별 tensor 반환.

        returns: (slots, slots, K_raw, V_raw)
        """
        slots = self.pooling(
            embedded,
            mask=attention_mask,
            focus_mask=focus_mask,
            query=query,
        )
        K_raw = self.ffn_k(slots)
        V_raw = self.ffn_v(slots)
        return slots, slots, K_raw, V_raw

    def normalize_kv(self, K, V):
        """SlotFFN.out_norm이 이미 LayerNorm을 적용하므로, 여기선 learnable
        scalar scaling만. unit-sphere collapse를 피하고 hook 주입 크기는 조정 가능."""
        return K * self.k_scale, V * self.v_scale

    def forward(self, embedded, attention_mask=None, query=None, focus_mask=None):
        _, _, K, V = self.encode_embedded(
            embedded,
            attention_mask=attention_mask,
            query=query,
            focus_mask=focus_mask,
        )
        return self.normalize_kv(K, V)

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
