import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import encode_passage_states
from .pooling import AttentivePooling
from .mlp import MLP
from .linearProjection import LinearProjection
from .config import (
    KV_PATH_MODE,
    POOLED_KV_SKIP_SCALE,
    POOLED_K_SKIP_SCALE,
    POOLED_V_SKIP_SCALE,
    USE_K_RMS_CLAMP,
    USE_POOLED_KV_SKIP,
    USE_V_RMS_CLAMP,
    K_RMS_CLAMP,
    V_RMS_CLAMP,
)


class HyperNetwork(nn.Module):
    """논문 `HyperKVGeneratorFixed` (MhQA_hypernetwork)를 충실히 재현.

    passage token embedding → AttentivePooling → MLP → LinearProjection → (K, V)

    논문과의 차이:
    - `ffn_k`, `ffn_v` 별칭 (debug/test 스크립트 호환용) — 내부적으로 동일한
      `self.mlp + self.lp` 경로를 사용.
    """

    def __init__(self, d_model, k=1, hidden_dim=1024):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.kv_path_mode = KV_PATH_MODE
        self.use_pooled_kv_skip = USE_POOLED_KV_SKIP
        self.pooled_kv_skip_scale = POOLED_KV_SKIP_SCALE
        self.pooled_k_skip_scale = POOLED_K_SKIP_SCALE
        self.pooled_v_skip_scale = POOLED_V_SKIP_SCALE
        self.use_k_rms_clamp = USE_K_RMS_CLAMP
        self.k_rms_clamp = K_RMS_CLAMP
        self.use_v_rms_clamp = USE_V_RMS_CLAMP
        self.v_rms_clamp = V_RMS_CLAMP
        self.pooling = AttentivePooling(d_model)
        self.mlp = MLP(d_model, hidden_dim=hidden_dim)
        # V는 현재 collapse가 더 심해서 K와 분리된 hidden 경로를 둔다.
        self.v_mlp = MLP(d_model, hidden_dim=hidden_dim)
        self.query_to_v = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
        )
        self.lp = LinearProjection(hidden_dim, d_model, num_kv=k)
        if self.use_pooled_kv_skip:
            self.pooled_to_k = nn.Linear(d_model, k * d_model)
            self.pooled_to_v = nn.Linear(d_model, k * d_model)

    def encode_embedded(self, embedded, attention_mask=None, query=None, focus_mask=None):
        """debug/train 호환 — 중간 stage tensor 반환.

        returns: (pooled, hidden, K_raw, V_raw)
        """
        stats = self.encode_embedded_components(
            embedded,
            attention_mask=attention_mask,
            query=query,
            focus_mask=focus_mask,
        )
        return stats["pooled"], stats["hidden"], stats["K_raw"], stats["V_raw"]

    def encode_embedded_components(self, embedded, attention_mask=None, query=None, focus_mask=None):
        """중간 구성요소를 모두 반환.

        debug 용도:
        - pooled 단계에서 살아있는 차이가 mlp 경로에서 죽는지
        - pooled->kv skip 이 실제로 차이를 보존하는지
        """
        pooled = self.pooling(
            embedded, mask=attention_mask, focus_mask=focus_mask, query=query
        )
        hidden = self.mlp(pooled)
        hidden_v = self.v_mlp(pooled)
        if query is not None:
            hidden_v = hidden_v + self.query_to_v(query)
        K_mlp, V_mlp = self.lp(hidden, hidden_v)
        K_skip = V_skip = None
        K_raw, V_raw = K_mlp, V_mlp
        if self.use_pooled_kv_skip:
            B = pooled.size(0)
            K_skip = self.pooled_to_k(pooled).view(B, self.k, self.d_model)
            V_skip = self.pooled_to_v(pooled).view(B, self.k, self.d_model)
            if self.kv_path_mode == "pooled_only":
                K_raw = self.pooled_k_skip_scale * K_skip
                V_raw = self.pooled_v_skip_scale * V_skip
            elif self.kv_path_mode == "hybrid":
                K_raw = K_raw + self.pooled_k_skip_scale * K_skip
                V_raw = V_raw + self.pooled_v_skip_scale * V_skip
            elif self.kv_path_mode == "mlp_only":
                pass
            elif self.kv_path_mode == "k_mlp_v_hybrid":
                K_raw = K_mlp
                V_raw = V_mlp + self.pooled_v_skip_scale * V_skip
            elif self.kv_path_mode == "k_mlp_v_skip":
                K_raw = K_mlp
                V_raw = self.pooled_v_skip_scale * V_skip
            elif self.kv_path_mode == "k_hybrid_v_skip":
                K_raw = K_mlp + self.pooled_k_skip_scale * K_skip
                V_raw = self.pooled_v_skip_scale * V_skip
            else:
                raise ValueError(f"Unsupported MERGEPRAG_KV_PATH_MODE: {self.kv_path_mode}")
        return {
            "pooled": pooled,
            "hidden": hidden,
            "hidden_v": hidden_v,
            "K_mlp": K_mlp,
            "V_mlp": V_mlp,
            "K_skip": K_skip,
            "V_skip": V_skip,
            "K_raw": K_raw,
            "V_raw": V_raw,
        }

    def normalize_kv(self, K, V):
        """K/V RMS clamp로 hook 주입과 attention logit 폭주를 막는다.

        V는 cross-attention 출력의 내용을 직접 결정하고, K는 attention logit을
        결정한다. 둘 중 하나가 과도하게 커지면 passage 차이보다 norm이 지배한다.
        """
        if self.use_k_rms_clamp:
            k_rms = K.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            scale = torch.clamp(self.k_rms_clamp / k_rms, max=1.0)
            K = K * scale
        if self.use_v_rms_clamp:
            v_rms = V.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
            scale = torch.clamp(self.v_rms_clamp / v_rms, max=1.0)
            V = V * scale
        return K, V

    def forward(self, embedded, attention_mask=None, query=None, focus_mask=None):
        _, _, K, V = self.encode_embedded(
            embedded,
            attention_mask=attention_mask,
            query=query,
            focus_mask=focus_mask,
        )
        return self.normalize_kv(K, V)

    # ── debug 스크립트 호환용 별칭 ──
    def ffn_k(self, pooled_or_slots):
        """debug에서 `hypernet.ffn_k(slots)` 호출 호환. slots = pooled일 수도 있어
        shape 검사해서 자동 dispatch."""
        if pooled_or_slots.dim() == 2:
            hidden = self.mlp(pooled_or_slots)
        else:
            # [B, k, d]가 들어오면 k를 batch로 흡수해서 처리
            B, k, d = pooled_or_slots.shape
            hidden = self.mlp(pooled_or_slots.view(B * k, d)).view(B, k, -1)
        K, _ = self.lp(hidden) if hidden.dim() == 2 else (None, None)
        return K

    def ffn_v(self, pooled_or_slots):
        if pooled_or_slots.dim() == 2:
            hidden = self.v_mlp(pooled_or_slots)
        else:
            B, k, d = pooled_or_slots.shape
            hidden = self.v_mlp(pooled_or_slots.view(B * k, d)).view(B, k, -1)
        _, V = self.lp(hidden, hidden) if hidden.dim() == 2 else (None, None)
        return V

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
