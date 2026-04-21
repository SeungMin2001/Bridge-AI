import torch
import torch.nn as nn


class AttentivePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.att_pool = nn.Linear(d_model, 1)
        # Max pooling과 결합하기 위한 projection
        self.combine = nn.Linear(d_model * 2, d_model)

    def forward(self, H, mask=None, focus_mask=None, **_ignored):
        """
        Hybrid pooling: attention pooling + max pooling

        문제: pure attention pooling은 전체를 weighted average하면서
              핵심 단어 차이를 희석시킴
        해결: max pooling으로 가장 두드러진 feature를 보존하고,
              attention pooling으로 맥락을 결합
        """
        # Attention pooling (기존)
        scores = self.att_pool(H)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        att_pooled = (H * weights).sum(dim=1)  # [B, d_model]

        # Max pooling (새로 추가)
        # focus_mask가 있으면 passage 부분만, 없으면 전체에서 max
        if focus_mask is not None and focus_mask.sum() > 0:
            # passage 영역만 max pooling
            H_masked = H.clone()
            H_masked[focus_mask == 0] = float("-inf")
            max_pooled = H_masked.max(dim=1)[0]  # [B, d_model]
        elif mask is not None:
            H_masked = H.clone()
            H_masked[mask.unsqueeze(-1) == 0] = float("-inf")
            max_pooled = H_masked.max(dim=1)[0]
        else:
            max_pooled = H.max(dim=1)[0]

        # 두 pooling 결합
        combined = torch.cat([att_pooled, max_pooled], dim=-1)  # [B, 2*d_model]
        return self.combine(combined)  # [B, d_model]
