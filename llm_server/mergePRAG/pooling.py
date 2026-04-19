import torch
import torch.nn as nn
import math

class AttentivePooling(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.Wa = nn.Linear(d_model, 1)
    self.fuse = nn.Linear(d_model * 3, d_model)
    self.ln = nn.LayerNorm(d_model)

  def forward(self, H, mask=None, query=None, focus_mask=None):
    # H: (B, T, d)
    score = self.Wa(H).squeeze(-1)
    if query is not None:
      guided = (H * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(H.size(-1))
      score = score + guided

    pooling_mask = focus_mask if focus_mask is not None else mask
    if pooling_mask is None:
      pooling_mask = torch.ones(H.shape[:2], device=H.device, dtype=torch.long)

    score = score.masked_fill(pooling_mask == 0, float('-inf'))
    valid = pooling_mask.unsqueeze(-1).to(dtype=H.dtype)
    denom = valid.sum(dim=1).clamp_min(1.0)
    mean_pool = (H * valid).sum(dim=1) / denom

    if mask is not None:
      last_source = pooling_mask if focus_mask is not None else mask
      last_idx = last_source.sum(dim=1).clamp_min(1) - 1
    else:
      last_idx = torch.full(
          (H.size(0),),
          H.size(1) - 1,
          device=H.device,
          dtype=torch.long,
      )

    alpha = torch.softmax(score, dim=1)
    attn_pool = torch.sum(alpha.unsqueeze(-1) * H, dim=1)

    batch_idx = torch.arange(H.size(0), device=H.device)
    last_pool = H[batch_idx, last_idx]

    fused = torch.cat([attn_pool, mean_pool, last_pool], dim=-1)
    pooled = self.ln(self.fuse(fused))
    return pooled
