import torch
import torch.nn as nn

class AttentivePooling(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.Wa = nn.Linear(d_model, 1)
    self.fuse = nn.Linear(d_model * 3, d_model)
    self.ln = nn.LayerNorm(d_model)

  def forward(self, H, mask=None):
    # H: (B, T, d)
    score = self.Wa(H).squeeze(-1)
    if mask is not None:
      score = score.masked_fill(mask == 0, float('-inf'))
      valid = mask.unsqueeze(-1).to(dtype=H.dtype)
      denom = valid.sum(dim=1).clamp_min(1.0)
      mean_pool = (H * valid).sum(dim=1) / denom
      last_idx = mask.sum(dim=1).clamp_min(1) - 1
    else:
      mean_pool = H.mean(dim=1)
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
