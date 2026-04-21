import torch
import torch.nn as nn


class AttentivePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.att_pool = nn.Linear(d_model, 1)

    def forward(self, H, mask=None, **_ignored):
        scores = self.att_pool(H)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        return (H * weights).sum(dim=1)
