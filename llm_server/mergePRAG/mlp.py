import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_model, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h):
        return self.net(h)
