import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, d_model, hidden_dim=1024):
    super().__init__()
    self.W = nn.Linear(d_model, hidden_dim)
    self.V = nn.Linear(hidden_dim, hidden_dim)
    self.res = nn.Linear(d_model, hidden_dim)
    self.ln = nn.LayerNorm(hidden_dim)
    self.act = nn.GELU()


  def forward(self,h):
    base = self.res(h)
    x = self.W(h)
    x = self.act(self.ln(x))
    x = self.V(x)
    return self.act(x + base)
