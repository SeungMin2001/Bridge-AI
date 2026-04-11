import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, d_model, hidden_dim=256):
    super().__init__()
    self.W  = nn.Linear(d_model, hidden_dim)
    self.V  = nn.Linear(hidden_dim, hidden_dim)
    self.ln = nn.LayerNorm(hidden_dim)
    self.relu = nn.ReLU()


  def forward(self,h):
    # 논문수식을 그대로 적용. 이때 d_model은 512로 고정.
    # 선형변환도 d_model->d_model로 설정.
    res=self.relu(self.V(self.ln(self.relu(self.W(h)))))

    return res