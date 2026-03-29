import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self,d_model):
    super().__init__()
    self.W=nn.Linear(d_model,d_model)
    self.V=nn.Linear(d_model,d_model)
    self.ln=nn.LayerNorm(d_model)
    self.relu=nn.ReLU()


  def forward(self,h):
    # 논문수식을 그대로 적용. 이때 d_model은 512로 고정.
    # 선형변환도 d_model->d_model로 설정.
    res=self.relu(self.V(self.ln(self.relu(self.W(h)))))

    return res