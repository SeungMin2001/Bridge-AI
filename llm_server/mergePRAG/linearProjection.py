import torch
import torch.nn as nn

class LinearProjection(nn.Module):
  def __init__(self,d_model1,d_model2,k):
    super().__init__()
    self.k=k
    self.K=nn.Linear(d_model1,k*d_model2)
    self.V=nn.Linear(d_model1,k*d_model2)
    self.d=d_model2
    self.reset_parameters()

  def reset_parameters(self):
    # Small projections keep injected deltas from overwhelming the frozen base model early in training.
    nn.init.normal_(self.K.weight, mean=0.0, std=0.01)
    nn.init.normal_(self.V.weight, mean=0.0, std=0.01)
    nn.init.zeros_(self.K.bias)
    nn.init.zeros_(self.V.bias)

  def forward(self,h):
    B,d=h.size()
    res_k=self.K(h).view(B,self.k,self.d)
    res_v=self.V(h).view(B,self.k,self.d)

    return res_k,res_v
