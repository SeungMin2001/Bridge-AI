import torch
import torch.nn as nn

class LinearProjection(nn.Module):
  def __init__(self,d_model1,d_model2,k):
    super().__init__()
    self.k=k
    self.K=nn.Linear(d_model1,k*d_model2)
    self.V=nn.Linear(d_model1,k*d_model2)
    self.d=d_model2

  def forward(self,h):
    B,d=h.size()
    res_k=self.K(h).view(B,self.k,self.d_model2)
    res_v=self.V(h).view(B,self.k,self.d_model2)

    return res_k,res_v
