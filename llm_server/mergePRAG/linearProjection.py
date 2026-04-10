import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProjection(nn.Module):
  def __init__(self,d_model1,d_model2,k):
    super().__init__()
    self.k=k
    self.K=nn.Linear(d_model1,k*d_model2)
    self.V=nn.Linear(d_model1,k*d_model2)
    self.d=d_model2

  def forward(self,h):
    B,d=h.size()
    res_k=self.K(h).view(B,self.k,self.d)
    res_v=self.V(h).view(B,self.k,self.d)

    # L2 정규화: K,V 각 벡터의 norm을 1로 제한 → norm 폭발 방지
    res_k=F.normalize(res_k,dim=-1)
    res_v=F.normalize(res_v,dim=-1)

    return res_k,res_v
