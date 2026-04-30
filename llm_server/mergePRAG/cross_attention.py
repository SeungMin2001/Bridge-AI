import torch
import math

def cross_attention(Q,K,V,head=8):

  # 2D → 3D 보정 (hook에서 batch 차원 없이 올 수 있음)
  squeezed = False
  if Q.dim() == 2:
    Q = Q.unsqueeze(0)
    squeezed = True
  if K.dim() == 2:
    K = K.unsqueeze(0)
  if V.dim() == 2:
    V = V.unsqueeze(0)

  B,q,d=Q.shape #from transformer
  k=K.shape[1] #k=16
  d_k=d//head

  Q=Q.view(B,q,head,d_k).transpose(1,2) #d=>head*d_k, [B,head,q,d_k]
  K=K.view(B,k,head,d_k).transpose(1,2)
  V=V.view(B,k,head,d_k).transpose(1,2)

  attention=(Q@K.transpose(2,3))/(math.sqrt(d_k))
  attention=torch.softmax(attention,dim=-1)@V

  out=attention.transpose(1,2).contiguous().view(B,q,d)

  if squeezed:
    out = out.squeeze(0)

  return out