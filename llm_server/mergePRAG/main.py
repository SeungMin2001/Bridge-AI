import torch
from .cross_attention import cross_attention
from .hypernetwork import HyperNetwork
from .orthogonal_merge import orthogonal_merging

def make_hook(K,V): #hypernetwork 출력 전달해주기
  def hook(module,input,output):

    Q=output
    K_ = K.to(device=Q.device, dtype=Q.dtype)
    V_ = V.to(device=Q.device, dtype=Q.dtype)
    return Q+0.01*cross_attention(Q,K_,V_) #W+delta W
  return hook