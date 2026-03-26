import torch
import torch.nn as nn

class AttentivePooling(nn.Module):
  def __init__(self, d_model):
    super().__init__()
    self.Wa=nn.Linear(d_model,1) #토큰의 임베딩벡터 -> 1(스칼라)

  def forward(self,H,mask=None):
    #H:(B,T,d),
    score=self.Wa(H).squeeze(-1) #마지막차원 1 지워주기
    if mask is not None:
      # padding 토큰 무시하기. -무한 넣어주면 0이됨
      score=score.masked_fill(mask==0,float('-inf'))

    # dim=1: 가로
    alpha=torch.softmax(score,dim=1)

    # alpha의 차원 하나 늘려주기. H랑 연산하려고.
    pooled=torch.sum(alpha.unsqueeze(-1)*H,dim=1)

    return pooled
# 여기까지 h 차원=(B,d_model)