import torch

def orthogonal_merging(WF:torch.Tensor|None,Wt:torch.tensor,eps=1e-6)->torch.Tensor:
  if WF==None: # 만약 merge된 벡터 없다면 wt바로 반환
    return Wt

  # Wf:[d,k]
  # Wt:[d,k]

  # d차원에서 정사영 해줘야하므로 행이 d가 되야함. 즉 전치
  A=WF.T # 기존 memory
  B=Wt.T # 추가할 memory

  k=A.size(1)
  d=A.size(0)

  gram=A.T@A #[k*k]
  gram+=eps*torch.eye(k,device=gram.device)
  gram=torch.linalg.inv(gram)

  P=A@gram@A.T #10번수식 적용 [d,d]

  res=(torch.eye(d,device=P.device)-P)@B #직교 성분 가져오기.

  A=A+res

  return A.T # 입력값 차원 그대로 다시 [k,d] 로 반환하기