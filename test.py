import torch

# 현재 사용 가능한 디바이스 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print("현재 사용 중인 디바이스:", device)

# 모델을 확인하고 싶다면
# model = ...  # 이미 정의된 모델
# print(next(model.parameters()).device)
