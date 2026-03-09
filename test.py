import torch

print("CUDA 사용 가능:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 장치 이름:", torch.cuda.get_device_name(0))
    print("현재 연산 디바이스:", torch.cuda.current_device())

# 간단한 GPU 연산 테스트
x = torch.rand(3, 3).to("cuda")
y = torch.rand(3, 3).to("cuda")
z = x + y
print("GPU 연산 결과:", z)
print("연산 디바이스:", z.device)
