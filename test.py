import torch

# 모델 없이도 디바이스 확인
if torch.cuda.is_available():
    print("CUDA 장치 이름:", torch.cuda.get_device_name(0))
    print("CUDA 장치 지원 여부:", torch.cuda.is_available())
    print("현재 연산 디바이스:", torch.cuda.current_device())
else:
    print("GPU 사용 불가, CPU 사용 중")
