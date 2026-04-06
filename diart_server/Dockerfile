# diart 화자 분리 서버 Dockerfile
# 잡음 제거 이후 화자 분리 단계. 독립 스케일링 가능.
# [필수] HF_TOKEN 환경변수로 HuggingFace 토큰 전달

FROM python:3.10-slim
# 시스템 의존성 (오디오 처리 + portaudio + git)
# git: pyannote 내부에서 git commit hash 확인 시도하므로 필수
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    gcc \
    g++ \
    git \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    portaudio19-dev \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# CPU 전용 torch (이미지 크기 최소화)
# 2.5.1 안정 버전 사용
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# pandas 1.4.4 빌드 에러 방지를 위해 setuptools 버전 낮추기 및 wheel 설치
RUN pip install --no-cache-dir "setuptools<70.0.0" wheel

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8003

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8003"]
