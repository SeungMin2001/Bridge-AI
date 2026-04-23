FROM python:3.11-slim

# 시스템 의존성 (오디오 처리용)
RUN apt-get update && apt-get install -y \
    gcc \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (캐시 활용)
COPY requirements.txt .

# CPU 전용 torch 먼저 설치 (이미지 크기 최소화)
RUN pip install --no-cache-dir \
    torch==2.8.0+cpu \
    torchaudio==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

EXPOSE 8002

# 서버 시작
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
