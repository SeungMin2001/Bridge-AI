# PyTorch 및 표준 머신러닝 라이브러리와 호환되는 Python 이미지 사용
FROM python:3.11-slim

# Python이 .pyc 파일을 생성하지 않고 표준 출력을 버퍼링하지 않도록 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# psycopg 빌드, 오디오 처리(Whisper) 등에 필요한 시스템 의존성 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 애플리케이션 코드 복사
COPY . .

# main.py가 있는 backend 폴더로 작업 디렉토리 설정
WORKDIR /app/backend

# FastAPI를 위한 포트 개방
EXPOSE 8000

# FastAPI 애플리케이션 시작
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]