# Real-time Lecture Assistant

교수자 음성을 실시간으로 전사하고, 전사된 문장 속 어려운 단어나 개념을 표시하여 클릭 시 간단한 설명을 제공하는 실시간 학습 보조 데모입니다.

## Setup
```bash
git clone <repository_url>
cd last_project

python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Run Backend
```bash
cd backend
uvicorn main:app --reload
```

Backend: http://127.0.0.1:8000

## Run Frontend
```bash
cd frontend
python -m http.server 5500
```

Frontend: http://127.0.0.1:5500

## Requirements
```
Python 3.10+
ffmpeg
브라우저 마이크 권한 허용
```
