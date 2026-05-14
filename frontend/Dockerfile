FROM node:20-alpine

WORKDIR /app

# 패키지 매니저 파일 복사
COPY package*.json ./

# 의존성 설치
RUN npm install

# 소스 코드 복사 (빌드용이 아닌 개발용이므로 사실 볼륨 바인딩 시 덮어써짐)
COPY . .

# Vite 개발 서버 포트 노출
EXPOSE 5173

# 호스트 0.0.0.0으로 모든 인터페이스에서 접속 가능하게 설정
# 컨테이너 시작 시 node_modules/.bin이 없으면 npm install을 실행하도록 설정
CMD ["sh", "-c", "if [ ! -d node_modules/.bin ] || [ ! -d node_modules/pdfjs-dist ] || [ ! -d node_modules/pptxviewjs ]; then npm install; fi; npm run dev -- --host 0.0.0.0"]