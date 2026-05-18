# 프로젝트 실행 흐름 가이드

이 문서는 이 프로젝트가 어떤 서버들로 나뉘어 돌아가는지, 각 포트가 무슨 역할을 하는지, 녹음된 음성이 어떤 과정을 거쳐 전사/화자분리/요약까지 이어지는지 초보자 기준으로 정리한 문서입니다.

## 한 줄 요약

이 프로젝트는 대략 아래 구조로 동작합니다.

```text
브라우저 프론트엔드(5173)
  -> FastAPI 백엔드(8000)
      -> PostgreSQL DB(5432)
      -> LLM 서버(8001 또는 11434)
      -> 화자분리 서버(8003)
```

## 포트별 역할

| 포트 | 서버 | 역할 |
| --- | --- | --- |
| `5173` | Vite/Vue 프론트엔드 | 사용자가 보는 화면. 녹음 버튼, 전사 탭, 요약 탭 등이 여기에 있음 |
| `8000` | FastAPI 백엔드 | 프로젝트의 중심 서버. STT, DB 저장, 요약 API, 일정 API, 워크스페이스 API, WebSocket 처리 |
| `8001` | LLM 서버 | Qwen 계열 모델을 OpenAI 호환 API로 서빙하는 서버로 가정 |
| `8003` | Diart/Pyannote 화자분리 서버 | 음성에서 `SPEAKER_00`, `SPEAKER_01` 같은 화자 구간을 추출 |
| `5432` | PostgreSQL | 세션, 전사문, 요약, 일정 등 저장 |
| `11434` | Ollama | `8001` 대신 Ollama를 LLM 서버로 쓸 때 사용 가능 |
| `8002` | denoise_server | 잡음 제거 서버. Docker 설정에는 있지만 현재 백엔드 음성 흐름에서는 직접 사용하지 않음 |

## 프론트엔드 실행

위치:

```bash
cd "/Users/changyoung/Voice Project/Group-Chat-agent/frontend"
npm run dev
```

접속:

```text
http://localhost:5173
```

프론트는 API를 직접 DB에 보내지 않습니다. `frontend/vite.config.js`의 proxy 설정을 통해 `/workspace`, `/summary`, `/schedule`, `/chat`, `/ws` 요청을 백엔드 `8000`으로 넘깁니다.

현재 `vite.config.js` 기준:

```text
로컬 127.0.0.1:8000이 살아있으면 -> http://127.0.0.1:8000 사용
로컬 8000이 없으면 -> VITE_BACKEND_URL 환경변수 또는 http://127.0.0.1:8000 사용
```

원격 `100.104.164.84:8000` fallback 코드는 파일 안에 주석으로 남아 있지만 현재는 꺼져 있습니다.

## 백엔드 8000 실행

위치:

```bash
cd "/Users/changyoung/Voice Project/Group-Chat-agent/backend"
```

로컬 DB, Ollama, 화자분리 서버를 쓰는 예시:

```bash
DB_HOST=localhost \
DB_PORT=5432 \
DB_NAME=shin \
DB_USER=changyoung \
DB_PASSWORD= \
SCHEDULE_MOCK_MODE=false \
SUMMARY_MOCK_MODE=false \
QUIZ_MOCK_MODE=false \
SUMMARY_FALLBACK_ON_ERROR=true \
LLM_URL=http://localhost:11434 \
LLM_MODEL=qwen2.5:1.5b \
DIARIZE_ENABLED=true \
DIARIZE_URL=http://127.0.0.1:8003/diart/raw \
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

`8001` LLM 서버를 쓰는 예시:

```bash
DB_HOST=localhost \
DB_PORT=5432 \
DB_NAME=shin \
DB_USER=changyoung \
DB_PASSWORD= \
SCHEDULE_MOCK_MODE=false \
SUMMARY_MOCK_MODE=false \
QUIZ_MOCK_MODE=false \
SUMMARY_FALLBACK_ON_ERROR=true \
LLM_URL=http://localhost:8001 \
LLM_MODEL=QuantTrio/Qwen3.5-4B-AWQ \
DIARIZE_ENABLED=true \
DIARIZE_URL=http://127.0.0.1:8003/diart/raw \
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

확인:

```bash
curl http://127.0.0.1:8000/docs
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

`--reload`로 실행하면 보통 프로세스가 2개처럼 보일 수 있습니다. 하나는 reloader, 하나는 실제 server process입니다.

## 화자분리 서버 8003 실행

위치:

```bash
cd "/Users/changyoung/Voice Project/Group-Chat-agent/diart_server"
```

실행:

```bash
HF_TOKEN=본인_허깅페이스_토큰 python server.py
```

주의:

- `HF_TOKEN`은 Hugging Face에서 발급한 토큰입니다.
- `pyannote/speaker-diarization-3.1` 모델 페이지에서 사용 조건 동의가 필요합니다.
- 토큰은 코드에 직접 저장하지 말고 실행할 때 환경변수로 넣는 것이 좋습니다.

확인:

```bash
curl http://127.0.0.1:8003/health
```

정상 예시:

```json
{
  "status": "ok",
  "pipeline_loaded": true,
  "sample_rate": 16000,
  "pipeline_type": "SpeakerDiarization"
}
```

`pipeline_loaded`가 `false`면 화자분리 모델이 제대로 로드되지 않은 상태입니다.

## LLM 서버 8001 또는 11434

이 프로젝트에서 LLM은 다음 기능에 사용됩니다.

- AI 채팅
- 실시간/최종 요약
- 화자별 요약
- 일정 추출
- 퀴즈 생성

중요한 점:

```text
STT 모델과 LLM 모델은 다릅니다.
```

STT는 음성을 텍스트로 바꾸는 모델이고, LLM은 텍스트를 이해해서 요약/질문응답/일정추출을 하는 모델입니다.

### 현재 코드 기본값

`backend/main.py`, `backend/summary/summary_service.py`, `backend/schedule/schedule_service.py`, `backend/quiz/quiz_service.py` 기준 기본값은 다음 쪽입니다.

```text
LLM_URL=http://localhost:8001
LLM_MODEL=QuantTrio/Qwen3.5-4B-AWQ
```

하지만 실행 명령어에서 아래처럼 바꾸면 Ollama를 쓸 수 있습니다.

```bash
LLM_URL=http://localhost:11434
LLM_MODEL=qwen2.5:1.5b
```

백엔드는 OpenAI 호환 형식인 아래 endpoint로 요청합니다.

```text
{LLM_URL}/v1/chat/completions
```

따라서 `8001`을 쓰든 `11434`를 쓰든 해당 서버가 OpenAI 호환 API를 받아야 합니다.

## AI 모델 정리

| 기능 | 사용 모델/기술 | 위치 |
| --- | --- | --- |
| 음성 전사 STT | `faster-whisper`의 `large-v3-turbo` | `backend/main.py` |
| 화자분리 | `pyannote/speaker-diarization-3.1`, 실패 시 `diart` 폴백 | `diart_server/server.py` |
| 요약/채팅/일정/퀴즈 | `LLM_MODEL` 환경변수 값 | `backend/*_service.py`, `backend/main.py` |
| 전사 교정 | KoBART 교정 모델, 없으면 비활성화 | `backend/correction.py` |
| RAG 임베딩/검색 | HuggingFace embedding + vector store | `backend/rag_search.py` |

현재 STT 모델:

```text
large-v3-turbo
```

현재 기본 LLM 모델:

```text
QuantTrio/Qwen3.5-4B-AWQ
```

로컬 Ollama로 자주 쓰는 모델 예시:

```text
qwen2.5:1.5b
qwen3:4b
```

## 음성 처리 전체 흐름

녹음 버튼을 누르면 브라우저에서 마이크 오디오를 가져옵니다.

```text
1. 프론트에서 녹음 시작
2. 브라우저 AudioWorklet이 마이크 음성을 PCM 데이터로 변환
3. WebSocket /ws 로 백엔드 8000에 전송
4. 백엔드가 2.5초 정도의 chunk로 오디오를 나눔
5. 48kHz 오디오를 16kHz로 resample
6. 화자분리 사용 여부에 따라 처리 방식 분기
7. faster-whisper large-v3-turbo로 STT 전사
8. KoBART 교정 모델이 있으면 교정, 없으면 raw text 그대로 사용
9. 프론트 전사 탭에 표시
10. JSONL 파일과 DB transcripts 테이블에 저장
11. RAG 검색용 문서로 추가
12. 요약 탭에서 8초마다 요약 요청
```

## 화자분리를 켰을 때 현재 흐름

현재 `backend/main.py` 기준 화자분리 모드는 아래처럼 동작합니다.

```text
오디오 수신
-> 16kHz 변환
-> STT용 2.5초 청크는 바로 전사해서 프론트로 전송
-> 동시에 화자분리용 버퍼에는 8초 단위로 오디오를 모음
-> 8초가 모이면 8003 /diart/raw 호출
-> SPEAKER_00, SPEAKER_01 같은 구간 결과 받음
-> 이미 표시된 전사 chunk와 시간대를 비교해 speaker_id를 보정
-> speaker_update 메시지로 프론트 말풍선의 화자 라벨을 갱신
```

즉 현재 구조는 아래에 가깝습니다.

```text
오디오 -> STT 먼저 표시
오디오 -> 8초 단위 화자분리 -> 기존 전사 speaker_id 보정
```

그래서 화자분리 서버가 느려도 전사 자체는 먼저 표시됩니다. 다만 화자 라벨은 몇 초 늦게 붙거나 나중에 보정될 수 있습니다.

녹음 종료 시에는 파일 저장 없이 RAM에 모아 둔 전체 오디오 버퍼를 한 번 더 `8003 /diart/raw`로 보내 전체 기준 speaker_id를 다시 계산합니다. 이 결과로 기존 DB `transcripts.speaker_id` 값을 업데이트하고, 프론트에 최종 `speaker_update`를 보냅니다.

## 화자분리를 끄면

화자분리 사용을 선택하지 않으면 백엔드 WebSocket query에 아래처럼 전달됩니다.

```text
diarize=false
```

이 경우 흐름은 더 단순합니다.

```text
오디오 -> STT -> 프론트
```

speaker_id는 붙지 않고, 요약은 화자별 요약이 아니라 전체 세션 요약 방향으로 처리됩니다.

## 전사 필터 로그 뜻

백엔드 로그에 이런 것이 보일 수 있습니다.

```text
[필터] avg_logprob=-1.20 → 제거: '...'
[필터] no_speech_prob=0.85 → 제거: '...'
```

뜻:

- `avg_logprob`가 너무 낮으면 Whisper가 자신 없어 하는 문장이라 제거
- `no_speech_prob`가 높으면 무음/잡음일 가능성이 높아서 제거

즉 음성을 잘못 알아들은 것 같은 문장을 화면에 보내지 않기 위한 필터입니다.

## 요약 흐름

프론트에서는 녹음 중 8초마다 현재 전사 내용을 요약 API로 보냅니다.

화자분리 사용 시:

```text
전사 목록
-> speaker_id 기준으로 묶음
-> /summary/speaker/generate 호출
-> 화자별 요약 생성
-> 요약 탭에는 화자별 최신 요약 카드 표시
```

화자분리 미사용 시:

```text
전사 목록 전체
-> /summary/session/text/generate 호출
-> 전체 세션 요약 생성
```

녹음 종료 후에도 마지막으로 한 번 더 요약 생성을 시도합니다. 화자분리 사용 모드에서는 종료 직전 전체 오디오 기준 speaker_id 보정을 기다린 뒤, 보정된 전사 목록으로 최종 요약을 만듭니다.

현재 프론트 요약 화면은 단순화되어 있습니다.

```text
화자분리 사용: 화자별 최신 요약 카드
화자분리 미사용: 전체 녹음 요약 카드
```

최근 요약 5개 흐름, 화자별 키워드, 전체 핵심 키워드는 현재 화면에서 제거된 상태입니다.

## DB 연결

DB 설정은 `backend/db.py`에서 환경변수로 읽습니다.

기본값:

```text
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag
DB_USER=postgres
DB_PASSWORD=1234
```

현재 로컬 실행 예시에서는 아래 값을 사용하고 있습니다.

```text
DB_NAME=shin
DB_USER=changyoung
DB_PASSWORD=
```

전사문은 주로 `transcripts` 테이블에 저장됩니다. 코드에서 `speaker_id`, `recording_id` 컬럼이 없으면 자동으로 추가하려고 시도합니다.

## 자주 쓰는 확인 명령어

프론트 확인:

```bash
cd "/Users/changyoung/Voice Project/Group-Chat-agent/frontend"
npm run dev
```

백엔드 8000 확인:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

화자분리 8003 확인:

```bash
curl http://127.0.0.1:8003/health
```

프론트 빌드 확인:

```bash
cd "/Users/changyoung/Voice Project/Group-Chat-agent/frontend"
npm run build
```

백엔드 문법 확인:

```bash
cd "/Users/changyoung/Voice Project/Group-Chat-agent"
python -m py_compile backend/main.py
```

## 문제 상황별 빠른 판단

### 프론트만 켰는데 DB가 바뀌는 것처럼 보일 때

프론트가 직접 DB에 접근하는 것은 아닙니다. 이미 `8000` 백엔드가 켜져 있으면 프론트 요청이 Vite proxy를 통해 `8000`으로 가고, 백엔드가 DB를 만집니다.

### `8000`이 두 개 보일 때

`uvicorn --reload`는 reloader process와 server process를 같이 띄웁니다. 그래서 `lsof`에 2개처럼 보일 수 있습니다.

### `8003 pipeline_loaded=false`

화자분리 모델이 로딩되지 않은 상태입니다.

확인할 것:

- `HF_TOKEN`이 있는지
- Hugging Face 모델 사용 조건에 동의했는지
- `pyannote.audio`, `diart`, `torch`, `torchaudio` 설치가 맞는지

### 전사가 중간에 멈추는 느낌이 날 때

화자분리 사용 모드에서 `8003` 분석이 오래 걸리면 `8000`의 전사 흐름도 같이 늦어질 수 있습니다. 현재 구조가 `화자분리 결과를 기다린 뒤 STT를 처리`하는 방식이기 때문입니다.

화자분리를 끄면 전사는 더 빠르게 흘러갑니다.

### 요약 500 에러

대부분 LLM 서버 연결 문제입니다.

확인할 것:

- `LLM_URL`이 실제 켜져 있는 서버인지
- `LLM_MODEL` 이름이 해당 서버에 존재하는지
- `SUMMARY_MOCK_MODE=false`면 진짜 LLM 요청이 나가므로 LLM 서버가 반드시 필요

## 전체 실행 순서 추천

처음부터 로컬에서 확인할 때는 아래 순서가 좋습니다.

1. PostgreSQL DB 실행
2. LLM 서버 실행
3. 화자분리를 쓸 거면 `8003` 실행
4. 백엔드 `8000` 실행
5. 프론트 `5173` 실행
6. 브라우저에서 `http://localhost:5173` 접속

화자분리를 테스트하지 않을 때는 `8003` 없이 백엔드를 `DIARIZE_ENABLED=false`로 실행하면 됩니다.
