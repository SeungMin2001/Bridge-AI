# LLM 서빙 환경 최적화 보고서

## 하드웨어 환경

| 항목 | 사양 |
|------|------|
| OS | Windows 11 |
| GPU | NVIDIA RTX 5070 Ti |
| VRAM | 16GB (16,303 MiB) |
| CPU | - |
| RAM | - |
| 서빙 방식 | Docker Desktop (WSL2 백엔드) + vLLM |
| 네트워크 | Mac (프론트엔드) ↔ SSH ↔ Windows (백엔드 + LLM 서버 + DB) |

## GPU 메모리 분배 구조

RTX 5070 Ti 16GB에서 동시 실행되는 프로세스:

```
┌────────────────────────────────────────────────┐
│              RTX 5070 Ti 16GB VRAM              │
├────────────────────────────────────────────────┤
│ Windows 시스템 (dwm, explorer 등)  │ ~600 MiB  │
│ Docker vLLM (LLM 서빙)            │ ~6-8 GB   │
│ Whisper large-v3-turbo (STT)       │ ~3 GB     │
│ KoBART (교정)                      │ ~0.5 GB   │
│ bge-m3 (RAG 임베딩)               │ ~2 GB     │
│ 여유                               │ ~2-3 GB   │
└────────────────────────────────────────────────┘
```

**중요: vLLM 컨테이너를 먼저 시작한 후 Backend 서버를 시작해야 함.**
순서가 바뀌면 Whisper가 VRAM을 먼저 점유하여 vLLM 초기화 실패.

---

## 모델 선택 과정

### 테스트한 모델들

| 모델 | 크기 | 양자화 | VRAM 필요 | 결과 |
|------|------|--------|----------|------|
| Qwen3.5-27B | 27B | bitsandbytes 4-bit | ~14 GB | **실패** - VRAM 초과, 다른 프로세스와 공존 불가 |
| Qwen3.5-9B | 9B | bitsandbytes 4-bit | ~6 GB | **성공 (단독)** - transformers generate로 직접 서빙. 단, 속도 느림 (~3s) |
| Qwen3.5-9B | 9B | AWQ 4-bit | ~6 GB | **실패 (vLLM)** - vLLM 오버헤드 포함 시 다른 프로세스와 VRAM 경합 |
| **Qwen3.5-4B** | **4B** | **AWQ 4-bit** | **~3 GB** | **성공 (vLLM)** - 최적 조합. 다른 프로세스와 공존 가능 |

### 모델별 VRAM 사용량 (vLLM 기준)

| 모델 | 모델 가중치 | KV Cache (1024 ctx) | vLLM 오버헤드 | 총 필요 |
|------|-----------|-------------------|-------------|---------|
| Qwen3.5-27B AWQ | ~8 GB | ~2 GB | ~1 GB | ~11 GB |
| Qwen3.5-9B AWQ | ~5 GB | ~1.5 GB | ~1 GB | ~7.5 GB |
| **Qwen3.5-4B AWQ** | **~2.5 GB** | **~0.8 GB** | **~1 GB** | **~4.3 GB** |

---

## 서빙 방식 비교

### transformers generate (이전 방식)

```python
# llm_server/run_model.py + api.py
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", load_in_4bit=True)
output = model.generate(**inputs, max_new_tokens=512)
```

| 항목 | 값 |
|------|-----|
| 모델 | Qwen3.5-9B (bitsandbytes 4-bit) |
| 응답 시간 | ~3초 |
| VRAM | ~6 GB |
| 장점 | 설치 간단, 윈도우 직접 실행 가능 |
| 단점 | 느림, KV cache 비효율, Python GIL |

### Docker + vLLM (현재 방식)

```powershell
docker run -d --gpus all --ipc=host -p 8001:8000 `
  --name shin `
  vllm/vllm-openai:latest `
  --model QuantTrio/Qwen3.5-4B-AWQ `
  --quantization awq `
  --dtype half `
  --gpu-memory-utilization 0.9 `
  --max-model-len 1024 `
  --max-num-seqs 1 `
  --enforce-eager `
  --api-key test-key
```

| 항목 | 값 |
|------|-----|
| 모델 | Qwen3.5-4B (AWQ 4-bit) |
| 응답 시간 | ~1초 |
| VRAM | ~4-5 GB |
| 장점 | 2~3배 빠름, OpenAI 호환 API, PagedAttention |
| 단점 | Docker 필요, 윈도우에서 WSL2 경유 |

---

## vLLM 파라미터 최적화 (핵심)

### gpu-memory-utilization

vLLM이 전체 VRAM 중 몇 %를 사용할지 결정.

| 값 | 상태 | 비고 |
|----|------|------|
| 0.5 | **실패** | KV cache 할당 불가. 모델 가중치 로드 후 cache 공간 부족 |
| 0.6 | **실패** | max-model-len 512에서만 성공. 1024 이상 실패 |
| 0.85 | **실패** | max-model-len 2048에서 실패 |
| **0.9** | **성공** | max-model-len 1024 + enforce-eager 조합에서 안정 |
| 0.95 | 미테스트 | 다른 프로세스(Whisper 등) VRAM 부족 위험 |

### max-model-len

입력 + 출력 토큰의 최대 합. KV cache 크기에 직접 영향.

| 값 | 상태 | 비고 |
|----|------|------|
| 512 | **성공** | gpu-memory-utilization 0.6에서도 작동. 단, RAG context 포함 시 부족 |
| **1024** | **성공** | gpu-memory-utilization 0.9 + enforce-eager 필요. RAG 운용에 충분 |
| 2048 | **실패** | gpu-memory-utilization 0.85에서도 KV cache 메모리 부족 |
| 4096 | **실패** | gpu-memory-utilization 0.85에서도 메모리 부족 |

### enforce-eager

CUDA Graph 사전 컴파일 비활성화.

| 값 | 상태 | 비고 |
|----|------|------|
| 미설정 (CUDA Graph ON) | **실패** | CUDA Graph가 추가 VRAM ~1-2GB 점유하여 cache 할당 실패 |
| **--enforce-eager (OFF)** | **성공** | VRAM 절약. 속도 약 10~20% 감소하나 체감 미미 (max-num-seqs 1이므로) |

### max-num-seqs

동시 처리 요청 수.

| 값 | 상태 | 비고 |
|----|------|------|
| **1** | **성공** | 단일 사용자 서비스에 적합. VRAM 최소 사용 |
| 2+ | 미테스트 | 동시 요청 시 KV cache 추가 필요. VRAM 부족 위험 |

### dtype

모델 연산 정밀도.

| 값 | 상태 | 비고 |
|----|------|------|
| **half (float16)** | **성공** | AWQ 양자화 모델과 호환. 표준 설정 |
| bfloat16 | 미테스트 | RTX 5070 Ti 지원 가능하나 AWQ와의 호환성 확인 필요 |
| float32 | 비권장 | VRAM 2배 사용. 16GB에서 불가능 |

---

## 파라미터 조합 매트릭스

### Qwen3.5-4B AWQ 기준

| gpu-mem-util | max-model-len | enforce-eager | 결과 |
|-------------|--------------|---------------|------|
| 0.5 | 512 | X | **실패** |
| 0.6 | 512 | X | **성공** (RAG context 부족) |
| 0.6 | 1024 | X | **실패** |
| 0.6 | 1024 | O | **실패** |
| 0.85 | 1024 | X | **실패** |
| 0.85 | 2048 | O | **실패** |
| **0.9** | **1024** | **O** | **성공 (최적)** |
| 0.9 | 2048 | O | **실패** |
| 0.95 | 1024 | O | 성공하나 Whisper 등과 VRAM 경합 위험 |

### 최적 파라미터 조합 (최종)

```
모델:               QuantTrio/Qwen3.5-4B-AWQ
양자화:             AWQ 4-bit
gpu-memory-util:   0.9
max-model-len:     1024
max-num-seqs:      1
enforce-eager:     ON
dtype:             half (float16)
```

---

## 오류 유형 정리

### 1. VRAM 부족 (가장 빈번)
```
ValueError: No available memory for the cache blocks.
Try increasing `gpu_memory_utilization` when initializing the engine.
```
**원인**: gpu-memory-utilization이 낮거나 max-model-len이 너무 큼
**해결**: gpu-memory-utilization 올리기 + max-model-len 줄이기 + enforce-eager 추가

### 2. Engine Core 초기화 실패
```
RuntimeError: Engine core initialization failed. See root cause above.
```
**원인**: 위 VRAM 부족의 상위 에러. 또는 다른 프로세스가 GPU 점유
**해결**: nvidia-smi로 GPU 사용 현황 확인 → 불필요한 프로세스 종료 → vLLM 먼저 시작

### 3. Context Length 초과
```
This model's maximum context length is 512 tokens.
However, you requested 512 output tokens and your prompt contains 403 characters
```
**원인**: max-model-len < (입력 토큰 + max_tokens)
**해결**: max-model-len 늘리거나 API 호출 시 max_tokens 줄이기

### 4. Docker Credential 에러
```
docker: error getting credentials - err: exit status 1,
out: `A specified logon session does not exist.`
```
**원인**: Docker Desktop의 credsStore 설정 문제
**해결**: `%USERPROFILE%\.docker\config.json`에서 `"credsStore": ""` 로 변경

### 5. 모델 크기 초과
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**원인**: 모델 자체가 VRAM을 초과 (Qwen3.5-27B 4-bit ≈ 14GB → 다른 프로세스 공간 없음)
**해결**: 더 작은 모델 사용 (27B → 9B → 4B)

### 6. bitsandbytes 호환 문제 (transformers 방식)
```
TypeError: _is_hf_initialized
```
**원인**: bitsandbytes/transformers 버전 불일치
**해결**: vLLM + AWQ 양자화로 전환하여 근본 해결

---

## 서버 기동 순서 (필수)

```
1. Docker Desktop 실행 (Windows 부팅 시 자동 시작 권장)
2. vLLM 컨테이너 시작:  docker start shin
3. vLLM 로딩 완료 대기: docker logs -f shin
   → "Uvicorn running on http://0.0.0.0:8000" 확인
4. Backend 서버 시작:   uvicorn main:app --host 0.0.0.0 --port 8000
5. Frontend 시작 (Mac): npm run dev
```

**순서를 지키지 않으면:**
- Backend 먼저 → Whisper가 VRAM 점유 → vLLM 메모리 부족
- vLLM 로딩 완료 전 요청 → "Server disconnected without sending a response"

---

## Docker 컨테이너 관리

```bash
# 최초 1회 생성
docker run -d --gpus all --ipc=host -p 8001:8000 \
  --name shin \
  -v "${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN="${env:HF_TOKEN}" \
  vllm/vllm-openai:latest \
  --model QuantTrio/Qwen3.5-4B-AWQ \
  --quantization awq \
  --dtype half \
  --gpu-memory-utilization 0.9 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --enforce-eager \
  --api-key test-key

# 이후 제어
docker start shin      # 시작
docker stop shin       # 중지
docker logs -f shin    # 로그 실시간 확인
docker rm shin         # 삭제 (재생성 필요)

# SSH(맥)에서도 동일하게 제어 가능
# 단, Docker Desktop이 윈도우에서 실행 중이어야 함
```

---

## Backend API 설정 (main.py)

vLLM은 OpenAI 호환 API를 제공하므로 기존 llm_server/api.py와 형식이 다름.

| 항목 | 기존 (transformers) | 현재 (vLLM) |
|------|-------------------|-------------|
| URL | `localhost:8001/generate` | `localhost:8001/v1/chat/completions` |
| Request | `{"prompt": "..."}` | `{"model": "...", "messages": [...]}` |
| Response | `{"thinking": ..., "answer": ...}` | `{"choices": [{"message": {"content": ...}}]}` |
| 인증 | 없음 | `Authorization: Bearer test-key` |
| System Prompt | llm_server/api.py 내부 | main.py messages 배열에 포함 |
| Think Mode | `enable_thinking=True/False` | `chat_template_kwargs: {"enable_thinking": false}` |

---

## 결론

RTX 5070 Ti 16GB 환경에서 STT + 교정 + RAG + LLM을 동시 운용하기 위한 최적 조합:

```
LLM: Qwen3.5-4B-AWQ (AWQ 4-bit 양자화)
서빙: Docker + vLLM (OpenAI 호환)
핵심 파라미터:
  - gpu-memory-utilization: 0.9
  - max-model-len: 1024
  - enforce-eager: ON
  - max-num-seqs: 1

→ 응답 속도 ~1초, VRAM ~5GB 사용, 나머지 ~11GB를 STT/교정/RAG가 분배
```
