# MergePRAG 인수인계 README

이 문서는 이 프로젝트 전체 소개가 아니라, 현재 우리가 집중하고 있는 `llm_server/mergePRAG` 코드 기준의 진행 상황 정리 문서다.  
다른 AI나 개발자가 이 파일만 읽어도 `HyperNetwork`, `K/V 생성`, `critical layer 주입`, `과목별 memory`, `현재 막히는 문제`를 바로 이해하고 다음 작업을 이어갈 수 있게 쓰는 것이 목적이다.

프로젝트의 전체 서비스 구조가 필요하면 아래 문서를 같이 보면 된다.

- `docs/system_model.md`
- `docs/llm_pipeline.md`
- `docs/critical_layer_method.md`

## 1. 현재 상태 한 줄 요약

현재 mergePRAG 코드는 "기본 뼈대 실험" 단계는 이미 넘겼다.  
즉, `Qwen 고정 + HyperNetwork 학습 + K/V 생성 + critical layer hook 주입 + 과목별 memory 관리 + API 연결 + 진단 스크립트`까지는 구현되어 있다.

다만 아직 완전히 끝난 상태는 아니다.  
현재 핵심 과제는 다음이다.

- passage마다 정말 다른 `K/V`가 만들어지는가
- 그 차이가 실제 답변 분포를 바꾸는가
- 학습 때 본 형식과 추론 때 쓰는 형식이 일치하는가
- lecture 서비스용 한국어 환경에서도 이 방식이 안정적으로 먹히는가

즉, "코드는 있다"가 아니라 "이 코드가 실제로 passage-grounded memory injection으로 안정 동작하게 다듬는 중"이라고 이해하면 맞다.

## 2. 우리가 이미 구현한 것

### 2-1. HyperNetwork 기반 K/V 생성 경로 구현

`llm_server/mergePRAG/hypernetwork.py`

현재 passage는 아래 경로로 `K, V`로 변환된다.

1. `encode_passage_states()`로 passage hidden/token embedding 생성
2. `AttentivePooling`으로 `[B, T, d] -> [B, d]`
3. `MLP`로 hidden 변환
4. `LinearProjection`으로 `[B, d] -> K[B, k, d], V[B, k, d]`
5. 필요하면 pooled skip 경로를 섞음
6. `V RMS clamp`로 과도한 주입을 억제

즉 지금 구현은 단순 랜덤 K/V 실험이 아니라, 실제 학습 가능한 passage -> memory 변환기 형태다.

### 2-2. Critical Layer 탐색 자동화

`llm_server/mergePRAG/find_critical_layers.py`  
`llm_server/mergePRAG/critical_layers.json`

처음에는 임의 레이어 주입 수준이었지만, 지금은 각 레이어를 짧게 학습/검증해서 validation loss가 가장 잘 나오는 레이어를 고르는 방식으로 바뀌어 있다.

현재 `critical_layers.json` 기준 top layer는 다음 순서다.

- 11
- 13
- 12
- 17
- 16

실제 기본 주입 레이어는 `config.load_critical_layer()`가 이 파일의 첫 번째 값을 읽어오므로 현재는 사실상 `11번 레이어`가 기본이다.

### 2-3. Question-conditioned memory 경로 구현

`llm_server/mergePRAG/embedding.py`

현재 memory 생성은 두 가지 모드가 있다.

- 질문 없이 passage만 인코딩
- 질문과 passage를 같이 넣되 mask로 역할을 분리하는 question-conditioned memory

구현 방식은 다음과 같다.

- `Question: ...`
- `Passage: ...`

형태로 하나의 시퀀스를 만들고,

- `question_mask`
- `passage_mask`

를 따로 만들어서,

- query는 question 토큰 평균으로 만들고
- pooling은 passage 토큰에만 집중하게 한다

이 구조는 "질문이 어떤 정보에 주목해야 하는지 힌트를 주되, memory 자체는 passage 중심으로 만들자"는 의도다.

### 2-4. Passage collapse 대응용 보강 구현

초기 mergePRAG 실험에서 가장 큰 문제는 서로 다른 passage가 거의 비슷한 memory로 무너지는 collapse였다.  
이걸 줄이기 위해 현재 코드에는 아래 장치들이 추가돼 있다.

- `USE_CONTEXTUAL_PASSAGE_ENCODER`
  - token embedding만 쓰지 않고 Qwen contextual hidden을 passage encoder로 쓰는 옵션
- `USE_POOLED_KV_SKIP`
  - pooled 표현을 K/V로 바로 보내는 skip 경로
- `KV_PATH_MODE`
  - `mlp_only`, `hybrid`, `k_mlp_v_hybrid` 등으로 경로 제어
- `USE_V_RMS_CLAMP`
  - V의 RMS를 제한해서 hook 출력 폭주 방지
- `NEGATIVE_LOSS_WEIGHT`, `REPULSION_LOSS_WEIGHT`
  - 다른 passage는 다른 memory가 되도록 유도하는 추가 loss

즉 현재 코드는 논문 재현 100% 고정본이 아니라, 실제 Qwen 환경에서 passage 분리 신호를 살리기 위해 여러 안전장치를 얹은 상태다.

### 2-5. Orthogonal Merging 기반 과목 memory 관리

`llm_server/mergePRAG/main.py`  
`llm_server/mergePRAG/orthogonal_merge.py`

한 과목에 passage가 여러 개 들어오면 매번 새로 덮어쓰지 않고 기존 memory에 직교 성분만 추가하는 방식으로 누적한다.

흐름:

1. 새 passage -> `new_K, new_V`
2. 기존 과목 memory가 없으면 그대로 저장
3. 있으면 `orthogonal_merging(existing, new)` 수행
4. `course_id`별로 `K`, `V`, `count`, `passages`를 캐시

이 구조 덕분에 과목별 누적 memory 실험이 가능하다.

### 2-6. API 추론 연결 완료

`llm_server/api.py`

현재 API 수준에서 아래 세 모드가 연결되어 있다.

- `/generate`
  - LLM only
- `/generate/rag`
  - context를 prompt에 직접 삽입
- `/generate/mergeprag`
  - course memory에서 `K,V`를 꺼내 critical layer에 hook 주입

또한 스트리밍 endpoint도 있다.

- `/generate/stream`

memory 관리 endpoint도 구현되어 있다.

- `/memory/add`
- `/memory/add-batch`
- `/memory/list`
- `/memory/clear`

즉 mergePRAG는 연구 코드에서 끝난 게 아니라 API 서버까지 한 번 관통해 둔 상태다.

## 3. 현재 코드의 핵심 메커니즘

### 3-1. 학습 메커니즘

`llm_server/mergePRAG/train.py`

학습 시 base LLM은 고정된다.

- Qwen 파라미터 `requires_grad = False`
- 학습 대상은 HyperNetwork만

학습 루프 핵심:

1. sample에서 `question`, `passage`, `answer` 추출
2. passage를 HyperNetwork에 넣어 `delta_K`, `delta_V` 생성
3. `target_layer.register_forward_hook(make_hook(delta_K, delta_V))`
4. QA prompt를 LLM에 넣고 answer 토큰 loss 계산
5. negative passage도 같이 넣어 grounding/repulsion loss 계산
6. HyperNetwork만 업데이트

즉 학습 objective는 "이 passage memory를 꽂았을 때 정답 answer loss가 줄어들고, 틀린 passage memory를 꽂았을 때는 덜 유리하게 만드는 것"이다.

### 3-2. Hook 주입 메커니즘

`llm_server/mergePRAG/main.py`  
`llm_server/mergePRAG/train.py`  
`llm_server/mergePRAG/cross_attention.py`

hook는 critical layer output의 hidden state에 아래 연산을 더한다.

```python
delta = cross_attention(hidden, K, V)
hidden = hidden + alpha * delta
```

의미는 다음과 같다.

- `Q = 현재 디코더 hidden`
- `K,V = passage로부터 만든 external memory`
- 현재 hidden이 memory를 참조하도록 cross-attention을 한 번 더 수행
- 그 결과를 residual처럼 더함

즉 프롬프트 텍스트를 길게 붙이는 대신, 모델 내부 representation에 memory를 바로 삽입하는 구조다.

### 3-3. 추론 메커니즘

`llm_server/api.py`  
`llm_server/mergePRAG/main.py`

추론 시 흐름:

1. `course_id`에 대해 memory 존재 여부 확인
2. 필요하면 새 passage들을 먼저 add
3. `memory_manager.get_memory(course_id, question=req.prompt)` 호출
4. critical layer에 hook 등록
5. `model.generate()` 수행
6. 종료 후 hook 제거

중요한 점:

- 메모리에 원본 passage 문자열이 남아 있으면 질문별로 다시 `K,V`를 재계산할 수 있다
- 질문 없이 저장된 cached `K,V`를 그대로 쓸 수도 있다

즉 현재 memory manager는 "정적 merged memory"와 "질문 조건 재생성 memory" 둘 다 지원하는 중간 단계 구조다.

## 4. 주요 파일 역할 정리

### 코어 구현

- `llm_server/mergePRAG/config.py`
  - 실험 설정, 기본 하이퍼파라미터, weight/checkpoint 로드 규칙
- `llm_server/mergePRAG/embedding.py`
  - token/contextual embedding, question-conditioned tokenization
- `llm_server/mergePRAG/pooling.py`
  - attentive pooling
- `llm_server/mergePRAG/mlp.py`
  - pooled vector 변환용 MLP
- `llm_server/mergePRAG/linearProjection.py`
  - hidden -> K,V projection
- `llm_server/mergePRAG/hypernetwork.py`
  - 전체 HyperNetwork 조립, skip path, V clamp
- `llm_server/mergePRAG/cross_attention.py`
  - hook에서 쓰는 attention 계산
- `llm_server/mergePRAG/main.py`
  - memory manager, hook 생성, course memory 로직
- `llm_server/mergePRAG/orthogonal_merge.py`
  - 기존 memory와 새 memory 직교 병합

### 학습/실험

- `llm_server/mergePRAG/train.py`
  - 메인 학습 스크립트
- `llm_server/mergePRAG/find_critical_layers.py`
  - critical layer 탐색
- `llm_server/mergePRAG/prepare_squad.py`
  - SQuAD -> mergePRAG 학습 포맷 변환
- `llm_server/mergePRAG/prepare_hotpotqa.py`
  - HotpotQA -> 약한 supervision 포맷 변환
- `llm_server/mergePRAG/diagnose_hotpot.py`
  - 실제 가공 데이터에서 memory 분리력 점검
- `llm_server/test_mergeprag.py`
  - K/V 차이와 답변 변화 여부를 직접 점검
- `llm_server/debug_mergeprag.py`
  - collapse가 어느 stage에서 생기는지 세밀 진단

### 서버 연결

- `llm_server/run_model.py`
  - Qwen 로드
- `llm_server/api.py`
  - mergePRAG API 및 streaming 연결

## 5. 지금 코드 기준으로 이해해야 할 중요한 설계 포인트

### 5-1. 논문 재현 그대로가 아니라 "Qwen 적응형 구현"이다

코드 주석을 보면 현재 구현은 단순한 논문 복제가 아니다.

- token embedding only 대신 contextual hidden 옵션 사용
- pure MLP path 대신 pooled skip path 추가
- V 폭주를 막기 위한 RMS clamp 추가
- 단순 CE만이 아니라 negative/repulsion loss 추가

즉 현재 branch의 방향은:

"논문 구현을 가져와서 그대로 끝"  
가 아니라

"Qwen + lecture QA 환경에서 실제로 passage-specific memory가 살아남도록 실용적으로 변형"

이다.

### 5-2. Multi-hop보다 single-passage grounding 쪽으로 학습 구조를 정리하고 있다

`prepare_squad.py` 주석과 `train.py`를 보면, 현재 팀은 "passage 하나가 answer를 결정할 수 있어야 HyperNetwork가 passage-specific memory를 배운다"는 쪽으로 많이 기울어 있다.

즉 현재 판단은 이렇다.

- HotpotQA 원형 그대로는 multi-hop이라 passage 하나만으로 answer 결정이 약함
- 그러면 CE loss만으로는 서로 다른 passage의 K/V를 강하게 구분시키기 어려움
- 그래서 SQuAD 같은 `(question, passage, answer)` 결정성이 높은 데이터가 더 안정적일 수 있음

즉 현 시점의 학습 방향은 "먼저 passage-grounded memory injection 자체를 살려라" 쪽이다.

### 5-3. 서비스용 RAG와 mergePRAG는 아직 완전히 같은 입력 분포가 아니다

서비스는 한국어 강의 assistant이고, mergePRAG 학습/디버깅은 영어 QA 데이터가 중심이다.

이건 지금 코드 구조상 꽤 중요한 사실이다.

- 학습 데이터는 주로 SQuAD/HotpotQA
- 서비스 prompt는 한국어 강의 assistant
- 추론 API는 chat template 사용
- 학습 prompt는 `Question: ...\nAnswer:`

즉 mergePRAG 성능이 애매할 때, 원인이 architecture인지 데이터 분포 차이인지 prompt mismatch인지 분리해서 봐야 한다.

## 6. 반드시 주의해야 할 점

### 6-1. `critical_layers.json`와 현재 기본 config가 서로 안 맞을 수 있다

현재 `critical_layers.json`에는 아래 정보가 들어 있다.

- `k = 16`
- `alpha = 1.0`

그런데 `config.py` 기본값은 현재:

- `NUM_KV = 1`
- `ALPHA = 0.1`

즉 아래 상황이 충분히 발생할 수 있다.

- 이전에 `k=16`으로 학습한 weight를
- 현재 기본 `k=1` 설정으로 로드하려다가 shape mismatch
- 혹은 weight는 맞아도 실험 해석이 완전히 달라짐

다음 작업 전에 반드시 확인할 것:

- 실제 weight 파일이 어떤 `NUM_KV`로 학습되었는지
- 현재 환경변수와 `critical_layers.json`이 같은 실험 세팅을 가리키는지

이건 지금 가장 먼저 확인해야 하는 함정이다.

### 6-2. 학습 prompt와 추론 prompt가 다르다

`train.py`는 기본적으로 아래 형식으로 학습한다.

```text
Question: ...
Answer:
```

반면 `api.py`는 `build_chat_text()`를 통해 chat template 기반 추론을 한다.

디버그 스크립트 `test_mergeprag.py`에도 이 문제를 직접 언급하는 주석이 있다.  
즉, 현재 추론에서 hook 효과가 약하면 architecture보다 먼저 이 mismatch를 의심해야 한다.

실무적으로는 둘 중 하나를 해야 한다.

- 추론 프롬프트를 학습 형식에 맞춘다
- 아니면 학습 자체를 chat template 분포로 다시 맞춘다

### 6-3. question-conditioned memory는 DB 복구 시 완전하지 않다

`CourseMemoryManager.load_from_db()`는 `K`, `V`, `count`만 복구하고 `passages`는 비어 있는 리스트로 넣는다.

즉 question-conditioned mode를 켰을 때의 의미는:

- 메모리에 원문 passage가 남아 있을 때만 질문별 재계산이 가능
- DB에서 tensor만 복구한 상태에서는 질문별 재생성이 불가

즉 DB persistence는 아직 "정적 merged tensor 캐시" 수준이지, 완전한 question-conditioned memory persistence는 아니다.

### 6-4. merge 순서와 passage 구성에 영향을 받는다

`orthogonal_merging()`은 새 passage를 기존 memory에 직교 성분으로 더하는 방식이라, 어떤 passage를 어떤 순서로 넣느냐가 결과에 영향을 줄 수 있다.

즉 현재 과목 memory는 완전 순서불변 구조라고 보면 안 된다.

### 6-5. 이 구현은 hook 가능한 local Transformers 환경이 전제다

현재 mergePRAG는 `model.model.layers[...]`에 직접 forward hook을 걸기 때문에 다음 환경에서는 그대로 못 쓴다.

- vLLM OpenAI 호환 API만 사용하는 환경
- 외부 SaaS LLM API
- hidden hook 제어가 안 되는 추론 엔진

즉 mergePRAG 자체는 현재 `transformers` 기반 로컬 모델 실행을 전제로 한다.

## 7. 지금 우리가 해결 중인 핵심 문제

### 문제 1. passage마다 K/V가 진짜 달라지는가

이건 가장 본질적인 문제다.  
겉보기로는 K/V가 만들어져도, cosine similarity가 너무 높으면 사실상 같은 memory다.

현재 이 문제를 보기 위해 다음 도구들이 있다.

- `llm_server/test_mergeprag.py`
- `llm_server/debug_mergeprag.py`
- `llm_server/mergePRAG/diagnose_hotpot.py`

현재 방향은:

- pooled 단계 차이가 MLP/Projection에서 죽는지 확인
- K는 살아 있는데 V가 collapse하는지 확인
- skip path가 실제 차이를 보존하는지 확인

### 문제 2. K/V 차이가 있어도 답변이 안 바뀌는가

이 경우는 두 가지 가능성이 있다.

- hook 세기가 너무 약하다
- K/V는 달라 보여도 model hidden distribution을 충분히 못 밀어준다

그래서 현재 코드에 다음 실험 흔적이 있다.

- `ALPHA` 축소/조정
- `V_RMS_CLAMP`
- layer별 진단
- `delta_ratio`, `KL`, answer flip 체크

즉 지금은 단순 accuracy보다 "memory injection이 실제 token distribution을 바꾸는가"를 먼저 보고 있다.

### 문제 3. train-time objective가 service objective와 어긋나는가

현재 학습은 QA answer loss 중심이고, 서비스는 lecture assistant 질의응답이다.

즉 아직 남은 질문:

- QA 데이터에서 배운 memory injection이 강의 기반 답변에도 전이되는가
- 한국어/lecture domain에서도 같은 메커니즘이 유지되는가
- RAG context 삽입보다 실제로 더 이득이 있는가

### 문제 4. memory persistence와 서비스 연결이 아직 반쪽짜리다

API는 붙어 있지만 아래는 아직 더 다듬어야 한다.

- course memory의 영속 저장/복구 흐름 완성
- 실제 lecture chunk를 어떻게 memory add 할지 정책 확정
- RAG 검색 결과와 mergePRAG 메모리를 함께 쓸지, 분리할지 설계 확정

## 8. 다음 AI가 우선적으로 확인해야 할 것

### 우선순위 1. 설정 일치성부터 확인

아래 값이 현재 weight와 일치하는지 먼저 확인할 것.

- `MERGEPRAG_NUM_KV`
- `MERGEPRAG_ALPHA`
- `MERGEPRAG_CRITICAL_LAYER`
- `MERGEPRAG_LOAD_SOURCE`
- `MERGEPRAG_USE_CONTEXTUAL_ENCODER`
- `MERGEPRAG_USE_QUESTION_CONDITIONED_MEMORY`
- `MERGEPRAG_KV_PATH_MODE`

특히 `NUM_KV`, `alpha`, `critical layer`는 weight 파일과 다르면 비교 자체가 무의미해진다.

### 우선순위 2. train/infer prompt mismatch 해결

가장 먼저 실험할 가치가 큰 것은 이것이다.

1. 추론도 학습과 같은 `Question: ...\nAnswer:` 형식으로 통일해서 성능 확인
2. 성능이 오르면 chat template mismatch가 주요 원인
3. 그 다음에 chat template 기반 재학습 여부 결정

### 우선순위 3. answer flip이 실제로 나는지 다시 확인

아래 스크립트로 점검:

- `python llm_server/test_mergeprag.py`
- `python llm_server/debug_mergeprag.py`

보고 싶은 지표:

- `cos(K), cos(V)`가 지나치게 높지 않은가
- `delta_ratio`가 너무 작지 않은가
- `hook with sample1 memory`와 `hook with sample2 memory` 답변이 실제로 갈리는가

### 우선순위 4. lecture-domain 데이터로 옮길 준비

mergePRAG 메커니즘이 QA에서 먼저 안정화되면 그 다음은 lecture chunk 기반 데이터셋으로 넘어가야 한다.

필요한 작업:

- 강의 chunk -> question/answer/passage 포맷 정의
- lecture-specific negative sample 전략 설계
- RAG 결과 chunk와 course memory를 어떻게 연결할지 확정

## 9. 빠른 실행 포인트

### 모델 서버 실행

```bash
cd llm_server
uvicorn api:app --host 0.0.0.0 --port 8001
```

### HyperNetwork 학습

```bash
python -m llm_server.mergePRAG.train
```

### Critical layer 재탐색

```bash
python -m llm_server.mergePRAG.find_critical_layers
```

### 핵심 진단

```bash
python llm_server/test_mergeprag.py
python llm_server/debug_mergeprag.py
python -m llm_server.mergePRAG.diagnose_hotpot
```

## 10. 결론

현재 mergePRAG 코드는 "아이디어만 있는 상태"는 아니다.  
`HyperNetwork 학습`, `K/V 생성`, `critical layer hook`, `과목별 memory`, `API 추론`, `디버그 도구`까지 이미 꽤 많이 깔려 있다.

하지만 진짜 남은 핵심은 구조 구현이 아니라 "passage별 memory 차이가 실제 생성 결과 차이로 이어지는지"를 확실하게 만드는 것이다.

다음 작업자는 아래 순서로 접근하는 것이 가장 안전하다.

1. 설정 일치성 확인
2. train/infer prompt mismatch 해결
3. K/V separation과 answer flip 재검증
4. 그 다음에 lecture-domain memory 실험으로 확장

이 문서 기준으로 보면 현재 프로젝트 방향은 명확하다.  
"mergePRAG를 서비스 코드에 붙여 놓은 상태"에서, 이제 진짜로 작동하는 memory injection으로 다듬는 단계다.

## 11. 2026-04-25 이후 개발 방향 업데이트

현재 진단 결과 기준으로 결론은 명확하다.
우리가 원하는 서비스는 "발화자가 말한 내용을 모델 내부 memory로 주입하고, 사용자가 질문하면 그 발화 기반으로 설명하는 것"이다.
따라서 논문처럼 반드시 multi-hop/sub-question 구조부터 재현할 필요는 없다.

우선순위는 다음이다.

1. single-hop lecture QA로 passage-grounding을 먼저 안정화한다.
2. 같은 질문에서 passage만 바뀌면 answer도 바뀌는 hard negative/counterfactual 데이터를 반드시 넣는다.
3. question-conditioned memory로 학습해서 서비스 추론 시 질문별로 같은 발화 중 필요한 부분을 다르게 뽑게 한다.
4. multi-hop은 "여러 발화/여러 chunk를 조합해야 하는 질문"으로 넘어갈 때 확장한다.

즉 지금 당장 필요한 데이터는 HotpotQA식 full multi-hop이 아니라, 아래처럼 발화 chunk 하나가 답을 결정할 수 있는 구조다.

```json
{
  "course_id": "os-2026",
  "chunk_id": "week01-003",
  "speaker": "교수",
  "utterance": "프로세스는 실행 중인 프로그램입니다.",
  "question": "교수님은 프로세스를 뭐라고 설명했어?",
  "answer": "교수님은 프로세스를 실행 중인 프로그램이라고 설명했습니다.",
  "contrast_id": "process-definition",
  "hard_negatives": [
    {
      "passage": "교수: 스레드는 프로세스 안의 실행 흐름입니다.",
      "answer": "스레드"
    }
  ]
}
```

### 11-1. 현재 collapse의 가장 큰 원인

1000~1500 step 실험에서 hook은 작동했다.
`delta_ratio`가 충분히 컸고, 정답 후보 loss도 움직였다.
문제는 memory가 passage의 관계를 제대로 담지 못했다는 점이다.

특히 아래 조합이 위험했다.

- `question_conditioned=False`
- SQuAD처럼 같은 passage에 여러 question/answer가 붙는 데이터
- 쉬운 global negative 위주 sampling
- `num_kv=16`이지만 slot별 다양성 loss가 없음
- "Manchester United defeated Chelsea" vs "Chelsea defeated Manchester United" 같은 near-counterfactual 관계를 학습하지 않음

이 상태에서는 모델이 "passage에 나온 엔티티/일반 answer prior"는 배우지만, "누가 누구를 이겼는지" 같은 관계 반전은 배우기 어렵다.
그래서 Kcos/Vcos가 계속 커지고, compare passage를 넣어도 같은 답을 고르는 현상이 생긴다.

### 11-2. 코드에 새로 반영된 보강

`llm_server/mergePRAG/train.py`

- `hard_negatives`, `negative_passage`, `counterfactual_passage`를 negative sample로 우선 사용한다.
- `contrast_id` 또는 `group_id`가 같은 샘플을 hard negative 후보로 우선 사용한다.
- 같은 `question`을 가진 다른 passage/answer를 global negative보다 먼저 사용한다.
- `utterance`, `speaker`, `messages`, `turns` 같은 서비스 transcript 형태도 passage로 읽을 수 있다.
- `num_kv>1`에서 slot들이 같은 방향으로 복제되는 것을 줄이기 위해 slot diversity loss를 추가했다.
- checkpoint에 config snapshot을 저장하고, 설정이 다르면 기본적으로 자동 resume하지 않는다.

`llm_server/mergePRAG/config.py`

- loss weight, slot diversity, prompt format, checkpoint resume 정책을 환경변수로 조절할 수 있게 했다.
- `MERGEPRAG_TRAIN_PROMPT_FORMAT=chat`을 켜면 API의 chat template과 더 가까운 형식으로 학습한다.

`llm_server/mergePRAG/main.py`

- API 추론에서 question-conditioned memory는 `MERGEPRAG_USE_QUESTION_CONDITIONED_MEMORY=true`일 때만 사용한다.
- 즉, question-conditioned로 학습하지 않은 checkpoint에 질문 조건 주입을 실수로 섞는 위험을 줄였다.

`llm_server/mergePRAG/prepare_lecture_qa.py`

- lecture/service QA JSON 또는 JSONL을 MergePRAG 학습 포맷으로 변환하는 스크립트를 추가했다.

### 11-3. 다음 실험의 권장 set 값

기존 1500 step checkpoint는 `question_conditioned=False` 기반이라 서비스 방향으로 계속 이어 학습하기보다 새 실험을 권장한다.
또한 `num_kv=1` 실험에서 K는 분리됐지만 V가 거의 같은 방향으로 붙는 문제가 확인됐다. `num_kv=1`에서는 attention slot이 하나뿐이라 K가 선택 역할을 못 하고 V 하나가 그대로 주입되므로, 현재 코드 기본값은 아래 방향으로 바꿨다.

- `NUM_KV=4`
- `USE_QUESTION_CONDITIONED_MEMORY=true`
- `KV_PATH_MODE=k_mlp_v_skip`
- `POOLED_V_SKIP_SCALE=1.0`
- `V_SIM_TARGET=0.65`
- `V_REPULSION_MULTIPLIER=4.0`
- `REPULSION_LOSS_WEIGHT=1.0`
- `SLOT_DIVERSITY_LOSS_WEIGHT=0.1`
- `SLOT_DIVERSITY_TARGET=0.5`
- `TRAIN_PROMPT_FORMAT=chat`

SQuAD baseline 결과 K/V norm 폭주는 잡혔지만, main/compare memory가 여전히 거의 같은 분포를 만들었다. 그래서 현재 코드의 기본 학습 데이터도 HotPot processed로 옮겼다. Windows CMD 기준으로 이제 매번 모든 값을 `set`할 필요는 없다. 기존 checkpoint/weights만 백업하고 바로 학습하면 된다.

```bat
cd C:\Users\user\Documents\last_project\Group-Chat-agent

if not exist llm_server\mergePRAG\backup_pt mkdir llm_server\mergePRAG\backup_pt
if exist llm_server\mergePRAG\hypernet_checkpoint.pt move llm_server\mergePRAG\hypernet_checkpoint.pt llm_server\mergePRAG\backup_pt\hypernet_checkpoint_prev.pt
if exist llm_server\mergePRAG\hypernet_weights.pt move llm_server\mergePRAG\hypernet_weights.pt llm_server\mergePRAG\backup_pt\hypernet_weights_prev.pt

python -m llm_server.mergePRAG.train
```

만약 다른 실험값을 임시로 덮고 싶을 때만 `set MERGEPRAG_...=...`를 사용한다.

`NUM_KV=16` 실험을 다시 할 때만:

```bat
set MERGEPRAG_NUM_KV=16
set MERGEPRAG_SLOT_DIVERSITY_LOSS_WEIGHT=0.1
set MERGEPRAG_SLOT_DIVERSITY_TARGET=0.5
```

기존 legacy checkpoint를 정말 이어서 쓰고 싶을 때만:

```bat
set MERGEPRAG_ALLOW_LEGACY_CHECKPOINT_RESUME=true
```

하지만 서비스 방향 실험에서는 이 값을 켜지 않는 것을 권장한다.
설정이 바뀐 checkpoint를 이어 학습하면 왜 좋아졌는지/나빠졌는지 원인 분석이 흐려진다.

### 11-4. lecture QA 데이터 변환

SQuAD는 학습 루프와 V/K clamp 설정이 안정적으로 도는지 확인하는 baseline 역할은 했다. 하지만 relation flip 진단은 통과하지 못했으므로, 현재 기본 데이터셋은 HotPot processed다.

다만 `A defeated B`와 `B defeated A`처럼 같은 엔티티가 나오지만 관계가 뒤집히는 테스트를 통과하려면 SQuAD만으로는 부족할 가능성이 높다. SQuAD에는 `contrast_id`나 explicit hard negative가 거의 없기 때문이다. 직접 lecture 데이터를 만들 수 없다면 다음 단계는 공개 데이터셋 중 `HotPot_train_processed.jsonl` 또는 `MuSiQue_train.jsonl`을 우선 audit하고, 더 hard-negative가 많은 쪽으로 바꾸는 것이다.

공개 데이터셋 전환 후보:

```bat
python -m llm_server.mergePRAG.audit_dataset C:\Users\user\Documents\last_project\data\HotPot_train_processed.jsonl --limit 1000
python -m llm_server.mergePRAG.audit_dataset C:\Users\user\Documents\last_project\data\MuSiQue_train.jsonl --limit 1000
```

HotPot은 이제 코드 기본 경로라 별도 `set` 없이 실행된다. MuSiQue로 바꿔 실험할 때만 경로를 지정한다.

```bat
python -m llm_server.mergePRAG.train
```

MuSiQue가 현재 loader에서 정상 audit되면 아래도 가능하다.

```bat
set MERGEPRAG_TRAIN_DATA_PATH=C:\Users\user\Documents\last_project\data\MuSiQue_train.jsonl
set MERGEPRAG_VALID_DATA_PATH=C:\Users\user\Documents\last_project\data\MuSiQue_valid.jsonl
python -m llm_server.mergePRAG.train
```

입력 lecture QA가 준비되어 있으면 아래처럼 변환한다.

```bat
python -m llm_server.mergePRAG.prepare_lecture_qa C:\path\lecture_train_raw.jsonl C:\path\lecture_train_processed.jsonl
python -m llm_server.mergePRAG.prepare_lecture_qa C:\path\lecture_valid_raw.jsonl C:\path\lecture_valid_processed.jsonl
```

그 다음 학습 데이터 경로를 지정한다.

```bat
set MERGEPRAG_TRAIN_DATA_PATH=C:\path\lecture_train_processed.jsonl
set MERGEPRAG_VALID_DATA_PATH=C:\path\lecture_valid_processed.jsonl
python -m llm_server.mergePRAG.train
```

학습 후 진단에서 중간 checkpoint를 직접 볼 때만 아래 값을 추가한다.

```bat
set MERGEPRAG_LOAD_SOURCE=checkpoint
python llm_server\test_mergeprag.py
```

데이터 품질은 먼저 audit으로 확인한다.

```bat
python -m llm_server.mergePRAG.audit_dataset %MERGEPRAG_TRAIN_DATA_PATH%
```

확인해야 할 값:

- `has_explicit_hard_negative`가 너무 낮으면 passage flip 학습 신호가 부족하다.
- `has_contrast_id`가 낮으면 같은 질문/다른 passage 구조가 부족하다.
- `same_source_valid_negative`가 낮으면 같은 강의/같은 주제 안에서 비교 학습이 약하다.
- `passage_contains_answer`만 높고 hard negative가 없으면 정답 문자열 매칭으로 흐를 수 있다.

### 11-5. 앞으로 해결할 문제

가장 먼저 봐야 할 성공 조건은 free generation이 아니라 candidate choice다.

- main passage memory에서 main answer loss가 낮아지는가
- compare passage memory에서 compare answer loss가 낮아지는가
- `Kcos/Vcos`가 훈련이 진행될수록 0.99로 붙지 않는가
- `slotK/slotV`가 `num_kv=16`에서 계속 0.9 이상으로 붙지 않는가
- qKcos/qVcos가 question-conditioned 학습에서 실제 숫자로 나오고, 같은 passage의 다른 질문을 구분하는가

이 기준을 통과한 다음에야 `/generate/mergeprag`의 자유 생성 품질을 보는 것이 맞다.
자유 생성은 base model prior, decoding, prompt format 영향을 많이 받기 때문에 초반 진단 지표로 쓰면 원인을 헷갈리기 쉽다.
