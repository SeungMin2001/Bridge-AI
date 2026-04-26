# MergePRAG 현재 인수인계 README

이 문서는 프로젝트 전체 소개가 아니라, 현재 우리가 집중하고 있는 `llm_server/mergePRAG` 개발 상태를 다른 AI나 개발자가 바로 이어받기 위한 인수인계 문서다. 목표는 "수업 발화 passage를 모델 내부 K/V memory로 주입하고, 사용자의 질문에 대해 그 발화 내용을 근거로 답하게 하는 것"이다.

현재 결론부터 말하면, 코드는 단순 실험 뼈대를 넘어서 실제 학습, 진단, API 연결까지 구현되어 있다. 다만 아직 최종 성공은 아니며, 최근 병목은 더 구체화됐다. `k_mlp_v_hybrid`로 V 경로를 바꿔도 고정 진단 pair에서는 `pooled` 단계가 이미 main/compare를 거의 같은 벡터로 만들어 `K/V`가 함께 collapse된다. 즉 지금 1순위는 V projection 자체가 아니라 **same-question swapped-role passage에서 single pooled vector가 관계를 보존하지 못하는 문제**다.

## 현재 목표

서비스 목표:

```text
발화자/교수/조교가 말한 수업 내용 passage를 memory로 주입한다.
사용자가 질문하면 모델이 일반 지식이 아니라 주입된 발화 내용을 기준으로 답한다.
비슷한 passage라도 작은 사실 차이가 있으면 답이 정확히 바뀌어야 한다.
```

현재 테스트 예시:

```text
question: When is the homework due?

main passage:
Professor Lee said the homework is due on Monday.
The project proposal is due on Friday.

compare passage:
Professor Lee said the homework is due on Friday.
The project proposal is due on Monday.

원하는 결과:
main memory    -> Monday
compare memory -> Friday
```

## 현재 중요한 진단 결과

최근 `k_mlp_v_hybrid` + checkpoint 500 기준 진단:

```text
hypernet=checkpoint, checkpoint step=500
kv_path_mode=k_mlp_v_hybrid

direct main  | Monday
direct comp  | Friday

cos(pooled)=0.9999
cos(hidden)=0.9998
cos(K)=0.9997
cos(V)=1.0000

Same Passage / Different Question:
cos(K)=0.8451
cos(V)=0.8145
```

해석:

- `direct main=Monday`, `direct comp=Friday`: base LLM은 passage를 prompt에 직접 넣으면 내용을 이해한다.
- 같은 passage에서 질문만 바꾸면 K/V가 어느 정도 갈라진다. question conditioning 자체는 동작한다.
- 같은 질문에서 passage의 날짜/역할만 뒤집으면 `pooled`부터 0.9999로 붙는다. 따라서 hypernetwork 뒤쪽을 더 학습하기보다 pooling 입력에서 answer 주변 token을 살려야 한다.

최근 대응:

- `KV_PATH_MODE` 기본값을 `k_mlp_v_hybrid`로 바꿨다.
- `embedding.py`에서 질문 핵심 단어가 passage에 등장한 주변 window를 `query_focus_mask`로 만든다.
- `pooling.py`에서 `query_focus_mask` 위치에 attention score boost를 준다.
- 500 step 진단에서 `cos(pooled)=0.9909`, `cos(V)=0.9985`로 여전히 same-question swapped passage가 붙어 있어, single pooled vector 병목을 깨기 위해 slot-wise pooling을 추가했다.
- slot-wise pooling은 `num_kv=4`일 때 4개 slot이 각자 다른 attention map으로 passage를 pooling한 뒤 K/V로 projection한다. 기존처럼 pooled 하나를 4개 slot으로 펼치지 않는다.
- slot-wise 500 step에서도 `cos(V)=0.9986`으로 높게 유지되어, contextual hidden에 raw token embedding을 더하는 `TOKEN_EMBED_SKIP_SCALE=1.0`을 추가했다. 목적은 Monday/Friday 같은 표면 token identity가 V에 남게 하는 것이다.
- 이 변경은 gold answer를 사용하지 않으므로 inference에도 적용 가능하다.
- 기존 checkpoint/weights는 새 pooling 구조 기준으로 다시 학습해야 한다.

## 핵심 파일 지도

### 설정과 공통 케이스

- `llm_server/mergePRAG/config.py`
  - 현재 하이퍼파라미터 기본값, 경로, system prompt, checkpoint/weights 로딩 정책.
  - 기본 로딩은 `hypernet_weights.pt` 우선이다.
- `llm_server/mergePRAG/eval_cases.py`
  - 학습 데이터 생성, `test_mergeprag.py`, `debug_mergeprag.py`가 공유하는 공통 진단 케이스.
  - 질문/passage 진단 케이스를 바꿀 때는 이 파일을 먼저 바꿔야 한다.

### HyperNetwork / K,V 생성

- `llm_server/mergePRAG/embedding.py`
  - passage 또는 question-conditioned memory sequence를 tokenization한다.
  - 영어/한국어에 따라 memory instruction과 label을 바꾼다.
  - `question_mask`, `passage_mask`를 만들어 query와 pooling 영역을 분리한다.
  - `query_focus_mask`를 만들어 질문 핵심 단어가 passage에 나온 주변 token window를 표시한다.
- `llm_server/mergePRAG/pooling.py`
  - `AttentivePooling`.
  - question query와 passage token 유사도를 attention score에 더한다.
  - `query_focus_mask`가 있으면 해당 window에 추가 attention boost를 준다.
  - `USE_SLOTWISE_POOLING=True`이면 `num_kv`개 attention map을 만들어 pooled를 `[B, num_kv, d]`로 반환한다.
- `llm_server/mergePRAG/hypernetwork.py`
  - passage hidden -> pooled -> MLP -> K/V projection.
  - 현재 `k_mlp_v_hybrid` 기본값:
    - `K_raw = K_mlp`
    - `V_raw = V_mlp + V_skip`
  - K/V RMS clamp를 적용한다.
- `llm_server/mergePRAG/cross_attention.py`
  - hook에서 사용하는 cross-attention 계산.

### 학습/데이터

- `llm_server/mergePRAG/prepare_service_hardpairs.py`
  - 현재 핵심 학습 데이터 생성기.
  - 영어/한국어 service-style hard pair를 만든다.
  - 모든 row는 `hard_negatives`와 `contrast_id`를 가진다.
  - `eval_cases.py`의 공통 진단 케이스도 학습 데이터에 포함한다.
- `llm_server/mergePRAG/train.py`
  - Qwen base model은 freeze하고 HyperNetwork만 학습한다.
  - positive answer CE, negative grounding, answer-rank, K/V repulsion, question-conditioned repulsion, slot diversity를 사용한다.
  - validation도 이제 hard negative와 answer-rank를 포함한다.
- `llm_server/mergePRAG/audit_dataset.py`
  - 데이터셋 중복, hard negative 유무, answer/passage 문제 진단.

### 진단

- `llm_server/test_mergeprag.py`
  - 빠른 진단.
  - `no_hook`, `direct main`, `direct comp`, hook generation, candidate choice, K/V cosine을 본다.
- `llm_server/debug_mergeprag.py`
  - 심층 진단.
  - 어느 stage에서 passage 차이가 사라지는지, hook이 logit을 바꾸는지, slot이 실제로 기여하는지 본다.
- `llm_server/mergePRAG/diagnose_hotpot.py`
  - HotPot processed 데이터용 진단.

### 서비스/API

- `llm_server/mergePRAG/main.py`
  - course memory manager와 hook inference.
  - question-conditioned memory가 켜져 있으면 raw passage를 보관하고 질문마다 K/V를 재계산하는 경로가 중요하다.
- `llm_server/api.py`
  - `/generate`, `/generate/rag`, `/generate/mergeprag`, memory add/list/clear endpoint 연결.

## 현재 기본 파라미터

`config.py` 기준 현재 기본값:

```text
MODEL_NAME = Qwen/Qwen3.5-4B
NUM_KV = 4
ALPHA = 0.1
MAX_SEQ_LEN = 512
critical layer = critical_layers.json 첫 번째 값, 현재 11
USE_CONTEXTUAL_PASSAGE_ENCODER = True
TOKEN_EMBED_SKIP_SCALE = 1.0
USE_QUESTION_CONDITIONED_MEMORY = True
QUERY_POOL_SCALE = 4.0
USE_QUERY_LEXICAL_FOCUS = True
QUERY_LEXICAL_FOCUS_SCALE = 6.0
QUERY_LEXICAL_FOCUS_WINDOW = 6
USE_SLOTWISE_POOLING = True
KV_PATH_MODE = k_mlp_v_hybrid
USE_POOLED_KV_SKIP = True
POOLED_K_SKIP_SCALE = 1.0
POOLED_V_SKIP_SCALE = 1.0
USE_K_RMS_CLAMP = True
K_RMS_CLAMP = 0.25
USE_V_RMS_CLAMP = True
V_RMS_CLAMP = 0.25
TRAIN_PROMPT_FORMAT = chat
NEGATIVE_MARGIN = 0.2
NEGATIVE_LOSS_WEIGHT = 0.25
ANSWER_RANK_MARGIN = 0.2
ANSWER_RANK_LOSS_WEIGHT = 1.0
REPULSION_LOSS_WEIGHT = 1.0
V_SIM_TARGET = 0.65
V_REPULSION_MULTIPLIER = 4.0
QUESTION_NEGATIVE_LOSS_WEIGHT = 0.50
QUESTION_REPULSION_LOSS_WEIGHT = 1.25
SLOT_DIVERSITY_LOSS_WEIGHT = 0.1
SLOT_DIVERSITY_TARGET = 0.5
```

기본 데이터 경로:

```text
C:\Users\user\Documents\last_project\data\ServiceHardPair_train.jsonl
C:\Users\user\Documents\last_project\data\ServiceHardPair_valid.jsonl
```

기본 로딩 정책:

```text
MERGEPRAG_LOAD_SOURCE 기본값 = weights
```

즉 진단/서비스는 기본적으로 validation best인 `hypernet_weights.pt`를 읽는다. `hypernet_checkpoint.pt`는 재개용 중간 체크포인트 성격이 강하다.

## 현재 학습 objective

`train.py`에서 base LLM은 freeze된다. 학습되는 것은 HyperNetwork뿐이다.

학습 흐름:

```text
sample question/passage/answer
-> question-conditioned memory query 생성
-> passage -> HyperNetwork -> delta_K, delta_V
-> target layer hook 등록
-> answer token CE loss 계산
-> hard negative passage도 K/V 생성
-> positive answer가 negative memory에서 덜 나오게 grounding loss
-> negative answer가 positive memory에서 덜 나오고 negative memory에서는 잘 나오게 answer_rank_loss
-> K/V가 서로 너무 비슷하면 repulsion loss
-> slot diversity loss
-> HyperNetwork만 optimizer.step()
```

추가된 핵심 loss:

```text
positive memory: positive answer loss < negative answer loss
negative memory: negative answer loss < positive answer loss
```

로그에서 `arank`가 이 answer-rank loss다.

중요한 최근 수정:

- 예전 validation은 positive answer CE만 봐서, `main은 맞지만 compare flip은 실패하는 weights`가 best로 저장될 수 있었다.
- 현재 validation은 hard negative와 answer-rank까지 포함한 `hard_pair_answer_rank_v1` objective를 사용한다.
- 따라서 이 수정 이후에는 기존 weights를 믿지 말고 재학습해야 한다.

## 데이터셋 전략

외부 데이터셋만 그대로 쓰는 것은 현재 목표에 부족하다.

이유:

- SQuAD/HotPotQA는 일반 QA나 multi-hop에는 좋지만, near-counterfactual passage flip 신호가 약하다.
- 우리 서비스는 "거의 같은 발화에서 날짜/승자/범위/정의/비교 방향 하나가 바뀌면 답도 바뀌는지"가 핵심이다.

현재 전략:

```text
1. ServiceHardPair로 passage-specific grounding 먼저 학습
2. 필요하면 HotPotQA/MuSiQue/NarrativeQA로 다양성 보강
3. 마지막은 다시 ServiceHardPair로 짧게 마무리
```

둘 중 하나만 고르면 `외부 데이터셋 -> ServiceHardPair` 순서가 더 안전하다. 마지막 학습이 hard pair여야 service-critical passage flip 능력이 유지된다.

## 영어/한국어 데이터 처리

이 프로젝트는 영어/한국어 mixed service data를 의도적으로 사용한다. 중요한 것은 한 샘플 내부에서 질문, passage, answer, hard negative가 충돌하지 않는 것이다.

이미 고친 문제:

- 한국어 row에 `Session 1.` 같은 영어 prefix가 붙던 문제를 `수업 1회차.`로 수정.
- 한국어 질문에 `BFS`, `SSD`, `TCP` 같은 영문 약어 단독 답변이 생기던 문제를 `BFS 방식`, `SSD 저장장치`, `TCP 프로토콜`처럼 수정.
- 한국어 질문은 `질문/답변`, 한국어 memory instruction, 한국어 system prompt를 사용한다.
- 영어 질문은 `Question/Answer`, 영어 instruction/prompt를 사용한다.

검증한 기준:

```text
q_passage_mismatch = 0
english_prefix_on_ko = 0
ko_answer_no_hangul = 0
negative_lang_mismatch = 0
```

## 실행 명령

Windows CMD 기준.

### 1. 데이터 재생성

```bat
cd C:\Users\user\Documents\last_project\Group-Chat-agent
python -m llm_server.mergePRAG.prepare_service_hardpairs --output-dir C:\Users\user\Documents\last_project\data
```

기본 반복 수 기준 예상:

```text
train 약 2520개
valid 약 120개
```

### 2. 학습

```bat
python -m llm_server.mergePRAG.train
```

현재 config 기본값이 코드에 들어가 있으므로 일반적으로 `set MERGEPRAG_...`는 필요 없다. 예전에 터미널에 남긴 환경변수가 있으면 새 터미널에서 실행하는 것이 안전하다.

### 3. best weights 기준 진단

```bat
set MERGEPRAG_LOAD_SOURCE=weights
python llm_server\test_mergeprag.py
```

정상 출력 기준:

```text
hypernet=weights
```

만약 아래처럼 나오면 checkpoint를 본 것이다.

```text
hypernet=checkpoint
```

이 경우 best validation weights가 아니라 마지막 checkpoint일 수 있으므로 결과 해석에 주의해야 한다.

### 4. 사전지식 없는 synthetic 진단

요일/마감일처럼 모델 prior가 섞일 수 있는 케이스와 별도로, 무의미한 코드명 매핑으로 passage 주입만 검사할 수 있다.

```bat
cd C:\Users\user\Documents\last_project\Group-Chat-agent\llm_server
set MERGEPRAG_LOAD_SOURCE=checkpoint
python test_mergeprag_synthetic.py
```

사용되는 핵심 구조:

```text
question: What code word is assigned to the daxmel marker?
main passage:    daxmel -> virel,  norqu -> jandor
compare passage: daxmel -> jandor, norqu -> virel

원하는 결과:
main memory    -> virel
compare memory -> jandor
```

같은 스크립트를 환경변수로도 선택할 수 있다.

```bat
set MERGEPRAG_DIAGNOSTIC_CASE=synthetic
python test_mergeprag.py
```

### 5. 심층 디버그

```bat
set MERGEPRAG_LOAD_SOURCE=weights
python llm_server\debug_mergeprag.py
```

synthetic 심층 디버그:

```bat
cd C:\Users\user\Documents\last_project\Group-Chat-agent\llm_server
set MERGEPRAG_LOAD_SOURCE=checkpoint
python debug_mergeprag_synthetic.py
```

## 진단 출력 해석법

`test_mergeprag.py`에서 가장 먼저 볼 것:

```text
no_hook
direct main
direct comp
main hook
compare hook
Candidate Choice
cos(K)
cos(V)
```

의미:

- `no_hook`: passage 없이 모델이 답하는지 본다. "Not provided"류면 좋다.
- `direct main/direct comp`: passage를 prompt에 직접 넣었을 때 base LLM이 이해 가능한지 보는 상한선이다.
- `main hook`: K/V 주입만으로 main answer가 나오는지 본다.
- `compare hook`: K/V 주입만으로 compare answer로 뒤집히는지 본다.
- `Candidate Choice`: free generation보다 더 믿을 수 있는 정량 지표다.
- `cos(V)`: 현재 가장 중요한 실패 지표다. main/compare 간 V cosine이 0.99 이상이면 사실상 같은 내용을 주입하는 것이다.

성공에 가까운 상태:

```text
no_hook      -> Not provided
direct main  -> Monday
direct comp  -> Friday
main hook    -> Monday
compare hook -> Friday
main_choice=Monday
compare_choice=Friday
cos(V) ideally < 0.95, 더 좋으면 < 0.8
```

현재 마지막 알려진 실패 상태:

```text
kv_path_mode=k_mlp_v_hybrid
cos(pooled)=0.9999
cos(K)=0.9997
cos(V)=1.0000
main/compare Candidate Choice가 동일하게 움직임
```

## 지금 남은 핵심 문제

### 1. Pooling collapse

현재 가장 큰 병목은 V만이 아니라 `pooled` 단계가 passage별 관계를 보존하지 못하는 것이다.

관찰:

```text
cos(pooled)=0.9999
cos(hidden)=0.9998
cos(K)=0.9997
cos(V)=1.0000
```

같은 질문에서 passage만 바뀔 때 이미 pooled가 같아진다. cross-attention에서 K는 "어디를 볼지", V는 "무엇을 주입할지"에 가까우므로, pooled가 같으면 K/V도 같이 붙고 compare memory도 같은 답으로 밀린다.

최근 적용한 작업:

- `query_focus_mask`: gold answer 없이 질문 핵심 단어 주변 passage window를 표시.
- `AttentivePooling`: 해당 window에 `QUERY_LEXICAL_FOCUS_SCALE`만큼 attention boost.
- `slot-wise pooling`: `num_kv=4` slot마다 별도 attention map을 사용해 passage의 다른 위치를 직접 보게 함.
- `token embedding skip`: contextual hidden에 raw token embedding을 더해 날짜/entity token identity 보존.
- `KV_PATH_MODE`: 기본값을 `k_mlp_v_hybrid`로 변경.

다음 판단:

- 이 코드 변경 후 기존 `.pt`를 제거하거나 이름을 바꾸고 처음부터 재학습한다.
- 500 step에서 `test_mergeprag.py`를 `checkpoint` 기준으로 본다.
- 성공 신호는 `cos(pooled)`가 0.9999에서 내려가고, Candidate Choice가 `main=Monday`, `compare=Friday`로 갈리는 것이다.

### 2. Validation 주기

현재 코드에서 validation은 `EVAL_EVERY = 500`이다. 이 값은 gradient 학습에는 직접 영향을 주지 않는다. 하지만 best weights 저장과 early stopping에는 영향을 준다.

현재 데이터셋은 작고 overfitting이 빠르므로 실험 단계에서는 100 또는 250이 더 안전할 수 있다.

추천:

```text
EVAL_EVERY = 100 또는 250
SAVE_EVERY = 250 또는 500
```

### 3. 기존 weights 해석 주의

`train.py`의 validation objective가 최근 바뀌었다. 기존 `hypernet_weights.pt`는 새 hard-pair validation 기준으로 고른 best가 아닐 수 있다.

따라서 README 업데이트 시점 이후의 판단은 반드시 새 학습으로 얻은 weights 기준이어야 한다.

## 주의사항

- 질문/passage 진단 케이스를 바꿀 때 `test_mergeprag.py`만 직접 바꾸지 말 것.
  - `llm_server/mergePRAG/eval_cases.py`를 수정해야 train/test/debug가 같이 맞춰진다.
- `MERGEPRAG_LOAD_SOURCE=checkpoint`가 남아 있으면 마지막 checkpoint를 읽어 결과가 달라진다.
  - 진단은 기본적으로 `weights`를 보게 해야 한다.
- `.pt`, `train_log.json`, `train_loss_curve.png`는 실험 산출물이다.
  - 코드 커밋 시 실수로 포함하지 않도록 주의.
- `num_kv=1`은 K 선택 역할이 사라져 V collapse에 취약했다.
  - 현재 기본은 `num_kv=4`.
- `alpha=1.0`은 hidden을 과하게 덮어써 이상한 생성이 나올 수 있다.
  - 서비스/진단 기본은 `alpha=0.1`.

## 다음 AI가 바로 해야 할 일

1. 현재 README 기준 코드로 ServiceHardPair 데이터를 다시 생성한다.
2. 기존 weights/checkpoint가 새 validation 기준 이전 것이라면 새로 학습한다.
3. `MERGEPRAG_LOAD_SOURCE=weights`로 `test_mergeprag.py`를 실행한다.
4. `main_choice=Monday`, `compare_choice=Friday`가 되는지 확인한다.
5. 실패하면 `debug_mergeprag.py`에서 V가 어디서 collapse되는지 본다.
6. 우선순위는 pooling collapse 해소다. `cos(pooled)`가 0.9999면 뒤쪽 K/V 학습만으로는 해결이 어렵다.

현재 프로젝트의 방향은 "외부 데이터셋 일반 QA 성능"보다 "주입된 발화 passage가 답변을 실제로 뒤집는가"에 맞춰져 있다. 다른 AI가 이어받을 때도 이 기준을 최우선으로 봐야 한다.
