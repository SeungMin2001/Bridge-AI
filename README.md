# MergePRAG 현재 인수인계 README

이 문서는 프로젝트 전체 소개가 아니라, 현재 우리가 집중하고 있는 `llm_server/mergePRAG` 개발 상태를 다른 AI나 개발자가 바로 이어받기 위한 인수인계 문서다. 목표는 "수업 발화 passage를 모델 내부 K/V memory로 주입하고, 사용자의 질문에 대해 그 발화 내용을 근거로 답하게 하는 것"이다.

현재 결론부터 말하면, 개발 방향을 다시 정리했다. 기존 `train.py` 계열은 여러 보조장치를 붙이며 실험한 버전이고, `train2.py`/`train2_repair.py`는 로컬 클론 논문 코드(`MhQA_hypernetwork-B31F`)를 최대한 반영한 baseline이다. 이 baseline 분석 결과, **논문식 single attentive pooling -> MLP -> K/V 구조는 현재 ServiceHardPair에서 passage-specific memory를 만들지 못하고 K/V가 거의 같은 벡터로 붕괴**했다. 그래서 현재 주 진행은 새 서비스 목표용 구조인 `service_memory.py`, `train_service_memory.py`, `test_service_memory.py`로 옮겼다.

## 최신 진행 방향: Service Memory 구조

서비스에서 진짜 성공으로 보는 목표는 아래 하나다.

```text
교수님/조교가 말한 수업 passage를 HyperNetwork가 K/V memory로 만든다.
나중에 사용자가 그 passage에 관한 질문을 하면, 모델 prompt에는 passage를 다시 넣지 않고 K/V만 주입한다.
모델은 주입된 memory를 근거로 passage 내용을 설명하거나 정답을 말한다.
```

즉 "A라는 지식 문장을 K/V로 저장하고, 사용자가 A가 뭐냐고 물으면 passage 없이도 주입 memory로 답하는 것"이 목표다. near-counterfactual hard-pair는 이 능력을 검증하기 위한 강한 테스트다.

### 왜 train2 baseline을 버리지 않고 넘어갔는가

로컬 논문 코드 기준 baseline:

```text
passage token ids
-> model.model.embed_tokens(...)
-> single attentive pooling
-> MLP
-> linear_K / linear_V
-> target decoder layer forward hook
-> hidden_states + cross_attention(hidden_states, K, V)
```

`train2.py`와 `train2_repair.py`는 이 구조를 기준으로 만들었다. 그러나 실제 진단 결과:

```text
direct passage gold preference: 18/20 = 0.900
main memory gold preference: 10/20 = 0.500
negative memory negative preference: 10/20 = 0.500
bidirectional flip success: 0/20 = 0.000
avg memory cosine: K≈0.998, V≈0.999
```

해석:

- base model은 passage를 prompt로 직접 받으면 대체로 정답을 고른다.
- 그러나 HyperNetwork를 통과한 main/negative passage의 K/V가 거의 동일하다.
- CE-only는 answer-like distribution을 올릴 수 있지만 passage별 사실을 뒤집도록 강제하지 못한다.
- repair objective는 K/V cosine을 일부 낮췄지만, `num_kv=1` single pooling 구조에서는 validation flip이 여전히 0에 머물렀다.

따라서 현재는 논문 구조를 그대로 재현하는 것이 아니라, 논문 아이디어를 출발점으로 삼되 **우리 서비스 목표에 맞는 memory encoder**를 구현하는 방향이다.

### 새 구조

새 파일:

```text
llm_server/mergePRAG/service_memory.py
llm_server/mergePRAG/train_service_memory.py
llm_server/mergePRAG/test_service_memory.py
```

새 구조:

```text
passage
-> base model raw token embedding
-> base model contextual hidden
-> concat(raw, contextual)
-> token-level 또는 multi-slot memory projection
-> token/slot별 MLP
-> token/slot별 K/V
-> K/V RMS clamp
-> alpha-scaled cross-attention hook
```

중요한 차이:

- passage 전체를 single pooled vector 하나로 압축하지 않는다.
- 최신 기본값은 `MERGEPRAG_SERVICE_POOLING_MODE=token`이다.
- token mode에서는 passage token마다 K/V를 만들기 때문에 `A=B`, `C=D` 같은 관계가 8개 slot 압축 과정에서 사라지는 병목을 우회한다.
- 예전 slot mode는 `MERGEPRAG_SERVICE_POOLING_MODE=slot`으로 다시 켤 수 있다.
- Qwen 계열 사용 방식에 맞춰 학습/진단 프롬프트는 chat template 기반이다.
- 학습 objective는 service hard-pair에 맞춰 dual CE + bidirectional ranking + memory separation + slot diversity를 사용한다.

현재 기본값:

```text
MERGEPRAG_SERVICE_NUM_KV = 8
MERGEPRAG_SERVICE_HIDDEN_DIM = 1024
MERGEPRAG_SERVICE_ALPHA = 0.3
MERGEPRAG_SERVICE_USE_CONTEXTUAL = 1
MERGEPRAG_SERVICE_POOLING_MODE = token
MERGEPRAG_SERVICE_MAX_MEMORY_TOKENS = 128
MERGEPRAG_SERVICE_RMS_CLAMP = 0.5
MERGEPRAG_SERVICE_SKIP_SCALE = 0.5
MERGEPRAG_SERVICE_RANK_MARGIN = 0.5
MERGEPRAG_SERVICE_RANK_WEIGHT = 1.0
MERGEPRAG_SERVICE_SEPARATION_WEIGHT = 0.25
MERGEPRAG_SERVICE_DIVERSITY_WEIGHT = 0.05
```

실행:

```bat
python -m llm_server.mergePRAG.train_service_memory
```

진단:

```bat
python -m llm_server.mergePRAG.test_service_memory --split valid --max-samples 240
```

빠른 진단:

```bat
python -m llm_server.mergePRAG.test_service_memory --split valid --max-samples 20 --show-examples 20 --show-generations 5
```

새 산출물:

```text
llm_server/mergePRAG/service_memory_weights.pt
llm_server/mergePRAG/service_memory_checkpoint.pt
llm_server/mergePRAG/service_memory_log.json
```

기존 `hypernet_weights.pt`, `hypernet_checkpoint.pt`, `hypernet_train2_*.pt`, `hypernet_train2_repair_*.pt`는 지울 필요 없다. 서로 다른 실험 산출물이다.

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

최근 진행 흐름:

```text
1) k_mlp_v_hybrid + query lexical focus, checkpoint 500
   cos(pooled)=0.9909, cos(K)=0.9949, cos(V)=0.9985
   Same Passage / Different Question: cos(K)=0.6621, cos(V)=0.5169
   -> 질문 조건 분리는 좋아졌지만, same-question passage flip은 실패.

2) slot-wise pooling 추가 후 checkpoint 500
   cos(pooled)=0.9951, cos(K)=0.9901, cos(V)=0.9986
   Same Passage / Different Question: cos(K)=0.5018, cos(V)=0.4969
   -> slot별 분리는 생겼지만, main/compare 후보 선택은 여전히 같이 움직임.

3) token embedding skip + synthetic_ko checkpoint 500
   cos(pooled)=0.9798, cos(K)=0.9793, cos(V)=0.9780
   direct main=정답, direct comp=반대 정답, no_hook=내용 모름
   -> V cosine은 드디어 0.99 아래로 내려왔지만 compare memory가 아직 반대 답으로 뒤집히지 않음.
```

해석:

- `direct main=Monday`, `direct comp=Friday`: base LLM은 passage를 prompt에 직접 넣으면 내용을 이해한다.
- 같은 passage에서 질문만 바꾸면 K/V가 크게 갈라진다. question conditioning 자체는 동작한다.
- 같은 질문에서 passage의 관계만 뒤집는 경우가 아직 핵심 실패 케이스다.
- synthetic_ko처럼 모델 사전지식이 거의 없는 케이스에서도 direct prompt는 성공하므로, base LLM 문제가 아니라 주입 memory가 passage relation을 충분히 담지 못하는 문제다.

최근 대응:

- `KV_PATH_MODE` 기본값을 `k_mlp_v_hybrid`로 바꿨다.
- `embedding.py`에서 질문 핵심 단어가 passage에 등장한 주변 window를 `query_focus_mask`로 만든다.
- `pooling.py`에서 `query_focus_mask` 위치에 attention score boost를 준다.
- 500 step 진단에서 `cos(pooled)=0.9909`, `cos(V)=0.9985`로 여전히 same-question swapped passage가 붙어 있어, single pooled vector 병목을 깨기 위해 slot-wise pooling을 추가했다.
- slot-wise pooling은 `num_kv=4`일 때 4개 slot이 각자 다른 attention map으로 passage를 pooling한 뒤 K/V로 projection한다. 기존처럼 pooled 하나를 4개 slot으로 펼치지 않는다.
- slot-wise 500 step에서도 `cos(V)=0.9986`으로 높게 유지되어, contextual hidden에 raw token embedding을 더하는 `TOKEN_EMBED_SKIP_SCALE=1.0`을 추가했다. 목적은 Monday/Friday 같은 표면 token identity가 V에 남게 하는 것이다.
- 이 변경은 gold answer를 사용하지 않으므로 inference에도 적용 가능하다.
- 한국어/영어 code-word hard pair와 장소, 점수, 정책, 담당자 도메인을 추가해 사전지식 없는 arbitrary mapping도 학습 데이터에 포함했다.
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

- `llm_server/mergePRAG/service_memory.py`
    - 현재 주 진행 구조.
    - raw token embedding과 contextual hidden을 함께 사용한다.
    - multi-slot learned pooling으로 passage의 여러 token 영역을 K/V slot으로 보존한다.
    - `make_memory_hook`, service chat prompt, answer loss, slot diversity 등 service memory 학습/진단 공통 함수를 포함한다.
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
- `llm_server/mergePRAG/train2.py`
    - 로컬 클론 논문 코드의 `KV_train.py`, `HyperKVGeneratorFixed`를 최대한 따른 baseline.
    - token embedding -> single attentive pooling -> MLP -> linear K/V -> layer hook.
    - 현재 hard-pair service 목표에는 K/V collapse로 실패한 baseline으로 남긴다.
- `llm_server/mergePRAG/train2_repair.py`
    - `train2.py`와 같은 구조를 유지하되 hard-pair bidirectional rank objective만 추가한 repair baseline.
    - K/V cosine은 일부 낮췄지만 validation flip은 아직 0에 가까웠다.
- `llm_server/mergePRAG/train_service_memory.py`
    - 현재 주 진행 학습 스크립트.
    - service-oriented multi-slot memory encoder를 학습한다.
    - 산출물은 `service_memory_weights.pt`, `service_memory_checkpoint.pt`, `service_memory_log.json`.
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
- `llm_server/mergePRAG/test_train2.py`
    - train2/train2_repair baseline 전용 진단.
    - direct passage score, memory score, K/V cosine, bidirectional flip success를 본다.
- `llm_server/mergePRAG/test_service_memory.py`
    - 현재 주 진행 구조 전용 진단.
    - service memory weights를 로드해 passage 없이 K/V 주입만으로 답 후보가 뒤집히는지 본다.

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

## 데이터셋 확장 가이드

다른 AI에게 데이터셋 생성을 맡길 때는 일반 QA를 많이 만드는 것보다, **거의 같은 passage에서 핵심 사실 하나만 바뀌면 답도 반드시 바뀌는 hard-pair 데이터**를 만들게 해야 한다. MergePRAG의 현재 병목은 모델 지식 부족이 아니라 K/V memory가 passage 차이를 보존하지 못하는 것이므로, 데이터도 이 능력을 직접 가르쳐야 한다.

### 수정할 파일

기본 생성기는 아래 파일이다.

```text
llm_server/mergePRAG/prepare_service_hardpairs.py
```

이 파일 안에 도메인별 seed 리스트를 추가하고, `build_rows()`에서 `add_pair(...)`를 호출하면 된다. 현재 들어있는 예시는 다음 계열이다.

```text
EN_TEAMS / KO_TEAMS              경기 승자/패자
DEADLINES_EN / DEADLINES_KO      과제/보고서/퀴즈 마감일
SCOPES_EN / SCOPES_KO            시험/발표 범위
DEFINITIONS_EN / DEFINITIONS_KO  개념 정의
COMPARISONS_EN / COMPARISONS_KO  비교 우위
CODEWORDS_EN / CODEWORDS_KO      사전지식 없는 임의 표식-암호어 매핑
ROOMS_EN / ROOMS_KO              보강/실습/상담 장소 배정
SCORES_EN / SCORES_KO            항목별 점수/가중치
POLICIES_EN / POLICIES_KO        수업 정책/규칙
ASSIGNMENTS_EN / ASSIGNMENTS_KO  담당자/역할 배정
```

데이터를 늘릴 때는 이 리스트들을 직접 확장하거나, 같은 방식의 새 리스트를 추가한다. 예를 들어 `FORMULAS_KO`, `ERROR_CODES_KO`, `EQUIPMENT_KO`, `RUBRICS_KO`, `SCHEDULES_KO` 같은 도메인을 만들 수 있다.

### JSONL 스키마

최종 학습 파일은 아래 두 경로에 저장된다.

```text
C:\Users\user\Documents\last_project\data\ServiceHardPair_train.jsonl
C:\Users\user\Documents\last_project\data\ServiceHardPair_valid.jsonl
```

각 row는 반드시 이 형식이어야 한다.

```json
{
  "source_id": "ko_codeword_0:a",
  "task": "final_qa",
  "question": "라멜 표식에 배정된 암호어는 뭐야?",
  "answer": "루반",
  "passage": "이 교수: 비공개 기록 Q-0에는 라멜 표식의 암호어가 루반이라고 적혀 있습니다. 소핀 표식의 암호어는 가딘입니다.",
  "contrast_id": "ko_codeword_0:라멜 표식에 배정된 암호어는 뭐야?",
  "hard_negatives": [
    {
      "passage": "이 교수: 비공개 기록 Q-0에는 라멜 표식의 암호어가 가딘이라고 적혀 있습니다. 소핀 표식의 암호어는 루반입니다.",
      "answer": "가딘"
    }
  ]
}
```

중요한 필드:

- `question`: 사용자가 실제로 물을 질문. 답을 직접 포함하면 안 된다.
- `answer`: positive passage 기준 정답.
- `passage`: 정답을 포함한 근거 발화. 모델 사전지식이 아니라 이 문장에서만 답이 결정되어야 한다.
- `hard_negatives`: 같은 질문에 대해 답이 뒤집히는 near-counterfactual passage. 반드시 `answer`도 반대 정답으로 넣는다.
- `contrast_id`: 같은 hard-pair를 묶는 id. `add_pair()`를 쓰면 자동으로 만들어진다.
- `task`: 일반 최종 QA는 `final_qa`로 둔다.

가능하면 직접 JSONL을 쓰기보다 `prepare_service_hardpairs.py`의 `add_pair()`를 사용한다. `add_pair()`는 positive row와 negative row를 양방향으로 자동 생성한다.

### 좋은 데이터 예시

좋은 샘플은 같은 질문에서 passage만 바뀌고 답이 바뀐다.

```text
question:
보강 수업 장소는 어디야?

passage_a:
민아 조교: 보강 수업 장소는 새빛관 204호입니다. 실습 모임은 해오름관 101호입니다.
answer_a:
새빛관 204호

passage_b:
민아 조교: 보강 수업 장소는 해오름관 101호입니다. 실습 모임은 새빛관 204호입니다.
answer_b:
해오름관 101호
```

영어도 같은 구조로 만든다.

```text
question:
Which room is assigned to the review session?

passage_a:
TA Mina: The review session is assigned to Room N-204. The lab meeting is assigned to Room H-101.
answer_a:
Room N-204

passage_b:
TA Mina: The review session is assigned to Room H-101. The lab meeting is assigned to Room N-204.
answer_b:
Room H-101
```

### 추천 도메인

다양성을 늘릴 때는 아래처럼 “관계 + 정답 후보가 뒤집히는” 도메인을 우선한다.

- 마감/일정: 과제 마감일, 발표일, 시험일, 상담 시간, 보강 날짜.
- 장소/배정: 강의실, 조별 발표 순서, 실습실, 회의실, 좌석 번호.
- 코드/식별자: 표식-암호어, 에러코드-조치, 실험 샘플-라벨, 장비-식별번호.
- 수치/점수: 가중치, 제한 시간, 제출 횟수, 배점, 임계값.
- 정책/규칙: 지각 처리, 재제출 허용 여부, 감점 기준, 출석 인정 조건.
- 정의/개념: A는 무엇으로 정의됐는지, B와 C의 차이, 특정 용어의 의미.
- 비교/선택: 어느 알고리즘이 빠른지, 어떤 장비가 적합한지, 어떤 방법이 안정적인지.
- 인물/역할: 담당 조교, 발표자, 리뷰어, 제출 담당자.

각 도메인은 한국어/영어를 모두 만들되, 한 row 안에서는 언어를 섞지 않는 것이 좋다.

### 생성 요령

- 한 pair는 `passage_a`와 `passage_b`가 거의 같고, 핵심 값만 서로 바뀌어야 한다.
- `question`은 두 passage에 모두 동일하게 적용되어야 한다.
- `answer_a`와 `answer_b`는 반드시 달라야 한다.
- `passage_a`에는 `answer_a`, `passage_b`에는 `answer_b`가 명시적으로 포함되어야 한다.
- 같은 passage 안에 distractor도 넣어야 한다. 예: 과제 마감일을 묻는데 프로젝트 제안서 마감일도 같이 넣기.
- 정답은 너무 generic하지 않게 한다. `월요일`, `3`, `예`만 반복하면 answer prior가 생긴다.
- 한국어 답변은 가능하면 한국어를 포함한다. 단 `BFS 방식`, `TCP 프로토콜`처럼 약어는 설명어를 붙인다.
- 모델이 이미 알 만한 공개 사실은 피한다. `대한민국 수도=서울` 같은 데이터는 passage 주입 검사용으로 부적합하다.
- 임의 고유명사, 임의 코드, 임의 강의실, 임의 날짜처럼 passage 없이는 알 수 없는 값을 많이 넣는다.
- valid split에는 train과 다른 이름/코드/장소를 넣어야 한다. train에 나온 exact pair를 valid에 그대로 복사하지 않는다.

### 나쁜 데이터 예시

아래는 피해야 한다.

```text
question: 과제는 언제야?
passage: 과제는 월요일입니다.
answer: 월요일
```

문제점: distractor가 없고, hard negative가 없어서 passage flip을 배우지 못한다.

```text
question: TCP는 무엇인가?
passage_a: TCP는 신뢰성 있는 프로토콜입니다.
passage_b: UDP는 빠른 프로토콜입니다.
```

문제점: 질문 대상이 passage_b에서 바뀌어 같은 질문에 대한 counterfactual이 아니다.

```text
question: 라멜 표식의 암호어는 루반이야?
answer: 예
```

문제점: 질문에 답이 들어가고, `예/아니오` prior가 생긴다.

### 데이터 양 늘리는 방법

현재 기본 생성량은 `prepare_service_hardpairs.py` 기준으로 아래 정도다.

```text
base rows: train 164개, valid 48개
repeat 적용 후: train 4920개, valid 240개
```

양을 늘리는 가장 안전한 방법:

```text
1. 도메인 seed 리스트를 늘린다.
2. 각 seed마다 add_pair()로 양방향 hard-pair를 만든다.
3. train/valid는 서로 다른 seed를 사용한다.
4. repeats는 마지막에만 늘린다.
```

단순히 `--train-repeats`만 크게 늘리면 같은 샘플 반복이 많아져 overfitting이 빨라질 수 있다. 먼저 seed 다양성을 늘리고, 그 다음 반복 수를 조절한다.

재생성 명령:

```bat
cd C:\Users\user\Documents\last_project\Group-Chat-agent
python -m llm_server.mergePRAG.prepare_service_hardpairs --output-dir C:\Users\user\Documents\last_project\data
```

현재 코드가 최신이면 출력은 아래처럼 나와야 한다.

```text
[service_hardpairs] train=4920 -> C:\Users\user\Documents\last_project\data\ServiceHardPair_train.jsonl
[service_hardpairs] valid=240 -> C:\Users\user\Documents\last_project\data\ServiceHardPair_valid.jsonl
```

만약 `train=3480`, `valid=160`이 나오면 확장 도메인 코드가 적용되기 전 파일로 생성했거나, 오래된 브랜치/터미널 상태일 가능성이 있다. 이때는 아래 확인을 먼저 한다.

```bat
findstr /n "ROOMS_KO SCORES_KO POLICIES_KO ASSIGNMENTS_KO" llm_server\mergePRAG\prepare_service_hardpairs.py
findstr /n "ko_room ko_score ko_policy ko_assignment" llm_server\mergePRAG\prepare_service_hardpairs.py
```

반복 수를 바꾸고 싶으면:

```bat
python -m llm_server.mergePRAG.prepare_service_hardpairs --output-dir C:\Users\user\Documents\last_project\data --train-repeats 40 --valid-repeats 5
```

데이터를 크게 바꾼 뒤에는 기존 `.pt`를 삭제하고 처음부터 재학습한다.

```bat
del llm_server\mergePRAG\hypernet_checkpoint.pt
del llm_server\mergePRAG\hypernet_weights.pt
python -m llm_server.mergePRAG.train
```

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
train 4920개
valid 240개
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

한국어 synthetic 진단:

```bat
cd C:\Users\user\Documents\last_project\Group-Chat-agent\llm_server
set MERGEPRAG_LOAD_SOURCE=checkpoint
python test_mergeprag_synthetic_ko.py
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

한국어 synthetic 구조:

```text
question: 테바 표식에 배정된 암호어는 뭐야?
main passage:    테바 -> 가론, 모린 -> 리펜
compare passage: 테바 -> 리펜, 모린 -> 가론

원하는 결과:
main memory    -> 가론
compare memory -> 리펜
```

같은 스크립트를 환경변수로도 선택할 수 있다.

```bat
set MERGEPRAG_DIAGNOSTIC_CASE=synthetic
python test_mergeprag.py

set MERGEPRAG_DIAGNOSTIC_CASE=synthetic_ko
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

python debug_mergeprag_synthetic_ko.py
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
- `cos(V)`: main/compare memory가 얼마나 다른지 보는 핵심 보조 지표다. 0.99 이상이면 거의 같은 내용을 주입하는 것이고, 0.98 근처까지 내려와도 Candidate Choice가 안 갈리면 아직 relation grounding은 실패로 본다.

성공에 가까운 상태:

```text
no_hook      -> Not provided
direct main  -> Monday
direct comp  -> Friday
main hook    -> Monday
compare hook -> Friday
main_choice=Monday
compare_choice=Friday
synthetic_ko main_choice=가론
synthetic_ko compare_choice=리펜
cos(V) ideally < 0.95, 더 좋으면 < 0.8
```

현재 마지막 알려진 실패 상태:

```text
token embedding skip + slot-wise pooling + synthetic_ko, checkpoint 500
cos(pooled)=0.9798
cos(K)=0.9793
cos(V)=0.9780
direct main/direct comp는 성공
Candidate Choice는 main/compare가 여전히 같은 답으로 움직임
```

## 지금 남은 핵심 문제

### 1. Passage-relation grounding 부족

초기에는 V만 높은 것이 문제처럼 보였지만, 진단을 분해해 보면 더 정확한 병목은 "질문 대상과 passage 안의 값 사이 관계"가 K/V memory에 안정적으로 들어가지 않는 것이다.

현재까지 관찰:

```text
direct prompt: passage를 직접 넣으면 base LLM은 정답/반대정답을 구분한다.
no_hook: passage 없이는 synthetic_ko 내용을 모른다.
hook: K/V 주입은 logit을 크게 바꾸지만 main memory와 compare memory의 분포 차이는 아직 작다.
same passage + different question: 분리력이 좋아졌다.
same question + swapped passage: 아직 반대 답으로 flip이 약하다.
```

즉 hook 자체는 동작한다. 문제는 hook이 주입하는 memory가 "닥스멜/테바가 어떤 값에 연결되는지" 같은 relation을 충분히 분리하지 못한다는 점이다.

최근 적용한 작업:

- `query_focus_mask`: gold answer 없이 질문 핵심 단어 주변 passage window를 표시.
- `AttentivePooling`: 해당 window에 `QUERY_LEXICAL_FOCUS_SCALE`만큼 attention boost.
- `slot-wise pooling`: `num_kv=4` slot마다 별도 attention map을 사용해 passage의 다른 위치를 직접 보게 함.
- `token embedding skip`: contextual hidden에 raw token embedding을 더해 날짜/entity token identity 보존.
- `ServiceHardPair` 데이터에 영어/한국어 임의 code-word hard pair를 추가해 `테바 -> 가론` 같은 사전지식 없는 매핑도 학습 패턴에 포함.
- 장소, 점수, 정책, 담당자 도메인 hard pair를 추가해 relation flip 패턴을 넓혔다.
- `KV_PATH_MODE`: 기본값을 `k_mlp_v_hybrid`로 변경.

다음 판단:

- 현재 코드 변경과 데이터셋 확장은 기존 `.pt`와 호환되는 실험이 아니므로 기존 `.pt`를 제거하고 처음부터 재학습한다.
- 재생성된 데이터셋 크기가 `train=4920`, `valid=240`인지 먼저 확인한다.
- 500 step에서 `test_mergeprag.py`, `test_mergeprag_synthetic.py`, `test_mergeprag_synthetic_ko.py`를 `checkpoint` 기준으로 본다.
- 성공 신호는 service case에서 `main=Monday`, `compare=Friday`, synthetic_ko에서 `main=가론`, `compare=리펜`으로 Candidate Choice와 generation이 동시에 갈리는 것이다.

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

1. `prepare_service_hardpairs.py`에 `ROOMS`, `SCORES`, `POLICIES`, `ASSIGNMENTS` 계열이 들어있는지 확인한다.
2. ServiceHardPair 데이터를 다시 생성하고 출력이 `train=4920`, `valid=240`인지 확인한다.
3. 기존 `hypernet_checkpoint.pt`, `hypernet_weights.pt`를 제거한 뒤 처음부터 재학습한다.
4. 500 step에서 `MERGEPRAG_LOAD_SOURCE=checkpoint`로 service, synthetic, synthetic_ko 진단을 모두 실행한다.
5. service case는 `main=Monday`, `compare=Friday`, synthetic_ko는 `main=가론`, `compare=리펜`으로 Candidate Choice가 갈리는지 본다.
6. 실패하면 `debug_mergeprag.py`와 `debug_mergeprag_synthetic_ko.py`에서 `KL(main || compare)`, `cos(pooled)`, `cos(K)`, `cos(V)`, slot ablation을 같이 본다.
7. 우선순위는 "K/V cosine 숫자만 낮추기"가 아니라 "주입 후 답이 passage에 맞게 뒤집히는지"다.

현재 프로젝트의 방향은 "외부 데이터셋 일반 QA 성능"보다 "주입된 발화 passage가 답변을 실제로 뒤집는가"에 맞춰져 있다. 다른 AI가 이어받을 때도 이 기준을 최우선으로 봐야 한다.