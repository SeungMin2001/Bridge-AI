# PRAG 실험 계획 및 진행 기록

이 문서는 기존 인수인계 문서와 별도로, 현재 진행 중인 PRAG 실험의 방향, 학습 순서, 비교 실험 조건, 진단 기준을 계속 업데이트하기 위한 실험 기록 문서다.

기존 `README.md`는 `인수인계.md`로 이동했다. 이 문서는 앞으로 실험 계획과 결과를 누적하는 용도로 사용한다.

## 1. 최종 목표

우리 서비스의 목표는 사용자가 수업/회의 전사 내용에 대해 질문했을 때, 관련 passage를 K/V memory로 주입하고 모델이 그 주입 내용을 기반으로 자연스럽게 설명형 답변을 생성하는 것이다.

성공 기준은 단순 후보 선택이 아니라 아래 형태다.

```text
입력 passage: 철수는 선문대학교 학생이다.
질문: 철수는 어느 학교 학생이야?
모델 답변: 철수는 선문대학교 학생입니다.
```

즉, 모델 prompt에 passage를 직접 넣지 않아도 K/V 주입만으로 passage의 정보를 반영한 자유생성 답변이 나와야 한다.

## 2. 나중에 진행할 핵심 비교 실험

나중에 아래 두 하이퍼네트워크를 같은 조건에서 비교한다.

### A. 기존 하이퍼네트워크

passage만 기반으로 K/V memory를 만드는 baseline이다. 질문 조건부 정보가 K/V 생성 과정에 직접 반영되지 않거나 약하게 반영되는 구조를 비교 기준으로 둔다.

### B. 질문+passage 조건부 하이퍼네트워크

현재 주력 구조다. 사용자 질문과 관련 passage를 함께 사용해 K/V memory를 만든다.

```text
question + passage
-> base LLM contextual hidden state
-> HyperNetwork
-> K/V memory
-> target layer injection
-> answer generation
```

비교 실험에서는 두 구조 모두 같은 모델, 같은 데이터셋, 같은 train/valid split, 같은 step 수, 같은 평가 스크립트를 사용해야 한다.

## 3. 공정 비교를 위한 고정 조건

비교 실험에서 반드시 고정할 조건은 다음과 같다.

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Critical layer: `19`
- `num_kv`: `16`
- `hidden_dim`: `1024`
- `alpha`: `1.0`
- Dataset: 동일한 `PRAG_multifact_augmented_train.jsonl`, `PRAG_multifact_augmented_valid.jsonl`
- Evaluation: 동일한 `test.py`, `test_single_ko.py`
- 핵심 지표: `candidate_flip_ok`와 별도로 `main_kv_generation_hits`, `neg_kv_generation_hits`, `gen-val main_kv/neg_kv`

비교 대상이 되는 구조만 바꾸고, 나머지 조건은 최대한 동일하게 유지한다.

## 4. 현재 학습 방향

현재는 먼저 자유생성 답변을 성공시키는 것이 우선이다. 비교 실험은 자유생성 학습 방향이 어느 정도 안정된 뒤 진행한다.

현재 학습 흐름은 다음과 같다.

```text
1. multi-fact 데이터셋 구축
2. multifact 기준 4 epoch 학습
3. 자유생성 답변 개선용 추가 학습
4. 실제 전사문 스타일 transcript 데이터셋 추가학습
5. KorQuAD 추가학습
6. 기존 하이퍼네트워크 vs 질문+passage 하이퍼네트워크 비교
```

## 5. Multi-fact 기본 학습

현재 주 데이터셋은 single-fact가 아니라 multi-fact 데이터셋이다.

```text
train: C:\Users\user\Documents\last_project\data\PRAG_multifact_augmented_train.jsonl
valid: C:\Users\user\Documents\last_project\data\PRAG_multifact_augmented_valid.jsonl
```

데이터 규모는 현재 로그 기준 다음과 같다.

```text
train rows: 3678
valid rows: 919
expanded train examples: 14688
expanded valid examples: 3666
group train examples: 3678
group valid examples: 919
```

기본 학습은 multifact 4 epoch 방향으로 진행했다.

## 6. 자유생성 답변 개선용 학습

기본 ranking 성능은 높지만, 자유생성 답변이 주입 passage의 핵심 구절을 정확히 꺼내지 못하는 문제가 있었다. 이를 개선하기 위해 아래 loss를 추가한 학습을 진행 중이다.

```text
short_answer_weight = 1.0
answer_prefix_weight = 2.0
answer_prefix_tokens = 3
answer_target = full_answer
```

의도는 다음과 같다.

- `full_answer`: 서비스형 자연어 답변을 학습
- `short_answer_weight`: 짧은 핵심 정답 구절도 같이 학습
- `answer_prefix_weight`: 답변 시작 부분에 정답 구절이 나오도록 강화

현재 자유생성 개선용 run은 `step=8000` checkpoint에서 이어서 진행 중이며, 목표 총 step은 약 `18366`이다.

이어 학습 명령어:

```bash
python -m llm_server.PRAG.train --multifact --epochs 1 --short-answer-weight 1.0 --answer-prefix-weight 2.0 --answer-prefix-tokens 3
```

정상 resume이면 아래 로그가 나와야 한다.

```text
[PRAG:train] resumed step=8000 best_val=...
```

현재 run에서는 `gen-val main_kv`, `gen-val neg_kv`가 가장 중요하다. `candidate_flip_ok`가 높아도 자유생성 답변이 실패하면 서비스 목표에는 아직 부족하다.

## 7. 자유생성 학습 완료 후 진단

자유생성 개선용 학습이 끝나면 먼저 multifact valid 기준으로 진단한다.

```bash
python -m llm_server.PRAG.test --multifact --max-samples 300 --show 20 --alpha 1.0 --answer-target auto --max-new-tokens 64
```

한국어 단일 샘플 진단:

```bash
python -m llm_server.PRAG.test_single_ko --weights llm_server/PRAG/prag_multifact_memory_weights.pt --max-new-tokens 64 --alpha 1.0
```

진단에서 봐야 할 핵심:

- `candidate_main_ok`
- `candidate_neg_ok`
- `candidate_flip_ok`
- `shown_main_kv_generation_hits`
- `shown_neg_kv_generation_hits`
- `shown_direct_passage_hits`
- `shown_no_memory_hits`
- `shown_zero_kv_hits`

좋은 상태는 `no_memory`와 `zero_kv`는 낮고, `main_kv_generation_hits`와 `neg_kv_generation_hits`가 올라가는 것이다.

## 8. 실제 전사문 스타일 transcript 추가학습

최근 진단에서 중요한 차이가 확인됐다.

```text
학습 데이터와 비슷한 구조:
  passage: '긴급 문의 응답 시간'은 '3일 이내'가 아니라 '30분 이내'
  expected: 긴급 문의 응답 시간은 30분 이내이다.
  with_KV_loss: 0.0364
  direct_recovery: 1.250
  generation_hit: True

실제 수업 전사문에 가까운 구조:
  passage: 어 오늘 수업 녹화는요, 끝나고 나면 이캠퍼스 자료실에 올려둘게요...
  expected: 수업 녹화 파일은 이캠퍼스 자료실에 올라옵니다.
  with_KV_loss: 2.0312
  direct_recovery: 0.009
  generation_hit: False
```

해석은 명확하다. 현재 multifact 학습은 구조화된 데이터셋 분포에서는 K/V 주입이 잘 되지만, 실제 강의/회의 전사문처럼 filler, 반복, 구어체 설명, 자연스러운 문맥이 섞인 passage에는 일반화가 약하다. 따라서 우리 서비스 목표에는 transcript-style 한국어 데이터셋이 별도로 필요하다.

### 8-1. transcript source 데이터셋

새 transcript 데이터셋은 기존 `PRAG_multifact_*` 파일을 덮어쓰지 않고 완전히 분리한다.

```text
source: C:\Users\user\Documents\last_project\data\PRAG_transcript_sources.jsonl
train:  C:\Users\user\Documents\last_project\data\PRAG_transcript_augmented_train.jsonl
valid:  C:\Users\user\Documents\last_project\data\PRAG_transcript_augmented_valid.jsonl
```

source 생성은 한국어 전용으로 진행한다. 서비스가 한국어 수업/회의 전사문을 대상으로 하므로, 현재 단계에서는 영어를 섞지 않는다.

```bash
python -m llm_server.PRAG.build_transcript_sources --rows-per-domain 80 --facts-per-passage 3 --languages ko --seed 42
```

현재 설정 기준 source row 목표는 약 `3280`개다. 각 row는 하나의 강의/회의 장면이며, 내부에 3개 fact를 가진다.

### 8-2. LLM 증강 방식

source row는 바로 학습하지 않고, 더 좋은 LLM인 `Qwen/Qwen3.5-4B`가 실제 전사문 스타일로 다시 쓴다.

증강 결과 row 구조:

```text
passage:
  실제 수업/회의 발화처럼 filler, 설명, 반복, 비유가 포함된 전사문

atomic_qas:
  passage 안의 개별 fact마다 question / answer / full_answer / sub_passage 생성

final_qas:
  한 passage 안의 여러 fact를 묶어 설명하는 최종 질문/답변 생성

hard_negatives:
  가능하면 counterfactual passage를 생성하지만, 품질이 깨지면 positive-only 학습을 우선한다.
```

증강 명령어:

```bash
python -m llm_server.PRAG.augment_transcript --backend vllm --vllm-url http://localhost:8001/v1/chat/completions --model Qwen/Qwen3.5-4B --max-new-tokens 2048 --no-resume
```

중간에 끊긴 뒤 이어서 증강:

```bash
python -m llm_server.PRAG.augment_transcript --backend vllm --vllm-url http://localhost:8001/v1/chat/completions --model Qwen/Qwen3.5-4B --max-new-tokens 2048
```

소량 샘플 디버그:

```bash
python -m llm_server.PRAG.augment_transcript --backend vllm --vllm-url http://localhost:8001/v1/chat/completions --model Qwen/Qwen3.5-4B --max-new-tokens 2048 --max-samples 10 --no-resume --debug-invalid-raw
```

증강 샘플 확인:

```bash
python -m llm_server.PRAG.preview_data --transcript --split train --samples 3 --max-qas 4
```

검증:

```bash
python -m llm_server.PRAG.validate --transcript --show 5
```

### 8-3. transcript 추가학습

transcript 학습은 기존 multifact 가중치를 초기값으로 사용하되, 결과 파일은 별도 저장한다.

```text
weights:    llm_server/PRAG/prag_transcript_memory_weights.pt
checkpoint: llm_server/PRAG/prag_transcript_memory_checkpoint.pt
log:        llm_server/PRAG/prag_transcript_train_log.json
```

추천 시작 명령어:

```bash
python -m llm_server.PRAG.train --transcript --epochs 1 --no-resume --init-weights llm_server/PRAG/prag_multifact_memory_weights.pt --lr 2e-5 --positive-only --short-answer-weight 1.0 --answer-prefix-weight 2.0 --answer-prefix-tokens 3 --eval-generation-samples 20 --eval-generation-every 250 --eval-generation-max-new-tokens 64
```

현재 transcript 증강에서는 negative 품질이 불안정할 수 있으므로 `--positive-only`를 우선 사용한다. 목표는 negative flip보다 실제 전사문 passage의 핵심 내용을 K/V 주입만으로 자연스럽게 답변하는 능력이다.

### 8-4. transcript 진단 기준

통계 진단:

```bash
python -m llm_server.PRAG.test --transcript --max-samples 300 --show 20 --alpha 1.0 --answer-target auto --max-new-tokens 64
```

한국어 단일 샘플 진단:

```bash
python -m llm_server.PRAG.test_single_ko --transcript --max-new-tokens 64 --alpha 1.0
```

transcript 단계에서 가장 중요한 지표:

- `with_KV_loss`: K/V 주입 후 gold 답변 loss
- `memory_gain`: no-memory 대비 K/V가 답변 loss를 얼마나 낮췄는지
- `direct_recovery`: direct passage prompt 성능을 K/V가 얼마나 회복했는지
- `answer_prefix_loss@3`: 답변 초반이 정답 구절로 시작하도록 학습되는지
- `generation_hit`: 자유생성 답변에 기대 핵심 구절이 들어갔는지

이 단계에서는 `candidate_flip_ok`보다 `generation_hit`, `direct_recovery`, `gen-val main_kv`를 더 중요하게 본다.

### 8-5. 주입 연산 방식 ablation

기존 PRAG 경로는 target layer에서 `hidden + alpha * cross_attention(hidden, K, V)` 방식으로 K/V memory를 읽는다. 그러나 실제 전사문 스타일 진단에서 자유생성 답변이 여전히 passage 핵심 구절을 놓치는 경우가 있어, 데이터셋과 모델은 그대로 두고 주입 연산만 바꿔 비교할 수 있도록 `--injection-mode`를 추가했다.

지원 모드:

- `attention`: 기존 PRAG-style cross-attention 주입
- `add_all`: `V` 평균 벡터를 모든 토큰 hidden state에 직접 더하는 additive-only ablation
- `add_last`: 마지막 토큰 hidden state에만 `V` 평균 벡터를 더하는 ablation
- `hybrid`: attention 주입 후 additive bias를 추가하는 혼합 방식

중요한 점은 inference에서만 덧셈을 적용하면 공정한 비교가 아니라는 것이다. `add_all`의 가능성을 보려면 학습부터 진단까지 같은 `--injection-mode add_all`을 사용해야 한다.

additive-only transcript 실험 예시:

```bash
python -m llm_server.PRAG.train --transcript --epochs 1 --no-resume --init-weights llm_server/PRAG/prag_multifact_memory_weights.pt --lr 2e-5 --positive-only --short-answer-weight 2.0 --answer-prefix-weight 10.0 --answer-prefix-tokens 10 --eval-generation-samples 30 --eval-generation-every 250 --eval-generation-max-new-tokens 128 --injection-mode add_all
```

진단 예시:

```bash
python -m llm_server.PRAG.test_single_ko --weights llm_server/PRAG/prag_transcript_memory_checkpoint.pt --synthetic-case process_restaurant --injection-mode add_all --max-new-tokens 64 --alpha 1.0
```

## 9. KorQuAD 추가학습 계획

자유생성 개선용 multifact 학습이 끝난 뒤 KorQuAD를 추가학습한다.

KorQuAD는 실제 한국어 context/question/answer를 사용한다. synthetic으로 비슷하게 만든 데이터가 아니라, KorQuAD 안의 실제 passage 내용을 우리 PRAG augmented schema로 변환해 사용한다.

KorQuAD 변환:

```bash
python -m llm_server.PRAG.prepare_korquad --max-train-records 3000 --max-valid-records 600 --qas-per-row 3 --include-final
```

샘플 확인:

```bash
python -m llm_server.PRAG.preview_data C:\Users\user\Documents\last_project\data\PRAG_korquad_augmented_train.jsonl --samples 3 --max-qas 4
```

검증:

```bash
python -m llm_server.PRAG.validate C:\Users\user\Documents\last_project\data\PRAG_korquad_augmented_train.jsonl C:\Users\user\Documents\last_project\data\PRAG_korquad_augmented_valid.jsonl --show 0
```

KorQuAD 추가학습:

```bash
python -m llm_server.PRAG.train --korquad --epochs 1 --no-resume --init-weights llm_server/PRAG/prag_multifact_memory_weights.pt --lr 1e-5 --short-answer-weight 1.0 --answer-prefix-weight 2.0 --answer-prefix-tokens 3 --final-weight 0.25
```

KorQuAD 진단:

```bash
python -m llm_server.PRAG.test --korquad --max-samples 300 --show 20 --alpha 1.0 --answer-target auto --max-new-tokens 64
```

KorQuAD는 서비스 데이터와 분포가 다르므로, 기존 multifact grounding을 망치지 않도록 낮은 learning rate와 낮은 final weight로 시작한다.

## 10. 나중에 비교 실험에 사용할 학습 레시피

자유생성 답변이 안정되면 아래 레시피를 기준으로 두 구조를 비교한다.

### 공통 레시피

```text
dataset: PRAG_multifact_augmented_train/valid
base training: multifact 4 epoch
generation finetune: short_answer_weight=1.0, answer_prefix_weight=2.0, answer_prefix_tokens=3
transcript finetune: Korean transcript-style positive-only 추가학습
optional extra finetune: KorQuAD low-lr 추가학습
evaluation: same alpha, same max_new_tokens, same valid split
```

### 비교 대상

```text
baseline: passage-only hypernetwork
ours: question+passage conditioned hypernetwork
```

### 비교 지표

```text
candidate_main_ok
candidate_neg_ok
candidate_flip_ok
atomic/final split 성능
direct passage generation hit
main K/V generation hit
negative K/V generation hit
no-memory hit
zero-KV hit
test_single_ko 자유생성 결과
학습 시간
추론 시간
```

## 11. 문서 업데이트 규칙

앞으로 다음 상황이 생기면 이 문서를 업데이트한다.

- 학습 명령어가 바뀐 경우
- 데이터셋이 바뀐 경우
- checkpoint/weights 경로가 바뀐 경우
- 중요한 진단 결과가 나온 경우
- 비교 실험 조건이 확정된 경우
- transcript-style 데이터셋/학습 결과가 나온 경우
- KorQuAD 또는 외부 QA 추가학습 결과가 나온 경우

이 문서는 논문 작성과 실험 재현을 위한 기준 문서로 사용한다.
