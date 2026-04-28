# MergePRAG Service Memory 구현 문제점 보고서

## 문서 목적

본 문서는 현재 `service_memory.py` / `service_memory_train.py` 코드베이스의 문제를 진단하고, 코딩 작업을 수행할 AI가 **올바른 방향으로 수정**하도록 지시하기 위한 보고서이다. 단순한 버그 리스트가 아니라 **각 문제의 근본 원인 → 수정 방향 → 검증 방법**을 함께 제공한다.

---

## 0. 프로젝트 목표 (재확인)

- **Use case**: 학습 지원 서비스. 교수가 한 passage(예: "프로세스는 실행 중인 프로그램이다")를 제공하면, 그 정보를 모델 파라미터 공간에 K/V 메모리로 주입한다. 사용자가 이후 질문할 때 모델은 prompt에 passage 없이도 그 정보를 활용해 답해야 한다.
- **현재 상태**: hypernetwork가 학습은 진행되는 듯하지만 inference 시 passage 정보가 모델에 제대로 주입되지 않음.
- **기반 논문**: MergePRAG (ICLR 2026, Liu et al., JBNU). 다만 multi-hop / orthogonal merging은 사용하지 않고, single-passage K/V injection 메커니즘만 차용함.

---

## 1. 핵심 진단: 한 줄 요약

> **Hypernetwork가 학습 신호를 받기 전에, loss가 K/V를 우회해서 줄어들 수 있는 경로가 여러 개 있다. 그리고 loss objective가 너무 많은 항으로 쪼개져 어느 항도 dominant하지 못해 학습 신호가 분산된다.**

이 두 문제 때문에 hypernetwork가 "passage를 K/V로 압축한다"는 핵심 작업을 학습하지 못하고 있을 가능성이 매우 높다.

---

## 2. 문제 분류

문제는 다음 4개 카테고리로 분류된다. 우선순위는 ★표시(★★★ = 가장 치명적)로 표기한다.

| # | 카테고리 | 우선순위 | 핵심 문제 |
|---|--------|--------|----------|
| A | 학습 신호 누설 | ★★★ | K/V 무시해도 loss가 줄어듦 |
| B | Loss objective 구성 | ★★★ | 6개 항이 동시에 작용, 서로 충돌 |
| C | Hypernetwork 구조 | ★★ | contextual feature, skip connection이 학습 우회로 제공 |
| D | Cross-attention 주입 | ★★ | head 수 하드코딩, GQA 무시, LM attention space와 정렬 안 됨 |

---

## 3. 문제 A: 학습 신호 누설 (★★★)

### 3-1. 문제 A-1: Teacher distillation이 정답 누설 가능성

**위치**: `service_memory_train.py`의 `hard_pair_objective()`

```python
direct_gold_tok = tokenize_direct_qa(tokenizer, question, sample["passage"], gold, device)
# direct_gold_tok은 prompt에 passage 통째로 포함

with torch.no_grad():
    direct_gold_logits = model(**direct_gold_tok)["logits"]

teacher_kl_main = answer_kl_loss(
    main_gold_logits,      # student: K/V 주입 + question만 (passage 없음)
    gold_tok["labels"],
    direct_gold_logits,    # teacher: passage 인-컨텍스트 + question
    ...
)
```

**원리상 의도**: DistilledPRAG처럼 "passage를 본 모델"을 teacher로, "K/V만 받은 모델"을 student로 두고 정합 학습.

**실제로 일어나는 일**:
- Teacher는 passage를 prompt에서 직접 보므로 정답에 매우 confident한 distribution을 만든다
- Student는 K/V가 부실해도 question만으로 가능한 답을 생성하면서 teacher distribution에 KL로 끌려간다
- 결과적으로 **K/V는 teacher 모방의 보조 신호 정도가 되고, 핵심 학습 신호가 되지 못한다**

이 자체는 잘 설계되면 작동할 수 있으나, 문제 A-2와 결합되면 K/V가 사실상 무시될 수 있다.

### 3-2. 문제 A-2: CE loss가 K/V 없이도 줄어드는 구조

**위치**: `service_memory_train.py`의 `hard_pair_objective()`, 그리고 `service_memory.py`의 `tokenize_qa()`

```python
# tokenize_qa는 prompt에 passage를 포함시키지 않음 - 이건 OK
prompt = build_chat_prompt(tokenizer, question)  # passage 없음

# 그런데 이런 prompt 형태로:
# system: "Use only the injected lecture memory or the provided lecture content.
#          If the answer is not explicitly supported, answer exactly: Unknown."
# user: "Question: {question}\nAnswer with only the short final answer."
# assistant: <answer>  ← 학습 시 정답이 labels로 주어짐
```

**실제로 일어나는 일**:
- 학습 시 모델은 `answer` 토큰들을 labels로 받아서 cross-entropy를 최소화한다
- Question + system prompt만으로도 일반 상식 질문이라면 정답에 가까운 분포를 만들 수 있다
- 더 결정적으로: **transformer는 다음 토큰 예측에서 이전 정답 토큰을 보고 다음 정답 토큰을 예측한다 (teacher forcing)**. 즉 정답이 labels로 들어가면, 모델은 정답 sequence를 생성하는 데 K/V 도움을 거의 안 받는다.
- K/V를 0으로 만들어도 CE loss는 거의 비슷하게 줄어들 가능성이 매우 높다

### 3-3. 수정 방향

**수정 A-1 (필수)**: K/V 무용성 검증 테스트를 먼저 실행한다. (자세한 코드는 § 7-1)

**수정 A-2 (필수)**: 학습 데이터 구조를 다음과 같이 강화한다.
- 질문은 passage 내용 없이는 답할 수 없도록 설계 (entity-specific QA)
- 정답은 짧은 phrase (단어 1~5개)로 통일
- 모델의 사전 지식만으로는 절대 답할 수 없는 entity 사용 (예: 가상 인물명, 가상 정의)

```json
// 좋은 예시 (모델 사전지식으로 답할 수 없음)
{
  "passage": "교수: 우리 수업에서 'Ferganite'는 압력에 강한 가상 합금을 의미한다.",
  "question": "Ferganite의 정의는?",
  "answer": "압력에 강한 가상 합금"
}

// 나쁜 예시 (모델이 이미 알고 있을 가능성)
{
  "passage": "프로세스는 실행 중인 프로그램이다.",
  "question": "프로세스란?",
  "answer": "실행 중인 프로그램"
}
```

**이유**: 모델이 사전지식만으로 답할 수 있는 데이터는 K/V 학습 검증에 부적합하다. 학습이 진짜로 일어나는지 보려면 **사전지식만으로는 답할 수 없는 데이터**가 필요하다.

**수정 A-3 (선택)**: Teacher KL은 일단 weight=0으로 비활성화한다. Baseline이 동작 확인된 후에만 다시 추가한다.

---

## 4. 문제 B: Loss Objective 구성 문제 (★★★)

### 4-1. 현재 objective의 구조

```python
objective = (
    main_gold                         # CE on (passage K/V → question → answer)
    + DUAL_CE_WEIGHT * neg_neg        # CE on (negative K/V → question → neg_answer)
    + RANK_WEIGHT * rank              # margin: main_gold < main_neg AND neg_neg < neg_gold
    + TEACHER_KL_WEIGHT * teacher_kl  # KL with passage-in-context teacher
    + SEPARATION_WEIGHT * sep         # K/V cosine 분리 (서로 다른 passage끼리)
    + DIVERSITY_WEIGHT * div          # slot 직교 (한 passage 안에서)
)
```

### 4-2. 문제점

**문제 B-1: 항이 너무 많아 학습 신호가 분산**
- 6개 항이 동시에 hypernetwork 하나의 파라미터를 잡아당김
- 가중치를 환경변수로 일일이 튜닝해야 함 → 디버깅 surface 폭발

**문제 B-2: 항 간 충돌**
- `sep`: 다른 passage들의 K/V를 서로 떼어놓으려 함
- `div`: 한 passage 안의 slot들끼리 직교화
- `main_gold` + `teacher_kl`: K/V가 정답 정보를 담도록 압박
- → Hypernetwork는 **K/V를 거의 random에 가깝게** 출력해도 sep, div를 동시에 만족시킬 수 있다. 그러면 정답 정보가 K/V에 거의 안 실린다.

**문제 B-3: Step당 forward pass 폭발**
- `main_gold`, `main_neg`, `neg_gold`, `neg_neg` (hook 등록/해제 포함) → 4번 forward
- `direct_gold`, `direct_neg` (no_grad teacher) → 2번 forward
- `encode_memory(main)`, `encode_memory(neg)` (use_contextual=True일 때 LM forward 포함) → 2번 forward
- **총 8번 / step**. 매우 느리고, 강제로 batch_size=1이 됨.

**문제 B-4: `slot_diversity_loss`가 "token" pooling mode에서 항상 0**
```python
if K.size(1) > SERVICE_NUM_KV:
    return K.new_tensor(0.0)
```
현재 default `pooling_mode="token"`에서는 K의 두 번째 차원이 token 수 (보통 > num_kv)이므로 `div` 항은 항상 0이다. 매 step 의미 없는 계산이 일어나고 있고, 로그의 div 값은 항상 0이다.

### 4-3. 수정 방향

**수정 B-1 (필수)**: **단계적 baseline 구축 전략을 채택한다.**

Phase 1 (Minimal baseline): 다음 weight를 모두 0으로 설정하고 학습한다.
```python
RANK_WEIGHT = 0.0
DUAL_CE_WEIGHT = 0.0
TEACHER_KL_WEIGHT = 0.0
SEPARATION_WEIGHT = 0.0
DIVERSITY_WEIGHT = 0.0
# 결과: objective = main_gold (단일 항 CE)
```

이 상태에서 **overfit test 통과**가 첫 번째 milestone이다. (§ 7-2 참고)

Phase 2: Phase 1이 통과되면 **한 번에 한 항씩만** 추가한다. 각 항 추가 후 validation metric이 **올라가는지** 측정. 안 올라가면 그 항은 제거한다.

추가 우선순위: `dual_ce` → `rank` → `teacher_kl` → `sep` → `div`

**수정 B-2 (필수)**: 항 추가 시 다음 sanity check를 매번 통과해야 한다:
- K/V를 0으로 강제하면 loss가 **눈에 띄게 악화**되어야 한다 (악화 안 되면 그 항은 K/V 학습에 기여 안 함)

**수정 B-3 (필수)**: `slot_diversity_loss`를 token mode에서는 코드에서 명시적으로 비활성화하거나, slot mode에서만 사용하도록 강제한다. 현재는 silent로 0을 반환하므로 디버깅이 혼란스럽다.

```python
# service_memory_train.py 학습 루프 진입 전에 명시적 검증
if SERVICE_POOLING_MODE == "token" and DIVERSITY_WEIGHT > 0:
    raise ValueError(
        "DIVERSITY_WEIGHT > 0 is incompatible with pooling_mode='token'. "
        "Set DIVERSITY_WEIGHT=0 or use pooling_mode='slot'."
    )
```

---

## 5. 문제 C: Hypernetwork 구조 문제 (★★)

### 5-1. 문제 C-1: Contextual feature가 학습 우회로 제공

**위치**: `service_memory.py`의 `passage_features()`

```python
if use_contextual:
    outputs = model.model(input_ids=..., ...)
    contextual = outputs.last_hidden_state.to(dtype=torch.float32)
    features = torch.cat([raw, contextual], dim=-1)  # [B, L, 2*d_model]
```

**문제**:
- LLaMA 마지막 layer의 hidden state는 **next-token prediction에 필요한 거의 모든 정보**를 담고 있다
- Hypernetwork 입력에 이게 포함되면, hypernetwork는 사실상 identity-like projection만 학습해도 "passage를 압축한 K/V"처럼 보이는 출력을 낼 수 있다
- 진짜 의미 있는 압축 학습이 일어나지 않음

**원본 MergePRAG 논문 코드 비교**: 원본은 `model.model.embed_tokens(input_ids)` 즉 **raw token embedding만** 사용한다. 이게 hypernetwork에게 "압축할 거리"를 제공하는 적절한 입력이다.

### 5-2. 문제 C-2: Skip connection이 raw passage를 그대로 흘림

**위치**: `service_memory.py`의 `ServiceMemoryHyperNetwork.forward()`

```python
K = self.linear_K(hidden) + self.skip_scale * self.skip_K(pooled)
V = self.linear_V(hidden) + self.skip_scale * self.skip_V(pooled)
```

**문제**:
- `pooled`는 `input_proj(passage_embedding)`의 결과
- Skip connection으로 raw passage projection이 K/V에 직접 더해짐 (`skip_scale=0.5`로 강하게)
- MLP가 학습 안 해도 K/V에 passage 정보가 흐르는 우회로
- MLP의 학습 신호가 약해짐

### 5-3. 문제 C-3: `pooling_mode = "token"`의 의미 모호성

**위치**: `service_memory.py`의 `ServiceMemoryHyperNetwork.forward()`

```python
if self.pooling_mode == "token":
    pooled = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
    hidden = self.mlp(pooled)
```

**문제**:
- "token" mode에서는 사실상 모든 token이 각각의 K/V slot이 됨
- `slot_queries` 파라미터가 사용되지 않음 (slot 메커니즘 무력화)
- `num_kv` 파라미터의 의미가 모호 (slot 수가 아니라 max token 수)
- `slot mode`와 완전히 다른 메커니즘이 같은 class의 분기로 처리됨 → 한 mode 디버깅이 다른 mode에 옮겨가지 않음

### 5-4. 수정 방향

**수정 C-1 (필수)**: Phase 1 baseline에서는 `use_contextual=False`로 강제한다.
```python
SERVICE_USE_CONTEXTUAL = False  # baseline
```

**수정 C-2 (필수)**: Phase 1 baseline에서는 `skip_scale=0.0`으로 강제한다.
```python
SERVICE_SKIP_SCALE = 0.0  # baseline
```

**수정 C-3 (필수)**: Phase 1 baseline에서는 `pooling_mode="slot"`을 사용한다. 이게 원래 설계 의도(고정 slot 개수의 K/V 메모리)에 맞다.
```python
SERVICE_POOLING_MODE = "slot"
SERVICE_NUM_KV = 8  # 또는 16
```

**중요**: Phase 1 baseline에서 동작이 확인된 후에만, contextual / skip / token mode를 ablation으로 추가하여 효과를 측정한다.

---

## 6. 문제 D: Cross-attention 주입 메커니즘 (★★)

### 6-1. 문제 D-1: `num_heads=8` 하드코딩

**위치**: `service_memory.py`의 `cross_attention()`

```python
def cross_attention(Q, K, V, num_heads: int = 8) -> torch.Tensor:
    ...
    Qh = Q.view(batch, query_len, num_heads, d_k).transpose(1, 2)
```

**문제**:
- LLaMA-2-7B: 32 attention heads
- LLaMA-3-8B: 32 query heads + 8 KV heads (Grouped Query Attention)
- Qwen2.5-7B: 28 query heads + 4 KV heads (GQA)
- `num_heads=8`은 어떤 모델 head 구조와도 정확히 일치하지 않음

**왜 중요한가**: 주입된 K/V는 LM의 attention space와 같은 head 구조에서 작동해야 의미적으로 정합된다. Head 수가 다르면 주입이 노이즈처럼 흩뿌려진다.

### 6-2. 문제 D-2: K, V가 LM attention의 K/V projection을 거치지 않음

```python
new_hidden = hidden + alpha * cross_attention(hidden, K_local, V_local)
```

**문제**:
- `K_local, V_local`은 hypernetwork의 raw 출력
- LM 자체의 attention block은 `Q = W_Q · h`, `K = W_K · h`, `V = W_V · h`로 projection 한다
- 이 projection을 거치지 않은 K/V는 **LM의 attention space와 정렬되지 않음**
- Hypernetwork가 처음부터 이 정렬을 학습해야 하므로 학습 부담이 매우 큼

### 6-3. 문제 D-3: Hook이 layer output 전체에 적용 (residual stream 직접 수정)

```python
def hook_fn(_module, _input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    ...
    new_hidden = hidden + alpha * cross_attention(hidden, K_local, V_local)
```

**문제**:
- LLaMA decoder layer는 `pre-norm` 구조: `x → norm → attn → x + attn → norm → ffn → x + ffn → output`
- Hook이 layer output에 걸리면, **그 다음 layer의 RMSNorm이 다시 적용**된다
- RMSNorm은 magnitude를 재정규화하므로 `alpha * cross_attention`의 효과가 norm에 의해 흐려짐
- `alpha=0.3`이 실제 hidden state에서 어느 정도 영향력인지 측정 안 된 상태

### 6-4. 수정 방향

**수정 D-1 (필수)**: `num_heads`를 모델 config에서 읽어오도록 수정한다.
```python
# service_memory.py에 추가
def cross_attention(Q, K, V, num_heads: int) -> torch.Tensor:
    ...

# 호출부 (make_memory_hook 등)
num_heads = model.config.num_attention_heads  # LM의 query head 수와 일치
```

또는 hypernetwork 학습 시 사용한 head 수와 inference 시 head 수를 명시적으로 동일하게 유지한다. **head 수 미스매치는 silent failure를 일으키므로 반드시 명시적으로 설정해야 한다.**

**수정 D-2 (선택, 후순위)**: 정렬 문제는 baseline 동작 확인 후 ablation으로 다룬다. 옵션:
- 옵션 A: hypernetwork 출력을 LM의 W_K, W_V로 한 번 더 projection (LM weight 그대로 활용)
- 옵션 B: 현재처럼 raw K/V 사용 (hypernetwork가 정렬을 학습)

옵션 B가 현재 코드의 방식이고, 학습이 충분하면 동작할 수 있다. 옵션 A가 더 안정적일 가능성이 있으나 baseline 우선.

**수정 D-3 (필수)**: Hook 호출 시 hidden state와 delta의 norm을 측정한다. (§ 7-3 참고)

---

## 7. 디버깅 코드: 반드시 실행해야 하는 검증

본 보고서가 가장 중요하게 강조하는 부분이다. **수정 전후 다음 검증을 반드시 거쳐야 한다.**

### 7-1. 검증 1: K/V Necessity Test (가장 중요)

**목적**: 학습된 hypernetwork의 K/V가 정말로 모델 출력에 영향을 주는지 확인.

**방법**: 학습 완료 후 같은 question에 대해 다음 3가지 inference 결과를 비교한다.

```python
# debug_kv_necessity.py
import torch
from llm_server.mergePRAG.service_memory import (
    encode_memory,
    forward_with_memory,
    tokenize_qa,
    SERVICE_ALPHA,
)

@torch.no_grad()
def test_kv_necessity(model, tokenizer, hypernet, target_layer, sample, device):
    """K/V가 출력에 실제로 영향을 주는지 검증."""
    question = sample["question"]
    passage = sample["passage"]
    
    # 1. 정상 K/V (passage 기반)
    mem = encode_memory(model, hypernet, tokenizer, passage, device)
    K_real, V_real = mem["K"], mem["V"]
    
    # 2. Zero K/V (주입 효과 없어야 함)
    K_zero = torch.zeros_like(K_real)
    V_zero = torch.zeros_like(V_real)
    
    # 3. Random K/V (정답 정보 없음)
    K_random = torch.randn_like(K_real) * K_real.std()
    V_random = torch.randn_like(V_real) * V_real.std()
    
    qa_tok = tokenize_qa(tokenizer, question, "", device)
    # answer 빈 문자열 - generation을 위해 prompt만 필요
    
    def generate_with_kv(K, V):
        from torch.nn import functional as F
        # 실제 generation 함수 사용
        # 또는 logits 비교: 첫 토큰의 top-5 logits 출력
        hook = target_layer.register_forward_hook(
            make_memory_hook(K, V, alpha=SERVICE_ALPHA)
        )
        try:
            output = model.generate(
                input_ids=qa_tok["input_ids"][:, :-1],  # answer 토큰 제외
                max_new_tokens=20,
                do_sample=False,
            )
        finally:
            hook.remove()
        return tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"Question: {question}")
    print(f"Expected: {sample['answer']}")
    print(f"Real K/V:   {generate_with_kv(K_real, V_real)}")
    print(f"Zero K/V:   {generate_with_kv(K_zero, V_zero)}")
    print(f"Random K/V: {generate_with_kv(K_random, V_random)}")
    print()
    print("Expected behavior:")
    print("  - Real K/V output should contain the correct answer")
    print("  - Zero K/V output should be vague/wrong (no passage info)")
    print("  - Random K/V output should be incoherent or wrong")
    print()
    print("If Real == Zero: hypernetwork is NOT learning. K/V has no effect.")
    print("If Real == Random: K/V structure has no semantic meaning.")
```

**합격 기준**:
- Real K/V output ≠ Zero K/V output (의미 있게 다름)
- Real K/V output에 정답 entity가 포함됨
- Zero K/V output에는 정답 entity가 없음 또는 "Unknown" 등이 출력됨

**불합격 시**: hypernetwork가 학습 신호를 받지 못하고 있다. 데이터 또는 loss objective 점검.

### 7-2. 검증 2: Single-Sample Overfit Test

**목적**: 메커니즘 자체가 동작하는지 (capacity, gradient flow) 확인.

**방법**: 한 샘플에 대해 200회 반복 학습 후 그 샘플에서 정답이 나오는지 확인.

```python
# 환경변수로 실행
# MERGEPRAG_SERVICE_OVERFIT_CASE=test_case_1 python -m llm_server.mergePRAG.service_memory_train

# 또는 코드 내에서
overfit_sample = {
    "question": "Ferganite는 무엇인가?",
    "answer": "압력에 강한 가상 합금",
    "passage": "교수: 우리 수업에서 'Ferganite'는 압력에 강한 가상 합금이다.",
    "hard_negatives": [{
        "passage": "교수: 우리 수업에서 'Quartzium'은 빛에 반응하는 가상 광물이다.",
        "answer": "빛에 반응하는 가상 광물"
    }]
}

# 200회 반복 학습 후 다음을 확인:
# 1. Final main_gold loss < 0.1 (거의 0)
# 2. K/V Necessity Test 통과
# 3. 학습된 K/V로 generation 시 정답 entity 포함
```

**합격 기준**: 위 3가지 모두 통과.

**불합격 시**: Phase 1 baseline 자체가 동작 안 함. 다음 순서로 점검:
1. Hook이 호출되는지 (print로 확인)
2. delta norm이 0이 아닌지 (§ 7-3)
3. Hypernetwork gradient가 흐르는지 (§ 7-4)

### 7-3. 검증 3: Hook 호출 및 Delta Norm 측정

**목적**: K/V 주입이 hidden state에 실제로 영향을 미치는 정도 측정.

**위치**: `service_memory.py`의 `make_memory_hook` 수정.

```python
# debug 모드 추가
import os
DEBUG_HOOK = os.getenv("MERGEPRAG_DEBUG_HOOK", "0") == "1"

def make_memory_hook(K, V, alpha=SERVICE_ALPHA):
    call_count = [0]
    def hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        K_local = K.to(device=hidden.device, dtype=hidden.dtype)
        V_local = V.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, K_local, V_local)
        new_hidden = hidden + alpha * delta
        
        if DEBUG_HOOK and call_count[0] < 3:  # 처음 3번만 로그
            with torch.no_grad():
                hidden_norm = hidden.norm(dim=-1).mean().item()
                delta_norm = (alpha * delta).norm(dim=-1).mean().item()
                ratio = delta_norm / max(hidden_norm, 1e-8)
                print(f"[HOOK call={call_count[0]}] hidden_norm={hidden_norm:.4f} "
                      f"delta_norm={delta_norm:.4f} ratio={ratio:.4f}")
            call_count[0] += 1
        
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden
    return hook_fn
```

**합격 기준**:
- Hook이 실제로 호출됨 (로그가 찍힘)
- `ratio` (delta_norm / hidden_norm)가 0.05 ~ 0.5 범위
  - < 0.01: 주입 효과 거의 없음
  - > 1.0: 주입이 hidden state를 망가뜨림

**불합격 시**:
- Hook 호출 안 됨: layer 인덱스 잘못 또는 hook 등록 위치 잘못
- ratio < 0.01: alpha를 키우거나 K/V scale 조정 (rms_clamp 완화)
- ratio > 1.0: alpha를 줄이거나 rms_clamp 강화

### 7-4. 검증 4: Hypernetwork Gradient Flow

**목적**: Hypernetwork 파라미터에 gradient가 실제로 도달하는지 확인.

**위치**: 학습 루프의 `optimizer.step()` 직전.

```python
# service_memory_train.py의 학습 루프에 추가
out["objective"].backward()

# Gradient flow 검증
if global_step <= 5:  # 처음 5 step만
    grad_norms = {}
    for name, param in hypernet.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
        else:
            grad_norms[name] = None
    
    print(f"[GRAD step={global_step}]")
    for name, norm in grad_norms.items():
        if norm is None:
            print(f"  {name}: NO GRADIENT (problematic!)")
        elif norm < 1e-8:
            print(f"  {name}: ZERO GRADIENT (problematic!)")
        else:
            print(f"  {name}: {norm:.6f}")

if GRAD_CLIP_NORM > 0:
    torch.nn.utils.clip_grad_norm_(hypernet.parameters(), GRAD_CLIP_NORM)
optimizer.step()
```

**합격 기준**:
- 모든 hypernetwork 파라미터에 gradient가 존재 (None 없음)
- Gradient norm이 0이 아님 (1e-8 미만이면 문제)
- `linear_K`, `linear_V`, `mlp` 레이어들의 gradient가 비슷한 magnitude

**불합격 시**:
- 일부 파라미터 gradient 없음: forward path에서 그 파라미터가 사용 안 됨 (구조 문제)
- 모든 gradient가 0: loss와 hypernetwork 사이 연결이 끊어짐 (`detach()` 등)

---

## 8. 데이터 점검 체크리스트

### 8-1. 학습 데이터의 Passage-Answer 의존성

현재 데이터:
```json
{
  "question": "교수님은 프로세스를 뭐라고 설명했어?",
  "answer": "교수님은 프로세스를 실행 중인 프로그램이라고 설명했습니다.",
  "passage": "프로세스는 실행 중인 프로그램입니다."
}
```

**문제**:
1. `answer`가 너무 길고 passage 표현을 거의 그대로 포함 → 표면적 따라하기만 학습됨
2. Passage 없이도 모델이 사전지식으로 답 가능 → K/V 학습 검증 불가

**수정안**:
```json
{
  "question": "프로세스의 정의는?",
  "answer": "실행 중인 프로그램",
  "passage": "프로세스는 실행 중인 프로그램이다."
}
```

또는 (검증 데이터용으로 더 강력):
```json
{
  "question": "교수가 정의한 'Ferganite'는?",
  "answer": "압력에 강한 가상 합금",
  "passage": "교수: 우리 수업에서 'Ferganite'는 압력에 강한 가상 합금이다."
}
```

### 8-2. 데이터 품질 검증 코드

```python
# debug_data_quality.py
@torch.no_grad()
def check_data_leakage(model, tokenizer, dataset, device, max_check=20):
    """Passage 없이 question만 줘서 모델이 답을 맞히는지 확인.
    너무 많이 맞히면 데이터가 K/V 학습 검증에 부적합."""
    model.eval()
    correct_without_passage = 0
    total = 0
    for sample in dataset[:max_check]:
        question = sample["question"]
        gold = sample["answer"]
        
        prompt = build_chat_prompt(tokenizer, question)
        tok = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        out = model.generate(
            **tok, max_new_tokens=30, do_sample=False,
        )
        pred = tokenizer.decode(out[0][tok.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # 정답 entity가 prediction에 포함되면 누설
        if gold.lower() in pred.lower():
            correct_without_passage += 1
        total += 1
    
    leakage_ratio = correct_without_passage / max(total, 1)
    print(f"Data leakage ratio: {leakage_ratio:.2%} ({correct_without_passage}/{total})")
    print()
    if leakage_ratio > 0.3:
        print("WARNING: > 30% of questions can be answered without the passage.")
        print("This dataset is unsuitable for verifying K/V injection.")
        print("Use entities/concepts that the model cannot know from pretraining.")
```

**합격 기준**: leakage_ratio < 20%

---

## 9. 수정 작업 순서 (코딩 AI에게 지시)

다음 순서를 **반드시 지킬 것**. 단계 건너뛰기 금지.

### Step 1: Baseline 단순화

다음 환경변수 / 기본값으로 설정:
```bash
export MERGEPRAG_SERVICE_USE_CONTEXTUAL=0
export MERGEPRAG_SERVICE_SKIP_SCALE=0.0
export MERGEPRAG_SERVICE_POOLING_MODE=slot
export MERGEPRAG_SERVICE_NUM_KV=8
export MERGEPRAG_SERVICE_HIDDEN_DIM=1024
export MERGEPRAG_SERVICE_ALPHA=0.3

# Loss weights: main_gold만 살림
export MERGEPRAG_SERVICE_DUAL_CE_WEIGHT=0.0
export MERGEPRAG_SERVICE_RANK_WEIGHT=0.0
export MERGEPRAG_SERVICE_TEACHER_KL_WEIGHT=0.0
export MERGEPRAG_SERVICE_SEPARATION_WEIGHT=0.0
export MERGEPRAG_SERVICE_DIVERSITY_WEIGHT=0.0
```

코드 수정:
1. `cross_attention`의 `num_heads`를 `model.config.num_attention_heads`에서 동적으로 받도록 수정
2. `slot_diversity_loss`가 token mode에서 사용될 때 명시적 에러 발생
3. `make_memory_hook`에 debug 로깅 추가 (§ 7-3)
4. 학습 루프에 gradient flow 검증 추가 (§ 7-4)

### Step 2: 데이터 품질 확보

1. 학습 데이터를 검토하여 짧은 정답 + passage 의존적 질문으로 재구성
2. 일부는 가상 entity (Ferganite 등) 사용해 사전지식 누설 차단
3. § 8-2의 `check_data_leakage` 실행하여 leakage_ratio < 20% 확인

### Step 3: Overfit Test (Phase 1 검증)

1. 단일 샘플 200회 반복 학습
2. § 7-3, § 7-4 로그 확인:
   - Hook ratio가 0.05 ~ 0.5 범위
   - 모든 hypernetwork 파라미터에 gradient 흐름
3. 학습 후 § 7-1 K/V Necessity Test 실행:
   - Real K/V output에 정답 포함
   - Zero K/V output과 명확히 다름

**Phase 1 통과 기준 = 위 모든 항목 통과**

### Step 4: 점진적 항 추가 (Phase 2)

Phase 1 통과 후에만 진행. 항을 하나씩 추가하며 매번 validation:

1. `DUAL_CE_WEIGHT=1.0` 추가 → val 성능 측정
2. (개선되면) `RANK_WEIGHT=1.0` 추가 → val 성능 측정
3. (개선되면) `TEACHER_KL_WEIGHT=1.0` 추가 → val 성능 측정
4. `SEPARATION_WEIGHT`, `DIVERSITY_WEIGHT`는 마지막

각 항 추가 시 검증:
- Validation의 main_ok, flip_ok 비율 상승해야 함
- 안 오르면 그 항은 weight=0으로 되돌림

### Step 5: 구조 개선 (Phase 3)

Phase 2까지 안정화된 후에만 다음 ablation 진행:
- `use_contextual=True` (도움 되는지 측정)
- `skip_scale=0.5` (도움 되는지 측정)
- 위 둘 다 도움 안 되면 baseline 유지

---

## 10. 최종 체크리스트

코딩 AI는 작업 완료 후 다음을 모두 답변할 수 있어야 함:

- [ ] § 7-1 K/V Necessity Test의 결과 (Real / Zero / Random K/V 각각의 generation 출력)
- [ ] § 7-2 Overfit Test의 final loss 값
- [ ] § 7-3 Hook의 ratio (delta_norm / hidden_norm) 평균
- [ ] § 7-4 Hypernetwork 파라미터별 gradient norm
- [ ] § 8-2 데이터 leakage_ratio
- [ ] § 4-3 어떤 loss 항을 활성화했고 각 항이 validation에 미친 효과
- [ ] § 5-4 use_contextual / skip_scale / pooling_mode 최종 설정
- [ ] § 6-4 num_heads 최종 설정 (모델 head 수와 일치 여부)

---

## 부록 A: 우선순위가 낮지만 점진 개선이 필요한 항목

다음 항목들은 Phase 1~3이 안정화된 후 다룬다:

1. **Causal tracing 재실행**: 현재 `single_layer`가 정말 최적인지 확인. 최적 layer는 모델별로 다르다 (LLaMA-3-8B는 보통 layer 15~25 부근).

2. **GQA 정렬**: hypernetwork 출력을 LM의 W_K, W_V projection을 통과시켜 attention space 정렬.

3. **Multi-passage incremental injection**: 한 번에 하나가 아닌 여러 passage 누적 주입 시 catastrophic interference 측정.

4. **Conflict-aware injection**: 모델 사전지식과 passage 정보가 충돌할 때 entropy 기반으로 K/V 주입 강도 조절 (논문화 가능한 contribution 후보).

---

## 부록 B: 즉시 적용 가능한 코드 패치

### B-1. `cross_attention` num_heads 동적 처리

```python
# service_memory.py
def cross_attention(Q, K, V, num_heads):  # num_heads 필수 인자로
    """
    Args:
        num_heads: 호출자가 model.config.num_attention_heads에서 받아 전달
    """
    squeezed = False
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
        squeezed = True
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if V.dim() == 2:
        V = V.unsqueeze(0)
    
    batch, query_len, d_model = Q.shape
    key_len = K.shape[1]
    if d_model % num_heads != 0:
        raise ValueError(
            f"d_model={d_model} must be divisible by num_heads={num_heads}. "
            f"Check model.config.num_attention_heads."
        )
    d_k = d_model // num_heads
    Qh = Q.view(batch, query_len, num_heads, d_k).transpose(1, 2)
    Kh = K.view(batch, key_len, num_heads, d_k).transpose(1, 2)
    Vh = V.view(batch, key_len, num_heads, d_k).transpose(1, 2)
    att = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(d_k)
    out = att.softmax(dim=-1) @ Vh
    out = out.transpose(1, 2).contiguous().view(batch, query_len, d_model)
    return out.squeeze(0) if squeezed else out


def make_memory_hook(K, V, num_heads, alpha=SERVICE_ALPHA):
    def hook_fn(_module, _input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        K_local = K.to(device=hidden.device, dtype=hidden.dtype)
        V_local = V.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, K_local, V_local, num_heads=num_heads)
        new_hidden = hidden + alpha * delta
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden
    return hook_fn


def forward_with_memory(model, target_layer, K, V, tok, alpha=SERVICE_ALPHA):
    num_heads = model.config.num_attention_heads
    hook = target_layer.register_forward_hook(
        make_memory_hook(K, V, num_heads=num_heads, alpha=alpha)
    )
    try:
        logits = model(**tok)["logits"]
    finally:
        hook.remove()
    return logits
```

### B-2. Loss objective 명시적 단순화

```python
# service_memory_train.py에서 Phase 1용 단순 objective 함수
def simple_objective(model, tokenizer, hypernet, target_layer, sample, device):
    """Phase 1 baseline: main_gold CE only."""
    main_mem = encode_memory(model, hypernet, tokenizer, sample["passage"], device)
    gold_tok = tokenize_qa(tokenizer, sample["question"], sample["answer"], device)
    
    main_gold_logits = forward_with_memory(
        model, target_layer, main_mem["K"], main_mem["V"], gold_tok
    )
    main_gold = compute_answer_loss(main_gold_logits, gold_tok["labels"])
    if main_gold is None:
        return None
    
    return {
        "objective": main_gold,
        "main_gold": main_gold,
        # placeholder for logging compatibility
        "main_neg": main_gold.detach(),
        "neg_gold": main_gold.detach(),
        "neg_neg": main_gold.detach(),
        "rank": torch.zeros_like(main_gold),
        "teacher_kl": torch.zeros_like(main_gold),
        "sep": torch.zeros_like(main_gold),
        "div": torch.zeros_like(main_gold),
        "k_cos": torch.zeros_like(main_gold),
        "v_cos": torch.zeros_like(main_gold),
    }


# Phase 선택용 환경변수
USE_PHASE_1_BASELINE = os.getenv("MERGEPRAG_SERVICE_PHASE_1", "0") == "1"

# 학습 루프 내에서
if USE_PHASE_1_BASELINE:
    out = simple_objective(model, tokenizer, hypernet, target_layer, sample, device)
else:
    out = hard_pair_objective(model, tokenizer, hypernet, target_layer, sample, device)
```

---

## 끝.

이 보고서는 **순차적**으로 따라야 한다. 특히 § 9의 Step 1 → 5 순서는 절대 건너뛰지 않는다. 각 Step의 성공 여부는 § 7의 검증 코드로 확인한다. Phase 1 baseline이 동작하지 않으면 Phase 2~3은 의미가 없다.
