# Critical Layer 탐색 방법

## 목적
Qwen 모델의 N개 디코더 레이어 중, 외부 knowledge 주입에 가장 민감한 레이어를 찾는다.

## 측정 방식
1. QA 샘플(passage + question + answer)로 **baseline loss** 측정 (hook 없이)
2. 각 레이어에 **랜덤 K,V를 cross-attention으로 주입**하는 hook을 걸고 loss 재측정
3. `delta = baseline_loss - hooked_loss` 계산
4. delta가 큰 레이어 = 외부 정보에 민감한 레이어 = **critical layer**

## 주입 방식
```
output = hidden_states + α × CrossAttention(Q=hidden_states, K=random, V=random)
```
- α = 0.01 (논문 기반 스케일)
- K, V: [1, 16, d_model] 랜덤 벡터 (전 레이어 동일)

## 데이터
강의 전사 chunk 기반 QA 4건 (컴공, 경제, 심리 과목 혼합)

## 결과
- top-5 레이어 번호를 `critical_layers.json`에 저장
- 이 레이어에만 HyperNetwork의 K,V를 주입하여 학습 및 추론에 사용
