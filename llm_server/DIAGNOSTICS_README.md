# MergePRAG Diagnostics

`test_mergeprag.py` now runs a quantitative diagnostic pipeline in one shot:

- K/V separation metrics: cosine, L2, MAE
- Hook perturbation strength: `delta_ratio = ||delta|| / ||hidden||`
- Output distribution shift: KL/JS divergence (A vs B, A/B vs base)
- Greedy generation comparison for both passages over alpha sweep
- JSON artifact save for regression tracking

## Quick run

```powershell
Set-Location "C:\Group-Chat-agent\llm_server"
python test_mergeprag.py
```

`.pt`가 없는 재학습 구간이면 `test_mergeprag.py`가 자동으로 우회(임베딩 프록시) 모드로 실행됩니다.

## Output

A JSON report is saved under:

- `llm_server/diagnostics/mergeprag_diag_YYYYMMDD_HHMMSS.json`

## How to read the report

- High `K/V cosine` + very low `MAE` => passage separation collapse likely.
- Very small `JS(A,B)` across alpha sweep => different KV injection does not change next-token distribution enough.
- Large `delta_ratio` (e.g., >0.25) => over-strong perturbation risk (instability).
- `answer_diff=false` for all alpha => semantic branch is not reaching decoding.

## Suggested next steps

1. Use this script as a fast gate before 30-hour E2E runs.
2. Keep only runs where `JS(A,B)` increases while `delta_ratio` remains moderate.
3. Promote top candidates to expensive E2E validation.

## Conflict-pair diagnostics (no .pt required)

```powershell
Set-Location "C:\Group-Chat-agent"
python -m llm_server.mergePRAG.find_passage_sim_elbow --max-pairs 200
```

기본적으로 스크립트 내부의 예시 conflict pair를 사용합니다.

커스텀 데이터(JSONL)를 쓰고 싶다면 `--input`을 추가하세요.

```powershell
python -m llm_server.mergePRAG.find_passage_sim_elbow --input C:\path\conflict_pairs.jsonl --max-pairs 200
```

`--input` JSONL 형식:

```json
{"question":"What color is the apple?","passage_a":"The apple is green.","passage_b":"The apple is red.","answer_a":"green","answer_b":"red"}
```

출력 JSON의 핵심 지표:

- `flip_accuracy`: passage A/B를 바꿨을 때 정답 선호가 뒤집히는 비율
- `same_answer_bias_rate`: passage가 바뀌어도 같은 답만 선호하는 실패 비율
- `embedding_cosine_mean`: 질문 조건 하 passage 임베딩 평균 코사인
- `high_sim_0_99_flip_accuracy`: 유사도 0.99 이상 고난도 샘플의 분기 성능

해당 지표를 먼저 개선한 뒤, 필요하면 운영 threshold를 적용하세요.

```powershell
$env:MERGEPRAG_PASSAGE_SIM_GATE="0.985"
python -m uvicorn llm_server.api:app --host 0.0.0.0 --port 8001
```




