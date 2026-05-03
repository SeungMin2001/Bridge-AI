# PRAG Service Memory Restart

This folder is the clean restart point for our service-specific implementation.
Keep `llm_server/mergePRAG` as historical/debugging reference, but do not import
legacy training code from it unless a file is deliberately copied and simplified.

## Goal

The product goal is single-hop lecture memory:

1. A professor gives one short passage/fact.
2. The system encodes that passage into learned memory.
3. Later, the user asks a question without the passage in the prompt.
4. The model answers from the injected memory.

This is not a multi-hop paper reproduction. PRAG/MergePRAG papers are references
for mechanisms, but the implementation should prioritize reliable single-hop
service grounding.

## Design Rules

- Use a strict single-hop data contract: `question`, `passage`, `answer`, and
  one counterfactual hard negative.
- Avoid hidden multi-hop fallbacks such as `hop_passages`, HotPot context
  expansion, or decomposed sub-question pipelines in the core service path.
- Keep phase-based validation: overfit first, then dataset validation, then
  hard-negative flip.
- Verify K/V necessity before trusting any loss metric: real memory should
  answer correctly, zero/random memory should fail.
- Keep Korean and English prompts explicit and separate enough to avoid mixed
  instruction behavior.
- Treat paper code as reference, not a target to clone exactly.

## Planned Files

- `config.py`: model, layer, paths, and phase defaults.
- `data.py`: strict single-hop hard-pair dataset loader.
- `memory.py`: passage-to-memory encoder and injection hook.
- `train.py`: minimal phase-based trainer.
- `test.py`: compact service diagnostics with generation and K/V necessity.

## Current Pipeline

The first clean implementation follows this hybrid strategy:

1. `augment.py` uses a local LLM to convert one professor passage into:
   - `rewrite`
   - `atomic_qas`
   - `final_qas`
   - `hard_negatives`
2. `memory.py` follows the local MergePRAG paper code style:
   - raw token embeddings only
   - attentive pooling
   - MLP
   - linear K/V projection
   - residual-stream cross-attention injection
3. `train.py` trains one HyperKV generator from all generated QA pairs:
   - positive passage memory should answer positive QA
   - counterfactual memory should answer counterfactual QA
   - margin rank encourages answer flipping
4. `test.py` checks:
   - main memory preference
   - negative memory preference
   - bidirectional flip
   - real K/V generation versus zero K/V generation

## Commands

Prepare raw passage data as JSONL, or reuse the current `ServiceHardPair`
dataset as seed input:

```json
{"source_id": "lecture_001", "speaker": "교수", "passage": "오늘 수업에서 ..."}
```

Generate augmented supervision:

```bash
python -m llm_server.PRAG.augment \
  --input data/ServiceHardPair_train.jsonl \
  --train-output data/PRAG_augmented_train.jsonl \
  --valid-output data/PRAG_augmented_valid.jsonl \
  --model Qwen/Qwen2.5-3B-Instruct \
  --max-samples 100
```

Validate before training:

```bash
python -m llm_server.PRAG.validate \
  data/PRAG_augmented_train.jsonl \
  data/PRAG_augmented_valid.jsonl
```

Preview samples by eye:

```bash
python -m llm_server.PRAG.preview_data \
  data/PRAG_augmented_train.jsonl \
  --samples 5 \
  --max-qas 4
```

Train HyperKV memory:

```bash
python -m llm_server.PRAG.train --multifact --no-resume
```

Default training settings are kept in `llm_server/PRAG/config.py`: `epochs=4`,
`num_kv=16`, `alpha=1.0`, `answer_target=full_answer`, and a small
free-generation validation probe every 1000 steps.

Evaluate:

```bash
python -m llm_server.PRAG.test \
  --data data/PRAG_augmented_valid.jsonl \
  --show 3
```
