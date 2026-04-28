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

