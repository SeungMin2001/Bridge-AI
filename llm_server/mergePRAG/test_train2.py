"""
Evaluate train2 paper-style HyperKV weights.

This script loads the train2 HyperKVGeneratorFixed-compatible weights and
measures whether memory injection actually selects the passage-grounded answer
over the hard-negative answer.

Run:
    python -m llm_server.mergePRAG.test_train2

Useful options:
    python -m llm_server.mergePRAG.test_train2 --split valid --max-samples 240
    python -m llm_server.mergePRAG.test_train2 --split train --max-samples 20 --show-generations 5
    python -m llm_server.mergePRAG.test_train2 --weights checkpoint
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_NAME, TRAIN_DATA_PATH, VALID_DATA_PATH, load_critical_layer
from .train2 import (
    CHECKPOINT_PATH,
    HIDDEN_DIM,
    NUM_KV,
    WEIGHTS_PATH,
    MergePRAGDataset,
    PaperHyperNetwork,
    build_direct_prompt,
    build_paper_prompt,
    compute_loss,
    encode_memory,
    extract_first_hard_negative,
    generate_plain,
    generate_with_memory,
    score_answer,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate train2 HyperKV weights.")
    parser.add_argument("--split", choices=["valid", "train"], default="valid")
    parser.add_argument("--max-samples", type=int, default=240)
    parser.add_argument("--weights", choices=["weights", "checkpoint", "latest"], default="weights")
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--show-examples", type=int, default=10)
    parser.add_argument("--show-generations", type=int, default=0)
    return parser.parse_args()


def resolve_weights_path(kind: str, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Explicit weights path not found: {path}")
        return path

    weights_path = Path(WEIGHTS_PATH)
    checkpoint_path = Path(CHECKPOINT_PATH)
    if kind == "weights":
        if weights_path.exists():
            return weights_path
        if checkpoint_path.exists():
            print(f"[test_train2] weights missing; falling back to checkpoint: {checkpoint_path}")
            return checkpoint_path
    if kind == "checkpoint":
        if checkpoint_path.exists():
            return checkpoint_path
        if weights_path.exists():
            print(f"[test_train2] checkpoint missing; falling back to weights: {weights_path}")
            return weights_path
    if kind == "latest":
        candidates = [p for p in (weights_path, checkpoint_path) if p.exists()]
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(
        "No train2 weights found. Expected one of:\n"
        f"  {weights_path}\n"
        f"  {checkpoint_path}"
    )


def load_hypernet_state(path: Path, map_location):
    state = torch.load(path, map_location=map_location)
    if isinstance(state, dict) and "hypernet" in state:
        return state["hypernet"], state
    return state, {"step": None, "val_loss": None, "config": None}


def fmt(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def cosine_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def tokenize_direct_qa(tokenizer, question: str, passage: str, answer: str, device, model_name: str):
    prompt = build_direct_prompt(question, passage)
    if "llama" in model_name.lower():
        answer_text = f" {answer}{tokenizer.eos_token}"
    else:
        answer_text = f"{answer}{tokenizer.eos_token}"

    tok_prompt = tokenizer(prompt, return_tensors="pt")
    tok_answer = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)
    input_ids = torch.cat((tok_prompt["input_ids"], tok_answer["input_ids"][:, :-1]), dim=-1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = torch.cat(
        (
            torch.full(tok_prompt["input_ids"].shape, -100, dtype=torch.long)[:, 1:],
            tok_answer["input_ids"],
        ),
        dim=-1,
    ).to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def score_direct_answer(model, tokenizer, question: str, passage: str, answer: str, device):
    tok = tokenize_direct_qa(tokenizer, question, passage, answer, device, MODEL_NAME)
    logits = model(**tok)["logits"]
    loss = compute_loss(logits, tok["labels"])
    return float("nan") if loss is None else loss.item()


def main():
    args = parse_args()
    dataset_path = VALID_DATA_PATH if args.split == "valid" else TRAIN_DATA_PATH
    weights_path = resolve_weights_path(args.weights, args.weights_path)

    print(f"[test_train2] model={MODEL_NAME}")
    print(f"[test_train2] split={args.split}, data={dataset_path}")
    print(f"[test_train2] weights={weights_path}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    device = next(model.parameters()).device
    d_model = model.config.hidden_size
    layer_idx = load_critical_layer()
    target_layer = model.model.layers[layer_idx]

    hypernet = PaperHyperNetwork(d_model, num_kv=NUM_KV, hidden_dim=HIDDEN_DIM).to(device).float()
    state_dict, meta = load_hypernet_state(weights_path, map_location=device)
    hypernet.load_state_dict(state_dict)
    hypernet.eval()
    print(
        f"[test_train2] loaded step={meta.get('step')} "
        f"val_loss={meta.get('val_loss')} layer={layer_idx} num_kv={NUM_KV}"
    )

    dataset = MergePRAGDataset(dataset_path, max_samples=args.max_samples)
    total = 0
    with_negative = 0
    memory_gain_sum = 0.0
    direct_ok = 0
    main_ok = 0
    negative_ok = 0
    flip_ok = 0
    base_pref_gold = 0
    k_cos_sum = 0.0
    v_cos_sum = 0.0
    pooled_cos_sum = 0.0
    hidden_cos_sum = 0.0
    shown = 0

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            question = sample["question"]
            passage = sample["passage"]
            gold = sample["answer"]
            negative = extract_first_hard_negative(sample)
            if not negative or not negative.get("answer") or not negative.get("passage"):
                continue

            neg_answer = negative["answer"]
            neg_passage = negative["passage"]

            base_gold = score_answer(model, tokenizer, hypernet, target_layer, question, None, gold, device)
            base_neg = score_answer(model, tokenizer, hypernet, target_layer, question, None, neg_answer, device)
            direct_gold = score_direct_answer(model, tokenizer, question, passage, gold, device)
            direct_neg = score_direct_answer(model, tokenizer, question, passage, neg_answer, device)
            main_gold = score_answer(model, tokenizer, hypernet, target_layer, question, passage, gold, device)
            main_neg = score_answer(model, tokenizer, hypernet, target_layer, question, passage, neg_answer, device)
            negmem_gold = score_answer(model, tokenizer, hypernet, target_layer, question, neg_passage, gold, device)
            negmem_neg = score_answer(model, tokenizer, hypernet, target_layer, question, neg_passage, neg_answer, device)
            _, pooled_main, hidden_main, k_main, v_main = encode_memory(
                model,
                hypernet,
                tokenizer,
                passage,
                device,
            )
            _, pooled_neg, hidden_neg, k_neg, v_neg = encode_memory(
                model,
                hypernet,
                tokenizer,
                neg_passage,
                device,
            )

            total += 1
            with_negative += 1
            memory_gain_sum += base_gold - main_gold
            base_pref = base_gold < base_neg
            direct_pref = direct_gold < direct_neg
            main_pref = main_gold < main_neg
            neg_pref = negmem_neg < negmem_gold
            both_pref = main_pref and neg_pref
            direct_ok += int(direct_pref)
            base_pref_gold += int(base_pref)
            main_ok += int(main_pref)
            negative_ok += int(neg_pref)
            flip_ok += int(both_pref)
            pooled_cos = cosine_flat(pooled_main, pooled_neg)
            hidden_cos = cosine_flat(hidden_main, hidden_neg)
            k_cos = cosine_flat(k_main, k_neg)
            v_cos = cosine_flat(v_main, v_neg)
            pooled_cos_sum += pooled_cos
            hidden_cos_sum += hidden_cos
            k_cos_sum += k_cos
            v_cos_sum += v_cos

            if shown < args.show_examples:
                print(f"\n[{idx}] q={question}")
                print(f"  gold={gold} | neg={neg_answer}")
                print(f"  no_hook      gold={fmt(base_gold)} neg={fmt(base_neg)} prefer_gold={base_pref}")
                print(f"  direct       gold={fmt(direct_gold)} neg={fmt(direct_neg)} prefer_gold={direct_pref}")
                print(f"  main_memory  gold={fmt(main_gold)} neg={fmt(main_neg)} prefer_gold={main_pref}")
                print(f"  neg_memory   gold={fmt(negmem_gold)} neg={fmt(negmem_neg)} prefer_neg={neg_pref}")
                print(f"  memory_cos   pooled={pooled_cos:.4f} hidden={hidden_cos:.4f} K={k_cos:.4f} V={v_cos:.4f}")
                print(f"  gain(no_hook_gold-main_gold)={fmt(base_gold - main_gold)} flip_ok={both_pref}")
                if shown < args.show_generations:
                    no_hook = generate_plain(model, tokenizer, build_paper_prompt(question), device)
                    direct = generate_plain(model, tokenizer, build_direct_prompt(question, passage), device)
                    hooked = generate_with_memory(model, tokenizer, hypernet, target_layer, question, passage, device)
                    print(f"  gen no_hook={no_hook}")
                    print(f"  gen direct ={direct}")
                    print(f"  gen memory ={hooked}")
                shown += 1

            if torch.cuda.is_available() and total % 50 == 0:
                torch.cuda.empty_cache()

    denom = max(with_negative, 1)
    print("\n[test_train2:summary]")
    print(f"  evaluated hard-pair rows: {with_negative}/{len(dataset)}")
    print(f"  avg memory gain: {memory_gain_sum / denom:.4f}")
    print(f"  avg memory cosine: pooled={pooled_cos_sum / denom:.4f}, hidden={hidden_cos_sum / denom:.4f}, K={k_cos_sum / denom:.4f}, V={v_cos_sum / denom:.4f}")
    print(f"  no_hook gold preference: {base_pref_gold}/{denom} = {base_pref_gold / denom:.3f}")
    print(f"  direct passage gold preference: {direct_ok}/{denom} = {direct_ok / denom:.3f}")
    print(f"  main memory gold preference: {main_ok}/{denom} = {main_ok / denom:.3f}")
    print(f"  negative memory negative preference: {negative_ok}/{denom} = {negative_ok / denom:.3f}")
    print(f"  bidirectional flip success: {flip_ok}/{denom} = {flip_ok / denom:.3f}")


if __name__ == "__main__":
    main()
