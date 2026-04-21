from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..run_model import run_model
from .embedding import encode_passage_states, tokenize_conditioned_memory


DEFAULT_CONFLICT_PAIRS = [
	{
		"question": "What color is the apple?",
		"passage_a": "The apple is green.",
		"passage_b": "The apple is red.",
		"answer_a": "green",
		"answer_b": "red",
	},
	{
		"question": "Who submitted the report?",
		"passage_a": "The report was submitted by Mina.",
		"passage_b": "The report was submitted by Jisoo.",
		"answer_a": "Mina",
		"answer_b": "Jisoo",
	},
	{
		"question": "What day is the deadline?",
		"passage_a": "The assignment deadline is Monday.",
		"passage_b": "The assignment deadline is Friday.",
		"answer_a": "Monday",
		"answer_b": "Friday",
	},
]


def iter_jsonl(path: str):
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				yield json.loads(line)
			except json.JSONDecodeError:
				continue


def iter_conflict_pairs(path: str, max_pairs: int):
	count = 0
	for row in iter_jsonl(path):
		q = str(row.get("question", "")).strip()
		pa = str(row.get("passage_a", "")).strip()
		pb = str(row.get("passage_b", "")).strip()
		aa = str(row.get("answer_a", "")).strip()
		ab = str(row.get("answer_b", "")).strip()
		if not (q and pa and pb and aa and ab):
			continue
		yield {
			"question": q,
			"passage_a": pa,
			"passage_b": pb,
			"answer_a": aa,
			"answer_b": ab,
		}
		count += 1
		if count >= max_pairs:
			break


@torch.no_grad()
def encode_pair_embedding(model, tokenizer, question: str, passage: str, device) -> torch.Tensor:
	encoded = tokenize_conditioned_memory(
		tokenizer,
		question,
		passage,
		device,
		max_length=512,
	)
	hidden = encode_passage_states(
		model,
		encoded["input_ids"],
		attention_mask=encoded["attention_mask"],
		use_contextual=True,
	)
	focus = encoded["passage_mask"].to(dtype=hidden.dtype)
	focus_weight = encoded.get("focus_weight")
	if focus_weight is None:
		focus_weight = focus
	weights = (focus_weight.to(dtype=hidden.dtype) * focus).unsqueeze(-1)
	denom = weights.sum(dim=1).clamp_min(1.0)
	pooled = (hidden * weights).sum(dim=1) / denom
	return F.normalize(pooled, dim=-1).squeeze(0)


def build_scoring_batch(tokenizer, prompt: str, answer_text: str, device):
	answer = f" {answer_text}{tokenizer.eos_token}"
	tok_prompt = tokenizer(prompt, return_tensors="pt", truncation=True)
	tok_answer = tokenizer(answer, return_tensors="pt", add_special_tokens=False, truncation=True)
	input_ids = torch.cat((tok_prompt["input_ids"], tok_answer["input_ids"][:, :-1]), dim=-1).to(device)
	labels = torch.cat(
		(
			torch.full((1, tok_prompt["input_ids"].shape[1] - 1), -100, dtype=torch.long),
			tok_answer["input_ids"],
		),
		dim=-1,
	).to(device)
	return input_ids, labels


@torch.no_grad()
def answer_logprob(model, tokenizer, question: str, passage: str, answer: str, device) -> float:
	prompt = (
		"Answer using the passage-grounded fact only.\n"
		f"Passage: {passage}\n"
		f"Question: {question}\n"
		"Answer:"
	)
	input_ids, labels = build_scoring_batch(tokenizer, prompt, answer, device)
	logits = model(input_ids=input_ids)["logits"]

	shift_logits = logits[:, :-1, :].contiguous()
	shift_labels = labels[:, 1:].contiguous()
	valid = shift_labels != -100
	if valid.sum() == 0:
		return float("-inf")

	selected_logits = shift_logits[valid]
	selected_labels = shift_labels[valid]
	log_probs = torch.log_softmax(selected_logits, dim=-1)
	return float(log_probs.gather(-1, selected_labels.unsqueeze(-1)).sum().item())


def summarize(per_pair: list[dict]) -> dict:
	if not per_pair:
		return {"count": 0}

	cos = np.asarray([x["embedding_cosine"] for x in per_pair], dtype=np.float64)
	margin_a = np.asarray([x["margin_a_over_b_when_passage_a"] for x in per_pair], dtype=np.float64)
	margin_b = np.asarray([x["margin_b_over_a_when_passage_b"] for x in per_pair], dtype=np.float64)
	flip_ok = np.asarray([1.0 if x["flip_correct"] else 0.0 for x in per_pair], dtype=np.float64)
	same_bias = np.asarray([1.0 if x["same_answer_bias"] else 0.0 for x in per_pair], dtype=np.float64)
	min_margin = np.minimum(margin_a, margin_b)

	hard_high_sim = cos >= 0.99
	hard_high_sim_count = int(hard_high_sim.sum())
	hard_high_sim_flip_acc = float(flip_ok[hard_high_sim].mean()) if hard_high_sim_count > 0 else None

	corr = None
	if len(cos) >= 2 and float(np.std(cos)) > 1e-12 and float(np.std(min_margin)) > 1e-12:
		corr = float(np.corrcoef(cos, min_margin)[0, 1])

	return {
		"count": int(len(per_pair)),
		"embedding_cosine_mean": float(cos.mean()),
		"embedding_cosine_p90": float(np.quantile(cos, 0.90)),
		"embedding_cosine_p95": float(np.quantile(cos, 0.95)),
		"mean_margin_a_over_b_when_passage_a": float(margin_a.mean()),
		"mean_margin_b_over_a_when_passage_b": float(margin_b.mean()),
		"mean_min_margin": float(min_margin.mean()),
		"flip_accuracy": float(flip_ok.mean()),
		"same_answer_bias_rate": float(same_bias.mean()),
		"high_sim_0_99_count": hard_high_sim_count,
		"high_sim_0_99_flip_accuracy": hard_high_sim_flip_acc,
		"corr_embedding_cosine_vs_min_margin": corr,
	}


def parse_args():
	parser = argparse.ArgumentParser(description="Conflict-pair factual distinguishability diagnostics.")
	parser.add_argument("--input", type=str, default="", help="Optional JSONL path with conflict pairs")
	parser.add_argument("--max-pairs", type=int, default=200)
	parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "diagnostics"))
	return parser.parse_args()


def main():
	args = parse_args()
	if args.input:
		pairs = list(iter_conflict_pairs(args.input, args.max_pairs))
		input_source = args.input
	else:
		pairs = DEFAULT_CONFLICT_PAIRS[: args.max_pairs]
		input_source = "<internal_default_pairs>"

	if len(pairs) < 1:
		raise ValueError(f"No usable conflict pairs found: {input_source}")

	print(f"[diag] usable conflict pairs: {len(pairs)}")
	model, tokenizer = run_model()
	device = next(model.parameters()).device

	per_pair = []
	for idx, row in enumerate(pairs, start=1):
		q = row["question"]
		pa = row["passage_a"]
		pb = row["passage_b"]
		aa = row["answer_a"]
		ab = row["answer_b"]

		ea = encode_pair_embedding(model, tokenizer, q, pa, device)
		eb = encode_pair_embedding(model, tokenizer, q, pb, device)
		emb_cos = float(F.cosine_similarity(ea.view(1, -1), eb.view(1, -1)).item())

		lp_aa = answer_logprob(model, tokenizer, q, pa, aa, device)
		lp_ab = answer_logprob(model, tokenizer, q, pa, ab, device)
		lp_ba = answer_logprob(model, tokenizer, q, pb, aa, device)
		lp_bb = answer_logprob(model, tokenizer, q, pb, ab, device)

		margin_a = lp_aa - lp_ab
		margin_b = lp_bb - lp_ba
		passage_a_prefers_a = lp_aa >= lp_ab
		passage_b_prefers_b = lp_bb >= lp_ba
		same_answer_bias = (passage_a_prefers_a and not passage_b_prefers_b) or (
			(not passage_a_prefers_a) and passage_b_prefers_b
		)

		per_pair.append(
			{
				"idx": idx,
				"question": q,
				"answer_a": aa,
				"answer_b": ab,
				"embedding_cosine": emb_cos,
				"logprob_answer_a_given_passage_a": lp_aa,
				"logprob_answer_b_given_passage_a": lp_ab,
				"logprob_answer_a_given_passage_b": lp_ba,
				"logprob_answer_b_given_passage_b": lp_bb,
				"margin_a_over_b_when_passage_a": margin_a,
				"margin_b_over_a_when_passage_b": margin_b,
				"flip_correct": bool((margin_a > 0.0) and (margin_b > 0.0)),
				"same_answer_bias": bool(same_answer_bias),
			}
		)

		if idx % 10 == 0:
			print(f"[diag] processed {idx}/{len(pairs)}")

	summary = summarize(per_pair)
	report = {
		"input": input_source,
		"max_pairs": args.max_pairs,
		"summary": summary,
		"per_pair": per_pair,
	}

	out_dir = Path(args.output_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_path = out_dir / f"conflict_diag_{ts}.json"
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=False, indent=2)

	print("\n[diag] done")
	print(f"flip_accuracy={summary.get('flip_accuracy', 0.0):.4f}")
	print(f"same_answer_bias_rate={summary.get('same_answer_bias_rate', 0.0):.4f}")
	print(f"embedding_cosine_mean={summary.get('embedding_cosine_mean', 0.0):.6f}")
	print(f"high_sim_0_99_count={summary.get('high_sim_0_99_count', 0)}")
	print(f"report={out_path}")


if __name__ == "__main__":
	main()



