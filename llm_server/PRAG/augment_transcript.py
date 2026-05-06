"""LLM augmentation for transcript-style PRAG training data.

This path prioritizes quality over speed. It asks a stronger local/vLLM model
to turn compact fact bundles into realistic lecture/meeting transcript snippets
plus service-style QA supervision. Outputs are intentionally separate from the
existing multi-fact files.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import requests
import torch

from .augment import (
    append_jsonl,
    apply_chat_template_no_thinking,
    contains_text,
    extract_json_object,
    format_progress,
    load_existing_outputs,
    load_local_model,
    progress_suffix,
)
from .config import (
    TRANSCRIPT_AUGMENTED_TRAIN_PATH,
    TRANSCRIPT_AUGMENTED_VALID_PATH,
    TRANSCRIPT_SOURCE_PATH,
)
from .data import default_full_answer, iter_json_records, jsonl_snapshot, write_jsonl


DEFAULT_AUGMENT_MODEL = os.getenv("PRAG_TRANSCRIPT_AUGMENT_MODEL", "Qwen/Qwen3.5-4B")


def facts_text(source: dict, key: str) -> str:
    facts = source.get(key)
    if not isinstance(facts, list):
        return ""
    lines = []
    for idx, item in enumerate(facts, start=1):
        if not isinstance(item, dict):
            continue
        fact_key = str(item.get("key") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if fact_key and answer:
            lines.append(f"{idx}. {fact_key}: {answer}")
    return "\n".join(lines)


def build_messages(source: dict) -> list[dict]:
    lang = str(source.get("language") or "ko").strip()
    is_ko = lang == "ko"
    positive = facts_text(source, "facts")
    negative = facts_text(source, "negative_facts")
    subject = str(source.get("subject") or source.get("domain") or "").strip()
    scene = str(source.get("scene") or "").strip()
    speaker = str(source.get("speaker") or ("교수님" if is_ko else "the instructor")).strip()
    pattern = str(source.get("speech_pattern") or "").strip()

    if is_ko:
        task = f"""
너는 PRAG K/V 주입 학습용 데이터를 만드는 데이터 증강기다.
목표는 실제 수업/회의 전사문처럼 보이는 passage를 만들고, 그 passage만 근거로 답할 수 있는 QA를 만드는 것이다.

[상황]
- 언어: 한국어
- 장면: {scene}
- 주제: {subject}
- 발화자: {speaker}
- 발화 유형: {pattern}

[반드시 포함해야 하는 실제 정보]
{positive}

[hard negative에서만 사용할 반대 정보]
{negative}

[규칙]
1. 새 지식이나 새로운 고유명사를 만들지 마라. 위 정보만 사용하라.
2. passage는 실제 전사문처럼 자연스럽게 써라. 예: "자", "음", "이 부분 중요해요", "다시 말하면" 같은 발화체를 적절히 포함하라.
3. passage는 너무 정리문처럼 쓰지 말고, 수업/회의에서 말하는 흐름을 가져라.
4. atomic_qas는 위 실제 정보 각각에 대응해야 한다.
5. 각 atomic answer는 해당 sub_passage 안에 글자 그대로 포함되어야 한다.
6. full_answer는 사용자가 보는 AI 선생님 답변처럼 자연스럽게 쓰되, passage 밖 정보를 넣지 마라.
7. final_qas는 전체 passage의 핵심을 묻는 질문 1개를 만들고, answer에는 핵심 정보를 세미콜론으로 요약하라.
8. hard_negatives도 같은 질문을 사용하되, 답과 passage는 반대 정보로만 바꿔라.
9. JSON 이외의 설명, markdown, 코드블록은 절대 쓰지 마라.

[출력 JSON 스키마]
{{
  "passage": "...실제 전사문 스타일 passage...",
  "rewrite": "...짧은 정리문...",
  "atomic_qas": [
    {{"sub_passage": "...answer가 정확히 포함된 passage 일부...", "question": "...", "answer": "...", "full_answer": "..."}}
  ],
  "final_qas": [
    {{"question": "...", "answer": "항목1: 값1; 항목2: 값2", "full_answer": "..."}}
  ],
  "hard_negatives": [
    {{
      "passage": "...반대 정보로 만든 전사문...",
      "atomic_qas": [
        {{"sub_passage": "...negative answer가 정확히 포함된 passage 일부...", "question": "positive와 동일 질문", "answer": "...", "full_answer": "..."}}
      ],
      "final_qas": [
        {{"question": "positive와 동일 질문", "answer": "...", "full_answer": "..."}}
      ]
    }}
  ]
}}
""".strip()
    else:
        task = f"""
You generate PRAG K/V injection training data.
Create a realistic lecture/meeting transcript passage and QA supervision grounded only in the provided facts.

[Context]
- Language: English
- Scene: {scene}
- Subject: {subject}
- Speaker: {speaker}
- Speech pattern: {pattern}

[Positive facts that must appear]
{positive}

[Counterfactual facts for hard negative only]
{negative}

[Rules]
1. Do not invent new facts or names.
2. Write passage in a realistic spoken transcript style, with light filler such as "okay", "so", or "let me put it this way".
3. Do not make passage look like a clean table.
4. Create one atomic QA per positive fact.
5. Each atomic answer must appear verbatim inside its sub_passage.
6. full_answer must be a natural service answer grounded only in the passage.
7. Create one final QA summarizing all key facts.
8. hard_negatives must use the same questions but counterfactual answers/passages.
9. Output raw JSON only. No markdown, no code fences.

[JSON schema]
{{
  "passage": "...realistic transcript-style passage...",
  "rewrite": "...short clean summary...",
  "atomic_qas": [
    {{"sub_passage": "...span containing answer verbatim...", "question": "...", "answer": "...", "full_answer": "..."}}
  ],
  "final_qas": [
    {{"question": "...", "answer": "key1: value1; key2: value2", "full_answer": "..."}}
  ],
  "hard_negatives": [
    {{
      "passage": "...counterfactual transcript...",
      "atomic_qas": [
        {{"sub_passage": "...span containing negative answer verbatim...", "question": "same as positive", "answer": "...", "full_answer": "..."}}
      ],
      "final_qas": [
        {{"question": "same as positive", "answer": "...", "full_answer": "..."}}
      ]
    }}
  ]
}}
""".strip()

    return [
        {
            "role": "system",
            "content": "You generate strict raw JSON only. Do not write analysis, markdown, or code fences.",
        },
        {"role": "user", "content": task},
    ]


@torch.no_grad()
def generate_json_transformers(model, tokenizer, source: dict, max_new_tokens: int) -> tuple[dict | None, str]:
    prompt = apply_chat_template_no_thinking(tokenizer, build_messages(source))
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    return extract_json_object(text), text


def generate_json_vllm(
    source: dict,
    model_name: str,
    url: str,
    max_new_tokens: int,
    timeout: float,
    json_mode: bool,
) -> tuple[dict | None, str]:
    payload = {
        "model": model_name,
        "messages": build_messages(source),
        "temperature": 0,
        "max_tokens": max_new_tokens,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    try:
        message = data["choices"][0]["message"]
        text = message.get("content") or message.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError):
        text = json.dumps(data, ensure_ascii=False)
    return extract_json_object(text), text


def answer_span(answer: str, passage: str) -> str:
    answer = str(answer or "").strip()
    passage = str(passage or "").strip()
    if not answer or not passage or not contains_text(answer, passage):
        return ""
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?。！？요다죠])\s+|(?<=[.!?。！？])", passage)
        if item.strip()
    ]
    for sentence in sentences:
        if contains_text(answer, sentence):
            return sentence
    idx = passage.casefold().find(answer.casefold())
    if idx < 0:
        return passage
    start = max(0, idx - 80)
    end = min(len(passage), idx + len(answer) + 80)
    return passage[start:end].strip()


def source_fact_answers(source: dict, key: str) -> list[str]:
    values = source.get(key)
    if not isinstance(values, list):
        return []
    answers = []
    for item in values:
        if isinstance(item, dict):
            answer = str(item.get("answer") or "").strip()
            if answer:
                answers.append(answer)
    return answers


def source_fact_items(source: dict, key: str) -> list[dict]:
    values = source.get(key)
    if not isinstance(values, list):
        return []
    out = []
    for item in values:
        if not isinstance(item, dict):
            continue
        fact_key = str(item.get("key") or "").strip()
        answer = str(item.get("answer") or "").strip()
        if fact_key and answer:
            out.append({"key": fact_key, "answer": answer})
    return out


def source_language(source: dict) -> str:
    return str(source.get("language") or "ko").strip()


def source_question(source: dict, fact_key: str) -> str:
    subject = str(source.get("subject") or source.get("domain") or "").strip()
    speaker = str(source.get("speaker") or "").strip()
    if source_language(source) == "ko":
        prefix = f"{speaker}가 " if speaker else ""
        scope = f"{subject}에서 " if subject else ""
        return f"{prefix}{scope}말한 {fact_key}은 무엇인가요?"
    prefix = f"According to {speaker}, " if speaker else "According to the speaker, "
    scope = f"in {subject}, " if subject else ""
    return f"{prefix}{scope}what is {fact_key}?"


def source_full_answer(source: dict, fact_key: str, answer: str) -> str:
    subject = str(source.get("subject") or source.get("domain") or "").strip()
    if source_language(source) == "ko":
        scope = f"{subject}에서 " if subject else ""
        return f"{scope}{fact_key}은 {answer}입니다."
    scope = f"In {subject}, " if subject else ""
    return f"{scope}{fact_key} is {answer}."


def build_atomic_from_source(source: dict, passage: str, key: str) -> list[dict] | None:
    items = source_fact_items(source, key)
    if not items:
        return None
    qas = []
    for item in items:
        sub_passage = answer_span(item["answer"], passage)
        if not sub_passage:
            return None
        question = source_question(source, item["key"])
        qas.append({
            "sub_passage": sub_passage,
            "question": question,
            "answer": item["answer"],
            "full_answer": source_full_answer(source, item["key"], item["answer"]),
        })
    return qas


def build_final_from_source(source: dict, key: str, *, question: str | None = None) -> list[dict] | None:
    items = source_fact_items(source, key)
    if not items:
        return None
    subject = str(source.get("subject") or source.get("domain") or "").strip()
    answer = "; ".join(f"{item['key']}: {item['answer']}" for item in items)
    if question is None:
        if source_language(source) == "ko":
            scope = f"{subject}에서 " if subject else ""
            question = f"{scope}언급된 핵심 내용은 무엇인가요?"
            full_answer = f"{scope}핵심 내용은 {answer}입니다."
        else:
            scope = f"in {subject} " if subject else ""
            question = f"What are the key points mentioned {scope}?".replace("  ", " ").strip()
            full_answer = f"The key points are {answer}."
    else:
        full_answer = default_full_answer(question, answer)
    return [{"question": question, "answer": answer, "full_answer": full_answer}]


def counterfactual_passage_from_source(source: dict, passage: str) -> str:
    negative = str(source.get("hard_negatives", [{}])[0].get("passage") if isinstance(source.get("hard_negatives"), list) and source.get("hard_negatives") else "").strip()
    rewritten = str(passage or "")
    for pos, neg in zip(source_fact_items(source, "facts"), source_fact_items(source, "negative_facts")):
        if pos["answer"] and neg["answer"] and pos["answer"] in rewritten:
            rewritten = rewritten.replace(pos["answer"], neg["answer"], 1)
    if rewritten != passage:
        return rewritten
    return negative


def normalize_qas(value, *, context_passage: str = "", require_sub_passage: bool = False) -> list[dict] | None:
    if not isinstance(value, list) or not value:
        return None
    out = []
    for item in value:
        if not isinstance(item, dict):
            return None
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        sub_passage = str(item.get("sub_passage") or "").strip()
        full_answer = str(item.get("full_answer") or "").strip()
        if not (question and answer):
            return None
        if not sub_passage or not contains_text(answer, sub_passage):
            sub_passage = answer_span(answer, context_passage)
        if require_sub_passage and not sub_passage:
            return None
        out.append({
            "sub_passage": sub_passage,
            "question": question,
            "answer": answer,
            "full_answer": full_answer or default_full_answer(question, answer),
        })
    return out


def normalize_generated(raw: dict, source: dict, source_id: str) -> dict | None:
    passage = str(raw.get("passage") or "").strip()
    if not passage:
        return None
    atomic = normalize_qas(raw.get("atomic_qas"), context_passage=passage, require_sub_passage=True)
    final = normalize_qas(raw.get("final_qas"))
    source_atomic = build_atomic_from_source(source, passage, "facts")
    if source_atomic and (not atomic or len(atomic) != len(source_atomic)):
        atomic = source_atomic
    elif source_atomic:
        # Keep the LLM's wording when valid, but force answer/sub-passage
        # alignment to the source facts so training targets cannot drift.
        fixed = []
        for generated, source_qa in zip(atomic, source_atomic):
            generated["answer"] = source_qa["answer"]
            generated["sub_passage"] = source_qa["sub_passage"]
            generated["full_answer"] = generated.get("full_answer") or source_qa["full_answer"]
            fixed.append(generated)
        atomic = fixed
    if not final:
        final = build_final_from_source(source, "facts")
    if not atomic or not final:
        return None
    negatives = raw.get("hard_negatives")
    neg = negatives[0] if isinstance(negatives, list) and negatives and isinstance(negatives[0], dict) else {}
    neg_passage = str(neg.get("passage") or "").strip() or counterfactual_passage_from_source(source, passage)
    neg_atomic = normalize_qas(neg.get("atomic_qas"), context_passage=neg_passage, require_sub_passage=True)
    neg_final = normalize_qas(neg.get("final_qas"))
    if not (neg_passage and neg_atomic and neg_final):
        source_neg_atomic = build_atomic_from_source(source, neg_passage, "negative_facts")
        if source_neg_atomic:
            neg_atomic = source_neg_atomic
            final_question = final[0]["question"] if final else None
            neg_final = build_final_from_source(source, "negative_facts", question=final_question)
    source_neg_atomic = build_atomic_from_source(source, neg_passage, "negative_facts")
    if source_neg_atomic and (not neg_atomic or len(neg_atomic) != len(atomic)):
        neg_atomic = source_neg_atomic
    elif source_neg_atomic:
        fixed = []
        for generated, source_qa, positive_qa in zip(neg_atomic, source_neg_atomic, atomic):
            generated["question"] = positive_qa["question"]
            generated["answer"] = source_qa["answer"]
            generated["sub_passage"] = source_qa["sub_passage"]
            generated["full_answer"] = generated.get("full_answer") or source_qa["full_answer"]
            fixed.append(generated)
        neg_atomic = fixed
    if not neg_final and final:
        neg_final = build_final_from_source(source, "negative_facts", question=final[0]["question"])
    if not (neg_passage and neg_atomic and neg_final):
        return None
    # Some LLM outputs omit one negative QA or paraphrase it too aggressively.
    # If the counterfactual passage contains the source negative answers, rebuild
    # negative atomics by aligning source facts to the positive questions.
    if len(neg_atomic) != len(atomic):
        rebuilt = []
        for question, answer in zip((qa["question"] for qa in atomic), source_fact_answers(source, "negative_facts")):
            sub_passage = answer_span(answer, neg_passage)
            if not sub_passage:
                break
            rebuilt.append({
                "sub_passage": sub_passage,
                "question": question,
                "answer": answer,
                "full_answer": default_full_answer(question, answer),
            })
        if len(rebuilt) == len(atomic):
            neg_atomic = rebuilt
    if len(neg_atomic) != len(atomic) or len(neg_final) != len(final):
        return None
    for idx, qa in enumerate(neg_atomic):
        qa["question"] = atomic[idx]["question"]
    for idx, qa in enumerate(neg_final):
        qa["question"] = final[idx]["question"]
    return {
        "source_id": source_id,
        "speaker": source.get("speaker", ""),
        "scene": source.get("scene", ""),
        "domain": source.get("domain", ""),
        "language": source.get("language", ""),
        "speech_pattern": source.get("speech_pattern", ""),
        "passage": passage,
        "rewrite": str(raw.get("rewrite") or "").strip(),
        "atomic_qas": atomic,
        "final_qas": final,
        "hard_negatives": [{
            "passage": neg_passage,
            "atomic_qas": neg_atomic,
            "final_qas": neg_final,
        }],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(TRANSCRIPT_SOURCE_PATH))
    parser.add_argument("--train-output", default=str(TRANSCRIPT_AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid-output", default=str(TRANSCRIPT_AUGMENTED_VALID_PATH))
    parser.add_argument("--model", default=DEFAULT_AUGMENT_MODEL)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--vllm-url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--vllm-timeout", type=float, default=300.0)
    parser.add_argument("--vllm-json-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--valid-every", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--debug-invalid-raw", action="store_true")
    parser.add_argument("--debug-raw-chars", type=int, default=1600)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run(args: argparse.Namespace) -> None:
    if args.resume:
        seen_ids, train_count, valid_count = load_existing_outputs(args.train_output, args.valid_output)
        existing_count = len(seen_ids)
        if seen_ids:
            print(
                f"[PRAG:transcript-augment] resume enabled: existing train={train_count} "
                f"valid={valid_count} seen={len(seen_ids)}"
            )
    else:
        seen_ids, train_count, valid_count = set(), 0, 0
        existing_count = 0
        write_jsonl(args.train_output, [])
        write_jsonl(args.valid_output, [])
        print("[PRAG:transcript-augment] resume disabled: output files were reset.")

    input_rows = list(iter_json_records(args.input))
    if args.shuffle:
        random.Random(args.seed).shuffle(input_rows)
    if args.max_samples:
        input_rows = input_rows[: args.max_samples]
    input_total = len(input_rows)
    input_ids = {
        str(row.get("source_id") or row.get("id") or f"raw_{idx}").strip()
        for idx, row in enumerate(input_rows)
    }
    existing_in_input = len(seen_ids & input_ids)
    print(f"[PRAG:transcript-augment] input={args.input} total_target={input_total}")
    print(f"[PRAG:transcript-augment] shuffle={args.shuffle} seed={args.seed}")
    print(
        f"[PRAG:transcript-augment] cumulative_start existing={existing_in_input}/{input_total} "
        f"({existing_in_input / input_total * 100:.1f}%) "
        f"all_existing_outputs={existing_count} train={train_count} valid={valid_count}"
        if input_total
        else f"[PRAG:transcript-augment] cumulative_start existing=0/0 all_existing_outputs={existing_count} train={train_count} valid={valid_count}"
    )

    model = tokenizer = None
    if args.backend == "transformers":
        model, tokenizer = load_local_model(args.model)
    else:
        print(f"[PRAG:transcript-augment] backend=vllm url={args.vllm_url} model={args.model}")

    made = skipped_seen = skipped_invalid = 0
    started_at = time.time()
    for idx, source in enumerate(input_rows):
        source_id = str(source.get("source_id") or source.get("id") or f"raw_{idx}")
        if source_id in seen_ids:
            skipped_seen += 1
            if skipped_seen % 50 == 0:
                processed = idx + 1
                done_total = min(existing_in_input + made, input_total)
                print(
                    f"[PRAG:transcript-augment] resume skip scan={format_progress(processed, input_total)} "
                    f"done_total={format_progress(done_total, input_total)} "
                    f"existing_start={existing_in_input}, new={made}, skipped_seen={skipped_seen}, "
                    f"train={train_count}, valid={valid_count}"
                )
            continue
        try:
            if args.backend == "vllm":
                generated, raw_text = generate_json_vllm(
                    source,
                    model_name=args.model,
                    url=args.vllm_url,
                    max_new_tokens=args.max_new_tokens,
                    timeout=args.vllm_timeout,
                    json_mode=args.vllm_json_mode,
                )
            else:
                generated, raw_text = generate_json_transformers(
                    model,
                    tokenizer,
                    source,
                    max_new_tokens=args.max_new_tokens,
                )
        except requests.RequestException as exc:
            skipped_invalid += 1
            processed = idx + 1
            print(
                f"[PRAG:transcript-augment] skip {source_id}: vLLM request failed: {exc} "
                f"({progress_suffix(processed, input_total, existing_in_input, made, skipped_seen, skipped_invalid, train_count, valid_count, started_at)})"
            )
            continue
        if generated is None:
            skipped_invalid += 1
            processed = idx + 1
            print(
                f"[PRAG:transcript-augment] skip {source_id}: invalid JSON "
                f"({progress_suffix(processed, input_total, existing_in_input, made, skipped_seen, skipped_invalid, train_count, valid_count, started_at)})"
            )
            if args.debug_invalid_raw:
                preview = " ".join(str(raw_text or "").split())
                print(f"[PRAG:transcript-augment:raw] {preview[:args.debug_raw_chars]}")
            continue
        row = normalize_generated(generated, source, source_id)
        if row is None:
            skipped_invalid += 1
            processed = idx + 1
            print(
                f"[PRAG:transcript-augment] skip {source_id}: missing required fields "
                f"({progress_suffix(processed, input_total, existing_in_input, made, skipped_seen, skipped_invalid, train_count, valid_count, started_at)})"
            )
            if args.debug_invalid_raw:
                raw_preview = " ".join(str(raw_text or "").split())
                parsed_preview = json.dumps(generated, ensure_ascii=False)[:args.debug_raw_chars] if generated is not None else "none"
                print(f"[PRAG:transcript-augment:parsed] {parsed_preview}")
                print(f"[PRAG:transcript-augment:raw] {raw_preview[:args.debug_raw_chars]}")
            continue

        is_valid = args.valid_every > 0 and idx % args.valid_every == args.valid_every - 1
        saved_to = args.valid_output if is_valid else args.train_output
        append_jsonl(saved_to, row)
        seen_ids.add(source_id)
        made += 1
        if is_valid:
            valid_count += 1
        else:
            train_count += 1
        processed = idx + 1
        print(
            f"[PRAG:transcript-augment] ok {source_id}: atomic={len(row['atomic_qas'])} "
            f"final={len(row['final_qas'])} -> {'valid' if is_valid else 'train'} "
            f"saved_to={saved_to} "
            f"({progress_suffix(processed, input_total, existing_in_input, made, skipped_seen, skipped_invalid, train_count, valid_count, started_at)})"
        )

    print(
        f"[PRAG:transcript-augment] done new={made} skipped_seen={skipped_seen} "
        f"skipped_invalid={skipped_invalid}"
    )
    print(f"[PRAG:transcript-augment] train={train_count} -> {args.train_output}")
    print(f"[PRAG:transcript-augment] valid={valid_count} -> {args.valid_output}")
    print(f"[PRAG:transcript-augment] train_snapshot={jsonl_snapshot(args.train_output)}")
    print(f"[PRAG:transcript-augment] valid_snapshot={jsonl_snapshot(args.valid_output)}")


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
