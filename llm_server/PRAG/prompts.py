"""Prompts for local-LLM augmentation and service QA."""

from __future__ import annotations

from .config import contains_hangul


def augmentation_prompt(
    passage: str,
    question: str = "",
    answer: str = "",
    negative_passage: str = "",
    negative_answer: str = "",
) -> str:
    seed_hint_ko = ""
    seed_hint_en = ""
    if question or answer or negative_passage or negative_answer:
        seed_hint_ko = f"""

기존 seed 정보:
- 대표 질문: {question or "(없음)"}
- 원본 답: {answer or "(없음)"}
- 반례 답: {negative_answer or "(없음)"}
- 반례 passage: {negative_passage or "(없음)"}

반례 passage가 제공되면 hard_negatives[0].passage에는 그 반례 passage를 그대로 사용하세요.
hard_negatives[0].atomic_qas와 final_qas는 원본 atomic/final 질문과 같은 순서, 같은 질문 문자열을 사용하고,
answer만 반례 passage 기준으로 바꾸세요.
"""
        seed_hint_en = f"""

Existing seed information:
- Representative question: {question or "(none)"}
- Original answer: {answer or "(none)"}
- Counterfactual answer: {negative_answer or "(none)"}
- Counterfactual passage: {negative_passage or "(none)"}

If a counterfactual passage is provided, use it exactly as hard_negatives[0].passage.
hard_negatives[0].atomic_qas and final_qas must use the same order and exactly
the same question strings as the original atomic/final QAs; only the answers
should change according to the counterfactual passage.
"""
    if contains_hangul(passage):
        return f"""다음 교수 발화 하나를 기반으로 학습용 JSON만 생성하세요.

목표:
- passage 안의 사실을 원자적 질문/답변으로 분해합니다.
- 이 passage는 발음 기반 자동 전사라 오탈자와 잘못 인식된 단어가 있을 수 있습니다.
  의미가 명확한 수업 일정, 과제, 평가, 개념 정의, 용어 관계, 실습 지시만 사용하고,
  불확실하거나 깨진 문장은 학습 사실로 만들지 마세요.
- 회화 예문, 인사말, 반복 발화, "다시 들어볼까요" 같은 진행 멘트는 학습 사실로 만들지 마세요.
- "왜/이유" 질문을 만들지 마세요. passage에 직접 적힌 값, 정의, 규칙, 일정, 조건만 묻는 질문을 만드세요.
- passage 안에 여러 사실이 있으면 가장 명확한 사실만 최대 3개 선택해서 atomic_qas를 만듭니다.
- 명확한 학습 사실이 없으면 atomic_qas, final_qas, hard_negatives를 빈 배열로 반환하세요.
- 각 원자적 질문/답변마다 sub_passage를 포함합니다. sub_passage는 원문 근거
  조각을 복사하거나 최소한으로 재작성한 문장이어야 하며, 그 조각만 보고 답할
  수 있어야 합니다.
- 각 atomic answer 문자열은 해당 sub_passage 안에 실제로 들어 있어야 합니다.
- atomic question들은 서로 다른 사실을 물어야 합니다. 같은 사실의 표현만 바꾼 질문을 여러 개 만들지 마세요.
- final_qas는 1개만 만들고, 여러 atomic fact를 종합해서 교수님의 전체 설명을 묻는 질문으로 만듭니다.
- hard negative의 atomic/final question은 원본과 정확히 같은 문자열과 같은 순서를 사용하고, answer만 반례 기준으로 바꿉니다.
- hard negative의 sub_passage에도 바뀐 answer 문자열이 실제로 들어 있어야 합니다.
- 원본과 hard negative의 모든 QA에는 answer뿐 아니라 AI 조교가 사용자에게 말할 자연스러운 완전문장 full_answer도 포함하세요.
- full_answer는 실제 서비스 답변처럼 핵심 답을 먼저 말하고, 필요하면 짧게 설명하는 한 문장으로 작성하세요.
- passage에 없는 사실을 만들지 마세요.
- 답변은 짧고 passage에 근거해야 합니다. placeholder나 설명문을 쓰지 마세요.
- JSON 앞뒤에 설명, 마크다운, 코드블록을 절대 붙이지 마세요.

출력은 아래 JSON 객체 하나만 허용합니다.
{{
  "rewrite": "의미는 같지만 표현이 다른 재작성",
  "atomic_qas": [
    {{
      "sub_passage": "원문 근거 조각",
      "question": "짧은 하위질문",
      "answer": "sub_passage에 실제로 포함된 짧은 답",
      "full_answer": "완전한 문장 답"
    }}
  ],
  "final_qas": [
    {{"question": "교수님 설명을 묻는 최종질문", "answer": "짧은 답", "full_answer": "교수님 설명 기반 답변"}}
  ],
  "hard_negatives": [
    {{
      "passage": "핵심 관계나 값이 뒤집힌 반례 passage",
      "atomic_qas": [
        {{
          "sub_passage": "뒤집힌 반례 근거 조각",
          "question": "원본 atomic_qas와 정확히 같은 질문",
          "answer": "반례 passage 기준 답",
          "full_answer": "반례 passage 기준 완전한 문장 답"
        }}
      ],
      "final_qas": [
        {{"question": "원본 final_qas와 정확히 같은 질문", "answer": "반례 passage 기준 답", "full_answer": "반례 passage 기준 교수님 설명 기반 답변"}}
      ]
    }}
  ]
}}

교수 발화:
{passage}
{seed_hint_ko}"""
    return f"""Generate one training JSON object from the lecture passage below.

Goals:
- Decompose the passage into atomic question/answer pairs.
- This passage may be a noisy speech transcript. Use only clearly supported
  facts such as schedules, assignments, evaluation rules, concept definitions,
  term relations, and practice instructions. Ignore uncertain or corrupted ASR
  fragments.
- Do not turn dialogue practice sentences, greetings, repeated utterances, or
  class-management remarks such as "let's listen again" into training facts.
- Do not create "why/reason" questions. Ask only for values, definitions, rules,
  schedules, or conditions that are directly stated in the passage.
- If the passage contains multiple facts, select at most three of the clearest
  facts and create one atomic QA for each selected fact.
- If there are no clear trainable facts, return empty arrays for atomic_qas,
  final_qas, and hard_negatives.
- For every atomic pair, include a sub_passage copied from or minimally
  rewritten from the lecture passage. The sub_passage alone must support the
  answer.
- The atomic answer string must actually appear in its sub_passage.
- Atomic questions must ask different facts. Do not create multiple paraphrases
  of the same fact.
- Create exactly one final QA that combines multiple atomic facts and asks for the
  instructor's overall explanation.
- hard_negatives must use exactly the same question strings and order as the
  original atomic/final QAs; only the answers should change under the
  counterfactual passage.
- Every hard-negative sub_passage must actually contain its changed answer.
- Every original and hard-negative QA must include both answer and full_answer.
- full_answer must be a natural service-style assistant sentence that starts
  with the core answer and then gives a concise grounded explanation if needed.
- Do not invent facts unsupported by the passage.
- Keep answers short and grounded in the passage. Do not use placeholders.
- Return raw JSON only. Do not add explanations, markdown, or code fences.

Return only this JSON object:
{{
  "rewrite": "same meaning, different wording",
  "atomic_qas": [
    {{
      "sub_passage": "short evidence chunk from the passage",
      "question": "short sub-question",
      "answer": "short answer that appears in sub_passage",
      "full_answer": "complete sentence answer"
    }}
  ],
  "final_qas": [
    {{"question": "final question asking for the instructor's explanation", "answer": "short answer", "full_answer": "grounded answer"}}
  ],
  "hard_negatives": [
    {{
      "passage": "counterfactual passage with the key relation/value flipped",
      "atomic_qas": [
        {{
          "sub_passage": "counterfactual evidence chunk",
          "question": "exact same question as the matching original atomic QA",
          "answer": "answer under the counterfactual passage",
          "full_answer": "complete sentence answer under the counterfactual passage"
        }}
      ],
      "final_qas": [
        {{"question": "exact same question as the original final QA", "answer": "answer under the counterfactual passage", "full_answer": "grounded answer under the counterfactual passage"}}
      ]
    }}
  ]
}}

Lecture passage:
{passage}
{seed_hint_en}"""


def system_prompt(question: str) -> str:
    if contains_hangul(question):
        return (
            "당신은 수업/회의 내용을 기억해 답하는 AI 조교입니다. "
            "반드시 한국어로만 답하세요. 중국어, 영어, 일본어, 한자 혼용을 절대 쓰지 마세요. "
            "답변은 한 문장으로 끝내고, 질문이 요구한 정답 정보만 말하세요. "
            "답변은 반드시 주입된 메모리에 담긴 내용만 근거로 하세요. "
            "질문에 해당하는 정보가 있으면 사용자의 질문과 같은 한국어로 자연스럽게 답하세요. "
            "짧은 사실 질문은 첫 문장을 반드시 답이 되는 핵심 구절로 시작하세요. "
            "설명이 필요한 질문도 핵심 답을 먼저 말한 뒤 최대 한 문장으로만 설명하세요. "
            "메모리에 없는 내용은 추측하지 말고 '모름'이라고만 답하세요. "
            "중국어, 한자, 영어 번역, 근거/질의/반증/레이블 같은 메타 표현, 특수기호, "
            "불필요한 접두어를 절대 붙이지 마세요."
        )
    return (
        "You are an AI assistant that answers from injected lecture or meeting memory. "
        "Use only the injected memory as evidence. "
        "If the answer is present, reply naturally in English. "
            "For short factual questions, the first sentence must start with the answer phrase. "
            "For explanation questions, give the answer first, then explain in one or two concise sentences. "
            "If unsupported by memory, answer exactly 'Unknown'. "
            "Do not add Chinese text, translations, evidence labels, meta words, special symbols, or unrelated prefixes."
    )


def user_prompt(question: str) -> str:
    if contains_hangul(question):
        return (
            f"질문: {question}\n"
            "주입된 메모리에서 이 질문에 직접 답하는 내용만 사용해 답하세요. "
            "반드시 한국어 한 문장으로만 답하세요. 외국어, 한자, 번역문, 부가 설명을 쓰지 마세요. "
            "첫 문장은 답이 되는 핵심 구절로 시작하세요. "
            "답을 알고 있으면 바로 답하고, 모르면 '모름'이라고만 답하세요."
        )
    return (
        f"Question: {question}\n"
        "Answer using only the injected memory. If the answer is present, answer directly; "
        "if not, answer exactly 'Unknown'."
    )
