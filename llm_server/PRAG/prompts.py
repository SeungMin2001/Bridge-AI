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

반례 passage가 제공되면 hard_negatives[0].passage에는 그 반례 passage를 우선 사용하고,
원본 atomic/final 질문에 대응되는 반례 기준 답변도 함께 만드세요.
"""
        seed_hint_en = f"""

Existing seed information:
- Representative question: {question or "(none)"}
- Original answer: {answer or "(none)"}
- Counterfactual answer: {negative_answer or "(none)"}
- Counterfactual passage: {negative_passage or "(none)"}

If a counterfactual passage is provided, prefer it as hard_negatives[0].passage
and create counterfactual answers for the same atomic/final questions.
"""
    if contains_hangul(passage):
        return f"""다음 교수 발화 하나를 기반으로 학습용 JSON만 생성하세요.

목표:
- passage 안의 사실을 원자적 질문/답변으로 분해합니다.
- 각 원자적 질문/답변마다 sub_passage를 포함합니다. sub_passage는 원문 근거
  조각을 복사하거나 최소한으로 재작성한 문장이어야 하며, 그 조각만 보고 답할
  수 있어야 합니다.
- 전체 설명 질문/답변도 만듭니다.
- 같은 질문에서 답이 뒤집히는 counterfactual hard negative passage도 만듭니다.
- passage에 없는 사실을 만들지 마세요.
- 답변은 짧고 passage에 근거해야 합니다.

출력은 아래 JSON 객체 하나만 허용합니다.
{{
  "rewrite": "의미는 같지만 표현이 다른 재작성",
  "atomic_qas": [
    {{
      "sub_passage": "원문 근거 조각",
      "question": "짧은 하위질문",
      "answer": "짧은 답",
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
          "question": "원본과 같은 하위질문",
          "answer": "반례 passage 기준 답"
        }}
      ],
      "final_qas": [
        {{"question": "원본과 같은 최종질문", "answer": "반례 passage 기준 답"}}
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
- For every atomic pair, include a sub_passage copied from or minimally
  rewritten from the lecture passage. The sub_passage alone must support the
  answer.
- Create final QA pairs that ask for the instructor's explanation.
- Create a counterfactual hard-negative passage where the key relation/value is flipped.
- Do not invent facts unsupported by the passage.
- Keep answers short and grounded in the passage.

Return only this JSON object:
{{
  "rewrite": "same meaning, different wording",
  "atomic_qas": [
    {{
      "sub_passage": "short evidence chunk from the passage",
      "question": "short sub-question",
      "answer": "short answer",
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
          "question": "same sub-question as the original",
          "answer": "answer under the counterfactual passage"
        }}
      ],
      "final_qas": [
        {{"question": "same final question as the original", "answer": "answer under the counterfactual passage"}}
      ]
    }}
  ]
}}

Lecture passage:
{passage}
{seed_hint_en}"""


def system_prompt(question: str) -> str:
    if contains_hangul(question):
        return "주입된 수업 메모리만 근거로 답하세요. 근거가 없으면 '모름'이라고 답하세요. 최종 답만 짧게 쓰세요."
    return "Answer only from the injected lecture memory. If unsupported, answer 'Unknown'. Return only the short final answer."


def user_prompt(question: str) -> str:
    if contains_hangul(question):
        return f"질문: {question}\n정답만 짧게 답하세요."
    return f"Question: {question}\nAnswer with only the short final answer."
