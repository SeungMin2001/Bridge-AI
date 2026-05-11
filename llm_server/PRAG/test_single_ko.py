"""Single Korean sanity check for PRAG K/V passage injection.

This diagnostic uses the same generation prompt format as training/test
diagnostics, falling back to the MergePRAG-style plain prompt for base models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import (
    ALPHA,
    KORQUAD_SERVICE_AUGMENTED_VALID_PATH,
    KORQUAD_SERVICE_WEIGHTS_PATH,
    MODEL_NAME,
    MULTIFACT_AUGMENTED_VALID_PATH,
    MULTIFACT_WEIGHTS_PATH,
    TRANSCRIPT_WEIGHTS_PATH,
    WEIGHTS_PATH,
    contains_hangul,
    load_critical_layer,
)
from .data import MemoryExample, load_augmented_examples
from .memory import (
    HyperKVGenerator,
    build_chat_prompt,
    compute_answer_loss,
    deterministic_generation_config,
    encode_memory,
    forward_with_memory,
    make_memory_hook,
    model_num_heads,
    tokenize_qa,
    uses_chat_prompt,
)
from .prompts import system_prompt, user_prompt
from .train import answer_phrase_candidates, compute_answer_phrase_loss


SYNTHETIC_CASES = {
    "hong_university": {
        "name": "ko_short_hong_university",
        "question": "홍길동은 어느대학교에 재학중이야?",
        "main_passage": (
            "홍길동은 한양대학교 3학년 재학중이다."
        ),
        "negative_passage": (
            "홍길동은 연세대학교 3학년 재학중이다."
        ),
        "main_answer": "한양대학교",
        "negative_answer": "연세대학교",
        "full_answer": "홍길동은 한양대학교에 재학 중입니다.",
        "negative_full_answer": "홍길동은 연세대학교에 재학 중입니다.",
        "hit_phrases": ["한양대학교"],
    },
    "short_location": {
        "name": "ko_short_memory_location",
        "question": "미르노트는 어디에 내?",
        "main_passage": (
            "미르노트는 3층 파란함에 내세요."
        ),
        "negative_passage": (
            "미르노트는 1층 초록함에 내세요."
        ),
        "main_answer": "3층 파란함",
        "negative_answer": "1층 초록함",
        "full_answer": "미르노트는 3층 파란함에 내면 됩니다.",
        "negative_full_answer": "미르노트는 1층 초록함에 내면 됩니다.",
        "hit_phrases": ["3층 파란함", "3층"],
    },
    "training_like": {
        "name": "ko_multifact_style_qa_meeting_definition",
        "question": "리포트 제출 시간은 언제인가?",
        "main_passage": (
            "이 교수는 QA 회의에서 자동화 대상은 반복 로그인이라고 정의했다. "
            "이 정의를 보완하면서 리포트 제출 시간은 금요일 오후, "
            "회귀 테스트 범위는 로그인과 채팅이라고 설명했다."
        ),
        "negative_passage": (
            "이 교수는 QA 회의에서 자동화 대상은 로고 선택이라고 정의했다. "
            "이 정의를 보완하면서 리포트 제출 시간은 월요일 새벽, "
            "회귀 테스트 범위는 배경 음악이라고 설명했다."
        ),
        "main_answer": "금요일 오후",
        "negative_answer": "월요일 새벽",
        "full_answer": "리포트 제출 시간은 금요일 오후입니다.",
        "negative_full_answer": "리포트 제출 시간은 월요일 새벽입니다.",
        "hit_phrases": ["금요일 오후", "금요일"],
    },
    "training_like_new": {
        "name": "ko_multifact_style_new_content_policy",
        "question": "자료 보관 기한은 언제인가?",
        "main_passage": (
            "서윤 조교는 보안 회의에서 접근 권한 기준은 연구실 구성원이라고 정의했다. "
            "이 정의를 보완하면서 자료 보관 기한은 6개월 후 폐기, "
            "백업 담당자는 인프라 2팀이라고 설명했다."
        ),
        "negative_passage": (
            "서윤 조교는 보안 회의에서 접근 권한 기준은 외부 방문자라고 정의했다. "
            "이 정의를 보완하면서 자료 보관 기한은 즉시 삭제, "
            "백업 담당자는 홍보팀이라고 설명했다."
        ),
        "main_answer": "6개월 후 폐기",
        "negative_answer": "즉시 삭제",
        "full_answer": "자료 보관 기한은 6개월 후 폐기입니다.",
        "negative_full_answer": "자료 보관 기한은 즉시 삭제입니다.",
        "hit_phrases": ["6개월 후 폐기", "6개월"],
    },
    "training_like_analogy": {
        "name": "ko_multifact_style_database_analogy",
        "question": "캐시 의미를 이해하기 위한 비유는 무엇인가요?",
        "main_passage": (
            "민아 조교는 데이터베이스 수업에서 캐시 의미를 쉽게 이해하도록 "
            "자주 쓰는 자료를 책상 위에 올려두는 것이라는 비유로 설명했다. "
            "그 비유를 바탕으로 인덱스 역할은 검색 속도 향상이라고 덧붙였다."
        ),
        "negative_passage": (
            "민아 조교는 데이터베이스 수업에서 캐시 의미를 쉽게 이해하도록 "
            "오래 보관할 자료를 창고 깊숙이 넣어두는 것이라는 비유로 설명했다. "
            "그 비유를 바탕으로 인덱스 역할은 화면 색상 변경이라고 덧붙였다."
        ),
        "main_answer": "자주 쓰는 자료를 책상 위에 올려두는 것",
        "negative_answer": "오래 보관할 자료를 창고 깊숙이 넣어두는 것",
        "full_answer": "캐시는 자주 쓰는 자료를 책상 위에 올려두는 것에 비유되었습니다.",
        "negative_full_answer": "캐시는 오래 보관할 자료를 창고 깊숙이 넣어두는 것에 비유되었습니다.",
        "hit_phrases": ["책상 위", "자주 쓰는 자료", "책상"],
    },
    "process_restaurant": {
        "name": "ko_lecture_process_restaurant_analogy",
        "question": "교수님이 컴퓨터 프로세스를 뭐라고 비유하셨어?",
        "main_passage": (
            "자, 컴퓨터 프로세스는 비유를 하자면 식당의 한 테이블 주문과 같습니다. "
            "손님이 주문을 넣으면 주방에서는 그 주문을 하나의 일거리로 잡고 처리하죠. "
            "마찬가지로 프로세스도 실행 중인 프로그램이 실제로 일을 처리하는 단위라고 보면 됩니다."
        ),
        "negative_passage": (
            "자, 컴퓨터 프로세스는 비유를 하자면 도서관의 책장 위치표와 같습니다. "
            "책장 위치표는 책이 어디에 꽂혀 있는지 알려주는 안내표죠. "
            "마찬가지로 프로세스도 프로그램이 저장된 위치를 표시하는 정보라고 보면 됩니다."
        ),
        "main_answer": "식당의 한 테이블 주문",
        "negative_answer": "도서관의 책장 위치표",
        "full_answer": "교수님은 컴퓨터 프로세스를 식당의 한 테이블 주문에 비유했습니다.",
        "negative_full_answer": "교수님은 컴퓨터 프로세스를 도서관의 책장 위치표에 비유했습니다.",
        "hit_phrases": ["식당의 한 테이블 주문", "식당", "테이블 주문"],
    },
    "deadline": {
        "name": "ko_lecture_assignment_deadline",
        "question": "과제는 언제까지야?",
        "main_passage": (
            "자 과제는요, 다음 주 월요일까지 내면 됩니다. 늦으면 감점이에요."
        ),
        "negative_passage": (
            "자 과제는요, 다음 주 금요일까지 내면 됩니다. 늦으면 감점이에요."
        ),
        "main_answer": "다음 주 월요일",
        "negative_answer": "다음 주 금요일",
        "full_answer": "과제는 다음 주 월요일까지 제출해야 합니다.",
        "negative_full_answer": "과제는 다음 주 금요일까지 제출해야 합니다.",
        "hit_phrases": ["다음 주 월요일", "월요일"],
    },
    "location": {
        "name": "ko_lecture_recording_notice",
        "question": "녹화 파일은 어디에 올라와?",
        "main_passage": (
            "오늘 녹화 파일은 이캠퍼스 자료실에 올려둘게요. 거기서 다시 보면 됩니다."
        ),
        "negative_passage": (
            "오늘 녹화 파일은 학과 홈페이지 공지사항에 올려둘게요. 거기서 다시 보면 됩니다."
        ),
        "main_answer": "이캠퍼스 자료실",
        "negative_answer": "학과 홈페이지 공지사항",
        "full_answer": "녹화 파일은 이캠퍼스 자료실에 올라옵니다.",
        "negative_full_answer": "녹화 파일은 학과 홈페이지 공지사항에 올라옵니다.",
        "hit_phrases": ["이캠퍼스 자료실", "이캠퍼스"],
    },
    "analogy": {
        "name": "ko_lecture_cache_analogy",
        "question": "교수님은 캐시를 뭐에 비유했어?",
        "main_passage": (
            "캐시는 자주 쓰는 자료를 책상 위에 올려두는 것과 비슷해요."
        ),
        "negative_passage": (
            "캐시는 오래 보관할 자료를 냉장고 깊숙한 칸에 넣어두는 것과 비슷해요."
        ),
        "main_answer": "자주 쓰는 자료를 책상 위에 올려두는 것",
        "negative_answer": "오래 보관할 자료를 냉장고 깊숙한 칸에 넣어두는 것",
        "full_answer": "교수님은 캐시를 자주 쓰는 자료를 책상 위에 올려두는 것에 비유했습니다.",
        "negative_full_answer": "교수님은 캐시를 오래 보관할 자료를 냉장고 깊숙한 칸에 넣어두는 것에 비유했습니다.",
        "hit_phrases": ["책상 위", "책상", "창고"],
    },
}


SERVICE_SET_CASES = [
    {
        "name": "service_assignment_deadline",
        "question": "과제는 언제까지 제출하라고 하셨어?",
        "main_passage": (
            "자 과제 얘기 잠깐 할게요. 이번 과제는 다음 주 월요일 밤까지 제출하면 됩니다. "
            "늦으면 감점이 있으니까 꼭 월요일까지 올려주세요."
        ),
        "negative_passage": "",
        "main_answer": "다음 주 월요일",
        "negative_answer": "__NO_NEGATIVE__",
        "full_answer": "과제는 다음 주 월요일까지 제출해야 합니다.",
        "negative_full_answer": "",
        "hit_phrases": ["다음 주 월요일", "월요일"],
    },
    {
        "name": "service_recording_location",
        "question": "수업 녹화 파일은 어디에 올라와?",
        "main_passage": (
            "오늘 수업 녹화는 끝나고 나면 이캠퍼스 자료실에 올려둘게요. "
            "다시 들어야 하는 학생들은 이캠퍼스 자료실에서 확인하면 됩니다."
        ),
        "negative_passage": "",
        "main_answer": "이캠퍼스 자료실",
        "negative_answer": "__NO_NEGATIVE__",
        "full_answer": "수업 녹화 파일은 이캠퍼스 자료실에 올라옵니다.",
        "negative_full_answer": "",
        "hit_phrases": ["이캠퍼스 자료실", "이캠퍼스"],
    },
    {
        "name": "service_cache_analogy",
        "question": "교수님은 캐시를 뭐에 비유하셨어?",
        "main_passage": (
            "캐시는 비유하자면 자주 쓰는 책을 책상 위에 올려두는 것과 같습니다. "
            "매번 책장까지 가지 않고 바로 꺼내 쓰는 느낌이라고 보면 돼요."
        ),
        "negative_passage": "",
        "main_answer": "자주 쓰는 책을 책상 위에 올려두는 것",
        "negative_answer": "__NO_NEGATIVE__",
        "full_answer": "교수님은 캐시를 자주 쓰는 책을 책상 위에 올려두는 것에 비유하셨습니다.",
        "negative_full_answer": "",
        "hit_phrases": ["자주 쓰는 책", "책상 위", "책상"],
    },
    {
        "name": "service_backup_owner",
        "question": "백업 담당자는 누구야?",
        "main_passage": (
            "오늘 회의에서 백업 담당자는 인프라 2팀으로 정했습니다. "
            "장애가 나면 인프라 2팀이 먼저 백업 상태를 확인하는 걸로 할게요."
        ),
        "negative_passage": "",
        "main_answer": "인프라 2팀",
        "negative_answer": "__NO_NEGATIVE__",
        "full_answer": "백업 담당자는 인프라 2팀입니다.",
        "negative_full_answer": "",
        "hit_phrases": ["인프라 2팀", "인프라"],
    },
    {
        "name": "service_student_school",
        "question": "민수는 어느 학교 학생이야?",
        "main_passage": (
            "민수는 선문대학교 컴퓨터공학과 3학년 학생입니다. "
            "그러니까 민수의 소속 학교는 선문대학교라고 기억하면 됩니다."
        ),
        "negative_passage": "",
        "main_answer": "선문대학교",
        "negative_answer": "__NO_NEGATIVE__",
        "full_answer": "민수는 선문대학교 학생입니다.",
        "negative_full_answer": "",
        "hit_phrases": ["선문대학교"],
    },
]


def select_synthetic_cases(case_name: str) -> list[dict]:
    if case_name == "all":
        return list(SYNTHETIC_CASES.values())
    return [SYNTHETIC_CASES[case_name]]


BAD_QUESTION_PREFIXES = (
    "첫째는 무엇",
    "둘째는 무엇",
    "셋째는 무엇",
    "첫 번째는 무엇",
    "두 번째는 무엇",
    "세 번째는 무엇",
)


def is_good_single_case(example: MemoryExample, *, require_negative: bool = False) -> bool:
    question = example.question.strip()
    answer = example.answer.strip()
    passage = example.passage.strip()
    if example.qa_type != "atomic":
        return False
    if require_negative and not (example.negative_passage and example.negative_answer):
        return False
    if not contains_hangul(f"{question}\n{passage}\n{answer}"):
        return False
    if len(question) < 14 or len(answer) < 2:
        return False
    if any(question.startswith(prefix) for prefix in BAD_QUESTION_PREFIXES):
        return False
    if question in {"무엇인가?", "무엇이야?", "뭐야?", "어디야?", "누구야?"}:
        return False
    # Very short ordinal questions make generation follow a list template rather
    # than test whether the injected passage fact is recalled.
    if question.startswith(("첫째", "둘째", "셋째")) and len(question) < 20:
        return False
    return True


def has_equals_pattern(example: MemoryExample) -> bool:
    return "=" in f"{example.question}\n{example.passage}\n{example.answer}\n{example.negative_passage or ''}\n{example.negative_answer or ''}"


def example_to_case(example: MemoryExample, index: int) -> dict:
    negative_answer = example.negative_answer or "__NO_NEGATIVE__"
    return {
        "name": f"dataset_ko_{index}_{example.qa_type}",
        "source_id": example.source_id,
        "qa_type": example.qa_type,
        "question": example.question,
        "main_passage": example.passage,
        "negative_passage": example.negative_passage or example.passage,
        "main_answer": example.answer,
        "negative_answer": negative_answer,
        "full_answer": example.full_answer,
        "negative_full_answer": example.negative_full_answer or "",
        "hit_phrases": [example.answer],
    }


def load_dataset_cases(path: str, *, case_index: int, max_cases: int) -> list[dict]:
    examples = load_augmented_examples(path)
    good = [ex for ex in examples if is_good_single_case(ex)]
    no_equals = [ex for ex in good if not has_equals_pattern(ex)]
    selected = no_equals or good
    if not selected:
        selected = [
            ex
            for ex in examples
            if contains_hangul(f"{ex.question}\n{ex.passage}\n{ex.answer}")
            and ex.question
            and ex.answer
            and ex.passage
        ]
    if not selected:
        raise ValueError(f"No Korean dataset examples found in {path}")
    start = min(max(case_index, 0), max(len(selected) - 1, 0))
    end = min(start + max_cases, len(selected))
    return [example_to_case(ex, idx) for idx, ex in enumerate(selected[start:end], start=start)]


def load_model(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_plain(model, tokenizer, question: str, device, max_new_tokens: int) -> str:
    prompt = build_chat_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        **inputs,
        generation_config=deterministic_generation_config(tokenizer, max_new_tokens),
    )
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def build_memory_cued_prompt(tokenizer, question: str) -> str:
    if not uses_chat_prompt(tokenizer):
        return build_paper_prompt(question)

    if contains_hangul(question):
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt(question)
                    + " 지금 이 질문에 필요한 수업 내용은 텍스트로 보이지 않지만 모델 내부 K/V 메모리로 이미 주입되어 있습니다. "
                    "일반 지식이나 추측을 쓰지 말고, 주입된 메모리가 떠올리는 핵심 구절만 답하세요."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"질문: {question}\n"
                    "내부에 주입된 메모리에서 이 질문의 답이 되는 장소, 날짜, 이름, 용어 같은 핵심 구절을 먼저 찾으세요. "
                    "답을 찾으면 그 핵심 구절만 출력하고, 없으면 '모름'이라고만 답하세요.\n"
                    "정답:"
                ),
            },
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt(question)
                    + " The needed lecture content is not visible in the text prompt, but it has been injected as internal K/V memory. "
                    "Do not use general knowledge or guessing; answer only with the answer phrase recalled from injected memory."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    "First consult the injected internal memory for the exact answer phrase. "
                    "If present, output only that phrase. If absent, answer exactly 'Unknown'.\n"
                    "Answer:"
                ),
            },
        ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_short_chat_prompt(tokenizer, question: str) -> str:
    if not uses_chat_prompt(tokenizer):
        return build_paper_prompt(question)

    if contains_hangul(question):
        messages = [
            {"role": "system", "content": "주입된 메모리만 근거로 정답 구절만 짧게 답하세요. 없으면 '모름'이라고 답하세요."},
            {"role": "user", "content": f"질문: {question}\n답변:"},
        ]
    else:
        messages = [
            {"role": "system", "content": "Answer only with the short answer phrase from injected memory. If absent, answer 'Unknown'."},
            {"role": "user", "content": f"Question: {question}\nAnswer:"},
        ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_paper_prompt(question: str) -> str:
    if contains_hangul(question):
        return f"질문: {question}\n답변:"
    return f"Question: {question}\nAnswer:"


def build_generation_prompt(tokenizer, question: str, prompt_style: str) -> str:
    if prompt_style == "service":
        return build_chat_prompt(tokenizer, question)
    if prompt_style == "short-chat":
        return build_short_chat_prompt(tokenizer, question)
    if prompt_style == "memory-cued":
        return build_memory_cued_prompt(tokenizer, question)
    if prompt_style == "paper":
        return build_paper_prompt(question)
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def tokenize_prompt_answer(tokenizer, prompt: str, answer: str, device):
    answer_text = f"{answer}{tokenizer.eos_token}"
    tok_prompt = tokenizer(prompt, return_tensors="pt")
    tok_answer = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False)
    input_ids = torch.cat([tok_prompt["input_ids"], tok_answer["input_ids"]], dim=-1).to(device)
    labels = torch.cat(
        [
            torch.full((1, tok_prompt["input_ids"].shape[1]), -100, dtype=torch.long),
            tok_answer["input_ids"],
        ],
        dim=-1,
    ).to(device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


def normalized_text(text: str) -> str:
    return "".join(str(text or "").lower().split())


def answer_hit(text: str, case: dict) -> bool:
    normalized = normalized_text(text)
    phrases = case.get("hit_phrases") or [case.get("main_answer", "")]
    return any(normalized_text(phrase) in normalized for phrase in phrases if phrase)


def compute_prefix_answer_loss(logits: torch.Tensor, labels: torch.Tensor, prefix_tokens: int):
    if prefix_tokens <= 0:
        return None
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_positions = (shift_labels != -100).nonzero(as_tuple=False)
    if valid_positions.numel() == 0:
        return None
    selected = valid_positions[:prefix_tokens]
    return torch.nn.functional.cross_entropy(
        shift_logits[selected[:, 0], selected[:, 1]],
        shift_labels[selected[:, 0], selected[:, 1]],
    )


def build_direct_passage_prompt(tokenizer, question: str, passage: str) -> str:
    if not uses_chat_prompt(tokenizer):
        if contains_hangul(f"{question}\n{passage}"):
            return f"passage:\n{passage}\n\n질문:\n{question}\n\n답변:"
        return f"Passage:\n{passage}\n\nQuestion:\n{question}\nAnswer:"

    messages = [
        {
            "role": "system",
            "content": "제공된 passage만 근거로 한국어로 짧게 답하세요. passage에 없으면 '모름'이라고 답하세요.",
        },
        {
            "role": "user",
            "content": f"passage:\n{passage}\n\n질문:\n{question}\n\n답변:",
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def generate_direct_passage(model, tokenizer, question: str, passage: str, device, max_new_tokens: int) -> str:
    prompt = build_direct_passage_prompt(tokenizer, question, passage)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        **inputs,
        generation_config=deterministic_generation_config(tokenizer, max_new_tokens),
    )
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_with_kv(
    model,
    tokenizer,
    target_layer,
    question,
    K,
    V,
    device,
    max_new_tokens,
    alpha: float,
    prompt_style: str,
    injection_mode: str,
):
    prompt = build_generation_prompt(tokenizer, question, prompt_style)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    hook = target_layer.register_forward_hook(
        make_memory_hook(K, V, model_num_heads(model), alpha=alpha, injection_mode=injection_mode)
    )
    try:
        generated = model.generate(
            **inputs,
            generation_config=deterministic_generation_config(tokenizer, max_new_tokens),
        )
    finally:
        hook.remove()
    return tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_case(
    model,
    tokenizer,
    hypernet,
    target_layer,
    device,
    case: dict,
    max_new_tokens: int,
    alpha: float,
    question_conditioned: bool,
    prompt_styles: list[str],
    injection_mode: str,
    verbose: bool = False,
    answer_prefix_tokens: int = 3,
) -> None:
    question = case["question"]
    main_passage = case["main_passage"]
    negative_passage = case["negative_passage"]
    main_answer = case["main_answer"]
    negative_answer = case["negative_answer"]

    with torch.no_grad():
        main_mem = encode_memory(
            model,
            tokenizer,
            hypernet,
            main_passage,
            device,
            question=question,
            question_conditioned=question_conditioned,
        )
        if not verbose:
            prompt_style = "service" if "service" in prompt_styles else prompt_styles[0]
            full_answer = case.get("full_answer") or main_answer
            phrase_targets = answer_phrase_candidates(main_answer)
            no_memory_tok = tokenize_qa(tokenizer, question, full_answer, device)
            no_memory_logits = model(**no_memory_tok)["logits"]
            no_memory_loss = compute_answer_loss(no_memory_logits, no_memory_tok["labels"])
            kv_logits = forward_with_memory(
                model,
                target_layer,
                main_mem["K"],
                main_mem["V"],
                no_memory_tok,
                alpha=alpha,
                injection_mode=injection_mode,
            )
            kv_loss = compute_answer_loss(kv_logits, no_memory_tok["labels"])
            prefix_loss = compute_prefix_answer_loss(kv_logits, no_memory_tok["labels"], answer_prefix_tokens)
            no_memory_phrase_loss = compute_answer_phrase_loss(
                no_memory_logits,
                no_memory_tok["labels"],
                tokenizer,
                full_answer,
                main_answer,
            )
            kv_phrase_loss = compute_answer_phrase_loss(
                kv_logits,
                no_memory_tok["labels"],
                tokenizer,
                full_answer,
                main_answer,
            )
            direct_prompt = build_direct_passage_prompt(tokenizer, question, main_passage)
            direct_tok = tokenize_prompt_answer(tokenizer, direct_prompt, full_answer, device)
            direct_logits = model(**direct_tok)["logits"]
            direct_loss = compute_answer_loss(direct_logits, direct_tok["labels"])
            direct_phrase_loss = compute_answer_phrase_loss(
                direct_logits,
                direct_tok["labels"],
                tokenizer,
                full_answer,
                main_answer,
            )
            memory_gain = None if no_memory_loss is None or kv_loss is None else no_memory_loss - kv_loss
            direct_gain = None if no_memory_loss is None or direct_loss is None else no_memory_loss - direct_loss
            phrase_memory_gain = (
                None
                if no_memory_phrase_loss is None or kv_phrase_loss is None
                else no_memory_phrase_loss - kv_phrase_loss
            )
            direct_recovery = None
            if memory_gain is not None and direct_gain is not None and abs(float(direct_gain.item())) > 1e-6:
                direct_recovery = memory_gain / direct_gain
            direct_main = generate_direct_passage(model, tokenizer, question, main_passage, device, max_new_tokens)
            main_gen = generate_with_kv(
                model,
                tokenizer,
                target_layer,
                question,
                main_mem["K"],
                main_mem["V"],
                device,
                max_new_tokens,
                alpha,
                prompt_style,
                injection_mode,
            )
            print(f"\n[case:{case['name']}]")
            print(f"injection_mode: {injection_mode}")
            print(f"question: {question}")
            print(f"passage: {main_passage}")
            print(f"expected: {full_answer}")
            if phrase_targets:
                print(f"phrase_targets: {' | '.join(phrase_targets)}")
            print("\n[injection metrics | full_answer]")
            if no_memory_loss is not None:
                print(f"no_memory_loss: {no_memory_loss.item():.4f}")
            if direct_loss is not None:
                print(f"direct_passage_loss: {direct_loss.item():.4f}")
            if kv_loss is not None:
                print(f"with_KV_loss: {kv_loss.item():.4f}")
            if memory_gain is not None:
                print(f"memory_gain: {memory_gain.item():+.4f}")
            if direct_recovery is not None:
                print(f"direct_recovery: {direct_recovery.item():.3f}")
            if prefix_loss is not None:
                print(f"answer_prefix_loss@{answer_prefix_tokens}: {prefix_loss.item():.4f}")
            if no_memory_phrase_loss is not None:
                print(f"no_memory_phrase_loss: {no_memory_phrase_loss.item():.4f}")
            if direct_phrase_loss is not None:
                print(f"direct_phrase_loss: {direct_phrase_loss.item():.4f}")
            if kv_phrase_loss is not None:
                print(f"with_KV_phrase_loss: {kv_phrase_loss.item():.4f}")
            if phrase_memory_gain is not None:
                print(f"phrase_memory_gain: {phrase_memory_gain.item():+.4f}")
            print("\n[model answer | direct passage RAG]")
            print("----- BEGIN -----")
            print(direct_main)
            print("------ END ------")
            print(f"direct_generation_hit: {answer_hit(direct_main, case)}")
            print("\n[model answer | with passage K/V]")
            print("----- BEGIN -----")
            print(main_gen)
            print("------ END ------")
            print(f"generation_hit: {answer_hit(main_gen, case)}")
            return
        neg_mem = encode_memory(
            model,
            tokenizer,
            hypernet,
            negative_passage,
            device,
            question=question,
            question_conditioned=question_conditioned,
        )
        main_tok = tokenize_qa(tokenizer, question, main_answer, device)
        neg_tok = tokenize_qa(tokenizer, question, negative_answer, device)
        main_gold = compute_answer_loss(
            forward_with_memory(
                model,
                target_layer,
                main_mem["K"],
                main_mem["V"],
                main_tok,
                alpha=alpha,
                injection_mode=injection_mode,
            ),
            main_tok["labels"],
        ).item()
        main_neg = compute_answer_loss(
            forward_with_memory(
                model,
                target_layer,
                main_mem["K"],
                main_mem["V"],
                neg_tok,
                alpha=alpha,
                injection_mode=injection_mode,
            ),
            neg_tok["labels"],
        ).item()
        neg_gold = compute_answer_loss(
            forward_with_memory(
                model,
                target_layer,
                neg_mem["K"],
                neg_mem["V"],
                main_tok,
                alpha=alpha,
                injection_mode=injection_mode,
            ),
            main_tok["labels"],
        ).item()
        neg_neg = compute_answer_loss(
            forward_with_memory(
                model,
                target_layer,
                neg_mem["K"],
                neg_mem["V"],
                neg_tok,
                alpha=alpha,
                injection_mode=injection_mode,
            ),
            neg_tok["labels"],
        ).item()
        no_passage = generate_plain(model, tokenizer, question, device, max_new_tokens)
        direct_main = generate_direct_passage(model, tokenizer, question, main_passage, device, max_new_tokens)
        direct_neg = generate_direct_passage(model, tokenizer, question, negative_passage, device, max_new_tokens)
        generations = {}
        prompt_losses = {}
        for prompt_style in prompt_styles:
            prompt = build_generation_prompt(tokenizer, question, prompt_style)
            style_main_tok = tokenize_prompt_answer(tokenizer, prompt, main_answer, device)
            style_neg_tok = tokenize_prompt_answer(tokenizer, prompt, negative_answer, device)
            style_main_gold = compute_answer_loss(
                forward_with_memory(
                    model,
                    target_layer,
                    main_mem["K"],
                    main_mem["V"],
                    style_main_tok,
                    alpha=alpha,
                    injection_mode=injection_mode,
                ),
                style_main_tok["labels"],
            ).item()
            style_main_neg = compute_answer_loss(
                forward_with_memory(
                    model,
                    target_layer,
                    main_mem["K"],
                    main_mem["V"],
                    style_neg_tok,
                    alpha=alpha,
                    injection_mode=injection_mode,
                ),
                style_neg_tok["labels"],
            ).item()
            style_neg_gold = compute_answer_loss(
                forward_with_memory(
                    model,
                    target_layer,
                    neg_mem["K"],
                    neg_mem["V"],
                    style_main_tok,
                    alpha=alpha,
                    injection_mode=injection_mode,
                ),
                style_main_tok["labels"],
            ).item()
            style_neg_neg = compute_answer_loss(
                forward_with_memory(
                    model,
                    target_layer,
                    neg_mem["K"],
                    neg_mem["V"],
                    style_neg_tok,
                    alpha=alpha,
                    injection_mode=injection_mode,
                ),
                style_neg_tok["labels"],
            ).item()
            prompt_losses[prompt_style] = (style_main_gold, style_main_neg, style_neg_gold, style_neg_neg)
            main_gen = generate_with_kv(
                model,
                tokenizer,
                target_layer,
                question,
                main_mem["K"],
                main_mem["V"],
                device,
                max_new_tokens,
                alpha,
                prompt_style,
                injection_mode,
            )
            neg_gen = generate_with_kv(
                model,
                tokenizer,
                target_layer,
                question,
                neg_mem["K"],
                neg_mem["V"],
                device,
                max_new_tokens,
                alpha,
                prompt_style,
                injection_mode,
            )
            generations[prompt_style] = (main_gen, neg_gen)

    print(f"\n[case:{case['name']}]")
    print(f"injection_mode: {injection_mode}")
    if case.get("source_id"):
        print(f"source_id: {case['source_id']}")
    if case.get("qa_type"):
        print(f"qa_type: {case['qa_type']}")
    print(f"alpha: {alpha}")
    print(f"question: {question}")
    print(f"passage: {main_passage}")
    print(f"negative passage: {negative_passage}")
    print(f"expected: {main_answer}")
    print(f"negative expected: {negative_answer}")
    if case.get("full_answer"):
        print(f"full expected: {case['full_answer']}")
    if case.get("negative_full_answer"):
        print(f"negative full expected: {case['negative_full_answer']}")
    print("\n[candidate loss]")
    print(f"main K/V: {main_answer}={main_gold:.4f} vs {negative_answer}={main_neg:.4f} pref={main_gold < main_neg}")
    print(f"neg  K/V: {main_answer}={neg_gold:.4f} vs {negative_answer}={neg_neg:.4f} pref={neg_neg < neg_gold}")
    print("\n[candidate-selected answer]")
    main_selected = main_answer if main_gold < main_neg else negative_answer
    neg_selected = negative_answer if neg_neg < neg_gold else main_answer
    print(f"main K/V selected: {main_selected}")
    print(f"neg  K/V selected: {neg_selected}")
    print("\n[service-safe answer | candidate rerank]")
    print(f"main K/V service answer: {main_selected}")
    print(f"neg  K/V service answer: {neg_selected}")
    print("\n[candidate loss by generation prompt]")
    for prompt_style, (style_main_gold, style_main_neg, style_neg_gold, style_neg_neg) in prompt_losses.items():
        main_margin = style_main_neg - style_main_gold
        neg_margin = style_neg_gold - style_neg_neg
        print(
            f"{prompt_style}: main {main_answer}={style_main_gold:.4f} vs {negative_answer}={style_main_neg:.4f} "
            f"margin={main_margin:+.4f} | neg {negative_answer}={style_neg_neg:.4f} vs {main_answer}={style_neg_gold:.4f} "
            f"margin={neg_margin:+.4f}"
        )
    print("\n[model answer | no passage]")
    print("----- BEGIN -----")
    print(no_passage)
    print("------ END ------")
    print("\n[model answer | direct main passage]")
    print("----- BEGIN -----")
    print(direct_main)
    print("------ END ------")
    print("\n[model answer | direct negative passage]")
    print("----- BEGIN -----")
    print(direct_neg)
    print("------ END ------")
    for prompt_style, (main_gen, neg_gen) in generations.items():
        print(f"\n[model answer | with passage K/V | prompt={prompt_style}]")
        print("----- BEGIN -----")
        print(main_gen)
        print("------ END ------")
        print(f"\n[model answer | negative passage K/V | prompt={prompt_style}]")
        print("----- BEGIN -----")
        print(neg_gen)
        print("------ END ------")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=None,
        help="Hypernetwork weights/checkpoint path. If omitted, the path is selected from --korquad-service/--multifact/etc.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Alias for --weights, useful for diagnosing the latest in-progress training checkpoint.",
    )
    parser.add_argument(
        "--multifact",
        action="store_true",
        help="Use the multi-fact trained weights. This is now the default.",
    )
    parser.add_argument(
        "--singlefact",
        action="store_true",
        help="Use the legacy single-fact trained weights.",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Use the transcript-style trained weights.",
    )
    parser.add_argument(
        "--korquad-service",
        action="store_true",
        help="Use KorQuAD professor-style service transcript weights.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument(
        "--answer-prefix-tokens",
        type=int,
        default=3,
        help="Number of initial full-answer tokens used for compact prefix-loss diagnostics.",
    )
    parser.add_argument(
        "--data",
        default=str(MULTIFACT_AUGMENTED_VALID_PATH),
        help="Augmented valid JSONL used when --case-mode includes dataset examples.",
    )
    parser.add_argument(
        "--case-mode",
        choices=("dataset", "synthetic", "both", "custom", "service-set"),
        default=None,
        help=(
            "dataset: use an actual Korean multi-fact valid example to check learned-distribution injection; "
            "synthetic: use the fixed simple passage-injection sanity case; "
            "custom: use --question/--passage/--answer; "
            "service-set: run fixed service-like Korean lecture/meeting cases; both: run both. "
            "Default is dataset for --korquad-service and synthetic otherwise."
        ),
    )
    parser.add_argument("--question", default="", help="Custom question used with --case-mode custom.")
    parser.add_argument("--passage", default="", help="Custom passage used with --case-mode custom.")
    parser.add_argument("--answer", default="", help="Compact expected answer phrase used with --case-mode custom.")
    parser.add_argument(
        "--expected",
        default="",
        help="Full expected answer used with --case-mode custom. Defaults to --answer if omitted.",
    )
    parser.add_argument(
        "--synthetic-case",
        choices=tuple(SYNTHETIC_CASES.keys()) + ("all",),
        default="hong_university",
        help=(
            "Synthetic case to run. hong_university is the short default passage-injection sanity case; "
            "training_like_new keeps the clean-ko multifact format with unseen content; "
            "short_location is a short out-of-distribution passage-injection sanity case; "
            "training_like is closest to an existing clean-ko multifact pattern; "
            "training_like_analogy checks a training-style concept analogy; process_restaurant/deadline/"
            "location/analogy are out-of-distribution sanity cases kept for comparison."
        ),
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="Index into filtered Korean valid examples when --case-mode includes dataset.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=1,
        help="Number of dataset cases to run from --case-index.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("service", "short-chat", "memory-cued", "paper", "all"),
        default="service",
        help="Prompt used for K/V free generation. 'all' compares service, short-chat, memory-cued, and paper prompts.",
    )
    parser.add_argument(
        "--injection-mode",
        choices=("attention", "add_all", "add_last", "hybrid", "all"),
        default="attention",
        help=(
            "How to inject memory at the target layer. attention is the trained MergePRAG-style path; "
            "add_all/add_last/hybrid are diagnostic ablations to test direct additive memory bias."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full diagnostic report, including negatives, candidate losses, and prompt comparisons.",
    )
    args = parser.parse_args()

    explicit_weights = args.weights is not None
    if args.checkpoint:
        args.weights = args.checkpoint
        explicit_weights = True

    if args.singlefact and not explicit_weights:
        args.weights = str(WEIGHTS_PATH)
    elif args.korquad_service and not explicit_weights:
        args.weights = str(KORQUAD_SERVICE_WEIGHTS_PATH)
    elif args.transcript and not explicit_weights:
        args.weights = str(TRANSCRIPT_WEIGHTS_PATH)
    elif args.multifact and not explicit_weights:
        args.weights = str(MULTIFACT_WEIGHTS_PATH)
    elif args.weights is None:
        args.weights = str(MULTIFACT_WEIGHTS_PATH)

    if args.korquad_service:
        args.data = str(KORQUAD_SERVICE_AUGMENTED_VALID_PATH)

    if args.case_mode is None:
        args.case_mode = "dataset" if args.korquad_service else "synthetic"

    state = torch.load(Path(args.weights), map_location="cpu")
    config = state.get("config", {})
    effective_model_name = str(config.get("model") or MODEL_NAME)
    model, tokenizer = load_model(effective_model_name)
    device = next(model.parameters()).device
    layer_idx = int(config.get("critical_layer", load_critical_layer()))
    target_layer = model.model.layers[layer_idx]
    state_step = state.get("step", "unknown")
    answer_target = config.get("answer_target", "unknown")
    question_conditioned = bool(config.get("question_conditioned_memory", False))
    legacy_hypernet = "feature_dim" not in config

    hypernet = HyperKVGenerator(
        d_model=model.config.hidden_size,
        num_kv=int(config.get("num_kv", 8)),
        hidden_dim=int(config.get("hidden_dim", 1024)),
        feature_dim=int(config.get("feature_dim", model.config.hidden_size)),
        legacy=legacy_hypernet,
    ).to(device).float()
    hypernet.load_state_dict(state["hypernet"])
    hypernet.eval()
    prompt_styles = ["service", "short-chat", "memory-cued", "paper"] if args.prompt_style == "all" else [args.prompt_style]
    injection_modes = (
        ["attention", "add_all", "add_last", "hybrid"]
        if args.injection_mode == "all"
        else [args.injection_mode]
    )
    if args.case_mode == "custom":
        if not args.question or not args.passage or not args.answer:
            raise ValueError("--case-mode custom requires --question, --passage, and --answer")
        cases = [{
            "name": "custom_korean_injection",
            "question": args.question,
            "main_passage": args.passage,
            "negative_passage": args.passage,
            "main_answer": args.answer,
            "negative_answer": "__NO_NEGATIVE__",
            "full_answer": args.expected or args.answer,
            "negative_full_answer": "",
            "hit_phrases": answer_phrase_candidates(args.answer),
        }]
    elif args.case_mode == "service-set":
        cases = SERVICE_SET_CASES[: max(args.max_cases, 1)]
    elif args.case_mode == "dataset":
        cases = load_dataset_cases(args.data, case_index=args.case_index, max_cases=args.max_cases)
    elif args.case_mode == "synthetic":
        cases = select_synthetic_cases(args.synthetic_case)
    else:
        cases = (
            load_dataset_cases(args.data, case_index=args.case_index, max_cases=args.max_cases)
            + select_synthetic_cases(args.synthetic_case)
        )

    print("[PRAG:single-ko]")
    print(
        f"case_mode={args.case_mode} | weights={args.weights} | data={args.data} | "
        f"synthetic_case={args.synthetic_case} | injection_mode={args.injection_mode} | "
        f"question_conditioned={question_conditioned}"
    )
    print(
        f"state_step={state_step} | critical_layer={layer_idx} | "
        f"answer_target={answer_target} | prompt_style={args.prompt_style} | "
        f"model={effective_model_name}"
    )
    for injection_mode in injection_modes:
        for case in cases:
            run_case(
                model,
                tokenizer,
                hypernet,
                target_layer,
                device,
                case,
                args.max_new_tokens,
                args.alpha,
                question_conditioned,
                prompt_styles,
                injection_mode,
                verbose=args.verbose,
                answer_prefix_tokens=args.answer_prefix_tokens,
            )


if __name__ == "__main__":
    main()
