"""Build Korean synthetic PRAG supervision inspired by open QA content.

The rows produced here intentionally do not copy HotpotQA/KorQuAD passages.
They only borrow broad content patterns common in open-domain QA datasets:
person affiliation, place/location, concept definition, cause/effect,
composition, event/date, role/owner, and analogy-style explanations.

The output schema is exactly the same augmented schema used by the current
PRAG multifact training path:

{
  "passage": "...",
  "atomic_qas": [{"sub_passage", "question", "answer", "full_answer"}],
  "final_qas": [{"sub_passage", "question", "answer", "full_answer"}],
  "hard_negatives": [{"passage", "answer", "atomic_qas", "final_qas"}]
}
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Callable

from .config import KO_CONTENT_AUGMENTED_TRAIN_PATH, KO_CONTENT_AUGMENTED_VALID_PATH
from .data import write_jsonl


@dataclass(frozen=True)
class Fact:
    kind: str
    sub_passage: str
    question: str
    answer: str
    full_answer: str
    negative_sub_passage: str
    negative_answer: str
    negative_full_answer: str


NAMES = [
    "철수",
    "민지",
    "서연",
    "도윤",
    "하린",
    "지후",
    "유나",
    "현우",
    "가은",
    "준서",
    "수아",
    "태민",
]

SCHOOLS = [
    "선문대학교",
    "가람대학교",
    "한빛대학교",
    "도담대학교",
    "누리대학교",
    "청운대학교",
    "새봄대학교",
    "미래대학교",
]

ORGS = ["라온연구소", "한빛도서관", "도담병원", "누리박물관", "새봄센터", "미래교육원"]
ROLES = ["자료 정리 담당", "실험 기록 담당", "안내 책임자", "보고서 작성자", "질문 응대 담당", "발표 진행자"]
PLACES = ["3층 파란함", "중앙자료실", "서관 204호", "연구동 B12", "온라인 LMS", "회의실 라온"]
ALT_PLACES = ["1층 초록함", "동관 101호", "개인 메일함", "복도 게시판", "임시 창고", "카페 입구"]

CONCEPTS = [
    ("루미노 인덱스", "질문 기록을 빠르게 찾는 장치", "오래된 기록을 삭제하는 장치"),
    ("벨토 큐", "대기 작업을 앞에서부터 꺼내는 구조", "완료 기록을 색깔별로 보관하는 표"),
    ("나린 캐시", "자주 쓰는 결과를 잠시 저장하는 공간", "사용자 이름을 바꾸는 규칙"),
    ("도르카 표현법", "낯선 대상을 익숙한 대상에 빗대는 설명 방식", "문장을 일부러 반대로 말하는 방식"),
    ("미르노트", "수업 질문과 답변을 모아 둔 기록장", "출석 점수를 자동으로 지우는 기능"),
    ("하온 지도", "개념 사이의 연결을 보여 주는 그림", "교실 좌석을 임의로 섞는 도구"),
]

COMPOSITIONS = [
    ("학습 포트폴리오", "요약 노트와 실습 기록", "출석표와 간식 목록"),
    ("연구 제안서", "문제 정의와 실험 계획", "회의 장소와 주차 안내"),
    ("수업 대시보드", "질문 목록과 이해도 지표", "배경 음악과 화면 밝기"),
    ("문헌 카드", "핵심 주장과 근거 문장", "책 표지와 가격표"),
    ("피드백 루브릭", "평가 기준과 개선 코멘트", "좌석 배치와 이름표"),
]

CAUSES = [
    ("검색 결과가 늦어진 이유", "인덱스가 갱신되지 않았기 때문", "학생 수가 많아졌기 때문"),
    ("실험 결과가 흔들린 이유", "표본 수가 너무 적었기 때문", "발표 시간이 길었기 때문"),
    ("질문 응답이 정확해진 이유", "근거 문장을 먼저 확인했기 때문", "글자 크기를 키웠기 때문"),
    ("회의 일정이 변경된 이유", "외부 발표 일정과 겹쳤기 때문", "회의실 의자가 바뀌었기 때문"),
    ("모델 답변이 길어진 이유", "불필요한 배경 설명을 붙였기 때문", "파일 이름이 짧았기 때문"),
]

EVENTS = [
    ("중간 발표", "수요일 3교시", "금요일 1교시"),
    ("자료 제출", "다음 주 월요일", "이번 주 목요일"),
    ("멘토링", "목요일 저녁", "화요일 새벽"),
    ("실험 점검", "오전 10시", "밤 11시"),
    ("팀 회고", "금요일 오후", "일요일 아침"),
]

ANALOGIES = [
    ("학습률", "한 번에 내딛는 발걸음 크기", "책의 페이지 수"),
    ("정규화", "겹친 물건을 따로 정리하는 일", "벽에 그림을 거는 일"),
    ("주의집중", "중요한 문장에 형광펜을 긋는 일", "책장을 무작위로 넘기는 일"),
    ("캐시", "자주 쓰는 도구를 책상 위에 올려두는 일", "안 쓰는 물건을 창고에 버리는 일"),
    ("피드백", "거울을 보고 자세를 고치는 일", "일정을 숨기는 일"),
]


def person_school(idx: int) -> Fact:
    name = NAMES[idx % len(NAMES)]
    school = SCHOOLS[idx % len(SCHOOLS)]
    neg = SCHOOLS[(idx + 3) % len(SCHOOLS)]
    return Fact(
        "person_school",
        f"{name}는 {school} 학생이다.",
        f"{name}는 어느 학교 학생이야?",
        school,
        f"{name}는 {school} 학생입니다.",
        f"{name}는 {neg} 학생이다.",
        neg,
        f"{name}는 {neg} 학생입니다.",
    )


def person_org_role(idx: int) -> Fact:
    name = NAMES[(idx + 2) % len(NAMES)]
    org = ORGS[idx % len(ORGS)]
    role = ROLES[(idx + 1) % len(ROLES)]
    neg_role = ROLES[(idx + 4) % len(ROLES)]
    return Fact(
        "person_role",
        f"{name}는 {org}에서 {role}를 맡고 있다.",
        f"{name}는 {org}에서 어떤 역할을 맡고 있어?",
        role,
        f"{name}는 {org}에서 {role}를 맡고 있습니다.",
        f"{name}는 {org}에서 {neg_role}를 맡고 있다.",
        neg_role,
        f"{name}는 {org}에서 {neg_role}를 맡고 있습니다.",
    )


def concept_definition(idx: int) -> Fact:
    concept, answer, neg_answer = CONCEPTS[idx % len(CONCEPTS)]
    return Fact(
        "concept_definition",
        f"{concept}은 {answer}이다.",
        f"{concept}은 무엇이야?",
        answer,
        f"{concept}은 {answer}입니다.",
        f"{concept}은 {neg_answer}이다.",
        neg_answer,
        f"{concept}은 {neg_answer}입니다.",
    )


def composition(idx: int) -> Fact:
    item, answer, neg_answer = COMPOSITIONS[idx % len(COMPOSITIONS)]
    return Fact(
        "composition",
        f"{item}은 {answer}로 구성된다.",
        f"{item}은 무엇으로 구성돼?",
        answer,
        f"{item}은 {answer}로 구성됩니다.",
        f"{item}은 {neg_answer}로 구성된다.",
        neg_answer,
        f"{item}은 {neg_answer}로 구성됩니다.",
    )


def cause_effect(idx: int) -> Fact:
    event, answer, neg_answer = CAUSES[idx % len(CAUSES)]
    return Fact(
        "cause",
        f"{event}는 {answer}이라고 설명했다.",
        f"{event}는 무엇 때문이야?",
        answer,
        f"{event}는 {answer}입니다.",
        f"{event}는 {neg_answer}이라고 설명했다.",
        neg_answer,
        f"{event}는 {neg_answer}입니다.",
    )


def event_time(idx: int) -> Fact:
    event, answer, neg_answer = EVENTS[idx % len(EVENTS)]
    return Fact(
        "event_time",
        f"{event} 시간은 {answer}이다.",
        f"{event} 시간은 언제야?",
        answer,
        f"{event} 시간은 {answer}입니다.",
        f"{event} 시간은 {neg_answer}이다.",
        neg_answer,
        f"{event} 시간은 {neg_answer}입니다.",
    )


def location_fact(idx: int) -> Fact:
    item = ["과제함", "미르노트", "실습 파일", "질문 카드", "회의록", "자료 묶음"][idx % 6]
    place = PLACES[idx % len(PLACES)]
    neg_place = ALT_PLACES[idx % len(ALT_PLACES)]
    return Fact(
        "location",
        f"{item} 제출 장소는 {place}이다.",
        f"{item}은 어디에 제출해?",
        place,
        f"{item} 제출 장소는 {place}입니다.",
        f"{item} 제출 장소는 {neg_place}이다.",
        neg_place,
        f"{item} 제출 장소는 {neg_place}입니다.",
    )


def analogy_fact(idx: int) -> Fact:
    concept, answer, neg_answer = ANALOGIES[idx % len(ANALOGIES)]
    speaker = ["이 교수", "민아 조교", "승민 팀장", "지영 매니저"][idx % 4]
    return Fact(
        "analogy",
        f"{speaker}는 {concept}을 쉽게 이해하도록 '{answer}'이라는 비유로 설명했다.",
        f"{speaker}가 {concept}을 어떤 비유로 설명했어?",
        answer,
        f"{speaker}는 {concept}을 '{answer}'이라는 비유로 설명했습니다.",
        f"{speaker}는 {concept}을 쉽게 이해하도록 '{neg_answer}'이라는 비유로 설명했다.",
        neg_answer,
        f"{speaker}는 {concept}을 '{neg_answer}'이라는 비유로 설명했습니다.",
    )


FACT_BUILDERS: list[Callable[[int], Fact]] = [
    person_school,
    person_org_role,
    concept_definition,
    composition,
    cause_effect,
    event_time,
    location_fact,
    analogy_fact,
]

CONTEXTS = [
    "수업 기록",
    "회의 메모",
    "학습 상담 기록",
    "프로젝트 안내",
    "질문 답변 노트",
    "운영 공지",
]


def make_final_question(row_idx: int) -> str:
    context = CONTEXTS[row_idx % len(CONTEXTS)]
    return f"이 {context}에서 확인해야 할 핵심 내용은 뭐야?"


def make_row(row_idx: int, facts_per_row: int, rng: random.Random) -> dict:
    builders = rng.sample(FACT_BUILDERS, k=min(facts_per_row, len(FACT_BUILDERS)))
    facts = [builder(row_idx + offset * 17) for offset, builder in enumerate(builders)]
    context = CONTEXTS[row_idx % len(CONTEXTS)]
    passage = f"{context}: " + " ".join(fact.sub_passage for fact in facts)
    negative_passage = f"{context}: " + " ".join(fact.negative_sub_passage for fact in facts)
    final_answer = "; ".join(fact.answer for fact in facts)
    negative_final_answer = "; ".join(fact.negative_answer for fact in facts)
    final_full = " ".join(fact.full_answer for fact in facts)
    negative_final_full = " ".join(fact.negative_full_answer for fact in facts)
    final_question = make_final_question(row_idx)

    atomic_qas = [
        {
            "sub_passage": fact.sub_passage,
            "question": fact.question,
            "answer": fact.answer,
            "full_answer": fact.full_answer,
        }
        for fact in facts
    ]
    negative_atomic_qas = [
        {
            "sub_passage": fact.negative_sub_passage,
            "question": fact.question,
            "answer": fact.negative_answer,
            "full_answer": fact.negative_full_answer,
        }
        for fact in facts
    ]
    final_qas = [{
        "sub_passage": passage,
        "question": final_question,
        "answer": final_answer,
        "full_answer": final_full,
    }]
    negative_final_qas = [{
        "sub_passage": negative_passage,
        "question": final_question,
        "answer": negative_final_answer,
        "full_answer": negative_final_full,
    }]
    return {
        "source_id": f"ko_content_synth_{row_idx:06d}",
        "task": "ko_content_service_memory",
        "content_note": "synthetic Korean content inspired by open-domain QA topics; not copied from external datasets",
        "passage": passage,
        "rewrite": passage,
        "answer": final_answer,
        "atomic_qas": atomic_qas,
        "final_qas": final_qas,
        "hard_negatives": [{
            "passage": negative_passage,
            "answer": negative_final_answer,
            "atomic_qas": negative_atomic_qas,
            "final_qas": negative_final_qas,
        }],
    }


def split_rows(rows: list[dict], valid_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    valid_n = max(1, int(round(len(shuffled) * valid_ratio)))
    return shuffled[valid_n:], shuffled[:valid_n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", default=str(KO_CONTENT_AUGMENTED_TRAIN_PATH))
    parser.add_argument("--valid-output", default=str(KO_CONTENT_AUGMENTED_VALID_PATH))
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--facts-per-row", type=int, default=3)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rows < 2:
        raise ValueError("--rows must be at least 2")
    if not 0 < args.valid_ratio < 1:
        raise ValueError("--valid-ratio must be between 0 and 1")
    rng = random.Random(args.seed)
    rows = [make_row(row_idx, args.facts_per_row, rng) for row_idx in range(args.rows)]
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed + 1)
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.valid_output, valid_rows)
    print(
        f"[PRAG:ko-content] train={len(train_rows)} -> {args.train_output}\n"
        f"[PRAG:ko-content] valid={len(valid_rows)} -> {args.valid_output}\n"
        f"[PRAG:ko-content] rows={len(rows)} facts_per_row={args.facts_per_row} "
        f"expanded_examples~train={len(train_rows) * (args.facts_per_row + 1)} "
        f"valid={len(valid_rows) * (args.facts_per_row + 1)}"
    )


if __name__ == "__main__":
    main()
