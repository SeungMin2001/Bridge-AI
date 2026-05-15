"""Build broad-question entity multi-fact PRAG datasets.

Each row describes one entity/topic with several related natural-language
passages. The atomic QA entries are evidence-only, while the trainable final
QAs ask broad questions such as "초콜렛을 뭐라고 설명했어?" and require the
model to recover multiple facts from the merged memory.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .data import write_jsonl


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_PREFIX = "PRAG_entity_multifact"


ENTITY_BANK = [
    ("dessert", "초콜렛", [("색깔", "검정색"), ("맛", "달콤하다"), ("비유", "작은 선물 상자"), ("특징", "입안에서 천천히 녹는다")]),
    ("dessert", "솜사탕", [("색깔", "분홍색"), ("맛", "가볍게 달콤하다"), ("비유", "구름 같은 간식"), ("특징", "입에 닿으면 빠르게 사라진다")]),
    ("dessert", "마카롱", [("색깔", "알록달록하다"), ("맛", "부드럽고 진하다"), ("비유", "작은 보석 상자"), ("특징", "겉은 바삭하고 속은 촉촉하다")]),
    ("dessert", "브라우니", [("색깔", "진한 갈색"), ("맛", "묵직하게 달콤하다"), ("비유", "초콜렛을 압축한 케이크"), ("특징", "속이 촉촉하게 남는다")]),
    ("os_concept", "프로세스", [("역할", "독립된 실행 공간을 가진다"), ("비유", "독립된 식당"), ("장점", "서로 영향을 줄이는 데 도움이 된다"), ("주의점", "자원을 너무 많이 쓰면 느려질 수 있다")]),
    ("os_concept", "스레드", [("역할", "프로세스 안에서 작업을 나누어 실행한다"), ("비유", "같은 식당 안에서 일하는 직원"), ("장점", "작업을 가볍게 나눌 수 있다"), ("주의점", "공유 자원 관리가 필요하다")]),
    ("os_concept", "캐시", [("역할", "자주 쓰는 정보를 가까이 둔다"), ("비유", "책상 위에 올려둔 책"), ("장점", "반복 접근을 빠르게 만든다"), ("주의점", "오래된 정보가 남을 수 있다")]),
    ("os_concept", "큐", [("역할", "작업을 순서대로 기다리게 한다"), ("비유", "번호표를 뽑고 기다리는 줄"), ("장점", "처리 순서를 명확하게 만든다"), ("주의점", "대기 시간이 길어질 수 있다")]),
    ("marketing", "신입생 캠페인", [("핵심 채널", "인스타그램"), ("대상", "신입생"), ("성과 지표", "전환율"), ("보상", "커피 쿠폰")]),
    ("marketing", "동아리 홍보", [("핵심 채널", "오픈채팅방"), ("대상", "동아리 회원"), ("성과 지표", "참여율"), ("보상", "굿즈 추첨권")]),
    ("marketing", "스터디 모집", [("핵심 채널", "학과 홈페이지"), ("대상", "스터디 신청자"), ("성과 지표", "완료율"), ("보상", "우선 신청권")]),
    ("marketing", "멘토링 이벤트", [("핵심 채널", "이메일 뉴스레터"), ("대상", "멘토링 참여 학생"), ("성과 지표", "재방문율"), ("보상", "포인트 500점")]),
    ("project", "최종 발표", [("담당", "기획 파트"), ("마감", "다음 주 월요일 밤"), ("확인 기준", "근거 제시 여부"), ("공유 위치", "팀 노션 페이지")]),
    ("project", "백엔드 배포", [("담당", "백엔드 팀"), ("마감", "수요일 오후 다섯 시"), ("확인 기준", "오류 발생률"), ("공유 위치", "깃허브 이슈 탭")]),
    ("project", "AI 모델 평가", [("담당", "AI 파트"), ("마감", "금요일 자정 전"), ("확인 기준", "정답률"), ("공유 위치", "회의록 폴더")]),
    ("project", "프론트엔드 점검", [("담당", "프론트엔드 파트"), ("마감", "다음 달 첫째 주 화요일"), ("확인 기준", "사용자 차단 여부"), ("공유 위치", "프로젝트 슬랙 채널")]),
    ("art_history", "바로크", [("핵심 특징", "극적 명암"), ("비유", "무대 조명이 한 장면을 강조하는 방식"), ("목적", "감정을 강하게 전달하는 것"), ("비교 포인트", "평면적 구도와 구분된다")]),
    ("art_history", "입체주의", [("핵심 특징", "여러 시점"), ("비유", "여러 방향에서 본 사물을 한 화면에 놓는 방식"), ("목적", "대상을 새롭게 해석하는 것"), ("비교 포인트", "단일 원근과 다르다")]),
    ("art_history", "인상주의", [("핵심 특징", "빛의 순간적인 인상"), ("비유", "햇빛이 흔들리는 수면에 비친 모습"), ("목적", "관찰 순간을 살리는 것"), ("비교 포인트", "정확한 윤곽보다 색감이 중요하다")]),
    ("art_history", "르네상스", [("핵심 특징", "인문주의"), ("비유", "사람을 화면 중앙에 다시 세우는 방식"), ("목적", "사람의 가치를 드러내는 것"), ("비교 포인트", "절대왕정 중심 설명과 다르다")]),
    ("statistics", "표본분산", [("의미", "흩어짐 측정"), ("비유", "점들이 얼마나 퍼졌는지 보는 자"), ("주의점", "순위 정렬과 다르다"), ("활용", "데이터 변동성을 볼 때 쓴다")]),
    ("statistics", "신뢰구간", [("의미", "모수 추정 범위"), ("비유", "정답이 있을 법한 범위를 그은 지도"), ("주의점", "정답 하나를 보장하지 않는다"), ("활용", "추정의 불확실성을 설명할 때 쓴다")]),
    ("statistics", "상관계수", [("의미", "선형 관계 강도"), ("비유", "두 변수의 움직임을 비교하는 나침반"), ("주의점", "원인 증명을 뜻하지 않는다"), ("활용", "관계 강도를 빠르게 살필 때 쓴다")]),
    ("statistics", "회귀분석", [("의미", "관계의 경향 설명"), ("비유", "흩어진 점 사이에 길을 그어 보는 일"), ("주의점", "모든 원인을 증명하지는 않는다"), ("활용", "예측 경향을 설명할 때 쓴다")]),
]

QUESTION_TEMPLATES = [
    "{entity}에 대해 뭐라고 설명하셨어?",
    "교수님이 {entity}에 대해 정리한 내용을 말해줘.",
    "{entity} 설명에서 기억해야 할 내용은 뭐야?",
    "{entity}의 특징을 어떻게 설명했어?",
]

SENTENCE_TEMPLATES = [
    "{speaker} 설명에서 {entity}의 {slot} 항목은 {quoted_value} 정리되었습니다.",
    "수업에서는 {entity}라는 주제를 다루며 {slot} 항목을 {quoted_value} 강조했습니다.",
    "학생들이 헷갈리지 않도록 {entity}의 {slot} 항목은 {quoted_value} 설명되었습니다.",
    "회의 기록에는 {entity}와 관련한 {slot} 항목이 {quoted_value} 적혔습니다.",
]

SPEAKERS = ["김 교수님", "박 교수님", "이 조교", "한 강사", "정 매니저", "최 팀장", "서 연구원"]
SETTINGS = ["기초 설명", "심화 설명", "오늘 수업", "지난 시간 복습", "프로젝트 회의", "서비스 회의", "중간 점검", "최종 정리"]


def has_batchim(text: str) -> bool:
    for ch in reversed(str(text or "").strip()):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            return (code - 0xAC00) % 28 != 0
    return False


def quoted(value: str) -> str:
    value = str(value or "").strip()
    if value.endswith("다"):
        return f"{value[:-1]}다고"
    return f"{value}{'이라고' if has_batchim(value) else '라고'}"


def build_answer(entity: str, selected_facts: list[tuple[str, str]]) -> str:
    parts = [f"{slot} 항목은 {quoted(value)}" for slot, value in selected_facts]
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + f", 그리고 {parts[-1]}"
    return f"{entity}에 대해 {body} 설명했습니다."


def build_row(index: int, rng: random.Random, *, source_prefix: str) -> dict:
    domain, base_entity, facts = rng.choice(ENTITY_BANK)
    entity = f"{base_entity} {index + 1}번 사례"
    setting = SETTINGS[index % len(SETTINGS)]
    speaker = rng.choice(SPEAKERS)

    selected_facts = list(facts)
    rng.shuffle(selected_facts)
    selected_facts = selected_facts[:4]

    atomic_qas = []
    passages = []
    for fact_idx, (slot, value) in enumerate(selected_facts):
        template = SENTENCE_TEMPLATES[(index + fact_idx) % len(SENTENCE_TEMPLATES)]
        sub_passage = template.format(
            speaker=speaker,
            entity=entity,
            slot=slot,
            value=value,
            quoted_value=quoted(value),
        )
        passages.append(sub_passage)
        atomic_qas.append(
            {
                "sub_passage": sub_passage,
                # Evidence-only: loaders skip these when entity_multifact=True.
                "question": f"{entity}의 {slot}은 무엇이라고 했어?",
                "answer": value,
                "full_answer": f"{entity}의 {slot} 항목은 {value}입니다.",
            }
        )

    passage = " ".join(passages)
    answer = build_answer(entity, selected_facts)
    final_qas = [
        {
            "question": template.format(entity=entity),
            "answer": answer,
            "full_answer": answer,
        }
        for template in QUESTION_TEMPLATES
    ]
    return {
        "source_id": f"{source_prefix}_{domain}_{index:04d}",
        "language": "ko",
        "clean_ko": True,
        "entity_multifact": True,
        "atomic_qas_as_evidence_only": True,
        "setting": setting,
        "entity": entity,
        "passage": passage,
        "rewrite": passage,
        "atomic_qas": atomic_qas,
        "final_qas": final_qas,
        "hard_negatives": [],
    }


def split_rows(rows: list[dict], train_count: int, valid_count: int, test_count: int) -> tuple[list[dict], list[dict], list[dict]]:
    train = rows[:train_count]
    valid = rows[train_count: train_count + valid_count]
    test = rows[train_count + valid_count: train_count + valid_count + test_count]
    return train, valid, test


def write_summary(path: Path, rows: list[dict]) -> None:
    atomic = sum(len(row.get("atomic_qas") or []) for row in rows)
    final = sum(len(row.get("final_qas") or []) for row in rows)
    print(
        f"[PRAG:entity-multifact] {path.name}: rows={len(rows)} "
        f"evidence_atomic={atomic} trainable_final={final}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-count", type=int, default=3200)
    parser.add_argument("--valid-count", type=int, default=400)
    parser.add_argument("--test-count", type=int, default=400)
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--source-prefix", default="entity_multifact")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    total = args.train_count + args.valid_count + args.test_count
    rng = random.Random(args.seed)
    rows = [build_row(i, rng, source_prefix=args.source_prefix) for i in range(total)]
    if args.shuffle:
        rng.shuffle(rows)

    train, valid, test = split_rows(rows, args.train_count, args.valid_count, args.test_count)
    output_dir = Path(args.output_dir)
    train_path = output_dir / f"{args.prefix}_train.jsonl"
    valid_path = output_dir / f"{args.prefix}_valid.jsonl"
    test_path = output_dir / f"{args.prefix}_test.jsonl"

    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)
    write_jsonl(test_path, test)

    print(
        "[PRAG:entity-multifact] wrote broad-question entity multi-fact splits "
        f"train={args.train_count} valid={args.valid_count} test={args.test_count}"
    )
    write_summary(train_path, train)
    write_summary(valid_path, valid)
    write_summary(test_path, test)


if __name__ == "__main__":
    main()
