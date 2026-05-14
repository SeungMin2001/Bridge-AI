"""Build hard distractor related-merge PRAG test rows.

The schema matches the existing augmented JSONL format, but each row contains
multiple related facts with the same answer type. This makes the merge memory
need to select the question-relevant fact instead of simply recalling any
obvious answer from the passage group.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SCENARIOS = [
    {
        "domain": "infra_schedule",
        "topic": "인프라 회의",
        "answer_type": "date",
        "facts": [
            ("배포 일정", "다음 주 수요일 오전"),
            ("점검 일정", "금요일 저녁"),
            ("롤백 리허설 일정", "월요일 오후"),
            ("모니터링 회고 일정", "목요일 오전"),
        ],
    },
    {
        "domain": "project_owner",
        "topic": "프로젝트 회의",
        "answer_type": "owner",
        "facts": [
            ("자료 업로드 담당자", "프론트엔드 팀"),
            ("실시간 요약 담당자", "AI 파트"),
            ("서버 배포 담당자", "백엔드 팀"),
            ("최종 검토 담당자", "기획 파트"),
        ],
    },
    {
        "domain": "assignment_policy",
        "topic": "과제 안내",
        "answer_type": "policy",
        "facts": [
            ("제출 기한", "다음 주 월요일 밤"),
            ("제출 위치", "이캠퍼스 과제함"),
            ("감점 기준", "지각 제출"),
            ("평가 기준", "근거 제시"),
        ],
    },
    {
        "domain": "marketing_metric",
        "topic": "마케팅 회의",
        "answer_type": "metric",
        "facts": [
            ("홍보 채널", "오픈채팅방"),
            ("타깃 고객", "신입생"),
            ("성과 지표", "전환율"),
            ("참여 보상", "굿즈 추첨권"),
        ],
    },
    {
        "domain": "security_response",
        "topic": "보안 회의",
        "answer_type": "security",
        "facts": [
            ("인증 방식", "이중 인증"),
            ("점검 대상", "관리자 계정"),
            ("위험 신호", "비정상 로그인"),
            ("대응 담당", "보안 팀"),
        ],
    },
    {
        "domain": "nlp_analogy",
        "topic": "자연어처리 수업",
        "answer_type": "analogy",
        "facts": [
            ("토큰화 비유", "문장을 작은 블록으로 나누는 과정"),
            ("임베딩 비유", "단어를 좌표 위에 올려두는 방식"),
            ("어텐션 비유", "중요한 단어에 형광펜을 치는 방식"),
            ("버전 관리 비유", "작업 과정을 남기는 연구 노트"),
        ],
    },
]


CONNECTORS = [
    "같은 회의에서 혼동하지 않도록",
    "비슷한 항목이 함께 언급되었지만",
    "교수님은 여러 후보를 나란히 설명하면서",
    "회의록에는 서로 다른 항목이 연속으로 정리되었고",
]

TOPIC_PREFIXES = [
    "1차",
    "2차",
    "3차",
    "긴급",
    "정기",
    "오전",
    "오후",
    "월간",
    "주간",
    "최종",
    "중간",
    "파일럿",
]

GROUP_NOUNS = [
    "검토",
    "정리",
    "점검",
    "운영",
    "개선",
    "실험",
    "리허설",
    "공유",
]

KOREAN_NUMBERS = [
    "첫 번째",
    "두 번째",
    "세 번째",
    "네 번째",
    "다섯 번째",
    "여섯 번째",
    "일곱 번째",
    "여덟 번째",
    "아홉 번째",
    "열 번째",
]

QUESTION_OPENERS = [
    "{topic}에서 {key}은 무엇이었어?",
    "{topic}에서 말한 {key}을 알려줘.",
    "교수님이 {topic}에서 {key}을 뭐라고 했어?",
    "{topic} 내용 중 {key}에 해당하는 답은 뭐야?",
    "{topic} 메모에서 {key}으로 적힌 내용은 뭐야?",
    "{topic} 설명을 기준으로 {key}에 맞는 값은 뭐야?",
    "{topic}에서 다른 항목 말고 {key}만 물으면 답이 뭐야?",
    "{topic}에서 {key} 항목의 정답만 말해줘.",
]

DATE_QUESTION_OPENERS = [
    "{topic}에서 {key}은 언제였어?",
    "{topic}에서 말한 {key} 날짜를 알려줘.",
    "교수님이 {topic}에서 {key}을 언제라고 했어?",
    "{topic} 내용 중 {key}에 해당하는 시점은 언제야?",
    "{topic} 기록에서 {key}으로 적힌 날짜는 언제야?",
    "{topic}에서 다른 일정 말고 {key}만 보면 언제야?",
    "{topic} 메모 기준 {key}의 시간만 답해줘.",
    "{topic}에서 {key} 항목의 날짜는 뭐야?",
]

OWNER_QUESTION_OPENERS = [
    "{topic}에서 {key}은 누구였어?",
    "{topic}에서 말한 {key} 주체를 알려줘.",
    "교수님이 {topic}에서 {key}을 누구라고 했어?",
    "{topic} 내용 중 {key}에 해당하는 팀은 어디야?",
    "{topic} 기록에서 {key}으로 적힌 담당은 누구야?",
    "{topic}에서 다른 담당 말고 {key}만 보면 누구야?",
    "{topic} 메모 기준 {key}의 담당자만 답해줘.",
    "{topic}에서 {key} 항목의 주체는 누구야?",
]

LOCATION_QUESTION_OPENERS = [
    "{topic}에서 {key}은 어디였어?",
    "{topic}에서 말한 {key} 장소를 알려줘.",
    "교수님이 {topic}에서 {key}을 어디라고 했어?",
    "{topic} 내용 중 {key}에 해당하는 위치는 어디야?",
    "{topic} 기록에서 {key}으로 적힌 장소는 어디야?",
    "{topic}에서 다른 위치 말고 {key}만 보면 어디야?",
    "{topic} 메모 기준 {key}의 위치만 답해줘.",
    "{topic}에서 {key} 항목의 장소는 어디야?",
]


def question_for(topic: str, key: str, *, paraphrase_idx: int) -> str:
    variants = QUESTION_OPENERS
    if "일정" in key or "기한" in key:
        variants = DATE_QUESTION_OPENERS
    elif "담당" in key or "고객" in key or "대상" in key:
        variants = OWNER_QUESTION_OPENERS
    elif "위치" in key:
        variants = LOCATION_QUESTION_OPENERS
    return variants[paraphrase_idx % len(variants)].format(topic=topic, key=key)


def sentence_for(topic: str, key: str, value: str, rng: random.Random) -> str:
    connector = rng.choice(CONNECTORS)
    templates = [
        f"{connector}, {topic}에서 {key}은 {value}로 정리되었습니다.",
        f"{topic} 설명에서는 {key}을 {value}라고 분명히 구분했습니다.",
        f"{topic} 기록에는 {key}이 {value}라고 따로 적혀 있었습니다.",
        f"{topic}에서는 {key}만 따로 보면 답이 {value}라고 안내되었습니다.",
    ]
    return rng.choice(templates)


def full_answer(key: str, value: str) -> str:
    return f"{key}은 {value}입니다."


def build_row(index: int, rng: random.Random, source_prefix: str) -> dict:
    scenario = rng.choice(SCENARIOS)
    topic_prefix = TOPIC_PREFIXES[index % len(TOPIC_PREFIXES)]
    group_noun = GROUP_NOUNS[(index // len(TOPIC_PREFIXES)) % len(GROUP_NOUNS)]
    round_name = KOREAN_NUMBERS[index % len(KOREAN_NUMBERS)]
    topic = f"{topic_prefix} {scenario['topic']} {group_noun}"
    facts = list(scenario["facts"])
    rng.shuffle(facts)
    selected = facts[:4]
    atomic_qas = []
    for fact_idx, (key, value) in enumerate(selected):
        sub_passage = sentence_for(topic, key, value, rng)
        question_index = index * len(selected) + fact_idx
        atomic_qas.append({
            "sub_passage": sub_passage,
            "question": f"{round_name}로 정리된 {question_for(topic, key, paraphrase_idx=question_index)}",
            "answer": value,
            "full_answer": full_answer(key, value),
        })
    passage = " ".join(qa["sub_passage"] for qa in atomic_qas)
    final_answer = " ".join(qa["full_answer"] for qa in atomic_qas)
    return {
        "source_id": f"{source_prefix}_{scenario['domain']}_{index:03d}",
        "language": "ko",
        "clean_ko": True,
        "hard_distractor": True,
        "passage": passage,
        "rewrite": passage,
        "atomic_qas": atomic_qas,
        "final_qas": [{
            "question": f"{topic}에서 헷갈리지 말아야 할 핵심 항목들은 무엇이었어?",
            "answer": final_answer,
            "full_answer": final_answer,
        }],
        "hard_negatives": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/PRAG_related_merge_hard_test.jsonl")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--source-prefix", default="related_hard_test")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [build_row(i, rng, args.source_prefix) for i in range(args.count)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[PRAG:related-hard-test] wrote rows={len(rows)} path={output}")


if __name__ == "__main__":
    main()
