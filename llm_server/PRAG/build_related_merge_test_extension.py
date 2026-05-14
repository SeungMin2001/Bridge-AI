"""Build additional related-merge PRAG test rows.

This is used when the existing related test set is already trained/evaluated
around 100 rows and we want to append more held-out rows with the same schema
and the same "one coherent topic with related passages" structure.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


DOMAINS = {
    "infra": {
        "topic": "인프라 회의",
        "items": [
            ("배포 일정", ["다음 주 수요일 오전", "금요일 저녁", "월말 점검일", "둘째 주 화요일"]),
            ("롤백 담당자", ["플랫폼 팀", "백엔드 파트", "운영 담당자", "인프라 2팀"]),
            ("모니터링 지표", ["오류율", "응답 시간", "CPU 사용률", "완료율"]),
            ("장애 기록 위치", ["회의록 폴더", "운영 로그", "장애 대응 문서", "공유 드라이브"]),
        ],
    },
    "marketing": {
        "topic": "마케팅 회의",
        "items": [
            ("홍보 채널", ["오픈채팅방", "인스타그램", "학과 게시판", "이메일 뉴스레터"]),
            ("타깃 고객", ["재학생 전체", "신입생", "동아리 운영진", "졸업 예정자"]),
            ("성과 지표", ["전환율", "참여율", "클릭률", "완료율"]),
            ("참여 보상", ["굿즈 추첨권", "포인트 500점", "커피 쿠폰", "추가 응모권"]),
        ],
    },
    "nlp": {
        "topic": "자연어처리 수업",
        "items": [
            ("토큰화 비유", ["문장을 작은 블록으로 나누는 과정", "레고 조각을 분리하는 일", "문장 재료를 손질하는 단계"]),
            ("임베딩 설명", ["단어의 의미 위치를 숫자로 표현하는 방법", "단어를 좌표 위에 올려두는 방식", "의미를 벡터 공간에 놓는 과정"]),
            ("모델 평가 기준", ["정확도", "재현율", "토큰 F1", "응답 일관성"]),
            ("주의해야 할 오류", ["환각 답변", "문맥 누락", "질문 오해", "근거 없는 확정 표현"]),
        ],
    },
    "project": {
        "topic": "프로젝트 회의",
        "items": [
            ("우선 구현 기능", ["자료 업로드", "실시간 요약", "AI 채팅", "퀴즈 생성"]),
            ("검토 담당자", ["기획 파트", "프론트엔드 팀", "백엔드 팀", "AI 파트"]),
            ("마감 기준", ["이번 주 금요일", "다음 주 월요일", "중간 발표 전날", "최종 제출 하루 전"]),
            ("위험 요소", ["일정 지연", "데이터 부족", "응답 속도 저하", "서버 연결 오류"]),
        ],
    },
    "assignment": {
        "topic": "과제 안내",
        "items": [
            ("제출 기한", ["다음 주 월요일 밤", "이번 주 금요일 오후", "수요일 자정", "발표 전날"]),
            ("제출 위치", ["이캠퍼스 과제함", "공유 드라이브", "강의 자료실", "팀 채널"]),
            ("평가 기준", ["근거 제시", "완성도", "분석 깊이", "제출 형식 준수"]),
            ("감점 사유", ["지각 제출", "출처 누락", "파일 형식 오류", "분량 부족"]),
        ],
    },
    "medicine": {
        "topic": "의학 수업",
        "items": [
            ("혈압 측정 자세", ["앉은 자세", "팔을 심장 높이에 둔 자세", "안정된 자세", "등을 기대는 자세"]),
            ("응급처치 순서", ["가슴압박 우선", "호흡 확인", "도움 요청", "자동심장충격기 준비"]),
            ("관리 지표", ["혈당", "체온", "맥박", "산소포화도"]),
            ("주의 증상", ["호흡 곤란", "의식 저하", "가슴 통증", "지속되는 어지럼"]),
        ],
    },
    "security": {
        "topic": "보안 회의",
        "items": [
            ("인증 방식", ["이중 인증", "일회용 비밀번호", "생체 인증", "보안 키"]),
            ("점검 대상", ["관리자 계정", "외부 공유 링크", "접속 로그", "권한 설정"]),
            ("위험 신호", ["비정상 로그인", "반복된 실패 기록", "낯선 위치 접속", "권한 상승 요청"]),
            ("대응 담당", ["보안 팀", "인프라 2팀", "운영 관리자", "서비스 담당자"]),
        ],
    },
    "community": {
        "topic": "커뮤니티 회의",
        "items": [
            ("공지 우선순위", ["일정 변경", "서비스 점검", "이벤트 안내", "규칙 개정"]),
            ("신고 처리 기준", ["욕설 여부", "반복 게시", "개인정보 노출", "광고성 글"]),
            ("이벤트 보상", ["포인트 500점", "굿즈 추첨권", "커피 쿠폰", "추가 배지"]),
            ("운영 원칙", ["빠른 안내", "공정한 처리", "근거 기록", "사용자 보호"]),
        ],
    },
}


def full_answer(key: str, value: str) -> str:
    return f"{key}은 {value}입니다."


def question_for(key: str, topic: str) -> str:
    if "기한" in key or "일정" in key or "시간" in key:
        return f"{topic}에서 {key}은 언제였어?"
    if "담당" in key or "대상" in key or "고객" in key:
        return f"{topic}에서 {key}은 누구였어?"
    if "위치" in key:
        return f"{topic}에서 {key}은 어디였어?"
    return f"{topic}에서 {key}은 무엇이었어?"


def sub_passage_for(topic: str, key: str, value: str, rng: random.Random) -> str:
    templates = [
        f"{topic}에서는 {key}을 {value}로 정리했습니다.",
        f"{topic}에서 {key}은 {value}라고 설명했습니다.",
        f"{topic}의 핵심 내용 중 {key}은 {value}로 안내되었습니다.",
        f"{topic}에서는 학생들이 기억해야 할 {key}을 {value}로 강조했습니다.",
    ]
    return rng.choice(templates)


def build_row(index: int, rng: random.Random, prefix: str) -> dict:
    domain_name = rng.choice(sorted(DOMAINS))
    domain = DOMAINS[domain_name]
    topic = domain["topic"]
    facts = rng.sample(domain["items"], 3)
    atomic_qas = []
    for key, values in facts:
        value = rng.choice(values)
        sub_passage = sub_passage_for(topic, key, value, rng)
        atomic_qas.append({
            "sub_passage": sub_passage,
            "question": question_for(key, topic),
            "answer": value,
            "full_answer": full_answer(key, value),
        })
    passage = " ".join(qa["sub_passage"] for qa in atomic_qas)
    final_answer = " ".join(qa["full_answer"] for qa in atomic_qas)
    return {
        "source_id": f"{prefix}_{domain_name}_{index:03d}",
        "language": "ko",
        "clean_ko": True,
        "passage": passage,
        "rewrite": passage,
        "atomic_qas": atomic_qas,
        "final_qas": [{
            "question": f"{topic}에서 정리한 핵심 내용은 무엇이었어?",
            "answer": final_answer,
            "full_answer": final_answer,
        }],
        "hard_negatives": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/PRAG_related_merge_test_extra300.jsonl")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--source-prefix", default="related_extra_test")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [build_row(i, rng, args.source_prefix) for i in range(args.count)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[PRAG:related-test-extension] wrote rows={len(rows)} path={output}")


if __name__ == "__main__":
    main()
