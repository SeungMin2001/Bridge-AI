"""Build a natural-language PRAG test set for final memory evaluation.

The existing multifact data intentionally contains compact relation/value
patterns. This generator creates a held-out style test set where every passage
and evidence span is written as natural Korean sentences instead of using
template markers such as "A: B" or "A=B".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import write_jsonl


DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "PRAG_natural_service_test_450.jsonl"

SPEAKERS = [
    "김 교수님",
    "박 교수님",
    "이 조교",
    "최 팀장",
    "정 매니저",
    "한 강사",
    "서 연구원",
    "민 코치",
    "지영 매니저",
    "승민 팀장",
]

TIMES = [
    "다음 주 월요일 밤",
    "수요일 오후 다섯 시",
    "금요일 자정 전",
    "이번 주 토요일 오전",
    "다음 달 첫째 주 화요일",
    "오늘 수업이 끝난 뒤",
    "중간고사 전날 오후",
    "프로젝트 회의 다음 날",
    "월말 점검 직후",
    "다음 주 목요일 아침",
]

LOCATIONS = [
    "이캠퍼스 자료실",
    "팀 노션 페이지",
    "깃허브 이슈 탭",
    "학과 공지 게시판",
    "구글 드라이브 공유 폴더",
    "프로젝트 슬랙 채널",
    "강의실 앞 제출함",
    "서비스 관리자 페이지",
    "회의록 폴더",
    "운영 대시보드",
]

OWNERS = [
    "QA 1팀",
    "인프라 2팀",
    "프론트엔드 파트",
    "백엔드 파트",
    "데이터 수집팀",
    "운영 지원팀",
    "보안 담당자",
    "수업 조교",
    "기획 파트",
    "디자인 파트",
]

ANALOGIES = [
    "독립된 식당",
    "책상 위에 올려둔 자주 쓰는 책",
    "도서관의 색인 카드",
    "택배 분류 벨트",
    "회의실 예약표",
    "주방의 주문서",
    "지도 위의 길찾기 표시",
    "냉장고의 자주 여는 칸",
    "공연장의 좌석 안내도",
    "은행 창구의 번호표",
]

CHANNELS = [
    "인스타그램",
    "카카오톡 채널",
    "학과 홈페이지",
    "이메일 뉴스레터",
    "유튜브 커뮤니티",
    "교내 포스터",
    "디스코드 공지방",
    "앱 푸시 알림",
    "블로그 공지",
    "오픈채팅방",
]

AUDIENCES = [
    "신입생",
    "재학생 전체",
    "프로젝트 참여자",
    "조교진",
    "운영진",
    "스터디 신청자",
    "동아리 회원",
    "수강 정정 학생",
    "멘토링 참여 학생",
    "현장 실습생",
]

METRICS = [
    "전환율",
    "응답 시간",
    "정답률",
    "완료율",
    "재방문율",
    "오류 발생률",
    "참여율",
    "처리 시간",
    "만족도",
    "클릭률",
]

REWARDS = [
    "포인트 500점",
    "추가 출석 점수",
    "커피 쿠폰 한 장",
    "과제 가산점 2점",
    "스터디 우선 신청권",
    "굿즈 추첨권",
    "피드백 우선권",
    "자료 열람권",
    "멘토링 우선권",
    "팀 배지",
]

CRITERIA = [
    "사용자 차단 여부",
    "욕설 포함 여부",
    "개인정보 노출 여부",
    "재현 가능성",
    "장애 지속 시간",
    "학습 목표와의 관련성",
    "중복 제보 여부",
    "보안 위험도",
    "제출 기한 준수 여부",
    "핵심 기능 영향도",
]

CONCEPTS = [
    ("프로세스", "독립된 식당", "스레드", "같은 식당 안에서 일하는 직원들"),
    ("캐시", "책상 위에 올려둔 자주 쓰는 책", "디스크", "창고 깊숙이 보관한 상자"),
    ("트랜잭션", "은행 창구의 한 번에 끝나는 업무 처리", "롤백", "잘못 적은 장부를 이전 상태로 되돌리는 일"),
    ("인덱스", "도서관의 색인 카드", "전체 탐색", "책장을 처음부터 끝까지 훑는 일"),
    ("어텐션", "중요한 문장에 형광펜을 긋는 과정", "임베딩", "단어의 의미 좌표"),
    ("정규화", "반복된 내용을 정리해 서랍을 나누는 일", "조인", "나뉜 서랍의 정보를 다시 맞추는 일"),
    ("로드밸런서", "손님을 빈 계산대로 안내하는 직원", "서버", "계산대"),
    ("큐", "번호표를 뽑고 순서대로 기다리는 줄", "스택", "위에 놓은 접시부터 꺼내는 더미"),
    ("버전 관리", "작업 과정을 남기는 연구 노트", "브랜치", "실험을 따로 진행하는 작업대"),
    ("컨테이너", "필요한 도구를 담은 이동식 작업 상자", "이미지", "작업 상자를 만들기 위한 설계도"),
]

SUBJECT_FACTS = [
    ("표본분산의 목적", "흩어짐 측정", "순위 정렬"),
    ("신뢰구간의 의미", "모수 추정 범위", "정답 하나"),
    ("상관계수의 의미", "선형 관계 강도", "원인 증명"),
    ("색채 대비의 목적", "시선 집중", "명암 제거"),
    ("바로크의 특징", "극적 명암", "평면적 구도"),
    ("입체주의의 특징", "여러 시점", "단일 원근"),
    ("대체재의 예시", "콜라와 사이다", "노트북과 충전기"),
    ("수요 법칙의 핵심", "가격이 오르면 수요량이 줄어드는 경향", "가격과 수요량이 함께 오르는 경향"),
    ("혈압 측정 자세", "앉은 자세", "누운 자세만 허용"),
    ("심폐소생술 순서", "가슴압박 우선", "호흡 확인만 반복"),
]


def pick(items: list[str], variant: int, shift: int = 0) -> str:
    return items[(variant + shift) % len(items)]


def jongseong(text: str) -> int:
    for ch in reversed(str(text or "").strip()):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            return (code - 0xAC00) % 28
        if ch.isdigit():
            return 1
    return 0


def has_batchim(text: str) -> bool:
    return jongseong(text) != 0


def eul(text: str) -> str:
    return "을" if has_batchim(text) else "를"


def eun(text: str) -> str:
    return "은" if has_batchim(text) else "는"


def iga(text: str) -> str:
    return "이" if has_batchim(text) else "가"


def euro(text: str) -> str:
    jong = jongseong(text)
    return "로" if jong == 0 or jong == 8 else "으로"


def iraneun(text: str) -> str:
    return "이라는" if has_batchim(text) else "라는"


def qa(question: str, answer: str, full_answer: str, sub_passage: str) -> dict:
    return {
        "sub_passage": sub_passage,
        "question": question,
        "answer": answer,
        "full_answer": full_answer,
    }


def make_row(source_id: str, passage: str, atomic_qas: list[dict], final_question: str, final_answer: str, final_full: str) -> dict:
    final_full = final_full if final_answer in final_full else f"{final_answer} {final_full}"
    return {
        "source_id": source_id,
        "passage": passage,
        "rewrite": passage,
        "atomic_qas": atomic_qas,
        "final_qas": [{
            "question": final_question,
            "answer": final_answer,
            "full_answer": final_full,
        }],
        "hard_negatives": [],
    }


def row_assignment(variant: int) -> dict:
    speaker = pick(SPEAKERS, variant)
    deadline = pick(TIMES, variant)
    location = pick(LOCATIONS, variant, 2)
    criterion = pick(CRITERIA, variant, 4)
    s1 = f"{speaker}은 이번 과제 제출 기한을 {deadline}{euro(deadline)} 안내했습니다."
    s2 = f"제출 장소는 {location}{euro(location)} 정해졌다고 말했습니다."
    s3 = f"늦은 제출을 판단할 때는 {criterion}{eul(criterion)} 함께 확인하겠다고 덧붙였습니다."
    passage = f"{s1} {s2} {s3} 학생들이 헷갈리지 않도록 마감 시간과 제출 위치를 한 번 더 풀어서 설명한 내용이었습니다."
    atomic = [
        qa("이번 과제 제출 기한은 언제라고 했어?", deadline, f"이번 과제 제출 기한은 {deadline}입니다.", s1),
        qa("과제 제출 장소는 어디라고 안내했어?", location, f"과제 제출 장소는 {location}입니다.", s2),
        qa("늦은 제출 판단 기준은 무엇이라고 했어?", criterion, f"늦은 제출 판단 기준은 {criterion}입니다.", s3),
    ]
    return make_row(
        f"natural_test_assignment_{variant:03d}",
        passage,
        atomic,
        "과제 안내에서 교수님이 정리한 핵심은 무엇이었어?",
        f"제출 기한은 {deadline}이고 제출 장소는 {location}이며 판단 기준은 {criterion}입니다.",
        f"과제 안내의 핵심은 제출 기한이 {deadline}이고 제출 장소가 {location}이며 판단 기준이 {criterion}{iraneun(criterion)} 점입니다.",
    )


def row_qa_meeting(variant: int) -> dict:
    owner = pick(OWNERS, variant)
    scope = "로그인과 채팅" if variant % 3 == 0 else "검색과 파일 업로드" if variant % 3 == 1 else "결제와 알림"
    criterion = pick(CRITERIA, variant)
    s1 = f"QA 회의에서 테스트 담당자는 {owner}{euro(owner)} 정해졌습니다."
    s2 = f"이번 회귀 테스트 범위는 {scope} 기능을 중심으로 보겠다고 설명했습니다."
    s3 = f"버그 우선순위는 {criterion}{eul(criterion)} 먼저 따져서 정하기로 했습니다."
    passage = f"{s1} {s2} {s3} 회의 참석자들은 이 기준을 다음 배포 전까지 공통 규칙으로 쓰기로 했습니다."
    atomic = [
        qa("QA 회의에서 테스트 담당자는 누구야?", owner, f"테스트 담당자는 {owner}입니다.", s1),
        qa("회귀 테스트 범위는 무엇이라고 했어?", scope, f"회귀 테스트 범위는 {scope}입니다.", s2),
        qa("버그 우선순위 기준은 무엇인가?", criterion, f"버그 우선순위 기준은 {criterion}입니다.", s3),
    ]
    return make_row(
        f"natural_test_qa_meeting_{variant:03d}",
        passage,
        atomic,
        "QA 회의에서 테스트 운영 기준을 어떻게 정했어?",
        f"테스트 담당자는 {owner}이고 회귀 테스트 범위는 {scope}이며 버그 우선순위 기준은 {criterion}입니다.",
        f"QA 회의에서는 {owner}{iga(owner)} 테스트를 맡고 {scope} 기능을 회귀 테스트하며 {criterion}{eul(criterion)} 기준으로 버그 우선순위를 정하기로 했습니다.",
    )


def row_marketing(variant: int) -> dict:
    channel = pick(CHANNELS, variant)
    audience = pick(AUDIENCES, variant, 2)
    metric = pick(METRICS, variant, 4)
    reward = pick(REWARDS, variant, 6)
    s1 = f"마케팅 회의에서 이번 홍보 채널은 {channel}{eul(channel)} 중심으로 운영하기로 했습니다."
    s2 = f"타깃 고객은 {audience}{euro(audience)} 잡는 것이 좋겠다고 설명했습니다."
    s3 = f"성과 지표는 {metric}{eul(metric)} 먼저 보고, 참여 보상은 {reward}{euro(reward)} 제공하기로 했습니다."
    passage = f"{s1} {s2} {s3} 담당자는 홍보 방식과 보상 기준을 학생들이 바로 이해할 수 있게 자연스럽게 풀어 말했습니다."
    atomic = [
        qa("마케팅 회의에서 홍보 채널은 무엇이었어?", channel, f"홍보 채널은 {channel}입니다.", s1),
        qa("마케팅 회의에서 타깃 고객은 누구야?", audience, f"타깃 고객은 {audience}입니다.", s2),
        qa("마케팅 회의에서 참여 보상은 무엇이었어?", reward, f"참여 보상은 {reward}입니다.", s3),
    ]
    return make_row(
        f"natural_test_marketing_{variant:03d}",
        passage,
        atomic,
        "마케팅 회의의 핵심 결정은 무엇이었어?",
        f"홍보 채널은 {channel}이고 타깃 고객은 {audience}이며 참여 보상은 {reward}입니다.",
        f"마케팅 회의에서는 {channel}{eul(channel)} 홍보 채널로 쓰고 {audience}{eul(audience)} 타깃으로 삼으며 참여 보상을 {reward}{euro(reward)} 제공하기로 했습니다.",
    )


def row_community(variant: int) -> dict:
    reward = pick(REWARDS, variant)
    priority = "일정 변경" if variant % 3 == 0 else "장소 변경" if variant % 3 == 1 else "참여 방법 변경"
    criterion = pick(CRITERIA, variant, 2)
    s1 = f"커뮤니티 회의에서 이번 이벤트 보상은 {reward}{euro(reward)} 정했습니다."
    s2 = f"공지 우선순위는 {priority}처럼 참여자에게 바로 영향을 주는 안내를 먼저 올리는 방식이라고 말했습니다."
    s3 = f"신고 처리 기준은 {criterion}{eul(criterion)} 확인하는 절차로 정리했습니다."
    passage = f"{s1} {s2} {s3} 운영진은 보상과 공지, 신고 기준을 하나의 운영 규칙으로 묶어 설명했습니다."
    atomic = [
        qa("커뮤니티 회의에서 이벤트 보상은 무엇인가?", reward, f"커뮤니티 회의에서 이벤트 보상은 {reward}입니다.", s1),
        qa("커뮤니티 회의에서 공지 우선순위는 무엇이라고 했어?", priority, f"공지 우선순위는 {priority}입니다.", s2),
        qa("신고 처리 기준은 무엇이라고 정리했어?", criterion, f"신고 처리 기준은 {criterion}입니다.", s3),
    ]
    return make_row(
        f"natural_test_community_{variant:03d}",
        passage,
        atomic,
        "커뮤니티 회의에서 운영 기준을 어떻게 정했어?",
        f"이벤트 보상은 {reward}이고 공지 우선순위는 {priority}이며 신고 처리 기준은 {criterion}입니다.",
        f"커뮤니티 회의에서는 이벤트 보상을 {reward}{euro(reward)} 정하고 {priority} 공지를 우선하며 {criterion}{eul(criterion)} 신고 처리 기준으로 삼았습니다.",
    )


def row_concept_analogy(variant: int) -> dict:
    concept, analogy, sub_concept, sub_analogy = CONCEPTS[variant % len(CONCEPTS)]
    speaker = pick(SPEAKERS, variant, 3)
    purpose = pick(METRICS, variant, 1)
    s1 = f"{speaker}은 {concept}{eul(concept)} 쉽게 이해시키기 위해 {analogy}에 비유했습니다."
    s2 = f"이어 {sub_concept}{eun(sub_concept)} {sub_analogy}에 가깝다고 설명했습니다."
    s3 = f"이 비유를 통해 학생들이 {purpose}{eul(purpose)} 기준으로 두 개념의 역할을 구분하도록 도왔습니다."
    passage = f"{s1} {s2} {s3} 설명은 기술 용어를 일상적인 장면으로 바꾸어 이해시키려는 의도였습니다."
    atomic = [
        qa(f"{speaker}은 {concept}을 무엇에 비유했어?", analogy, f"{concept}은 {analogy}에 비유되었습니다.", s1),
        qa(f"{sub_concept}은 무엇에 가깝다고 설명했어?", sub_analogy, f"{sub_concept}은 {sub_analogy}에 가깝다고 설명했습니다.", s2),
        qa("이 비유에서 구분 기준으로 삼은 것은 무엇이었어?", purpose, f"구분 기준으로 삼은 것은 {purpose}입니다.", s3),
    ]
    return make_row(
        f"natural_test_concept_analogy_{variant:03d}",
        passage,
        atomic,
        f"{concept}과 {sub_concept}{eul(sub_concept)} 어떻게 비유해서 설명했어?",
        f"{concept}은 {analogy}이고 {sub_concept}은 {sub_analogy}입니다.",
        f"{speaker}은 {concept}{eul(concept)} {analogy}에, {sub_concept}{eul(sub_concept)} {sub_analogy}에 비유해서 설명했습니다.",
    )


def row_database(variant: int) -> dict:
    concept, analogy, sub_concept, sub_analogy = CONCEPTS[(variant + 2) % len(CONCEPTS)]
    location = pick(LOCATIONS, variant, 5)
    metric = pick(METRICS, variant, 3)
    s1 = f"데이터베이스 수업에서 {concept}은 {analogy}처럼 생각하면 된다고 설명했습니다."
    s2 = f"{sub_concept}은 {sub_analogy}처럼 이전 상태를 떠올리면 이해하기 쉽다고 덧붙였습니다."
    s3 = f"실습 결과는 {location}에 올리고 성능 판단은 {metric}을 기준으로 보기로 했습니다."
    passage = f"{s1} {s2} {s3} 교수님은 개념 설명과 실습 안내를 같은 흐름 안에서 정리했습니다."
    atomic = [
        qa(f"데이터베이스 수업에서 {concept}은 무엇처럼 생각하면 된다고 했어?", analogy, f"{concept}은 {analogy}처럼 생각하면 된다고 했습니다.", s1),
        qa(f"{sub_concept}은 무엇처럼 이해하면 된다고 했어?", sub_analogy, f"{sub_concept}은 {sub_analogy}처럼 이해하면 된다고 했습니다.", s2),
        qa("실습 결과는 어디에 올리라고 했어?", location, f"실습 결과는 {location}에 올리라고 했습니다.", s3),
    ]
    return make_row(
        f"natural_test_database_{variant:03d}",
        passage,
        atomic,
        "데이터베이스 수업에서 개념 설명과 실습 안내를 어떻게 정리했어?",
        f"{concept}은 {analogy}처럼 설명했고 {sub_concept}은 {sub_analogy}처럼 설명했으며 실습 결과는 {location}에 올리라고 했습니다.",
        f"데이터베이스 수업에서는 {concept}을 {analogy}에, {sub_concept}을 {sub_analogy}에 비유하고 실습 결과를 {location}에 올리라고 안내했습니다.",
    )


def row_nlp(variant: int) -> dict:
    concept, analogy, sub_concept, sub_analogy = CONCEPTS[(variant + 4) % len(CONCEPTS)]
    criterion = pick(CRITERIA, variant, 5)
    metric = pick(METRICS, variant, 6)
    s1 = f"자연어처리 수업에서 {concept}은 {analogy}에 비유되었습니다."
    s2 = f"{sub_concept}은 {sub_analogy}처럼 문장의 의미를 다루는 과정이라고 설명했습니다."
    s3 = f"모델 결과를 볼 때는 {criterion}{eul(criterion)} 먼저 확인하고 {metric}도 함께 보자고 했습니다."
    passage = f"{s1} {s2} {s3} 이 설명은 모델 내부 과정을 학생들이 질문과 연결해 이해하도록 돕기 위한 것이었습니다."
    atomic = [
        qa(f"자연어처리 수업에서 {concept}은 무엇에 비유되었어?", analogy, f"{concept}은 {analogy}에 비유되었습니다.", s1),
        qa(f"{sub_concept}은 무엇처럼 설명했어?", sub_analogy, f"{sub_concept}은 {sub_analogy}처럼 설명했습니다.", s2),
        qa("모델 결과를 볼 때 먼저 확인할 기준은 무엇이었어?", criterion, f"먼저 확인할 기준은 {criterion}입니다.", s3),
    ]
    return make_row(
        f"natural_test_nlp_{variant:03d}",
        passage,
        atomic,
        "자연어처리 수업에서 모델 설명의 핵심은 무엇이었어?",
        f"{concept}은 {analogy}에 비유했고 {sub_concept}은 {sub_analogy}처럼 설명했으며 확인 기준은 {criterion}입니다.",
        f"자연어처리 수업에서는 {concept}{eul(concept)} {analogy}에 비유하고 {sub_concept}{eul(sub_concept)} {sub_analogy}처럼 설명했으며 결과 확인 기준으로 {criterion}{eul(criterion)} 강조했습니다.",
    )


def row_subject(variant: int) -> dict:
    first = SUBJECT_FACTS[variant % len(SUBJECT_FACTS)]
    second = SUBJECT_FACTS[(variant + 3) % len(SUBJECT_FACTS)]
    third = SUBJECT_FACTS[(variant + 6) % len(SUBJECT_FACTS)]
    speaker = pick(SPEAKERS, variant, 6)
    s1 = f"{speaker}은 {first[0]}{iga(first[0])} {first[2]}{iga(first[2])} 아니라 {first[1]}에 가깝다고 설명했습니다."
    s2 = f"또한 {second[0]}{eun(second[0])} {second[2]}보다 {second[1]}{euro(second[1])} 이해하는 편이 맞다고 정리했습니다."
    s3 = f"마지막으로 {third[0]}{eun(third[0])} {third[2]}보다는 {third[1]}이라는 점을 강조했습니다."
    passage = f"{s1} {s2} {s3} 수업에서는 헷갈리는 개념을 잘못된 해석과 바른 해석을 나란히 두고 비교했습니다."
    atomic = [
        qa(f"{first[0]}은 무엇에 가깝다고 했어?", first[1], f"{first[0]}은 {first[1]}에 가깝다고 했습니다.", s1),
        qa(f"{second[0]}은 무엇으로 이해하는 편이 맞다고 했어?", second[1], f"{second[0]}은 {second[1]}{euro(second[1])} 이해하는 편이 맞다고 했습니다.", s2),
        qa(f"{third[0]}은 무엇이라는 점을 강조했어?", third[1], f"{third[0]}은 {third[1]}이라는 점을 강조했습니다.", s3),
    ]
    return make_row(
        f"natural_test_subject_{variant:03d}",
        passage,
        atomic,
        "수업에서 비교해서 정리한 개념들은 무엇이었어?",
        f"{first[0]}은 {first[1]}이고 {second[0]}은 {second[1]}이며 {third[0]}은 {third[1]}입니다.",
        f"수업에서는 {first[0]}{eul(first[0])} {first[1]}{euro(first[1])}, {second[0]}{eul(second[0])} {second[1]}{euro(second[1])}, {third[0]}{eul(third[0])} {third[1]}{euro(third[1])} 정리했습니다.",
    )


def row_security(variant: int) -> dict:
    owner = pick(OWNERS, variant, 6)
    criterion = pick(CRITERIA, variant, 7)
    time = pick(TIMES, variant, 4)
    location = pick(LOCATIONS, variant, 8)
    s1 = f"보안 점검 회의에서 사고 대응 담당자는 {owner}{euro(owner)} 정했습니다."
    s2 = f"비밀번호 점검은 {criterion}{eul(criterion)} 기준으로 먼저 살피기로 했습니다."
    s3 = f"백업 상태는 {time}에 확인하고 결과는 {location}에 남기기로 했습니다."
    passage = f"{s1} {s2} {s3} 회의에서는 사고 대응과 예방 점검, 기록 위치를 한 번에 정리했습니다."
    atomic = [
        qa("보안 점검 회의에서 사고 대응 담당자는 누구야?", owner, f"사고 대응 담당자는 {owner}입니다.", s1),
        qa("비밀번호 점검 기준은 무엇이라고 했어?", criterion, f"비밀번호 점검 기준은 {criterion}입니다.", s2),
        qa("백업 상태는 언제 확인한다고 했어?", time, f"백업 상태는 {time}에 확인한다고 했습니다.", s3),
    ]
    return make_row(
        f"natural_test_security_{variant:03d}",
        passage,
        atomic,
        "보안 점검 회의에서 정한 운영 기준은 무엇이었어?",
        f"사고 대응 담당자는 {owner}이고 비밀번호 점검 기준은 {criterion}이며 백업 확인 시간은 {time}입니다.",
        f"보안 점검 회의에서는 {owner}{iga(owner)} 사고 대응을 맡고 {criterion}{eul(criterion)} 기준으로 비밀번호를 점검하며 {time}에 백업 상태를 확인하기로 했습니다.",
    )


def row_design(variant: int) -> dict:
    color = ["파란색", "초록색", "검은색", "주황색", "남색", "회색", "흰색", "민트색", "빨간색", "베이지색"][variant % 10]
    location = pick(LOCATIONS, variant, 1)
    criterion = pick(CRITERIA, variant, 1)
    metric = pick(METRICS, variant, 8)
    s1 = f"디자인 회의에서 주요 버튼 색은 {color}{euro(color)} 정했습니다."
    s2 = f"자료 공유 위치는 {location}{eul(location)} 쓰기로 했습니다."
    s3 = f"접근성 검토는 {criterion}{eul(criterion)} 기준으로 살피고, 개선 효과는 {metric}{euro(metric)} 확인하기로 했습니다."
    passage = f"{s1} {s2} {s3} 디자이너는 색상과 공유 위치, 접근성 기준을 서비스 화면 흐름에 맞춰 설명했습니다."
    atomic = [
        qa("디자인 회의에서 주요 버튼 색은 무엇으로 정했어?", color, f"주요 버튼 색은 {color}입니다.", s1),
        qa("디자인 자료 공유 위치는 어디로 정했어?", location, f"디자인 자료 공유 위치는 {location}입니다.", s2),
        qa("접근성 검토 기준은 무엇이라고 했어?", criterion, f"접근성 검토 기준은 {criterion}입니다.", s3),
    ]
    return make_row(
        f"natural_test_design_{variant:03d}",
        passage,
        atomic,
        "디자인 회의에서 화면 개선 기준을 어떻게 정했어?",
        f"버튼 색은 {color}이고 공유 위치는 {location}이며 접근성 검토 기준은 {criterion}입니다.",
        f"디자인 회의에서는 버튼 색을 {color}{euro(color)} 정하고 자료를 {location}에 공유하며 {criterion}{eul(criterion)} 기준으로 접근성을 검토하기로 했습니다.",
    )


def row_infra(variant: int) -> dict:
    time = pick(TIMES, variant, 2)
    owner = pick(OWNERS, variant, 1)
    metric = pick(METRICS, variant, 1)
    location = pick(LOCATIONS, variant, 6)
    s1 = f"인프라 회의에서 배포 시간은 {time}{euro(time)} 잡았습니다."
    s2 = f"롤백 담당자는 {owner}{iga(owner)} 맡기로 했습니다."
    s3 = f"모니터링 지표는 {metric}{eul(metric)} 중심으로 보고, 장애 기록은 {location}에 남기기로 했습니다."
    passage = f"{s1} {s2} {s3} 팀은 배포와 롤백, 모니터링을 같은 절차 안에서 움직이도록 정리했습니다."
    atomic = [
        qa("인프라 회의에서 배포 시간은 언제로 잡았어?", time, f"배포 시간은 {time}입니다.", s1),
        qa("롤백 담당자는 누구로 정했어?", owner, f"롤백 담당자는 {owner}입니다.", s2),
        qa("모니터링 지표는 무엇을 중심으로 보기로 했어?", metric, f"모니터링 지표는 {metric}입니다.", s3),
    ]
    return make_row(
        f"natural_test_infra_{variant:03d}",
        passage,
        atomic,
        "인프라 회의에서 배포 운영을 어떻게 정했어?",
        f"배포 시간은 {time}이고 롤백 담당자는 {owner}이며 모니터링 지표는 {metric}입니다.",
        f"인프라 회의에서는 {time}에 배포하고 {owner}{iga(owner)} 롤백을 맡으며 {metric}{eul(metric)} 중심으로 모니터링하기로 했습니다.",
    )


def row_project(variant: int) -> dict:
    owner = pick(OWNERS, variant, 4)
    metric = pick(METRICS, variant, 2)
    criterion = pick(CRITERIA, variant, 3)
    channel = pick(CHANNELS, variant, 5)
    s1 = f"프로젝트 회고에서 다음 스프린트 진행 담당자는 {owner}{euro(owner)} 정했습니다."
    s2 = f"성과를 판단할 때는 {metric}{eul(metric)} 먼저 확인하기로 했습니다."
    s3 = f"요구사항 변경은 {criterion}{eul(criterion)} 따져 본 뒤 {channel}{euro(channel)} 공유하기로 했습니다."
    passage = f"{s1} {s2} {s3} 회고에서는 다음 작업을 누가 맡고 어떤 기준으로 공유할지까지 정리했습니다."
    atomic = [
        qa("다음 스프린트 진행 담당자는 누구야?", owner, f"다음 스프린트 진행 담당자는 {owner}입니다.", s1),
        qa("프로젝트 성과 판단 지표는 무엇이었어?", metric, f"프로젝트 성과 판단 지표는 {metric}입니다.", s2),
        qa("요구사항 변경은 어디로 공유하기로 했어?", channel, f"요구사항 변경은 {channel}{euro(channel)} 공유하기로 했습니다.", s3),
    ]
    return make_row(
        f"natural_test_project_{variant:03d}",
        passage,
        atomic,
        "프로젝트 회고에서 다음 스프린트 기준을 어떻게 정했어?",
        f"진행 담당자는 {owner}이고 성과 지표는 {metric}이며 요구사항 변경 공유 채널은 {channel}입니다.",
        f"프로젝트 회고에서는 {owner}{iga(owner)} 다음 스프린트를 진행하고 {metric}{euro(metric)} 성과를 보며 요구사항 변경은 {channel}{euro(channel)} 공유하기로 했습니다.",
    )


def row_economics(variant: int) -> dict:
    first = SUBJECT_FACTS[(variant + 6) % len(SUBJECT_FACTS)]
    channel = pick(CHANNELS, variant, 3)
    audience = pick(AUDIENCES, variant, 1)
    metric = pick(METRICS, variant, 9)
    s1 = f"경제학 수업에서 {first[0]}{eun(first[0])} {first[1]}{euro(first[1])} 이해하면 된다고 설명했습니다."
    s2 = f"관련 사례 자료는 {channel}{eul(channel)} 통해 {audience}에게 먼저 공유하기로 했습니다."
    s3 = f"사례 분석 결과는 {metric}{eul(metric)} 기준으로 비교하자고 말했습니다."
    passage = f"{s1} {s2} {s3} 교수님은 이론과 사례, 분석 기준을 한 흐름으로 연결해 설명했습니다."
    atomic = [
        qa(f"경제학 수업에서 {first[0]}은 무엇으로 이해하면 된다고 했어?", first[1], f"{first[0]}은 {first[1]}{euro(first[1])} 이해하면 된다고 했습니다.", s1),
        qa("경제학 사례 자료는 어떤 채널로 공유한다고 했어?", channel, f"경제학 사례 자료는 {channel}{euro(channel)} 공유한다고 했습니다.", s2),
        qa("사례 분석 결과는 무엇을 기준으로 비교한다고 했어?", metric, f"사례 분석 결과는 {metric}{eul(metric)} 기준으로 비교한다고 했습니다.", s3),
    ]
    return make_row(
        f"natural_test_economics_{variant:03d}",
        passage,
        atomic,
        "경제학 수업에서 이론과 사례를 어떻게 연결했어?",
        f"{first[0]}은 {first[1]}{euro(first[1])} 설명했고 사례 자료는 {channel}{euro(channel)} 공유하며 분석 기준은 {metric}입니다.",
        f"경제학 수업에서는 {first[0]}{eul(first[0])} {first[1]}{euro(first[1])} 설명하고 사례 자료를 {channel}{euro(channel)} 공유하며 {metric}{eul(metric)} 기준으로 분석하기로 했습니다.",
    )


def row_medicine(variant: int) -> dict:
    posture = "앉은 자세" if variant % 2 == 0 else "팔을 심장 높이에 둔 자세"
    order = "가슴압박 우선" if variant % 3 == 0 else "의식 확인 후 도움 요청"
    metric = "혈당" if variant % 2 == 0 else "혈압 변화"
    owner = pick(OWNERS, variant, 7)
    s1 = f"의학 기초 수업에서 혈압 측정 자세는 {posture}{euro(posture)} 설명했습니다."
    s2 = f"심폐소생술 순서는 {order}{eul(order)} 먼저 기억하라고 했습니다."
    s3 = f"당뇨 관리 지표는 {metric}{eul(metric)} 중심으로 보고 실습 확인은 {owner}{iga(owner)} 맡기로 했습니다."
    passage = f"{s1} {s2} {s3} 강사는 절차를 단순 암기가 아니라 실제 상황에서 떠올릴 수 있는 규칙으로 설명했습니다."
    atomic = [
        qa("혈압 측정 자세는 무엇이라고 설명했어?", posture, f"혈압 측정 자세는 {posture}입니다.", s1),
        qa("심폐소생술 순서는 무엇을 먼저 기억하라고 했어?", order, f"심폐소생술 순서는 {order}{eul(order)} 먼저 기억하라고 했습니다.", s2),
        qa("당뇨 관리 지표는 무엇을 중심으로 본다고 했어?", metric, f"당뇨 관리 지표는 {metric}입니다.", s3),
    ]
    return make_row(
        f"natural_test_medicine_{variant:03d}",
        passage,
        atomic,
        "의학 기초 수업에서 실습 절차를 어떻게 설명했어?",
        f"혈압 측정 자세는 {posture}이고 심폐소생술 순서는 {order}이며 당뇨 관리 지표는 {metric}입니다.",
        f"의학 기초 수업에서는 혈압 측정 자세를 {posture}{euro(posture)} 설명하고 심폐소생술에서 {order}{eul(order)} 강조했으며 당뇨 관리 지표로 {metric}{eul(metric)} 보라고 했습니다.",
    )


def row_art(variant: int) -> dict:
    contrast = "시선 집중" if variant % 2 == 0 else "주제 강조"
    baroque = "극적 명암" if variant % 3 != 0 else "강한 움직임"
    cubism = "여러 시점" if variant % 2 == 0 else "분해된 형태"
    speaker = pick(SPEAKERS, variant, 1)
    s1 = f"{speaker}은 예술사에서 색채 대비의 목적을 {contrast}{euro(contrast)} 설명했습니다."
    s2 = f"바로크의 특징은 {baroque}{euro(baroque)} 기억하면 된다고 말했습니다."
    s3 = f"입체주의의 특징은 {cubism}{eul(cubism)} 한 화면에 담는 방식이라고 덧붙였습니다."
    passage = f"{s1} {s2} {s3} 수업에서는 미술 양식을 외우기보다 작품을 볼 때 확인할 단서로 이해하라고 안내했습니다."
    atomic = [
        qa("예술사에서 색채 대비의 목적은 무엇이라고 했어?", contrast, f"색채 대비의 목적은 {contrast}입니다.", s1),
        qa("바로크의 특징은 무엇으로 기억하면 된다고 했어?", baroque, f"바로크의 특징은 {baroque}입니다.", s2),
        qa("입체주의의 특징은 무엇이라고 했어?", cubism, f"입체주의의 특징은 {cubism}입니다.", s3),
    ]
    return make_row(
        f"natural_test_art_{variant:03d}",
        passage,
        atomic,
        "예술사 수업에서 미술 양식의 핵심을 어떻게 정리했어?",
        f"색채 대비의 목적은 {contrast}이고 바로크의 특징은 {baroque}이며 입체주의의 특징은 {cubism}입니다.",
        f"예술사 수업에서는 색채 대비를 {contrast}{euro(contrast)}, 바로크를 {baroque}{euro(baroque)}, 입체주의를 {cubism}{euro(cubism)} 정리했습니다.",
    )


def row_logistics(variant: int) -> dict:
    time = pick(TIMES, variant, 5)
    owner = pick(OWNERS, variant, 5)
    criterion = pick(CRITERIA, variant, 8)
    location = pick(LOCATIONS, variant, 4)
    s1 = f"물류 회의에서 라인 점검일은 {time}{euro(time)} 정했습니다."
    s2 = f"안전 교육 담당자는 {owner}{iga(owner)} 맡기로 했습니다."
    s3 = f"품질 기준은 {criterion}{eul(criterion)} 확인하고 기록은 {location}에 남기기로 했습니다."
    passage = f"{s1} {s2} {s3} 담당자는 점검 날짜와 교육 책임, 품질 확인 방식을 작업자들이 바로 볼 수 있게 설명했습니다."
    atomic = [
        qa("라인 점검일은 언제로 정했어?", time, f"라인 점검일은 {time}입니다.", s1),
        qa("안전 교육 담당자는 누구야?", owner, f"안전 교육 담당자는 {owner}입니다.", s2),
        qa("품질 기준은 무엇을 확인한다고 했어?", criterion, f"품질 기준은 {criterion}입니다.", s3),
    ]
    return make_row(
        f"natural_test_logistics_{variant:03d}",
        passage,
        atomic,
        "물류 회의에서 현장 운영 기준을 어떻게 정했어?",
        f"라인 점검일은 {time}이고 안전 교육 담당자는 {owner}이며 품질 기준은 {criterion}입니다.",
        f"물류 회의에서는 {time}에 라인을 점검하고 {owner}{iga(owner)} 안전 교육을 맡으며 {criterion}{eul(criterion)} 품질 기준으로 확인하기로 했습니다.",
    )


def row_education(variant: int) -> dict:
    method = ["짝 토론", "짧은 퀴즈", "개념 지도 그리기", "오답 설명", "실습 발표"][variant % 5]
    metric = pick(METRICS, variant, 5)
    location = pick(LOCATIONS, variant, 7)
    reward = pick(REWARDS, variant, 3)
    s1 = f"교육학 세미나에서 수업 참여 방식은 {method}{euro(method)} 진행하기로 했습니다."
    s2 = f"학습 효과는 {metric}{eul(metric)} 기준으로 확인한다고 설명했습니다."
    s3 = f"참여 자료는 {location}에 모으고 우수 참여 보상은 {reward}{euro(reward)} 정했습니다."
    passage = f"{s1} {s2} {s3} 세미나에서는 학생 참여와 평가, 자료 관리가 서로 이어지도록 안내했습니다."
    atomic = [
        qa("교육학 세미나에서 수업 참여 방식은 무엇이었어?", method, f"수업 참여 방식은 {method}입니다.", s1),
        qa("학습 효과는 무엇을 기준으로 확인한다고 했어?", metric, f"학습 효과는 {metric}{eul(metric)} 기준으로 확인한다고 했습니다.", s2),
        qa("우수 참여 보상은 무엇으로 정했어?", reward, f"우수 참여 보상은 {reward}입니다.", s3),
    ]
    return make_row(
        f"natural_test_education_{variant:03d}",
        passage,
        atomic,
        "교육학 세미나에서 참여와 평가 방식을 어떻게 정했어?",
        f"참여 방식은 {method}이고 학습 효과 기준은 {metric}이며 우수 참여 보상은 {reward}입니다.",
        f"교육학 세미나에서는 {method}{euro(method)} 참여를 진행하고 {metric}{euro(metric)} 학습 효과를 보며 우수 참여 보상을 {reward}{euro(reward)} 정했습니다.",
    )


BUILDERS = [
    row_assignment,
    row_qa_meeting,
    row_marketing,
    row_community,
    row_concept_analogy,
    row_database,
    row_nlp,
    row_subject,
    row_security,
    row_design,
    row_infra,
    row_project,
    row_economics,
    row_medicine,
    row_art,
    row_logistics,
    row_education,
]


def assert_natural_rows(rows: list[dict]) -> None:
    for row in rows:
        text_fields = [row["passage"], row.get("rewrite", "")]
        for qa_item in row["atomic_qas"]:
            text_fields.append(qa_item.get("sub_passage", ""))
        for text in text_fields:
            if "=" in text or ":" in text:
                raise ValueError(f"Non-natural marker found in {row['source_id']}: {text}")


def build_rows(count: int) -> list[dict]:
    rows: list[dict] = []
    variant_by_builder = {builder.__name__: 0 for builder in BUILDERS}
    for idx in range(count):
        builder = BUILDERS[idx % len(BUILDERS)]
        variant = variant_by_builder[builder.__name__]
        variant_by_builder[builder.__name__] += 1
        rows.append(builder(variant))
    assert_natural_rows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--count", type=int, default=450)
    args = parser.parse_args()

    rows = build_rows(args.count)
    write_jsonl(args.output, rows)
    print(f"[PRAG:natural-testset] wrote rows={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
