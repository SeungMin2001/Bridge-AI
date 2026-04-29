"""Build diverse lecture/meeting hard-pair source data for PRAG augmentation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .config import DIVERSE_SOURCE_PATH
from .data import write_jsonl


LECTURE_DOMAINS = {
    "economics": {
        "ko": ("경제학", ["수요 탄력성", "기회비용", "한계비용", "물가상승률"], ["높음", "낮음", "3%", "7%"]),
        "en": ("economics", ["demand elasticity", "opportunity cost", "marginal cost", "inflation rate"], ["high", "low", "3%", "7%"]),
    },
    "math": {
        "ko": ("수학", ["극한 단원", "미분 과제", "행렬 퀴즈", "확률 발표"], ["월요일", "수요일", "5장", "7장"]),
        "en": ("math", ["limit unit", "derivative homework", "matrix quiz", "probability talk"], ["Monday", "Wednesday", "chapter 5", "chapter 7"]),
    },
    "english": {
        "ko": ("영어", ["에세이 초안", "단어 시험", "발음 과제", "토론 주제"], ["금요일", "화요일", "환경", "기술"]),
        "en": ("English", ["essay draft", "vocab quiz", "pronunciation task", "debate topic"], ["Friday", "Tuesday", "environment", "technology"]),
    },
    "computer": {
        "ko": ("컴퓨터", ["캐시 히트", "스레드", "API 과제", "모델 평가"], ["데이터를 찾는 것", "가벼운 실행 흐름", "REST", "F1 점수"]),
        "en": ("computer science", ["cache hit", "thread", "API task", "model evaluation"], ["finding data", "lightweight execution flow", "REST", "F1 score"]),
    },
    "physics": {
        "ko": ("물리", ["실험 A", "가속도", "전압 측정", "보고서"], ["2.4m/s²", "오른쪽", "5V", "다음 주"]),
        "en": ("physics", ["experiment A", "acceleration", "voltage measure", "report"], ["2.4m/s^2", "rightward", "5V", "next week"]),
    },
    "biology": {
        "ko": ("생물", ["세포 호흡", "효소 실험", "유전자 단원", "관찰 노트"], ["미토콘드리아", "37도", "3장", "금요일"]),
        "en": ("biology", ["cell respiration", "enzyme lab", "gene unit", "observation note"], ["mitochondria", "37 degrees", "chapter 3", "Friday"]),
    },
}


MEETING_DOMAINS = {
    "product": {
        "ko": ("제품 회의", ["베타 배포", "온보딩 화면", "검색 기능", "알림 정책"], ["수요일", "민수", "우선순위 1", "즉시 알림"]),
        "en": ("product meeting", ["beta release", "onboarding screen", "search feature", "notification policy"], ["Wednesday", "Mina", "priority 1", "instant alert"]),
    },
    "finance": {
        "ko": ("재무 회의", ["예산 검토", "광고비", "출장비", "계약 승인"], ["300만원", "15%", "금요일", "팀장"]),
        "en": ("finance meeting", ["budget review", "ad spend", "travel cost", "contract approval"], ["3 million won", "15%", "Friday", "team lead"]),
    },
    "research": {
        "ko": ("연구 회의", ["실험군 A", "평가지표", "논문 초안", "데이터 정제"], ["정확도", "재현율", "서연", "화요일"]),
        "en": ("research meeting", ["group A", "metric", "paper draft", "data cleanup"], ["accuracy", "recall", "Soyeon", "Tuesday"]),
    },
    "operations": {
        "ko": ("운영 회의", ["서버 점검", "고객 응대", "배포 시간", "장애 보고"], ["새벽 2시", "지영", "목요일", "1시간 이내"]),
        "en": ("operations meeting", ["server check", "customer response", "deploy time", "incident report"], ["2 a.m.", "Jiyoung", "Thursday", "within one hour"]),
    },
    "marketing": {
        "ko": ("마케팅 회의", ["캠페인 문구", "타깃 고객", "이벤트 날짜", "성과 지표"], ["짧게", "신입생", "토요일", "전환율"]),
        "en": ("marketing meeting", ["campaign slogan", "target customer", "event date", "success metric"], ["short", "new students", "Saturday", "conversion rate"]),
    },
}


TEMPLATES = {
    "ko": [
        "{speaker}는 {domain}에서 {item}의 값은 {a}라고 말했다.",
        "{speaker}의 설명에 따르면 {item}은 {a}로 정리된다.",
        "{domain} 공지: {item}은 {a}이다.",
    ],
    "en": [
        "{speaker} said in the {domain} that {item} is {a}.",
        "According to {speaker}, {item} is {a} in the {domain}.",
        "{domain} note: {item} is {a}.",
    ],
}

QUESTIONS = {
    "ko": [
        "{item}은 뭐야?",
        "{item}은 어떻게 정리됐어?",
        "{speaker}가 말한 {item}은?",
    ],
    "en": [
        "What is {item}?",
        "How was {item} decided?",
        "What did {speaker} say about {item}?",
    ],
}

SPEAKERS = {
    "ko": ["이 교수", "박 강사", "민아 조교", "승민 팀장", "지영 매니저"],
    "en": ["Professor Lee", "Instructor Park", "TA Mina", "Lead Seungmin", "Manager Jiyoung"],
}


FACTS = [
    ("lecture", "economics", "경제학", "economics", "수요 탄력성", "demand elasticity", "높음", "high", "낮음", "low"),
    ("lecture", "economics", "경제학", "economics", "물가상승률", "inflation rate", "3%", "3%", "7%", "7%"),
    ("lecture", "math", "수학", "math", "미분 과제 마감일", "derivative homework deadline", "수요일", "Wednesday", "금요일", "Friday"),
    ("lecture", "math", "수학", "math", "행렬 퀴즈 범위", "matrix quiz scope", "5장", "chapter 5", "7장", "chapter 7"),
    ("lecture", "english", "영어", "English", "에세이 초안 마감일", "essay draft deadline", "금요일", "Friday", "화요일", "Tuesday"),
    ("lecture", "english", "영어", "English", "토론 주제", "debate topic", "환경", "environment", "기술", "technology"),
    ("lecture", "computer", "컴퓨터", "computer science", "캐시 히트 뜻", "cache hit meaning", "데이터를 찾는 것", "finding data", "데이터를 못 찾는 것", "missing data"),
    ("lecture", "computer", "컴퓨터", "computer science", "모델 평가 지표", "model evaluation metric", "F1 점수", "F1 score", "정확도", "accuracy"),
    ("lecture", "physics", "물리", "physics", "실험 A 가속도", "experiment A acceleration", "2.4m/s²", "2.4m/s^2", "1.8m/s²", "1.8m/s^2"),
    ("lecture", "physics", "물리", "physics", "전압 측정값", "voltage measurement", "5V", "5V", "9V", "9V"),
    ("lecture", "biology", "생물", "biology", "세포 호흡 위치", "cell respiration location", "미토콘드리아", "mitochondria", "리보솜", "ribosome"),
    ("lecture", "biology", "생물", "biology", "효소 실험 온도", "enzyme lab temperature", "37도", "37 degrees", "25도", "25 degrees"),
    ("meeting", "product", "제품 회의", "product meeting", "베타 배포일", "beta release date", "수요일", "Wednesday", "금요일", "Friday"),
    ("meeting", "product", "제품 회의", "product meeting", "검색 기능 우선순위", "search feature priority", "우선순위 1", "priority 1", "우선순위 3", "priority 3"),
    ("meeting", "finance", "재무 회의", "finance meeting", "광고비 증액률", "ad spend increase", "15%", "15%", "8%", "8%"),
    ("meeting", "finance", "재무 회의", "finance meeting", "예산 검토 금액", "budget review amount", "300만원", "3 million won", "500만원", "5 million won"),
    ("meeting", "research", "연구 회의", "research meeting", "평가지표", "evaluation metric", "재현율", "recall", "정밀도", "precision"),
    ("meeting", "research", "연구 회의", "research meeting", "논문 초안 담당자", "paper draft owner", "서연", "Soyeon", "도윤", "Doyoon"),
    ("meeting", "operations", "운영 회의", "operations meeting", "서버 점검 시간", "server check time", "새벽 2시", "2 a.m.", "오전 9시", "9 a.m."),
    ("meeting", "operations", "운영 회의", "operations meeting", "장애 보고 기한", "incident report deadline", "1시간 이내", "within one hour", "하루 이내", "within one day"),
    ("meeting", "marketing", "마케팅 회의", "marketing meeting", "타깃 고객", "target customer", "신입생", "new students", "졸업생", "graduates"),
    ("meeting", "marketing", "마케팅 회의", "marketing meeting", "성과 지표", "success metric", "전환율", "conversion rate", "조회수", "view count"),
    ("lecture", "identity", "강의 메모", "lecture note", "알파 변수", "alpha variable", "학습률", "learning rate", "배치 크기", "batch size"),
    ("lecture", "identity", "강의 메모", "lecture note", "베타 기호", "beta symbol", "정규화 계수", "normalization coefficient", "오차항", "error term"),
    ("lecture", "identity", "강의 메모", "lecture note", "RAG", "RAG", "검색 증강 생성", "retrieval augmented generation", "순환 그래프", "recurrent graph"),
    ("lecture", "identity", "강의 메모", "lecture note", "API", "API", "응용 프로그램 인터페이스", "application programming interface", "평균 정밀도 지표", "average precision metric"),
    ("lecture", "identity", "강의 메모", "lecture note", "오메가 표식", "omega marker", "기말 범위", "final exam scope", "출석 점수", "attendance score"),
    ("lecture", "identity", "강의 메모", "lecture note", "세타 코드", "theta code", "파란 파일", "blue file", "초록 폴더", "green folder"),
    ("meeting", "identity", "회의 메모", "meeting note", "프로젝트 루나", "Project Luna", "검색 개선 작업", "search improvement task", "결제 개선 작업", "payment improvement task"),
    ("meeting", "identity", "회의 메모", "meeting note", "담당자 A", "owner A", "민수", "Minsu", "지영", "Jiyoung"),
    ("meeting", "identity", "회의 메모", "meeting note", "릴리즈 R2", "release R2", "수요일 배포", "Wednesday deployment", "금요일 배포", "Friday deployment"),
    ("meeting", "identity", "회의 메모", "meeting note", "긴급 채널", "urgent channel", "슬랙 알림", "Slack alert", "이메일 알림", "email alert"),
    ("meeting", "identity", "회의 메모", "meeting note", "문서 D1", "document D1", "요구사항 명세서", "requirement specification", "회의록", "meeting minutes"),
    ("meeting", "identity", "회의 메모", "meeting note", "태스크 T7", "task T7", "프론트 수정", "frontend fix", "백엔드 점검", "backend check"),
]


def make_row(source_id: str, lang: str, domain_name: str, item: str, a: str, b: str, speaker: str, template_idx: int) -> dict:
    if "identity" in source_id:
        template = "{item} = {a}." if lang == "en" else "{item} = {a}."
    else:
        template = TEMPLATES[lang][template_idx % len(TEMPLATES[lang])]
    q_template = QUESTIONS[lang][template_idx % len(QUESTIONS[lang])]
    passage = template.format(speaker=speaker, domain=domain_name, item=item, a=a)
    negative = template.format(speaker=speaker, domain=domain_name, item=item, a=b)
    question = q_template.format(speaker=speaker, domain=domain_name, item=item)
    return {
        "source_id": source_id,
        "speaker": speaker,
        "task": "diverse_service_memory",
        "question": question,
        "answer": a,
        "passage": passage,
        "hard_negatives": [{"passage": negative, "answer": b}],
    }


def build_rows(limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for idx, (group_name, domain_key, domain_ko, domain_en, item_ko, item_en, a_ko, a_en, b_ko, b_en) in enumerate(FACTS):
        for variant in range(6):
            for lang in ("ko", "en"):
                domain_name = domain_ko if lang == "ko" else domain_en
                item = item_ko if lang == "ko" else item_en
                a = a_ko if lang == "ko" else a_en
                b = b_ko if lang == "ko" else b_en
                if variant % 2 == 1:
                    a, b = b, a
                speaker = SPEAKERS[lang][(idx + variant) % len(SPEAKERS[lang])]
                source_id = f"{group_name}_{lang}_{domain_key}_{idx}_{variant}"
                rows.append(make_row(source_id, lang, domain_name, item, a, b, speaker, idx + variant))

    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DIVERSE_SOURCE_PATH))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_rows(args.limit, args.seed)
    write_jsonl(args.output, rows)
    print(f"[PRAG:diverse-sources] rows={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
