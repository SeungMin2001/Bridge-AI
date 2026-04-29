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

IDENTITY_TEMPLATES = {
    "ko": [
        "{item} = {a}.",
        "{speaker}는 {item}는 {a}라고 설명했다.",
        "{domain}에서 {item}의 의미는 {a}이다.",
        "{speaker}의 발화 기준으로 {item}는 {a}이다.",
    ],
    "en": [
        "{item} = {a}.",
        "{speaker} explained that {item} means {a}.",
        "In the {domain}, {item} means {a}.",
        "According to {speaker}, {item} is {a}.",
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

EXTRA_FACTS = [
    ("lecture", "history", "역사", "history", "르네상스 발표 주제", "Renaissance presentation topic", "인문주의", "humanism", "절대왕정", "absolute monarchy"),
    ("lecture", "history", "역사", "history", "중간고사 핵심 사건", "midterm key event", "프랑스 혁명", "French Revolution", "산업 혁명", "Industrial Revolution"),
    ("lecture", "law", "법학", "law", "계약 성립 요건", "contract formation requirement", "청약과 승낙", "offer and acceptance", "손해배상", "damages"),
    ("lecture", "law", "법학", "law", "개인정보 처리 근거", "personal data processing basis", "동의", "consent", "소유권", "ownership"),
    ("lecture", "psychology", "심리학", "psychology", "조작적 조건형성", "operant conditioning", "보상과 처벌", "reward and punishment", "단기 기억", "short-term memory"),
    ("lecture", "psychology", "심리학", "psychology", "주의 실험 변수", "attention experiment variable", "반응 시간", "reaction time", "혈압", "blood pressure"),
    ("lecture", "statistics", "통계학", "statistics", "귀무가설", "null hypothesis", "차이가 없다", "no difference", "효과가 크다", "large effect"),
    ("lecture", "statistics", "통계학", "statistics", "유의수준", "significance level", "0.05", "0.05", "0.5", "0.5"),
    ("lecture", "chemistry", "화학", "chemistry", "산화 반응", "oxidation reaction", "전자를 잃는 과정", "loss of electrons", "전자를 얻는 과정", "gain of electrons"),
    ("lecture", "chemistry", "화학", "chemistry", "실험 용액 농도", "solution concentration", "0.2M", "0.2M", "1.0M", "1.0M"),
    ("lecture", "medicine", "의학", "medicine", "혈압 측정 자세", "blood pressure posture", "앉은 자세", "sitting position", "누운 자세", "lying position"),
    ("lecture", "medicine", "의학", "medicine", "감염 관리 핵심", "infection control focus", "손 위생", "hand hygiene", "수면 시간", "sleep duration"),
    ("lecture", "education", "교육학", "education", "형성평가 목적", "formative assessment purpose", "학습 과정 점검", "checking learning progress", "최종 등급 산출", "final grade calculation"),
    ("lecture", "education", "교육학", "education", "피드백 제출 방식", "feedback submission method", "LMS 댓글", "LMS comment", "종이 설문", "paper survey"),
    ("lecture", "design", "디자인", "design", "포스터 핵심 색상", "poster key color", "남색", "navy", "분홍", "pink"),
    ("lecture", "design", "디자인", "design", "타이포그래피 원칙", "typography principle", "가독성", "readability", "장식성", "ornamentation"),
    ("lecture", "business", "경영학", "business", "SWOT의 O", "O in SWOT", "기회", "opportunity", "위협", "threat"),
    ("lecture", "business", "경영학", "business", "고객 세그먼트", "customer segment", "초기 사용자", "early adopters", "공급업체", "suppliers"),
    ("meeting", "hr", "인사 회의", "HR meeting", "신입 온보딩 담당자", "new hire onboarding owner", "하은", "Haeun", "지훈", "Jihoon"),
    ("meeting", "hr", "인사 회의", "HR meeting", "면접 일정", "interview schedule", "목요일 오전", "Thursday morning", "월요일 오후", "Monday afternoon"),
    ("meeting", "legal", "법무 회의", "legal meeting", "계약 검토 담당", "contract review owner", "법무팀", "legal team", "마케팅팀", "marketing team"),
    ("meeting", "legal", "법무 회의", "legal meeting", "NDA 보관 위치", "NDA storage location", "공유 드라이브", "shared drive", "개인 메일함", "personal mailbox"),
    ("meeting", "sales", "영업 회의", "sales meeting", "이번 달 목표", "monthly target", "20건", "20 deals", "5건", "5 deals"),
    ("meeting", "sales", "영업 회의", "sales meeting", "우선 고객군", "priority customer group", "대학 연구실", "university labs", "동네 카페", "local cafes"),
    ("meeting", "security", "보안 회의", "security meeting", "2단계 인증 적용일", "two-factor rollout date", "다음 주 월요일", "next Monday", "다음 달 금요일", "next month's Friday"),
    ("meeting", "security", "보안 회의", "security meeting", "보안 점검 항목", "security check item", "접근 권한", "access permission", "폰트 크기", "font size"),
    ("meeting", "support", "고객지원 회의", "support meeting", "긴급 문의 응답 시간", "urgent ticket response time", "30분 이내", "within 30 minutes", "3일 이내", "within three days"),
    ("meeting", "support", "고객지원 회의", "support meeting", "FAQ 수정 담당자", "FAQ update owner", "유진", "Yujin", "민재", "Minjae"),
    ("meeting", "planning", "기획 회의", "planning meeting", "다음 스프린트 목표", "next sprint goal", "검색 정확도 개선", "improve search accuracy", "로고 색상 변경", "change logo color"),
    ("meeting", "planning", "기획 회의", "planning meeting", "마일스톤 M1", "milestone M1", "프로토타입 완성", "prototype completion", "운영 종료", "service shutdown"),
    ("meeting", "data", "데이터 회의", "data meeting", "라벨링 기준", "labeling rule", "교수 발화 우선", "professor utterance first", "학생 농담 우선", "student joke first"),
    ("meeting", "data", "데이터 회의", "data meeting", "중복 제거 기준", "deduplication rule", "같은 passage 제거", "remove same passage", "같은 길이 제거", "remove same length"),
    ("meeting", "qa", "QA 회의", "QA meeting", "회귀 테스트 범위", "regression test scope", "로그인과 채팅", "login and chat", "배경 음악", "background music"),
    ("meeting", "qa", "QA 회의", "QA meeting", "버그 우선순위 기준", "bug priority rule", "사용자 차단 여부", "whether it blocks users", "아이콘 모양", "icon shape"),
]

FACTS.extend(EXTRA_FACTS)
VARIANTS_PER_FACT = 10


def make_row(source_id: str, lang: str, domain_name: str, item: str, a: str, b: str, speaker: str, template_idx: int) -> dict:
    if "identity" in source_id:
        template = IDENTITY_TEMPLATES[lang][template_idx % len(IDENTITY_TEMPLATES[lang])]
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
        for variant in range(VARIANTS_PER_FACT):
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
