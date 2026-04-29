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

LECTURE_FACT_GROUPS = [
    ("lecture", "literature", "문학", "literature", [
        ("화자의 정서", "speaker emotion", "그리움", "longing", "분노", "anger"),
        ("상징 소재", "symbolic object", "갈대", "reed", "바위", "rock"),
        ("서술 시점", "narrative point of view", "1인칭", "first person", "3인칭", "third person"),
        ("작품 주제", "work theme", "상실 회복", "recovery from loss", "권력 비판", "critique of power"),
        ("비유법", "figurative device", "은유", "metaphor", "반어", "irony"),
        ("갈등 유형", "conflict type", "내적 갈등", "internal conflict", "사회적 갈등", "social conflict"),
        ("시상 전개", "poetic development", "대조", "contrast", "열거", "enumeration"),
        ("핵심 배경", "key setting", "겨울 강가", "winter riverside", "도시 광장", "city square"),
    ]),
    ("lecture", "database", "데이터베이스", "database", [
        ("정규화 목적", "normalization purpose", "중복 감소", "reducing redundancy", "응답 지연 증가", "increasing latency"),
        ("기본키 역할", "primary key role", "행 식별", "identifying rows", "컬럼 암호화", "encrypting columns"),
        ("인덱스 장점", "index benefit", "검색 속도 향상", "faster lookup", "저장 공간 제거", "removing storage"),
        ("트랜잭션 조건", "transaction property", "원자성", "atomicity", "임의성", "randomness"),
        ("조인 기준", "join condition", "공통 키", "shared key", "파일 크기", "file size"),
        ("ERD의 사각형", "rectangle in ERD", "엔티티", "entity", "관계", "relationship"),
        ("외래키 목적", "foreign key purpose", "참조 무결성", "referential integrity", "화면 디자인", "screen design"),
        ("뷰의 특징", "view property", "가상 테이블", "virtual table", "물리 디스크", "physical disk"),
    ]),
    ("lecture", "network", "네트워크", "networking", [
        ("TCP 특징", "TCP property", "신뢰성", "reliability", "방송 전송", "broadcasting"),
        ("UDP 특징", "UDP property", "낮은 지연", "low latency", "순서 보장", "ordered delivery"),
        ("DNS 역할", "DNS role", "도메인 변환", "domain resolution", "파일 압축", "file compression"),
        ("HTTP 상태 404", "HTTP status 404", "찾을 수 없음", "not found", "인증 성공", "authentication success"),
        ("라우터 역할", "router role", "경로 선택", "route selection", "화면 렌더링", "screen rendering"),
        ("서브넷 목적", "subnet purpose", "주소 범위 분리", "separating address ranges", "비밀번호 저장", "password storage"),
        ("TLS 목적", "TLS purpose", "통신 암호화", "encrypting communication", "이미지 편집", "image editing"),
        ("프록시 역할", "proxy role", "중간 전달", "intermediate forwarding", "배터리 충전", "battery charging"),
    ]),
    ("lecture", "ai", "인공지능", "artificial intelligence", [
        ("과적합 의미", "overfitting meaning", "훈련 데이터에 치우침", "bias toward training data", "데이터 부족 없음", "no data shortage"),
        ("검증셋 목적", "validation set purpose", "일반화 확인", "checking generalization", "최종 배포", "final deployment"),
        ("손실 함수 역할", "loss function role", "오차 측정", "measuring error", "데이터 저장", "storing data"),
        ("임베딩 의미", "embedding meaning", "벡터 표현", "vector representation", "파일 확장자", "file extension"),
        ("어텐션 목적", "attention purpose", "중요 토큰 가중", "weighting important tokens", "메모리 삭제", "memory deletion"),
        ("배치 크기 의미", "batch size meaning", "한 번에 보는 샘플 수", "samples per update", "모델 층 수", "number of layers"),
        ("학습률 의미", "learning rate meaning", "업데이트 크기", "update size", "문장 길이", "sentence length"),
        ("정규화 목적", "regularization purpose", "과적합 완화", "reducing overfitting", "정답 암기 강화", "strengthening memorization"),
    ]),
    ("lecture", "environment", "환경학", "environmental science", [
        ("탄소중립 목표", "carbon neutrality goal", "순배출 제로", "net zero emissions", "폐기물 증가", "more waste"),
        ("생물다양성 의미", "biodiversity meaning", "종 다양성", "species diversity", "단일 작물", "single crop"),
        ("미세먼지 원인", "fine dust cause", "연소 배출", "combustion emissions", "조수 간만", "tidal change"),
        ("재활용 핵심", "recycling focus", "분리배출", "separate disposal", "무작위 폐기", "random disposal"),
        ("기후 적응", "climate adaptation", "피해 줄이기", "reducing damage", "배출 늘리기", "increasing emissions"),
        ("수질 지표", "water quality indicator", "용존 산소", "dissolved oxygen", "화면 밝기", "screen brightness"),
        ("도시열섬 완화", "urban heat island mitigation", "녹지 확대", "more green space", "도로 확장", "road expansion"),
        ("순환경제 목표", "circular economy goal", "자원 재사용", "resource reuse", "일회용 증가", "more single-use items"),
    ]),
    ("lecture", "art", "예술사", "art history", [
        ("인상주의 특징", "impressionism feature", "빛의 순간", "momentary light", "정확한 해부", "precise anatomy"),
        ("입체주의 특징", "cubism feature", "여러 시점", "multiple viewpoints", "단일 원근", "single perspective"),
        ("바로크 특징", "baroque feature", "극적 명암", "dramatic contrast", "평면 장식", "flat decoration"),
        ("미니멀리즘 특징", "minimalism feature", "단순한 형태", "simple forms", "복잡한 서사", "complex narrative"),
        ("색채 대비 목적", "color contrast purpose", "시선 집중", "drawing attention", "정보 삭제", "removing information"),
        ("구도 중심", "composition focus", "균형", "balance", "무작위성", "randomness"),
        ("원근법 목적", "perspective purpose", "깊이 표현", "expressing depth", "소리 증폭", "sound amplification"),
        ("큐레이션 기준", "curation criterion", "주제 연결", "thematic connection", "파일 이름", "file name"),
    ]),
    ("lecture", "music", "음악", "music", [
        ("박자 의미", "meter meaning", "강약 패턴", "accent pattern", "음색 종류", "tone color type"),
        ("장조 느낌", "major key mood", "밝음", "bright", "불안", "anxious"),
        ("단조 느낌", "minor key mood", "어두움", "dark", "축제", "festive"),
        ("화성 역할", "harmony role", "음의 어울림", "combination of tones", "악기 보관", "instrument storage"),
        ("템포 의미", "tempo meaning", "빠르기", "speed", "음량", "volume"),
        ("동기 의미", "motif meaning", "짧은 핵심 선율", "short core melody", "무대 위치", "stage position"),
        ("대위법 특징", "counterpoint feature", "독립 선율 결합", "combining independent melodies", "리듬 삭제", "removing rhythm"),
        ("음색 의미", "timbre meaning", "소리의 색", "color of sound", "악보 크기", "score size"),
    ]),
    ("lecture", "philosophy", "철학", "philosophy", [
        ("공리주의 기준", "utilitarian criterion", "최대 행복", "greatest happiness", "혈통", "lineage"),
        ("의무론 기준", "deontology criterion", "도덕 법칙", "moral law", "시장 가격", "market price"),
        ("실존주의 핵심", "existentialism focus", "선택과 책임", "choice and responsibility", "기계 효율", "machine efficiency"),
        ("인식론 질문", "epistemology question", "앎의 조건", "conditions of knowledge", "건물 높이", "building height"),
        ("형이상학 주제", "metaphysics topic", "존재", "being", "광고 문구", "ad copy"),
        ("덕 윤리 초점", "virtue ethics focus", "성품", "character", "소득세", "income tax"),
        ("사회계약론 의미", "social contract meaning", "합의된 권위", "agreed authority", "무작위 추첨", "random drawing"),
        ("회의주의 태도", "skepticism stance", "확실성 의심", "doubting certainty", "속도 향상", "speed increase"),
    ]),
]

MEETING_FACT_GROUPS = [
    ("meeting", "startup", "스타트업 회의", "startup meeting", [
        ("이번 피벗 방향", "pivot direction", "B2B 교육", "B2B education", "게임 광고", "game ads"),
        ("핵심 지표", "core metric", "주간 활성 사용자", "weekly active users", "사무실 면적", "office area"),
        ("투자자 미팅일", "investor meeting date", "목요일", "Thursday", "일요일", "Sunday"),
        ("MVP 범위", "MVP scope", "질문 검색", "question search", "아바타 꾸미기", "avatar styling"),
        ("데모 담당자", "demo owner", "민아", "Mina", "준호", "Junho"),
        ("리스크 항목", "risk item", "데이터 부족", "data shortage", "로고 색상", "logo color"),
        ("고객 인터뷰 대상", "customer interview target", "교수자", "instructors", "배달 기사", "delivery riders"),
        ("출시 목표", "launch target", "6월 베타", "June beta", "내년 정식", "next year official"),
    ]),
    ("meeting", "university_admin", "학사 회의", "university admin meeting", [
        ("수강정정 마감", "course change deadline", "금요일", "Friday", "화요일", "Tuesday"),
        ("장학금 공지일", "scholarship notice date", "다음 주 월요일", "next Monday", "이번 주 토요일", "this Saturday"),
        ("출석 인정 기준", "attendance approval rule", "증빙 제출", "proof submission", "구두 신청", "verbal request"),
        ("졸업 심사 담당", "graduation review owner", "학사팀", "academic office", "동아리연합회", "club union"),
        ("강의실 변경", "classroom change", "공학관 204호", "Engineering 204", "도서관 1층", "Library first floor"),
        ("온라인 시험 플랫폼", "online exam platform", "LMS", "LMS", "개인 블로그", "personal blog"),
        ("상담 예약 방식", "advising booking method", "포털 신청", "portal request", "문자만 가능", "text only"),
        ("교환학생 설명회", "exchange info session", "수요일 오후", "Wednesday afternoon", "금요일 새벽", "Friday dawn"),
    ]),
    ("meeting", "hospital_ops", "병원 운영 회의", "hospital operations meeting", [
        ("야간 당직 담당", "night duty owner", "A팀", "team A", "C팀", "team C"),
        ("응급실 병상 기준", "ER bed rule", "중증도 우선", "severity first", "도착 순서만", "arrival order only"),
        ("소독 점검 시간", "sanitation check time", "오전 8시", "8 a.m.", "밤 11시", "11 p.m."),
        ("환자 안내 방식", "patient guidance method", "문자 알림", "text alert", "종이 쪽지", "paper note"),
        ("약품 재고 담당", "medicine stock owner", "약제팀", "pharmacy team", "홍보팀", "PR team"),
        ("검사 결과 전달", "test result delivery", "앱 알림", "app notification", "복도 게시", "hallway posting"),
        ("감염 의심 보고", "infection suspicion report", "즉시 보고", "immediate report", "월말 보고", "month-end report"),
        ("진료 대기 지표", "clinic wait metric", "평균 대기시간", "average wait time", "주차 대수", "parking count"),
    ]),
    ("meeting", "manufacturing", "제조 회의", "manufacturing meeting", [
        ("불량률 목표", "defect rate target", "1% 이하", "below 1%", "10% 이상", "above 10%"),
        ("라인 점검일", "line inspection date", "화요일", "Tuesday", "토요일", "Saturday"),
        ("안전 교육 담당", "safety training owner", "현장팀", "field team", "디자인팀", "design team"),
        ("부품 입고 시간", "parts arrival time", "오후 3시", "3 p.m.", "오전 6시", "6 a.m."),
        ("품질 기준", "quality criterion", "치수 오차", "dimension tolerance", "포스터 색상", "poster color"),
        ("재고 기준", "inventory rule", "2주치 확보", "two weeks of stock", "당일 소진", "same-day depletion"),
        ("설비 이상 보고", "equipment issue report", "즉시 알림", "immediate alert", "분기 보고", "quarterly report"),
        ("생산 우선 제품", "priority product", "모듈 A", "module A", "샘플 Z", "sample Z"),
    ]),
    ("meeting", "policy", "정책 회의", "policy meeting", [
        ("민원 처리 기한", "complaint handling deadline", "7일 이내", "within seven days", "60일 이내", "within sixty days"),
        ("공청회 장소", "public hearing venue", "시청 강당", "city hall auditorium", "체육관 창고", "gym storage"),
        ("우선 지원 대상", "priority support target", "소상공인", "small businesses", "대기업", "large corporations"),
        ("성과 평가 기준", "performance evaluation rule", "참여율", "participation rate", "건물 색상", "building color"),
        ("예산 배정 원칙", "budget allocation rule", "취약지역 우선", "vulnerable areas first", "무작위 배정", "random allocation"),
        ("홍보 채널", "publicity channel", "지역 방송", "local broadcast", "개인 메신저", "personal messenger"),
        ("자료 공개 방식", "data disclosure method", "홈페이지 게시", "website posting", "비공개 보관", "private storage"),
        ("사업 종료일", "program end date", "12월 31일", "December 31", "3월 1일", "March 1"),
    ]),
    ("meeting", "education_ops", "교육 운영 회의", "education operations meeting", [
        ("보충수업 대상", "makeup class target", "결석 학생", "absent students", "전체 교직원", "all staff"),
        ("과제 피드백 방식", "assignment feedback method", "루브릭 코멘트", "rubric comments", "점수만 공개", "score only"),
        ("학습자료 위치", "learning material location", "LMS 자료실", "LMS resources", "카페 메뉴판", "cafe menu"),
        ("퀴즈 재응시 조건", "quiz retake condition", "시스템 오류", "system error", "개인 변심", "personal preference"),
        ("조별 발표 순서", "team presentation order", "추첨", "random draw", "성적순", "by grade"),
        ("멘토링 시간", "mentoring time", "목요일 저녁", "Thursday evening", "월요일 새벽", "Monday dawn"),
        ("결석 인정 자료", "absence proof", "진단서", "medical note", "필기구", "stationery"),
        ("프로젝트 평가 비중", "project grading weight", "40%", "40%", "5%", "5%"),
    ]),
    ("meeting", "commerce", "커머스 회의", "commerce meeting", [
        ("쿠폰 적용 대상", "coupon target", "신규 가입자", "new users", "탈퇴 회원", "closed accounts"),
        ("무료배송 기준", "free shipping threshold", "3만원 이상", "over 30,000 won", "100만원 이상", "over 1,000,000 won"),
        ("반품 처리 기한", "return handling deadline", "3영업일", "three business days", "30영업일", "thirty business days"),
        ("추천 상품 기준", "recommendation criterion", "최근 조회", "recent views", "키보드 색상", "keyboard color"),
        ("재입고 알림 방식", "restock alert method", "앱 푸시", "app push", "우편 발송", "postal mail"),
        ("리뷰 노출 기준", "review display rule", "구매 인증", "verified purchase", "닉네임 길이", "nickname length"),
        ("정산 주기", "settlement cycle", "매주 금요일", "every Friday", "매년 1회", "once a year"),
        ("프로모션 목표", "promotion goal", "재구매율 증가", "higher repurchase rate", "창고 축소", "warehouse reduction"),
    ]),
    ("meeting", "content", "콘텐츠 회의", "content meeting", [
        ("이번 영상 주제", "video topic", "학습 루틴", "study routine", "주차 요금", "parking fee"),
        ("썸네일 색상", "thumbnail color", "노랑", "yellow", "회색", "gray"),
        ("업로드 요일", "upload day", "수요일", "Wednesday", "일요일", "Sunday"),
        ("대본 담당자", "script owner", "하린", "Harin", "민석", "Minseok"),
        ("검수 기준", "review criterion", "사실 오류", "factual errors", "배경음 길이", "background music length"),
        ("쇼츠 길이", "shorts length", "45초", "45 seconds", "5분", "5 minutes"),
        ("댓글 대응 원칙", "comment response rule", "질문 우선", "questions first", "이모지 우선", "emoji first"),
        ("성과 지표", "content metric", "완주율", "completion rate", "프린터 수", "printer count"),
    ]),
    ("meeting", "community", "커뮤니티 회의", "community meeting", [
        ("공지 우선순위", "announcement priority", "일정 변경", "schedule changes", "잡담 모음", "chat highlights"),
        ("신고 처리 기준", "report handling rule", "욕설 여부", "abusive language", "글자 수", "character count"),
        ("이벤트 보상", "event reward", "포인트 500점", "500 points", "빈 배지", "empty badge"),
        ("운영자 당번", "moderator shift", "서준", "Seojun", "나래", "Narae"),
        ("새 멤버 안내", "new member guide", "규칙 요약", "rule summary", "광고 링크", "ad links"),
        ("피드백 수집 방식", "feedback collection method", "익명 폼", "anonymous form", "전화 통화", "phone calls"),
        ("주간 회고 시간", "weekly retrospective time", "금요일 오후", "Friday afternoon", "화요일 새벽", "Tuesday dawn"),
        ("채널 정리 기준", "channel cleanup rule", "활동 없는 채널", "inactive channels", "가장 긴 이름", "longest name"),
    ]),
]


def expand_fact_groups(groups: list[tuple[str, str, str, str, list[tuple[str, str, str, str, str, str]]]]) -> list[tuple[str, str, str, str, str, str, str, str, str, str]]:
    facts = []
    for group_name, domain_key, domain_ko, domain_en, items in groups:
        for item_ko, item_en, a_ko, a_en, b_ko, b_en in items:
            facts.append((group_name, domain_key, domain_ko, domain_en, item_ko, item_en, a_ko, a_en, b_ko, b_en))
    return facts


BULK_FACTS = expand_fact_groups(LECTURE_FACT_GROUPS + MEETING_FACT_GROUPS)

FACTS.extend(EXTRA_FACTS)
FACTS.extend(BULK_FACTS)
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
