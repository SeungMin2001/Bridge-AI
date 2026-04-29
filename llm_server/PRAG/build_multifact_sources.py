"""Build multi-fact lecture/meeting source passages for PRAG augmentation.

Each source row contains several facts in one passage. The augmentation step can
then produce genuinely different atomic Q/A pairs instead of paraphrasing the
same single fact.
"""

from __future__ import annotations

import argparse
import random

from .config import MULTIFACT_SOURCE_PATH
from .data import write_jsonl


DEFAULT_OUTPUT = MULTIFACT_SOURCE_PATH

SPEAKERS = {
    "ko": ["이 교수", "박 강사", "민아 조교", "승민 팀장", "지영 매니저"],
    "en": ["Professor Lee", "Instructor Park", "TA Mina", "Lead Seungmin", "Manager Jiyoung"],
}


DOMAIN_FACTS = [
    ("lecture", "database", "데이터베이스", "database", [
        ("기본키 역할", "primary key role", "행 식별", "identifying rows", "컬럼 암호화", "encrypting columns"),
        ("외래키 목적", "foreign key purpose", "참조 무결성", "referential integrity", "화면 디자인", "screen design"),
        ("정규화 목적", "normalization purpose", "중복 감소", "reducing redundancy", "응답 지연 증가", "increasing latency"),
        ("인덱스 장점", "index benefit", "검색 속도 향상", "faster lookup", "저장 공간 제거", "removing storage"),
        ("트랜잭션 조건", "transaction property", "원자성", "atomicity", "임의성", "randomness"),
        ("뷰의 특징", "view property", "가상 테이블", "virtual table", "물리 디스크", "physical disk"),
    ]),
    ("lecture", "ai", "인공지능", "artificial intelligence", [
        ("임베딩 의미", "embedding meaning", "벡터 표현", "vector representation", "파일 확장자", "file extension"),
        ("어텐션 목적", "attention purpose", "중요 토큰 가중", "weighting important tokens", "메모리 삭제", "memory deletion"),
        ("검증셋 목적", "validation set purpose", "일반화 확인", "checking generalization", "최종 배포", "final deployment"),
        ("과적합 의미", "overfitting meaning", "훈련 데이터에 치우침", "bias toward training data", "데이터 부족 없음", "no data shortage"),
        ("학습률 의미", "learning rate meaning", "업데이트 크기", "update size", "문장 길이", "sentence length"),
        ("손실 함수 역할", "loss function role", "오차 측정", "measuring error", "데이터 저장", "storing data"),
    ]),
    ("lecture", "economics", "경제학", "economics", [
        ("수요 탄력성", "demand elasticity", "높음", "high", "낮음", "low"),
        ("기회비용 의미", "opportunity cost meaning", "포기한 대안의 가치", "value of the forgone option", "회계 장부의 총액", "total accounting amount"),
        ("한계비용 의미", "marginal cost meaning", "추가 1단위 비용", "cost of one additional unit", "고정비 총합", "total fixed cost"),
        ("물가상승률", "inflation rate", "3%", "3%", "7%", "7%"),
        ("균형가격 의미", "equilibrium price meaning", "수요와 공급이 만나는 가격", "price where demand meets supply", "정부가 항상 정하는 가격", "price always set by government"),
        ("GDP 의미", "GDP meaning", "국내총생산", "gross domestic product", "개인 저축률", "personal savings rate"),
    ]),
    ("lecture", "math", "수학", "mathematics", [
        ("극한 핵심", "limit focus", "가까워지는 값", "approaching value", "최댓값만", "maximum only"),
        ("미분 의미", "derivative meaning", "순간 변화율", "instantaneous rate of change", "전체 평균", "overall average"),
        ("행렬 곱셈 조건", "matrix multiplication condition", "앞 열과 뒤 행 일치", "first columns match second rows", "대각선만 같음", "same diagonal only"),
        ("확률 발표 범위", "probability presentation scope", "조건부 확률", "conditional probability", "복소수", "complex numbers"),
        ("벡터 내적 의미", "dot product meaning", "방향 유사도", "directional similarity", "도형 넓이만", "area only"),
        ("분산 의미", "variance meaning", "흩어진 정도", "degree of spread", "중앙값", "median"),
    ]),
    ("lecture", "network", "네트워크", "networking", [
        ("TCP 특징", "TCP property", "신뢰성", "reliability", "방송 전송", "broadcasting"),
        ("UDP 특징", "UDP property", "낮은 지연", "low latency", "순서 보장", "ordered delivery"),
        ("DNS 역할", "DNS role", "도메인 변환", "domain resolution", "파일 압축", "file compression"),
        ("TLS 목적", "TLS purpose", "통신 암호화", "encrypting communication", "이미지 편집", "image editing"),
        ("라우터 역할", "router role", "경로 선택", "route selection", "화면 렌더링", "screen rendering"),
        ("HTTP 404 의미", "HTTP 404 meaning", "찾을 수 없음", "not found", "인증 성공", "authentication success"),
    ]),
    ("lecture", "biology", "생물", "biology", [
        ("세포 호흡 위치", "cell respiration location", "미토콘드리아", "mitochondria", "리보솜", "ribosome"),
        ("효소 실험 온도", "enzyme lab temperature", "37도", "37 degrees", "25도", "25 degrees"),
        ("DNA 역할", "DNA role", "유전 정보 저장", "storing genetic information", "에너지 직접 생산", "direct energy production"),
        ("광합성 장소", "photosynthesis site", "엽록체", "chloroplast", "핵", "nucleus"),
        ("항체 역할", "antibody role", "항원 인식", "recognizing antigens", "산소 운반", "oxygen transport"),
        ("뉴런 신호", "neuron signal", "전기화학적 신호", "electrochemical signal", "단순 열전도", "simple heat conduction"),
    ]),
    ("lecture", "law", "법학", "law", [
        ("계약 성립 요건", "contract formation requirement", "청약과 승낙", "offer and acceptance", "손해배상", "damages"),
        ("개인정보 처리 근거", "personal data processing basis", "동의", "consent", "소유권", "ownership"),
        ("불법행위 요건", "tort requirement", "고의 또는 과실", "intent or negligence", "단순 호감", "simple preference"),
        ("형법의 원칙", "criminal law principle", "죄형법정주의", "principle of legality", "가격 균형", "price equilibrium"),
        ("저작권 보호 대상", "copyright protection target", "창작 표현", "creative expression", "단순 아이디어", "simple idea"),
        ("행정처분 구제", "administrative remedy", "행정심판", "administrative appeal", "상품 할인", "product discount"),
    ]),
    ("lecture", "psychology", "심리학", "psychology", [
        ("조작적 조건형성", "operant conditioning", "보상과 처벌", "reward and punishment", "단기 기억", "short-term memory"),
        ("주의 실험 변수", "attention experiment variable", "반응 시간", "reaction time", "혈압", "blood pressure"),
        ("작업기억 의미", "working memory meaning", "정보를 잠시 조작", "temporarily manipulating information", "장기 저장소", "long-term archive"),
        ("인지부하 의미", "cognitive load meaning", "처리 부담", "processing burden", "운동 속도", "movement speed"),
        ("강화의 효과", "reinforcement effect", "행동 증가", "increasing behavior", "행동 삭제", "deleting behavior"),
        ("사회적 촉진", "social facilitation", "타인 존재로 수행 변화", "performance change due to others", "혼자만의 기억", "private memory"),
    ]),
    ("lecture", "medicine", "의학", "medicine", [
        ("감염 관리 핵심", "infection control focus", "손 위생", "hand hygiene", "수면 시간", "sleep duration"),
        ("혈압 측정 자세", "blood pressure posture", "앉은 자세", "sitting position", "누운 자세", "lying position"),
        ("심폐소생술 순서", "CPR order", "가슴압박 우선", "chest compression first", "식사 우선", "meal first"),
        ("당뇨 관리 지표", "diabetes management metric", "혈당", "blood glucose", "시력", "vision"),
        ("항생제 사용 원칙", "antibiotic use principle", "필요 시 처방", "prescribed when needed", "항상 예방 복용", "always preventive use"),
        ("환자 확인 기준", "patient identification rule", "두 가지 정보 확인", "verify two identifiers", "옷 색 확인", "check clothing color"),
    ]),
    ("lecture", "design", "디자인", "design", [
        ("타이포그래피 원칙", "typography principle", "가독성", "readability", "장식성", "ornamentation"),
        ("포스터 핵심 색상", "poster key color", "남색", "navy", "분홍", "pink"),
        ("그리드 목적", "grid purpose", "배치 일관성", "layout consistency", "임의 배치", "random placement"),
        ("사용자 여정", "user journey", "사용 흐름", "usage flow", "가격표", "price tag"),
        ("대비의 목적", "contrast purpose", "중요 정보 강조", "highlight important information", "정보 숨김", "hiding information"),
        ("프로토타입 목적", "prototype purpose", "빠른 검증", "quick validation", "최종 계약", "final contract"),
    ]),
    ("meeting", "product", "제품 회의", "product meeting", [
        ("베타 배포일", "beta release date", "수요일", "Wednesday", "금요일", "Friday"),
        ("검색 기능 우선순위", "search feature priority", "우선순위 1", "priority 1", "우선순위 3", "priority 3"),
        ("온보딩 담당자", "onboarding owner", "민수", "Minsu", "지영", "Jiyoung"),
        ("핵심 지표", "core metric", "주간 활성 사용자", "weekly active users", "사무실 면적", "office area"),
        ("MVP 범위", "MVP scope", "질문 검색", "question search", "아바타 꾸미기", "avatar styling"),
        ("출시 목표", "launch target", "6월 베타", "June beta", "내년 정식", "next year official"),
    ]),
    ("meeting", "operations", "운영 회의", "operations meeting", [
        ("서버 점검 시간", "server check time", "새벽 2시", "2 a.m.", "오전 9시", "9 a.m."),
        ("장애 보고 기한", "incident report deadline", "1시간 이내", "within one hour", "하루 이내", "within one day"),
        ("배포 시간", "deploy time", "목요일", "Thursday", "월요일", "Monday"),
        ("고객 응대 담당", "customer response owner", "지영", "Jiyoung", "민수", "Minsu"),
        ("긴급 문의 응답 시간", "urgent ticket response time", "30분 이내", "within 30 minutes", "3일 이내", "within three days"),
        ("FAQ 수정 담당자", "FAQ update owner", "유진", "Yujin", "민재", "Minjae"),
    ]),
    ("meeting", "security", "보안 회의", "security meeting", [
        ("2단계 인증 적용일", "two-factor rollout date", "다음 주 월요일", "next Monday", "다음 달 금요일", "next month's Friday"),
        ("보안 점검 항목", "security check item", "접근 권한", "access permission", "폰트 크기", "font size"),
        ("NDA 보관 위치", "NDA storage location", "공유 드라이브", "shared drive", "개인 메일함", "personal mailbox"),
        ("비밀번호 정책", "password policy", "12자 이상", "at least 12 characters", "4자 고정", "fixed 4 characters"),
        ("로그 보관 기간", "log retention period", "90일", "90 days", "3일", "3 days"),
        ("권한 승인자", "permission approver", "보안팀", "security team", "마케팅팀", "marketing team"),
    ]),
    ("meeting", "education_ops", "교육 운영 회의", "education operations meeting", [
        ("보충수업 대상", "makeup class target", "결석 학생", "absent students", "전체 교직원", "all staff"),
        ("과제 피드백 방식", "assignment feedback method", "루브릭 코멘트", "rubric comments", "점수만 공개", "score only"),
        ("학습자료 위치", "learning material location", "LMS 자료실", "LMS resources", "카페 메뉴판", "cafe menu"),
        ("퀴즈 재응시 조건", "quiz retake condition", "시스템 오류", "system error", "개인 변심", "personal preference"),
        ("멘토링 시간", "mentoring time", "목요일 저녁", "Thursday evening", "월요일 새벽", "Monday dawn"),
        ("프로젝트 평가 비중", "project grading weight", "40%", "40%", "5%", "5%"),
    ]),
    ("meeting", "finance", "재무 회의", "finance meeting", [
        ("예산 검토 금액", "budget review amount", "300만원", "3 million won", "500만원", "5 million won"),
        ("광고비 증액률", "ad spend increase", "15%", "15%", "8%", "8%"),
        ("출장비 승인자", "travel cost approver", "팀장", "team lead", "인턴", "intern"),
        ("계약 승인 기한", "contract approval deadline", "금요일", "Friday", "다음 달", "next month"),
        ("정산 주기", "settlement cycle", "매주 금요일", "every Friday", "매년 1회", "once a year"),
        ("비용 절감 목표", "cost reduction target", "10%", "10%", "1%", "1%"),
    ]),
    ("meeting", "research", "연구 회의", "research meeting", [
        ("평가지표", "evaluation metric", "재현율", "recall", "정밀도", "precision"),
        ("논문 초안 담당자", "paper draft owner", "서연", "Soyeon", "도윤", "Doyoon"),
        ("실험군 A 조건", "group A condition", "새 모델 적용", "new model applied", "기존 모델 유지", "old model kept"),
        ("데이터 정제 기한", "data cleanup deadline", "화요일", "Tuesday", "일요일", "Sunday"),
        ("오류 분석 기준", "error analysis criterion", "실패 유형", "failure type", "파일 이름", "file name"),
        ("후속 실험 목표", "follow-up experiment goal", "일반화 확인", "checking generalization", "색상 비교", "color comparison"),
    ]),
    ("meeting", "marketing", "마케팅 회의", "marketing meeting", [
        ("타깃 고객", "target customer", "신입생", "new students", "졸업생", "graduates"),
        ("성과 지표", "success metric", "전환율", "conversion rate", "조회수", "view count"),
        ("이벤트 날짜", "event date", "토요일", "Saturday", "월요일", "Monday"),
        ("캠페인 문구", "campaign slogan", "짧게", "short", "길게", "long"),
        ("홍보 채널", "promotion channel", "인스타그램", "Instagram", "팩스", "fax"),
        ("쿠폰 대상", "coupon target", "신규 가입자", "new users", "탈퇴 회원", "closed accounts"),
    ]),
    ("meeting", "hospital_ops", "병원 운영 회의", "hospital operations meeting", [
        ("야간 당직 담당", "night duty owner", "A팀", "team A", "C팀", "team C"),
        ("응급실 병상 기준", "ER bed rule", "중증도 우선", "severity first", "도착 순서만", "arrival order only"),
        ("소독 점검 시간", "sanitation check time", "오전 8시", "8 a.m.", "밤 11시", "11 p.m."),
        ("환자 안내 방식", "patient guidance method", "문자 알림", "text alert", "종이 쪽지", "paper note"),
        ("검사 결과 전달", "test result delivery", "앱 알림", "app notification", "복도 게시", "hallway posting"),
        ("진료 대기 지표", "clinic wait metric", "평균 대기시간", "average wait time", "주차 대수", "parking count"),
    ]),
    ("meeting", "manufacturing", "제조 회의", "manufacturing meeting", [
        ("불량률 목표", "defect rate target", "1% 이하", "below 1%", "10% 이상", "above 10%"),
        ("라인 점검일", "line inspection date", "화요일", "Tuesday", "토요일", "Saturday"),
        ("안전 교육 담당", "safety training owner", "현장팀", "field team", "디자인팀", "design team"),
        ("품질 기준", "quality criterion", "치수 오차", "dimension tolerance", "포스터 색상", "poster color"),
        ("재고 기준", "inventory rule", "2주치 확보", "two weeks of stock", "당일 소진", "same-day depletion"),
        ("생산 우선 제품", "priority product", "모듈 A", "module A", "샘플 Z", "sample Z"),
    ]),
]


DOMAIN_FACTS.extend([
    ("lecture", "statistics", "통계학", "statistics", [
        ("귀무가설 의미", "null hypothesis meaning", "차이가 없다는 가정", "assumption of no difference", "항상 참인 결론", "always true conclusion"),
        ("유의수준", "significance level", "0.05", "0.05", "0.5", "0.5"),
        ("p값 의미", "p-value meaning", "관측 결과의 희귀성", "rarity of observed result", "평균값", "mean value"),
        ("표본분산 목적", "sample variance purpose", "흩어짐 측정", "measuring spread", "순위 정렬", "ranking"),
        ("신뢰구간 의미", "confidence interval meaning", "모수 추정 범위", "parameter estimate range", "정답 하나", "single answer"),
        ("상관계수 의미", "correlation coefficient meaning", "선형 관계 강도", "strength of linear relation", "원인 증명", "proof of causation"),
    ]),
    ("lecture", "chemistry", "화학", "chemistry", [
        ("산화 반응", "oxidation reaction", "전자를 잃는 과정", "loss of electrons", "전자를 얻는 과정", "gain of electrons"),
        ("환원 반응", "reduction reaction", "전자를 얻는 과정", "gain of electrons", "전자를 잃는 과정", "loss of electrons"),
        ("몰 농도 의미", "molar concentration meaning", "용액 1L당 몰수", "moles per liter", "분자 질량", "molecular mass"),
        ("촉매 역할", "catalyst role", "반응 속도 변화", "changing reaction rate", "반응물 소비", "being consumed as reactant"),
        ("pH 7 의미", "pH 7 meaning", "중성", "neutral", "강산성", "strong acid"),
        ("공유결합 의미", "covalent bond meaning", "전자쌍 공유", "sharing electron pairs", "전자 완전 이동", "complete electron transfer"),
    ]),
    ("lecture", "physics", "물리", "physics", [
        ("뉴턴 제2법칙", "Newton's second law", "힘은 질량 곱하기 가속도", "force equals mass times acceleration", "속도는 항상 일정", "speed is always constant"),
        ("운동량 의미", "momentum meaning", "질량과 속도의 곱", "mass times velocity", "위치와 시간의 합", "position plus time"),
        ("전압 의미", "voltage meaning", "전위차", "electric potential difference", "전류의 총량", "total current"),
        ("파장 의미", "wavelength meaning", "파동 한 주기의 길이", "length of one wave cycle", "소리의 크기", "sound volume"),
        ("마찰력 방향", "friction direction", "운동을 방해하는 방향", "opposes motion", "항상 아래쪽", "always downward"),
        ("에너지 보존", "energy conservation", "총 에너지는 일정", "total energy remains constant", "질량이 사라짐", "mass disappears"),
    ]),
    ("lecture", "history", "역사", "history", [
        ("르네상스 핵심", "Renaissance focus", "인문주의", "humanism", "절대왕정", "absolute monarchy"),
        ("산업혁명 변화", "Industrial Revolution change", "기계 생산 확대", "expanded machine production", "봉건제 강화", "stronger feudalism"),
        ("프랑스 혁명 구호", "French Revolution slogan", "자유 평등 박애", "liberty equality fraternity", "정복과 복종", "conquest and obedience"),
        ("냉전 특징", "Cold War feature", "이념 대립", "ideological conflict", "단일 제국", "single empire"),
        ("개항의 영향", "port opening impact", "무역 확대", "expanded trade", "농업만 유지", "only farming maintained"),
        ("근대화 과제", "modernization task", "제도 개혁", "institutional reform", "문자 폐지", "abolishing writing"),
    ]),
    ("lecture", "literature", "문학", "literature", [
        ("화자의 정서", "speaker emotion", "그리움", "longing", "분노", "anger"),
        ("서술 시점", "narrative point of view", "1인칭", "first person", "3인칭", "third person"),
        ("상징 소재", "symbolic object", "갈대", "reed", "바위", "rock"),
        ("작품 주제", "work theme", "상실 회복", "recovery from loss", "권력 비판", "critique of power"),
        ("비유법", "figurative device", "은유", "metaphor", "반어", "irony"),
        ("갈등 유형", "conflict type", "내적 갈등", "internal conflict", "사회적 갈등", "social conflict"),
    ]),
    ("lecture", "english", "영어", "English", [
        ("에세이 초안 마감일", "essay draft deadline", "금요일", "Friday", "화요일", "Tuesday"),
        ("토론 주제", "debate topic", "환경", "environment", "기술", "technology"),
        ("단어 시험 범위", "vocab quiz scope", "unit 4", "unit 4", "unit 9", "unit 9"),
        ("발음 과제", "pronunciation task", "강세 연습", "stress practice", "번역 제출", "translation submission"),
        ("읽기 전략", "reading strategy", "주제문 찾기", "finding topic sentences", "철자 세기", "counting spelling"),
        ("발표 평가 기준", "presentation criterion", "논리성", "logical flow", "목소리 크기만", "voice volume only"),
    ]),
    ("lecture", "education", "교육학", "education", [
        ("형성평가 목적", "formative assessment purpose", "학습 과정 점검", "checking learning progress", "최종 등급 산출", "final grade calculation"),
        ("피드백 제출 방식", "feedback submission method", "LMS 댓글", "LMS comment", "종이 설문", "paper survey"),
        ("협동학습 핵심", "cooperative learning focus", "상호 의존", "interdependence", "개별 경쟁", "individual competition"),
        ("수업 목표 작성", "lesson objective writing", "관찰 가능한 행동", "observable behavior", "추상적 소망", "abstract wish"),
        ("루브릭 역할", "rubric role", "평가 기준 명확화", "clarifying criteria", "출석 대체", "replacing attendance"),
        ("플립러닝 특징", "flipped learning feature", "사전 학습 후 활동", "pre-study then activity", "시험만 반복", "exam repetition only"),
    ]),
    ("lecture", "philosophy", "철학", "philosophy", [
        ("공리주의 기준", "utilitarian criterion", "최대 행복", "greatest happiness", "혈통", "lineage"),
        ("의무론 기준", "deontology criterion", "도덕 법칙", "moral law", "시장 가격", "market price"),
        ("실존주의 핵심", "existentialism focus", "선택과 책임", "choice and responsibility", "기계 효율", "machine efficiency"),
        ("인식론 질문", "epistemology question", "앎의 조건", "conditions of knowledge", "건물 높이", "building height"),
        ("덕 윤리 초점", "virtue ethics focus", "성품", "character", "소득세", "income tax"),
        ("사회계약론 의미", "social contract meaning", "합의된 권위", "agreed authority", "무작위 추첨", "random drawing"),
    ]),
    ("lecture", "environment", "환경학", "environmental science", [
        ("탄소중립 목표", "carbon neutrality goal", "순배출 제로", "net zero emissions", "폐기물 증가", "more waste"),
        ("생물다양성 의미", "biodiversity meaning", "종 다양성", "species diversity", "단일 작물", "single crop"),
        ("미세먼지 원인", "fine dust cause", "연소 배출", "combustion emissions", "조수 간만", "tidal change"),
        ("재활용 핵심", "recycling focus", "분리배출", "separate disposal", "무작위 폐기", "random disposal"),
        ("기후 적응", "climate adaptation", "피해 줄이기", "reducing damage", "배출 늘리기", "increasing emissions"),
        ("수질 지표", "water quality indicator", "용존 산소", "dissolved oxygen", "화면 밝기", "screen brightness"),
    ]),
    ("lecture", "business", "경영학", "business", [
        ("SWOT의 O", "O in SWOT", "기회", "opportunity", "위협", "threat"),
        ("고객 세그먼트", "customer segment", "초기 사용자", "early adopters", "공급업체", "suppliers"),
        ("가치제안 의미", "value proposition meaning", "고객에게 주는 가치", "value delivered to customers", "재무제표 형식", "financial statement format"),
        ("손익분기점", "break-even point", "수익과 비용이 같은 지점", "where revenue equals cost", "최대 매출일", "day of maximum sales"),
        ("브랜드 포지셔닝", "brand positioning", "고객 인식 속 위치", "place in customer perception", "창고 위치", "warehouse location"),
        ("리텐션 의미", "retention meaning", "사용자 유지", "keeping users", "사용자 삭제", "removing users"),
    ]),
    ("lecture", "art", "예술사", "art history", [
        ("인상주의 특징", "impressionism feature", "빛의 순간", "momentary light", "정확한 해부", "precise anatomy"),
        ("입체주의 특징", "cubism feature", "여러 시점", "multiple viewpoints", "단일 원근", "single perspective"),
        ("바로크 특징", "baroque feature", "극적 명암", "dramatic contrast", "평면 장식", "flat decoration"),
        ("미니멀리즘 특징", "minimalism feature", "단순한 형태", "simple forms", "복잡한 서사", "complex narrative"),
        ("색채 대비 목적", "color contrast purpose", "시선 집중", "drawing attention", "정보 삭제", "removing information"),
        ("원근법 목적", "perspective purpose", "깊이 표현", "expressing depth", "소리 증폭", "sound amplification"),
    ]),
    ("meeting", "startup", "스타트업 회의", "startup meeting", [
        ("이번 피벗 방향", "pivot direction", "B2B 교육", "B2B education", "게임 광고", "game ads"),
        ("투자자 미팅일", "investor meeting date", "목요일", "Thursday", "일요일", "Sunday"),
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
    ]),
    ("meeting", "hr", "인사 회의", "HR meeting", [
        ("신입 온보딩 담당자", "new hire onboarding owner", "하은", "Haeun", "지훈", "Jihoon"),
        ("면접 일정", "interview schedule", "목요일 오전", "Thursday morning", "월요일 오후", "Monday afternoon"),
        ("휴가 승인 기준", "vacation approval rule", "팀장 승인", "manager approval", "동료 투표", "peer vote"),
        ("교육 이수 기한", "training completion deadline", "2주 이내", "within two weeks", "6개월 이내", "within six months"),
        ("성과 면담 방식", "performance review method", "1대1 면담", "one-on-one meeting", "단체 공지", "group notice"),
        ("채용 우선 직무", "priority hiring role", "백엔드 개발자", "backend developer", "행사 진행자", "event host"),
    ]),
    ("meeting", "legal", "법무 회의", "legal meeting", [
        ("계약 검토 담당", "contract review owner", "법무팀", "legal team", "마케팅팀", "marketing team"),
        ("NDA 보관 위치", "NDA storage location", "공유 드라이브", "shared drive", "개인 메일함", "personal mailbox"),
        ("개인정보 조항 검토", "privacy clause review", "필수", "required", "생략", "omitted"),
        ("분쟁 대응 창구", "dispute response channel", "법무 메일", "legal email", "SNS 댓글", "social comments"),
        ("계약 갱신 알림", "contract renewal alert", "30일 전", "30 days before", "당일", "same day"),
        ("표준계약서 버전", "standard contract version", "v3", "v3", "v1", "v1"),
    ]),
    ("meeting", "sales", "영업 회의", "sales meeting", [
        ("이번 달 목표", "monthly target", "20건", "20 deals", "5건", "5 deals"),
        ("우선 고객군", "priority customer group", "대학 연구실", "university labs", "동네 카페", "local cafes"),
        ("후속 연락일", "follow-up date", "수요일", "Wednesday", "토요일", "Saturday"),
        ("제안서 담당", "proposal owner", "영업 2팀", "sales team 2", "디자인팀", "design team"),
        ("할인 승인 기준", "discount approval rule", "10% 초과", "over 10%", "모든 요청", "all requests"),
        ("CRM 입력 기한", "CRM entry deadline", "당일", "same day", "월말", "month-end"),
    ]),
    ("meeting", "policy", "정책 회의", "policy meeting", [
        ("민원 처리 기한", "complaint handling deadline", "7일 이내", "within seven days", "60일 이내", "within sixty days"),
        ("공청회 장소", "public hearing venue", "시청 강당", "city hall auditorium", "체육관 창고", "gym storage"),
        ("우선 지원 대상", "priority support target", "소상공인", "small businesses", "대기업", "large corporations"),
        ("성과 평가 기준", "performance evaluation rule", "참여율", "participation rate", "건물 색상", "building color"),
        ("예산 배정 원칙", "budget allocation rule", "취약지역 우선", "vulnerable areas first", "무작위 배정", "random allocation"),
        ("자료 공개 방식", "data disclosure method", "홈페이지 게시", "website posting", "비공개 보관", "private storage"),
    ]),
    ("meeting", "commerce", "커머스 회의", "commerce meeting", [
        ("쿠폰 적용 대상", "coupon target", "신규 가입자", "new users", "탈퇴 회원", "closed accounts"),
        ("무료배송 기준", "free shipping threshold", "3만원 이상", "over 30,000 won", "100만원 이상", "over 1,000,000 won"),
        ("반품 처리 기한", "return handling deadline", "3영업일", "three business days", "30영업일", "thirty business days"),
        ("추천 상품 기준", "recommendation criterion", "최근 조회", "recent views", "키보드 색상", "keyboard color"),
        ("재입고 알림 방식", "restock alert method", "앱 푸시", "app push", "우편 발송", "postal mail"),
        ("리뷰 노출 기준", "review display rule", "구매 인증", "verified purchase", "닉네임 길이", "nickname length"),
    ]),
    ("meeting", "content", "콘텐츠 회의", "content meeting", [
        ("이번 영상 주제", "video topic", "학습 루틴", "study routine", "주차 요금", "parking fee"),
        ("썸네일 색상", "thumbnail color", "노랑", "yellow", "회색", "gray"),
        ("업로드 요일", "upload day", "수요일", "Wednesday", "일요일", "Sunday"),
        ("대본 담당자", "script owner", "하린", "Harin", "민석", "Minseok"),
        ("검수 기준", "review criterion", "사실 오류", "factual errors", "배경음 길이", "background music length"),
        ("쇼츠 길이", "shorts length", "45초", "45 seconds", "5분", "5 minutes"),
    ]),
    ("meeting", "community", "커뮤니티 회의", "community meeting", [
        ("공지 우선순위", "announcement priority", "일정 변경", "schedule changes", "잡담 모음", "chat highlights"),
        ("신고 처리 기준", "report handling rule", "욕설 여부", "abusive language", "글자 수", "character count"),
        ("이벤트 보상", "event reward", "포인트 500점", "500 points", "빈 배지", "empty badge"),
        ("운영자 당번", "moderator shift", "서준", "Seojun", "나래", "Narae"),
        ("새 멤버 안내", "new member guide", "규칙 요약", "rule summary", "광고 링크", "ad links"),
        ("피드백 수집 방식", "feedback collection method", "익명 폼", "anonymous form", "전화 통화", "phone calls"),
    ]),
    ("meeting", "data", "데이터 회의", "data meeting", [
        ("라벨링 기준", "labeling rule", "교수 발화 우선", "professor utterance first", "학생 농담 우선", "student joke first"),
        ("중복 제거 기준", "deduplication rule", "같은 passage 제거", "remove same passage", "같은 길이 제거", "remove same length"),
        ("검수 샘플 비율", "review sample ratio", "10%", "10%", "1%", "1%"),
        ("오류 태그", "error tag", "근거 불일치", "evidence mismatch", "글꼴 불일치", "font mismatch"),
        ("데이터 공개 범위", "data release scope", "비식별 샘플", "de-identified samples", "원본 전체", "full raw data"),
        ("품질 지표", "quality metric", "정답 포함률", "answer coverage", "파일 크기", "file size"),
    ]),
    ("meeting", "qa", "QA 회의", "QA meeting", [
        ("회귀 테스트 범위", "regression test scope", "로그인과 채팅", "login and chat", "배경 음악", "background music"),
        ("버그 우선순위 기준", "bug priority rule", "사용자 차단 여부", "whether it blocks users", "아이콘 모양", "icon shape"),
        ("테스트 담당자", "test owner", "QA 1팀", "QA team 1", "영업팀", "sales team"),
        ("릴리즈 승인 조건", "release approval condition", "치명 버그 0개", "zero critical bugs", "댓글 10개", "ten comments"),
        ("자동화 대상", "automation target", "반복 로그인", "repeated login", "로고 선택", "logo selection"),
        ("리포트 제출 시간", "report submission time", "금요일 오후", "Friday afternoon", "월요일 새벽", "Monday dawn"),
    ]),
])


def fact_value(fact: tuple[str, str, str, str, str, str], lang: str, flipped: bool) -> tuple[str, str]:
    item_ko, item_en, a_ko, a_en, b_ko, b_en = fact
    item = item_ko if lang == "ko" else item_en
    value = (b_ko if lang == "ko" else b_en) if flipped else (a_ko if lang == "ko" else a_en)
    return item, value


def sentence(domain: str, item: str, value: str, lang: str) -> str:
    if lang == "ko":
        return f"{domain}에서 {item}: {value}"
    return f"in {domain}, {item} is {value}"


def build_passage(domain: str, speaker: str, chosen: list[tuple[str, str, str, str, str, str]], lang: str, flipped: bool) -> str:
    parts = [sentence(domain, *fact_value(fact, lang, flipped), lang) for fact in chosen]
    if lang == "ko":
        return f"{speaker}의 설명 내용은 다음과 같다. " + "; ".join(parts) + "."
    return f"{speaker} explained the following points: " + "; ".join(parts) + "."


def build_answer(chosen: list[tuple[str, str, str, str, str, str]], lang: str, flipped: bool) -> str:
    pairs = [f"{item}: {value}" for item, value in (fact_value(fact, lang, flipped) for fact in chosen)]
    return "; ".join(pairs)


def build_question(domain: str, speaker: str, lang: str) -> str:
    if lang == "ko":
        return f"{speaker}의 {domain} 설명에서 핵심 내용은 뭐야?"
    return f"What key points did {speaker} explain in {domain}?"


def build_rows(rows_per_domain: int, facts_per_passage: int, seed: int, limit: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for domain_idx, (group, domain_key, domain_ko, domain_en, facts) in enumerate(DOMAIN_FACTS):
        if len(facts) < facts_per_passage:
            continue
        for variant in range(rows_per_domain):
            start = variant % len(facts)
            chosen = [facts[(start + offset) % len(facts)] for offset in range(facts_per_passage)]
            if variant % 3 == 2:
                chosen = rng.sample(facts, facts_per_passage)
            for lang in ("ko", "en"):
                domain = domain_ko if lang == "ko" else domain_en
                speaker = SPEAKERS[lang][(domain_idx + variant) % len(SPEAKERS[lang])]
                source_id = f"multifact_{group}_{lang}_{domain_key}_{domain_idx}_{variant}"
                rows.append({
                    "source_id": source_id,
                    "speaker": speaker,
                    "task": "multifact_service_memory",
                    "question": build_question(domain, speaker, lang),
                    "answer": build_answer(chosen, lang, flipped=False),
                    "passage": build_passage(domain, speaker, chosen, lang, flipped=False),
                    "hard_negatives": [{
                        "passage": build_passage(domain, speaker, chosen, lang, flipped=True),
                        "answer": build_answer(chosen, lang, flipped=True),
                    }],
                })
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rows-per-domain", type=int, default=60)
    parser.add_argument("--facts-per-passage", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = build_rows(
        rows_per_domain=args.rows_per_domain,
        facts_per_passage=args.facts_per_passage,
        seed=args.seed,
        limit=args.limit,
    )
    write_jsonl(args.output, rows)
    print(
        f"[PRAG:multifact-sources] rows={len(rows)} domains={len(DOMAIN_FACTS)} "
        f"facts_per_passage={args.facts_per_passage} -> {args.output}"
    )


if __name__ == "__main__":
    main()
