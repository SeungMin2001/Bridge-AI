"""
일정 추출 서비스

파이프라인:
  1. 세션 전사문을 LLM에 전달
  2. LLM이 일정 관련 문장(시험, 과제, 발표 등)을 추출하여 JSON으로 반환
  3. 이전에 무시(ignored)된 동일 일정은 필터링하여 알림에서 제외
  4. 나머지 일정을 DB에 pending 상태로 저장
  5. 프론트에서 확정/무시 선택 → calendar_flag 업데이트

LLM 추출 JSON 구조:
[
  {
    "title": "중간고사",
    "description": "데이터베이스 중간고사 시험",
    "event_type": "시험",
    "due_date": "YYYY-MM-DDTHH:MM:SS",
    "source_text": "전사문에서 일정이 언급된 원문"
  }
]
"""
import json
import re
import logging
import httpx
import os
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
DEMO_PIPELINE_LOG = True


def _demo_log(stage: str, message: str) -> None:
    if DEMO_PIPELINE_LOG:
        print(f"[DEMO:{stage}] {message}", flush=True)


def _preview(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact if len(compact) <= limit else f"{compact[:limit - 3]}..."

# ── 설정 ──
MOCK_MODE = os.getenv("SCHEDULE_MOCK_MODE", "false").lower() == "true"

DEFAULT_LLM_URL = "http://localhost:8001"
DEFAULT_LLM_MODEL = "bridgeprag-qwen25-3b-kv64"

LLM_URL = os.getenv("LLM_URL", DEFAULT_LLM_URL)
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
LLM_API_KEY = os.getenv("LLM_API_KEY", "test-key")
SCHEDULE_SEMANTIC_DUP_FILTER = os.getenv("SCHEDULE_SEMANTIC_DUP_FILTER", "0").strip().lower() in {"1", "true", "yes", "on"}
SCHEDULE_RULE_FIRST = os.getenv("SCHEDULE_RULE_FIRST", "true").strip().lower() in {"1", "true", "yes", "on"}


#  LLM 프롬프트 템플릿
SCHEDULE_SYSTEM_PROMPT = """당신은 대학 강의 전사문에서 **학사 일정**만 추출하는 AI 비서입니다.

추출 대상 (학사 일정만):
- 시험(중간고사, 기말고사, 퀴즈, 쪽지시험)
- 과제(보고서, 레포트, 리포트, 제출, 마감)
- 프로젝트(팀플, 팀프로젝트, 설계)
- 발표(중간발표, 최종발표, 세미나)
- 수업 관련(보강, 휴강, 실습, 특강)

절대 추출하지 마세요 (비학사 일정):
- 친구 만남, 약속, 모임, 여행, 식사 등 개인 일정
- 잡담, 인사, 간식, 농담
- 일반적인 개념 설명, 학습 조언
- 날짜가 없는 할 일이나 계획

핵심 규칙:
- 전사문에 명시된 날짜/시간/마감 표현이 있는 항목만 추출하세요.
- 전사문에 없는 시험/과제/날짜를 추측하거나 만들어내면 안 됩니다.
- event_type은 전사문 원문 내용을 기반으로 분류하세요. 원문에 "시험"이 없으면 "시험"으로 분류하지 마세요.
- 제목은 전사문에 실제로 나온 표현만 사용하세요. 새로 만들지 마세요.
- 반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 텍스트는 절대 포함하지 마세요.
- 일정이 없으면 빈 배열 []을 반환하세요."""

SCHEDULE_USER_PROMPT_TEMPLATE = """오늘 날짜는 {today}입니다.

아래는 강의 전사문입니다:

{transcript_text}

위 전사문에서 **학사 일정**만 찾아 아래 JSON 배열 형식으로 추출하세요.

추출 규칙:
- 시험, 과제, 발표, 프로젝트, 보강처럼 학사와 관련된 항목만 추출하세요.
- 친구 만남, 개인 약속, 여행, 식사 등 비학사 일정은 절대 추출하지 마세요.
- 반드시 날짜나 시간이 있는 문장만 추출하세요. 날짜/시간이 없으면 제외하세요.
- "내일", "다음 주" 같은 상대 날짜는 오늘 날짜를 기준으로 YYYY-MM-DDTHH:MM:SS로 바꾸세요.
- 제목(title)은 전사문에 실제로 나온 과목명/대상을 포함해 작성하세요.
- 제목에 전사문에 없는 과목명, 약어, 주제어를 만들지 마세요.
- event_type은 전사문 원문에 해당 키워드가 있을 때만 해당 타입으로 분류하세요:
  - "시험" → 원문에 시험/고사/퀴즈가 있을 때만
  - "과제" → 원문에 과제/제출/마감/보고서/레포트가 있을 때만
  - "프로젝트" → 원문에 프로젝트/팀플/설계가 있을 때만
  - "발표" → 원문에 발표/세미나가 있을 때만
  - "기타" → 위에 해당하지 않는 학사 일정 (보강, 휴강, 실습 등)
- source_text는 반드시 전사문에 실제로 나온 문장이어야 합니다.
- 같은 일정이 여러 번 언급되면 1개만 추출하세요.
- 일정이 없으면 []만 반환하세요.
날짜 형식은 "YYYY-MM-DD" 또는 "YYYY-MM-DDTHH:MM:SS"로 작성하세요.

[
  {{
    "title": "일정 제목 (전사문 원문 기반, 간결하게)",
    "description": "일정에 대한 상세 설명",
    "event_type": "시험|과제|프로젝트|발표|기타",
    "due_date": "YYYY-MM-DDTHH:MM:SS" 또는 null,
    "source_text": "전사문에서 해당 일정이 언급된 원문 문장"
  }}
]"""


# 청크 분할 설정
CHUNK_SIZE = 4000       # 각 청크의 최대 글자 수
CHUNK_OVERLAP = 500     # 청크 간 오버랩 글자 수 (경계 일정 누락 방지)


def _split_transcript_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    긴 전사문을 오버랩이 있는 청크로 분할한다.
    경계에 걸린 일정이 누락되지 않도록 overlap을 적용한다.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def _build_schedule_prompt(transcript_text: str) -> list[dict]:
    """LLM에 보낼 일정 추출 프롬프트 messages 배열을 구성한다."""
    user_prompt = SCHEDULE_USER_PROMPT_TEMPLATE.format(
        today=datetime.now().strftime("%Y-%m-%d"),
        transcript_text=transcript_text
    )
    return [
        {"role": "system", "content": SCHEDULE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_schedule_json(raw_text: str) -> list[dict]:
    """
    LLM 응답에서 일정 JSON 배열을 추출하여 파싱한다.
    <think> 태그, 마크다운 코드블록 등의 노이즈를 자동 제거한다.
    """
    # <think>...</think> 제거
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    # 마크다운 코드블록 제거
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # JSON 배열 추출
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        schedules = json.loads(text)
    except json.JSONDecodeError as first_error:
        try:
            schedules = json.loads(_escape_control_chars_in_json_strings(text))
        except json.JSONDecodeError as e:
            logger.error(f"일정 JSON 파싱 실패: {e}\n원본: {raw_text[:500]}")
            raise ValueError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {first_error}")

    if not isinstance(schedules, list):
        raise ValueError("LLM 응답이 JSON 배열이 아닙니다")

    # 필수 필드 기본값 보장
    for s in schedules:
        s.setdefault("title", "제목 없음")
        s.setdefault("description", "")
        s.setdefault("event_type", "기타")
        s.setdefault("due_date", None)
        s.setdefault("source_text", "")

    return _filter_valid_schedules(schedules)


def _escape_control_chars_in_json_strings(text: str) -> str:
    """LLM이 JSON 문자열 안에 실제 줄바꿈을 넣은 경우 유효한 JSON으로 보정한다."""
    result = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                result.append(char)
                escaped = False
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                continue
            if char == '"':
                result.append(char)
                in_string = False
                continue
            if char == "\n":
                result.append("\\n")
                continue
            if char == "\r":
                result.append("\\r")
                continue
            if char == "\t":
                result.append("\\t")
                continue

            result.append(char)
            continue

        result.append(char)
        if char == '"':
            in_string = True

    return "".join(result)


ACADEMIC_EVENT_KEYWORDS = (
    "과제", "제출", "마감", "보고서", "레포트", "리포트",
    "시험", "고사", "중간고사", "기말고사", "퀴즈", "쪽지시험",
    "발표", "프로젝트", "팀플", "회의",
    "수업", "보강", "휴강", "실습", "특강"
)

GENERIC_SCHEDULE_KEYWORDS = (
    "일정", "기한", "데드라인"
)

SCHEDULE_EVENT_KEYWORDS = (
    *ACADEMIC_EVENT_KEYWORDS,
    *GENERIC_SCHEDULE_KEYWORDS,
)

TITLE_GENERIC_WORDS = (
    "일정", "과제", "제출", "마감", "보고서", "레포트", "리포트",
    "시험", "고사", "퀴즈", "발표", "프로젝트", "회의",
    "수업", "보강", "휴강", "실습", "특강", "중간", "기말",
    "중간고사", "기말고사", "기한", "데드라인", "날짜"
)

ASSIGNMENT_CONTEXT_HINT_PATTERN = re.compile(
    r"(이\s*내용|오늘\s*내용|이번\s*내용|배운\s*내용|수업\s*내용|강의\s*내용|해당\s*내용|앞에서\s*배운\s*내용)"
)

SCHEDULE_DETAIL_CONTEXT_PATTERN = re.compile(
    r"(범위|단원|챕터|chapter|장|주제|내용|자료|준비|복습|공부|예습|읽어\s*오|읽어오|정리)"
)

SCHEDULE_COMMITMENT_PATTERN = re.compile(
    r"(봅니다|보겠습니다|치릅니다|치르겠습니다|진행|진행합니다|예정|있습니다|있어요|"
    r"제출|마감|까지|전까지|발표|시작|끝납니다|열립니다)"
)

SCHEDULE_CONTEXT_ONLY_PATTERN = re.compile(
    r"(설명|배웠|학습|정리|살펴|공부|다뤘|범위|단원|내용|자료|준비)"
)

EXPLICIT_CALENDAR_DATE_PATTERN = re.compile(
    r"(\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
    r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|"
    r"\d{1,2}\s*월\s*\d{1,2}\s*일|"
    r"\d{1,2}[./]\d{1,2}|"
    r"내일|모레|다음\s*달\s*\d{1,2}\s*일|"
    r"다다음\s*주|다음\s*주|다다음주|다음주|"
    r"월요일|화요일|수요일|목요일|금요일|토요일|일요일|"
    r"까지|전까지|마감)"
)

LECTURE_TOPIC_END_PATTERN = re.compile(
    r"(?:에\s*대해\s*)?(?:배웠습니다|배웠어요|학습했습니다|학습했어요|다뤘습니다|다루었습니다|설명했습니다|설명했어요|정리했습니다|살펴봤습니다|공부했습니다|준비해\s*주세요|준비해주세요|복습해\s*주세요|복습해주세요|공부해\s*오세요|읽어\s*오세요|읽어오세요)$"
)

SCHEDULE_DATE_HINT_PATTERN = re.compile(
    r"(\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
    r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일|"
    r"\d{1,2}\s*월\s*\d{1,2}\s*일|"
    r"\d{1,2}[./]\d{1,2}|"
    r"오늘|내일|모레|다음\s*달\s*\d{1,2}\s*일|"
    r"다다음\s*주|다음\s*주|이번\s*주|다다음주|다음주|이번주|"
    r"월요일|화요일|수요일|목요일|금요일|토요일|일요일|"
    r"오전\s*\d{1,2}\s*시|오후\s*\d{1,2}\s*시|"
    r"\d{1,2}\s*시|"
    r"오전\s*(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|열한|열두)\s*시|"
    r"오후\s*(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|열한|열두)\s*시|"
    r"\d{1,2}\s*분|까지|전까지|마감)"
)

TIME_HINT_PATTERN = re.compile(
    r"(오전|오후)?\s*(\d{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|열한|열두)\s*시"
    r"(?:\s*(\d{1,2})\s*분)?"
)
DEADLINE_HINT_PATTERN = re.compile(r"(까지|마감|제출|전까지)")

KOREAN_HOUR_WORDS = {
    "한": 1,
    "두": 2,
    "세": 3,
    "네": 4,
    "다섯": 5,
    "여섯": 6,
    "일곱": 7,
    "여덟": 8,
    "아홉": 9,
    "열": 10,
    "열한": 11,
    "열두": 12,
}


WEEKDAY_INDEX = {
    "월": 0,
    "월요일": 0,
    "화": 1,
    "화요일": 1,
    "수": 2,
    "수요일": 2,
    "목": 3,
    "목요일": 3,
    "금": 4,
    "금요일": 4,
    "토": 5,
    "토요일": 5,
    "일": 6,
    "일요일": 6,
}


# 비학사 일정 감지용 키워드
NON_ACADEMIC_KEYWORDS = (
    "친구", "만남", "약속", "모임", "여행", "식사", "밥", "카페",
    "영화", "데이트", "쇼핑", "운동", "헬스", "게임", "놀",
)


def _validate_event_type(event_type: str, source_text: str) -> str:
    """
    LLM이 분류한 event_type이 source_text 내용과 일치하는지 검증한다.
    source_text에 해당 키워드가 없으면 "기타"로 보정한다.
    """
    source_lower = source_text.lower()

    type_keywords = {
        "시험": ("시험", "고사", "퀴즈", "쪽지"),
        "과제": ("과제", "제출", "마감", "보고서", "레포트", "리포트"),
        "프로젝트": ("프로젝트", "팀플", "설계"),
        "발표": ("발표", "세미나", "프레젠테이션"),
    }

    clean_type = event_type.strip()
    if clean_type in type_keywords:
        keywords = type_keywords[clean_type]
        if not any(kw in source_lower for kw in keywords):
            logger.info(f"[SCHEDULE] event_type 보정: '{clean_type}' -> '기타' (source_text에 근거 없음)")
            return "기타"

    return clean_type


def _is_non_academic(source_text: str, title: str, description: str = "") -> bool:
    """source_text, title, description에 비학사 일정 키워드가 포함되어 있는지 판단한다.
    LLM이 제목이나 설명에 학사 키워드("시험", "과제" 등)를 환각으로 추가했을 수 있으므로,
    비학사 감지와 학사 감지는 오직 원본 텍스트(source_text)만을 기준으로 해야 신뢰할 수 있습니다.
    """
    source_lower = source_text.lower()
    has_non_academic = any(kw in source_lower for kw in NON_ACADEMIC_KEYWORDS)
    has_academic = any(kw in source_lower for kw in ACADEMIC_EVENT_KEYWORDS)
    # 비학사 키워드가 있고 학사 키워드가 없으면 비학사 일정
    return has_non_academic and not has_academic


def _split_schedule_sentences(text: str) -> list[str]:
    """전사문을 일정 후보를 찾기 좋은 짧은 문장 단위로 나눈다."""
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []

    sentences = re.findall(r"[^.!?。！？\n]+(?:[.!?。！？]+|$)", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _infer_event_type_from_text(text: str) -> str:
    """원문 키워드로 일정 유형을 보수적으로 분류한다."""
    if any(keyword in text for keyword in ("중간고사", "기말고사", "시험", "고사", "퀴즈", "쪽지시험")):
        return "시험"
    if any(keyword in text for keyword in ("과제", "제출", "마감", "보고서", "레포트", "리포트", "기한", "데드라인")):
        return "과제"
    if any(keyword in text for keyword in ("프로젝트", "팀플", "설계")):
        return "프로젝트"
    if any(keyword in text for keyword in ("발표", "세미나", "프레젠테이션")):
        return "발표"
    return "기타"


def _clean_rule_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title).strip(" .,!?;:，。！？、")
    cleaned = re.sub(r"(은|는|이|가|을|를|의|에|으로|로|도|만|까지|전까지)$", "", cleaned).strip()
    return cleaned[:48].strip() or "일정"


def _infer_rule_title(sentence: str, event_type: str) -> str:
    """명시적 일정 문장에서 화면에 표시할 간결한 제목을 뽑는다."""
    title_patterns = (
        r"([A-Za-z0-9+#.\s가-힣]{0,24}?(?:중간고사|기말고사|쪽지시험|퀴즈|시험|고사))",
        r"([A-Za-z0-9+#.\s가-힣]{0,24}?(?:과제|보고서|레포트|리포트|제출|마감))",
        r"([A-Za-z0-9+#.\s가-힣]{0,24}?(?:프로젝트|팀플|설계))",
        r"([A-Za-z0-9+#.\s가-힣]{0,24}?(?:중간발표|최종발표|발표|세미나|프레젠테이션))",
        r"([A-Za-z0-9+#.\s가-힣]{0,24}?(?:보강|휴강|실습|특강|수업|일정))",
    )
    for pattern in title_patterns:
        match = re.search(pattern, sentence)
        if match:
            return _clean_rule_title(match.group(1))

    return {
        "시험": "시험 일정",
        "과제": "과제 제출",
        "프로젝트": "프로젝트 일정",
        "발표": "발표 일정",
    }.get(event_type, "일정")


def _extract_rule_based_schedules(transcript_text: str) -> list[dict]:
    """명시적 키워드와 날짜가 있는 일정은 LLM 없이도 놓치지 않도록 추출한다."""
    candidates = []
    for sentence in _split_schedule_sentences(transcript_text):
        if not any(keyword in sentence for keyword in SCHEDULE_EVENT_KEYWORDS):
            continue
        if not SCHEDULE_DATE_HINT_PATTERN.search(sentence):
            continue

        due_date = parse_due_date(sentence)
        if due_date is None:
            continue

        event_type = _infer_event_type_from_text(sentence)
        if _is_non_academic(sentence, sentence):
            continue

        candidates.append({
            "title": _infer_rule_title(sentence, event_type),
            "description": sentence,
            "event_type": event_type,
            "due_date": due_date.isoformat(),
            "source_text": sentence,
        })

    return _filter_valid_schedules(candidates)



def _is_fuzzy_duplicate(new_item: dict, existing_items: list[dict]) -> bool:
    """
    새 일정이 기존 목록의 항목과 유사한 중복인지 판단한다.
    같은 날짜 + (제목 유사 or source_text 포함 관계)이면 중복.
    """
    new_date = new_item.get("due_date", "")
    new_title = new_item.get("title", "")
    new_source = (new_item.get("source_text") or "").replace(" ", "")

    for existing in existing_items:
        ex_date = existing.get("due_date", "")
        ex_title = existing.get("title", "")
        ex_source = (existing.get("source_text") or "").replace(" ", "")

        # 날짜가 다르면 중복 아님
        if new_date and ex_date and new_date[:10] != ex_date[:10]:
            continue

        # source_text 포함 관계 체크
        if new_source and ex_source:
            if new_source in ex_source or ex_source in new_source:
                return True

        # 제목 토큰 기반 유사도 체크 (한국어 2자 이상 단어 단위)
        if new_title and ex_title:
            new_tokens = set(re.findall(r'[\uac00-\ud7a3]{2,}|[A-Za-z]+', new_title))
            ex_tokens = set(re.findall(r'[\uac00-\ud7a3]{2,}|[A-Za-z]+', ex_title))
            if new_tokens and ex_tokens:
                intersection = len(new_tokens & ex_tokens)
                union = len(new_tokens | ex_tokens)
                similarity = intersection / union if union > 0 else 0
                if similarity >= 0.5:
                    return True

    return False


def _filter_valid_schedules(schedules: list[dict]) -> list[dict]:
    """LLM이 과하게 뽑은 후보를 저장 전에 한 번 더 걸러낸다."""
    valid = []

    for item in schedules:
        title = str(item.get("title") or "").strip()
        description = str(item.get("description") or "").strip()
        event_type = str(item.get("event_type") or "").strip()
        due_date = item.get("due_date")
        source_text = str(item.get("source_text") or "").strip()

        if not title or not source_text:
            logger.info(f"[SCHEDULE] 후보 제외: 제목/source_text 없음 - {item}")
            continue

        parsed_due_date = parse_due_date(str(due_date)) if due_date else None
        parsed_source_date = parse_due_date(source_text) if SCHEDULE_DATE_HINT_PATTERN.search(source_text) else None
        if (
            parsed_due_date is not None
            and parsed_source_date is not None
            and not _has_time_hint(str(due_date))
            and _has_time_hint(source_text)
        ):
            parsed_due_date = parsed_source_date
        elif parsed_due_date is None:
            parsed_due_date = parsed_source_date

        if parsed_due_date is None:
            logger.info(f"[SCHEDULE] 후보 제외: 파싱 가능한 날짜 없음 - {title}")
            continue
        item["due_date"] = parsed_due_date.isoformat()

        # event_type 검증 및 보정 (비학사 필터링 전에 먼저 실행)
        item["event_type"] = _validate_event_type(event_type, source_text)

        # 비학사 일정 필터링 (event_type 보정 후 실행)
        if _is_non_academic(source_text, title, description):
            logger.info(f"[SCHEDULE] 후보 제외: 비학사 일정 감지 - {title}")
            continue

        # 학사 키워드 체크 (event_type 보정 후 combined에 반영)
        combined = " ".join([title, description, item["event_type"], source_text])
        if not any(keyword in combined for keyword in SCHEDULE_EVENT_KEYWORDS):
            logger.info(f"[SCHEDULE] 후보 제외: 일정 키워드 없음 - {title}")
            continue

        if not SCHEDULE_DATE_HINT_PATTERN.search(source_text):
            logger.info(f"[SCHEDULE] 후보 제외: source_text에 날짜/마감 표현 없음 - {title}")
            continue

        item["title"] = _normalize_schedule_title(title, item["event_type"], source_text)

        # 유사도 기반 중복 제거 (같은 날짜 + 제목 유사 or source_text 포함)
        if _is_fuzzy_duplicate(item, valid):
            logger.info(f"[SCHEDULE] 후보 제외: 유사 중복 감지 - {title}")
            continue

        valid.append(item)

    return valid


def _has_time_hint(text: str | None) -> bool:
    """문장에 명시적인 시간 표현이 있는지 확인한다."""
    return bool(text and TIME_HINT_PATTERN.search(str(text)))


def _title_has_unsupported_subject(title: str, source_text: str) -> bool:
    """제목에 원문에 없는 과목명/약어가 들어가면 예시 복사를 의심한다."""
    source_compact = source_text.replace(" ", "").lower()
    title_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.-]*|[가-힣]{2,}", title)

    for token in title_tokens:
        token_lower = token.lower()
        if token in TITLE_GENERIC_WORDS:
            continue
        if token_lower not in source_compact:
            return True

    return False


def _normalize_schedule_title(title: str, event_type: str, source_text: str) -> str:
    """LLM이 프롬프트 예시의 과목명을 베껴 쓴 경우 안전한 제목으로 바꾼다."""
    clean_title = title.strip()
    clean_event_type = event_type.strip()
    if not _title_has_unsupported_subject(clean_title, source_text):
        return clean_title

    if "시험" in source_text or clean_event_type == "시험":
        safe_title = "시험 일정"
    elif any(word in source_text for word in ("과제", "제출", "마감")) or clean_event_type == "과제":
        safe_title = "과제 제출"
    elif "발표" in source_text or clean_event_type == "발표":
        safe_title = "발표 일정"
    elif "회의" in source_text or clean_event_type == "회의":
        safe_title = "회의 일정"
    elif any(word in source_text for word in ("수업", "보강")):
        safe_title = "수업 일정"
    else:
        safe_title = clean_event_type or "일정"

    logger.info(f"[SCHEDULE] 원문에 없는 제목 보정: '{clean_title}' -> '{safe_title}'")
    return safe_title


def _parse_time_number(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return KOREAN_HOUR_WORDS.get(text)


def _split_schedule_sentences(text: str) -> list[str]:
    """STT 문장을 일정 추출에 적합한 짧은 후보 문장으로 나눈다."""
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []

    pieces = re.split(r"\n+|(?<=[.!?。])\s+|(?<=다)\s+|(?<=요)\s+", normalized)
    sentences = []
    for piece in pieces:
        clean = piece.strip(" \t\r\n,")
        if not clean:
            continue
        if len(clean) > 220:
            clauses = re.split(r"\s*(?:그리고|또한|다음으로|마지막으로)\s*", clean)
            sentences.extend(clause.strip(" ,") for clause in clauses if clause.strip(" ,"))
        else:
            sentences.append(clean)
    return sentences


def _classify_schedule_from_text(source_text: str) -> str:
    """원문 키워드 기준으로 일정 유형을 안정적으로 분류한다."""
    if re.search(r"시험|고사|퀴즈|쪽지시험", source_text):
        return "시험"
    if re.search(r"과제|제출|마감|보고서|레포트|리포트", source_text):
        return "과제"
    if re.search(r"발표|세미나|프레젠테이션", source_text):
        return "발표"
    if re.search(r"프로젝트|팀플|설계", source_text):
        return "프로젝트"
    if re.search(r"수업|보강|휴강|실습|특강", source_text):
        return "기타"
    return "기타"


def _make_rule_based_title(event_type: str, source_text: str) -> str:
    """LLM 환각 없이 원문 기반의 짧은 일정 제목을 만든다."""
    if event_type == "과제":
        assignment_topic = _extract_assignment_topic(source_text)
        if assignment_topic:
            return f"{assignment_topic} 과제"
        return "과제 제출"
    if "기말고사" in source_text:
        scope = _extract_exam_scope(source_text)
        if scope:
            return f"{scope} 기말고사"
        return "기말고사"
    if "중간고사" in source_text:
        scope = _extract_exam_scope(source_text)
        if scope:
            return f"{scope} 중간고사"
        return "중간고사"
    if "쪽지시험" in source_text:
        scope = _extract_exam_scope(source_text)
        if scope:
            return f"{scope} 쪽지시험"
        return "쪽지시험"
    if "퀴즈" in source_text:
        scope = _extract_exam_scope(source_text)
        if scope:
            return f"{scope} 퀴즈"
        return "퀴즈"
    if event_type == "시험":
        scope = _extract_exam_scope(source_text)
        if scope:
            return f"{scope} 시험"
        return "시험 일정"
    if "프로젝트" in source_text and event_type == "발표":
        return "프로젝트 발표"
    if event_type == "프로젝트":
        return "프로젝트 일정"
    if event_type == "발표":
        return "발표 일정"
    if "보강" in source_text:
        return "보강 일정"
    if "휴강" in source_text:
        return "휴강 일정"
    if "실습" in source_text:
        return "실습 일정"
    return "수업 일정"


def _strip_schedule_date_phrases(text: str) -> str:
    """일정 제목 후보에서 날짜/마감 표현을 제거한다."""
    cleaned = re.sub(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", " ", text)
    cleaned = re.sub(r"\d{1,2}\s*월\s*\d{1,2}\s*일", " ", cleaned)
    cleaned = re.sub(r"(오늘|내일|모레|이번\s*주|다음\s*주|다다음\s*주)\s*(월요일|화요일|수요일|목요일|금요일|토요일|일요일|월|화|수|목|금|토|일)?", " ", cleaned)
    cleaned = re.sub(r"(오전|오후)?\s*\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?", " ", cleaned)
    cleaned = re.sub(r"(까지|마감|제출\s*기한|기한|제출해야\s*합니다|제출해야합니다|제출하세요|내세요|입니다|합니다)", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" .,，")


def _clean_context_topic(text: str) -> str:
    """이전 수업 문장에서 과제 제목으로 쓸 수 있는 핵심 주제를 정리한다."""
    cleaned = _strip_schedule_date_phrases(text)
    cleaned = re.sub(r"^(오늘|이번\s*시간|이번\s*주|이번\s*강의|수업|강의)\s*(은|는|에서|에서는)?\s*", "", cleaned)
    cleaned = re.sub(r"^(은|는|이|가|인|에서|에서는)\s*", "", cleaned)
    cleaned = re.sub(r"^(먼저|다음으로|그리고|또한|마지막으로)\s*", "", cleaned)
    cleaned = LECTURE_TOPIC_END_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"(시험\s*)?범위\s*(은|는|:)?", " ", cleaned)
    cleaned = re.sub(r"(준비|복습|공부|예습)\s*(해\s*주세요|해주세요|해\s*오세요|해오세요)?", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,，")
    cleaned = re.sub(r"(입니다|합니다|이에요|예요)$", "", cleaned).strip(" .,，")
    cleaned = re.sub(r"(을|를)$", "", cleaned).strip(" .,，")
    return cleaned[:36].rstrip()


def _is_weak_assignment_topic(topic: str) -> bool:
    compact = re.sub(r"\s+", "", topic or "")
    if len(compact) < 4:
        return True
    weak_prefixes = (
        "이내용", "오늘내용", "이번내용", "배운내용", "수업내용", "강의내용",
        "해당내용", "앞에서배운내용", "내용은", "내용을", "내용으로",
        "설명", "을설명", "를설명", "정리", "작성"
    )
    return compact.startswith(weak_prefixes) or compact in {"내용정리", "자료정리", "문제풀이"}


def _extract_context_topic(source_text: str) -> str:
    """마감 문장 주변의 강의 내용에서 '어떤 과제인지'를 보완한다."""
    for sentence in reversed(_split_schedule_sentences(source_text)):
        if re.search(r"과제|숙제|보고서|레포트|리포트|제출|마감|기한|데드라인|시험|고사|퀴즈|발표|프로젝트", sentence):
            continue
        topic = _clean_context_topic(sentence)
        if topic and not _is_weak_assignment_topic(topic):
            return topic
    return ""


def _extract_exam_scope(source_text: str) -> str:
    """시험 날짜 문장 주변에서 시험 범위나 준비 단원을 추출한다."""
    sentences = _split_schedule_sentences(source_text)
    for sentence in reversed(sentences):
        if not SCHEDULE_DETAIL_CONTEXT_PATTERN.search(sentence):
            continue

        scope_match = re.search(r"(?:범위|단원|주제|내용|자료)\s*(?:은|는|:)?\s*(.{2,80})", sentence)
        candidate = scope_match.group(1) if scope_match else sentence
        topic = _clean_context_topic(candidate)
        if not topic or _is_weak_assignment_topic(topic):
            topic = _clean_context_topic(sentence)
        if not topic:
            continue

        compact = re.sub(r"\s+", "", topic)
        if re.search(r"시험|고사|퀴즈", compact):
            continue
        if compact in {"범위", "단원", "내용", "자료", "준비"}:
            continue
        return topic[:28].rstrip()

    return ""


def _is_context_only_schedule_sentence(sentence: str) -> bool:
    """
    '오늘은 기말고사 범위를 설명했습니다' 같은 강의 내용 문장을 일정으로 오인하지 않는다.
    실제 일정은 명시 날짜/마감 표현 또는 '봅니다/진행합니다' 같은 실행 동사를 함께 가져야 한다.
    """
    if EXPLICIT_CALENDAR_DATE_PATTERN.search(sentence):
        return False
    if SCHEDULE_COMMITMENT_PATTERN.search(sentence):
        return False
    return bool(SCHEDULE_CONTEXT_ONLY_PATTERN.search(sentence))


def _compact_assignment_topic(text: str) -> str:
    """과제 내용을 알림 제목에 넣기 좋은 길이로 정리한다."""
    cleaned = _strip_schedule_date_phrases(text)
    cleaned = re.sub(r"^(저희|이번|다음|오늘|여러분|교수님이|교수님께서)\s*(은|는|이|가|의)?\s*", "", cleaned)
    cleaned = re.sub(r"^(은|는|이|가|의)\s*", "", cleaned)
    cleaned = re.sub(r"(과제|숙제|보고서|레포트|리포트)\s*(내용|주제)?\s*(은|는|입니다|이에요|으로|로|을|를|:)?", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,，")
    cleaned = re.sub(r"^(은|는|이|가|을|를|의)\s*", "", cleaned)
    cleaned = re.sub(r"(하는\s*)?것$", "", cleaned).strip(" .,，")
    cleaned = re.sub(r"(을|를)$", "", cleaned).strip(" .,，")
    if not cleaned:
        return ""
    return cleaned[:32].rstrip()


def _extract_assignment_topic(source_text: str) -> str:
    """source_text에서 '무슨 과제인지'를 뽑아 알림 제목에 반영한다."""
    sentences = _split_schedule_sentences(source_text)
    context_topic = _extract_context_topic(source_text)
    topic_candidates = []

    for sentence in sentences or [source_text]:
        if "과제" not in sentence and not re.search(r"보고서|레포트|리포트", sentence):
            continue

        before_match = re.search(r"(.{3,80}?)(?:과제|숙제|보고서|레포트|리포트)\s*(?:은|는|입니다|이에요|으로|로)?", sentence)
        if before_match:
            topic_candidates.append(before_match.group(1))

        after_match = re.search(r"(?:과제|숙제|보고서|레포트|리포트)\s*(?:내용|주제)?\s*(?:은|는|입니다|이에요|:)?\s*(.{3,100})", sentence)
        if after_match:
            topic_candidates.append(after_match.group(1))

    for candidate in topic_candidates:
        topic = _compact_assignment_topic(candidate)
        if topic and not _is_weak_assignment_topic(topic) and not re.fullmatch(r"(저희|이번|다음|오늘)?\s*", topic):
            return topic

    if context_topic:
        return context_topic

    for candidate in topic_candidates:
        topic = _compact_assignment_topic(candidate)
        if topic and not _is_weak_assignment_topic(topic) and not re.fullmatch(r"(저희|이번|다음|오늘)?\s*", topic):
            return topic

    return ""


def _needs_assignment_context(sentence: str) -> bool:
    """마감 문장만 있고 과제 내용이 부족한 경우 주변 문장을 같이 사용한다."""
    if not re.search(r"과제|제출|마감|보고서|레포트|리포트", sentence):
        return False
    return not _extract_assignment_topic(sentence)


def _build_rule_source_text(sentences: list[str], index: int, event_type: str) -> str:
    """날짜 문장에 세부 범위/내용이 부족하면 앞뒤 문맥을 붙인다."""
    sentence = sentences[index]
    context = []
    should_expand = (
        event_type == "과제"
        and (_needs_assignment_context(sentence) or ASSIGNMENT_CONTEXT_HINT_PATTERN.search(sentence) or re.search(r"제출|마감|기한|데드라인", sentence))
    )
    if event_type in {"시험", "발표", "프로젝트"}:
        should_expand = True
    if not should_expand:
        return sentence

    for prev_index in range(max(0, index - 2), index):
        prev = sentences[prev_index]
        if len(prev) <= 180 and not _is_non_academic(prev, "", "") and (
            event_type == "과제" or SCHEDULE_DETAIL_CONTEXT_PATTERN.search(prev)
        ):
            context.append(prev)
    context.append(sentence)
    if index + 1 < len(sentences):
        next_sentence = sentences[index + 1]
        if len(next_sentence) <= 180 and not _is_non_academic(next_sentence, "", "") and (
            re.search(r"과제|제출|마감|보고서|레포트|리포트", next_sentence)
            or (event_type in {"시험", "발표", "프로젝트"} and SCHEDULE_DETAIL_CONTEXT_PATTERN.search(next_sentence))
        ):
            context.append(next_sentence)
    return " ".join(context)


def _extract_rule_based_schedules(transcript_text: str) -> list[dict]:
    """
    시연 안정성을 위해 명확한 날짜+학사 키워드 문장은 LLM 전에 deterministic하게 추출한다.
    예: "저희 과제는 6월 4일까지 제출해야합니다"
    """
    candidates = []
    sentences = _split_schedule_sentences(transcript_text)
    for index, sentence in enumerate(sentences):
        if not SCHEDULE_DATE_HINT_PATTERN.search(sentence):
            continue
        if not any(keyword in sentence for keyword in SCHEDULE_EVENT_KEYWORDS):
            continue
        if _is_non_academic(sentence, "", ""):
            continue
        if _is_context_only_schedule_sentence(sentence):
            logger.info(f"[SCHEDULE] 문맥 설명 문장 제외: {sentence[:80]}")
            continue

        due_date = parse_due_date(sentence)
        if due_date is None:
            continue

        event_type = _classify_schedule_from_text(sentence)
        source_text = _build_rule_source_text(sentences, index, event_type)
        title = _make_rule_based_title(event_type, source_text)
        candidates.append({
            "title": title,
            "description": source_text,
            "event_type": event_type,
            "due_date": due_date.isoformat(),
            "source_text": source_text,
        })

    return _filter_valid_schedules(candidates)


def _parse_time_from_text(text: str) -> tuple[int, int]:
    """문장 안의 한국어 시간 표현을 찾고, 없으면 오전 9시로 둔다."""
    hour = 9
    minute = 0
    time_match = TIME_HINT_PATTERN.search(text)
    if not time_match:
        return hour, minute

    meridiem, raw_hour, raw_minute = time_match.groups()
    parsed_hour = _parse_time_number(raw_hour)
    if parsed_hour is None:
        return hour, minute

    hour = parsed_hour
    minute = int(raw_minute or 0)
    if meridiem == "오후" and hour != 12:
        hour += 12
    if meridiem == "오전" and hour == 12:
        hour = 0
    return hour, minute


def _default_time_for_text(text: str) -> tuple[int, int]:
    """시간이 없는 마감/제출 문장은 하루 끝으로, 그 외 일정은 오전 9시로 보정한다."""
    if DEADLINE_HINT_PATTERN.search(text) and not _has_time_hint(text):
        return 23, 59
    return 9, 0


def _build_date_with_time(year: int, month: int, day: int, source_text: str) -> datetime | None:
    hour, minute = _parse_time_from_text(source_text) if _has_time_hint(source_text) else _default_time_for_text(source_text)
    try:
        return datetime(year, month, day, hour, minute, 0)
    except ValueError:
        return None


def _parse_explicit_date_in_text(text: str) -> datetime | None:
    """문장 중간에 포함된 명시적 날짜 표현을 찾아 datetime으로 변환한다."""
    if not text:
        return None

    clean = str(text).strip()
    hour, minute = _parse_time_from_text(clean) if _has_time_hint(clean) else _default_time_for_text(clean)

    ymd_match = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", clean)
    if ymd_match:
        year, month, day = map(int, ymd_match.groups())
        try:
            return datetime(year, month, day, hour, minute, 0)
        except ValueError:
            return None

    korean_match = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", clean)
    if korean_match:
        month, day = map(int, korean_match.groups())
        try:
            return datetime.now().replace(month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            return None

    slash_match = re.search(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)", clean)
    if slash_match:
        month, day = map(int, slash_match.groups())
        try:
            return datetime.now().replace(month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            return None

    return None


def _parse_weekday_relative_date(text: str) -> datetime | None:
    """'이번주 일요일', '다음 주 목요일', '다다음주 월요일' 같은 표현을 날짜로 변환한다."""
    match = re.search(
        r"(이번|다다음|다음)\s*주.{0,30}?"
        r"(월요일|화요일|수요일|목요일|금요일|토요일|일요일|월|화|수|목|금|토|일)",
        text,
    )
    if not match:
        return None

    week_word, weekday_text = match.groups()
    target_weekday = WEEKDAY_INDEX[weekday_text]
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())
    if week_word == "다음":
        week_start += timedelta(days=7)
    elif week_word == "다다음":
        week_start += timedelta(days=14)

    target_date = week_start + timedelta(days=target_weekday)
    hour, minute = _parse_time_from_text(text)
    return target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _parse_standalone_weekday_date(text: str) -> datetime | None:
    """'금요일 오후 6시까지'처럼 주차 표현 없이 나온 요일을 가장 가까운 해당 요일로 변환한다."""
    match = re.search(r"(월요일|화요일|수요일|목요일|금요일|토요일|일요일)", text)
    if not match:
        return None

    target_weekday = WEEKDAY_INDEX[match.group(1)]
    today = datetime.now()
    days_ahead = (target_weekday - today.weekday()) % 7
    target_date = today + timedelta(days=days_ahead)
    hour, minute = _parse_time_from_text(text)
    return target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)



def parse_due_date(date_str: str | None) -> datetime | None:
    """
    다양한 날짜 포맷을 datetime으로 변환한다.
    LLM이 반환하는 날짜 형식이 일정하지 않을 수 있으므로 여러 포맷을 시도한다.
    """
    if not date_str:
        return None

    normalized = str(date_str).strip()
    normalized = normalized.replace("오정", "오전")

    week_relative_date = _parse_weekday_relative_date(normalized)
    if week_relative_date is not None:
        return week_relative_date

    today = datetime.now()

    next_month_match = re.search(r"다음\s*달\s*(\d{1,2})\s*일", normalized)
    if next_month_match:
        month = today.month + 1
        year = today.year
        if month > 12:
            month = 1
            year += 1
        parsed = _build_date_with_time(year, month, int(next_month_match.group(1)), normalized)
        if parsed is not None:
            return parsed

    relative_base = None
    if "오늘" in normalized:
        relative_base = today
    elif "내일" in normalized:
        relative_base = today + timedelta(days=1)
    elif "모레" in normalized:
        relative_base = today + timedelta(days=2)

    if relative_base is not None:
        hour, minute = _parse_time_from_text(normalized)
        return relative_base.replace(hour=hour, minute=minute, second=0, microsecond=0)

    exact_formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]
    for fmt in exact_formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    full_korean_match = re.search(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월\s*(\d{1,2})\s*일", normalized)
    if full_korean_match:
        raw_year, raw_month, raw_day = full_korean_match.groups()
        parsed = _build_date_with_time(
            int(raw_year) if raw_year else today.year,
            int(raw_month),
            int(raw_day),
            normalized,
        )
        if parsed is not None:
            return parsed

    full_numeric_match = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", normalized)
    if full_numeric_match:
        year, month, day = map(int, full_numeric_match.groups())
        parsed = _build_date_with_time(year, month, day, normalized)
        if parsed is not None:
            return parsed

    month_day_numeric_match = re.search(r"(?<!\d)(\d{1,2})[./](\d{1,2})(?!\d)", normalized)
    if month_day_numeric_match:
        month, day = map(int, month_day_numeric_match.groups())
        parsed = _build_date_with_time(today.year, month, day, normalized)
        if parsed is not None:
            return parsed

    weekday_date = _parse_standalone_weekday_date(normalized)
    if weekday_date is not None:
        return weekday_date

    # 연도 없는 형식 (M/D, M월 D일) → 현재 연도 보정
    year_less_formats = [
        ("%m/%d", None),
        ("%m월 %d일", None),
    ]
    # 공백 없는 한국어 날짜 ("5월15일" 같은 형태)
    korean_no_space = re.match(r"(\d{1,2})월(\d{1,2})일", date_str.replace(" ", ""))
    if korean_no_space:
        month, day = int(korean_no_space.group(1)), int(korean_no_space.group(2))
        return _build_date_with_time(today.year, month, day, normalized)

    for fmt, _ in year_less_formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            hour, minute = _parse_time_from_text(normalized)
            return parsed.replace(year=today.year, hour=hour, minute=minute, second=0, microsecond=0)
        except ValueError:
            continue

    logger.debug(f"날짜 파싱 실패, None 반환: {date_str}")
    return None


def find_source_in_transcripts(source_text: str, transcripts: list[dict]) -> dict | None:
    """
    추출된 일정의 source_text가 어느 전사문 청크에서 나왔는지 찾는다.
    확정된 일정을 클릭하면 원본 전사문 위치로 이동할 수 있도록 한다.
    부분 문자열 포함 검사를 사용한다.
    """
    if not source_text:
        return None

    # 공백 제거 후 비교 (STT 전사문 특성상 공백이 달라질 수 있음)
    clean_source = source_text.replace(" ", "")

    for t in transcripts:
        t_text = (t.get("text") or "").replace(" ", "")
        if clean_source in t_text or t_text in clean_source:
            return {
                "transcript_id": t["transcript_id"],
                "chunk_index": t["chunk_index"],
                "start_time": t["start_time"],
                "end_time": t["end_time"],
            }

    # 여러 짧은 청크에 걸친 문장도 찾는다.
    for window_size in range(2, 7):
        for index in range(0, max(0, len(transcripts) - window_size + 1)):
            window = transcripts[index:index + window_size]
            window_text = "".join((t.get("text") or "") for t in window).replace(" ", "")
            if clean_source in window_text or window_text in clean_source:
                return {
                    "transcript_id": window[0]["transcript_id"],
                    "chunk_index": window[0]["chunk_index"],
                    "start_time": window[0]["start_time"],
                    "end_time": window[-1]["end_time"],
                }

    # 완전 매칭이 안 되면 핵심 키워드 3개 이상 겹치는 청크 탐색
    source_words = set(source_text.split())
    best_match = None
    best_overlap = 0

    for t in transcripts:
        t_words = set((t.get("text") or "").split())
        overlap = len(source_words & t_words)
        if overlap > best_overlap and overlap >= 2:
            best_overlap = overlap
            best_match = {
                "transcript_id": t["transcript_id"],
                "chunk_index": t["chunk_index"],
                "start_time": t["start_time"],
                "end_time": t["end_time"],
            }

    return best_match


def filter_already_ignored(
    extracted: list[dict],
    ignored_titles: set[str],
) -> tuple[list[dict], list[dict]]:
    """
    추출된 일정 중 이전에 무시(ignored)된 동일 제목의 일정을 분리한다.
    반환: (알림 대상 일정 리스트, 자동 무시된 일정 리스트)
    """
    to_notify = []
    auto_ignored = []

    for schedule in extracted:
        title_clean = schedule["title"].strip()
        if title_clean in ignored_titles:
            auto_ignored.append(schedule)
            logger.info(f"[SCHEDULE] 이전 무시 일정 자동 필터: '{title_clean}'")
        else:
            to_notify.append(schedule)

    return to_notify, auto_ignored


#  일정 추출
async def extract_schedules(transcript_text: str) -> list[dict]:
    """
    전사문 텍스트에서 일정 관련 정보를 추출한다.
    MOCK_MODE=true이면 목업 데이터를 반환하고,
    false이면 LLM을 호출하여 실제 추출한다.

    긴 전사문은 청크로 분할하여 개별 LLM 호출 후 결과를 병합한다.
    병합 후 유사도 기반 중복 제거를 수행한다.
    """
    if MOCK_MODE:
        logger.info("[SCHEDULE] MOCK_MODE: 목업 일정 데이터 반환")
        return _generate_mock_schedules()

    sentences = _split_schedule_sentences(transcript_text)
    _demo_log("SCHEDULE", f"1) 전사문 수신: chars={len(transcript_text)}, sentences={len(sentences)}")
    for index, sentence in enumerate(sentences[:6], start=1):
        _demo_log("SCHEDULE", f"   문장#{index}: {_preview(sentence, 130)}")

    rule_schedules = _extract_rule_based_schedules(transcript_text)
    _demo_log("SCHEDULE", f"2) 규칙 기반 날짜/학사키워드 추출: candidates={len(rule_schedules)}")
    for index, schedule in enumerate(rule_schedules[:5], start=1):
        _demo_log(
            "SCHEDULE",
            f"   규칙후보#{index}: title='{schedule.get('title')}', date={schedule.get('due_date')}, type={schedule.get('event_type')}",
        )
    if rule_schedules:
        logger.info(f"[SCHEDULE] 규칙 기반 일정 {len(rule_schedules)}개 감지")
        if SCHEDULE_RULE_FIRST:
            _demo_log("SCHEDULE", "3) RULE_FIRST 활성화 -> LLM 호출 없이 규칙 기반 일정 반환")
            return rule_schedules

    if _is_non_academic(transcript_text, "", ""):
        logger.info("[SCHEDULE] 비학사 일정 문맥으로 판단되어 추출 생략")
        _demo_log("SCHEDULE", "3) 비학사 문맥 감지 -> 일정 추출 생략")
        return []

    # 긴 텍스트를 청크로 분할
    chunks = _split_transcript_chunks(transcript_text)
    logger.info(f"[SCHEDULE] 전사문 {len(transcript_text)}자 → {len(chunks)}개 청크로 분할")
    _demo_log("SCHEDULE", f"3) LLM 추출용 청크 분할: chunks={len(chunks)}")

    all_schedules = list(rule_schedules)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0)) as client:
            for i, chunk in enumerate(chunks):
                messages = _build_schedule_prompt(chunk)
                logger.info(f"[SCHEDULE] 청크 {i+1}/{len(chunks)} LLM 호출 ({len(chunk)}자)")
                _demo_log("SCHEDULE", f"4) 청크 {i+1}/{len(chunks)} LLM 일정 추출 요청: chars={len(chunk)}")

                res = await client.post(
                    f"{LLM_URL}/v1/chat/completions",
                    json={
                        "model": LLM_MODEL,
                        "messages": messages,
                        "max_tokens": 2048,
                        "temperature": 0.1,  # 정확한 추출을 위해 낮은 temperature
                        "bridgeprag_alpha": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                )
                res.raise_for_status()

                data = res.json()
                raw_answer = data["choices"][0]["message"]["content"]
                logger.info(f"[SCHEDULE] 청크 {i+1} LLM 응답: {len(raw_answer)} chars")
                _demo_log("SCHEDULE", f"5) 청크 {i+1} LLM 응답 수신: chars={len(raw_answer)}")

                try:
                    chunk_schedules = _parse_schedule_json(raw_answer)
                except ValueError as e:
                    logger.warning(f"[SCHEDULE] 청크 {i+1} LLM JSON 파싱 실패, 해당 청크 건너뜀: {e}")
                    _demo_log("SCHEDULE", f"6) 청크 {i+1} JSON 파싱 실패 -> skip: {e}")
                    continue
                _demo_log("SCHEDULE", f"6) 청크 {i+1} 일정 후보 파싱 완료: candidates={len(chunk_schedules)}")
                all_schedules.extend(chunk_schedules)

        # 모든 청크 결과를 합친 후 전체 중복 제거 (_filter_valid_schedules에서 처리됨)
        logger.info(f"[SCHEDULE] 전체 {len(all_schedules)}개 일정 추출 완료 ({len(chunks)}개 청크)")
        _demo_log("SCHEDULE", f"7) 최종 일정 후보 반환: schedules={len(all_schedules)}")
        for index, schedule in enumerate(all_schedules[:5], start=1):
            _demo_log(
                "SCHEDULE",
                f"   최종후보#{index}: title='{schedule.get('title')}', date={schedule.get('due_date')}, type={schedule.get('event_type')}",
            )
        return all_schedules

    except httpx.HTTPError as e:
        logger.error(f"[SCHEDULE] LLM 호출 실패: {e}")
        return []
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"[SCHEDULE] 일정 추출 중 예상치 못한 오류: {e}")
        raise RuntimeError(f"일정 추출 실패: {e}")


# 시멘틱 유사도 필터링 기능
async def get_embedding(text: str) -> list[float]:
    """LLM API를 호출하여 텍스트의 임베딩 벡터를 가져옵니다."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        res = await client.post(
            f"{LLM_URL}/v1/embeddings", 
            json={"input": text, "model": LLM_MODEL},
            headers={"Authorization": f"Bearer {LLM_API_KEY}"}
        )
        res.raise_for_status()
        data = res.json()
        return data["data"][0]["embedding"]

def calculate_cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    # 수학적 공식: similarity = cos(θ) = (A · B) / (||A|| ||B||)
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

async def filter_already_ignored_semantic(
    extracted: list[dict],
    ignored_metadata: list[dict],
    threshold: float = 0.85
) -> tuple[list[dict], list[dict]]:
    """
    날짜 대조 및 시멘틱 유사도를 이용해 중복 일정을 필터링한다.
    
    Args:
        extracted: LLM이 방금 추출한 일정 리스트
        ignored_metadata: DB에서 가져온 무시된 일정 리스트 (title, due_date 포함)
        threshold: 중복으로 판단할 유사도 기준 (0.85 권장)
    """
    to_notify = []
    auto_ignored = []

    if not ignored_metadata:
        return extracted, []

    if not SCHEDULE_SEMANTIC_DUP_FILTER:
        logger.info("[SCHEDULE] semantic duplicate filter disabled; skipping embeddings")
        return extracted, []

    # 1. 무시된 일정들의 임베딩을 미리 생성 (비교 최적화)
    ignored_embeddings = []
    try:
        for item in ignored_metadata:
            emb = await get_embedding(item["title"])
            ignored_embeddings.append({
                "emb": emb,
                "due_date": item["due_date"]  # DB에서 가져온 값 (이미 datetime이거나 변환된 상태)
            })
    except Exception as e:
        logger.warning(f"[SCHEDULE] ignored 일정 임베딩 생성 실패, 중복 필터 생략: {e}")
        return extracted, []

    for new_s in extracted:
        is_duplicate = False
        new_date = parse_due_date(new_s.get("due_date"))
        try:
            new_title_emb = await get_embedding(new_s["title"])
        except Exception as e:
            logger.warning(f"[SCHEDULE] 새 일정 임베딩 생성 실패, 알림 대상으로 유지: {e}")
            to_notify.append(new_s)
            continue

        for ign in ignored_embeddings:
            # 날짜 비교: 두 날짜가 모두 존재하는데 다르면 무조건 새 일정
            if new_date and ign["due_date"]:
                # DB 날짜 포맷이 문자열일 경우를 대비해 변환 확인 필요
                ign_date = ign["due_date"]
                if isinstance(ign_date, str):
                    ign_date = parse_due_date(ign_date)
                
                if new_date != ign_date:
                    continue

            # 벡터 유사도 비교
            similarity = calculate_cosine_similarity(new_title_emb, ign["emb"])
            if similarity >= threshold:
                logger.info(f"[SCHEDULE] 중복 감지: '{new_s['title']}' (유사도: {similarity:.2f})")
                is_duplicate = True
                break

        if is_duplicate:
            auto_ignored.append(new_s)
        else:
            to_notify.append(new_s)

    return to_notify, auto_ignored


#  목업 데이터 (LLM 미연결 시)
def _generate_mock_schedules() -> list[dict]:
    """프론트엔드 개발/테스트용 목업 일정 데이터를 반환한다."""
    exam_date = (datetime.now() + timedelta(days=7)).replace(hour=9, minute=0, second=0, microsecond=0)
    assignment_date = (datetime.now() + timedelta(days=3)).replace(hour=23, minute=59, second=0, microsecond=0)
    presentation_date = (datetime.now() + timedelta(days=14)).replace(hour=14, minute=0, second=0, microsecond=0)
    return [
        {
            "title": "목업 시험 일정",
            "description": "SCHEDULE_MOCK_MODE=true일 때만 표시되는 개발용 시험 일정",
            "event_type": "시험",
            "due_date": exam_date.isoformat(),
            "source_text": "개발용 목업 시험 일정입니다.",
        },
        {
            "title": "목업 과제 제출",
            "description": "SCHEDULE_MOCK_MODE=true일 때만 표시되는 개발용 과제 일정",
            "event_type": "과제",
            "due_date": assignment_date.isoformat(),
            "source_text": "개발용 목업 과제 일정입니다.",
        },
        {
            "title": "목업 발표 일정",
            "description": "SCHEDULE_MOCK_MODE=true일 때만 표시되는 개발용 발표 일정",
            "event_type": "발표",
            "due_date": presentation_date.isoformat(),
            "source_text": "개발용 목업 발표 일정입니다.",
        },
    ]
