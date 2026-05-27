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
    "due_date": "2026-05-15",
    "source_text": "중간고사는 5월 15일에 치릅니다"
  }
]
"""
import json
import re
import logging
import httpx
import os
import asyncio
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

# ── 설정 ──
MOCK_MODE = os.getenv("SCHEDULE_MOCK_MODE", "false").lower() == "true"

DEFAULT_LLM_URL = "http://localhost:8001"
DEFAULT_LLM_MODEL = "bridgeprag-qwen25-3b-kv64"

LLM_URL = os.getenv("LLM_URL", DEFAULT_LLM_URL)
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
LLM_API_KEY = os.getenv("LLM_API_KEY", "test-key")


#  LLM 프롬프트 템플릿
SCHEDULE_SYSTEM_PROMPT = """당신은 전사문에서 **학사 일정**만 추출하는 AI입니다.
어떠한 설명이나 <think> 태그도 쓰지 말고, 오직 JSON 배열만 출력하세요.

추출 대상: 시험, 과제, 프로젝트, 발표, 보강 등 학사 일정 (날짜가 명시된 것만)
제외 대상: 개인 약속, 식사, 날짜 없는 할 일 등 비학사 일정"""

SCHEDULE_USER_PROMPT_TEMPLATE = """오늘 날짜: {today}

전사문:
{transcript_text}

위 전사문에서 학사 일정을 찾아 아래 JSON 배열로만 출력하세요. 일정이 없으면 []를 반환하세요.
상대 날짜(내일 등)는 오늘을 기준으로 YYYY-MM-DD 형식으로 변환하세요.

[
  {{
    "title": "일정 제목",
    "description": "상세 설명",
    "event_type": "시험|과제|프로젝트|발표|기타",
    "due_date": "YYYY-MM-DD",
    "source_text": "전사문에 언급된 원문 문장"
  }}
]"""


# 청크 분할 설정
CHUNK_SIZE = 2000       # 각 청크의 최대 글자 수
CHUNK_OVERLAP = 300     # 청크 간 오버랩 글자 수 (경계 일정 누락 방지)


def filter_schedule_relevant_text(text: str) -> str:
    """
    전사문 텍스트에서 학사 일정(시험, 과제, 등)과 관련된 
    핵심 문장만 필터링하여 반환합니다. 키워드가 전혀 없으면 빈 문자열을 반환합니다.
    """
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    relevant = []
    
    core_keywords = ["시험", "고사", "퀴즈", "과제", "제출", "마감", "보고서", "레포트", "리포트", "프로젝트", "팀플", "설계", "발표", "보강", "휴강"]
    date_keywords = ["일", "월", "주", "내일", "모레", "오늘", "오전", "오후", "시", "까지"]
    
    for sent in sentences:
        if not sent.strip():
            continue
            
        has_core = any(kw in sent for kw in core_keywords)
        has_date = any(kw in sent for kw in date_keywords)
        
        if has_core and has_date:
            relevant.append(sent.strip())
            
    return " ".join(relevant)

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


SCHEDULE_EVENT_KEYWORDS = (
    "과제", "제출", "마감", "보고서", "레포트", "리포트",
    "시험", "고사", "퀴즈", "발표", "프로젝트", "회의",
    "수업", "보강", "실습"
)

TITLE_GENERIC_WORDS = (
    "일정", "과제", "제출", "마감", "보고서", "레포트", "리포트",
    "시험", "고사", "퀴즈", "발표", "프로젝트", "회의",
    "수업", "보강", "실습", "중간", "기말", "중간고사", "기말고사"
)

SCHEDULE_DATE_HINT_PATTERN = re.compile(
    r"(\d{4}[./-]\d{1,2}[./-]\d{1,2}|"
    r"\d{1,2}\s*월\s*\d{1,2}\s*일|"
    r"오늘|내일|모레|다음\s*주|이번\s*주|"
    r"오전\s*\d{1,2}\s*시|오후\s*\d{1,2}\s*시|"
    r"\d{1,2}\s*시|\d{1,2}\s*분|까지|마감)"
)

TIME_HINT_PATTERN = re.compile(r"(오전|오후)?\s*\d{1,2}\s*시(?:\s*\d{1,2}\s*분)?")


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
    has_academic = any(kw in source_lower for kw in SCHEDULE_EVENT_KEYWORDS)
    # 비학사 키워드가 있고 학사 키워드가 없으면 비학사 일정
    return has_non_academic and not has_academic



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


def _parse_time_from_text(text: str) -> tuple[int, int]:
    """문장 안의 한국어 시간 표현을 찾고, 없으면 오전 9시로 둔다."""
    hour = 9
    minute = 0
    time_match = re.search(r"(오전|오후)?\s*(\d{1,2})\s*시(?:\s*(\d{1,2})\s*분)?", text)
    if not time_match:
        return hour, minute

    meridiem, raw_hour, raw_minute = time_match.groups()
    hour = int(raw_hour)
    minute = int(raw_minute or 0)
    if meridiem == "오후" and hour != 12:
        hour += 12
    if meridiem == "오전" and hour == 12:
        hour = 0
    return hour, minute


def _parse_weekday_relative_date(text: str) -> datetime | None:
    """'이번주 일요일', '다음 주 목요일', '다다음주 월요일' 같은 표현을 날짜로 변환한다."""
    match = re.search(r"(이번|다다음|다음)\s*주\s*(월요일|화요일|수요일|목요일|금요일|토요일|일요일|월|화|수|목|금|토|일)", text)
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

    relative_base = None
    if "오늘" in normalized:
        relative_base = datetime.now()
    elif "내일" in normalized:
        relative_base = datetime.now() + timedelta(days=1)
    elif "모레" in normalized:
        relative_base = datetime.now() + timedelta(days=2)

    if relative_base is not None:
        hour, minute = _parse_time_from_text(normalized)
        return relative_base.replace(hour=hour, minute=minute, second=0, microsecond=0)

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # 연도 없는 형식 (M/D, M월 D일) → 현재 연도 보정
    year_less_formats = [
        ("%m/%d", None),
        ("%m월 %d일", None),
    ]
    # 공백 없는 한국어 날짜 ("5월15일" 같은 형태)
    korean_no_space = re.match(r"(\d{1,2})월(\d{1,2})일", date_str.replace(" ", ""))
    if korean_no_space:
        month, day = int(korean_no_space.group(1)), int(korean_no_space.group(2))
        return datetime.now().replace(month=month, day=day, hour=9, minute=0, second=0, microsecond=0)

    for fmt, _ in year_less_formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.replace(year=datetime.now().year)
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

    # 긴 텍스트를 청크로 분할
    chunks = _split_transcript_chunks(transcript_text)
    logger.info(f"[SCHEDULE] 전사문 {len(transcript_text)}자 → {len(chunks)}개 청크로 분할")

    all_schedules = []

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0)) as client:
            for i, chunk in enumerate(chunks):
                messages = _build_schedule_prompt(chunk)
                import time
                chunk_start_time = time.time()
                logger.info(f"[SCHEDULE] 청크 {i+1}/{len(chunks)} LLM 호출 시작 ({len(chunk)}자)")

                res = await client.post(
                    f"{LLM_URL}/v1/chat/completions",
                    json={
                        "model": LLM_MODEL,
                        "messages": messages,
                        "max_tokens": 1024,
                        "temperature": 0.1,  # 정확한 추출을 위해 낮은 temperature
                        "stream": False,
                        "bridgeprag_alpha": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                )
                res.raise_for_status()

                chunk_duration = time.time() - chunk_start_time
                data = res.json()
                raw_answer = data["choices"][0]["message"]["content"]
                logger.info(f"[SCHEDULE] 청크 {i+1} LLM 응답: {len(raw_answer)} chars (소요시간: {chunk_duration:.2f}초)")

                chunk_schedules = _parse_schedule_json(raw_answer)
                all_schedules.extend(chunk_schedules)

        # 모든 청크 결과를 합친 후 전체 중복 제거 (_filter_valid_schedules에서 처리됨)
        logger.info(f"[SCHEDULE] 전체 {len(all_schedules)}개 일정 추출 완료 ({len(chunks)}개 청크)")
        return all_schedules

    except httpx.HTTPError as e:
        logger.error(f"[SCHEDULE] LLM 호출 실패: {e}")
        raise RuntimeError(f"LLM 서버 연결 실패: {e}")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"[SCHEDULE] 일정 추출 중 예상치 못한 오류: {e}")
        raise RuntimeError(f"일정 추출 실패: {e}")


# 시멘틱 유사도 필터링 기능
async def get_embedding(text: str) -> list[float]:
    """텍스트의 임베딩 벡터를 반환한다."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{LLM_URL}/v1/embeddings",
                json={"input": text, "model": "text-embedding-3-small"},
                headers={"Authorization": f"Bearer {LLM_API_KEY}"}
            )
            res.raise_for_status()
            return res.json()["data"][0]["embedding"]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # 로컬 LLM 서버가 임베딩을 지원하지 않는 경우 조용히 넘어감
            pass
        else:
            logger.warning(f"[SCHEDULE] 임베딩 실패 (상태 코드 {e.response.status_code})")
        return []
    except Exception as e:
        logger.warning(f"[SCHEDULE] 임베딩 실패 (유사도 필터링 스킵): {e}")
        return []

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
    날짜 대조 및 시멘틱 유사도(또는 자카드 유사도 폴백)를 이용해 중복 일정을 필터링한다.
    
    Args:
        extracted: LLM이 방금 추출한 일정 리스트
        ignored_metadata: DB에서 가져온 무시된 일정 리스트 (title, due_date 포함)
        threshold: 중복으로 판단할 유사도 기준 (0.85 권장)
    """
    to_notify = []
    auto_ignored = []

    if not ignored_metadata:
        return extracted, []

    # 1. 무시된 일정들의 임베딩을 병렬 생성 (asyncio.gather 사용)
    ignored_embeddings = []
    embedding_failed = False
    
    async def _safe_get_embedding(item):
        try:
            emb = await get_embedding(item["title"])
            return {"emb": emb, "title": item["title"], "due_date": item["due_date"]}
        except Exception as e:
            logger.debug(f"[SCHEDULE] '{item['title']}' 임베딩 생성 실패 (폴백 적용 예정): {e}")
            return {"emb": None, "title": item["title"], "due_date": item["due_date"]}

    try:
        tasks = [_safe_get_embedding(item) for item in ignored_metadata]
        results = await asyncio.gather(*tasks)
        ignored_embeddings = [r for r in results if r["emb"] is not None]
        # 모든 임베딩 호출이 실패했다면 임베딩 서버가 꺼진 것으로 판단
        if not ignored_embeddings and len(ignored_metadata) > 0:
            embedding_failed = True
    except Exception as e:
        logger.warning(f"[SCHEDULE] ignored 일정 임베딩 병렬 생성 중 오류, 텍스트 폴백 모드로 전환: {e}")
        embedding_failed = True

    # 2. 각 신규 일정 비교
    for new_s in extracted:
        is_duplicate = False
        new_date = parse_due_date(new_s.get("due_date"))
        
        # 임베딩 서버가 정상 작동하는 경우 시멘틱 비교 수행
        if not embedding_failed:
            try:
                new_title_emb = await get_embedding(new_s["title"])
                for ign in ignored_embeddings:
                    # 날짜 비교: 두 날짜가 모두 존재하는데 다르면 무조건 새 일정
                    if new_date and ign["due_date"]:
                        ign_date = ign["due_date"]
                        if isinstance(ign_date, str):
                            ign_date = parse_due_date(ign_date)
                        
                        if new_date != ign_date:
                            continue

                    # 벡터 유사도 비교
                    similarity = calculate_cosine_similarity(new_title_emb, ign["emb"])
                    if similarity >= threshold:
                        logger.info(f"[SCHEDULE] 중복 감지(시멘틱 유사도 {similarity:.2f}): '{new_s['title']}' == 무시된 일정 '{ign['title']}'")
                        is_duplicate = True
                        break
            except Exception as e:
                logger.warning(f"[SCHEDULE] 새 일정 임베딩 생성 실패, 텍스트 폴백 모드 전환: {e}")
                embedding_failed = True

        # 임베딩 비교 결과 중복이 아니라면 텍스트 폴백 필터링 수행
        if is_duplicate:
            auto_ignored.append(new_s)
            continue
            
        # 텍스트 기반 중복 제거 (Fallback)
        for ign in ignored_metadata:
            if new_date and ign["due_date"]:
                ign_date = ign["due_date"]
                if isinstance(ign_date, str):
                    ign_date = parse_due_date(ign_date)
                
                if new_date != ign_date:
                    continue

            # 제목 자카드 유사도 비교 (글자 수준 공통 비율 0.70 이상이면 중복 판단)
            similarity = jaccard_similarity(new_s["title"], ign["title"])
            if similarity >= 0.70 or new_s["title"].strip() == ign["title"].strip():
                logger.info(f"[SCHEDULE] 중복 감지(텍스트 폴백 유사도 {similarity:.2f}): '{new_s['title']}' == 무시된 일정 '{ign['title']}'")
                is_duplicate = True
                break

        if is_duplicate:
            auto_ignored.append(new_s)
        else:
            to_notify.append(new_s)

    return to_notify, auto_ignored


def jaccard_similarity(str1: str, str2: str) -> float:
    """두 문자열 간의 글자 기반 자카드 유사도를 계산합니다."""
    s1 = set(str1.strip().lower())
    s2 = set(str2.strip().lower())
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


# 실시간 일정 추출을 위한 키워드 사전 정의
SCHEDULE_CORE_KEYWORDS = ["시험", "고사", "퀴즈", "쪽지", "과제", "제출", "마감", "레포트", "리포트", "프로젝트", "팀플", "설계", "발표", "세미나", "보강", "휴강", "실습", "특강"]
SCHEDULE_DATE_KEYWORDS = ["월", "일", "내일", "오늘", "다음주", "다다음주", "요일", "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일", "까지"]



#  목업 데이터 (LLM 미연결 시)
def _generate_mock_schedules() -> list[dict]:
    """프론트엔드 개발/테스트용 목업 일정 데이터를 반환한다."""
    return [
        {
            "title": "데이터베이스 중간고사",
            "description": "Chapter 1~5 범위, 정규화와 SQL 중심",
            "event_type": "시험",
            "due_date": "2026-05-15",
            "source_text": "중간고사는 5월 15일에 치르겠습니다. 범위는 1장부터 5장까지입니다.",
        },
        {
            "title": "ERD 설계 과제 제출",
            "description": "팀별 ERD 설계 결과물 제출",
            "event_type": "과제",
            "due_date": "2026-05-08",
            "source_text": "ERD 설계 과제는 다음주 목요일까지 제출해주세요.",
        },
        {
            "title": "프로젝트 중간 발표",
            "description": "팀 프로젝트 진행 상황 발표",
            "event_type": "발표",
            "due_date": "2026-05-20",
            "source_text": "프로젝트 중간 발표는 5월 20일로 예정되어 있습니다.",
        },
    ]
