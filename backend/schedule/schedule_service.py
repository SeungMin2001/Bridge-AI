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
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

# ── 설정 ──
MOCK_MODE = os.getenv("SCHEDULE_MOCK_MODE", "true").lower() == "true"

LLM_URL = os.getenv("LLM_URL", "http://localhost:8001")
LLM_MODEL = os.getenv("LLM_MODEL", "QuantTrio/Qwen3.5-4B-AWQ")
LLM_API_KEY = os.getenv("LLM_API_KEY", "test-key")


#  LLM 프롬프트 템플릿
SCHEDULE_SYSTEM_PROMPT = """당신은 대학 강의 전사문에서 일정 관련 정보를 추출하는 AI 비서입니다.
시험, 과제, 프로젝트, 발표, 제출 마감일 등 학사 일정을 정확하게 찾아내세요.
반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 텍스트는 절대 포함하지 마세요.
일정이 없으면 빈 배열 []을 반환하세요."""

SCHEDULE_USER_PROMPT_TEMPLATE = """아래는 강의 전사문입니다:

{transcript_text}

위 전사문에서 일정 관련 내용을 모두 찾아 아래 JSON 배열 형식으로 추출하세요.
날짜가 명시되지 않은 경우 due_date는 null로 설정하세요.
날짜 형식은 "YYYY-MM-DD" 또는 "YYYY-MM-DDTHH:MM:SS"로 작성하세요.

[
  {{
    "title": "일정 제목 (간결하게)",
    "description": "일정에 대한 상세 설명",
    "event_type": "시험|과제|프로젝트|발표|기타",
    "due_date": "2026-05-15" 또는 null,
    "source_text": "전사문에서 해당 일정이 언급된 원문 문장"
  }}
]"""


def _build_schedule_prompt(transcript_text: str) -> list[dict]:
    """LLM에 보낼 일정 추출 프롬프트 messages 배열을 구성한다."""
    user_prompt = SCHEDULE_USER_PROMPT_TEMPLATE.format(
        transcript_text=transcript_text[:6000]  # 토큰 제한 고려
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
    except json.JSONDecodeError as e:
        logger.error(f"일정 JSON 파싱 실패: {e}\n원본: {raw_text[:500]}")
        raise ValueError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {e}")

    if not isinstance(schedules, list):
        raise ValueError("LLM 응답이 JSON 배열이 아닙니다")

    # 필수 필드 기본값 보장
    for s in schedules:
        s.setdefault("title", "제목 없음")
        s.setdefault("description", "")
        s.setdefault("event_type", "기타")
        s.setdefault("due_date", None)
        s.setdefault("source_text", "")

    return schedules


def parse_due_date(date_str: str | None) -> datetime | None:
    """
    다양한 날짜 포맷을 datetime으로 변환한다.
    LLM이 반환하는 날짜 형식이 일정하지 않을 수 있으므로 여러 포맷을 시도한다.
    """
    if not date_str:
        return None

    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%m/%d",
        "%m월 %d일",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    logger.warning(f"날짜 파싱 실패, None 반환: {date_str}")
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

    # 완전 매칭이 안 되면 핵심 키워드 3개 이상 겹치는 청크 탐색
    source_words = set(source_text.split())
    best_match = None
    best_overlap = 0

    for t in transcripts:
        t_words = set((t.get("text") or "").split())
        overlap = len(source_words & t_words)
        if overlap > best_overlap and overlap >= 3:
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
    """
    if MOCK_MODE:
        logger.info("[SCHEDULE] MOCK_MODE: 목업 일정 데이터 반환")
        return _generate_mock_schedules()

    messages = _build_schedule_prompt(transcript_text)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0)) as client:
            res = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.2,  # 정확한 추출을 위해 낮은 temperature
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            )
            res.raise_for_status()

        data = res.json()
        raw_answer = data["choices"][0]["message"]["content"]
        logger.info(f"[SCHEDULE] LLM 응답 수신: {len(raw_answer)} chars")

        schedules = _parse_schedule_json(raw_answer)
        logger.info(f"[SCHEDULE] {len(schedules)}개 일정 추출 완료")
        return schedules

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

    # 1. 무시된 일정들의 임베딩을 미리 생성 (비교 최적화)
    ignored_embeddings = []
    for item in ignored_metadata:
        emb = await get_embedding(item["title"])
        ignored_embeddings.append({
            "emb": emb, 
            "due_date": item["due_date"]  # DB에서 가져온 값 (이미 datetime이거나 변환된 상태)
        })

    for new_s in extracted:
        is_duplicate = False
        new_date = parse_due_date(new_s.get("due_date"))
        new_title_emb = await get_embedding(new_s["title"])

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