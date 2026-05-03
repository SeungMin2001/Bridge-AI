"""
일정 API 라우터

엔드포인트:
  POST /schedule/extract                - 세션 전사문에서 일정 추출 (녹음 종료 시 호출)
  GET  /schedule/                       - 전체 일정 조회 (일정 관리 화면)
  POST /schedule/manual                 - 사용자가 직접 추가한 일정 저장
  GET  /schedule/calendar               - 확정된 일정만 반환 (달력 표시용)
  GET  /schedule/session/{session_id}   - 세션별 전체 일정 조회
  PUT  /schedule/{schedule_id}/confirm  - 일정 확정 (달력에 표시)
  PUT  /schedule/{schedule_id}/ignore   - 일정 무시 (달력에 미표시, DB 보존)
  GET  /schedule/{schedule_id}          - 일정 단건 조회 (전사문 출처 포함)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import uuid
import logging

from db import get_transcripts_by_session
from schedule.schedule_db import (
    save_schedule,
    get_all_schedules,
    get_schedule,
    get_schedules_by_session,
    get_confirmed_schedules,
    update_schedule_status, # 일정 상태 업데이트
    get_ignored_schedules_metadata, # 무시된 일정 메타데이터 조회
    get_schedule_with_transcript, # 전사문과 함께 일정 조회
)
from schedule.schedule_service import (
    extract_schedules, # 일정 추출
    parse_due_date, # due_date 파싱
    find_source_in_transcripts, # 전사문에서 일정 출처 찾기
    filter_already_ignored_semantic, # 시멘틱 필터링 함수
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schedule", tags=["schedule"])


# ── 요청 스키마 ──
class ScheduleExtractRequest(BaseModel):
    """세션 기반 일정 추출 요청 (녹음 종료 시 프론트에서 호출)"""
    session_id: str


class ScheduleExtractTextRequest(BaseModel):
    """직접 텍스트로 일정 추출 (테스트용)"""
    text: str = Field(..., min_length=10, description="일정 추출 대상 텍스트")
    session_id: str | None = None


class ScheduleManualRequest(BaseModel):
    """사용자가 직접 입력한 일정 저장 요청"""
    title: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    event_type: str | None = None
    due_date: str | None = None
    status: str = "confirmed"
    session_id: str | None = None
    transcript_id: str | None = None
    source_start_time: float | None = None
    source_end_time: float | None = None
    source_text: str | None = None


VALID_SCHEDULE_STATUSES = {"pending", "confirmed", "ignored"}



#  녹음 종료 → 전사문에서 일정 자동 추출
@router.post("/extract")
async def schedule_extract(req: ScheduleExtractRequest):
    """
    녹음 세션 종료 시 호출한다.
    전사문을 LLM에 보내 일정을 추출하고, 이전에 무시한 동일 일정(의미적 유사성 포함)은
    자동 필터링하여 알림 대상에서 제외한다.
    새로운 일정만 pending 상태로 DB에 저장하고, 프론트에 알림 데이터를 반환한다.
    """
    logger.info(f"[SCHEDULE] 일정 추출 요청: session_id={req.session_id}")

    # 1. 세션 전사문 조회
    transcripts = await get_transcripts_by_session(req.session_id)
    if not transcripts:
        raise HTTPException(
            status_code=404,
            detail=f"세션 '{req.session_id}'에 해당하는 전사문이 없습니다."
        )

    # 2. 전사문 합치기
    transcript_text = "\n".join(t["text"] for t in transcripts if t["text"])
    if len(transcript_text.strip()) < 20:
        raise HTTPException(status_code=400, detail="전사문이 너무 짧아 일정을 추출할 수 없습니다.")

    # 3. LLM으로 일정 추출
    try:
        extracted = await extract_schedules(transcript_text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not extracted:
        return {
            "session_id": req.session_id,
            "notifications": [],
            "auto_ignored": [],
            "total_extracted": 0,
        }

    # 4. 이전 무시 일정 필터링 (시멘틱 유사도 기반)
    # DB에서 무시된 일정의 메타데이터(제목, 날짜)를 가져옴
    ignored_metadata = await get_ignored_schedules_metadata()
    
    # due_date 확인 + 벡터 유사도 계산을 통해 중복 분리
    to_notify, auto_ignored = await filter_already_ignored_semantic(
        extracted, 
        ignored_metadata, 
        threshold=0.85 # 일단 0.85로 설정, 더 높여도도미
    )

    # 5. 전사문 출처 매칭 + DB 저장 (알림 대상만)
    notifications = []
    for s in to_notify:
        schedule_id = str(uuid.uuid4())

        source_match = find_source_in_transcripts(s.get("source_text", ""), transcripts)
        if source_match is None:
            logger.info(f"[SCHEDULE] 출처 매칭 실패 후보 제외: {s.get('title')}")
            continue
        transcript_id = source_match["transcript_id"] if source_match else None

        await save_schedule(
            schedule_id=schedule_id,
            session_id=req.session_id,
            title=s["title"],
            description=s.get("description"),
            event_type=s.get("event_type"),
            due_date=parse_due_date(s.get("due_date")),
            source_start_time=source_match["start_time"] if source_match else None,
            source_end_time=source_match["end_time"] if source_match else None,
            source_text=s.get("source_text"),
            transcript_id=transcript_id,
        )

        notifications.append({
            "schedule_id": schedule_id,
            "title": s["title"],
            "description": s.get("description"),
            "event_type": s.get("event_type"),
            "due_date": s.get("due_date"),
            "source_text": s.get("source_text"),
            "status": "pending",
        })

    # 6. 자동 무시 일정도 DB에 ignored 상태로 저장 (이력 보존)
    for s in auto_ignored:
        schedule_id = str(uuid.uuid4())
        source_match = find_source_in_transcripts(s.get("source_text", ""), transcripts)

        await save_schedule(
            schedule_id=schedule_id,
            session_id=req.session_id,
            title=s["title"],
            description=s.get("description"),
            event_type=s.get("event_type"),
            due_date=parse_due_date(s.get("due_date")),
            source_text=s.get("source_text"),
            transcript_id=source_match["transcript_id"] if source_match else None,
        )
        # 즉시 ignored로 업데이트
        await update_schedule_status(schedule_id, "ignored")

    logger.info(
        f"[SCHEDULE] 추출 완료: {len(notifications)}개 알림, "
        f"{len(auto_ignored)}개 자동무시(시멘틱 필터링), 총 {len(extracted)}개"
    )

    return {
        "session_id": req.session_id,
        "notifications": notifications,
        "auto_ignored_count": len(auto_ignored),
        "total_extracted": len(extracted),
    }


#  직접 텍스트 → 일정 추출 (테스트용)
@router.post("/extract/text")
async def schedule_extract_from_text(req: ScheduleExtractTextRequest):
    """텍스트를 직접 입력하여 일정을 추출한다. LLM 연동 테스트용."""
    logger.info(f"[SCHEDULE] 텍스트 기반 일정 추출: {len(req.text)} chars")

    try:
        extracted = await extract_schedules(req.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    session_id = req.session_id or str(uuid.uuid4())

    notifications = []
    for s in extracted:
        schedule_id = str(uuid.uuid4())
        await save_schedule(
            schedule_id=schedule_id,
            session_id=session_id,
            title=s["title"],
            description=s.get("description"),
            event_type=s.get("event_type"),
            due_date=parse_due_date(s.get("due_date")),
            source_text=s.get("source_text"),
        )
        notifications.append({
            "schedule_id": schedule_id,
            "title": s["title"],
            "description": s.get("description"),
            "event_type": s.get("event_type"),
            "due_date": s.get("due_date"),
            "source_text": s.get("source_text"),
            "status": "pending",
        })

    return {
        "session_id": session_id,
        "notifications": notifications,
        "total_extracted": len(extracted),
    }


@router.get("/")
async def schedule_list():
    """전체 일정을 반환한다. 일정 관리 화면의 기본 데이터로 사용한다."""
    schedules = await get_all_schedules()
    return {
        "count": len(schedules),
        "schedules": schedules,
    }


@router.post("/manual")
async def schedule_create_manual(req: ScheduleManualRequest):
    """사용자가 직접 추가한 일정을 저장한다."""
    if req.status not in VALID_SCHEDULE_STATUSES:
        raise HTTPException(status_code=422, detail=f"지원하지 않는 일정 상태입니다: {req.status}")

    schedule_id = str(uuid.uuid4())

    await save_schedule(
        schedule_id=schedule_id,
        session_id=req.session_id,
        title=req.title,
        description=req.description,
        event_type=req.event_type,
        due_date=parse_due_date(req.due_date),
        source_start_time=req.source_start_time,
        source_end_time=req.source_end_time,
        source_text=req.source_text,
        transcript_id=req.transcript_id,
    )

    if req.status != "pending":
        await update_schedule_status(schedule_id, req.status)

    schedule = await get_schedule(schedule_id)
    return schedule or {"schedule_id": schedule_id, "status": req.status}


#  확정된 일정만 (달력 표시용)
@router.get("/calendar")
async def schedule_calendar():
    """확정(confirmed)된 일정만 due_date 오름차순으로 반환한다. 달력 UI에 표시할 데이터."""
    schedules = await get_confirmed_schedules()
    return {
        "count": len(schedules),
        "schedules": schedules,
    }


#  세션별 전체 일정
@router.get("/session/{session_id}")
async def schedule_list_by_session(session_id: str):
    """세션에서 추출된 모든 일정(pending, confirmed, ignored)을 반환한다."""
    schedules = await get_schedules_by_session(session_id)
    return {
        "session_id": session_id,
        "count": len(schedules),
        "schedules": schedules,
    }


#  일정 확정 (달력에 표시)
@router.put("/{schedule_id}/confirm")
async def schedule_confirm(schedule_id: str):
    """일정을 확정한다."""
    existing = await get_schedule(schedule_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"일정을 찾을 수 없습니다: {schedule_id}")

    result = await update_schedule_status(schedule_id, "confirmed")
    logger.info(f"[SCHEDULE] 일정 확정: {schedule_id} ('{existing['title']}')")
    return result


#  일정 무시 (달력 미표시, DB 보존)
@router.put("/{schedule_id}/ignore")
async def schedule_ignore(schedule_id: str):
    """일정을 무시한다. 향후 동일 일정 추출 시 시멘틱 필터링의 기준이 된다."""
    existing = await get_schedule(schedule_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"일정을 찾을 수 없습니다: {schedule_id}")

    result = await update_schedule_status(schedule_id, "ignored")
    logger.info(f"[SCHEDULE] 일정 무시: {schedule_id} ('{existing['title']}')")
    return result


#  일정 단건 조회 (전사문 출처 포함)
@router.get("/{schedule_id}")
async def schedule_get(schedule_id: str):
    """일정 상세를 조회한다."""
    schedule = await get_schedule_with_transcript(schedule_id)
    if schedule is None:
        raise HTTPException(status_code=404, detail=f"일정을 찾을 수 없습니다: {schedule_id}")
    return schedule
