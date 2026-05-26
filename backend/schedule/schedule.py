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
  PUT  /schedule/{schedule_id}/notion   - 노션 페이지 ID 연동
  POST /schedule/{schedule_id}/sync-notion  - 백엔드에서 노션 직접 등록
  POST /schedule/sync-notion-import     - 노션 캘린더 → 우리 DB 일정 가져오기
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
    update_schedule_notion_id, # 노션 연동 ID 업데이트
    get_all_notion_page_ids, # 노션 페이지 ID 중복 체크
)
from schedule.schedule_service import (
    extract_schedules, # 일정 추출
    parse_due_date, # due_date 파싱
    find_source_in_transcripts, # 전사문에서 일정 출처 찾기
    filter_already_ignored_semantic, # 시멘틱 필터링 함수
    SESSION_SCHEDULE_CACHE, # 실시간 추출 캐시
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schedule", tags=["schedule"])


# ── 요청 스키마 ──
class ScheduleExtractRequest(BaseModel):
    """세션 기반 일정 추출 요청 (녹음 종료 시 프론트에서 호출)"""
    session_id: str
    recording_id: str | None = None


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
    recording_id: str | None = None
    transcript_id: str | None = None
    source_start_time: float | None = None
    source_end_time: float | None = None
    source_text: str | None = None
    notion_page_id: str | None = None


VALID_SCHEDULE_STATUSES = {"pending", "confirmed", "ignored"}



#  녹음 종료 → 전사문에서 일정 자동 추출
@router.post("/extract")
async def schedule_extract(req: ScheduleExtractRequest):
    """
    녹음 세션 종료 시 호출한다.
    실시간 일정 감지 캐시가 있다면 즉각 로드하며, 없으면 전체 전사문에서 추출을 백업 시도한다.
    이전에 무시한 동일 일정(의미적 유사성/텍스트 폴백 포함)은 자동 필터링하여 알림 대상에서 제외한다.
    새로운 일정만 pending 상태로 DB에 저장하고, 프론트에 알림 데이터를 반환한다.
    """
    logger.info(f"[SCHEDULE] 일정 추출 요청: session_id={req.session_id}, recording_id={req.recording_id}")

    # session_id, recording_id 문자열 포맷 정규화 (대소문자, 하이픈 제거)
    norm_session_id = str(req.session_id).lower().replace("-", "")
    norm_recording_id = str(req.recording_id).lower().replace("-", "") if req.recording_id else ""
    cache_key = (norm_session_id, norm_recording_id)
    cached_extracted = SESSION_SCHEDULE_CACHE.pop(cache_key, None)

    # 1. 캐싱된 실시간 추출 일정이 있는지 확인 (비어있지 않은 실제 데이터가 있는 경우에만 활용)
    if cached_extracted:
        logger.info(f"[SCHEDULE] 실시간 캐시 로드 성공! 캐싱된 일정 수: {len(cached_extracted)}")
        extracted = cached_extracted
    else:
        # 캐싱이 없거나 누락된 경우 전체 전사문에서 백업 LLM 추출 수행
        logger.info("[SCHEDULE] 실시간 캐시 누락. 전체 전사문에서 일정 추출을 수행합니다.")
        
        # 1-1. 세션 전사문 조회
        transcripts = await get_transcripts_by_session(req.session_id, req.recording_id)
        if not transcripts:
            raise HTTPException(
                status_code=404,
                detail=f"세션 '{req.session_id}'에 해당하는 전사문이 없습니다."
            )

        # 1-2. 전사문 합치기
        transcript_text = "\n".join(t["text"] for t in transcripts if t["text"])
        if len(transcript_text.strip()) < 5:
            raise HTTPException(status_code=400, detail="전사문이 너무 짧아 일정을 추출할 수 없습니다.")

        # 1-3. LLM으로 일정 추출
        try:
            extracted = await extract_schedules(transcript_text)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    # 2. 모든 전사문(세션/녹음 범위) 조회 (출처 매칭을 위한 용도)
    transcripts = await get_transcripts_by_session(req.session_id, req.recording_id)

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

        source_match = find_source_in_transcripts(s.get("source_text", ""), transcripts) if transcripts else None
        if source_match is None:
            logger.info(f"[SCHEDULE] 출처 매칭 실패 (알림은 유지): {s.get('title')}")
        transcript_id = source_match["transcript_id"] if source_match else None

        await save_schedule(
            schedule_id=schedule_id,
            session_id=req.session_id,
            recording_id=req.recording_id,
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
            recording_id=req.recording_id,
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
        recording_id=req.recording_id,
        title=req.title,
        description=req.description,
        event_type=req.event_type,
        due_date=parse_due_date(req.due_date),
        source_start_time=req.source_start_time,
        source_end_time=req.source_end_time,
        source_text=req.source_text,
        transcript_id=req.transcript_id,
        notion_page_id=req.notion_page_id,
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

# 노션 페이지 ID로 일정 업데이트
class NotionSyncRequest(BaseModel):
    notion_page_id: str | None = Field(..., description="연동된 노션 페이지 ID")


@router.put("/{schedule_id}/notion")
async def schedule_update_notion(schedule_id: str, req: NotionSyncRequest):
    """일정의 노션 연동 ID를 업데이트한다. (클라이언트/MCP 단에서 등록한 경우 호출)"""
    existing = await get_schedule(schedule_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"일정을 찾을 수 없습니다: {schedule_id}")

    result = await update_schedule_notion_id(schedule_id, req.notion_page_id)
    logger.info(f"[SCHEDULE] 일정 노션 ID 업데이트: {schedule_id} -> {req.notion_page_id}")
    return result


@router.post("/{schedule_id}/sync-notion")
async def schedule_sync_notion(schedule_id: str):
    """일정을 백엔드에서 노션 API를 통해 직접 등록하고 결과를 저장한다."""
    existing = await get_schedule(schedule_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"일정을 찾을 수 없습니다: {schedule_id}")

    if existing.get("notion_page_id"):
        return {
            "result": "already_synced",
            "message": "이미 노션에 등록된 일정입니다.",
            "notion_page_id": existing["notion_page_id"],
            "schedule_status": existing.get("status")
        }

    try:
        from schedule.notion_service import sync_schedule_to_notion
        page_id = await sync_schedule_to_notion(existing)

        # DB 업데이트 (노션 ID 등록)
        await update_schedule_notion_id(schedule_id, page_id)

        # 노션에 등록된 일정은 즉시 'confirmed' 상태로 확실하게 통일시킵니다.
        if existing.get("status") != "confirmed":
            await update_schedule_status(schedule_id, "confirmed")

        return {
            "result": "success",
            "message": "노션 등록 성공 및 confirmed 상태 확정",
            "notion_page_id": page_id,
            "schedule_status": "confirmed"
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=502, detail=str(re))
    except Exception as e:
        logger.error(f"[SCHEDULE] 노션 동기화 실패: {e}")
        raise HTTPException(status_code=500, detail=f"예기치 못한 노션 동기화 실패: {e}")


@router.post("/sync-notion-confirmed")
async def schedule_sync_confirmed_to_notion():
    """노션에 아직 등록되지 않은 확정(confirmed) 일정을 모두 노션 데이터베이스에 저장한다."""
    schedules = await get_all_schedules()
    targets = [
        schedule for schedule in schedules
        if schedule.get("status") == "confirmed" and not schedule.get("notion_page_id")
    ]

    if not targets:
        return {
            "result": "already_synced",
            "message": "노션에 새로 저장할 확정 일정이 없습니다.",
            "synced_count": 0,
            "failed_count": 0,
            "synced": [],
            "failed": [],
        }

    try:
        from schedule.notion_service import sync_schedule_to_notion
    except Exception as e:
        logger.error(f"[SCHEDULE] 노션 서비스 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"노션 서비스 로드 실패: {e}")

    synced = []
    failed = []
    for schedule in targets:
        try:
            page_id = await sync_schedule_to_notion(schedule)
            await update_schedule_notion_id(schedule["schedule_id"], page_id)
            synced.append({
                "schedule_id": schedule["schedule_id"],
                "title": schedule["title"],
                "notion_page_id": page_id,
            })
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"[SCHEDULE] 확정 일정 노션 동기화 실패: {schedule.get('schedule_id')} {e}")
            failed.append({
                "schedule_id": schedule.get("schedule_id"),
                "title": schedule.get("title"),
                "error": str(e),
            })

    return {
        "result": "success" if not failed else "partial_success",
        "message": f"노션에 {len(synced)}개 일정을 저장했습니다.",
        "synced_count": len(synced),
        "failed_count": len(failed),
        "synced": synced,
        "failed": failed,
    }


# 노션 캘린더 → 우리 DB 일정 가져오기
@router.post("/sync-notion-import")
async def schedule_import_from_notion():
    """
    노션 캘린더 데이터베이스의 일정을 조회하여,
    우리 DB에 아직 없는 일정만 confirmed 상태로 저장한다.
    notion_page_id 기준으로 중복을 방지한다.
    """
    try:
        from schedule.notion_service import fetch_schedules_from_notion
        notion_schedules = await fetch_schedules_from_notion()
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=502, detail=str(re))

    if not notion_schedules:
        return {
            "result": "success",
            "message": "노션 캘린더에 일정이 없습니다.",
            "imported_count": 0,
            "imported": [],
        }

    # DB에 이미 연동된 notion_page_id 목록 조회
    existing_ids = await get_all_notion_page_ids()

    # 노션에만 있는 일정 필터링
    new_schedules = [
        s for s in notion_schedules
        if s["notion_page_id"] not in existing_ids
    ]

    if not new_schedules:
        return {
            "result": "success",
            "message": "모든 노션 일정이 이미 DB에 존재합니다.",
            "imported_count": 0,
            "imported": [],
            "total_in_notion": len(notion_schedules),
        }

    # 새 일정을 confirmed 상태로 저장
    imported = []
    for s in new_schedules:
        schedule_id = str(uuid.uuid4())
        # 날짜 형식이 다를수도 있음(시간 저장이 안될때 try로 변경)
        due_date = parse_due_date(s.get("due_date")) if s.get("due_date") else None

        await save_schedule(
            schedule_id=schedule_id,
            session_id=None,
            title=s["title"],
            description=s.get("description"),
            event_type=s.get("event_type"),
            due_date=due_date,
            notion_page_id=s["notion_page_id"],
        )
        # 노션에서 가져온 일정은 바로 confirmed 처리
        await update_schedule_status(schedule_id, "confirmed")

        imported.append({
            "schedule_id": schedule_id,
            "title": s["title"],
            "due_date": s.get("due_date"),
            "event_type": s.get("event_type"),
            "notion_page_id": s["notion_page_id"],
        })

    logger.info(
        f"[SCHEDULE] 노션 캘린더에서 {len(imported)}개 일정 가져오기 완료 "
        f"(노션 전체: {len(notion_schedules)}개, 기존 중복: {len(notion_schedules) - len(new_schedules)}개)"
    )

    return {
        "result": "success",
        "message": f"노션에서 {len(imported)}개 일정을 가져왔습니다.",
        "imported_count": len(imported),
        "imported": imported,
        "total_in_notion": len(notion_schedules),
        "skipped_duplicates": len(notion_schedules) - len(new_schedules),
    }