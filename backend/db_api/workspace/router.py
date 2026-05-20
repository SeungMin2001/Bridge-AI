from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import logging

from db_api.workspace.common import WorkspaceApiError
from db_api.workspace.courses_api import create_course, delete_course, update_course
from db_api.workspace.files_api import get_workspace_material_file, get_workspace_recording_file, save_workspace_material
from db_api.workspace.recording_transcription_api import transcribe_session_recording
from db_api.workspace.sessions_api import create_session_file, delete_session_file, delete_session_recording, update_session_file, update_session_resources, upload_session_recording
from db_api.workspace.tree_api import get_workspace_tree


router = APIRouter(prefix="/workspace", tags=["workspace"])
logger = logging.getLogger(__name__)


class CreateCourseRequest(BaseModel):
    # 새 폴더를 COURSES 테이블에 저장할 때 프론트에서 받는 값입니다.
    user_id: str | None = None
    parent_course_id: str | None = None
    title: str
    type: str | None = None
    description: str | None = None
    color: str | None = None
    icon: str | None = None


class UpdateCourseRequest(BaseModel):
    # 폴더 수정 시 프론트에서 받는 값입니다.
    title: str
    color: str | None = None
    icon: str | None = None


class CreateSessionFileRequest(BaseModel):
    # 새 파일을 SESSIONS 테이블에 저장할 때 프론트에서 받는 값
    course_id: str | None = None
    title: str
    file_kind: str | None = "lecture"
    tag: str | None = None
    icon: str | None = None
    color: str | None = None
    status: str | None = None


class UpdateSessionFileRequest(BaseModel):
    title: str


class UpdateSessionResourcesRequest(BaseModel):
    # 현재 파일 내부 주차/자료/녹음본 구조
    weeks: list


def _raise_http_error(error: Exception) -> None:
    if isinstance(error, WorkspaceApiError):
        raise HTTPException(status_code=error.status_code, detail=str(error))

    # 신창영: 수정 이유 - 프론트에는 500만 보이므로 백엔드 콘솔에 실제 DB/SQL 에러를 남겨 원인을 확인합니다.
    logger.exception("[WORKSPACE] API 처리 실패")
    raise HTTPException(status_code=500, detail=str(error))


@router.get("/tree")
async def workspace_tree():
    # COURSES와 SESSIONS를 프론트 fileTree 구조로 조회합니다.
    try:
        return await get_workspace_tree()
    except Exception as error:
        _raise_http_error(error)


@router.post("/courses")
async def workspace_create_course(req: CreateCourseRequest):
    # 새 폴더 생성 요청을 COURSES 테이블에 저장합니다.
    try:
        return await create_course(req.model_dump())
    except Exception as error:
        _raise_http_error(error)


@router.put("/courses/{course_id}")
async def workspace_update_course(course_id: str, req: UpdateCourseRequest):
    # 폴더 수정 요청을 COURSES 테이블에 저장합니다.
    try:
        return await update_course(course_id, req.model_dump())
    except Exception as error:
        _raise_http_error(error)


@router.delete("/courses/{course_id}")
async def workspace_delete_course(course_id: str):
    # 폴더 삭제 요청을 COURSES 테이블 삭제로 연결합니다.
    try:
        return await delete_course(course_id)
    except Exception as error:
        _raise_http_error(error)


@router.post("/sessions")
async def workspace_create_session(req: CreateSessionFileRequest):
    # 새 파일 생성 요청을 SESSIONS 테이블에 저장합니다.
    try:
        return await create_session_file(req.model_dump())
    except Exception as error:
        _raise_http_error(error)


@router.delete("/sessions/{session_id}")
async def workspace_delete_session(session_id: str):
    # 파일 삭제 요청을 SESSIONS 테이블 삭제로 연결
    try:
        return await delete_session_file(session_id)
    except Exception as error:
        _raise_http_error(error)


@router.put("/sessions/{session_id}")
async def workspace_update_session(session_id: str, req: UpdateSessionFileRequest):
    # 파일 제목 같은 SESSIONS 기본 정보를 수정합니다.
    try:
        return await update_session_file(session_id, req.model_dump())
    except Exception as error:
        _raise_http_error(error)


@router.put("/sessions/{session_id}/resources")
async def workspace_update_session_resources(session_id: str, req: UpdateSessionResourcesRequest):
    # 현재 파일 내부 강의자료/녹음본 구조를 SESSIONS 테이블에 저장
    try:
        return await update_session_resources(session_id, req.model_dump())
    except Exception as error:
        _raise_http_error(error)


@router.delete("/sessions/{session_id}/recordings/{recording_id}")
async def workspace_delete_session_recording(session_id: str, recording_id: str):
    # 녹음본 하나를 삭제할 때 session_voicefile JSON과 연결된 전사/RAG/일정/요약을 함께 정리
    try:
        return await delete_session_recording(session_id, recording_id)
    except Exception as error:
        _raise_http_error(error)


@router.post("/sessions/{session_id}/materials")
async def workspace_upload_material(session_id: str, file: UploadFile = File(...)):
    # 강의자료 파일을 서버에 저장하고 접근 URL 반환
    try:
        return await save_workspace_material(session_id, file)
    except Exception as error:
        _raise_http_error(error)


@router.post("/sessions/{session_id}/recordings")
async def workspace_upload_recording(
    session_id: str,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    duration_seconds: float | None = Form(None),
):
    # 업로드한 음성파일을 서버에 저장하고 현재 파일의 녹음본 목록에 연결
    try:
        return await upload_session_recording(
            session_id,
            file,
            title=title,
            duration_seconds=duration_seconds,
        )
    except Exception as error:
        _raise_http_error(error)


@router.post("/sessions/{session_id}/recordings/{recording_id}/transcribe")
async def workspace_transcribe_recording(session_id: str, recording_id: str):
    # 업로드된 음성파일을 Whisper로 전사하고 transcripts/RAG/session_voicefile을 동기화
    try:
        return await transcribe_session_recording(session_id, recording_id)
    except Exception as error:
        _raise_http_error(error)


@router.get("/uploads/materials/{stored_name}")
async def workspace_material_file(stored_name: str):
    # 저장된 강의자료 파일 반환
    try:
        return get_workspace_material_file(stored_name)
    except Exception as error:
        _raise_http_error(error)


@router.get("/uploads/recordings/{stored_name}")
async def workspace_recording_file(stored_name: str):
    # 저장된 음성파일 반환
    try:
        return get_workspace_recording_file(stored_name)
    except Exception as error:
        _raise_http_error(error)
