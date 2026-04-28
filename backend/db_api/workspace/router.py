from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db_api.workspace.common import WorkspaceApiError
from db_api.workspace.courses_api import create_course
from db_api.workspace.sessions_api import create_session_file, delete_session_file
from db_api.workspace.tree_api import get_workspace_tree


router = APIRouter(prefix="/workspace", tags=["workspace"])


class CreateCourseRequest(BaseModel):
    # 새 폴더를 COURSES 테이블에 저장할 때 프론트에서 받는 값입니다.
    user_id: str | None = None
    parent_course_id: str | None = None
    title: str
    type: str | None = None
    description: str | None = None
    color: str | None = None
    icon: str | None = None


class CreateSessionFileRequest(BaseModel):
    # 새 파일을 SESSIONS 테이블에 저장할 때 프론트에서 받는 값입니다.
    course_id: str | None = None
    title: str
    file_kind: str | None = "lecture"
    tag: str | None = None
    icon: str | None = None
    color: str | None = None
    status: str | None = None


def _raise_http_error(error: Exception) -> None:
    if isinstance(error, WorkspaceApiError):
        raise HTTPException(status_code=error.status_code, detail=str(error))

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


@router.post("/sessions")
async def workspace_create_session(req: CreateSessionFileRequest):
    # 새 파일 생성 요청을 SESSIONS 테이블에 저장합니다.
    try:
        return await create_session_file(req.model_dump())
    except Exception as error:
        _raise_http_error(error)


@router.delete("/sessions/{session_id}")
async def workspace_delete_session(session_id: str):
    # 파일 삭제 요청을 SESSIONS 테이블 삭제로 연결합니다.
    try:
        return await delete_session_file(session_id)
    except Exception as error:
        _raise_http_error(error)
