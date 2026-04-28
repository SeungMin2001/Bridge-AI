from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from db_api.workspace.common import WorkspaceApiError
from db_api.workspace.courses_api import create_course, delete_course, update_course
from db_api.workspace.files_api import get_workspace_material_file, save_workspace_material
from db_api.workspace.sessions_api import create_session_file, delete_session_file, update_session_resource_tree
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


class UpdateSessionResourceTreeRequest(BaseModel):
    # 현재 파일 내부 주차/자료/녹음본 구조
    resource_tree: list


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


@router.put("/sessions/{session_id}/resource-tree")
async def workspace_update_session_resource_tree(session_id: str, req: UpdateSessionResourceTreeRequest):
    # 현재 파일 내부 폴더 구조를 SESSIONS 테이블에 저장
    try:
        return await update_session_resource_tree(session_id, req.model_dump())
    except Exception as error:
        _raise_http_error(error)


@router.post("/sessions/{session_id}/materials")
async def workspace_upload_material(session_id: str, file: UploadFile = File(...)):
    # 강의자료 파일을 서버에 저장하고 접근 URL 반환
    try:
        return await save_workspace_material(session_id, file)
    except Exception as error:
        _raise_http_error(error)


@router.get("/uploads/materials/{stored_name}")
async def workspace_material_file(stored_name: str):
    # 저장된 강의자료 파일 반환
    try:
        return get_workspace_material_file(stored_name)
    except Exception as error:
        _raise_http_error(error)
