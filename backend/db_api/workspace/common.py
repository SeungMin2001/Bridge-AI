from uuid import UUID


class WorkspaceApiError(Exception):
    # 워크스페이스 API에서 클라이언트에게 내려줄 오류 상태를 함께 보관.
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def uuid_or_none(value, field_name: str):
    # 프론트에서 빈 문자열로 넘어온 UUID 값은 DB NULL로 저장.
    if value in (None, ""):
        return None

    try:
        return UUID(str(value))
    except ValueError as exc:
        raise WorkspaceApiError(f"{field_name} must be a valid UUID.") from exc


def required_text(payload: dict, field_name: str) -> str:
    # title처럼 반드시 필요한 문자열 필드를 검증합니다.
    value = str(payload.get(field_name) or "").strip()
    if not value:
        raise WorkspaceApiError(f"{field_name} is required.")
    return value
