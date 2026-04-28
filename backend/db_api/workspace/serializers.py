import json
from datetime import datetime


def format_display_date(value) -> str:
    # DB timestamp를 기존 프론트 fileTree 날짜 문자열로 변환
    if not value:
        return ""

    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value

    period = "오후" if value.hour >= 12 else "오전"
    hour = value.hour % 12 or 12
    minute = f"{value.minute:02d}"
    return f"{value.year}. {value.month}. {value.day}. {period} {hour}:{minute}"


def json_value(value, fallback):
    # JSONB 컬럼 값을 프론트에서 바로 쓸 수 있는 list/dict로 정리
    if value in (None, ""):
        return fallback
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return fallback


def course_node(row) -> dict:
    # COURSES row를 프론트 폴더 노드 구조로 변환
    return {
        "id": str(row["course_id"]),
        "type": "folder",
        "name": row["title"] or "새 폴더",
        "date": format_display_date(row["created_at"]),
        "color": row["color"] or "#3b82f6",
        "icon": row["icon"] or "folder",
        "expanded": False,
        "children": [],
    }


def session_node(row) -> dict:
    # SESSIONS row를 프론트 파일 노드 구조로 변환
    resource_tree = json_value(row["resource_tree"], [])
    return {
        "id": str(row["session_id"]),
        "type": "file",
        "fileKind": row["file_kind"] or "lecture",
        "name": row["title"] or "새 파일",
        "date": format_display_date(row["created_at"]),
        "color": row["color"] or "#3b82f6",
        "tag": row["tag"] or ("회의" if row["file_kind"] == "meeting" else "수업"),
        "fileIcon": row["icon"] or ("groups_2" if row["file_kind"] == "meeting" else "article"),
        "content": "",
        "attachments": json_value(row["session_pdf"], []),
        "summaryNotes": json_value(row["summary_notes"], []),
        "weeks": resource_tree,
    }
