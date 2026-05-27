import json
from datetime import datetime

from db_api.workspace.common import DEFAULT_FOLDER_DESCRIPTION


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


def week_resource_key(week: dict) -> str:
    return str(
        week.get("weekId")
        or week.get("id")
        or week.get("weekKey")
        or week.get("label")
        or ""
    )


def split_week_resources(weeks: list, resource_key: str) -> list:
    resources = []
    for week in weeks:
        if not isinstance(week, dict):
            continue

        items = week.get(resource_key)
        if not isinstance(items, list) or not items:
            continue

        resources.append({
            "weekId": week.get("id"),
            "weekKey": week.get("weekKey"),
            "label": week.get("label"),
            "dateLabel": week.get("dateLabel"),
            resource_key: items,
        })

    return resources


def flatten_week_resources(value: list, resource_key: str) -> list:
    if not isinstance(value, list):
        return []

    items = []
    for entry in value:
        if isinstance(entry, dict) and isinstance(entry.get(resource_key), list):
            items.extend(entry[resource_key])
        elif isinstance(entry, dict):
            items.append(entry)

    return items


def grouped_week_resources(value: list, resource_key: str) -> dict:
    if not isinstance(value, list):
        return {}

    grouped = {}
    for entry in value:
        if not isinstance(entry, dict) or not isinstance(entry.get(resource_key), list):
            continue

        grouped[week_resource_key(entry)] = entry[resource_key]

    return grouped


def week_shells_from_media(*media_groups: list) -> list:
    weeks = []
    seen = set()
    for media_group in media_groups:
        if not isinstance(media_group, list):
            continue

        for entry in media_group:
            if not isinstance(entry, dict):
                continue

            key = week_resource_key(entry)
            if not key or key in seen:
                continue

            seen.add(key)
            weeks.append({
                "id": entry.get("weekId") or entry.get("id") or entry.get("weekKey"),
                "type": "week",
                "label": entry.get("label") or entry.get("weekLabel") or "1주차",
                "weekKey": entry.get("weekKey"),
                "dateLabel": entry.get("dateLabel"),
                "expanded": True,
                "materialFolderExpanded": True,
                "recordingFolderExpanded": True,
            })

    return weeks


def _light_recording(recording: dict) -> dict:
    """Initial tree payload should keep recording metadata but not full transcript text."""
    if not isinstance(recording, dict):
        return recording

    next_recording = dict(recording)
    transcriptions = next_recording.pop("transcriptions", None)
    if isinstance(transcriptions, list):
        next_recording["transcriptionCount"] = len(transcriptions)
        if transcriptions and not next_recording.get("transcriptionStatus"):
            next_recording["transcriptionStatus"] = "done"
    return next_recording


def _light_voicefile_resources(session_voicefile: list) -> list:
    if not isinstance(session_voicefile, list):
        return []

    light_items = []
    for entry in session_voicefile:
        if not isinstance(entry, dict):
            continue

        next_entry = dict(entry)
        recordings = next_entry.get("recordings")
        if isinstance(recordings, list):
            next_entry["recordings"] = [_light_recording(recording) for recording in recordings]
        else:
            next_entry = _light_recording(next_entry)
        light_items.append(next_entry)
    return light_items


def merge_session_resources(session_pdf: list, session_voicefile: list) -> list:
    weeks = week_shells_from_media(session_pdf, session_voicefile)
    materials_by_week = grouped_week_resources(session_pdf, "materials")
    recordings_by_week = grouped_week_resources(session_voicefile, "recordings")

    for week in weeks:
        key = week_resource_key(week)
        if key in materials_by_week:
            week["materials"] = materials_by_week[key]
        else:
            week["materials"] = week.get("materials") if isinstance(week.get("materials"), list) else []

        if key in recordings_by_week:
            week["recordings"] = recordings_by_week[key]
        else:
            week["recordings"] = week.get("recordings") if isinstance(week.get("recordings"), list) else []

    return weeks


def course_node(row) -> dict:
    # COURSES row를 프론트 폴더 노드 구조로 변환
    description = row["description"]
    return {
        "id": str(row["course_id"]),
        "type": "folder",
        "name": row["title"] or "새 폴더",
        "description": description,
        "isDefaultFolder": description == DEFAULT_FOLDER_DESCRIPTION,
        "date": format_display_date(row["created_at"]),
        "color": row["color"] or "#3b82f6",
        "icon": row["icon"] or "folder",
        "expanded": False,
        "children": [],
    }


def session_node(row, *, include_transcriptions: bool = True) -> dict:
    # SESSIONS row를 프론트 파일 노드 구조로 변환
    session_pdf = json_value(row["session_pdf"], [])
    session_voicefile = json_value(row["session_voicefile"], [])
    if not include_transcriptions:
        session_voicefile = _light_voicefile_resources(session_voicefile)
    weeks = merge_session_resources(session_pdf, session_voicefile)

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
        "attachments": flatten_week_resources(session_pdf, "materials"),
        "recordings": flatten_week_resources(session_voicefile, "recordings"),
        "summaryNotes": json_value(row["summary_notes"], []),
        "weeks": weeks,
        "resourcesLoaded": include_transcriptions,
    }
