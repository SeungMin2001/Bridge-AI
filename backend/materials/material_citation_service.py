from urllib.parse import quote


def format_material_citation(result: dict) -> str:
    title = result.get("material_name") or result.get("file_title") or "강의자료"
    page = _safe_int(result.get("page"))
    if page > 0:
        return f"{title} > p.{page}"
    return title


def build_material_citation(
    result: dict,
    *,
    citation: str | None = None,
    session_id: str | None = None,
    search_scope: str = "",
) -> dict:
    text = str(result.get("text") or "")
    stored_name = str(result.get("stored_name") or result.get("storedName") or "")
    page = _safe_int(result.get("page"))
    material_name = result.get("material_name") or result.get("file_title") or "강의자료"

    return {
        "text": text,
        "citation": citation or format_material_citation(result),
        "course_title": result.get("course_title", ""),
        "session_title": result.get("session_title", ""),
        "session_date": result.get("session_date", ""),
        "start_time": 0,
        "end_time": 0,
        "transcript_id": "",
        "recording_id": "",
        "material_id": result.get("material_id", ""),
        "material_name": material_name,
        "file_title": material_name,
        "stored_name": stored_name,
        "material_url": _material_url(stored_name),
        "page": page,
        "page_label": f"PDF page {page}" if page > 0 else "PDF 원문",
        "chunk_index": result.get("chunk_index", None),
        "created_at": result.get("created_at", ""),
        "session_id": result.get("session_id") or session_id or "",
        "search_scope": search_scope,
        "source_type": "material",
        "full_transcript": result.get("full_transcript") or text,
    }


def _material_url(stored_name: str) -> str:
    if not stored_name:
        return ""
    return f"/workspace/uploads/materials/{quote(stored_name)}"


def _safe_int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
