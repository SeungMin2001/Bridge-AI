"""AI 채팅용 근거 컨텍스트 생성 서비스.

RAG 검색 결과, 선택 PDF 보강 컨텍스트, 현재 워크스페이스의 저장 목록을 합쳐
LLM에게 전달할 최종 prompt와 프론트에 보여줄 citations를 만듭니다.
"""

import json
import logging
import os

from rag_search import search as rag_search


logger = logging.getLogger(__name__)

CHAT_EVIDENCE_TOP_K = int(os.getenv("CHAT_EVIDENCE_TOP_K", "5"))
CHAT_SELECTED_MATERIAL_CONTEXT_CHARS = int(os.getenv("CHAT_SELECTED_MATERIAL_CONTEXT_CHARS", "12000"))
CHAT_SELECTED_MATERIAL_CONTEXT_PER_FILE_CHARS = int(os.getenv("CHAT_SELECTED_MATERIAL_CONTEXT_PER_FILE_CHARS", "4000"))
CHAT_WORKSPACE_INVENTORY_MAX_ITEMS = int(os.getenv("CHAT_WORKSPACE_INVENTORY_MAX_ITEMS", "40"))


async def build_prompt_and_citations(
    question: str,
    session_id: str | None = None,
    source_filter: dict | None = None,
) -> tuple[str, list[dict]]:
    """질문, 현재 파일, 선택 자료 기준으로 LLM prompt와 citation 목록을 구성합니다."""
    has_selected_material = source_filter_has_material(source_filter)
    inventory_context = await _build_workspace_inventory_context(session_id)
    rag_result = rag_search(
        question,
        top_k=CHAT_EVIDENCE_TOP_K,
        session_id=session_id,
        source_filter=source_filter,
    )
    context = rag_result["context"]
    citations = rag_result["citations"]

    if has_selected_material and not _has_material_citation(citations):
        material_context, material_citations = await _build_material_context_fallback(
            question,
            session_id,
            source_filter,
        )
        if not material_context:
            material_context, material_citations = await _build_selected_material_context(
                session_id,
                source_filter,
            )
        if material_context:
            context = "\n\n".join(part for part in (context, material_context) if part)
            citations = _merge_citations(citations, material_citations)[:CHAT_EVIDENCE_TOP_K]

    reference_context = "\n\n".join(
        part for part in (
            inventory_context,
            f"[검색된 참고자료]\n{context}" if context else "",
        )
        if part
    )

    if reference_context:
        missing_selected_material_note = (
            "현재 선택된 PDF에 대해 질문과 직접 관련된 근거 페이지를 찾지 못했습니다. "
            "저장 목록은 파일명/개수/시간 같은 메타데이터만 담고 있으므로, PDF 내용 질문이면 근거를 찾지 못했다고 답하세요. "
            if has_selected_material and not context
            else ""
        )
        prompt = (
            f"다음은 현재 워크스페이스 파일의 저장 목록과 강의 내용에서 검색된 참고자료입니다:\n\n{reference_context}\n\n"
            f"사용자가 현재 파일에 저장된 자료/녹음본 목록, 개수, 이름을 물으면 "
            f"[현재 워크스페이스 파일 저장 목록]을 우선 기준으로 답하세요. "
            f"저장 목록은 강의 내용 근거가 아니라 파일명/메타데이터 목록입니다. "
            f"{missing_selected_material_note}"
            f"사용자가 강의 내용, PDF 페이지, 전사 내용의 의미를 물으면 [검색된 참고자료]를 바탕으로 답변하세요. "
            f"선택된 PDF/녹음본 밖의 내용은 추측하지 마세요. "
            f"PDF 근거가 있으면 자료명과 p.페이지 번호를 답변 본문에 반드시 포함하세요. "
            f"페이지 위치를 묻는 질문이면 관련 페이지 번호를 먼저 답하세요. "
            f"답변 마지막에 참고한 출처를 '[출처]' 형식으로 표시해주세요.\n\n"
            f"질문: {question}"
        )
    elif has_selected_material:
        prompt = (
            "사용자가 PDF 자료를 선택했지만, 선택된 PDF 안에서 질문과 직접 관련된 근거 페이지를 찾지 못했습니다. "
            "외부 웹사이트나 일반 지식으로 대체하지 말고, 선택된 PDF에서 근거를 찾지 못했다고 짧게 답하세요.\n\n"
            f"질문: {question}"
        )
    else:
        prompt = question

    return prompt, citations


async def ensure_material_rag_for_chat(session_id: str | None, source_filter: dict | None):
    """선택된 PDF 자료가 있으면 채팅 전에 해당 세션 자료의 RAG 인덱싱을 보장합니다."""
    if not session_id or not source_filter_has_material(source_filter):
        return
    try:
        from materials.material_rag_service import ensure_session_materials_indexed

        await ensure_session_materials_indexed(session_id)
    except Exception as exc:
        logger.warning("[CHAT] PDF RAG 인덱싱 확인 실패: session=%s, error=%s", session_id, exc)


def source_filter_has_any_source(source_filter: dict | None) -> bool:
    """source_filter에 PDF/녹음본/전사 등 하나 이상의 선택 근거가 들어있는지 확인합니다."""
    if not isinstance(source_filter, dict):
        return False
    return any(
        source_filter.get(key)
        for key in ("material_ids", "stored_names", "recording_ids", "transcript_ids")
    )


def source_filter_has_material(source_filter: dict | None) -> bool:
    """source_filter에 선택된 PDF 자료 식별자가 들어있는지 확인합니다."""
    if not isinstance(source_filter, dict):
        return False
    return bool(source_filter.get("material_ids") or source_filter.get("stored_names"))


async def _build_material_context_fallback(
    question: str,
    session_id: str | None,
    source_filter: dict | None,
) -> tuple[str, list[dict]]:
    """RAG에서 PDF citation을 못 찾았을 때 페이지 단위 PDF 검색으로 보강합니다."""
    if not session_id or not source_filter_has_material(source_filter):
        return "", []

    try:
        from materials.material_page_search_service import build_material_page_context

        return await build_material_page_context(question, session_id, source_filter)
    except Exception as exc:
        logger.warning("[CHAT] PDF 페이지 검색 fallback 실패: session=%s, error=%s", session_id, exc)
        return "", []


def _has_material_citation(citations: list[dict]) -> bool:
    """citation 목록에 PDF 자료 근거가 하나라도 있는지 확인합니다."""
    return any((item or {}).get("source_type") == "material" for item in citations or [])


def _truncate_chat_context(text: str, max_chars: int) -> str:
    """LLM prompt가 너무 길어지지 않도록 컨텍스트 문자열을 지정 길이로 자릅니다."""
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n\n...[자료가 길어 일부만 사용했습니다]"


def _chat_json_value(value, fallback):
    """DB에 JSON 문자열 또는 이미 파싱된 객체로 저장된 값을 안전하게 Python 값으로 변환합니다."""
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


def _compact_chat_text(value, fallback: str = "") -> str:
    """목록 표시용 텍스트에서 연속 공백을 정리하고 비어 있으면 fallback을 반환합니다."""
    text = " ".join(str(value or "").split())
    return text or fallback


def _iter_week_resource_items(value, resource_key: str):
    """주차별 저장 구조에서 자료/녹음본 항목을 week_label과 함께 순회합니다."""
    for entry in _chat_json_value(value, []):
        if not isinstance(entry, dict):
            continue

        week_label = _compact_chat_text(
            entry.get("label")
            or entry.get("weekLabel")
            or entry.get("weekKey")
            or entry.get("dateLabel")
        )
        items = entry.get(resource_key)
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    yield week_label, item
        else:
            yield week_label, entry


def _format_inventory_detail(parts: list[str]) -> str:
    """저장 목록 한 줄 뒤에 붙일 괄호형 메타데이터 문자열을 만듭니다."""
    clean_parts = [part for part in parts if part]
    return f" ({', '.join(clean_parts)})" if clean_parts else ""


def _format_material_inventory_line(index: int, week_label: str, material: dict) -> str:
    """강의자료 항목 하나를 LLM이 읽기 쉬운 저장 목록 문자열로 포맷합니다."""
    name = _compact_chat_text(
        material.get("name")
        or material.get("title")
        or material.get("storedName")
        or material.get("url"),
        f"강의자료 {index}",
    )
    details = _format_inventory_detail([
        f"주차: {week_label}" if week_label else "",
        _compact_chat_text(material.get("type") or material.get("fileType") or material.get("mimeType")),
    ])
    return f"- {name}{details}"


def _format_recording_inventory_line(index: int, week_label: str, recording: dict) -> str:
    """녹음본 항목 하나를 LLM이 읽기 쉬운 저장 목록 문자열로 포맷합니다."""
    title = _compact_chat_text(
        recording.get("title")
        or recording.get("name")
        or recording.get("recordingTitle")
        or recording.get("recordingId")
        or recording.get("id"),
        f"녹음본 {index}",
    )
    material_names = recording.get("materialNames")
    if isinstance(material_names, list):
        material_label = ", ".join(_compact_chat_text(name) for name in material_names if _compact_chat_text(name))
    else:
        material_label = _compact_chat_text(material_names)

    details = _format_inventory_detail([
        f"주차: {week_label}" if week_label else "",
        f"시간: {_compact_chat_text(recording.get('startedAt') or recording.get('createdAt') or recording.get('uploadedAt'))}"
        if (recording.get("startedAt") or recording.get("createdAt") or recording.get("uploadedAt"))
        else "",
        f"길이: {_compact_chat_text(recording.get('durationText') or recording.get('duration') or recording.get('durationSec'))}"
        if (recording.get("durationText") or recording.get("duration") or recording.get("durationSec"))
        else "",
        f"연결 자료: {material_label}" if material_label else "",
    ])
    return f"- {title}{details}"


async def _build_workspace_inventory_context(session_id: str | None) -> str:
    """현재 워크스페이스 파일에 저장된 PDF/녹음본 목록 메타데이터를 prompt context로 만듭니다."""
    if not session_id:
        return ""

    try:
        from db import get_pool

        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT title, session_pdf, session_voicefile
                FROM sessions
                WHERE session_id = $1::uuid
                """,
                session_id,
            )
    except Exception as exc:
        logger.warning("[CHAT] 워크스페이스 저장 목록 조회 실패: session=%s, error=%s", session_id, exc)
        return ""

    if row is None:
        return ""

    material_lines = []
    recording_lines = []
    for index, (week_label, material) in enumerate(
        _iter_week_resource_items(row["session_pdf"], "materials"),
        1,
    ):
        if index > CHAT_WORKSPACE_INVENTORY_MAX_ITEMS:
            material_lines.append("- ...[강의자료 목록 일부 생략]")
            break
        material_lines.append(_format_material_inventory_line(index, week_label, material))

    for index, (week_label, recording) in enumerate(
        _iter_week_resource_items(row["session_voicefile"], "recordings"),
        1,
    ):
        if index > CHAT_WORKSPACE_INVENTORY_MAX_ITEMS:
            recording_lines.append("- ...[녹음본 목록 일부 생략]")
            break
        recording_lines.append(_format_recording_inventory_line(index, week_label, recording))

    file_title = _compact_chat_text(row["title"], "현재 파일")
    return "\n".join([
        "[현재 워크스페이스 파일 저장 목록]",
        f"파일명: {file_title}",
        "",
        "[저장된 녹음본]",
        *(recording_lines or ["- 없음"]),
        "",
        "[저장된 강의자료]",
        *(material_lines or ["- 없음"]),
    ]).strip()


async def _build_selected_material_context(
    session_id: str | None,
    source_filter: dict | None,
) -> tuple[str, list[dict]]:
    """선택 PDF 검색이 실패했을 때 PDF 전체 텍스트 일부를 fallback 근거로 구성합니다."""
    if not session_id or not source_filter_has_material(source_filter):
        return "", []

    try:
        from materials.material_citation_service import build_material_citation, format_material_citation
        from materials.material_text_service import (
            MaterialTextError,
            extract_pdf_text,
            get_session_pdf_materials,
            stored_name_from_material,
        )

        material_ids = source_filter.get("material_ids") if isinstance(source_filter, dict) else None
        stored_names = source_filter.get("stored_names") if isinstance(source_filter, dict) else None
        materials = await get_session_pdf_materials(
            session_id,
            material_ids=material_ids,
            stored_names=stored_names,
        )
    except Exception as exc:
        logger.warning("[CHAT] 선택 PDF 전체 context fallback 준비 실패: session=%s, error=%s", session_id, exc)
        return "", []

    context_parts = []
    citations = []
    used_chars = 0
    for index, material in enumerate(materials, 1):
        try:
            material_text = extract_pdf_text(material)
        except MaterialTextError as exc:
            logger.warning("[CHAT] 선택 PDF 텍스트 추출 실패: material=%s, error=%s", material.get("name"), exc)
            continue

        remaining_chars = CHAT_SELECTED_MATERIAL_CONTEXT_CHARS - used_chars
        if remaining_chars <= 0:
            break

        material_name = material.get("name") or material.get("title") or stored_name_from_material(material) or "강의자료"
        excerpt_limit = min(CHAT_SELECTED_MATERIAL_CONTEXT_PER_FILE_CHARS, remaining_chars)
        excerpt = _truncate_chat_context(material_text, excerpt_limit)
        if not excerpt:
            continue

        stored_name = stored_name_from_material(material) or ""
        citation_result = {
            "text": excerpt,
            "full_transcript": material_text,
            "material_id": str(material.get("id") or ""),
            "material_name": material_name,
            "stored_name": stored_name,
            "page": 0,
            "chunk_index": None,
        }
        citation = format_material_citation(citation_result)
        context_parts.append(
            f"[PDF 전체 {index}] 자료명: {material_name}\n"
            f"{excerpt}\n"
            f"(출처: {citation})"
        )
        citations.append(build_material_citation(
            citation_result,
            citation=citation,
            session_id=session_id,
            search_scope="selected_pdf_context",
        ))
        used_chars += len(excerpt)

    return "\n\n".join(context_parts), citations


def _merge_citations(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """중복 citation을 제거하면서 기존 검색 근거와 fallback 근거를 합칩니다."""
    merged = []
    seen = set()
    for item in [*(primary or []), *(secondary or [])]:
        key = (
            item.get("source_type", ""),
            item.get("material_id", ""),
            item.get("stored_name", ""),
            item.get("recording_id", ""),
            item.get("transcript_id", ""),
            item.get("page", ""),
            item.get("citation", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged
