import logging
import os
import re
from datetime import datetime

from db import get_pool
from materials.material_text_service import (
    MaterialTextError,
    extract_pdf_text,
    get_session_pdf_materials,
    stored_name_from_material,
)
from rag_search import add_document as rag_add_document


logger = logging.getLogger(__name__)

MATERIAL_RAG_CHUNK_CHARS = int(os.getenv("MATERIAL_RAG_CHUNK_CHARS", "900"))
MATERIAL_RAG_CHUNK_OVERLAP = int(os.getenv("MATERIAL_RAG_CHUNK_OVERLAP", "120"))


def _rag_table_name() -> str:
    raw_name = os.getenv("RAG_TABLE_NAME", "rag")
    safe_name = re.sub(r"[^A-Za-z0-9_]", "", raw_name) or "rag"
    return f"data_{safe_name}"


def _quote_ident(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _material_key(material: dict) -> tuple[str, str]:
    material_id = str(material.get("id") or "").strip()
    stored_name = stored_name_from_material(material) or ""
    return material_id, stored_name


def _chunk_text_by_page(material_text: str) -> list[dict]:
    """PDF 추출 텍스트를 페이지/길이 기준으로 RAG용 작은 조각으로 나눈다."""
    chunks = []
    page_blocks = re.split(r"(?=\[PDF page \d+\])", material_text or "")
    chunk_index = 0

    for block in page_blocks:
        text = block.strip()
        if not text:
            continue

        page_match = re.search(r"\[PDF page (\d+)\]", text)
        page = int(page_match.group(1)) if page_match else None
        text = re.sub(r"\[PDF page \d+\]\s*", "", text).strip()
        if not text:
            continue

        start = 0
        step = max(1, MATERIAL_RAG_CHUNK_CHARS - MATERIAL_RAG_CHUNK_OVERLAP)
        while start < len(text):
            part = text[start:start + MATERIAL_RAG_CHUNK_CHARS].strip()
            if part:
                chunks.append({
                    "text": part,
                    "page": page,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1
            start += step

    return chunks


async def delete_session_material_rag(session_id: str) -> int:
    """해당 세션의 기존 PDF RAG 조각을 지워 중복/삭제 자료 노출을 막는다."""
    pool = await get_pool()
    table_name = _rag_table_name()
    async with pool.acquire() as conn:
        try:
            result = await conn.execute(
                f"""
                DELETE FROM {table_name}
                WHERE metadata_->>'session_id' = $1
                  AND metadata_->>'source_type' = 'material'
                """,
                str(session_id),
            )
            return int(result.rsplit(" ", 1)[-1])
        except Exception as exc:
            logger.warning("[MATERIAL:RAG] 기존 PDF RAG 삭제 스킵: %s", exc)
            return 0


async def _session_material_rag_counts(session_id: str) -> dict[tuple[str, str], int]:
    """현재 세션에 이미 인덱싱된 PDF 조각 수를 material 단위로 조회한다."""
    pool = await get_pool()
    table_name = _quote_ident(_rag_table_name())
    async with pool.acquire() as conn:
        try:
            rows = await conn.fetch(
                f"""
                SELECT
                    COALESCE(metadata_->>'material_id', '') AS material_id,
                    COALESCE(metadata_->>'stored_name', '') AS stored_name,
                    COUNT(*) AS chunk_count
                FROM {table_name}
                WHERE metadata_->>'session_id' = $1
                  AND metadata_->>'source_type' = 'material'
                GROUP BY 1, 2
                """,
                str(session_id),
            )
        except Exception as exc:
            logger.info("[MATERIAL:RAG] PDF RAG 기존 인덱스 조회 스킵: %s", exc)
            return {}

    return {
        (str(row["material_id"] or ""), str(row["stored_name"] or "")): int(row["chunk_count"] or 0)
        for row in rows
    }


async def _delete_stale_session_material_rag(
    session_id: str,
    active_keys: set[tuple[str, str]],
    existing_counts: dict[tuple[str, str], int],
) -> int:
    """현재 세션에서 더 이상 연결되지 않은 PDF RAG 조각만 삭제한다."""
    stale_keys = [key for key in existing_counts if key not in active_keys]
    if not stale_keys:
        return 0

    pool = await get_pool()
    table_name = _quote_ident(_rag_table_name())
    deleted_count = 0
    async with pool.acquire() as conn:
        for material_id, stored_name in stale_keys:
            try:
                result = await conn.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE metadata_->>'session_id' = $1
                      AND metadata_->>'source_type' = 'material'
                      AND COALESCE(metadata_->>'material_id', '') = $2
                      AND COALESCE(metadata_->>'stored_name', '') = $3
                    """,
                    str(session_id),
                    material_id,
                    stored_name,
                )
                deleted_count += int(result.rsplit(" ", 1)[-1])
            except Exception as exc:
                logger.warning("[MATERIAL:RAG] 제거된 PDF RAG 삭제 스킵: %s", exc)

    return deleted_count


async def count_session_material_rag(session_id: str) -> int:
    pool = await get_pool()
    table_name = _rag_table_name()
    async with pool.acquire() as conn:
        try:
            return await conn.fetchval(
                f"""
                SELECT COUNT(*)
                FROM {table_name}
                WHERE metadata_->>'session_id' = $1
                  AND metadata_->>'source_type' = 'material'
                """,
                str(session_id),
            )
        except Exception as exc:
            logger.info("[MATERIAL:RAG] PDF RAG 카운트 스킵: %s", exc)
            return 0


async def sync_session_materials_to_rag(session_id: str) -> dict:
    """
    세션에 연결된 PDF 강의자료를 RAG 벡터 스토어에 등록한다.
    이미 인덱싱된 PDF는 유지하고, 새 PDF만 추가하며, 연결 해제된 PDF 조각만 삭제한다.
    """
    try:
        materials = await get_session_pdf_materials(session_id)
    except MaterialTextError as exc:
        logger.info("[MATERIAL:RAG] 인덱싱할 PDF 없음: session=%s, reason=%s", session_id, exc)
        return {
            "deleted": await delete_session_material_rag(session_id),
            "indexed_materials": 0,
            "indexed_chunks": 0,
            "skipped_materials": 0,
        }

    existing_counts = await _session_material_rag_counts(session_id)
    active_keys = {_material_key(material) for material in materials}
    deleted_count = await _delete_stale_session_material_rag(session_id, active_keys, existing_counts)

    indexed_materials = 0
    indexed_chunks = 0
    skipped_materials = 0
    indexed_at = datetime.now().isoformat()

    for material in materials:
        stored_name = stored_name_from_material(material)
        material_name = material.get("name") or material.get("title") or stored_name or "강의자료"
        material_id = str(material.get("id") or "")
        material_key = (material_id, stored_name or "")
        if existing_counts.get(material_key, 0) > 0:
            skipped_materials += 1
            continue

        try:
            material_text = extract_pdf_text(material)
        except MaterialTextError as exc:
            logger.warning("[MATERIAL:RAG] PDF 텍스트 추출 실패: %s (%s)", material_name, exc)
            continue

        chunks = _chunk_text_by_page(material_text)
        if not chunks:
            continue

        for chunk in chunks:
            rag_add_document(chunk["text"], {
                "source_type": "material",
                "session_id": str(session_id),
                "material_id": material_id,
                "material_name": material_name,
                "stored_name": stored_name or "",
                "page": chunk.get("page") or 0,
                "chunk_index": chunk.get("chunk_index"),
                "created_at": indexed_at,
            })
            indexed_chunks += 1

        indexed_materials += 1
        existing_counts[material_key] = len(chunks)

    logger.info(
        "[MATERIAL:RAG] PDF 인덱싱 완료: session=%s, indexed=%s, skipped=%s, chunks=%s, deleted=%s",
        session_id,
        indexed_materials,
        skipped_materials,
        indexed_chunks,
        deleted_count,
    )
    return {
        "deleted": deleted_count,
        "indexed_materials": indexed_materials,
        "indexed_chunks": indexed_chunks,
        "skipped_materials": skipped_materials,
    }


async def ensure_session_materials_indexed(session_id: str) -> dict:
    # 채팅 직전에도 증분 sync만 수행하므로 이미 저장된 PDF는 다시 파싱하지 않는다.
    return await sync_session_materials_to_rag(session_id)
