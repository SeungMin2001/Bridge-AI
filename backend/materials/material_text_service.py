import json
import os
import re
from statistics import median
from pathlib import Path
from typing import Any
from uuid import UUID

from pypdf import PdfReader

from db import get_pool
from db_api.workspace.files_api import MATERIAL_UPLOAD_DIR


PDF_TEXT_MAX_CHARS = int(os.getenv("MATERIAL_PDF_TEXT_MAX_CHARS", "120000"))
PDF_TEXT_MAX_PAGES = int(os.getenv("MATERIAL_PDF_TEXT_MAX_PAGES", "80"))


class MaterialTextError(Exception):
    # API 레이어에서 그대로 HTTP 상태 코드로 변환할 수 있도록 상태를 함께 담는다.
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def _json_value(value: Any, fallback):
    # DB 컬럼이 JSON 문자열/객체/빈 값 중 무엇으로 와도 호출부는 같은 형태로 다루게 한다.
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


def _iter_materials(session_pdf) -> list[dict]:
    # session_pdf는 과거/현재 저장 구조가 섞일 수 있어 중첩된 materials 배열까지 풀어낸다.
    materials = []
    for entry in _json_value(session_pdf, []):
        if not isinstance(entry, dict):
            continue

        items = entry.get("materials")
        if isinstance(items, list):
            materials.extend(item for item in items if isinstance(item, dict))
        else:
            materials.append(entry)
    return materials


def stored_name_from_material(material: dict) -> str | None:
    # 우선 저장 파일명을 쓰고, 없는 경우 업로드 URL 끝부분에서 파일명을 복원한다.
    stored_name = material.get("storedName")
    if stored_name:
        return Path(str(stored_name)).name

    url = material.get("url")
    if isinstance(url, str) and "/workspace/uploads/materials/" in url:
        return Path(url.rsplit("/", 1)[-1]).name

    return None


def is_pdf_material(material: dict, stored_name: str = "") -> bool:
    name = str(material.get("name") or "")
    content_type = str(material.get("type") or "")
    return (
        "pdf" in content_type.lower()
        or name.lower().endswith(".pdf")
        or stored_name.lower().endswith(".pdf")
    )


def _material_path(stored_name: str) -> Path:
    # 파일명만 허용해서 ../ 같은 경로 조작이 업로드 디렉터리 밖으로 나가지 못하게 한다.
    safe_name = Path(stored_name).name
    if safe_name != stored_name:
        raise MaterialTextError("잘못된 강의자료 파일명입니다.", status_code=400)

    upload_root = MATERIAL_UPLOAD_DIR.resolve()
    target_path = (upload_root / safe_name).resolve()
    # resolve 이후에도 최종 경로가 업로드 루트 내부인지 다시 확인한다.
    if upload_root not in target_path.parents and target_path != upload_root:
        raise MaterialTextError("강의자료 파일 경로가 올바르지 않습니다.", status_code=400)
    if not target_path.is_file():
        raise MaterialTextError("강의자료 파일을 찾을 수 없습니다.", status_code=404)

    return target_path


def _clean_pdf_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _font_name(font_dict) -> str:
    if not font_dict:
        return ""

    for key in ("/BaseFont", "/FontName", "/Name"):
        value = font_dict.get(key)
        if value:
            return str(value)
    return ""


def _is_bold_font(font_name: str) -> bool:
    normalized = font_name.lower()
    return any(token in normalized for token in ("bold", "black", "heavy", "semibold", "demi"))


def _looks_like_code_or_formula(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False

    formula_chars = sum(1 for char in compact if char in "[]=+*/{};<>")
    if formula_chars >= 3:
        return True
    if formula_chars >= 2 and re.search(r"[A-Za-z_]\w*|\d", compact):
        return True
    if re.search(r"[A-Za-z_]\w*(?:\[[^\]]+\])+\s*=", compact):
        return True
    if re.search(r"\b(for|while|if|else|return|int|float|double|char|void)\b", text):
        return True
    return False


def _normalize_heading_hint(text: str) -> str:
    text = _clean_pdf_text(text)
    text = re.sub(r"^[❖▪•\-\s]+", "", text)
    return text.strip()


def _is_clean_heading_hint(text: str) -> bool:
    text = _normalize_heading_hint(text)
    if not text or len(text) > 80:
        return False
    if _looks_like_code_or_formula(text):
        return False
    if not re.search(r"[가-힣A-Za-z]", text):
        return False
    if len(re.sub(r"[^\w가-힣]", "", text)) < 2:
        return False
    return True


def _extract_page_fragments(page) -> list[dict]:
    fragments = []

    def visitor_text(text, cm, tm, font_dict, font_size):
        cleaned = _clean_pdf_text(text)
        if not cleaned:
            return

        font_name = _font_name(font_dict)
        try:
            x = float(tm[4])
            y = float(tm[5])
            size = float(font_size or 0)
        except (TypeError, ValueError, IndexError):
            x = 0.0
            y = 0.0
            size = 0.0

        fragments.append(
            {
                "text": cleaned,
                "x": x,
                "y": y,
                "size": size,
                "bold": _is_bold_font(font_name),
            }
        )

    try:
        page.extract_text(visitor_text=visitor_text)
    except TypeError:
        return []
    except Exception:
        return []

    return fragments


def _group_fragments_into_lines(fragments: list[dict]) -> list[dict]:
    if not fragments:
        return []

    sizes = [fragment["size"] for fragment in fragments if fragment.get("size", 0) > 0]
    body_size = median(sizes) if sizes else 0
    y_tolerance = max(2.0, body_size * 0.35) if body_size else 3.0

    lines = []
    current = []
    current_y = None
    for fragment in sorted(fragments, key=lambda item: (-item["y"], item["x"])):
        if current_y is None or abs(fragment["y"] - current_y) <= y_tolerance:
            current.append(fragment)
            current_y = fragment["y"] if current_y is None else current_y
            continue

        lines.append(_merge_line_fragments(current))
        current = [fragment]
        current_y = fragment["y"]

    if current:
        lines.append(_merge_line_fragments(current))

    return [line for line in lines if line["text"]]


def _merge_line_fragments(fragments: list[dict]) -> dict:
    fragments = sorted(fragments, key=lambda item: item["x"])
    parts = [fragment["text"] for fragment in fragments if fragment.get("text")]
    sizes = [fragment["size"] for fragment in fragments if fragment.get("size", 0) > 0]
    return {
        "text": _clean_pdf_text(" ".join(parts)),
        "size": max(sizes) if sizes else 0,
        "bold": any(fragment.get("bold") for fragment in fragments),
    }


def _classify_pdf_line(line: dict, body_size: float) -> str:
    text = line["text"]
    size = line.get("size", 0)
    bold = line.get("bold", False)
    shortish = len(text) <= 90

    if not body_size or size <= 0:
        return "TEXT"
    if _looks_like_code_or_formula(text):
        return "TEXT"
    if shortish and size >= body_size * 1.55:
        return "H1"
    if shortish and (size >= body_size * 1.28 or (bold and size >= body_size * 1.12)):
        return "H2"
    if shortish and bold:
        return "H3"
    return "TEXT"


def _format_styled_page_text(page, page_index: int) -> tuple[str, int]:
    try:
        plain_text = _clean_pdf_text(page.extract_text() or "")
    except Exception:
        plain_text = ""

    fragments = _extract_page_fragments(page)
    lines = _group_fragments_into_lines(fragments)
    sizes = [line["size"] for line in lines if line.get("size", 0) > 0]
    body_size = median(sizes) if sizes else 0

    if not lines:
        return plain_text, len(plain_text)

    heading_hints = []
    seen_hints = set()

    for line in lines:
        text = line["text"]
        line_type = _classify_pdf_line(line, body_size)

        if line_type == "TEXT" and not line.get("bold"):
            continue
        if not _is_clean_heading_hint(text):
            continue

        hint = _normalize_heading_hint(text)
        if hint in seen_hints:
            continue
        seen_hints.add(hint)
        heading_hints.append(hint)
        if len(heading_hints) >= 6:
            break

    if heading_hints and plain_text:
        hint_text = " / ".join(heading_hints)
        return f"[PDF heading hints p={page_index + 1}] {hint_text}\n{plain_text}", len(plain_text)

    return plain_text, len(plain_text)


def _matches_material(material: dict, material_ids: set[str], stored_names: set[str]) -> bool:
    # 요청 조건이 없으면 세션에 연결된 모든 자료를 대상으로 삼는다.
    stored_name = stored_name_from_material(material)
    if material_ids and str(material.get("id") or "") in material_ids:
        return True
    if stored_names and stored_name in stored_names:
        return True
    return not material_ids and not stored_names


async def get_session_pdf_materials(
    session_id: str,
    material_ids: list[str] | None = None,
    stored_names: list[str] | None = None,
) -> list[dict]:
    try:
        session_uuid = UUID(str(session_id))
    except ValueError as exc:
        raise MaterialTextError(f"session_id 형식이 올바르지 않습니다: {exc}", status_code=422) from exc

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT session_pdf
            FROM sessions
            WHERE session_id = $1
            """,
            session_uuid,
        )

    if row is None:
        raise MaterialTextError("세션 파일을 찾을 수 없습니다.", status_code=404)

    # 요청 값은 공백을 제거하고, 파일명은 basename만 남겨 이후 경로 검증과 기준을 맞춘다.
    requested_ids = {str(item).strip() for item in (material_ids or []) if str(item).strip()}
    requested_names = {
        Path(str(item).strip()).name
        for item in (stored_names or [])
        if str(item).strip()
    }

    materials = [
        material
        for material in _iter_materials(row["session_pdf"])
        if _matches_material(material, requested_ids, requested_names)
    ]

    # 세션 메타데이터에 아직 반영되지 않은 업로드 파일도 storedName으로 직접 요청할 수 있게 한다.
    if not materials and requested_names:
        materials = [
            {
                "id": None,
                "name": stored_name,
                "type": "application/pdf",
                "storedName": stored_name,
            }
            for stored_name in requested_names
        ]

    if not materials:
        raise MaterialTextError("선택한 강의자료를 세션에서 찾을 수 없습니다.", status_code=404)

    pdf_materials = []
    for material in materials:
        # 저장 파일명이 없는 메타데이터는 실제 파일을 열 수 없으므로 건너뛴다.
        stored_name = stored_name_from_material(material)
        if not stored_name:
            continue
        if is_pdf_material(material, stored_name):
            pdf_materials.append({**material, "storedName": stored_name})

    if not pdf_materials:
        raise MaterialTextError("선택한 강의자료 중 PDF 파일이 없습니다.", status_code=422)

    return pdf_materials


def extract_pdf_text(material: dict) -> str:
    stored_name = stored_name_from_material(material)
    if not stored_name:
        raise MaterialTextError("PDF 저장 파일명을 찾을 수 없습니다.", status_code=422)

    path = _material_path(stored_name)
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise MaterialTextError(f"PDF 파일을 읽지 못했습니다: {exc}", status_code=422) from exc

    page_texts = []
    extracted_chars = 0
    truncated = False
    page_limit = min(len(reader.pages), PDF_TEXT_MAX_PAGES)
    for page_index in range(page_limit):
        text, page_chars = _format_styled_page_text(reader.pages[page_index], page_index)
        extracted_chars += page_chars
        if text:
            page_texts.append(f"[PDF page {page_index + 1}]\n{text}")
        else:
            page_texts.append(
                f"[PDF page {page_index + 1}]\n(텍스트 추출 없음: 이미지/스캔/빈 페이지일 수 있음)"
            )

        # 긴 PDF가 프롬프트를 과도하게 키우지 않도록 누적 길이 기준으로 조기 종료한다.
        if len("\n".join(page_texts)) >= PDF_TEXT_MAX_CHARS:
            truncated = True
            break

    pdf_text = "\n".join(page_texts).strip()[:PDF_TEXT_MAX_CHARS]
    # 텍스트가 거의 없으면 이미지 기반 스캔 PDF일 가능성이 높아 생성 전에 중단한다.
    if extracted_chars < 20:
        raise MaterialTextError(
            "PDF에서 추출된 텍스트가 너무 짧습니다. 스캔 PDF라면 OCR이 필요합니다.",
            status_code=422,
        )

    title = material.get("name") or material.get("title") or stored_name
    page_notice = f"전체 페이지: {len(reader.pages)} / 추출 대상: {page_limit}페이지"
    if len(reader.pages) > page_limit:
        page_notice += f" (최대 {PDF_TEXT_MAX_PAGES}페이지 제한)"
    if truncated:
        page_notice += f" / 텍스트 {PDF_TEXT_MAX_CHARS:,}자 제한으로 이후 내용 일부 생략"
    return f"자료명: {title}\n{page_notice}\n{pdf_text}"


async def build_material_text(
    session_id: str,
    material_ids: list[str] | None = None,
    stored_names: list[str] | None = None,
) -> tuple[str, list[dict]]:
    # 여러 PDF의 텍스트를 하나의 입력 문자열로 묶어 퀴즈/요약이 함께 사용할 수 있게 한다.
    materials = await get_session_pdf_materials(session_id, material_ids, stored_names)
    texts = [extract_pdf_text(material) for material in materials]
    return "\n\n".join(texts).strip(), materials
