import logging
import os
from io import BytesIO

from materials.material_text_service import MaterialTextError, _material_path, stored_name_from_material


logger = logging.getLogger(__name__)

OCR_ENABLED = os.getenv("MATERIAL_OCR_ENABLED", "true").lower() == "true"
OCR_LANG = os.getenv("MATERIAL_OCR_LANG", "kor+eng")
OCR_RENDER_SCALE = float(os.getenv("MATERIAL_OCR_RENDER_SCALE", "2.0"))

_ocr_unavailable_reason: str | None = None


def ocr_pdf_page(material: dict, page_number: int) -> str:
    """선택된 PDF 한 페이지를 이미지로 렌더링한 뒤 OCR 텍스트를 추출한다."""
    if not OCR_ENABLED:
        return ""
    if page_number <= 0:
        return ""

    global _ocr_unavailable_reason
    if _ocr_unavailable_reason:
        return ""

    try:
        import fitz
        import pytesseract
        from PIL import Image
    except Exception as exc:
        _ocr_unavailable_reason = str(exc)
        logger.warning("[MATERIAL:OCR] OCR 의존성이 없어 스킵합니다: %s", exc)
        return ""

    stored_name = stored_name_from_material(material)
    if not stored_name:
        return ""

    try:
        path = _material_path(stored_name)
        document = fitz.open(str(path))
    except MaterialTextError:
        raise
    except Exception as exc:
        logger.warning("[MATERIAL:OCR] PDF 열기 실패: %s", exc)
        return ""

    try:
        page_index = page_number - 1
        if page_index < 0 or page_index >= document.page_count:
            return ""

        page = document.load_page(page_index)
        matrix = fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        text = pytesseract.image_to_string(image, lang=OCR_LANG)
        return _clean_ocr_text(text)
    except Exception as exc:
        # tesseract 실행 파일이 없거나 언어 데이터가 없을 때도 여기로 들어온다.
        logger.warning("[MATERIAL:OCR] p.%s OCR 실패: %s", page_number, exc)
        return ""
    finally:
        try:
            document.close()
        except Exception:
            pass


def _clean_ocr_text(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        cleaned = " ".join(line.split()).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines).strip()
