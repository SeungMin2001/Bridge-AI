from materials.material_text_service import (
    MaterialTextError,
    build_material_text,
    extract_pdf_text,
    get_session_pdf_materials,
    is_pdf_material,
    stored_name_from_material,
)


MaterialQuizError = MaterialTextError


async def build_material_quiz_text(
    session_id: str,
    material_ids: list[str] | None = None,
    stored_names: list[str] | None = None,
) -> tuple[str, list[dict]]:
    return await build_material_text(session_id, material_ids, stored_names)
