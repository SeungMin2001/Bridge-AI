"""AI 채팅용 근거 컨텍스트 생성 서비스.

RAG 검색 결과, 선택 PDF 보강 컨텍스트, 현재 워크스페이스의 저장 목록을 합쳐
LLM에게 전달할 최종 prompt와 프론트에 보여줄 citations를 만듭니다.
"""

import json
import logging
import os
import re
import time

from rag_search import search as rag_search


logger = logging.getLogger(__name__)

CHAT_EVIDENCE_TOP_K = max(3, int(os.getenv("CHAT_EVIDENCE_TOP_K", "5")))
CHAT_EVIDENCE_BROAD_TOP_K = max(CHAT_EVIDENCE_TOP_K, int(os.getenv("CHAT_EVIDENCE_BROAD_TOP_K", "6")))
CHAT_SELECTED_MATERIAL_CONTEXT_CHARS = int(os.getenv("CHAT_SELECTED_MATERIAL_CONTEXT_CHARS", "6000"))
CHAT_SELECTED_MATERIAL_CONTEXT_PER_FILE_CHARS = int(os.getenv("CHAT_SELECTED_MATERIAL_CONTEXT_PER_FILE_CHARS", "2000"))
CHAT_WORKSPACE_INVENTORY_MAX_ITEMS = int(os.getenv("CHAT_WORKSPACE_INVENTORY_MAX_ITEMS", "40"))
CHAT_MATERIAL_RAG_CHECK_TTL_SEC = int(os.getenv("CHAT_MATERIAL_RAG_CHECK_TTL_SEC", "300"))
_material_rag_checked_at: dict[str, float] = {}
_ALNUM_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")
_LOCATOR_SUBJECT_STOPWORDS = {
    "혹시",
    "무엇",
    "뭐",
    "어디",
    "어느",
    "위치",
    "찾",
    "찾아",
    "검색",
    "언급",
    "부분",
    "구간",
    "파일",
    "녹음",
    "녹음본",
    "전사",
    "자료",
    "내용",
    "말",
    "나오",
    "포함",
    "포함되",
    "있는",
    "있어",
    "관련",
    "해당",
    "대한",
    "대해",
    "대해서",
    "특정",
    "단어",
    "표현",
    "키워드",
    "근거",
    "링크",
    "보여",
    "어떻게",
    "왜",
    "설명",
    "설명했어",
}
_GROUNDED_CONTENT_QUESTION_TERMS = (
    "누구",
    "언제",
    "언제까지",
    "몇 시",
    "몇시",
    "마감",
    "마감일",
    "기한",
    "제출",
    "제출일",
    "제출해야",
    "까지",
    "무엇",
    "뭐야",
    "뭐여",
    "뭐냐",
    "뭐임",
    "뭐에요",
    "뭐예요",
    "뭔가",
    "뭔데",
    "무슨",
    "의미",
    "정의",
    "설명",
    "알려",
    "개념",
    "뜻",
    "이란",
    "란",
    "요약",
    "정리",
)
_EVIDENCE_EXPLANATION_TERMS = (
    "근거",
    "출처",
    "증거",
    "원문",
    "인용",
    "정확히",
    "뭐라고",
    "뭐라",
    "말했",
    "그런 말",
    "맥락",
    "의도",
    "해석",
    "팩트",
)

FAST_RAG_STYLE_PROMPT = (
    "[통합 답변 규칙]\n"
    "- 검색된 참고자료에 직접 나온 내용만 사용해 답하세요.\n"
    "- 각 [n]의 '핵심 참고문장'은 질문과 가장 직접적으로 관련된 근거이므로 답변에 우선 반영하세요.\n"
    "- 핵심 참고문장이 여러 문장으로 제공되면 첫 문장만 답하지 말고, 정의와 원인/역할/관계까지 2~4문장으로 함께 설명하세요.\n"
    "- '보조 문맥'은 핵심 참고문장을 해석하는 데만 사용하고, 질문과 관련 없는 문장은 버리세요.\n"
    "- 검색된 참고자료가 여러 개이면 한두 문장만 고르지 말고, 서로 다른 근거의 핵심 개념을 종합해 압축적으로 설명하세요.\n"
    "- 특정 키워드나 넓은 개념을 물으면 정의, 특징, 차이, 조건, 예시처럼 근거에 나온 관련 하위 개념을 빠뜨리지 말고 묶어서 설명하세요.\n"
    "- 약어 또는 용어의 의미를 묻는 질문은 풀네임만 말하지 말고, 참고자료에 나온 역할, 동작 방식, 관련 개념까지 함께 설명하세요.\n"
    "- 약어의 풀네임은 참고자료에 명시된 표현만 그대로 사용하고, 참고자료에 없는 풀네임이나 외부 지식은 절대 만들지 마세요.\n"
    "- 질문의 핵심 용어가 참고자료 문장에 직접 등장하면, 그 문장을 최우선 근거로 삼아 재구성하고 다른 분야의 정의를 섞지 마세요.\n"
    "- 핵심 용어의 정의를 답할 때는 참고문장의 의미를 바꾸지 말고, 원문 표현을 최대한 유지해 정확하게 풀어 쓰세요.\n"
    "- 질문/참고자료/시스템 지시문을 반복하지 말고 최종 답변만 작성하세요.\n"
    "- 참고자료에 불릿, 번호, 학습목표, 목차, 단계처럼 목록형 정보가 있으면 항목을 빠짐없이 불릿으로 나열하세요.\n"
    "- 목록형 정보는 원문 표현을 최대한 그대로 유지하고, 없는 항목을 새로 만들지 마세요.\n"
    "- 정의/설명 질문은 한 문장으로 끝내지 말고, 근거에 함께 나온 연결 개념을 포함해 간결하게 설명하세요.\n"
    "- 관련 없는 근거는 무시하고, 근거가 없으면 찾지 못했다고 답하세요.\n"
    "- 같은 근거를 여러 문장에서 사용하더라도 citation 번호는 답변 끝에 한 번만 붙이세요.\n"
    "- 서로 다른 근거를 함께 사용한 경우에만 답변 끝에 [1] [2]처럼 필요한 번호를 모아 붙이세요.\n"
    "- 별도 출처 목록은 만들지 마세요."
)

RAG_USER_PROMPT_TAIL = (
    "위 근거 문장만 사용해 질문에 답하세요. "
    "근거 문장에 나온 정의와 함께 붙어 있는 설명을 빠뜨리지 말고 2~5문장으로 압축하세요. "
    "규칙, 라벨, 참고자료 원문은 반복하지 말고 최종 답변만 작성하세요."
)


async def build_prompt_and_citations(
    question: str,
    session_id: str | None = None,
    source_filter: dict | None = None,
) -> tuple[str, list[dict]]:
    """질문, 현재 파일, 선택 자료 기준으로 LLM prompt와 citation 목록을 구성합니다."""
    has_selected_material = source_filter_has_material(source_filter)
    grounded_content_question = _is_grounded_content_question(question)
    factual_grounded_question = _is_factual_grounded_question(question)
    inventory_context = (
        await _build_workspace_inventory_context(session_id)
        if _should_include_inventory_context(question)
        else ""
    )
    # 넓은 개념 설명 질문은 상위 1-2개 근거만 쓰면 일부 개념이 빠지기 쉬워 근거 폭을 넓힙니다.
    evidence_top_k = _chat_evidence_top_k(question)
    rag_result = rag_search(
        question,
        top_k=evidence_top_k,
        session_id=session_id,
        source_filter=source_filter,
    )
    context = rag_result["context"]
    citations = rag_result["citations"]

    use_material_fallback = has_selected_material and _should_use_material_fallback(question, context)
    if use_material_fallback and not _has_material_citation(citations):
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
            citations = _merge_citations(citations, material_citations)[:evidence_top_k]

    reference_context = "\n\n".join(
        part for part in (
            inventory_context,
            f"[검색된 참고자료]\n{context}" if context else "",
        )
        if part
    )

    if reference_context:
        required_answer_points = _build_required_answer_points(citations)
        reference_intro = (
            "다음은 현재 워크스페이스 파일의 저장 목록과 강의 내용에서 검색된 참고자료입니다"
            if inventory_context
            else (
                "다음은 워크스페이스 전체 강의 내용에서 검색된 참고자료입니다"
                if not session_id
                else "다음은 현재 워크스페이스 파일의 강의 내용에서 검색된 참고자료입니다"
            )
        )
        inventory_instruction = (
            "사용자가 현재 파일에 저장된 자료/녹음본 목록, 개수, 이름을 물으면 "
            "[현재 워크스페이스 파일 저장 목록]을 우선 기준으로 답하세요. "
            "저장 목록은 강의 내용 근거가 아니라 파일명/메타데이터 목록입니다. "
            if inventory_context
            else ""
        )
        scope_instruction = (
            "사용자가 파일 위치나 어느 파일에 언급됐는지 물으면 [검색된 참고자료]의 파일명, 녹음본명, 시간 범위를 기준으로 답하세요. "
            if not session_id and _is_locator_question(question)
            else ""
        )
        scope_boundary_instruction = (
            "검색된 참고자료 밖의 내용은 추측하지 마세요. "
            if not session_id
            else "선택된 PDF/녹음본 밖의 내용은 추측하지 마세요. "
        )
        missing_locator_note = (
            "사용자가 특정 표현/주제/파일이 어디에 언급됐는지 묻는데 [검색된 참고자료]가 없으면, "
            "저장 목록이나 파일명만 보고 있다고 추측하지 말고 현재 선택된 파일에서 해당 언급을 찾지 못했다고 답하세요. "
            if _is_locator_question(question) and not context
            else ""
        )
        missing_selected_material_note = (
            "현재 선택된 PDF에 대해 질문과 직접 관련된 근거 페이지를 찾지 못했습니다. "
            "저장 목록은 파일명/개수/시간 같은 메타데이터만 담고 있으므로, PDF 내용 질문이면 근거를 찾지 못했다고 답하세요. "
            if has_selected_material and not context
            else ""
        )
        grounded_answer_instruction = (
            ""
        )
        prompt = (
            f"{reference_intro}:\n\n{reference_context}\n\n"
            f"{required_answer_points}"
            f"{inventory_instruction}"
            f"{scope_instruction}"
            f"{missing_selected_material_note}"
            f"{missing_locator_note}"
            f"{grounded_answer_instruction}"
            f"{scope_boundary_instruction}"
            f"{RAG_USER_PROMPT_TAIL}\n\n"
            f"질문: {question}"
        )
    elif has_selected_material and _is_current_scope_question(question):
        prompt = (
            "사용자가 PDF 자료를 선택했지만, 선택된 PDF 안에서 질문과 직접 관련된 근거 페이지를 찾지 못했습니다. "
            "외부 웹사이트나 일반 지식으로 대체하지 말고, 선택된 PDF에서 근거를 찾지 못했다고 짧게 답하세요.\n\n"
            f"질문: {question}"
        )
    elif _is_locator_question(question):
        prompt = (
            "워크스페이스 전체에서 질문과 직접 관련된 검색 근거를 찾지 못했습니다. "
            "파일명이나 일반 지식으로 추측하지 말고, 저장된 자료/녹음본에서 해당 언급을 찾지 못했다고 짧게 답하세요.\n\n"
            f"질문: {question}"
        )
    elif _is_grounded_content_question(question):
        prompt = (
            "저장된 자료/녹음본/PDF에서 질문과 직접 관련된 검색 근거를 찾지 못했습니다. "
            "그래도 사용자의 질문에는 일반 지식으로 짧고 명확하게 답하세요. "
            "단, 저장 자료 근거가 없다는 점을 첫 문장에 간단히 밝히고 citation 번호는 붙이지 마세요.\n\n"
            f"질문: {question}"
        )
    else:
        prompt = question

    return prompt, citations


def _build_required_answer_points(citations: list[dict], *, max_points: int = 6) -> str:
    """검색 근거 중 답변에 반드시 들어가야 할 문장을 별도 포인트로 제공합니다."""
    if not citations:
        return ""

    points: list[str] = []
    seen = set()
    for index, citation in enumerate(citations, 1):
        evidence = _clean_evidence_text((citation or {}).get("text", ""))
        if not evidence:
            continue
        for sentence in _split_prompt_sentences(evidence):
            compact = _trim_sentence(sentence, 220)
            key = compact.casefold()
            if not compact or key in seen:
                continue
            seen.add(key)
            points.append(f"- [{index}] {compact}")
            if len(points) >= max_points:
                break
        if len(points) >= max_points:
            break

    if not points:
        return ""
    return (
        "[답변 필수 반영 포인트]\n"
        + "\n".join(points)
        + "\n\n"
    )


def _split_prompt_sentences(text: str) -> list[str]:
    """citation 텍스트를 답변 필수 포인트용 문장으로 나눕니다."""
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return []
    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?。？！])\s+|\n+", value)
        if part.strip()
    ]
    return parts or [value]


def _chat_evidence_top_k(question: str) -> int:
    """질문이 넓은 설명/정리형이면 더 많은 근거를 LLM에 전달합니다."""
    text = str(question or "")
    focused_terms = (
        "뭐야",
        "뭐였",
        "뭐였지",
        "뭔지",
        "무슨 뜻",
        "뜻",
        "의미",
        "정의",
        "란",
        "이란",
    )
    if any(term in text for term in focused_terms):
        return min(CHAT_EVIDENCE_TOP_K, 3)

    broad_terms = (
        "정리",
        "요약",
        "전체",
        "전반",
        "다양",
        "관련",
        "차이",
        "비교",
        "구조",
        "과정",
        "원리",
        "특징",
        "종류",
        "개념들",
        "설명해",
        "알려줘",
    )
    return CHAT_EVIDENCE_BROAD_TOP_K if any(term in text for term in broad_terms) else CHAT_EVIDENCE_TOP_K


async def ensure_material_rag_for_chat(session_id: str | None, source_filter: dict | None):
    """현재 파일에 PDF 자료가 있으면 채팅 전에 해당 세션 자료의 RAG 인덱싱을 보장합니다."""
    if not session_id:
        return
    now = time.monotonic()
    last_checked_at = _material_rag_checked_at.get(str(session_id))
    if last_checked_at and (now - last_checked_at) < CHAT_MATERIAL_RAG_CHECK_TTL_SEC:
        return
    try:
        from materials.material_rag_service import ensure_session_materials_indexed

        await ensure_session_materials_indexed(session_id)
        _material_rag_checked_at[str(session_id)] = now
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


def _is_current_scope_question(question: str) -> bool:
    """현재 선택 파일/PDF/녹음본으로 범위를 한정한 질문인지 확인합니다."""
    text = str(question or "")
    current_scope_terms = (
        "현재 파일",
        "이 파일",
        "여기 파일",
        "현재 여기에",
        "선택된",
        "열려 있는",
        "열려있는",
        "지금 파일",
        "이 강의",
        "이 자료",
        "이 pdf",
        "이 PDF",
        "현재 자료",
        "현재 pdf",
        "현재 PDF",
        "여기 내용",
        "현재 내용",
    )
    return any(term in text for term in current_scope_terms)


def build_direct_locator_answer(question: str, citations: list[dict]) -> str | None:
    """언급 위치 찾기 질문은 citation 메타데이터만으로 짧고 안정적인 답변을 만듭니다."""
    if not _is_locator_question(question) or _is_evidence_explanation_question(question):
        return None

    transcript_citations = [
        cite for cite in citations or []
        if (cite or {}).get("source_type", "transcript") == "transcript"
    ]
    if not transcript_citations:
        return None

    first = transcript_citations[0]
    subject = _extract_locator_subject(question)
    source_label = _locator_source_label(first)
    time_range = _format_citation_time_range(first)

    if not source_label or not time_range:
        return None

    return f"{subject}는 {source_label}에서 언급됩니다.\n해당 구간은 {time_range}입니다."


def build_direct_factual_answer(question: str, citations: list[dict]) -> str | None:
    """짧은 사실형 질문은 top citation만으로 바로 답해 LLM 생성 지연을 줄입니다."""
    if not citations:
        return None
    if _is_locator_question(question) or _is_evidence_explanation_question(question):
        return None

    text = str(question or "").strip()
    if not (
        _is_factual_grounded_question(text)
        or _is_assignment_question(text)
        or _is_identity_question(text)
    ):
        return None

    first = citations[0] or {}
    evidence = _clean_evidence_text(first.get("text", ""))
    if not evidence:
        return None

    sentence = _select_evidence_sentence(text, evidence)
    citation_no = "[1]"

    deadline = _extract_deadline_phrase(sentence or evidence)
    if deadline and _is_assignment_question(text):
        subject = "수학 과제" if "수학" in f"{text} {evidence}" and "과제" in f"{text} {evidence}" else "과제"
        if "제출" in f"{text} {evidence}":
            return f"{subject}는 {deadline} 제출하면 됩니다. {citation_no}"
        return f"{subject}는 {deadline} 하면 됩니다. {citation_no}"

    if deadline and any(term in text for term in ("언제", "언제까지", "몇 시", "몇시", "마감", "기한")):
        return f"근거에 따르면 기한은 {deadline}입니다. {citation_no}"

    if _is_identity_question(text):
        short_sentence = _trim_sentence(sentence or evidence, 120)
        return f"{short_sentence} {citation_no}"

    if sentence:
        return f"{_trim_sentence(sentence, 140)} {citation_no}"

    return None


def build_direct_smalltalk_answer(question: str) -> str | None:
    """인사/도움말은 RAG와 LLM을 거치지 않고 짧고 안정적으로 답합니다."""
    if not is_smalltalk_question(question):
        return None

    text = " ".join(str(question or "").strip().split())
    if not text:
        return "질문을 입력해 주세요."

    compact = re.sub(r"[\s!?.。！？~]+", "", text).lower()
    greetings = {"안녕", "안녕하세요", "하이", "hello", "hi", "ㅎㅇ"}
    thanks = {"고마워", "고맙습니다", "감사", "감사합니다", "땡큐", "thanks", "thankyou"}

    if compact in greetings or (len(compact) <= 12 and compact.startswith(("안녕", "하이"))):
        return (
            "안녕하세요. 저는 강의 녹음, 전사, PDF 자료를 바탕으로 "
            "요약, 퀴즈 생성, 일정 추출, 근거 기반 질문 답변을 도와드릴게요."
        )

    if compact in thanks:
        return "천만에요. 필요한 강의 내용이나 자료에 대해 질문해 주세요."

    help_patterns = (
        "뭐 할 수 있어",
        "무엇을 할 수 있어",
        "어떤 기능",
        "사용법",
        "도움말",
        "어떻게 쓰",
    )
    if any(pattern in text for pattern in help_patterns):
        return (
            "저는 선택한 강의 녹음과 PDF 자료를 기준으로 답변할 수 있습니다. "
            "전사 요약, 퀴즈 생성, 일정 추출, 자료 기반 질의응답을 지원하고, "
            "답변에는 가능한 경우 근거 링크를 함께 보여드립니다."
        )

    return None


def is_smalltalk_question(question: str) -> bool:
    """RAG 근거 없이도 답할 수 있는 인사/도움말성 질문인지 확인합니다."""
    text = " ".join(str(question or "").strip().split())
    if not text:
        return True

    compact = re.sub(r"[\s!?.。！？~]+", "", text).lower()
    greetings = {"안녕", "안녕하세요", "하이", "hello", "hi", "ㅎㅇ"}
    thanks = {"고마워", "고맙습니다", "감사", "감사합니다", "땡큐", "thanks", "thankyou"}
    if compact in greetings or compact in thanks:
        return True
    if len(compact) <= 12 and compact.startswith(("안녕", "하이")):
        return True

    help_patterns = (
        "뭐 할 수 있어",
        "무엇을 할 수 있어",
        "어떤 기능",
        "사용법",
        "도움말",
        "어떻게 쓰",
    )
    return any(pattern in text for pattern in help_patterns)


def build_direct_no_evidence_answer(question: str, citations: list[dict]) -> str | None:
    """근거 기반 질문인데 citation이 없으면 LLM 호출 없이 '찾지 못함'으로 답합니다."""
    if citations:
        return None
    if not (
        _is_locator_question(question)
        or _is_grounded_content_question(question)
        or _is_evidence_explanation_question(question)
    ):
        return None
    return "저장된 자료/녹음본에서 질문과 직접 관련된 근거를 찾지 못했습니다."


def _is_assignment_question(question: str) -> bool:
    text = str(question or "")
    return "과제" in text and any(term in text for term in ("언제", "언제까지", "어떻게", "해야", "제출", "마감", "기한"))


def _is_identity_question(question: str) -> bool:
    text = str(question or "")
    return any(term in text for term in (
        "누구",
        "어떤 사람",
        "무슨 사람",
        "뭐 하는 사람",
        "누군데",
        "뭐였",
        "뭐였지",
        "뭐야",
        "뭔지",
        "무슨 뜻",
        "의미",
        "정의",
    ))


def _clean_evidence_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"^선택된\s+녹음본\s+전체\s+전사\s*\(출처:\s*.*?\)\s*", "", text).strip()
    text = re.sub(r"^\[[0-9]+\]\s*", "", text).strip()
    return text


def _select_evidence_sentence(question: str, evidence: str) -> str:
    subjects = _extract_lookup_subjects(question)
    if "과제" in question and "과제" not in subjects:
        subjects.append("과제")

    parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?。！？])\s+|\n+", evidence)
        if part.strip()
    ]
    if not parts:
        parts = [evidence]

    if _is_assignment_question(question) or _is_factual_grounded_question(question):
        for part in parts:
            if any(term in part for term in ("까지", "전까지", "제출", "마감", "기한")):
                return part

    for subject in subjects:
        for part in parts:
            if subject and subject in part:
                return part

    for part in parts:
        if any(term in part for term in ("까지", "전까지", "제출", "마감", "기한")):
            return part

    return parts[0]


def _extract_deadline_phrase(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return None

    patterns = (
        r"((?:오늘|내일|모레|이번\s*주|다음\s*주|이번주|다음주|[월화수목금토일]요일|[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)[^.!?。！？]{0,40}?(?:전까지|까지))",
        r"((?:오전|오후)?\s*(?:[0-9]{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|열한|열두)\s*시[^.!?。！？]{0,20}?(?:전까지|까지))",
    )
    for pattern in patterns:
        match = re.search(pattern, compact)
        if match:
            return _trim_sentence(_normalize_deadline_phrase(match.group(1)), 60)
    return None


def _normalize_deadline_phrase(value: str) -> str:
    phrase = re.sub(r"\s+", " ", str(value or "")).strip()
    date_marker = r"(?:오늘|내일|모레|이번\s*주|다음\s*주|이번주|다음주|[월화수목금토일]요일|[0-9]{1,2}\s*월\s*[0-9]{1,2}\s*일)"
    date_matches = list(re.finditer(date_marker, phrase))
    if len(date_matches) > 1:
        phrase = phrase[date_matches[-1].start():].strip()
    phrase = re.sub(r"^(?:과제는|과제\s*는|수학\s*과제는|수학\s*과제\s*는)\s*", "", phrase).strip()
    return phrase


def _trim_sentence(text: str, limit: int) -> str:
    sentence = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(sentence) <= limit:
        return sentence
    return f"{sentence[:limit].rstrip()}..."


def _should_include_inventory_context(question: str) -> bool:
    """파일 목록/개수/저장 상태를 묻는 질문일 때만 워크스페이스 저장 목록을 prompt에 넣습니다."""
    text = str(question or "").strip()
    if not text:
        return False

    inventory_terms = (
        "파일",
        "자료",
        "녹음",
        "녹음본",
        "소스",
        "저장",
        "목록",
        "리스트",
        "몇 개",
        "몇개",
        "개수",
        "뭐 있어",
        "무엇이 있어",
        "올려",
        "업로드",
    )
    return any(term in text for term in inventory_terms) and not _is_locator_question(text)


def _is_factual_grounded_question(question: str) -> bool:
    """마감/일정/인물처럼 짧은 사실 답변이 필요한 근거 기반 질문인지 판별합니다."""
    text = str(question or "").strip()
    if not text:
        return False

    factual_terms = (
        "누구",
        "언제",
        "언제까지",
        "몇 시",
        "몇시",
        "마감",
        "마감일",
        "기한",
        "제출",
        "제출일",
    )
    return any(term in text for term in factual_terms)


def _is_elliptic_grounded_question(question: str) -> bool:
    """'선형리스트?', '연속공간은?'처럼 짧게 던진 개념 질문을 근거 기반 질문으로 봅니다."""
    text = str(question or "").strip()
    if not text:
        return False
    if any(term in text for term in _GROUNDED_CONTENT_QUESTION_TERMS):
        return False
    if _is_locator_question(text) or _is_evidence_explanation_question(text):
        return False

    cleaned = re.sub(r"[?!?.。！？\s]+$", "", text).strip()
    if not 2 <= len(cleaned) <= 30:
        return False
    if len(_extract_lookup_subjects(cleaned)) != 1 and not _ALNUM_TERM_RE.search(cleaned):
        return False

    return text.endswith(("?", "？")) or cleaned == text


def _is_locator_question(question: str) -> bool:
    """특정 표현이 어느 파일/녹음/자료에 있는지 묻는 질문인지 판별합니다."""
    text = str(question or "").strip()
    if not text:
        return False

    locator_terms = ("어디", "어느", "몇 분", "몇초", "몇 초", "위치", "찾", "검색", "근거", "링크", "보여")
    mention_terms = ("언급", "나오", "포함", "있어", "있는", "말했", "다룬", "등장")
    strong_mention_terms = ("언급", "나오", "포함", "말했", "다룬", "등장")
    target_terms = ("파일", "녹음", "전사", "자료", "내용", "부분", "구간", "문장", "대목")
    has_subject_hint = bool(_extract_lookup_subjects(text) or _ALNUM_TERM_RE.search(text))
    has_locator = any(term in text for term in locator_terms)
    has_mention = any(term in text for term in mention_terms)
    has_strong_mention = any(term in text for term in strong_mention_terms)
    has_target = any(term in text for term in target_terms)

    return (
        has_subject_hint
        and has_mention
        and (has_locator or has_target or has_strong_mention or "에 대한" in text or "에대한" in text or "라고" in text)
    )


def _is_grounded_content_question(question: str) -> bool:
    """저장된 강의/PDF/전사 내용의 의미나 정체를 묻는 질문인지 판별합니다."""
    text = str(question or "").strip()
    if not text:
        return False
    if _is_elliptic_grounded_question(text):
        return True
    if not any(term in text for term in _GROUNDED_CONTENT_QUESTION_TERMS):
        return False
    return bool(_extract_lookup_subjects(text) or _ALNUM_TERM_RE.search(text))


def _is_evidence_explanation_question(question: str) -> bool:
    """출처, 원문, 맥락, 해석을 요구하는 질문인지 판별합니다."""
    text = str(question or "").strip()
    if not text:
        return False
    if any(term in text for term in _EVIDENCE_EXPLANATION_TERMS):
        return True
    return "왜" in text and any(term in text for term in ("그렇게", "그런", "말", "해석", "판단"))


def _should_use_material_fallback(question: str, transcript_context: str) -> bool:
    """PDF 내용 질문일 때만 선택 PDF fallback을 붙입니다."""
    text = str(question or "").lower()
    material_terms = (
        "pdf",
        "페이지",
        "page",
        "p.",
        "자료",
        "강의자료",
        "슬라이드",
        "파일",
        "문서",
        "학습목표",
        "학습 목표",
        "목차",
        "항목",
    )
    has_material_signal = any(term in text for term in material_terms)
    if has_material_signal:
        return True

    # 전사 근거가 이미 있으면 PDF 전체/페이지 fallback을 추가하지 않는다.
    if transcript_context:
        return False

    return not _is_grounded_content_question(question)


def _extract_locator_subject(question: str) -> str:
    """질문에서 사용자가 찾으려는 표현명을 추출합니다."""
    text = str(question or "").strip()
    lookup_subjects = _extract_lookup_subjects(text)
    if lookup_subjects:
        return lookup_subjects[0]

    alnum_terms = _ALNUM_TERM_RE.findall(text)
    if alnum_terms:
        return alnum_terms[0]

    quoted_match = re.search(r"['\"“”‘’](.+?)['\"“”‘’]", text)
    if quoted_match:
        return quoted_match.group(1).strip()

    marker_match = re.search(r"(.+?)(?:라고|이라는|라는|에 대한|에대한|이 어디|가 어디|은 어디|는 어디)", text)
    if marker_match:
        candidate = re.sub(r"^(혹시|그럼|그러면|저기|혹은)\s*", "", marker_match.group(1)).strip()
        if candidate:
            return candidate

    return "질문하신 내용"


def _clean_lookup_subject(value: str) -> str:
    """질문에서 추출한 표현의 앞뒤 조사와 불필요한 말을 정리합니다."""
    subject = re.sub(r"^(혹시|그럼|그러면|저기|혹은|그|이|저)\s*", "", str(value or "")).strip()
    subject = re.sub(r"[\s,.:;!?？]+$", "", subject).strip()
    subject = re.sub(r"(은|는|이|가|을|를|에|에서|으로|로|도|만)$", "", subject).strip()
    return subject


def _append_lookup_subject(subjects: list[str], value: str):
    subject = _clean_lookup_subject(value)
    if len(subject) < 2:
        return
    if subject.lower() in _LOCATOR_SUBJECT_STOPWORDS or subject in _LOCATOR_SUBJECT_STOPWORDS:
        return
    if subject not in subjects:
        subjects.append(subject)


def _extract_lookup_subjects(question: str) -> list[str]:
    """언급/근거 찾기 질문에서 실제 찾으려는 표현을 추출합니다."""
    text = " ".join(str(question or "").split())
    subjects: list[str] = []

    for match in re.finditer(r"['\"“”‘’](.+?)['\"“”‘’]", text):
        _append_lookup_subject(subjects, match.group(1))

    marker_patterns = (
        r"([A-Za-z][A-Za-z0-9_+#.-]*|[가-힣A-Za-z0-9_+#.-]{2,30})\s*(?:에\s*대한|에대한|에\s*대해|에대해|에\s*대해서|에대해서)",
        r"([A-Za-z][A-Za-z0-9_+#.-]*|[가-힣A-Za-z0-9_+#.-]{2,30})\s*(?:라고|이라는|라는)",
        r"(?:단어|표현|키워드)\s*[\"'“”‘’]?\s*([A-Za-z][A-Za-z0-9_+#.-]*|[가-힣A-Za-z0-9_+#.-]{2,30})",
    )
    for pattern in marker_patterns:
        for match in re.finditer(pattern, text):
            _append_lookup_subject(subjects, match.group(1))

    if subjects:
        return subjects

    for match in re.finditer(r"[A-Za-z][A-Za-z0-9_+#.-]*|[가-힣A-Za-z0-9_+#.-]{2,30}", text):
        _append_lookup_subject(subjects, match.group(0))

    return subjects


def _format_seconds(seconds) -> str:
    """초 단위 숫자를 m:ss 문자열로 변환합니다."""
    try:
        total_seconds = max(0, int(float(seconds or 0)))
    except (TypeError, ValueError):
        total_seconds = 0
    return f"{total_seconds // 60}:{total_seconds % 60:02d}"


def _format_citation_time_range(citation: dict) -> str:
    """전사 citation의 start/end 값을 화면 답변용 시간 범위로 변환합니다."""
    if citation.get("start_time") is None or citation.get("end_time") is None:
        return ""
    return f"{_format_seconds(citation.get('start_time'))}~{_format_seconds(citation.get('end_time'))}"


def _locator_source_label(citation: dict) -> str:
    """언급 위치 답변에 사용할 녹음본/파일명을 고릅니다."""
    label = _compact_chat_text(
        citation.get("recording_title")
        or citation.get("session_title")
        or citation.get("file_title")
        or citation.get("citation"),
        "녹음본",
    )
    if "녹음" not in label:
        label = f"{label} 녹음본"
    return label


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
