"""
퀴즈 서비스 (Quiz Service)

역할:
  1. 세션 전사문을 기반으로 LLM에게 퀴즈 생성 요청
  2. LLM 응답을 지정된 JSON 템플릿으로 파싱
  3. 사용자 답안 채점

quiz_data JSONB 구조:
[
  {
    "question_index": 1,
    "type": "MULTIPLE_CHOICE" | "OX" | "SHORT_ANSWER",
    "question": "질문 텍스트",
    "options": ["1. ...", "2. ...", ...],
    "correct_answer": "정답",
    "user_answer": null | "사용자 답",
    "is_correct": null | true | false,
    "explanation": "해설"
  }
]

LLM이 아직 연결되지 않은 경우:
  - QUIZ_MOCK_MODE=true로 실행하면 generate_quiz()는 목업 데이터를 반환
  - 기본값은 MOCK_MODE=False이며 실제 LLM 서버를 호출
"""
import json
import re
import uuid
import logging
import httpx
import os

try:
    from kiwipiepy import Kiwi
except ImportError:
    Kiwi = None

logger = logging.getLogger(__name__)
DEMO_PIPELINE_LOG = True
_kiwi = Kiwi() if Kiwi is not None else None

# ── 설정 ──
MOCK_MODE = os.getenv("QUIZ_MOCK_MODE", "false").lower() == "true"

# vLLM OpenAI 호환 API
DEFAULT_LLM_URL = "http://localhost:8001"
DEFAULT_LLM_MODEL = "bridgeprag-qwen25-3b-kv64"

LLM_URL = os.getenv("LLM_URL", DEFAULT_LLM_URL)
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
LLM_API_KEY = os.getenv("LLM_API_KEY", "test-key")
LLM_MAX_TOKENS = int(os.getenv("QUIZ_MAX_TOKENS", "4096"))
QUIZ_CONTEXT_CHARS = int(os.getenv("QUIZ_CONTEXT_CHARS", "2800"))
QUIZ_MIN_OUTPUT_TOKENS = int(os.getenv("QUIZ_MIN_OUTPUT_TOKENS", "900"))
QUIZ_TOKENS_PER_QUESTION = int(os.getenv("QUIZ_TOKENS_PER_QUESTION", "320"))
_MORPHEME_LOG_TAGS = {"NNG", "NNP", "VV", "VA", "SL"}


def _demo_log(message: str) -> None:
    if DEMO_PIPELINE_LOG:
        print(f"[DEMO:QUIZ] {message}", flush=True)


def _preview(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    return compact if len(compact) <= limit else f"{compact[:limit - 3]}..."


def _tokenize_for_demo(sentence: str) -> list[str]:
    """시연 로그용 형태소 토큰을 추출합니다."""
    if _kiwi is None:
        return re.findall(r"[가-힣A-Za-z0-9_+#./-]{2,}", sentence)[:12]
    return [
        token.form
        for token in _kiwi.tokenize(sentence)
        if token.tag in _MORPHEME_LOG_TAGS and len(token.form) >= 2
    ][:12]


def _log_sentence_morphemes(sentences: list[str], *, label: str) -> None:
    """문장 분리 후 형태소 분석 산출물을 로그로 남깁니다."""
    if not sentences:
        _demo_log(f"{label} 형태소 분석 생략: 문장 없음")
        return
    _demo_log(f"{label} 형태소 분석 시작: sentences={len(sentences)}, sample_sentences={min(len(sentences), 5)}")
    for index, sentence in enumerate(sentences[:5], start=1):
        preprocessed = re.sub(r"\s+", " ", sentence).strip()
        tokens = _tokenize_for_demo(preprocessed)
        _demo_log(f"   전처리문장#{index}: '{_preview(preprocessed, 100)}'")
        _demo_log(f"   형태소분리#{index}: tokens={tokens}")
    _demo_log(f"{label} 형태소 분석 종료")

#  LLM 프롬프트 템플릿
QUIZ_SYSTEM_PROMPT = """당신은 대학 강의 내용을 기반으로 학습 퀴즈를 만드는 AI 교수입니다.
반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 텍스트는 절대 포함하지 마세요.
문제는 학습자가 실제로 복습할 수 있는 핵심 사실, 개념, 관계, 조건만 물어야 합니다.
모든 문항과 보기는 짧게 작성하고, JSON 문자열을 반드시 완성하세요."""

QUIZ_USER_PROMPT_TEMPLATE = """아래는 강의 전사문입니다:

{transcript_text}

위 강의 내용을 바탕으로 {num_questions}개의 퀴즈를 생성하세요.

문제 유형 배분:
- MULTIPLE_CHOICE (객관식, 3~4지선다): {mc_count}개
- OX (O/X 퀴즈): {ox_count}개
- SHORT_ANSWER (단답형): {sa_count}개

생성 규칙:
- 요청한 문항 수와 유형별 개수를 정확히 지키세요.
- 제공된 강의 내용에 명시된 사실만 사용하고, 외부 지식을 추가하지 마세요.
- 같은 질문이나 거의 같은 질문을 반복하지 마세요.
- 자료의 단순 제목/라벨/단어만 보고 "'제목'의 의미는 무엇입니까?" 같은 빈약한 문제를 만들지 마세요.
- 객관식 질문은 "강의 내용과 가장 일치하는 설명"처럼 전체 자료를 묻지 말고, 역전파/손실 함수처럼 특정 개념을 직접 물으세요.
- 객관식 오답도 반드시 선택 소스의 개념을 바탕으로 만든 짧은 지식 문장이어야 합니다.
- "자료에서 확인할 수 없는 별도의 개념", "자료의 설명과 반대" 같은 메타 문장을 보기로 쓰지 마세요.
- 각 question은 60자 이내, 각 option은 35자 이내, explanation은 50자 이내 한 문장으로 작성하세요.
- 객관식 보기는 긴 문장을 쓰지 말고 핵심어/짧은 구로 작성하세요.
- JSON 문자열 안에 실제 줄바꿈을 넣지 말고, 모든 따옴표와 대괄호를 반드시 닫으세요.
- MULTIPLE_CHOICE는 options를 반드시 3~4개 작성하고 correct_answer는 options 중 정확히 하나와 완전히 같아야 합니다.
- OX는 반드시 참/거짓을 판단할 수 있는 평서문으로 작성하세요. "무엇입니까?", "어디입니까?", "왜입니까?" 같은 의문문은 OX로 만들면 안 됩니다.
- OX의 options는 반드시 ["O", "X"]이고 correct_answer는 반드시 "O" 또는 "X"입니다.
- SHORT_ANSWER는 options를 []로 두고, correct_answer는 짧은 핵심 답안으로 작성하세요.

반드시 아래 JSON 배열 형식으로만 응답하세요:
[
  {{
    "question_index": 1,
    "type": "MULTIPLE_CHOICE",
    "question": "질문 텍스트",
    "options": ["1. 보기1", "2. 보기2", "3. 보기3", "4. 보기4"],
    "correct_answer": "1. 보기1",
    "explanation": "짧은 해설"
  }},
  {{
    "question_index": 2,
    "type": "OX",
    "question": "O/X 평서문 텍스트",
    "options": ["O", "X"],
    "correct_answer": "X",
    "explanation": "해설 텍스트"
  }},
  {{
    "question_index": 3,
    "type": "SHORT_ANSWER",
    "question": "단답형 질문 텍스트",
    "options": [],
    "correct_answer": "정답 텍스트",
    "explanation": "해설 텍스트"
  }}
]"""

QUIZ_REPAIR_PROMPT_TEMPLATE = """아래 퀴즈 JSON은 품질 검증에 실패했습니다.
문제점을 고쳐서 유효한 JSON 배열만 다시 응답하세요.

[원본 강의 내용]
{transcript_text}

[필수 문항 수]
- MULTIPLE_CHOICE: {mc_count}개
- OX: {ox_count}개
- SHORT_ANSWER: {sa_count}개

[검증 실패 사유]
{issues}

[수정할 JSON]
{quiz_json}

수정 규칙:
- OX는 반드시 참/거짓 평서문이어야 하며 options는 ["O", "X"]입니다.
- 객관식은 options 3~4개와 그중 하나와 완전히 같은 correct_answer가 필요합니다.
- 단답형은 options를 []로 두세요.
- question은 60자 이내, option은 35자 이내, explanation은 50자 이내로 짧게 쓰세요.
- JSON 외 텍스트는 쓰지 마세요."""

QUIZ_SOURCE_SUMMARY_SYSTEM_PROMPT = """당신은 강의자료와 전사문을 퀴즈 생성에 적합한 학습 요약본으로 정리하는 AI입니다.
반드시 JSON 객체만 응답하세요. JSON 외 텍스트는 포함하지 마세요.
목차, 페이지 번호, 깨진 문자, 장식 불릿은 제거하고 실제 학습 내용만 남기세요."""

QUIZ_SOURCE_SUMMARY_PROMPT_TEMPLATE = """아래는 사용자가 퀴즈 생성을 위해 선택한 원본 소스입니다.

[선택 소스 원문]
{source_text}

위 원문을 바탕으로 퀴즈 생성에 사용할 한국어 요약본을 작성하세요.
- 원문에 실제로 포함된 핵심 개념, 정의, 특징, 차이, 조건, 예시만 정리하세요.
- 단순 목차, 페이지 번호, 슬라이드 제목만으로 된 항목은 제외하세요.
- 같은 내용을 반복하지 마세요.
- 객관식 보기로 사용할 수 있도록 각 항목은 명확한 한 문장으로 작성하세요.
- 새로운 사실을 추가하지 마세요.

반드시 아래 JSON 객체 형식으로만 응답하세요:
{{
  "summary_text": "퀴즈 생성에 사용할 핵심 요약본"
}}"""

_OX_INTERROGATIVE_RE = re.compile(r"(무엇|어떤|어디|왜|어떻게|입니까|인가요|일까요|까요|[?？])")
_SENTENCE_END_FINDER_RE = re.compile(r".+?(?:다\.|요\.|[.!?。？！])(?:\s+|$)")
_SOURCE_BULLET_RE = re.compile(r"[❖▪•●○◆◇■□▶▷▸▹]")
_BAD_QUIZ_PHRASES = (
    "작성하세요",
    "설명해주세요",
    "요약하여",
    "문구가 나왔을 때",
    "자료에서 '",
    "JSON",
    "options",
    "correct_answer",
    "PDF page",
)
_BAD_OPTION_PHRASES = (
    "자료에서 확인할 수 없는",
    "자료의 설명과 반대",
    "선택한 소스의 핵심 설명",
    "핵심 설명과 일치하지 않는다",
    "관련이 없습니다",
    "관련이 있습니다",
    "관련이 없다",
    "관련이 있다",
    "별도의 개념이다",
    "외부 자료에만",
    "위 내용과 무관",
    "주차에서는",
    "강의에서는",
    "수업에서는",
    "하나의 고정값으로만",
    "모든 상황에서 같은 방식",
    "결과에 영향을 주지 않는 부가 정보",
    "원인과 결과의 관계를 고려하지 않는다",
    "측정이나 비교의 기준이 되지 않는다",
    "비례하지",
    "아닙니다",
    "하지 않습니다",
    "수 없습니다",
    "...",
    "…",
)
_GENERIC_MC_QUESTION_PHRASES = (
    "다음 중 선택한 자료의 설명과 일치",
    "자료에서 설명한 핵심 내용",
    "강의 내용과 가장 일치",
    "다음 중 자료의 핵심 개념",
    "선택한 소스의 내용으로 옳은 것",
    "자료의 핵심 내용으로 알맞은 것",
    "주차에서 정리한",
)
_GENERIC_SHORT_QUESTION_PHRASES = (
    "자료에서 설명한",
    "핵심 내용을 쓰세요",
    "강의 내용의 핵심",
    "선택 소스의 핵심",
    "요약하여 쓰세요",
    "설명하세요",
)
_EXPLANATORY_MARKERS = (
    "한다",
    "하다",
    "합니다",
    "된다",
    "됩니다",
    "이다",
    "입니다",
    "있다",
    "있습니다",
    "없다",
    "없습니다",
    "의미",
    "정의",
    "특징",
    "차이",
    "역할",
    "방식",
    "조건",
    "과정",
    "저장",
    "구현",
    "사용",
    "공유",
    "포함",
    "연결",
    "처리",
    "에너지",
    "물리량",
    "비례",
    "관계",
    "결과",
    "정답",
    "가중치",
    "속력",
)


class QuizParseError(ValueError):
    """LLM 응답을 JSON으로 파싱하지 못했을 때 원문을 함께 전달합니다."""

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


def _calculate_type_distribution(num_questions: int) -> tuple[int, int, int]:
    """문제 유형 배분 계산 (객관식 > OX > 단답형 비율)"""
    if num_questions <= 2:
        return num_questions, 0, 0
    mc_count = max(1, num_questions * 50 // 100)
    ox_count = max(1, num_questions * 30 // 100)
    sa_count = num_questions - mc_count - ox_count
    if sa_count < 0:
        ox_count += sa_count
        sa_count = 0
    return mc_count, ox_count, sa_count


QUIZ_TYPE_KEYS = ("MULTIPLE_CHOICE", "OX", "SHORT_ANSWER")


def _normalize_type_counts(
    num_questions: int = 5,
    type_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    if type_counts is None:
        mc, ox, sa = _calculate_type_distribution(num_questions)
        return {
            "MULTIPLE_CHOICE": mc,
            "OX": ox,
            "SHORT_ANSWER": sa,
        }

    counts = {}
    for key in QUIZ_TYPE_KEYS:
        raw_value = type_counts.get(key, 0)
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} 문항 수가 올바르지 않습니다.")
        if value < 0:
            raise ValueError(f"{key} 문항 수는 0 이상이어야 합니다.")
        counts[key] = value

    total = sum(counts.values())
    if total < 1:
        raise ValueError("퀴즈 문항 수는 1개 이상이어야 합니다.")
    if total > 20:
        raise ValueError("퀴즈 문항 수는 최대 20개까지 생성할 수 있습니다.")
    return counts


def count_quiz_types(quiz_data: list[dict]) -> dict[str, int]:
    counts = {key: 0 for key in QUIZ_TYPE_KEYS}
    for question in quiz_data:
        question_type = question.get("type")
        if question_type in counts:
            counts[question_type] += 1
    return counts


def _build_quiz_prompt(
    transcript_text: str,
    num_questions: int = 5,
    type_counts: dict[str, int] | None = None,
) -> list[dict]:
    """LLM에 보낼 messages 배열 구성"""
    counts = _normalize_type_counts(num_questions, type_counts)
    total_questions = sum(counts.values())
    user_prompt = QUIZ_USER_PROMPT_TEMPLATE.format(
        transcript_text=transcript_text[:QUIZ_CONTEXT_CHARS],  # 로컬 LLM JSON 완성률을 위해 입력 길이 제한
        num_questions=total_questions,
        mc_count=counts["MULTIPLE_CHOICE"],
        ox_count=counts["OX"],
        sa_count=counts["SHORT_ANSWER"],
    )
    return [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _remove_thinking_blocks(raw_text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    lower_text = text.lower()
    if '<think>' in lower_text:
        text = text[:lower_text.index('<think>')]
    return text


def _strip_json_code_fences(text: str) -> str:
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    return re.sub(r'```\s*', '', text).strip()


def _extract_balanced_json_candidates(text: str) -> list[str]:
    candidates = []
    stack = []
    start = None
    in_string = False
    escaped = False
    pairs = {"]": "[", "}": "{"}

    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char in "[{":
            if not stack:
                start = index
            stack.append(char)
            continue

        if char in "]}":
            if not stack or stack[-1] != pairs[char]:
                stack = []
                start = None
                continue
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:index + 1])
                start = None

    return candidates


def _normalize_json_candidate(candidate: str) -> str:
    candidate = _strip_json_code_fences(candidate)
    return re.sub(r",\s*([}\]])", r"\1", candidate)


def _coerce_quiz_questions(parsed: object) -> list[dict]:
    if isinstance(parsed, list):
        questions = parsed
    elif isinstance(parsed, dict):
        if any(key in parsed for key in ("question", "type", "correct_answer", "answer")):
            questions = [parsed]
        else:
            questions = None
        for key in ("quiz_data", "questions", "quizzes", "items", "data", "result"):
            if questions is not None:
                break
            value = parsed.get(key)
            if isinstance(value, list):
                questions = value
                break
        if questions is None:
            raise ValueError("LLM 응답이 퀴즈 JSON 배열을 포함하지 않습니다")
    else:
        raise ValueError("LLM 응답이 JSON 배열이 아닙니다")

    normalized_questions = []
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            raise ValueError("LLM 응답의 문항 형식이 올바르지 않습니다")
        next_question = dict(question)
        next_question.setdefault("question_index", index)
        next_question.setdefault("type", "MULTIPLE_CHOICE")
        next_question["type"] = _normalize_quiz_type(next_question.get("type"))
        next_question.setdefault("question", "")
        if "correct_answer" not in next_question and "answer" in next_question:
            next_question["correct_answer"] = next_question.get("answer")
        next_question.setdefault("correct_answer", "")
        next_question.setdefault("user_answer", None)
        next_question.setdefault("is_correct", None)
        next_question.setdefault("explanation", "")

        options = next_question.get("options")
        if isinstance(options, str):
            options = [options]
        elif not isinstance(options, list):
            options = []
        if next_question["type"] == "OX":
            options = ["O", "X"]
            if str(next_question.get("correct_answer") or "").upper() in {"O", "X"}:
                next_question["correct_answer"] = str(next_question["correct_answer"]).upper()
        if next_question["type"] == "SHORT_ANSWER":
            options = []
        next_question["options"] = options
        next_question = _coerce_correct_answer(next_question)
        normalized_questions.append(next_question)

    return normalized_questions


def _normalize_quiz_type(value) -> str:
    """LLM이 흔히 섞어 쓰는 문항 타입 표기를 내부 타입으로 정규화합니다."""
    normalized = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "MC": "MULTIPLE_CHOICE",
        "MULTIPLE": "MULTIPLE_CHOICE",
        "MULTIPLE_CHOICE": "MULTIPLE_CHOICE",
        "객관식": "MULTIPLE_CHOICE",
        "O/X": "OX",
        "OX": "OX",
        "TRUE_FALSE": "OX",
        "TRUE/FALSE": "OX",
        "단답형": "SHORT_ANSWER",
        "SHORT": "SHORT_ANSWER",
        "SHORT_ANSWER": "SHORT_ANSWER",
    }
    return aliases.get(normalized, "MULTIPLE_CHOICE")


def _trim_text(value: str, limit: int = 90) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _clip_text(value: str, limit: int = 90) -> str:
    """퀴즈 보기처럼 말줄임표가 위험한 곳에서는 문장을 자르되 ...를 붙이지 않습니다."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    text = text[:limit].rstrip(" ,:;./…")
    return text


def _clean_quiz_fragment(value: str) -> str:
    """PDF 표식, 깨진 문자, 장식 불릿을 제거해 퀴즈에 쓸 수 있는 조각으로 정리합니다."""
    text = str(value or "")
    text = re.sub(r"\[[^\]]*(?:PDF\s*page|page|페이지)[^\]]*\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = text.replace("�", " ")
    text = _SOURCE_BULLET_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n-–—•·,;:/|")


def _has_bad_quiz_artifact(value: str) -> bool:
    text = str(value or "")
    if not text or "�" in text:
        return True
    return any(phrase in text for phrase in _BAD_QUIZ_PHRASES)


def _has_bad_option_phrase(value: str) -> bool:
    text = _clean_quiz_fragment(value)
    return any(phrase in text for phrase in _BAD_OPTION_PHRASES)


def _is_generic_mc_question(value: str) -> bool:
    text = _clean_quiz_fragment(value)
    return any(phrase in text for phrase in _GENERIC_MC_QUESTION_PHRASES)


def _looks_like_heading_fragment(value: str) -> bool:
    """목차/슬라이드 제목처럼 보이는 조각은 보기와 질문 후보에서 제외합니다."""
    text = _clean_quiz_fragment(value)
    if not text or _has_bad_quiz_artifact(text):
        return True
    if re.search(r"(?:주차|강의|수업).{0,35}(?:정리|살펴|학습|다룹|소개)", text):
        return True
    if len(text) < 6:
        return True
    if re.fullmatch(r"\d+", text):
        return True
    if re.search(r"(학습목표|목차|contents|index)$", text, flags=re.IGNORECASE):
        return True
    if text.count(" - ") >= 2 or text.count("/") >= 3:
        return True
    if re.search(r"\s\d{1,2}$", text) and len(text) < 60:
        return True
    if "의 이해" in text and not re.search(r"(다|한다|된다|이다|있다|없다)[\.\s]*$", text):
        return True
    if re.search(r"(개념|이해|목표|목차|표현|형식|종류|구성)$", text) and not re.search(
        r"(다|한다|된다|이다|있다|없다|필요하다|가능하다)[\.\s]*$",
        text,
    ):
        return True
    if (
        len(text) < 32
        and not re.search(r"(은|는|이|가|을|를|으로|에서).*(다|한다|된다|이다|있다|없다)", text)
        and not any(marker in text for marker in _EXPLANATORY_MARKERS)
    ):
        return True
    if not any(marker in text for marker in _EXPLANATORY_MARKERS):
        # 짧은 명사구는 정답/오답으로는 빈약하므로 제외한다.
        return True
    return False


def _is_good_question_text(value: str) -> bool:
    text = _clean_quiz_fragment(value)
    if not text or _has_bad_quiz_artifact(text):
        return False
    if len(text) < 8 or len(text) > 100:
        return False
    if len(text) > 76 and any(phrase in text for phrase in ("옳은 것은", "알맞은 것은")):
        return False
    if re.search(r"(입니다|합니다|됩니다|있습니다|없습니다|비례합니다)\.\s*이에 대한 설명", text):
        return False
    if any(phrase in text for phrase in ("작성하세요", "설명해주세요", "요약하세요", "요약하여")):
        return False
    if _is_generic_mc_question(text):
        return False
    if re.fullmatch(r"\d+", text):
        return False
    if re.search(r"(학습목표|목차|contents|index)$", text, flags=re.IGNORECASE):
        return False
    if "?" not in text and "？" not in text and not any(
        phrase in text for phrase in ("옳은 것은", "알맞은 것은", "무엇", "어떤", "쓰세요")
    ):
        return False
    return True


def _is_recoverable_plain_question(value: str) -> bool:
    text = _clean_quiz_fragment(value)
    if not _is_good_question_text(text):
        return False
    if "?" not in text and "？" not in text and not text.endswith(("인가", "인가요", "무엇인가")):
        return False
    return True


def _is_good_option_text(value: str) -> bool:
    text = _clean_quiz_fragment(_strip_option_prefix(value))
    if _has_bad_option_phrase(text):
        return False
    if _looks_like_heading_fragment(text):
        return False
    if len(text) < 8 or len(text) > 64:
        return False
    return True


def _parse_summary_text_json(raw_text: str) -> str:
    text = _strip_json_code_fences(_remove_thinking_blocks(raw_text))
    candidates = [text, *_extract_balanced_json_candidates(text)]
    for candidate in candidates:
        candidate = _normalize_json_candidate(candidate)
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            summary_text = str(payload.get("summary_text") or "").strip()
            if summary_text:
                return summary_text
    raise QuizParseError("요약 JSON에 summary_text가 없습니다.", raw_text)


def _local_quiz_source_summary(source_text: str, *, max_sentences: int = 14) -> str:
    """LLM 요약 실패 시에도 퀴즈 생성에 쓸 수 있는 정제 요약본을 만듭니다."""
    sentences = _source_sentences(source_text, limit=max_sentences)
    if not sentences:
        cleaned = _clean_quiz_fragment(source_text)
        return _trim_text(cleaned, 1400)
    return "\n".join(f"- {sentence.rstrip('.。')}" for sentence in sentences)


async def _build_quiz_source_summary(
    client: httpx.AsyncClient,
    source_text: str,
) -> str:
    """원문을 바로 퀴즈에 넣지 않고 요약본으로 압축해 문제 품질을 안정화합니다."""
    fallback_summary = _local_quiz_source_summary(source_text)
    prompt_source = fallback_summary if len(fallback_summary) >= 80 else _clean_quiz_fragment(source_text)
    prompt_source = _trim_text(prompt_source, QUIZ_CONTEXT_CHARS)
    _demo_log(
        f"1) 선택 소스 정리: source_chars={len(source_text or '')}, "
        f"prompt_chars={len(prompt_source)}, fallback_lines={len(fallback_summary.splitlines())}"
    )
    for index, line in enumerate(fallback_summary.splitlines()[:5], start=1):
        _demo_log(f"   핵심문장#{index}: {_preview(line, 110)}")
    _log_sentence_morphemes(
        [line.lstrip("- ").strip() for line in fallback_summary.splitlines() if line.strip()],
        label="퀴즈 소스",
    )

    if not prompt_source.strip():
        raise ValueError("퀴즈 생성에 사용할 소스 내용이 없습니다.")

    try:
        _demo_log("2) 퀴즈용 요약본 생성을 위해 LLM에 전달")
        res = await client.post(
            f"{LLM_URL}/v1/chat/completions",
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": QUIZ_SOURCE_SUMMARY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": QUIZ_SOURCE_SUMMARY_PROMPT_TEMPLATE.format(source_text=prompt_source),
                    },
                ],
                "max_tokens": 900,
                "temperature": 0.0,
                "bridgeprag_alpha": 0.0,
                "bridgeprag_disable_memory": True,
                "demo_feature": "quiz",
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        )
        res.raise_for_status()
        raw_answer = res.json()["choices"][0]["message"]["content"]
        summary_text = _parse_summary_text_json(raw_answer)
        summary_text = _local_quiz_source_summary(summary_text, max_sentences=16)
        if len(summary_text) >= 40:
            logger.info("[QUIZ] 퀴즈용 요약본 생성 완료: %d chars", len(summary_text))
            _demo_log(f"3) 퀴즈용 요약본 생성 완료: chars={len(summary_text)}")
            for index, line in enumerate(summary_text.splitlines()[:5], start=1):
                _demo_log(f"   요약문장#{index}: {_preview(line, 110)}")
            return summary_text
    except Exception as exc:
        logger.warning("[QUIZ] 퀴즈용 LLM 요약 실패, 로컬 정제 요약본 사용: %s", exc)

    logger.info("[QUIZ] 로컬 정제 요약본 사용: %d chars", len(fallback_summary))
    _demo_log(f"3) 로컬 정제 요약본 사용: chars={len(fallback_summary)}")
    for index, line in enumerate(fallback_summary.splitlines()[:5], start=1):
        _demo_log(f"   로컬요약#{index}: {_preview(line, 110)}")
    return fallback_summary


def _source_sentences(transcript_text: str, *, limit: int = 20) -> list[str]:
    """선택 소스에서 퀴즈 fallback에 사용할 핵심 문장 후보를 추출합니다."""
    raw_text = str(transcript_text or "")
    raw_text = re.sub(r"\[[^\]]*(?:PDF\s*page|page|페이지)[^\]]*\]", "\n", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", raw_text)
    raw_text = raw_text.replace("�", " ")
    raw_text = _SOURCE_BULLET_RE.sub(". ", raw_text)
    cleaned = re.sub(r"\s+", " ", raw_text).strip()
    sentences = []
    seen = set()
    parts = [match.group(0) for match in _SENTENCE_END_FINDER_RE.finditer(cleaned)]
    if not parts:
        parts = re.split(r"\n+|[.;!?。？！]", raw_text)
    for part in parts:
        sentence = _clean_quiz_fragment(part)
        if not 8 <= len(sentence) <= 180:
            continue
        if _looks_like_heading_fragment(sentence):
            continue
        if sentence in seen:
            continue
        seen.add(sentence)
        sentences.append(sentence)
        if len(sentences) >= limit:
            break
    if not sentences and cleaned:
        sentences.append(_trim_text(cleaned, 120))
    return sentences


def _extract_topic(sentence: str, fallback: str) -> str:
    """문장에서 질문 주제로 쓸 만한 짧은 명사구를 추출합니다."""
    sentence = _clean_quiz_fragment(sentence)
    patterns = (
        r"([가-힣A-Za-z0-9_+#./ -]{2,24}?)(?:은|는)\s",
        r"([가-힣A-Za-z0-9_+#./ -]{2,24}?)(?:이|가)\s",
        r"([A-Za-z][A-Za-z0-9_+#./-]{1,24}|[가-힣]{2,12})",
    )
    for pattern in patterns:
        match = re.search(pattern, sentence)
        if match:
            topic = match.group(1).strip(" ,.:;")
            if len(topic) >= 2:
                return _trim_text(topic, 24)
    return fallback


def _source_phrase(sentence: str, limit: int = 34) -> str:
    """보기/단답 정답에 들어갈 짧은 원문 기반 표현을 만듭니다."""
    sentence = _clean_quiz_fragment(sentence)
    sentence = re.sub(r"^(따라서|그리고|또한|반면|하지만)\s*", "", sentence)
    sentence = sentence.rstrip(".。")
    return _clip_text(sentence, limit)


def _strip_sentence_ending(value: str) -> str:
    text = _clean_quiz_fragment(value).rstrip(".。")
    text = re.sub(r"(입니다|합니다|됩니다|있습니다|없습니다)$", "", text).strip()
    text = re.sub(r"(이다|한다|된다|있다|없다)$", "", text).strip()
    return text


def _topic_particle(topic: str) -> str:
    """주제어 뒤에 붙일 은/는 조사를 간단히 고릅니다."""
    topic = str(topic or "").strip()
    if not topic:
        return "은"
    last = topic[-1]
    if "가" <= last <= "힣":
        return "은" if (ord(last) - ord("가")) % 28 else "는"
    return "는"


def _split_subject_predicate(sentence: str) -> tuple[str, str]:
    """'A는 B입니다' 형태의 강의 문장을 질문 주제와 정답 후보로 분리합니다."""
    text = _clean_quiz_fragment(sentence).rstrip(".。")
    match = re.match(r"(.{2,32}?)(은|는|이|가)\s+(.{4,120})$", text)
    if not match:
        return "", ""
    subject = match.group(1).strip(" ,:;")
    subject = re.sub(r"^.*?에서\s+", "", subject).strip()
    predicate = _strip_sentence_ending(match.group(3))
    if len(subject) < 1 or len(predicate) < 4:
        return "", ""
    return subject, predicate


def _compact_answer_from_sentence(sentence: str, *, limit: int = 48) -> str:
    """객관식 보기에 들어가기 좋은 정답 표현을 원문 문장에서 추출합니다."""
    sentence = _clean_quiz_fragment(sentence)
    if "수직" in sentence and "일" in sentence and re.search(r"\b0\b|0이|영", sentence):
        return "일은 0이 된다"
    if "속력의 제곱" in sentence and "비례" in sentence:
        return "속력의 제곱에 비례한다"
    if "질량" in sentence and "속력" in sentence and "증가" in sentence:
        return "질량과 속력이 클수록 증가한다"
    if "역학적 에너지" in sentence and "보존" in sentence and "일정" in sentence:
        return "운동에너지와 위치에너지의 합이 일정하다"
    if "위치에너지" in sentence and "높이" in sentence:
        return "높이에 의해 저장되는 에너지"
    subject, predicate = _split_subject_predicate(sentence)
    if subject and predicate:
        if len(predicate) >= 8 and not predicate.startswith(("하나의", "모든")):
            return _clip_text(predicate, limit)
    return _source_phrase(sentence, limit)


def _mc_stem_and_answer_from_sentence(
    source_sentence: str,
    fallback_answer: str,
    type_index: int,
) -> tuple[str, str]:
    """정답 내용을 질문에 그대로 노출하지 않도록 개념형 질문과 짧은 정답을 구성합니다."""
    sentence = _clean_quiz_fragment(source_sentence or fallback_answer)
    answer = _compact_answer_from_sentence(sentence or fallback_answer, limit=54)
    subject, predicate = _split_subject_predicate(sentence)

    if "수직" in sentence and "일" in sentence and re.search(r"\b0\b|0이|영", sentence):
        return "힘이 이동 방향과 수직일 때 일은 어떻게 되는가?", "일은 0이 된다"
    if "속력의 제곱" in sentence and "운동에너지" in sentence:
        return "운동에너지는 속력과 어떤 관계를 가지는가?", "속력의 제곱에 비례한다"
    if "질량" in sentence and "속력" in sentence and "운동에너지" in sentence:
        return "운동에너지를 증가시키는 요인으로 옳은 것은?", "질량과 속력이 클수록 증가한다"
    if "역학적 에너지" in sentence and "보존" in sentence:
        return "역학적 에너지 보존에 대한 설명으로 옳은 것은?", "운동에너지와 위치에너지의 합이 일정하다"
    if "위치에너지" in sentence and "높이" in sentence:
        return "위치에너지에 대한 설명으로 옳은 것은?", "높이에 의해 저장되는 에너지"
    if subject and predicate:
        if "무엇" not in subject and len(subject) <= 24:
            return f"{subject}에 대한 설명으로 옳은 것은?", _clip_text(predicate, 54)

    topic = _extract_topic(sentence or fallback_answer, f"핵심 개념 {type_index + 1}")
    templates = (
        "{topic}에 대한 설명으로 옳은 것은?",
        "{topic}의 핵심 역할로 알맞은 것은?",
        "{topic}의 특징으로 옳은 것은?",
        "{topic}와 관련된 설명으로 옳은 것은?",
    )
    return templates[type_index % len(templates)].format(topic=topic), answer


def _mc_question_from_source(source_sentence: str, answer_text: str, type_index: int) -> str:
    """객관식 질문을 '강의 내용' 같은 메타 질문이 아니라 특정 개념 질문으로 만듭니다."""
    question, _ = _mc_stem_and_answer_from_sentence(source_sentence, answer_text, type_index)
    return question


def _short_answer_from_sentence(
    source_sentence: str,
    fallback_answer: str,
    type_index: int,
) -> tuple[str, str]:
    """단답형 문항을 서로 다른 지식/정보를 묻는 형태로 구성합니다."""
    sentence = _clean_quiz_fragment(source_sentence or fallback_answer)
    subject, predicate = _split_subject_predicate(sentence)

    if "수직" in sentence and "일" in sentence and re.search(r"\b0\b|0이|영", sentence):
        return "힘이 이동 방향과 수직일 때 일의 값은 얼마인가?", "0"
    if "속력의 제곱" in sentence and "운동에너지" in sentence:
        return "운동에너지는 속력에 대해 어떻게 비례하는가?", "속력의 제곱에 비례한다"
    if "질량" in sentence and "속력" in sentence and "운동에너지" in sentence:
        return "운동에너지를 증가시키는 두 요인은 무엇인가?", "질량과 속력"
    if "역학적 에너지" in sentence and "보존" in sentence:
        return "역학적 에너지 보존에서 일정하게 유지되는 값은 무엇인가?", "운동에너지와 위치에너지의 합"
    if "위치에너지" in sentence and "높이" in sentence:
        return "위치에너지는 어떤 조건 때문에 저장되는 에너지인가?", "위치나 높이"
    if subject and predicate:
        if subject == "일":
            return "물리에서 일은 무엇을 의미하는가?", _clip_text(predicate, 56)
        if any(marker in predicate for marker in ("에너지", "과정", "의미", "개념", "구조", "방식")):
            return f"{subject}{_topic_particle(subject)} 무엇을 의미하는가?", _clip_text(predicate, 56)
        templates = (
            "{subject}은 무엇을 의미하는가?",
            "{subject}의 핵심 특징은 무엇인가?",
            "{subject}은 어떤 역할을 하는가?",
            "{subject}와 관련된 핵심 조건은 무엇인가?",
        )
        return templates[type_index % len(templates)].format(subject=subject), _clip_text(predicate, 56)

    topic = _extract_topic(sentence, f"핵심 개념 {type_index + 1}")
    question_templates = (
        "{topic}의 핵심 의미는 무엇인가?",
        "{topic}에서 중요한 조건은 무엇인가?",
        "{topic}와 관련된 주요 결과는 무엇인가?",
        "{topic}의 대표적인 특징은 무엇인가?",
    )
    return question_templates[type_index % len(question_templates)].format(topic=topic), _compact_answer_from_sentence(sentence, limit=56)


def _is_generic_short_answer_question(value: str) -> bool:
    text = _clean_quiz_fragment(value)
    return any(phrase in text for phrase in _GENERIC_SHORT_QUESTION_PHRASES)


def _strip_option_prefix(value: str) -> str:
    text = " ".join(str(value or "").split()).strip()
    text = re.sub(r"^[A-Da-d]\s*[\.\)]\s*", "", text)
    text = re.sub(r"^[1-4]\s*[\.\)]\s*", "", text)
    return text.strip()


def _option_key(value: str) -> str:
    text = _strip_option_prefix(value)
    text = re.sub(r"\s+", "", text).lower()
    return re.sub(r"[^0-9a-z가-힣]", "", text)


def _keyword_root(token: str) -> str:
    token = token.lower().strip()
    for suffix in ("으로부터", "으로서", "으로써", "에서는", "에게는", "까지의", "부터", "에서", "으로", "에게", "이다", "이며", "이고", "라는", "까지", "보다", "처럼", "만큼", "에는", "은", "는", "이", "가", "을", "를", "의", "에", "와", "과", "도"):
        if len(token) > len(suffix) + 1 and token.endswith(suffix):
            token = token[: -len(suffix)]
            break
    return token


def _content_keywords(value: str) -> set[str]:
    stop_words = {
        "것",
        "것은",
        "설명",
        "자료",
        "강의",
        "내용",
        "핵심",
        "대한",
        "관련",
        "옳은",
        "알맞은",
        "입니다",
        "합니다",
        "됩니다",
        "있습니다",
        "없습니다",
    }
    keywords = set()
    for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", _clean_quiz_fragment(value)):
        root = _keyword_root(token)
        if len(root) >= 2 and root not in stop_words:
            keywords.add(root)
    return keywords


def _is_same_topic_option(answer_text: str, option_text: str) -> bool:
    """객관식 오답이 정답과 같은 개념권에서 만들어졌는지 확인합니다."""
    answer_keywords = _content_keywords(answer_text)
    option_keywords = _content_keywords(option_text)
    if not answer_keywords or not option_keywords:
        return False
    return bool(answer_keywords & option_keywords)


def _question_duplicates_option(question_text: str, options: list[str]) -> bool:
    question_key = _option_key(question_text)
    if len(question_key) < 14:
        return False
    for option in options:
        option_key = _option_key(option)
        if len(option_key) >= 14 and (question_key in option_key or option_key in question_key):
            return True
    return False


def _answer_matches_question_context(question_text: str, answer_text: str, source_sentence: str) -> bool:
    """LLM이 질문과 다른 개념을 정답으로 붙이는 경우를 걸러냅니다."""
    question_keywords = _content_keywords(question_text)
    answer_keywords = _content_keywords(answer_text)
    source_keywords = _content_keywords(source_sentence)
    if not question_keywords or not answer_keywords:
        return True
    if question_keywords & answer_keywords:
        return True
    if source_keywords and question_keywords & source_keywords and len(answer_keywords & source_keywords) >= 2:
        return True
    return False


def _append_direct_mc_option(options: list[str], value: str, *, limit: int = 64) -> None:
    """LLM이 직접 만든 객관식 보기를 보존하기 위한 완화된 정제 함수입니다."""
    option = _clean_quiz_fragment(_strip_option_prefix(value))
    option = option.strip(" ,:;\"'")
    if not option or _has_bad_quiz_artifact(option) or _has_bad_option_phrase(option):
        return
    if len(option) < 3:
        return
    option = _clip_text(option, limit)
    key = _option_key(option)
    if not key:
        return
    for existing in options:
        existing_key = _option_key(existing)
        if key == existing_key:
            return
        if len(key) >= 14 and len(existing_key) >= 14 and (key in existing_key or existing_key in key):
            return
    options.append(option)


def _normalize_llm_mc_direct(
    candidate: dict,
    transcript_text: str,
    type_index: int,
) -> dict | None:
    """소스 문장 후보가 부족해도 LLM이 만든 answer/distractors 기반 객관식을 우선 살립니다."""
    raw_options = candidate.get("options") if isinstance(candidate.get("options"), list) else []
    raw_distractors = candidate.get("distractors") if isinstance(candidate.get("distractors"), list) else []
    answer_text = (
        candidate.get("answer")
        or candidate.get("correct_answer")
        or candidate.get("정답")
        or ""
    )
    if not answer_text and raw_options:
        answer_text = raw_options[0]
    answer_text = _clean_quiz_fragment(_strip_option_prefix(str(answer_text)))
    if not answer_text or _has_bad_quiz_artifact(answer_text):
        return None

    question_text = _clean_quiz_fragment(candidate.get("question") or "")
    answer_was_replaced = False
    source_sentence = _best_source_sentence_for_question(
        question_text or answer_text,
        transcript_text,
        type_index,
    )
    if source_sentence and not _answer_matches_question_context(question_text, answer_text, source_sentence):
        source_answer = _compact_answer_from_sentence(source_sentence, limit=64)
        if source_answer:
            answer_text = source_answer
            answer_was_replaced = True

    option_texts: list[str] = []
    _append_direct_mc_option(option_texts, answer_text, limit=64)
    if not option_texts:
        return None

    # LLM이 준 보기와 distractor를 먼저 신뢰한다. 소스 문장이 적은 PDF/짧은 전사에서도 실패하지 않게 하기 위함이다.
    for value in [*raw_distractors, *raw_options]:
        if _option_key(value) == _option_key(answer_text):
            continue
        if answer_was_replaced and not _answer_matches_question_context(question_text, str(value), source_sentence):
            continue
        _append_direct_mc_option(option_texts, str(value), limit=64)
        if len(option_texts) >= 4:
            break

    # 부족한 경우에만 원문/규칙 기반 후보로 보충한다. LLM 생성값을 버리지는 않는다.
    if len(option_texts) < 4:
        for value in _make_domain_distractors(answer_text, source_sentence, limit=6):
            _append_direct_mc_option(option_texts, value, limit=64)
            if len(option_texts) >= 4:
                break
    if len(option_texts) < 4:
        for value in _make_rule_based_distractors(answer_text, limit=6):
            _append_direct_mc_option(option_texts, value, limit=64)
            if len(option_texts) >= 4:
                break
    if len(option_texts) < 4:
        for value in _make_contrastive_distractors(answer_text, limit=6):
            _append_direct_mc_option(option_texts, value, limit=64)
            if len(option_texts) >= 4:
                break
    if len(option_texts) < 4:
        return None

    option_texts = option_texts[:4]
    answer_index = max(0, min(3, type_index % 4))
    if answer_index != 0:
        option_texts[0], option_texts[answer_index] = option_texts[answer_index], option_texts[0]
    numbered_options = [f"{index}. {text}" for index, text in enumerate(option_texts, start=1)]
    correct_answer = numbered_options[answer_index]

    stem_question, _ = _mc_stem_and_answer_from_sentence(source_sentence, answer_text, type_index)
    if (
        not _is_good_question_text(question_text)
        or _question_duplicates_option(question_text, numbered_options)
        or _mc_question_leaks_answer(question_text, correct_answer, numbered_options)
    ):
        topic = _extract_topic(source_sentence or answer_text, f"핵심 개념 {type_index + 1}")
        question_text = stem_question or f"{topic}에 대한 설명으로 옳은 것은?"

    return {
        "question_index": candidate.get("question_index") or 1,
        "type": "MULTIPLE_CHOICE",
        "question": _clip_text(question_text, 70),
        "options": numbered_options,
        "correct_answer": correct_answer,
        "user_answer": None,
        "is_correct": None,
        "explanation": _trim_text(candidate.get("explanation") or answer_text, 70),
    }


def _mc_question_leaks_answer(question_text: str, answer_text: str, options: list[str]) -> bool:
    """질문이 정답 문장을 거의 그대로 포함하면 객관식 문제로 부적절합니다."""
    question = _clean_quiz_fragment(question_text)
    answer = _clean_quiz_fragment(_strip_option_prefix(answer_text))
    if not question or not answer:
        return False
    if len(answer) >= 12 and answer in question:
        return True
    if re.search(r"(입니다|합니다|됩니다|있습니다|없습니다|비례합니다)\.\s*이에 대한 설명", question):
        return True
    if len(question) > 76 and any(phrase in question for phrase in ("옳은 것은", "알맞은 것은")):
        return True

    question_keywords = _content_keywords(question)
    answer_keywords = _content_keywords(answer)
    if len(answer_keywords) >= 3 and len(question_keywords & answer_keywords) >= 3:
        return True
    for option in options:
        option_text = _clean_quiz_fragment(_strip_option_prefix(option))
        option_keywords = _content_keywords(option_text)
        if len(option_keywords) >= 3 and len(question_keywords & option_keywords) >= 3:
            return True
    return False


def _short_question_leaks_answer(question_text: str, answer_text: str) -> bool:
    """단답형 질문이 답을 그대로 포함하면 복습 문항으로 부적절합니다."""
    question = _clean_quiz_fragment(question_text)
    answer = _clean_quiz_fragment(answer_text)
    if not question or not answer:
        return False
    if len(answer) >= 10 and answer in question:
        return True
    question_keywords = _content_keywords(question)
    answer_keywords = _content_keywords(answer)
    if len(answer_keywords) >= 3 and len(question_keywords & answer_keywords) >= 3:
        return True
    return False


def _concept_signature(value: str) -> str:
    """문항이 같은 핵심 개념만 반복되는지 확인하기 위한 느슨한 서명입니다."""
    text = _clean_quiz_fragment(value)
    tokens = []
    stop_words = {"설명", "특징", "핵심", "역할", "알맞은", "옳은", "것은", "대한", "관련된"}
    for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", text):
        root = _keyword_root(token)
        if root and root not in stop_words and root not in tokens:
            tokens.append(root)
        if len(tokens) >= 3:
            break
    return "|".join(tokens)


def _question_concept_signature(question: dict) -> str:
    fields = [
        str(question.get("question") or ""),
        str(question.get("correct_answer") or ""),
        str(question.get("explanation") or ""),
    ]
    return _concept_signature(" ".join(fields))


def _same_concept_exists(question: dict, existing_items: list[str]) -> bool:
    signature = _question_concept_signature(question)
    if not signature:
        return False
    current = set(signature.split("|"))
    for item in existing_items:
        other = set(_concept_signature(item).split("|"))
        if not other:
            continue
        # 같은 핵심어 2개 이상이 겹치면 같은 정보 반복으로 본다.
        if len(current & other) >= 2:
            return True
    return False


def _make_false_ox_statement(
    sentence: str,
    transcript_text: str = "",
    type_index: int = 0,
) -> str:
    """O/X 문항이 전부 O로만 나오지 않도록 원문 기반의 거짓 진술을 만듭니다."""
    subject, _ = _split_subject_predicate(sentence)
    if subject and transcript_text:
        source_sentences = _source_sentences(transcript_text, limit=20)
        for offset in range(1, len(source_sentences) + 1):
            other_sentence = source_sentences[(type_index + offset) % len(source_sentences)]
            if _option_key(other_sentence) == _option_key(sentence):
                continue
            other_answer = _compact_answer_from_sentence(other_sentence, limit=58)
            if other_answer and _is_good_option_text(other_answer):
                return _trim_text(f"{subject}{_topic_particle(subject)} {other_answer}.", 90)

    correct = _clean_quiz_fragment(sentence).rstrip(".。")
    for candidate in [
        *_make_rule_based_distractors(_clean_quiz_fragment(sentence), limit=4),
        *_make_domain_distractors(_compact_answer_from_sentence(sentence, limit=70), sentence, limit=4),
    ]:
        candidate = _clean_quiz_fragment(candidate).rstrip(".。")
        if candidate and _is_good_option_text(candidate) and len(candidate) >= 10:
            return _trim_text(candidate + ".", 90)

    fallback_topic = _extract_topic(sentence, "해당 개념")
    return _trim_text(f"{fallback_topic}은 입력 조건과 관계없이 항상 일정하다.", 90)


def _append_unique_phrase(phrases: list[str], value: str, *, limit: int = 44) -> None:
    phrase = _clean_quiz_fragment(_strip_option_prefix(value))
    phrase = phrase.strip(" ,:;\"'")
    if not phrase or not _is_good_option_text(phrase):
        return
    phrase = _trim_text(phrase, limit)
    key = _option_key(phrase)
    if not key:
        return
    for item in phrases:
        item_key = _option_key(item)
        if item_key == key:
            return
        if len(key) >= 12 and len(item_key) >= 12 and (key in item_key or item_key in key):
            return
    if not key:
        return
    phrases.append(phrase)


def _source_option_phrases(transcript_text: str, *, limit: int = 12) -> list[str]:
    """원문 문장과 절을 짧은 보기 후보로 변환합니다."""
    phrases: list[str] = []
    for sentence in _source_sentences(transcript_text, limit=20):
        _append_unique_phrase(phrases, _compact_answer_from_sentence(sentence, limit=48), limit=48)
        clauses = re.split(r"(?:,|，|;|；|\s+반면\s+|\s+하지만\s+|\s+그리고\s+|\s+또한\s+|\s+때문에\s+)", sentence)
        for clause in clauses:
            clause = _clean_quiz_fragment(clause)
            if 8 <= len(clause) <= 90:
                _append_unique_phrase(phrases, _compact_answer_from_sentence(clause, limit=48), limit=48)
            if len(phrases) >= limit:
                return phrases
    return phrases


def _make_rule_based_distractors(answer_text: str, *, limit: int = 3) -> list[str]:
    """정답 문장을 바탕으로 의미가 달라지는 짧은 오답 후보를 생성합니다."""
    answer = _clean_quiz_fragment(answer_text)
    candidates: list[str] = []
    replacements = (
        ("출력층", "입력층"),
        ("입력층", "출력층"),
        ("은닉층", "출력층"),
        ("처음 위치", "평균 속도"),
        ("나중 위치", "가속도"),
        ("방향을 가진", "크기만 가진"),
        ("변화량", "고정값"),
        ("가속도가 일정하다고", "속도가 일정하다고"),
        ("가속도가 일정하다고", "가속도가 0이라고"),
        ("가속도가 일정하다고", "가속도가 계속 변한다고"),
        ("가속도가 일정", "가속도가 계속 변함"),
        ("일정", "계속 변함"),
        ("시간에 비례", "시간과 무관"),
        ("시간의 제곱 항", "시간의 세제곱 항"),
        ("비선형성", "선형성"),
        ("부여", "제거"),
        ("9.8미터", "0미터"),
        ("9.8m/s", "0m/s"),
        ("8미터", "0미터"),
        ("8m/s", "0m/s"),
        ("각 층", "마지막 층"),
        ("가중치가", "입력값이"),
        ("가중치를", "입력값을"),
        ("가중치는", "입력값은"),
        ("손실을", "정확도를"),
        ("손실이", "정확도가"),
        ("손실은", "정확도는"),
        ("손실에", "정확도에"),
        ("예측값", "입력값"),
        ("정답", "가중치"),
        ("역전파", "순전파"),
        ("순전파", "역전파"),
        ("경사하강법은", "활성화 함수는"),
        ("경사하강법을", "활성화 함수를"),
        ("활성화 함수는", "손실 함수는"),
        ("활성화 함수를", "손실 함수를"),
        ("논리적 순서", "임의 순서"),
        ("연속", "분산"),
        ("인접한", "서로 무관한"),
        ("순서대로", "무작위로"),
        ("같은", "서로 다른"),
        ("공유", "분리"),
        ("독립적인", "공유된"),
        ("선형 리스트", "트리"),
        ("순차", "연결"),
        ("배열", "그래프"),
        ("인덱스", "포인터"),
        ("증가", "감소"),
        ("감소", "증가"),
        ("가능", "불가능"),
        ("필요", "불필요"),
        ("메모리", "외부 저장소"),
        ("이동해야", "이동할 필요가 없어"),
    )
    for old, new in replacements:
        if old in answer:
            _append_unique_phrase(candidates, answer.replace(old, new, 1), limit=60)
        if len(candidates) >= limit:
            return candidates

    negated = ""
    if answer.endswith("수 있습니다"):
        negated = answer[:-6].rstrip() + " 수 없습니다"
    elif answer.endswith("수 없습니다"):
        negated = answer[:-6].rstrip() + " 수 있습니다"
    elif answer.endswith("수 있다"):
        negated = answer[:-4].rstrip() + " 수 없다"
    elif answer.endswith("입니다"):
        negated = answer[:-3].rstrip() + "이 아닙니다"
    elif answer.endswith("합니다"):
        negated = answer[:-3].rstrip() + "하지 않습니다"
    elif answer.endswith("됩니다"):
        negated = answer[:-3].rstrip() + "되지 않습니다"
    elif answer.endswith("있습니다"):
        negated = answer[:-4].rstrip() + "없습니다"
    elif answer.endswith("없습니다"):
        negated = answer[:-4].rstrip() + "있습니다"
    elif answer.endswith("이다"):
        negated = answer[:-2].rstrip() + "이 아니다"
    elif answer.endswith("한다"):
        negated = answer[:-2].rstrip() + "하지 않는다"
    elif answer.endswith("된다"):
        negated = answer[:-2].rstrip() + "되지 않는다"
    elif answer.endswith("있다"):
        negated = answer[:-2].rstrip() + "없다"
    elif answer.endswith("없다"):
        negated = answer[:-2].rstrip() + "있다"
    elif answer.endswith("다"):
        negated = answer[:-1].rstrip() + "지 않는다"
    if negated:
        _append_unique_phrase(candidates, negated, limit=60)
    return candidates[:limit]


def _make_domain_distractors(
    answer_text: str,
    source_sentence: str,
    *,
    limit: int = 3,
) -> list[str]:
    """강의 개념권 안에서 헷갈릴 만한 오답을 우선 생성합니다."""
    answer = _clean_quiz_fragment(answer_text)
    source = _clean_quiz_fragment(source_sentence)
    base = f"{answer} {source}"
    candidates: list[str] = []

    if "수직" in base and "일" in base:
        for value in ("일은 최대가 된다", "일은 항상 음수가 된다", "일은 질량에만 비례한다", "일은 속력에만 비례한다"):
            _append_unique_phrase(candidates, value, limit=48)
    if "속력의 제곱" in base or ("운동에너지" in base and "비례" in base):
        for value in ("속력에만 비례한다", "속력에 반비례한다", "속력과 무관하다", "질량과 무관하게 일정하다"):
            _append_unique_phrase(candidates, value, limit=48)
    if "운동에너지" in base and "에너지" in base:
        for value in ("위치 때문에 저장되는 에너지", "열로 전달된 에너지", "정지 상태에서만 가지는 에너지", "힘의 크기만 나타내는 물리량"):
            _append_unique_phrase(candidates, value, limit=48)
    if "위치에너지" in base or ("높이" in base and "에너지" in base):
        for value in ("물체가 운동하기 때문에 가지는 에너지", "힘이 이동시키며 전달하는 에너지", "속력의 제곱에 비례하는 에너지", "열로 전달되는 에너지"):
            _append_unique_phrase(candidates, value, limit=48)
    if "역학적 에너지" in base or "보존" in base:
        for value in ("운동에너지와 위치에너지의 합이 일정하다", "마찰이 클수록 항상 증가한다", "운동에너지만 일정하게 유지된다", "위치에너지만 일정하게 유지된다"):
            _append_unique_phrase(candidates, value, limit=48)
    if "일" in base and "에너지" in base and "힘" in base:
        for value in ("운동 때문에 가지는 에너지", "위치 때문에 저장되는 에너지", "힘의 크기만 나타내는 물리량", "이동 거리와 무관한 물리량"):
            _append_unique_phrase(candidates, value, limit=48)

    if "손실" in base or "loss" in base.lower():
        for value in ("예측값과 정답의 차이를 나타내는 값", "가중치를 무작위로 고정하는 과정", "입력값을 그대로 출력하는 함수"):
            _append_unique_phrase(candidates, value, limit=48)
    if "역전파" in base:
        for value in ("출력층에서 입력층 방향으로 오차를 전달한다", "입력 데이터를 정렬하는 과정", "학습률을 고정하는 함수"):
            _append_unique_phrase(candidates, value, limit=48)
    if "활성화" in base:
        for value in ("비선형성을 부여하는 함수", "손실을 직접 계산하는 기준", "데이터를 저장하는 배열 구조"):
            _append_unique_phrase(candidates, value, limit=48)

    if "배열" in base or "인덱스" in base:
        for value in ("인덱스로 원소에 접근한다", "포인터만으로 순차 접근한다", "트리 형태로 계층을 저장한다"):
            _append_unique_phrase(candidates, value, limit=48)
    if "연결 리스트" in base or "리스트" in base:
        for value in ("노드가 링크로 다음 노드를 가리킨다", "모든 원소가 연속된 메모리에 저장된다", "인덱스로 항상 즉시 접근한다"):
            _append_unique_phrase(candidates, value, limit=48)

    return candidates[:limit]


def _make_contrastive_distractors(answer_text: str, *, limit: int = 3) -> list[str]:
    """짧은 소스에서도 객관식이 실패하지 않도록 최후의 대비 오답을 만든다."""
    answer = _clean_quiz_fragment(answer_text)
    topic = _extract_topic(answer, "해당 개념")
    topic = re.sub(r"(은|는|이|가|을|를|의|에)$", "", topic).strip() or "해당 개념"
    candidates: list[str] = []

    templates = (
        "{topic}은 입력 조건과 관계없이 일정하다",
        "{topic}은 결과 변화와 직접 관련되지 않는다",
        "{topic}은 비교 기준 없이 단독으로 결정된다",
        "{topic}은 다른 요소의 영향을 받지 않는다",
    )
    for template in templates:
        _append_unique_phrase(candidates, template.format(topic=topic), limit=60)
        if len(candidates) >= limit:
            return candidates[:limit]
    return candidates[:limit]


def _keywords_for_match(value: str) -> set[str]:
    return {
        _keyword_root(token.lower())
        for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", str(value or ""))
        if _keyword_root(token.lower()) not in {"자료", "설명한", "내용", "다음", "것", "어떤", "형태"}
    }


def _best_source_sentence_for_question(question: str, transcript_text: str, type_index: int) -> str:
    sentences = _source_sentences(transcript_text, limit=20)
    if not sentences:
        return ""
    question_keywords = _keywords_for_match(question)
    best_sentence = sentences[type_index % len(sentences)]
    best_score = -1
    for sentence in sentences:
        sentence_keywords = _keywords_for_match(sentence)
        score = len(question_keywords & sentence_keywords)
        if score > best_score:
            best_score = score
            best_sentence = sentence
    return best_sentence


def _plain_text_to_candidate(
    raw_text: str,
    question_type: str,
    transcript_text: str,
    type_index: int,
) -> dict | None:
    """JSON을 무시한 LLM 자연어 응답에서 질문 문장을 회수해 문항으로 변환합니다."""
    text = _strip_json_code_fences(_remove_thinking_blocks(raw_text))
    lines = []
    for raw_line in text.splitlines():
        line = _clean_quiz_fragment(re.sub(r"^\s*[-*•\d]+[.)]?\s*", "", raw_line))
        if not line or re.match(r"^\[(?:PDF\s+page|page|페이지)\s*\d+\]$", line, flags=re.IGNORECASE):
            continue
        if len(line) < 6:
            continue
        lines.append(line)
    if not lines:
        text = " ".join(text.split())
        if text:
            lines.append(text)
    if not lines:
        return None

    question_line = next((line for line in lines if _is_recoverable_plain_question(line)), "")
    if not question_line:
        return None
    question_line = _trim_text(question_line, 70)
    source_sentence = _best_source_sentence_for_question(question_line, transcript_text, type_index)
    answer = _source_phrase(source_sentence, 60)
    if not answer:
        return None

    candidate = {
        "question_index": 1,
        "type": question_type,
        "question": question_line,
        "answer": answer,
        "correct_answer": answer,
        "distractors": [],
        "explanation": _trim_text(source_sentence, 70),
    }
    return _normalize_single_llm_question(candidate, question_type, transcript_text, type_index)


def _numbered_mc_question(
    question: dict,
    answer_text: str,
    distractors: list[str],
    transcript_text: str,
    answer_slot: int,
) -> dict | None:
    raw_answer = _strip_option_prefix(answer_text)
    if not raw_answer:
        return None
    source_sentence = _best_source_sentence_for_question(
        f"{question.get('question') or ''} {raw_answer}",
        transcript_text,
        answer_slot,
    )
    stem_question, answer = _mc_stem_and_answer_from_sentence(source_sentence, raw_answer, answer_slot)

    option_texts: list[str] = []
    _append_unique_phrase(option_texts, answer, limit=60)
    if not option_texts:
        return None
    for distractor in distractors:
        if _is_same_topic_option(answer, distractor):
            _append_unique_phrase(option_texts, distractor, limit=52)

    if len(option_texts) < 4:
        for distractor in _make_domain_distractors(answer, source_sentence, limit=5):
            _append_unique_phrase(option_texts, distractor, limit=52)
            if len(option_texts) >= 4:
                break

    if len(option_texts) < 4:
        for distractor in _make_rule_based_distractors(answer, limit=4):
            _append_unique_phrase(option_texts, distractor, limit=60)
            if len(option_texts) >= 4:
                break

    if len(option_texts) < 4:
        for source_phrase in _source_option_phrases(transcript_text, limit=16):
            _append_unique_phrase(option_texts, source_phrase, limit=48)
            if len(option_texts) >= 4:
                break

    if len(option_texts) < 4:
        for distractor in _make_contrastive_distractors(answer, limit=4):
            _append_unique_phrase(option_texts, distractor, limit=60)
            if len(option_texts) >= 4:
                break

    if len(option_texts) < 4:
        for source_phrase in _source_option_phrases(transcript_text, limit=20):
            _append_unique_phrase(option_texts, source_phrase, limit=48)
            if len(option_texts) >= 4:
                break

    if len(option_texts) < 4:
        return None

    option_texts = option_texts[:4]
    answer_text = option_texts[0]
    answer_index = max(0, min(3, answer_slot % 4))
    if answer_index != 0:
        option_texts[0], option_texts[answer_index] = option_texts[answer_index], option_texts[0]

    numbered_options = [
        f"{index}. {text}"
        for index, text in enumerate(option_texts, start=1)
    ]
    correct_answer = numbered_options[answer_index]

    next_question = dict(question)
    question_text = _clean_quiz_fragment(next_question.get("question") or "")
    if (
        not _is_good_question_text(question_text)
        or _question_duplicates_option(question_text, numbered_options)
        or _mc_question_leaks_answer(question_text, correct_answer, numbered_options)
    ):
        question_text = stem_question

    next_question["type"] = "MULTIPLE_CHOICE"
    next_question["question"] = _clip_text(question_text, 70)
    next_question["options"] = numbered_options
    next_question["correct_answer"] = correct_answer
    next_question["explanation"] = _trim_text(next_question.get("explanation") or answer_text, 70)
    return next_question


def _normalize_single_llm_question(
    candidate: dict,
    question_type: str,
    transcript_text: str,
    type_index: int,
) -> dict | None:
    """LLM이 만든 의미 단위 응답을 프론트가 풀 수 있는 퀴즈 JSON으로 변환합니다."""
    next_question = dict(candidate)
    next_question["type"] = question_type

    if question_type == "MULTIPLE_CHOICE":
        direct_question = _normalize_llm_mc_direct(next_question, transcript_text, type_index)
        if direct_question is not None:
            return direct_question

        raw_options = next_question.get("options") if isinstance(next_question.get("options"), list) else []
        raw_distractors = next_question.get("distractors") if isinstance(next_question.get("distractors"), list) else []
        answer_text = (
            next_question.get("answer")
            or next_question.get("correct_answer")
            or next_question.get("정답")
            or ""
        )
        if not answer_text and raw_options:
            answer_text = raw_options[0]

        distractors: list[str] = []
        for value in [*raw_distractors, *raw_options]:
            if _option_key(value) != _option_key(answer_text):
                distractors.append(str(value))
        return _numbered_mc_question(next_question, str(answer_text), distractors, transcript_text, type_index)

    if question_type == "OX":
        if _OX_INTERROGATIVE_RE.search(str(next_question.get("question") or "")):
            source_sentence = _best_source_sentence_for_question(str(next_question.get("question") or ""), transcript_text, type_index)
            next_question["question"] = source_sentence or next_question.get("question") or ""
        next_question["options"] = ["O", "X"]
        next_question = _coerce_correct_answer(next_question)
        next_question["question"] = _trim_text(next_question.get("question") or "", 90)
        next_question["explanation"] = _trim_text(next_question.get("explanation") or "", 70)
        return next_question

    next_question["options"] = []
    next_question = _coerce_correct_answer(next_question)
    question_text = _clean_quiz_fragment(next_question.get("question") or "")
    answer_text = _clean_quiz_fragment(next_question.get("correct_answer") or next_question.get("answer") or "")
    source_sentence = _best_source_sentence_for_question(f"{question_text} {answer_text}", transcript_text, type_index)
    fallback_question, fallback_answer = _short_answer_from_sentence(source_sentence, answer_text, type_index)
    if (
        not _is_good_question_text(question_text)
        or _is_generic_short_answer_question(question_text)
        or _short_question_leaks_answer(question_text, answer_text)
    ):
        question_text = fallback_question
    if not answer_text or len(answer_text) > 72 or _has_bad_quiz_artifact(answer_text):
        answer_text = fallback_answer
    next_question["question"] = _clip_text(question_text, 80)
    next_question["correct_answer"] = _clip_text(answer_text, 60)
    next_question["explanation"] = _trim_text(next_question.get("explanation") or next_question["correct_answer"], 70)
    return next_question


def _coerce_correct_answer(question: dict) -> dict:
    """LLM이 정답을 보기 번호나 참/거짓으로 적은 경우 프론트 채점 형식에 맞춥니다."""
    question_type = question.get("type")
    correct_answer = str(question.get("correct_answer") or "").strip()
    options = [str(option).strip() for option in question.get("options", []) if str(option).strip()]

    if question_type == "MULTIPLE_CHOICE":
        question["options"] = options
        if correct_answer in options:
            return question

        answer_key = correct_answer.rstrip(".").rstrip(")").strip().upper()
        for option in options:
            normalized_option = option.strip().upper()
            if normalized_option.startswith(f"{answer_key}.") or normalized_option.startswith(f"{answer_key})"):
                question["correct_answer"] = option
                return question
            if correct_answer and correct_answer in option:
                question["correct_answer"] = option
                return question

        if options:
            question["correct_answer"] = options[0]
        return question

    if question_type == "OX":
        normalized = correct_answer.strip().upper()
        ox_aliases = {
            "O": "O",
            "X": "X",
            "TRUE": "O",
            "FALSE": "X",
            "T": "O",
            "F": "X",
            "YES": "O",
            "NO": "X",
            "참": "O",
            "거짓": "X",
            "맞음": "O",
            "틀림": "X",
            "맞다": "O",
            "틀리다": "X",
        }
        question["options"] = ["O", "X"]
        question["correct_answer"] = ox_aliases.get(normalized, correct_answer)
        return question

    if question_type == "SHORT_ANSWER":
        question["options"] = []
        question["correct_answer"] = correct_answer

    return question


def _is_valid_question_shape(question: dict) -> bool:
    question_type = question.get("type")
    question_text = " ".join(str(question.get("question") or "").split())
    options = question.get("options") if isinstance(question.get("options"), list) else []
    correct_answer = str(question.get("correct_answer") or "").strip()
    option_keys = [_option_key(option) for option in options]

    if not question_text:
        return False
    if question_type == "MULTIPLE_CHOICE":
        return (
            len(options) == 4
            and correct_answer in options
            and len(set(option_keys)) == 4
            and all(option_keys)
            and _is_good_question_text(question_text)
            and not _question_duplicates_option(question_text, options)
            and not _mc_question_leaks_answer(question_text, correct_answer, options)
            and all(_is_good_option_text(option) for option in options)
        )
    if question_type == "OX":
        return options == ["O", "X"] and correct_answer in {"O", "X"} and not _OX_INTERROGATIVE_RE.search(question_text)
    if question_type == "SHORT_ANSWER":
        return (
            not options
            and bool(correct_answer)
            and _is_good_question_text(question_text)
            and not _is_generic_short_answer_question(question_text)
            and not _short_question_leaks_answer(question_text, correct_answer)
        )
    return False


def _build_fallback_question(
    question_type: str,
    type_index: int,
    transcript_text: str,
) -> dict:
    """LLM이 개수/형식을 끝까지 못 맞출 때 선택 소스 문장으로 문항을 보충합니다."""
    sentences = _source_sentences(transcript_text, limit=16)
    if not sentences:
        raise ValueError("퀴즈 생성에 사용할 소스 문장을 찾지 못했습니다.")
    sentence = sentences[type_index % len(sentences)]
    correct_phrase = _source_phrase(sentence, 60)

    if question_type == "MULTIPLE_CHOICE":
        question = {
            "question_index": 0,
            "type": "MULTIPLE_CHOICE",
            "question": _mc_question_from_source(sentence, correct_phrase, type_index),
            "user_answer": None,
            "is_correct": None,
            "explanation": _trim_text(sentence, 50),
        }
        normalized = _numbered_mc_question(question, correct_phrase, [], transcript_text, type_index)
        if normalized is not None:
            return normalized
        direct = _normalize_llm_mc_direct(
            {
                "question_index": 0,
                "type": "MULTIPLE_CHOICE",
                "question": question["question"],
                "answer": correct_phrase,
                "distractors": [
                    *_make_domain_distractors(correct_phrase, sentence, limit=4),
                    *_make_rule_based_distractors(correct_phrase, limit=4),
                    *_make_contrastive_distractors(correct_phrase, limit=4),
                ],
                "explanation": sentence,
            },
            transcript_text,
            type_index,
        )
        if direct is not None:
            return direct
        raise ValueError("LLM이 객관식 보기 4개를 완성하지 못했습니다. 선택 소스 내용을 조금 더 포함해 주세요.")

    if question_type == "OX":
        if type_index % 2 == 1:
            false_statement = _make_false_ox_statement(sentence, transcript_text, type_index)
            return {
                "question_index": 0,
                "type": "OX",
                "question": false_statement,
                "options": ["O", "X"],
                "correct_answer": "X",
                "user_answer": None,
                "is_correct": None,
                "explanation": _trim_text(sentence, 50),
            }
        return {
            "question_index": 0,
            "type": "OX",
            "question": _trim_text(sentence.rstrip(".。") + ".", 90),
            "options": ["O", "X"],
            "correct_answer": "O",
            "user_answer": None,
            "is_correct": None,
            "explanation": _trim_text(sentence, 50),
        }

    question_text, answer_text = _short_answer_from_sentence(sentence, correct_phrase, type_index)
    return {
        "question_index": 0,
        "type": "SHORT_ANSWER",
        "question": _clip_text(question_text, 80),
        "options": [],
        "correct_answer": _clip_text(answer_text, 60),
        "user_answer": None,
        "is_correct": None,
        "explanation": _trim_text(sentence, 50),
    }


def _fit_quiz_to_expected_counts(
    quiz_data: list[dict],
    expected_counts: dict[str, int],
    transcript_text: str,
) -> list[dict]:
    """재작성 후에도 문항 수/형식이 틀리면 유효 문항만 살리고 부족분을 보충합니다."""
    buckets = {key: [] for key in QUIZ_TYPE_KEYS}
    seen_questions = set()
    seen_concepts: list[str] = []

    for question in quiz_data:
        next_question = _coerce_correct_answer(dict(question))
        question_type = next_question.get("type")
        question_text = " ".join(str(next_question.get("question") or "").split())
        if question_type not in buckets or question_text in seen_questions:
            continue
        if not _is_valid_question_shape(next_question):
            continue
        if _same_concept_exists(next_question, seen_concepts):
            continue
        seen_questions.add(question_text)
        seen_concepts.append(
            " ".join(
                str(value or "")
                for value in (
                    next_question.get("question"),
                    next_question.get("correct_answer"),
                    next_question.get("explanation"),
                )
            )
        )
        buckets[question_type].append(next_question)

    fitted = []
    for question_type in QUIZ_TYPE_KEYS:
        expected = expected_counts.get(question_type, 0)
        selected = buckets[question_type][:expected]
        while len(selected) < expected:
            fallback_index = len(fitted) + len(selected)
            fallback = _build_fallback_question(question_type, fallback_index, transcript_text)
            if _same_concept_exists(fallback, seen_concepts):
                for offset in range(1, 12):
                    candidate = _build_fallback_question(question_type, fallback_index + offset, transcript_text)
                    if not _same_concept_exists(candidate, seen_concepts):
                        fallback = candidate
                        break
            fallback_text = " ".join(str(fallback.get("question") or "").split())
            if fallback_text in seen_questions:
                if question_type == "MULTIPLE_CHOICE":
                    source_hint = str(fallback.get("explanation") or "")
                    answer_hint = _strip_option_prefix(str(fallback.get("correct_answer") or ""))
                    for offset in range(1, 6):
                        candidate_text = _mc_question_from_source(source_hint, answer_hint, fallback_index + offset)
                        if candidate_text not in seen_questions:
                            fallback["question"] = candidate_text
                            break
                if " ".join(str(fallback.get("question") or "").split()) in seen_questions:
                    fallback["question"] = f"{fallback_text} - 추가 확인"
            seen_questions.add(" ".join(str(fallback.get("question") or "").split()))
            seen_concepts.append(
                " ".join(
                    str(value or "")
                    for value in (
                        fallback.get("question"),
                        fallback.get("correct_answer"),
                        fallback.get("explanation"),
                    )
                )
            )
            selected.append(fallback)
        fitted.extend(selected)

    for index, question in enumerate(fitted, start=1):
        question["question_index"] = index
        question.setdefault("user_answer", None)
        question.setdefault("is_correct", None)
        question.setdefault("explanation", "")

    return fitted


def _validate_quiz_quality(quiz_data: list[dict], expected_counts: dict[str, int]) -> list[str]:
    """생성된 퀴즈가 프론트에서 풀 수 있는 형태인지 검증합니다."""
    issues = []
    seen_questions = set()
    actual_counts = count_quiz_types(quiz_data)

    for key, expected in expected_counts.items():
        if actual_counts.get(key, 0) != expected:
            issues.append(f"{key} 문항 수가 요청({expected})과 다릅니다: {actual_counts.get(key, 0)}개")

    for index, question in enumerate(quiz_data, start=1):
        question_type = question.get("type")
        question_text = " ".join(str(question.get("question") or "").split())
        options = question.get("options") if isinstance(question.get("options"), list) else []
        correct_answer = str(question.get("correct_answer") or "").strip()

        if not question_text:
            issues.append(f"{index}번 문항 질문이 비어 있습니다.")
        elif question_text in seen_questions:
            issues.append(f"{index}번 문항이 이전 문항과 중복됩니다.")
        seen_questions.add(question_text)

        if question_type == "MULTIPLE_CHOICE":
            option_keys = [_option_key(option) for option in options]
            if not _is_good_question_text(question_text):
                issues.append(f"{index}번 객관식 문항의 질문이 구체적인 개념 질문이 아닙니다.")
            if _question_duplicates_option(question_text, options):
                issues.append(f"{index}번 객관식 문항의 질문이 보기 문장과 중복됩니다.")
            if len(options) != 4:
                issues.append(f"{index}번 객관식 문항의 보기가 4개가 아닙니다.")
            if len(set(option_keys)) != len(option_keys):
                issues.append(f"{index}번 객관식 문항에 중복 보기가 있습니다.")
            if correct_answer not in options:
                issues.append(f"{index}번 객관식 문항의 correct_answer가 options 중 하나와 일치하지 않습니다.")
            for option in options:
                if not _is_good_option_text(option):
                    issues.append(f"{index}번 객관식 문항에 학습 내용이 아닌 보기가 포함되어 있습니다.")
                    break
        elif question_type == "OX":
            if options != ["O", "X"]:
                issues.append(f"{index}번 O/X 문항의 options가 ['O', 'X']가 아닙니다.")
            if correct_answer not in {"O", "X"}:
                issues.append(f"{index}번 O/X 문항의 correct_answer가 O 또는 X가 아닙니다.")
            if _OX_INTERROGATIVE_RE.search(question_text):
                issues.append(f"{index}번 O/X 문항이 참/거짓 평서문이 아니라 의문문입니다.")
        elif question_type == "SHORT_ANSWER":
            if options:
                issues.append(f"{index}번 단답형 문항의 options는 비어 있어야 합니다.")
            if not correct_answer:
                issues.append(f"{index}번 단답형 문항의 correct_answer가 비어 있습니다.")
            if not _is_good_question_text(question_text):
                issues.append(f"{index}번 단답형 문항의 질문이 구체적인 지식 질문이 아닙니다.")
            if _is_generic_short_answer_question(question_text):
                issues.append(f"{index}번 단답형 문항이 일반적인 핵심 내용 질문으로 반복됩니다.")
            if _short_question_leaks_answer(question_text, correct_answer):
                issues.append(f"{index}번 단답형 문항의 질문이 정답을 포함합니다.")
        else:
            issues.append(f"{index}번 문항 타입이 지원되지 않습니다: {question_type}")

    return issues


def _valid_questions_by_type(quiz_data: list[dict]) -> dict[str, list[dict]]:
    """유효한 문항만 유형별로 묶고 중복 질문은 제거합니다."""
    buckets = {key: [] for key in QUIZ_TYPE_KEYS}
    seen_questions = set()
    for question in quiz_data:
        next_question = _coerce_correct_answer(dict(question))
        question_type = next_question.get("type")
        question_text = " ".join(str(next_question.get("question") or "").split())
        if question_type not in buckets or not question_text or question_text in seen_questions:
            continue
        if not _is_valid_question_shape(next_question):
            continue
        seen_questions.add(question_text)
        buckets[question_type].append(next_question)
    return buckets


def _flatten_quiz_buckets(
    buckets: dict[str, list[dict]],
    expected_counts: dict[str, int],
) -> list[dict]:
    fitted = []
    for question_type in QUIZ_TYPE_KEYS:
        fitted.extend(buckets.get(question_type, [])[: expected_counts.get(question_type, 0)])
    for index, question in enumerate(fitted, start=1):
        question["question_index"] = index
        question.setdefault("user_answer", None)
        question.setdefault("is_correct", None)
        question.setdefault("explanation", "")
    return fitted


def _quiz_generation_payload(messages: list[dict], max_tokens: int, *, temperature: float = 0.0) -> dict:
    return {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "bridgeprag_alpha": 0.0,
        "bridgeprag_disable_memory": True,
        "demo_feature": "quiz",
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _type_instruction(question_type: str) -> str:
    if question_type == "MULTIPLE_CHOICE":
        return (
            "MULTIPLE_CHOICE 객관식만 생성하세요. "
            "question은 특정 개념/정의/특징을 직접 묻고, answer와 distractors 3개는 모두 선택 소스의 학습 내용에서 만든 지식 문장으로 작성하세요. "
            "\"자료에서 확인할 수 없는\", \"반대되는 방식\", \"관련이 있다/없다\" 같은 메타 보기는 금지합니다."
        )
    if question_type == "OX":
        return (
            "OX 문항만 생성하세요. question은 참/거짓 판단이 가능한 평서문이어야 하고, "
            "options는 [\"O\", \"X\"], correct_answer는 \"O\" 또는 \"X\"입니다."
        )
    return (
        "SHORT_ANSWER 단답형만 생성하세요. options는 반드시 []이고 correct_answer는 짧은 핵심 답안입니다. "
        "\"핵심 내용을 쓰세요\"처럼 포괄적으로 묻지 말고 정의, 값, 조건, 관계, 결과 중 하나를 구체적으로 물으세요."
    )


def _build_type_specific_messages(
    transcript_text: str,
    question_type: str,
    count: int,
    existing_questions: list[str],
) -> list[dict]:
    existing_block = "\n".join(f"- {item}" for item in existing_questions[:12]) or "- 없음"
    user_prompt = f"""아래 선택 소스 내용을 읽고 {question_type} 유형 퀴즈를 정확히 {count}개 생성하세요.

[선택 소스]
{transcript_text[:QUIZ_CONTEXT_CHARS]}

[이미 사용한 질문]
{existing_block}

[생성 규칙]
- {_type_instruction(question_type)}
- 다른 유형의 문항은 절대 만들지 마세요.
- 제공된 선택 소스에 있는 내용만 사용하세요.
- 이미 사용한 질문과 중복되거나 거의 같은 질문은 만들지 마세요.
- 객관식은 특정 개념을 묻고, 정답과 오답 모두 학습 내용이 담긴 짧은 설명으로 쓰세요.
- "자료에서 확인할 수 없는 별도의 개념이다"처럼 형식만 맞춘 보기는 만들지 마세요.
- question은 60자 이내, option은 35자 이내, explanation은 50자 이내로 짧게 쓰세요.
- JSON 배열 외 텍스트는 쓰지 마세요.

반드시 아래처럼 JSON 배열만 응답하세요:
[
  {{
    "question_index": 1,
    "type": "{question_type}",
    "question": "질문",
    "options": [],
    "correct_answer": "정답",
    "explanation": "짧은 해설"
  }}
]"""
    return [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _build_single_question_messages(
    transcript_text: str,
    question_type: str,
    existing_questions: list[str],
    type_index: int,
) -> list[dict]:
    """소형 LLM 안정성을 위해 퀴즈를 한 문항씩 생성하는 프롬프트를 만듭니다."""
    existing_block = "\n".join(f"- {item}" for item in existing_questions[:20]) or "- 없음"
    source_sentences = _source_sentences(transcript_text, limit=20)
    focus_sentence = source_sentences[type_index % len(source_sentences)] if source_sentences else ""
    if question_type == "MULTIPLE_CHOICE":
        option_rule = (
            "answer에는 정답이 되는 짧은 핵심 설명을 쓰고, distractors에는 같은 주제권에서 헷갈릴 수 있지만 정답과 다른 오답 후보 3개를 쓰세요. "
    "모든 distractor는 선택 소스의 개념을 바탕으로 한 지식 문장이어야 하며, 보기 번호와 options/correct_answer는 쓰지 마세요. "
    "\"관련이 있다/없다\"처럼 관계 여부만 말하는 보기는 금지합니다."
        )
        response_example = """{
  "question_index": 1,
  "type": "MULTIPLE_CHOICE",
  "question": "질문",
  "answer": "짧은 정답",
  "distractors": ["짧은 오답1", "짧은 오답2", "짧은 오답3"],
  "explanation": "짧은 해설"
}"""
    elif question_type == "OX":
        option_rule = "options는 정확히 [\"O\", \"X\"]이고 correct_answer는 \"O\" 또는 \"X\"입니다."
        response_example = """{
  "question_index": 1,
  "type": "OX",
  "question": "참/거짓을 판단할 수 있는 평서문",
  "options": ["O", "X"],
  "correct_answer": "O",
  "explanation": "짧은 해설"
}"""
    else:
        option_rule = (
            "options는 정확히 []이고 correct_answer는 짧은 핵심 답안입니다. "
            "질문은 정의/값/조건/관계/결과 중 하나를 직접 물어야 하며, 이전 단답형과 다른 개념을 물어야 합니다."
        )
        response_example = """{
  "question_index": 1,
  "type": "SHORT_ANSWER",
  "question": "질문",
  "options": [],
  "correct_answer": "짧은 정답",
  "explanation": "짧은 해설"
}"""

    user_prompt = f"""아래 선택 소스 요약본에서 {question_type} 퀴즈 1개만 생성하세요.

[선택 소스 요약본]
{transcript_text[:QUIZ_CONTEXT_CHARS]}

[이번 문항에서 우선 참고할 문장]
{focus_sentence or "선택 소스 전체"}

[이미 만든 질문]
{existing_block}

[규칙]
- 반드시 {question_type} 유형 1개만 생성하세요.
- {_type_instruction(question_type)}
- {option_rule}
- 선택 소스 요약본에 실제로 나온 내용만 사용하세요.
- 이번 문항에서 우선 참고할 문장을 중심으로 만들되, 필요하면 요약본의 다른 문장도 참고하세요.
- 슬라이드 제목이나 목차 조각을 그대로 질문/보기로 쓰지 말고, 개념을 묻는 퀴즈 문장으로 바꾸세요.
- 객관식 질문은 특정 개념명을 포함하세요. "강의 내용과 가장 일치하는 설명은?" 같은 포괄 질문은 금지합니다.
    - 객관식 보기에는 "자료에서 확인할 수 없는", "자료의 설명과 반대", "관련이 있다/없다" 같은 메타 문장을 넣지 마세요.
- 이미 만든 질문과 중복되거나 거의 같은 질문은 만들지 마세요.
- question은 60자 이내, option은 35자 이내, explanation은 50자 이내로 짧게 쓰세요.
- JSON 객체 1개만 응답하세요. JSON 외 텍스트는 쓰지 마세요.

응답 형식:
{response_example}"""
    return [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


async def _request_quiz_json(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int,
    *,
    temperature: float = 0.0,
) -> list[dict]:
    res = await client.post(
        f"{LLM_URL}/v1/chat/completions",
        json=_quiz_generation_payload(messages, max_tokens, temperature=temperature),
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
    )
    res.raise_for_status()
    data = res.json()
    raw_answer = data["choices"][0]["message"]["content"]
    logger.info(f"[QUIZ] LLM JSON 응답 수신: {len(raw_answer)} chars")
    return _parse_quiz_json(raw_answer)


def _same_question_exists(question: dict, existing_questions: list[str]) -> bool:
    text = re.sub(r"\s+", "", str(question.get("question") or "")).strip()
    if not text:
        return True
    return any(text == re.sub(r"\s+", "", item).strip() for item in existing_questions)


def _question_shape_issues(question: dict) -> list[str]:
    question_type = question.get("type")
    options = question.get("options") if isinstance(question.get("options"), list) else []
    correct_answer = str(question.get("correct_answer") or "").strip()
    issues = []
    if not str(question.get("question") or "").strip():
        issues.append("empty_question")
    if question_type == "MULTIPLE_CHOICE":
        option_keys = [_option_key(option) for option in options]
        if len(options) != 4:
            issues.append(f"options={len(options)}")
        if len(set(option_keys)) != len(option_keys):
            issues.append("duplicate_options")
        if correct_answer not in options:
            issues.append("answer_not_in_options")
    elif question_type == "OX":
        if options != ["O", "X"]:
            issues.append("bad_ox_options")
        if correct_answer not in {"O", "X"}:
            issues.append("bad_ox_answer")
    elif question_type == "SHORT_ANSWER":
        if options:
            issues.append("short_answer_has_options")
        if not correct_answer:
            issues.append("empty_short_answer")
    else:
        issues.append(f"bad_type={question_type}")
    return issues


async def _generate_single_question(
    client: httpx.AsyncClient,
    transcript_text: str,
    question_type: str,
    type_index: int,
    existing_questions: list[str],
) -> dict:
    """한 문항씩 LLM 생성하고, 실패 시 선택 소스 기반 문항으로 복구합니다."""
    max_tokens = min(LLM_MAX_TOKENS, 760)
    for attempt in range(2):
        try:
            _demo_log(
                f"4) {question_type} 문항 LLM 전달: index={type_index + 1}, "
                f"attempt={attempt + 1}, context_chars={len(transcript_text)}"
            )
            candidates = await _request_quiz_json(
                client,
                _build_single_question_messages(transcript_text, question_type, existing_questions, type_index),
                max_tokens,
                temperature=0.0,
            )
            rejected_reasons: list[str] = []
            rejected_sample: dict | None = None
            for candidate in candidates:
                candidate = dict(candidate)
                if _normalize_quiz_type(candidate.get("type")) != question_type:
                    rejected_reasons.append("type_mismatch")
                    rejected_sample = rejected_sample or candidate
                    continue
                normalized_candidate = _normalize_single_llm_question(candidate, question_type, transcript_text, type_index)
                if normalized_candidate is None:
                    rejected_reasons.append("normalize_failed")
                    rejected_sample = rejected_sample or candidate
                    continue
                if _same_question_exists(normalized_candidate, existing_questions):
                    rejected_reasons.append("duplicate_question")
                    rejected_sample = rejected_sample or normalized_candidate
                    continue
                if _same_concept_exists(normalized_candidate, existing_questions):
                    rejected_reasons.append("duplicate_concept")
                    rejected_sample = rejected_sample or normalized_candidate
                    continue
                if _is_valid_question_shape(normalized_candidate):
                    _demo_log(
                        f"5) {question_type} 문항 검증 성공: "
                        f"question='{_preview(normalized_candidate.get('question'), 100)}'"
                    )
                    return normalized_candidate
                rejected_reasons.extend(_question_shape_issues(normalized_candidate))
                rejected_sample = rejected_sample or normalized_candidate
            sample = rejected_sample or (candidates[0] if candidates else {})
            logger.debug(
                "[QUIZ] %s 문항 후보 재생성: attempt=%s reasons=%s sample=%s",
                question_type,
                attempt + 1,
                rejected_reasons or _question_shape_issues(sample if isinstance(sample, dict) else {}),
                json.dumps(sample, ensure_ascii=False)[:400] if isinstance(sample, dict) else str(sample)[:400],
            )
        except QuizParseError as exc:
            recovered = _plain_text_to_candidate(exc.raw_text, question_type, transcript_text, type_index)
            if (
                recovered is not None
                and not _same_question_exists(recovered, existing_questions)
                and not _same_concept_exists(recovered, existing_questions)
                and _is_valid_question_shape(recovered)
            ):
                logger.info(
                    "[QUIZ] %s JSON이 아닌 LLM 응답을 문항으로 회수했습니다: %s",
                    question_type,
                    recovered.get("question"),
                )
                return recovered
            logger.warning(
                "[QUIZ] %s 자연어 LLM 응답 회수 실패: attempt=%s raw=%s",
                question_type,
                attempt + 1,
                exc.raw_text[:300],
            )
        except Exception as exc:
            logger.warning("[QUIZ] %s 단일 문항 LLM 생성 실패: attempt=%s error=%s", question_type, attempt + 1, exc)

    fallback = _build_fallback_question(question_type, type_index, transcript_text)
    if _same_concept_exists(fallback, existing_questions):
        for offset in range(1, 8):
            candidate = _build_fallback_question(question_type, type_index + offset, transcript_text)
            if not _same_concept_exists(candidate, existing_questions):
                fallback = candidate
                break
    logger.info("[QUIZ] %s 문항을 선택 소스 기반으로 보충했습니다: %s", question_type, fallback.get("question"))
    _demo_log(f"5) {question_type} 문항 보충 생성: question='{_preview(fallback.get('question'), 100)}'")
    return fallback


async def _generate_quiz_one_by_one(
    client: httpx.AsyncClient,
    transcript_text: str,
    expected_counts: dict[str, int],
) -> list[dict]:
    """요청한 유형/개수대로 한 문항씩 생성하여 전체 실패 가능성을 낮춥니다."""
    quiz_data: list[dict] = []
    existing_questions: list[str] = []

    for question_type in QUIZ_TYPE_KEYS:
        for type_index in range(expected_counts.get(question_type, 0)):
            question = await _generate_single_question(
                client,
                transcript_text,
                question_type,
                type_index,
                existing_questions,
            )
            quiz_data.append(question)
            existing_questions.append(
                " ".join(
                    str(value or "")
                    for value in (
                        question.get("question"),
                        question.get("correct_answer"),
                        question.get("explanation"),
                    )
                )
            )

    for index, question in enumerate(quiz_data, start=1):
        question["question_index"] = index
        question.setdefault("user_answer", None)
        question.setdefault("is_correct", None)
        question.setdefault("explanation", "")

    issues = _validate_quiz_quality(quiz_data, expected_counts)
    if issues:
        logger.warning("[QUIZ] 단일 문항 생성 후 보정 필요: %s", "; ".join(issues[:5]))
        quiz_data = _fit_quiz_to_expected_counts(quiz_data, expected_counts, transcript_text)
    return quiz_data


async def _complete_expected_counts_with_llm(
    client: httpx.AsyncClient,
    transcript_text: str,
    quiz_data: list[dict],
    expected_counts: dict[str, int],
    max_tokens: int,
) -> list[dict]:
    """초기 LLM 응답에서 부족한 유형을 유형별 LLM 호출로 보강합니다."""
    buckets = _valid_questions_by_type(quiz_data)
    for question_type in QUIZ_TYPE_KEYS:
        expected = expected_counts.get(question_type, 0)
        if expected <= 0:
            buckets[question_type] = []
            continue
        buckets[question_type] = buckets.get(question_type, [])[:expected]
        missing = expected - len(buckets[question_type])
        if missing <= 0:
            continue

        existing_questions = [
            str(item.get("question") or "")
            for items in buckets.values()
            for item in items
            if item.get("question")
        ]
        logger.info("[QUIZ] %s 부족 문항 LLM 보강: %s개", question_type, missing)
        try:
            supplemental = await _request_quiz_json(
                client,
                _build_type_specific_messages(transcript_text, question_type, missing, existing_questions),
                min(LLM_MAX_TOKENS, max(700, 450 + missing * 260)),
                temperature=0.0,
            )
            for item in supplemental:
                item["type"] = question_type
            new_bucket = _valid_questions_by_type([*buckets[question_type], *supplemental])[question_type]
            buckets[question_type] = new_bucket[:expected]
        except Exception as exc:
            logger.warning("[QUIZ] %s 부족 문항 LLM 보강 실패: %s", question_type, exc)

    completed = _flatten_quiz_buckets(buckets, expected_counts)
    issues = _validate_quiz_quality(completed, expected_counts)
    if not issues:
        return completed

    logger.warning("[QUIZ] LLM 보강 후에도 부족, 소스 기반 보충 사용: %s", "; ".join(issues[:5]))
    return _fit_quiz_to_expected_counts(completed, expected_counts, transcript_text)


def _parse_quiz_json(raw_text: str) -> list[dict]:
    """
    LLM 응답에서 JSON 배열만 추출하여 파싱.
    <think>...</think> 태그, 마크다운 코드블록 등을 제거한 후 파싱 시도.
    """
    text = _strip_json_code_fences(_remove_thinking_blocks(raw_text))
    candidates = [text, *_extract_balanced_json_candidates(text)]
    last_error = None

    for candidate in candidates:
        candidate = _normalize_json_candidate(candidate)
        if not candidate:
            continue
        try:
            return _coerce_quiz_questions(json.loads(candidate))
        except json.JSONDecodeError as exc:
            last_error = exc
        except ValueError as exc:
            last_error = exc

    logger.error(f"퀴즈 JSON 파싱 실패: {last_error}\n원본: {raw_text[:500]}")
    raise QuizParseError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {last_error}", raw_text)


#  퀴즈 생성 (LLM 호출)
async def generate_quiz(
    transcript_text: str,
    num_questions: int = 5,
    type_counts: dict[str, int] | None = None,
) -> list[dict]:
    """
    전사문 텍스트를 기반으로 퀴즈를 생성합니다.

    Args:
        transcript_text: 강의 전사문 전체 텍스트
        num_questions: 생성할 문제 수 (기본 5)
        type_counts: 문제 유형별 개수. 없으면 기본 배분을 사용.

    Returns:
        quiz_data: 퀴즈 JSON 배열 (JSONB에 저장될 형태)
    """
    normalized_counts = _normalize_type_counts(num_questions, type_counts)
    total_questions = sum(normalized_counts.values())

    if MOCK_MODE:
        logger.info("[QUIZ] MOCK_MODE: 목업 퀴즈 데이터 반환")
        return _generate_mock_quiz(total_questions, normalized_counts)

    try:
        _demo_log(
            f"퀴즈 생성 시작: requested={total_questions}, counts={normalized_counts}, "
            f"input_chars={len(transcript_text or '')}"
        )
        _log_sentence_morphemes(
            _source_sentences(transcript_text, limit=14),
            label="퀴즈 원문",
        )
        logger.info(
            "[QUIZ] 단일 문항 반복 생성 시작: total=%s counts=%s",
            total_questions,
            normalized_counts,
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(6.0, read=60.0)) as client:
            quiz_source_summary = await _build_quiz_source_summary(client, transcript_text)
            quiz_data = await _generate_quiz_one_by_one(client, quiz_source_summary, normalized_counts)

        quality_issues = _validate_quiz_quality(quiz_data, normalized_counts)
        if quality_issues:
            logger.warning("[QUIZ] 최종 품질 검증 실패, 소스 기반 보정 재적용: %s", "; ".join(quality_issues[:5]))
            quiz_data = _fit_quiz_to_expected_counts(quiz_data, normalized_counts, transcript_text)
            quality_issues = _validate_quiz_quality(quiz_data, normalized_counts)
            if quality_issues:
                raise ValueError("퀴즈 문항 형식이 올바르지 않습니다: " + "; ".join(quality_issues[:5]))

        logger.info(f"[QUIZ] {len(quiz_data)}개 문제 파싱 완료")
        _demo_log(f"6) 퀴즈 생성 완료: total={len(quiz_data)}")
        for item in quiz_data[:8]:
            _demo_log(
                f"   Q{item.get('question_index')}: type={item.get('type')} "
                f"question='{_preview(item.get('question'), 100)}' "
                f"answer='{_preview(item.get('correct_answer'), 80)}'"
            )
            if item.get("type") == "MULTIPLE_CHOICE":
                _demo_log(f"      options={[ _preview(option, 60) for option in item.get('options', []) ]}")
            elif item.get("type") == "OX":
                _demo_log(f"      options={item.get('options', [])}")
        return quiz_data

    except httpx.HTTPError as e:
        logger.warning("[QUIZ] LLM 서버 연결 실패, 선택 소스 기반 문항으로 생성합니다: %s", e)
        quiz_source_summary = _local_quiz_source_summary(transcript_text)
        quiz_data = _fit_quiz_to_expected_counts([], normalized_counts, quiz_source_summary)
        logger.info(f"[QUIZ] 소스 기반 {len(quiz_data)}개 문제 생성 완료")
        return quiz_data
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"[QUIZ] 퀴즈 생성 중 예상치 못한 오류: {e}")
        raise RuntimeError(f"퀴즈 생성 실패: {e}")


#  채점 로직
def grade_quiz(quiz_data: list[dict], answers: dict[str, str]) -> tuple[list[dict], int]:
    """
    사용자 답안을 채점합니다.

    Args:
        quiz_data: 기존 퀴즈 데이터 (DB에서 조회한 quiz_data)
        answers: { "1": "사용자답", "2": "사용자답", ... }
                 key = question_index (문자열), value = 사용자 응답

    Returns:
        (updated_quiz_data, correct_count)
        - updated_quiz_data: user_answer, is_correct 필드가 채워진 quiz_data
        - correct_count: 정답 개수
    """
    correct_count = 0

    for q in quiz_data:
        idx_str = str(q["question_index"])
        user_answer = answers.get(idx_str)

        if user_answer is None:
            q["user_answer"] = None
            q["is_correct"] = False
            continue

        q["user_answer"] = user_answer

        # 정답 비교: 공백/대소문자 무시, 양쪽 공백 제거
        correct = q.get("correct_answer", "").strip()
        submitted = user_answer.strip()

        if q["type"] == "SHORT_ANSWER":
            # 단답형: 정답에 핵심 키워드가 포함되면 정답 처리
            is_correct = _fuzzy_match(correct, submitted)
        else:
            # 객관식/OX: 정확히 일치
            is_correct = correct == submitted

        q["is_correct"] = is_correct
        if is_correct:
            correct_count += 1

    return quiz_data, correct_count


def _fuzzy_match(correct: str, submitted: str) -> bool:
    """
    단답형 퍼지 매칭:
    - 정확히 일치하면 True
    - 정답이 사용자 답에 포함되거나, 사용자 답이 정답에 포함되면 True
    - 숫자 등 핵심 키워드 비교
    """
    c = correct.replace(" ", "").lower()
    s = submitted.replace(" ", "").lower()

    if c == s:
        return True
    if c in s or s in c:
        return True

    return False


#  목업 데이터 (LLM 미연결 시)
def _generate_mock_quiz(
    num_questions: int = 5,
    type_counts: dict[str, int] | None = None,
) -> list[dict]:
    """프론트엔드 개발/테스트용 목업 퀴즈 데이터"""
    counts = _normalize_type_counts(num_questions, type_counts)
    mc = counts["MULTIPLE_CHOICE"]
    ox = counts["OX"]
    sa = counts["SHORT_ANSWER"]
    questions = []
    idx = 1

    # 객관식
    mock_mc = [
        {
            "question": "데이터베이스 정규화의 주된 목적은 무엇인가?",
            "options": ["1. 데이터 중복 제거", "2. 처리 속도 향상", "3. 데이터 백업", "4. 네트워크 최적화"],
            "correct_answer": "1. 데이터 중복 제거",
            "explanation": "정규화는 데이터의 중복을 제거하고 이상 현상(anomaly)을 방지하기 위한 과정입니다.",
        },
        {
            "question": "SQL에서 데이터를 조회하는 명령어는?",
            "options": ["1. INSERT", "2. SELECT", "3. UPDATE", "4. DELETE"],
            "correct_answer": "2. SELECT",
            "explanation": "SELECT문은 데이터베이스에서 원하는 데이터를 조회하는 데 사용됩니다.",
        },
        {
            "question": "인덱스(Index)를 사용하는 주된 이유는?",
            "options": ["1. 데이터 무결성 보장", "2. 검색 속도 향상", "3. 저장 공간 절약", "4. 보안 강화"],
            "correct_answer": "2. 검색 속도 향상",
            "explanation": "인덱스는 테이블의 검색 속도를 높이기 위한 자료구조입니다.",
        },
    ]
    for i in range(mc):
        m = mock_mc[i % len(mock_mc)]
        questions.append({
            "question_index": idx,
            "type": "MULTIPLE_CHOICE",
            "user_answer": None,
            "is_correct": None,
            **m,
        })
        idx += 1

    # OX
    mock_ox = [
        {
            "question": "PostgreSQL은 NoSQL 데이터베이스이다.",
            "options": ["O", "X"],
            "correct_answer": "X",
            "explanation": "PostgreSQL은 대표적인 관계형 데이터베이스(RDBMS)입니다.",
        },
        {
            "question": "트랜잭션의 ACID 속성 중 'I'는 Isolation(격리성)을 의미한다.",
            "options": ["O", "X"],
            "correct_answer": "O",
            "explanation": "ACID는 Atomicity, Consistency, Isolation, Durability의 약자입니다.",
        },
    ]
    for i in range(ox):
        m = mock_ox[i % len(mock_ox)]
        questions.append({
            "question_index": idx,
            "type": "OX",
            "user_answer": None,
            "is_correct": None,
            **m,
        })
        idx += 1

    # 단답형
    mock_sa = [
        {
            "question": "모든 릴레이션이 원자값만을 가지도록 하는 정규형은?",
            "options": [],
            "correct_answer": "제1정규형",
            "explanation": "도메인이 원자값이어야 한다는 것은 제1정규형(1NF)의 정의입니다.",
        },
        {
            "question": "기본 키(Primary Key)가 만족해야 하는 두 가지 제약조건은 유일성과 무엇인가?",
            "options": [],
            "correct_answer": "최소성",
            "explanation": "기본 키는 유일성(uniqueness)과 최소성(minimality)을 만족해야 합니다.",
        },
    ]
    for i in range(sa):
        m = mock_sa[i % len(mock_sa)]
        questions.append({
            "question_index": idx,
            "type": "SHORT_ANSWER",
            "user_answer": None,
            "is_correct": None,
            **m,
        })
        idx += 1

    return questions[:num_questions]
