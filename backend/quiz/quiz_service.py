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

logger = logging.getLogger(__name__)

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

_OX_INTERROGATIVE_RE = re.compile(r"(무엇|어떤|어디|왜|어떻게|입니까|인가요|일까요|까요|[?？])")
_SENTENCE_END_FINDER_RE = re.compile(r".+?(?:다\.|요\.|[.!?。？！])(?:\s+|$)")


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


def _source_sentences(transcript_text: str, *, limit: int = 20) -> list[str]:
    """선택 소스에서 퀴즈 fallback에 사용할 핵심 문장 후보를 추출합니다."""
    cleaned = re.sub(r"\[[^\]]+\]", " ", str(transcript_text or ""))
    cleaned = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    sentences = []
    seen = set()
    parts = [match.group(0) for match in _SENTENCE_END_FINDER_RE.finditer(cleaned)]
    if not parts:
        parts = re.split(r"\n+|[.;!?。？！]", cleaned)
    for part in parts:
        sentence = " ".join(part.split()).strip(" -•·")
        if not 8 <= len(sentence) <= 180:
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
    sentence = " ".join(str(sentence or "").split())
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
    sentence = re.sub(r"^(따라서|그리고|또한|반면|하지만)\s*", "", str(sentence or "").strip())
    sentence = sentence.rstrip(".。")
    return _trim_text(sentence, limit)


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

    if not question_text:
        return False
    if question_type == "MULTIPLE_CHOICE":
        return len(options) >= 3 and correct_answer in options
    if question_type == "OX":
        return options == ["O", "X"] and correct_answer in {"O", "X"} and not _OX_INTERROGATIVE_RE.search(question_text)
    if question_type == "SHORT_ANSWER":
        return not options and bool(correct_answer)
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
    topic = _extract_topic(sentence, f"핵심 개념 {type_index + 1}")
    correct_phrase = _source_phrase(sentence)

    if question_type == "MULTIPLE_CHOICE":
        correct_answer = f"1. {correct_phrase}"
        option_phrases = [correct_phrase]
        for candidate in sentences:
            phrase = _source_phrase(candidate)
            if phrase and phrase not in option_phrases:
                option_phrases.append(phrase)
            if len(option_phrases) >= 4:
                break
        while len(option_phrases) < 4:
            option_phrases.append(f"자료에 없는 설명 {len(option_phrases)}")
        options = [f"{idx}. {phrase}" for idx, phrase in enumerate(option_phrases[:4], start=1)]
        return {
            "question_index": 0,
            "type": "MULTIPLE_CHOICE",
            "question": f"자료에서 '{topic}'에 대해 설명한 내용은?",
            "options": options,
            "correct_answer": correct_answer,
            "user_answer": None,
            "is_correct": None,
            "explanation": _trim_text(sentence, 50),
        }

    if question_type == "OX":
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

    return {
        "question_index": 0,
        "type": "SHORT_ANSWER",
        "question": f"자료에서 '{topic}'에 대해 설명한 핵심 내용을 쓰세요.",
        "options": [],
        "correct_answer": correct_phrase,
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

    for question in quiz_data:
        next_question = _coerce_correct_answer(dict(question))
        question_type = next_question.get("type")
        question_text = " ".join(str(next_question.get("question") or "").split())
        if question_type not in buckets or question_text in seen_questions:
            continue
        if not _is_valid_question_shape(next_question):
            continue
        seen_questions.add(question_text)
        buckets[question_type].append(next_question)

    fitted = []
    for question_type in QUIZ_TYPE_KEYS:
        expected = expected_counts.get(question_type, 0)
        selected = buckets[question_type][:expected]
        while len(selected) < expected:
            selected.append(_build_fallback_question(question_type, len(selected), transcript_text))
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
            if len(options) < 3:
                issues.append(f"{index}번 객관식 문항의 보기가 3개 미만입니다.")
            if correct_answer not in options:
                issues.append(f"{index}번 객관식 문항의 correct_answer가 options 중 하나와 일치하지 않습니다.")
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
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _type_instruction(question_type: str) -> str:
    if question_type == "MULTIPLE_CHOICE":
        return (
            "MULTIPLE_CHOICE 객관식만 생성하세요. "
            "각 문항은 options 4개를 갖고, correct_answer는 options 중 하나와 완전히 같아야 합니다."
        )
    if question_type == "OX":
        return (
            "OX 문항만 생성하세요. question은 참/거짓 판단이 가능한 평서문이어야 하고, "
            "options는 [\"O\", \"X\"], correct_answer는 \"O\" 또는 \"X\"입니다."
        )
    return "SHORT_ANSWER 단답형만 생성하세요. options는 반드시 []이고 correct_answer는 짧은 핵심 답안입니다."


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
) -> list[dict]:
    """소형 LLM 안정성을 위해 퀴즈를 한 문항씩 생성하는 프롬프트를 만듭니다."""
    existing_block = "\n".join(f"- {item}" for item in existing_questions[:20]) or "- 없음"
    if question_type == "MULTIPLE_CHOICE":
        option_rule = (
            "options는 정확히 4개이며, 각 보기는 '1. ...' 형식의 짧은 구입니다. "
            "correct_answer는 options 중 하나와 완전히 같아야 합니다."
        )
        example_options = '["1. 정답 보기", "2. 오답 보기", "3. 오답 보기", "4. 오답 보기"]'
        example_answer = "1. 정답 보기"
    elif question_type == "OX":
        option_rule = "options는 정확히 [\"O\", \"X\"]이고 correct_answer는 \"O\" 또는 \"X\"입니다."
        example_options = '["O", "X"]'
        example_answer = "O"
    else:
        option_rule = "options는 정확히 []이고 correct_answer는 짧은 핵심 답안입니다."
        example_options = "[]"
        example_answer = "정답"

    user_prompt = f"""아래 선택 소스에서 {question_type} 퀴즈 1개만 생성하세요.

[선택 소스]
{transcript_text[:QUIZ_CONTEXT_CHARS]}

[이미 만든 질문]
{existing_block}

[규칙]
- 반드시 {question_type} 유형 1개만 생성하세요.
- {_type_instruction(question_type)}
- {option_rule}
- 선택 소스에 실제로 나온 내용만 사용하세요.
- 이미 만든 질문과 중복되거나 거의 같은 질문은 만들지 마세요.
- question은 60자 이내, option은 35자 이내, explanation은 50자 이내로 짧게 쓰세요.
- JSON 객체 1개만 응답하세요. JSON 외 텍스트는 쓰지 마세요.

응답 형식:
{{
  "question_index": 1,
  "type": "{question_type}",
  "question": "질문",
  "options": {example_options},
  "correct_answer": "{example_answer}",
  "explanation": "짧은 해설"
}}"""
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
            candidates = await _request_quiz_json(
                client,
                _build_single_question_messages(transcript_text, question_type, existing_questions),
                max_tokens,
                temperature=0.0,
            )
            for candidate in candidates:
                candidate = dict(candidate)
                if _normalize_quiz_type(candidate.get("type")) != question_type:
                    continue
                candidate["type"] = question_type
                candidate = _coerce_correct_answer(candidate)
                if _same_question_exists(candidate, existing_questions):
                    continue
                if _is_valid_question_shape(candidate):
                    return candidate
            logger.warning("[QUIZ] %s 단일 문항 검증 실패: attempt=%s", question_type, attempt + 1)
        except Exception as exc:
            logger.warning("[QUIZ] %s 단일 문항 LLM 생성 실패: attempt=%s error=%s", question_type, attempt + 1, exc)

    fallback = _build_fallback_question(question_type, type_index, transcript_text)
    logger.info("[QUIZ] %s 문항을 선택 소스 기반으로 보충했습니다: %s", question_type, fallback.get("question"))
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
            existing_questions.append(str(question.get("question") or ""))

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
    raise ValueError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {last_error}")


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
        logger.info(
            "[QUIZ] 단일 문항 반복 생성 시작: total=%s counts=%s",
            total_questions,
            normalized_counts,
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(6.0, read=60.0)) as client:
            quiz_data = await _generate_quiz_one_by_one(client, transcript_text, normalized_counts)

        quality_issues = _validate_quiz_quality(quiz_data, normalized_counts)
        if quality_issues:
            logger.warning("[QUIZ] 최종 품질 검증 실패, 소스 기반 보정 재적용: %s", "; ".join(quality_issues[:5]))
            quiz_data = _fit_quiz_to_expected_counts(quiz_data, normalized_counts, transcript_text)
            quality_issues = _validate_quiz_quality(quiz_data, normalized_counts)
            if quality_issues:
                raise ValueError("퀴즈 문항 형식이 올바르지 않습니다: " + "; ".join(quality_issues[:5]))

        logger.info(f"[QUIZ] {len(quiz_data)}개 문제 파싱 완료")
        return quiz_data

    except httpx.HTTPError as e:
        logger.warning("[QUIZ] LLM 서버 연결 실패, 선택 소스 기반 문항으로 생성합니다: %s", e)
        quiz_data = _fit_quiz_to_expected_counts([], normalized_counts, transcript_text)
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
