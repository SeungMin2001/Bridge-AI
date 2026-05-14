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
LLM_URL = os.getenv("LLM_URL", "http://localhost:8001")
LLM_MODEL = os.getenv("LLM_MODEL", "QuantTrio/Qwen3.5-4B-AWQ")
LLM_API_KEY = os.getenv("LLM_API_KEY", "test-key")
LLM_MAX_TOKENS = int(os.getenv("QUIZ_MAX_TOKENS", "4096"))

#  LLM 프롬프트 템플릿
QUIZ_SYSTEM_PROMPT = """당신은 대학 강의 내용을 기반으로 학습 퀴즈를 만드는 AI 교수입니다.
반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 텍스트는 절대 포함하지 마세요."""

QUIZ_USER_PROMPT_TEMPLATE = """아래는 강의 전사문입니다:

{transcript_text}

위 강의 내용을 바탕으로 {num_questions}개의 퀴즈를 생성하세요.

문제 유형 배분:
- MULTIPLE_CHOICE (객관식, 3~4지선다): {mc_count}개
- OX (O/X 퀴즈): {ox_count}개
- SHORT_ANSWER (단답형): {sa_count}개

반드시 아래 JSON 배열 형식으로만 응답하세요:
[
  {{
    "question_index": 1,
    "type": "MULTIPLE_CHOICE",
    "question": "질문 텍스트",
    "options": ["1. 보기1", "2. 보기2", "3. 보기3", "4. 보기4"],
    "correct_answer": "1. 보기1",
    "explanation": "해설 텍스트"
  }},
  {{
    "question_index": 2,
    "type": "OX",
    "question": "O/X 질문 텍스트",
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
        transcript_text=transcript_text[:6000],  # 토큰 제한 고려
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
        questions = None
        for key in ("quiz_data", "questions", "quizzes", "items", "data", "result"):
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
        next_question.setdefault("question", "")
        next_question.setdefault("correct_answer", "")
        next_question.setdefault("user_answer", None)
        next_question.setdefault("is_correct", None)
        next_question.setdefault("explanation", "")

        options = next_question.get("options")
        if isinstance(options, str):
            options = [options]
        elif not isinstance(options, list):
            options = []
        if next_question["type"] == "OX" and not options:
            options = ["O", "X"]
        if next_question["type"] == "SHORT_ANSWER":
            options = []
        next_question["options"] = options
        normalized_questions.append(next_question)

    return normalized_questions


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

    messages = _build_quiz_prompt(transcript_text, total_questions, normalized_counts)

    try:
        max_tokens = min(LLM_MAX_TOKENS, max(1400, 400 + (total_questions * 350)))
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0)) as client:
            res = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,  # 정확한 JSON 생성을 위해 낮은 temperature
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            )
            res.raise_for_status()

            data = res.json()
            raw_answer = data["choices"][0]["message"]["content"]
            logger.info(f"[QUIZ] LLM 응답 수신: {len(raw_answer)} chars")

            try:
                quiz_data = _parse_quiz_json(raw_answer)
            except ValueError as first_error:
                logger.warning(f"[QUIZ] LLM 응답 JSON 파싱 실패, 복구 요청 시도: {first_error}")
                repair_res = await client.post(
                    f"{LLM_URL}/v1/chat/completions",
                    json={
                        "model": LLM_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": "입력 텍스트에서 퀴즈 문항만 추출해 유효한 JSON 배열로만 응답하세요.",
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"아래 응답을 {total_questions}개 이하의 퀴즈 JSON 배열로 고쳐주세요. "
                                    "JSON 외 텍스트는 쓰지 마세요.\n\n"
                                    f"{raw_answer[:8000]}"
                                ),
                            },
                        ],
                        "max_tokens": max_tokens,
                        "temperature": 0.0,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                )
                repair_res.raise_for_status()
                repair_data = repair_res.json()
                raw_answer = repair_data["choices"][0]["message"]["content"]
                logger.info(f"[QUIZ] LLM 복구 응답 수신: {len(raw_answer)} chars")
                quiz_data = _parse_quiz_json(raw_answer)

        logger.info(f"[QUIZ] {len(quiz_data)}개 문제 파싱 완료")
        return quiz_data

    except httpx.HTTPError as e:
        logger.error(f"[QUIZ] LLM 호출 실패: {e}")
        raise RuntimeError(f"LLM 서버 연결 실패: {e}")
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
