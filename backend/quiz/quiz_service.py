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
  - generate_quiz()는 목업 데이터를 반환 (MOCK_MODE=True)
  - 프론트/LLM 연결 시 MOCK_MODE=False로 전환하면 즉시 실서비스 전환
"""
import json
import re
import uuid
import logging
import httpx
import os

logger = logging.getLogger(__name__)

# ── 설정 ──
# LLM 서버가 준비되면 False로 변경
MOCK_MODE = os.getenv("QUIZ_MOCK_MODE", "true").lower() == "true"

# vLLM OpenAI 호환 API (main.py와 동일한 설정)
LLM_URL = os.getenv("LLM_URL", "http://localhost:8001")
LLM_MODEL = os.getenv("LLM_MODEL", "QuantTrio/Qwen3.5-4B-AWQ")
LLM_API_KEY = os.getenv("LLM_API_KEY", "test-key")


# ══════════════════════════════════════
#  LLM 프롬프트 템플릿
# ══════════════════════════════════════
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


def _build_quiz_prompt(transcript_text: str, num_questions: int = 5) -> list[dict]:
    """LLM에 보낼 messages 배열 구성"""
    mc, ox, sa = _calculate_type_distribution(num_questions)
    user_prompt = QUIZ_USER_PROMPT_TEMPLATE.format(
        transcript_text=transcript_text[:6000],  # 토큰 제한 고려
        num_questions=num_questions,
        mc_count=mc,
        ox_count=ox,
        sa_count=sa,
    )
    return [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_quiz_json(raw_text: str) -> list[dict]:
    """
    LLM 응답에서 JSON 배열만 추출하여 파싱.
    <think>...</think> 태그, 마크다운 코드블록 등을 제거한 후 파싱 시도.
    """
    # <think>...</think> 제거
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    # 마크다운 코드블록 제거
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # JSON 배열 추출: 가장 바깥쪽 [ ... ] 를 찾음
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        questions = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"퀴즈 JSON 파싱 실패: {e}\n원본: {raw_text[:500]}")
        raise ValueError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {e}")

    if not isinstance(questions, list):
        raise ValueError("LLM 응답이 JSON 배열이 아닙니다")

    # 각 문제에 user_answer, is_correct 기본값 추가
    for q in questions:
        q.setdefault("user_answer", None)
        q.setdefault("is_correct", None)
        # options가 없으면 빈 배열로
        q.setdefault("options", [])
        q.setdefault("explanation", "")

    return questions


# ══════════════════════════════════════
#  퀴즈 생성 (LLM 호출)
# ══════════════════════════════════════
async def generate_quiz(
    transcript_text: str,
    num_questions: int = 5,
) -> list[dict]:
    """
    전사문 텍스트를 기반으로 퀴즈를 생성합니다.

    Args:
        transcript_text: 강의 전사문 전체 텍스트
        num_questions: 생성할 문제 수 (기본 5)

    Returns:
        quiz_data: 퀴즈 JSON 배열 (JSONB에 저장될 형태)
    """
    if MOCK_MODE:
        logger.info("[QUIZ] MOCK_MODE: 목업 퀴즈 데이터 반환")
        return _generate_mock_quiz(num_questions)

    messages = _build_quiz_prompt(transcript_text, num_questions)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0)) as client:
            res = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.3,  # 정확한 JSON 생성을 위해 낮은 temperature
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            )
            res.raise_for_status()

        data = res.json()
        raw_answer = data["choices"][0]["message"]["content"]
        logger.info(f"[QUIZ] LLM 응답 수신: {len(raw_answer)} chars")

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


# ══════════════════════════════════════
#  채점 로직
# ══════════════════════════════════════
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


# ══════════════════════════════════════
#  목업 데이터 (LLM 미연결 시)
# ══════════════════════════════════════
def _generate_mock_quiz(num_questions: int = 5) -> list[dict]:
    """프론트엔드 개발/테스트용 목업 퀴즈 데이터"""
    mc, ox, sa = _calculate_type_distribution(num_questions)
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
