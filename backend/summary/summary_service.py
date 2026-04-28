"""
요약 서비스 (Summary Service)

파이프라인:
  1. 세션 전사문 조회
  2. kiwipiepy 형태소 분리 → 명사/동사 중심 전처리
  3. TextRank (textrankr) 핵심 문장 추출
  4. LLM에 핵심 문장 전달 → 5가지 템플릿 구조의 요약 생성
  5. SUMMARIES + KEY_SENTENCES 테이블에 저장

요약 응답 JSON 구조:
{
  "core_concepts": ["핵심 개념1", "핵심 개념2"],
  "key_sentences": ["주요 문장1", "주요 문장2"],
  "review_points": ["복습 포인트1", "복습 포인트2"],
  "detailed_explanation": "세부 설명 텍스트",
  "exam_points": ["시험 포인트1", "시험 포인트2"]
}

LLM이 아직 연결되지 않은 경우:
  - MOCK_MODE=True → 목업 요약 데이터 반환
"""
import json
import re
import logging
import httpx
import os

logger = logging.getLogger(__name__)

# ── 설정 ──
MOCK_MODE = os.getenv("SUMMARY_MOCK_MODE", "true").lower() == "true"

LLM_URL = os.getenv("LLM_URL", "http://localhost:8001")
LLM_MODEL = os.getenv("LLM_MODEL", "QuantTrio/Qwen3.5-4B-AWQ")
LLM_API_KEY = os.getenv("LLM_API_KEY", "test-key")


# ══════════════════════════════════════
#  kiwipiepy 형태소 분석
# ══════════════════════════════════════
from kiwipiepy import Kiwi

_kiwi = Kiwi()
_NOUN_TAGS = {"NNG", "NNP"}  # 일반명사, 고유명사
_VERB_TAGS = {"VV", "VA"}    # 동사, 형용사


def extract_nouns(text: str) -> list[str]:
    """명사만 추출 (요약 전처리용)"""
    tokens = _kiwi.tokenize(text)
    return [t.form for t in tokens if t.tag in _NOUN_TAGS and len(t.form) >= 2]


def split_sentences(text: str) -> list[str]:
    """kiwipiepy의 문장 분리기를 사용하여 텍스트를 문장 단위로 분할"""
    results = []
    for sent in _kiwi.split_into_sents(text):
        s = sent.text.strip()
        if len(s) >= 10:  # 너무 짧은 문장 제외
            results.append(s)
    return results


# ══════════════════════════════════════
#  TextRank 핵심 문장 추출
# ══════════════════════════════════════
from textrankr import TextRank


class KiwiTokenizer:
    """textrankr에 전달할 토크나이저 어댑터"""
    def __call__(self, text: str) -> list[str]:
        tokens = _kiwi.tokenize(text)
        return [t.form for t in tokens if len(t.form) >= 2]


_textrank = TextRank(KiwiTokenizer())


def extract_key_sentences(text: str, num_sentences: int = 10) -> list[dict]:
    """
    TextRank 알고리즘으로 핵심 문장을 추출합니다.

    Args:
        text: 전체 전사문 텍스트
        num_sentences: 추출할 핵심 문장 수

    Returns:
        [{"text": "문장 텍스트", "score": 0.85, "rank": 1}, ...]
    """
    # textrankr.summarize()는 요약문 문자열 반환 → 개별 문장 + 점수를 얻으려면 내부 접근
    sentences = split_sentences(text)

    if len(sentences) <= num_sentences:
        # 문장 수가 추출 목표보다 적으면 전부 반환
        return [
            {"text": s, "score": 1.0, "rank": i + 1}
            for i, s in enumerate(sentences)
        ]

    # TextRank로 요약 추출
    summarized = _textrank.summarize(text, num_sentences)
    summarized_sentences = split_sentences(summarized)

    results = []
    for i, sent in enumerate(summarized_sentences):
        results.append({
            "text": sent,
            "score": round(1.0 - (i * 0.05), 2),  # 순위 기반 근사 점수
            "rank": i + 1,
        })

    return results[:num_sentences]


# ══════════════════════════════════════
#  LLM 프롬프트 템플릿
# ══════════════════════════════════════
SUMMARY_SYSTEM_PROMPT = """당신은 대학 강의 내용을 분석하고 구조화된 요약을 작성하는 AI 교수입니다.
반드시 아래 JSON 형식으로만 응답하세요. JSON 외의 텍스트는 절대 포함하지 마세요."""

SUMMARY_USER_PROMPT_TEMPLATE = """아래는 강의에서 추출된 핵심 문장들입니다:

{key_sentences_text}

위 핵심 문장들을 기반으로 강의 내용을 구조화하여 요약하세요.
반드시 아래 JSON 형식으로만 응답하세요:

{{
  "core_concepts": ["핵심 개념 3~5개를 간결하게 나열"],
  "key_sentences": ["강의에서 가장 중요한 문장 3~5개"],
  "review_points": ["복습 시 반드시 확인할 포인트 3~5개"],
  "detailed_explanation": "전체 강의 내용을 200자 이내로 요약한 설명",
  "exam_points": ["시험에 출제될 가능성이 높은 포인트 3~5개"]
}}"""


def _build_summary_prompt(key_sentences: list[dict]) -> list[dict]:
    """LLM에 보낼 messages 배열 구성"""
    sentences_text = "\n".join(
        f"[{s['rank']}] {s['text']}" for s in key_sentences
    )
    user_prompt = SUMMARY_USER_PROMPT_TEMPLATE.format(
        key_sentences_text=sentences_text
    )
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _parse_summary_json(raw_text: str) -> dict:
    """LLM 응답에서 JSON 객체를 추출하여 파싱"""
    # <think>...</think> 제거
    text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    # 마크다운 코드블록 제거
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # JSON 객체 추출: 가장 바깥쪽 { ... } 를 찾음
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        summary = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"요약 JSON 파싱 실패: {e}\n원본: {raw_text[:500]}")
        raise ValueError(f"LLM 응답을 JSON으로 파싱할 수 없습니다: {e}")

    if not isinstance(summary, dict):
        raise ValueError("LLM 응답이 JSON 객체가 아닙니다")

    # 필수 필드 기본값 보장
    summary.setdefault("core_concepts", [])
    summary.setdefault("key_sentences", [])
    summary.setdefault("review_points", [])
    summary.setdefault("detailed_explanation", "")
    summary.setdefault("exam_points", [])

    return summary


# ══════════════════════════════════════
#  요약 생성 (메인 함수)
# ══════════════════════════════════════
async def generate_summary(
    transcript_text: str,
    num_key_sentences: int = 10,
) -> dict:
    """
    전사문 텍스트를 기반으로 요약을 생성합니다.

    Args:
        transcript_text: 강의 전사문 전체 텍스트
        num_key_sentences: TextRank로 추출할 핵심 문장 수

    Returns:
        {
            "summary": { ... 5가지 템플릿 ... },
            "key_sentences": [ {"text": ..., "score": ..., "rank": ...} ],
            "nouns": ["명사1", "명사2", ...]   # 형태소 분석 결과
        }
    """
    # 1. 형태소 분석 — 핵심 명사 추출
    nouns = extract_nouns(transcript_text)
    unique_nouns = list(dict.fromkeys(nouns))[:30]  # 중복 제거, 최대 30개
    logger.info(f"[SUMMARY] 형태소 분석 완료: {len(unique_nouns)}개 명사 추출")

    # 2. TextRank 핵심 문장 추출
    key_sentences = extract_key_sentences(transcript_text, num_key_sentences)
    logger.info(f"[SUMMARY] TextRank 완료: {len(key_sentences)}개 핵심 문장 추출")

    # 3. LLM 구조화 요약 (또는 Mock)
    if MOCK_MODE:
        logger.info("[SUMMARY] MOCK_MODE: 목업 요약 데이터 반환")
        summary = _generate_mock_summary(key_sentences, unique_nouns)
    else:
        summary = await _call_llm_summary(key_sentences)

    return {
        "summary": summary,
        "key_sentences": key_sentences,
        "nouns": unique_nouns,
    }


async def _call_llm_summary(key_sentences: list[dict]) -> dict:
    """LLM 호출하여 구조화된 요약 생성"""
    messages = _build_summary_prompt(key_sentences)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0)) as client:
            res = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "max_tokens": 2048,
                    "temperature": 0.3,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            )
            res.raise_for_status()

        data = res.json()
        raw_answer = data["choices"][0]["message"]["content"]
        logger.info(f"[SUMMARY] LLM 응답 수신: {len(raw_answer)} chars")

        return _parse_summary_json(raw_answer)

    except httpx.HTTPError as e:
        logger.error(f"[SUMMARY] LLM 호출 실패: {e}")
        raise RuntimeError(f"LLM 서버 연결 실패: {e}")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"[SUMMARY] 요약 생성 중 예상치 못한 오류: {e}")
        raise RuntimeError(f"요약 생성 실패: {e}")


# ══════════════════════════════════════
#  목업 데이터 (LLM 미연결 시)
# ══════════════════════════════════════
def _generate_mock_summary(key_sentences: list[dict], nouns: list[str]) -> dict:
    """프론트엔드 개발/테스트용 목업 요약 데이터"""
    top_nouns = nouns[:5] if nouns else ["데이터베이스", "정규화", "트랜잭션", "인덱스", "SQL"]
    top_sentences = [s["text"] for s in key_sentences[:5]] if key_sentences else [
        "데이터베이스 정규화는 중복을 제거하는 과정입니다.",
        "트랜잭션은 ACID 속성을 만족해야 합니다.",
        "인덱스는 검색 성능을 향상시킵니다.",
    ]

    return {
        "core_concepts": top_nouns,
        "key_sentences": top_sentences,
        "review_points": [
            f"{n}의 정의와 특징을 정리하세요." for n in top_nouns[:3]
        ],
        "detailed_explanation": (
            f"이번 강의에서는 {', '.join(top_nouns[:3])} 등의 개념을 다루었습니다. "
            f"특히 {top_nouns[0]}에 대해 중점적으로 설명하였으며, "
            f"관련 예제와 함께 실제 활용 방법을 소개하였습니다."
        ),
        "exam_points": [
            f"{n} 관련 문제가 출제될 수 있습니다." for n in top_nouns[:3]
        ],
    }
