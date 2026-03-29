import logging
import httpx
import os
from data.save_chunk_to_db import get_conn
# 데이터 폴더에 있는 임베딩 생성 함수 임포트
from data.embeded_test import get_embedding

# 로거 설정
logger = logging.getLogger(__name__)

# Docker 환경변수가 없으면 로컬 호스트의 Qwen 주소를 기본값으로 사용(변경해야함)
LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")

def retrieve_similar_chunks(query: str, limit: int = 3) -> list[str]:
    """
    사용자의 질문(query)을 벡터로 변환하고 DB에서 가장 유사한 청크(문서 조각)들을 가져옵니다.
    """
    if not query:
        return []

    # 1. 기존 로직(embeded_test.py)을 사용하여 사용자 질문에 대한 임베딩(벡터) 생성
    query_vector = get_embedding(query)
    if query_vector is None:
        logger.error("질문에 대한 임베딩 생성에 실패했습니다.")
        return []

    # 2. DB 연결
    conn = get_conn()
    if conn is None:
        return []

    try:
        cur = conn.cursor()
        # 주의: 테이블 이름이 'chunks'이고 벡터 컬럼이 'embedding'이어야 합니다.
        # pgvector의 코사인 유사도 연산자(<=>)를 사용하여 가장 유사한 데이터를 찾습니다. (L2 거리는 <-> 사용)
        cur.execute(
            """
            SELECT chunk_text
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_vector, limit)
        )

        results = cur.fetchall()

        # results는 [('chunk_text1',), ('chunk_text2',)] 형태의 튜플 리스트로 반환되므로 텍스트만 추출
        retrieved_texts = [row[0] for row in results if row[0] is not None]
        return retrieved_texts

    except Exception as e:
        logger.error(f"벡터 검색 중 오류 발생: {e}")
        return []
    finally:
        if conn:
            conn.close()

async def generate_llm_response(query: str, contexts: list[str]) -> str | None:
    """
    로컬 Qwen LLM으로 프롬프트(질문+문맥)를 전송하여 답변을 생성합니다.
    """
    # 검색된 문맥들을 하나의 문자열로 합칩니다.
    combined_context = "\n".join(contexts) if contexts else "참고할 만한 문맥이 없습니다."

    prompt = f"다음 문맥을 참고하여 질문에 답하세요.\n\n문맥:\n{combined_context}\n\n질문: {query}\n\n답변:"

    payload = {
        "model": "qwen",
        "prompt": prompt,
        "stream": False # 스트리밍(한 글자씩 출력) 사용 여부
    }

    try:
        # 비동기 HTTP 클라이언트를 사용하여 LLM 서버에 요청 전송
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(LLM_API_URL, json=payload)
            response.raise_for_status() # HTTP에러 발생 시 예외 처리

            data = response.json()
            return data.get("response", "LLM으로부터 응답이 없습니다.")
    except Exception as e:
        logger.error(f"LLM 응답 생성 실패: {e}")
        return None