"""
transcripts 테이블의 데이터를 읽어서 bge-m3 임베딩을 생성하고
llama_index PGVectorStore에 저장하는 스크립트.

사용법: python rag_embed_chunks.py
"""
import psycopg2
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# 임베딩 모델
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# DB에서 transcripts 읽기
conn = psycopg2.connect(
    host="localhost", port=5432,
    database="rag", user="postgres", password="1234"
)
cur = conn.cursor()
cur.execute("""
    SELECT t.transcript_id,
           t.session_id,
           t.chunk_index,
           t.start_time,
           t.end_time,
           COALESCE(t.corrected_text, t.chunk_text) AS chunk_text,
           s.title as session_title,
           s.session_date,
           co.title as course_title
    FROM transcripts t
    JOIN sessions s ON t.session_id = s.session_id
    JOIN courses co ON s.course_id = co.course_id
    ORDER BY t.session_id, t.chunk_index
""")
rows = cur.fetchall()
cur.close()
conn.close()

print(f"총 {len(rows)}개 transcript 로드됨")

# Document 변환
documents = []
for row in rows:
    transcript_id, session_id, chunk_index, start_time, end_time, chunk_text, session_title, session_date, course_title = row
    doc = Document(
        text=chunk_text,
        metadata={
            "transcript_id": str(transcript_id),
            "session_id": str(session_id),
            "course_title": course_title,
            "session_title": session_title,
            "session_date": str(session_date),
            "chunk_index": chunk_index,
            "start_time": float(start_time),
            "end_time": float(end_time),
        },
    )
    documents.append(doc)

print(f"{len(documents)}개 Document 생성 완료, 임베딩 시작...")

# PGVectorStore에 저장
vector_store = PGVectorStore.from_params(
    database="rag",
    host="localhost",
    password="1234",
    port=5432,
    user="postgres",
    table_name="rag",
    embed_dim=1024,
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True,
)

print(f"완료! {len(documents)}개 transcript가 임베딩되어 rag 테이블에 저장됨")
