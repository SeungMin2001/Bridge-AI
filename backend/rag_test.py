from llama_index.vector_stores.postgres import PGVectorStore

vector_store = PGVectorStore.from_params(
    database="rag",
    host="localhost",
    password="1234",
    port=5432,
    user="postgres",
    table_name="test",
    embed_dim=1024
)

print("PGVectorStore 연결 객체 생성 완료")