from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

# 1) 임베딩 모델 설정
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)

# 2) PostgreSQL vector store 다시 연결
vector_store = PGVectorStore.from_params(
    database="shin",
    host="localhost",
    password="1234",
    port=5432,
    user="postgres",
    table_name="shin",
    embed_dim=1024
)

# 3) 저장된 vector store로부터 index 만들기
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 4) retriever 만들기
retriever = index.as_retriever(similarity_top_k=3)

# 5) 검색
nodes = retriever.retrieve("기억과 인출")

print("검색 결과 개수:", len(nodes))
print()

for i, node in enumerate(nodes, 1):
    print(f"[{i}] 내용:")
    print(node.text)
    print("metadata:", node.metadata)
    print("-" * 50)