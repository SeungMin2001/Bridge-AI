import json
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)

file_path="./data/transcripts/60db78d8-10fe-48f1-8f05-9d31a8f561a5.jsonl"
documents=[]

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)

        doc = Document(
            text=item["text"],
            metadata={
                "session_id": item["session_id"],
                "start_time": item["start_time"],
                "end_time": item["end_time"],
                "raw_text": item["raw_text"],
            }
        )
        documents.append(doc)

vector_store = PGVectorStore.from_params(
    database="shin",
    host="localhost",
    password="1234",
    port=5432,
    user="postgres",
    table_name="test",
    embed_dim=1024
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

print("완료")