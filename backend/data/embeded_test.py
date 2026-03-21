from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ",device)
model = SentenceTransformer("BAAI/bge-m3",device=device)

def get_embedding(text: str):
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()



text = "DBMS는 데이터의 무결성을 관리한다."
embedding = get_embedding(text)

print(len(embedding))
print(embedding[:5])