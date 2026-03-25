from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ",device)
model = SentenceTransformer("BAAI/bge-m3",device=device)

def get_embedding(text: str):
    embedding = model.encode(text, normalize_embeddings=True) #공식문서 참조하면 코사인 유사도보다 내적이 더 효과적이라고 말하고있음.
    return embedding.tolist()
