from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ",device)
model = SentenceTransformer("BAAI/bge-m3",device=device)

def get_embedding(text: str):
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
