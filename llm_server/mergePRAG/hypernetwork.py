import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_server.run_model import run_model
from mergePRAG.embedding import embedding

model,tokenizer=run_model()

embedded=embedding(model,tokenizer,text="test")

print(embedded)