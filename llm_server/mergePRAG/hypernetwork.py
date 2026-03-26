import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_server.run_model import run_model
from .embedding import embedding
from .pooling import AttentivePooling



model,tokenizer=run_model() #모델 실행(qwen 3.5 9B)
device = next(model.parameters()).device # cuda

text="test"

embedded=embedding(model,tokenizer,text).to(device) #embedding from qwen 3.5

pooling=AttentivePooling(model.config.hidden_size).to(device)
res=pooling.forward(embedded)

print(res)


