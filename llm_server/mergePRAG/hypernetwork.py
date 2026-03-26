import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm_server.run_model import run_model

model=run_model()
print(model)