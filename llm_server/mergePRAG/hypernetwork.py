from llm_server.run_model import run_model
from .embedding import embedding
from .pooling import AttentivePooling
from .mlp import MLP
from .linearProjection import LinearProjection

model,tokenizer=run_model() #모델 실행(qwen 3.5 9B)
device = next(model.parameters()).device #cuda
dtype = next(model.parameters()).dtype #float16
k=16 #논문 그대로.

def HyperNetwork(text):
    embedded=embedding(model,tokenizer,text).to(device=device,dtype=dtype) #embedding from qwen 3.5

    d_model=model.config.hidden_size

    pooling=AttentivePooling(d_model).to(device=device,dtype=dtype)
    res=pooling.forward(embedded)

    mlp=MLP(d_model).to(device=device,dtype=dtype)
    res=mlp.forward(res)

    lp=LinearProjection(d_model,d_model,k).to(device=device,dtype=dtype)
    res=lp.forward(res)

    K,V=res
    
    return K,V
