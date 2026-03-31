from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import torch
from run_model import run_model

model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model, tokenizer = run_model()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256


@app.post("/generate")
async def generate(req: GenerateRequest):
    messages = [
        {"role": "system", "content": "You are a helpful lecture assistant. Answer in Korean."},
        {"role": "user", "content": req.prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    def run_generation():
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, run_generation)

    return {"answer": answer}
