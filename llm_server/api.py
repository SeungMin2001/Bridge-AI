from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio, re, torch
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
    max_new_tokens: int = 512


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
        return tokenizer.decode(generated_ids, skip_special_tokens=False)

    loop = asyncio.get_event_loop()
    raw_output = await loop.run_in_executor(None, run_generation)

    # think / answer 분리
    think_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    answer = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL)
    # special token 제거
    answer = re.sub(r'<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>', '', answer).strip()

    return {"thinking": thinking, "answer": answer}
