from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from run_model import run_model
import logging
import uvicorn

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 글로벌 변수로 모델과 토크나이저 저장
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    logger.info("모델 로딩 중...")
    model, tokenizer = run_model()
    logger.info("모델 로딩 완료.")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 입력 프롬프트를 제외한 생성된 텍스트만 추출
        prompt_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][prompt_length:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return {"response": response_text.strip()}
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
