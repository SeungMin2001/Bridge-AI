# corrector.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "OpenLLM-Korea/kanana-1.5-2.1b-instruct-2505"  # 8B보다 가벼운 쪽 추천

print("교정 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
corrector = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print("교정 모델 로드 완료")

@torch.inference_mode()
def correct_text(text: str) -> str:
    inputs = tokenizer(f"다음 문장의 명백한 음성인식 오류만 고쳐라:\n{text}", return_tensors="pt").to(corrector.device)
    outputs = corrector.generate(**inputs, max_new_tokens=30, do_sample=False)
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return result if result else text