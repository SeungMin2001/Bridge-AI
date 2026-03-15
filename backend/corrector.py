# corrector.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "OpenLLM-Korea/kanana-1.5-2.1b-instruct-2505"

print("교정 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
corrector = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
corrector.eval()
print("교정 모델 로드 완료")


SYSTEM_PROMPT = """
너는 한국어 음성 전사 교정기다.
역할은 오직 하나다:
발음대로 잘못 적힌 단어를 문맥에 맞는 올바른 단어로 고쳐라.

반드시 지켜야 할 규칙:
- 발음 오류처럼 보이는 단어만 교정하라.
- 원래 문장의 의미를 바꾸지 마라.
- 없는 정보를 추가하지 마라.
- 문장을 새로 꾸미지 마라.
- 설명, 해설, 답변, 인사말을 쓰지 마라.
- 교정된 최종 문장만 출력하라.
- 교정할 부분이 없으면 원문 그대로 출력하라.
""".strip()


@torch.inference_mode()
def correct_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""다음 전사 문장을 교정하라.

전사문:
{text}

교정문:"""
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = next(corrector.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    outputs = corrector.generate(
        **inputs,
        max_new_tokens=max(16, min(len(text) + 10, 80)),
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # 첫 줄만 사용
    result = result.splitlines()[0].strip()

    # 접두어 제거
    for prefix in ["교정문:", "출력:", "정답:", "교정:", "수정문:"]:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()

    # 이상한 출력 방어
    if not result:
        return text

    # 원문보다 지나치게 길면 교정이 아니라 생성으로 판단
    if len(result) > max(len(text) + 12, int(len(text) * 1.4)):
        return text

    banned_starts = ["네", "알겠습니다", "이해했습니다", "저는", "설명", "답변"]
    if any(result.startswith(x) for x in banned_starts):
        return text

    return result