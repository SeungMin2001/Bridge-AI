import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# 주의:
# 1) 아래 MODEL_NAME에는 "ASR 교정용으로 파인튜닝된 KoBART 체크포인트"를 넣는 것이 맞다.
# 2) 처음 테스트용으로는 gogamza/kobart-base-v2 같은 base를 넣어 연결만 확인할 수는 있지만
#    실제 교정 성능은 기대하면 안 된다.

MODEL_NAME = "gogamza/kobart-base-v2"  # 나중에 네가 파인튜닝한 체크포인트로 교체

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
model.eval()

@torch.inference_mode()
def correct_text(text: str, max_length: int = 128) -> str:
    text = text.strip()
    if not text:
        return text

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        do_sample=False,
        early_stopping=True,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
    )

    corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return corrected if corrected else text