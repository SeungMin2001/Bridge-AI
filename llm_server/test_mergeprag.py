"""
MergePRAG 동작 테스트 스크립트

3가지 방식으로 같은 질문에 답변 비교:
  A. LLM Only (아무 정보 없이)
  B. RAG 방식 (프롬프트에 텍스트 삽입)
  C. MergePRAG (K,V inject)

사용법: cd llm_server && python test_mergeprag.py
"""
import torch
from run_model import run_model
from mergePRAG.main import CourseMemoryManager, make_hook, CRITICAL_LAYER

# ── 테스트 데이터 ──
TEST_PASSAGE = "운영체제에서 프로세스는 실행 중인 프로그램의 인스턴스이다. 각 프로세스는 고유한 PID를 가지며, PCB(Process Control Block)에 프로세스의 상태, 프로그램 카운터, 레지스터 정보가 저장된다."
TEST_QUESTION = "프로세스가 뭐야?"

print("=" * 60)
print("MergePRAG 동작 테스트")
print("=" * 60)

# ── 모델 로드 ──
print("\n[1/4] 모델 로딩...")
model, tokenizer = run_model()
device = next(model.parameters()).device

# ── HyperNetwork 로드 ──
print("[2/4] HyperNetwork 로딩...")
mm = CourseMemoryManager(model, tokenizer, device)

# ── 과목 메모리에 passage 추가 ──
print("[3/4] passage를 과목 메모리에 추가...")
mm.add_passage("test_course", TEST_PASSAGE)
print(f"  passage: {TEST_PASSAGE[:50]}...")

# ── 프롬프트 구성 ──
SYSTEM = "You are a helpful lecture assistant. Answer in Korean. 반드시 3문장 이내로 핵심만 답변해."

def make_prompt(question, context=""):
    if context:
        user = f"참고자료:\n{context}\n\n질문: {question}"
    else:
        user = question
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def generate(input_text, max_new_tokens=128):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True)
    # think 태그 제거
    import re
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return raw

# ── 테스트 실행 ──
print(f"\n[4/4] 테스트 시작: '{TEST_QUESTION}'")
print("=" * 60)

# A. LLM Only
print("\n[A] LLM Only (사전학습 지식만)")
prompt_a = make_prompt(TEST_QUESTION)
answer_a = generate(prompt_a)
print(f"  → {answer_a}")

# B. RAG 방식 (텍스트 삽입)
print("\n[B] RAG + LLM (프롬프트에 텍스트 삽입)")
prompt_b = make_prompt(TEST_QUESTION, context=TEST_PASSAGE)
answer_b = generate(prompt_b)
print(f"  → {answer_b}")

# C. MergePRAG (K,V inject)
print("\n[C] MergePRAG + LLM (Critical Layer inject)")
K, V = mm.get_memory("test_course")
print(f"  K shape: {K.shape}, V shape: {V.shape}")
print(f"  K norm: {K.norm().item():.4f}, V norm: {V.norm().item():.4f}")
print(f"  K has NaN: {K.isnan().any().item()}, V has NaN: {V.isnan().any().item()}")

target_layer = model.model.layers[CRITICAL_LAYER]
hook = target_layer.register_forward_hook(make_hook(K, V))
prompt_c = make_prompt(TEST_QUESTION)  # context 없이!

# raw output 확인 (skip_special_tokens=False)
inputs = tokenizer(prompt_c, return_tensors="pt").to(device)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
gen_ids = output_ids[0][inputs["input_ids"].shape[1]:]
raw_c = tokenizer.decode(gen_ids, skip_special_tokens=False)
hook.remove()

print(f"  raw output: {repr(raw_c[:200])}")
import re
answer_c = re.sub(r'<think>.*?</think>', '', raw_c, flags=re.DOTALL).strip()
answer_c = re.sub(r'<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>', '', answer_c).strip()
print(f"  → {answer_c}")

# ── 결과 비교 ──
print("\n" + "=" * 60)
print("비교 결과")
print("=" * 60)
print(f"\n질문: {TEST_QUESTION}")
print(f"passage: {TEST_PASSAGE[:80]}...")
print(f"\n[A] LLM Only:   {answer_a[:100]}")
print(f"[B] RAG+LLM:    {answer_b[:100]}")
print(f"[C] MergePRAG:  {answer_c[:100]}")
print("\n※ [C]가 passage 내용(PID, PCB 등)을 언급하면 MergePRAG inject 성공!")
