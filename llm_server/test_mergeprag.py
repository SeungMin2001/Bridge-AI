"""
MergePRAG Alpha Sweep 테스트

A. LLM Only → B. RAG → C~. MergePRAG alpha별 비교
사용법: cd llm_server && python test_mergeprag.py
"""
import torch
import re
from run_model import run_model
from mergePRAG.main import CourseMemoryManager, CRITICAL_LAYER
from mergePRAG.cross_attention import cross_attention

# ── 테스트 데이터 ──
TEST_PASSAGE = "운영체제에서 프로세스는 실행 중인 프로그램의 인스턴스이다. 각 프로세스는 고유한 PID를 가지며, PCB(Process Control Block)에 프로세스의 상태, 프로그램 카운터, 레지스터 정보가 저장된다."
TEST_QUESTION = "프로세스가 뭐야?"

print("=" * 60)
print("MergePRAG Alpha Sweep 테스트")
print("=" * 60)

# ── 모델 로드 ──
print("\n[1/3] 모델 로딩...")
model, tokenizer = run_model()
device = next(model.parameters()).device

print("[2/3] HyperNetwork 로딩...")
mm = CourseMemoryManager(model, tokenizer, device)

print("[3/3] passage 추가...")
mm.add_passage("test_course", TEST_PASSAGE)
K, V = mm.get_memory("test_course")
print(f"  K norm: {K.norm().item():.2f}, V norm: {V.norm().item():.2f}")

# ── 유틸 ──
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

def clean(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    return re.sub(r'<\|im_end\|>|<\|endoftext\|>|<\|im_start\|>', '', text).strip()

def generate(input_text, max_new=128):
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return clean(tokenizer.decode(gen_ids, skip_special_tokens=False))

def make_alpha_hook(delta_K, delta_V, alpha):
    logged = [False]
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        Kd = delta_K.to(device=hidden.device, dtype=hidden.dtype)
        Vd = delta_V.to(device=hidden.device, dtype=hidden.dtype)
        delta = cross_attention(hidden, Kd, Vd)
        if not logged[0]:
            ad = (alpha * delta).norm().item()
            print(f"    hidden={hidden.norm().item():.2f}, alpha*delta={ad:.2f}, 기여비={ad/hidden.norm().item():.1%}")
            logged[0] = True
        result = hidden + alpha * delta
        if isinstance(output, tuple):
            return (result,) + output[1:]
        return result
    return hook_fn

# ── 테스트 ──
print(f"\n질문: '{TEST_QUESTION}'")
print("=" * 60)

print("\n[A] LLM Only")
answer_a = generate(make_prompt(TEST_QUESTION))
print(f"  → {answer_a}")

print("\n[B] RAG + LLM")
answer_b = generate(make_prompt(TEST_QUESTION, context=TEST_PASSAGE))
print(f"  → {answer_b}")

# Alpha sweep
target_layer = model.model.layers[CRITICAL_LAYER]
prompt = make_prompt(TEST_QUESTION)
results = {}

for alpha in [0.1, 0.3, 0.5, 1.0]:
    print(f"\n[MergePRAG alpha={alpha}]")
    hook = target_layer.register_forward_hook(make_alpha_hook(K, V, alpha))
    ans = generate(prompt)
    hook.remove()
    results[alpha] = ans
    print(f"  → {ans}")

# ── 결과 요약 ──
print("\n" + "=" * 60)
print("결과 요약")
print("=" * 60)
print(f"질문: {TEST_QUESTION}")
print(f"passage: {TEST_PASSAGE}")
print(f"\n[A] LLM Only:\n  {answer_a}")
print(f"\n[B] RAG+LLM:\n  {answer_b}")
for alpha, ans in results.items():
    print(f"\n[α={alpha}]:\n  {ans}")
print("\n※ PID, PCB 언급하는 alpha = 최적값 → main.py ALPHA에 반영")
