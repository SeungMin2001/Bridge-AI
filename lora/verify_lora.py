"""
MergePRAG LoRA E2E 검증 스크립트

vLLM(8001) + LoRA 서비스(9001)를 사용하여
HyperNetwork → K,V → LoRA 변환 → vLLM 핫로드 → 답변 비교까지
전체 파이프라인을 CMD에서 검증한다.

사용법: python verify_lora.py
"""
import re
import requests

# ── 서버 주소 ──
VLLM_URL = "http://localhost:8001"
LORA_SERVICE_URL = "http://localhost:9001"
BASE_MODEL = "Qwen/Qwen2.5-3B"
COURSE_ID = "test_verify"
SYSTEM_PROMPT = "질문에 한국어로 간단히 답하세요."
USE_CHAT_API = "instruct" in BASE_MODEL.lower()
FEWSHOT_Q = "대한민국의 수도는 무엇인가요?"
FEWSHOT_A = "서울입니다."
DEFAULT_GEN_PARAMS = {
    "max_tokens": 128,
    "temperature": 0.0,
}
FALLBACK_GEN_PARAMS = {
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "presence_penalty": 0.2,
    "frequency_penalty": 0.2,
}
COMPLETION_STOP = ["\n질문:", "\n\n"]
_HANGUL_RE = re.compile(r"[가-힣]")

# 모델이 절대 모르는 가상 지식
PASSAGE = (
    "mtg 알고리즘(MTG-Algo)은 데이터베이스 트랜잭션의 처리 속도를 "
    "기존 대비 500% 향상시킨 스케줄링 기법입니다. 이 알고리즘은 기존의 ACID 속성에 "
    "Q(Quantum) 속성을 추가하여 ACID-Q 모델을 제안했습니다."
)
QUESTION = "mtg 알고리즘(MTG-Algo)이 제안한 모델의 이름은 무엇인지 알려주세요."


def box(title):
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def _looks_garbled(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text or "")
    if len(cleaned) < 40:
        return False
    for size in range(2, 7):
        unit = cleaned[:size]
        if unit and unit * (len(cleaned) // size) == cleaned[: (len(cleaned) // size) * size]:
            return True
    return False


def _needs_retry(text: str) -> bool:
    return _looks_garbled(text) or not _HANGUL_RE.search(text or "")


def _chat_messages(prompt: str, fewshot: bool) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if fewshot:
        messages.extend([
            {"role": "user", "content": FEWSHOT_Q},
            {"role": "assistant", "content": FEWSHOT_A},
        ])
    messages.append({"role": "user", "content": prompt})
    return messages


def _completion_prompt(prompt: str, fewshot: bool) -> str:
    if fewshot:
        return (
            f"{SYSTEM_PROMPT}\n"
            "예시:\n"
            f"질문: {FEWSHOT_Q}\n"
            f"답변: {FEWSHOT_A}\n"
            f"질문: {prompt}\n"
            "답변:"
        )
    return (
        f"{SYSTEM_PROMPT}\n"
        f"질문: {prompt}\n"
        "답변:"
    )


def vllm_chat(prompt, model_name=BASE_MODEL, gen_params=None, fewshot=False):
    """vLLM OpenAI 호환 API로 질문을 보낸다."""
    payload = {
        "model": model_name,
        "messages": _chat_messages(prompt, fewshot=fewshot),
        **(DEFAULT_GEN_PARAMS | (gen_params or {})),
    }
    resp = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def vllm_completion(prompt, model_name=BASE_MODEL, gen_params=None, fewshot=False):
    """vLLM completions API로 질문을 보낸다."""
    completion_prompt = _completion_prompt(prompt, fewshot=fewshot)
    payload = {
        "model": model_name,
        "prompt": completion_prompt,
        "stop": COMPLETION_STOP,
        **(DEFAULT_GEN_PARAMS | (gen_params or {})),
    }
    resp = requests.post(
        f"{VLLM_URL}/v1/completions",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()


def vllm_ask(prompt, model_name=BASE_MODEL):
    """모델 특성에 맞는 API로 질문한다."""
    if USE_CHAT_API:
        answer = vllm_chat(prompt, model_name=model_name)
        if _needs_retry(answer):
            print("⚠️  응답이 반복 패턴이라 재샘플링합니다.")
            answer = vllm_chat(
                prompt,
                model_name=model_name,
                gen_params=FALLBACK_GEN_PARAMS,
                fewshot=True,
            )
        return answer
    answer = vllm_completion(prompt, model_name=model_name)
    if _needs_retry(answer):
        print("⚠️  응답이 반복 패턴이라 재샘플링합니다.")
        answer = vllm_completion(
            prompt,
            model_name=model_name,
            gen_params=FALLBACK_GEN_PARAMS,
            fewshot=True,
        )
    return answer


def main():
    box("MergePRAG LoRA E2E 검증")
    print(f"vLLM: {VLLM_URL}")
    print(f"LoRA Service: {LORA_SERVICE_URL}")

    # Step 1: 연결 확인
    box("Step 1: 서버 연결 확인")
    try:
        models = requests.get(f"{VLLM_URL}/v1/models", timeout=5).json()
        print(f"vLLM: {[m['id'] for m in models['data']]}")
    except Exception as e:
        print(f"vLLM 연결 실패: {e}")
        return

    try:
        health = requests.get(f"{LORA_SERVICE_URL}/health", timeout=5).json()
        print(f"LoRA Service: {health}")
    except Exception as e:
        print(f"LoRA Service 연결 실패: {e}")
        print("   → cd lora && python lora_service.py")
        return

    # Step 2: 주입 전 답변
    box("Step 2: 주입 전 일반 답변")
    print(f"질문: {QUESTION}")
    print(f"모드: {'chat' if USE_CHAT_API else 'completion'}")
    base_answer = vllm_ask(QUESTION)
    print(f"답변: {base_answer}")

    # Step 3: LoRA 변환 + 핫로드
    box("Step 3: HyperNetwork → K,V → LoRA → vLLM 핫로드")
    print(f"주입: {PASSAGE[:80]}...")
    resp = requests.post(
        f"{LORA_SERVICE_URL}/inject",
        json={"course_id": COURSE_ID, "passage": PASSAGE},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()

    lora = result.get("lora", {})
    hotload = result.get("hotload", {})
    print(f"\n  Passages: {result.get('passage_count')}")
    print(f"  ΔW Norm: {lora.get('delta_w_norm')}")
    if lora.get("delta_w_norm_raw") is not None:
        print(f"  ΔW Norm Raw: {lora.get('delta_w_norm_raw')}")
    if lora.get("delta_w_scale") is not None:
        print(f"  ΔW Scale: {lora.get('delta_w_scale')}")
    print(f"  Energy Ratio: {lora.get('energy_ratio')}")
    print(f"  Rank: {lora.get('rank')}")
    print(f"  핫로드: {'성공' if hotload.get('success') else '❌ 실패: ' + str(hotload.get('error', ''))}")
    adapter_model = (
        hotload.get("adapter_name")
        or lora.get("adapter_name")
        or f"mergeprag-{COURSE_ID}"
    )
    print(f"  Adapter: {adapter_model}")

    # Step 4: 모델 목록 확인
    box("Step 4: vLLM 모델 목록")
    models = requests.get(f"{VLLM_URL}/v1/models", timeout=5).json()
    ids = [m["id"] for m in models["data"]]
    print(f"모델: {ids}")
    if adapter_model in ids:
        print(f"'{adapter_model}' 확인됨!")
    else:
        print(f"'{adapter_model}' 목록에 없음")

    # Step 5: 주입 후 답변
    box("Step 5: LoRA 모델 답변")
    print(f"모델: {adapter_model}")
    try:
        lora_answer = vllm_ask(QUESTION, model_name=adapter_model)
        print(f"답변: {lora_answer}")
    except Exception as e:
        print(f"LoRA 답변 실패: {e}")
        lora_answer = ""

    # 결과
    box("결과 비교")
    print(f"[일반] {base_answer[:100]}")
    print(f"[LoRA] {lora_answer[:100]}")
    if "ACID-Q" in lora_answer or "acid-q" in lora_answer.lower():
        print("\n성공! 가상 지식(ACID-Q)을 정확히 답변했습니다.")
    elif lora_answer and lora_answer != base_answer:
        print("\n답변이 변했지만 정확한 정답이 아닙니다. alpha/rank 조정 필요.")
    else:
        print("\n답변 변화 없음. 파이프라인 점검 필요.")


if __name__ == "__main__":
    main()
