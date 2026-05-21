"""
MergePRAG LoRA E2E 검증 스크립트

vLLM(8001) + LoRA 서비스(9001)를 사용하여
HyperNetwork → K,V → LoRA 변환 → vLLM 핫로드 → 답변 비교까지
전체 파이프라인을 CMD에서 검증한다.

Qwen/Qwen2.5-3B는 베이스 모델(Instruct 아님)이므로
/v1/completions + Qwen 전용 chat template으로 프롬프트를 구성한다.

사용법: python verify_lora.py
"""
import requests
import sys

# ── 서버 주소 ──
VLLM_URL = "http://localhost:8001"
LORA_SERVICE_URL = "http://localhost:9001"
BASE_MODEL = "Qwen/Qwen2.5-3B"
COURSE_ID = "test_verify"

# 모델이 절대 모르는 가상 지식
PASSAGE = (
    "2026년 새롭게 발표된 mtg 알고리즘(MTG-Algo)은 데이터베이스 트랜잭션의 처리 속도를 "
    "기존 대비 500% 향상시킨 혁신적인 스케줄링 기법입니다. 이 알고리즘은 기존의 ACID 속성에 "
    "Q(Quantum) 속성을 추가하여 ACID-Q 모델을 제안했습니다."
)
QUESTION = "mtg 알고리즘(MTG-Algo)이 제안한 새로운 트랜잭션 모델의 이름은 무엇인가요?"
EXPECTED_ANSWER = "ACID-Q"


def box(title):
    """구분선 출력"""
    print(f"\n{'='*60}\n {title}\n{'='*60}")


def vllm_generate(prompt: str, model_name: str = BASE_MODEL) -> str:
    """vLLM /v1/completions API로 텍스트를 생성한다.

    베이스 모델은 chat API가 아닌 completion API를 써야 한다.
    Qwen2.5 chat template 형식으로 프롬프트를 구성한다.
    """
    # Qwen2.5 chat template 형식
    formatted_prompt = (
        "<|im_start|>system\n"
        "당신은 유능한 AI 조교입니다. 질문에 한국어로 간결하게 답하세요.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    resp = requests.post(
        f"{VLLM_URL}/v1/completions",
        json={
            "model": model_name,
            "prompt": formatted_prompt,
            "max_tokens": 128,
            "temperature": 0,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()


def main():
    box("MergePRAG LoRA E2E 검증")
    print(f"vLLM: {VLLM_URL}")
    print(f"LoRA Service: {LORA_SERVICE_URL}")
    print(f"Model: {BASE_MODEL} (base, completion API + Qwen chat template)")

    # ── Step 1: 연결 확인 ──
    box("Step 1: 서버 연결 확인")
    try:
        models = requests.get(f"{VLLM_URL}/v1/models", timeout=5).json()
        model_ids = [m["id"] for m in models["data"]]
        print(f"✅ vLLM: {model_ids}")
    except Exception as e:
        print(f"❌ vLLM 연결 실패: {e}")
        return

    try:
        health = requests.get(f"{LORA_SERVICE_URL}/health", timeout=5).json()
        print(f"✅ LoRA Service: {health}")
    except Exception as e:
        print(f"❌ LoRA Service 연결 실패: {e}")
        print("   → cd lora && python lora_service.py")
        return

    # ── Step 1.5: 기존 메모리 클리어 (깨끗한 테스트) ──
    try:
        requests.post(
            f"{LORA_SERVICE_URL}/memory/clear",
            json={"course_id": COURSE_ID},
            timeout=5,
        )
        print(f"✅ 기존 메모리 클리어: {COURSE_ID}")
    except:
        pass

    # ── Step 2: 주입 전 답변 (모델이 모르는 지식) ──
    box("Step 2: 주입 전 일반 답변 (모르는 지식)")
    print(f"질문: {QUESTION}")
    try:
        base_answer = vllm_generate(QUESTION)
        print(f"\n답변: {base_answer}")
        if EXPECTED_ANSWER.lower() in base_answer.lower():
            print(f"\n⚠️ 모델이 이미 '{EXPECTED_ANSWER}'를 알고 있습니다! 다른 가상 지식이 필요합니다.")
    except Exception as e:
        print(f"❌ 일반 답변 생성 실패: {e}")
        base_answer = f"[ERROR: {e}]"

    # ── Step 3: LoRA 변환 + 핫로드 ──
    box("Step 3: HyperNetwork → K,V → LoRA 변환 → vLLM 핫로드")
    print(f"주입: {PASSAGE[:80]}...")

    try:
        resp = requests.post(
            f"{LORA_SERVICE_URL}/inject",
            json={"course_id": COURSE_ID, "passage": PASSAGE},
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"❌ 주입 실패: {e}")
        return

    lora_info = result.get("lora", {})
    hotload_info = result.get("hotload", {})

    print(f"\n  [HyperNetwork → K,V]")
    print(f"    Passage Count: {result.get('passage_count')}")

    print(f"\n  [K,V → LoRA 변환 (SVD)]")
    print(f"    ΔW Norm: {lora_info.get('delta_w_norm')}")
    if lora_info.get("delta_w_norm_raw") is not None:
        print(f"    ΔW Norm Raw: {lora_info.get('delta_w_norm_raw')}")
    if lora_info.get("delta_w_scale") is not None:
        print(f"    ΔW Scale: {lora_info.get('delta_w_scale')}")
    print(f"    Energy Ratio: {lora_info.get('energy_ratio')}")
    print(f"    Rank: {lora_info.get('rank')}")
    print(f"    LoRA A: {lora_info.get('lora_A_shape')}")
    print(f"    LoRA B: {lora_info.get('lora_B_shape')}")

    print(f"\n  [vLLM 핫로드]")
    adapter_model = (
        hotload_info.get("adapter_name")
        or lora_info.get("adapter_name")
        or f"mergeprag-{COURSE_ID}"
    )
    if hotload_info.get("success"):
        print(f"    ✅ 성공! Adapter: {adapter_model}")
    else:
        print(f"    ❌ 실패: {hotload_info.get('error')}")
        print("    Docker에 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True 필요")

    # ── Step 4: 모델 목록 확인 ──
    box("Step 4: vLLM 모델 목록")
    try:
        models = requests.get(f"{VLLM_URL}/v1/models", timeout=5).json()
        ids = [m["id"] for m in models["data"]]
        print(f"모델: {ids}")
        if adapter_model in ids:
            print(f"✅ '{adapter_model}' 확인됨!")
        else:
            print(f"⚠️ '{adapter_model}' 목록에 없음")
    except Exception as e:
        print(f"모델 목록 조회 실패: {e}")

    # ── Step 5: 주입 후 LoRA 모델 답변 ──
    box("Step 5: LoRA 모델 답변")
    print(f"모델: {adapter_model}")
    try:
        lora_answer = vllm_generate(QUESTION, model_name=adapter_model)
        print(f"\n답변: {lora_answer}")
    except Exception as e:
        print(f"❌ LoRA 모델 답변 실패: {e}")
        lora_answer = ""

    # ── 결과 비교 ──
    box("결과 비교")
    print(f"기대 정답: {EXPECTED_ANSWER}")
    print(f"[일반] {base_answer[:150]}")
    print(f"[LoRA] {lora_answer[:150]}")

    if EXPECTED_ANSWER.lower() in lora_answer.lower():
        print(f"\n🎉 성공! LoRA 모델이 '{EXPECTED_ANSWER}'를 정확히 답변했습니다.")
        print("   파이프라인 동작 확인 완료!")
    elif lora_answer and lora_answer != base_answer:
        print("\n⚠️ 답변이 변했지만 정확한 정답이 포함되지 않았습니다.")
        print("   → lora_service.py의 LORA_ALPHA, LORA_RANK 조정을 시도하세요.")
    else:
        print("\n❌ 답변 변화 없음. 파이프라인 점검 필요.")

    # ── LoRA 서비스 통계 ──
    box("LoRA 서비스 통계")
    try:
        resp = requests.get(f"{LORA_SERVICE_URL}/stats", timeout=5)
        stats = resp.json()
        for k, v in stats.items():
            print(f"  {k}: {v}")
    except:
        print("  통계 조회 실패")


if __name__ == "__main__":
    main()
