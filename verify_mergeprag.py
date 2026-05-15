import requests
import json
import time

# LLM 서버 주소 (기본 포트 8001)
URL = "http://localhost:8001"
COURSE_ID = "test_verify_course"

PASSAGE = (
    "2026년 새롭게 발표된 mtg 알고리즘(MTG-Algo)은 데이터베이스 트랜잭션의 처리 속도를 "
    "기존 대비 500% 향상시킨 혁신적인 스케줄링 기법입니다. 이 알고리즘은 기존의 ACID 속성에 "
    "Q(Quantum) 속성을 추가하여 ACID-Q 모델을 제안했습니다."
)

QUESTION = "mtg 알고리즘(MTG-Algo)이 제안한 새로운 트랜잭션 모델의 이름은 무엇인가요?"

def print_box(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def main():
    print_box("실시간 MergePRAG 가중치 주입 검증 스크립트")
    print(f"목표 LLM 서버: {URL}\n")

    # 1. 일반 LLM 답변 (지식 주입 전)
    print("▶ [Step 1] 주입 전 일반 모델에게 질문합니다 (모델이 모르는 가상의 지식)")
    print(f"질문: {QUESTION}")
    try:
        res = requests.post(
            f"{URL}/generate", 
            json={"prompt": QUESTION, "enable_thinking": False},
            timeout=30
        )
        base_answer = res.json().get("answer", "")
        print(f"\n일반 모델 답변:\n{base_answer}")
    except Exception as e:
        print(f"오류: LLM 서버({URL})와 연결할 수 없습니다. 서버가 켜져 있는지 확인하세요. ({e})")
        return

    # 2. HyperNetwork 메모리 주입 (K,V 가중치 추출)
    print_box("▶ [Step 2] 실시간 전사문(메모리)을 주입합니다")
    print(f"주입 내용: {PASSAGE}")
    res = requests.post(
        f"{URL}/memory/add",
        json={"course_id": COURSE_ID, "passage": PASSAGE},
        timeout=30
    )
    add_result = res.json()
    print(f"\n주입 결과: Passage Count = {add_result.get('passage_count')}")
    print("(HyperNetwork가 텍스트를 인코딩하여 K,V 벡터를 추출 및 저장했습니다.)")

    # 3. K,V 가중치 확인
    print_box("▶ [Step 3] 추출된 메모리(가중치)의 상태를 확인합니다")
    res = requests.get(f"{URL}/memory/stats/{COURSE_ID}")
    stats = res.json()
    print(f"K 텐서 Shape: {stats.get('k_shape')}")
    print(f"V 텐서 Shape: {stats.get('v_shape')}")
    print(f"K 텐서 Norm:  {stats.get('k_norm')}")
    
    if stats.get("k_shape") == [16, 2048]:
        print("\n성공! HyperNetwork가 Qwen2.5-3B에 맞는 [16, 2048] 형태의 가중치를 정상 추출했습니다.")
    else:
        print("\n경고: 추출된 가중치 형태가 예상과 다릅니다.")

    # 4. MergePRAG LLM 답변 (지식 주입 후)
    print_box("▶ [Step 4] 주입된 모델에게 다시 질문합니다 (메모리 주입 작동 확인)")
    print("과정: Layer 19에 추출된 가중치(K,V)를 강제로 끼워넣어 예측을 조작합니다.")
    print(f"질문: {QUESTION}")
    
    res = requests.post(
        f"{URL}/generate/mergeprag",
        json={"course_id": COURSE_ID, "prompt": QUESTION, "enable_thinking": False},
        timeout=30
    )
    prag_answer = res.json().get("answer", "")
    print(f"\n주입 후 모델 답변:\n{prag_answer}")

    print_box("결과 요약")
    if "ACID-Q" in prag_answer or "acid-q" in prag_answer.lower():
        print("모델이 주입된 가상의 지식을 정확히 사용하여 답변했습니다.")
        print("HyperNetwork 추출 및 vLLM(Transformers) 레이어 주입이 완벽하게 동작합니다!")
    else:
        print("주의: 답변이 주입된 정보를 포함하지 않을 수 있습니다. 결과를 확인해 주세요.")

if __name__ == "__main__":
    main()
