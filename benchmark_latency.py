"""
RAG + vLLM 응답 속도 벤치마크

측정 항목:
  - RAG 검색 시간
  - TTFT (Time to First Token): 요청 → 첫 토큰 도착
  - 총 응답 시간: 요청 → 마지막 토큰 도착
  - 토큰 수 & 처리량 (tokens/sec)

사용법: python benchmark_latency.py
  (backend 서버가 localhost:8000에서 ���행 중이어야 함)
"""
import httpx
import time
import json
import sys

BACKEND_URL = "http://localhost:8000"

# 벤치마크용 키워드 (실제 사용: 키워드 클릭 → 설명)
QUESTIONS = [
    "바이트",
    "TCP",
    "기회비용",
    "조건형성",
    "프로세스",
    "수요",
    "행동주의",
    "데드락",
    "GDP",
    "기억",
]


def measure_non_streaming(question: str) -> dict:
    """비스트리밍 /chat 엔드포인트 측정"""
    start = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(10.0, read=300.0)) as client:
        res = client.post(f"{BACKEND_URL}/chat", json={
            "question": question,
            "is_thinking": False,
        })
    end = time.perf_counter()

    data = res.json()
    return {
        "question": question,
        "total_time": end - start,
        "answer": data.get("answer", ""),
        "citations_count": len(data.get("citations", [])),
    }


def measure_streaming(question: str) -> dict:
    """스트리밍 /chat/stream 엔드포인트 측정 (TTFT 포함)"""
    ttft = None
    tokens = []
    total_tokens = 0

    start = time.perf_counter()

    with httpx.Client(timeout=httpx.Timeout(10.0, read=300.0)) as client:
        with client.stream("POST", f"{BACKEND_URL}/chat/stream", json={
            "question": question,
            "is_thinking": False,
        }) as stream:
            for line in stream.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break

                chunk = json.loads(payload)

                if chunk.get("type") == "token":
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    tokens.append(chunk["token"])
                    total_tokens += 1

    end = time.perf_counter()
    total_time = end - start
    answer = "".join(tokens)
    generation_time = total_time - (ttft or total_time)
    tps = total_tokens / generation_time if generation_time > 0 else 0

    return {
        "question": question,
        "ttft": ttft,
        "total_time": total_time,
        "generation_time": generation_time,
        "total_tokens": total_tokens,
        "tokens_per_sec": tps,
        "answer": answer,
    }


def run_benchmark():
    print("=" * 60)
    print("RAG + vLLM 응답 속도 벤치마크")
    print("=" * 60)

    # 서버 상태 확인
    try:
        httpx.get(f"{BACKEND_URL}/docs", timeout=5)
    except Exception:
        print(f"[ERROR] 서버 연결 불가: {BACKEND_URL}")
        print("  backend 서버를 먼저 실행하세요.")
        sys.exit(1)

    # 워밍업 (첫 요청은 느릴 수 있음)
    print("\n[워밍업] 첫 요청 실행 중...")
    warmup = measure_streaming(QUESTIONS[0])
    print(f"  워��업 완료: TTFT={warmup['ttft']:.3f}s, 총={warmup['total_time']:.3f}s\n")

    # 스트리�� 벤치마크
    print("-" * 60)
    print(f"{'질문':<30} {'TTFT':>7} {'총시간':>7} {'토큰수':>5} {'tok/s':>6}")
    print("-" * 60)

    results = []
    for q in QUESTIONS:
        r = measure_streaming(q)
        results.append(r)
        ttft_str = f"{r['ttft']:.3f}s" if r['ttft'] else "N/A"
        print(f"{q:<30} {ttft_str:>7} {r['total_time']:>6.3f}s {r['total_tokens']:>5} {r['tokens_per_sec']:>5.1f}")

    # 요약 통계
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    totals = [r["total_time"] for r in results]
    tps_list = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0]

    print("\n" + "=" * 60)
    print("요약 통계")
    print("=" * 60)
    if ttfts:
        print(f"  TTFT  (첫 토큰):  평균 {sum(ttfts)/len(ttfts):.3f}s | "
              f"���소 {min(ttfts):.3f}s | 최대 {max(ttfts):.3f}s")
    print(f"  총 응답 시간:      평균 {sum(totals)/len(totals):.3f}s | "
          f"최소 {min(totals):.3f}s | 최대 {max(totals):.3f}s")
    if tps_list:
        print(f"  처리량 (tok/s):    평균 {sum(tps_list)/len(tps_list):.1f} | "
              f"최소 {min(tps_list):.1f} | 최대 {max(tps_list):.1f}")

    # 결과 JSON 저장
    output = {
        "config": {
            "backend_url": BACKEND_URL,
            "questions_count": len(QUESTIONS),
        },
        "summary": {
            "ttft_avg": round(sum(ttfts) / len(ttfts), 4) if ttfts else None,
            "ttft_min": round(min(ttfts), 4) if ttfts else None,
            "ttft_max": round(max(ttfts), 4) if ttfts else None,
            "total_time_avg": round(sum(totals) / len(totals), 4),
            "total_time_min": round(min(totals), 4),
            "total_time_max": round(max(totals), 4),
            "tokens_per_sec_avg": round(sum(tps_list) / len(tps_list), 1) if tps_list else None,
        },
        "results": [
            {
                "question": r["question"],
                "ttft": round(r["ttft"], 4) if r["ttft"] else None,
                "total_time": round(r["total_time"], 4),
                "total_tokens": r["total_tokens"],
                "tokens_per_sec": round(r["tokens_per_sec"], 1),
                "answer": r["answer"],
            }
            for r in results
        ],
    }

    with open("benchmark_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[저장] benchmark_result.json")


if __name__ == "__main__":
    run_benchmark()
