import os
import time
import torch
import whisper
from diart import SpeakerDiarization, SpeakerDiarizationConfig

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("❌ 에러: 환경 변수 HF_TOKEN이 설정되지 않았습니다.")

INPUT_AUDIO = "a7.m4a"  # 테스트할 원본 오디오 파일

def measure_pipeline_latency():
    if not os.path.exists(INPUT_AUDIO):
        print(f"❌ 에러: {INPUT_AUDIO} 파일이 없습니다. 테스트용 오디오 파일을 준비해주세요.")
        return

    print("🚀 [모델 로딩 중] (초기 로딩 시간은 지연 시간에 포함하지 않습니다)...")

    # 1. 모델 초기화 (GPU 우선 할당)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"장치 확인: {device}")

    diart_config = SpeakerDiarizationConfig(hf_token=HF_TOKEN)
    diart_pipeline = SpeakerDiarization(diart_config)

    whisper_model = whisper.load_model("base", device=device) # base 모델 기준 (필요시 small/medium 변경)

    print("\n==================================================")
    print("⏱️ 본격적인 파이프라인 지연 시간(Latency) 측정 시작")
    print("==================================================\n")

    total_start_time = time.time()

    # ---------------------------------------------------------
    # 단계 1: Diart (화자 분리)
    # ---------------------------------------------------------
    t1_start = time.time()
    print("🗣️ [1/2] Diart 화자 분리 진행 중...")
    # 참고: 실제 스트리밍에서는 diart.stream을 쓰지만, 지연 시간 측정을 위해 파일 기반으로 구동합니다.
    diarization_result = diart_pipeline(INPUT_AUDIO)
    t1_end = time.time()
    diart_latency = t1_end - t1_start
    print(f"✅ [1/2] Diart 화자 분리 완료: {diart_latency:.2f}초")

    # ---------------------------------------------------------
    # 단계 2: Whisper (STT 전사)
    # ---------------------------------------------------------
    t2_start = time.time()
    print("📝 [2/2] Whisper STT 전사 진행 중...")
    whisper_result = whisper_model.transcribe(INPUT_AUDIO)
    t2_end = time.time()
    whisper_latency = t2_end - t2_start
    print(f"✅ [2/2] Whisper STT 전사 완료: {whisper_latency:.2f}초")

    total_end_time = time.time()
    total_latency = total_end_time - total_start_time

    print("\n==================================================")
    print("📊 [최종 측정 결과]")
    print(f"- 화자 분리 소요 시간   : {diart_latency:.2f}초")
    print(f"- 음성 전사 소요 시간   : {whisper_latency:.2f}초")
    print(f"🔥 총 파이프라인 지연 시간: {total_latency:.2f}초")
    print("==================================================")

    # 전사 결과 샘플 출력
    print("\n[STT 텍스트 결과 샘플]")
    print(whisper_result["text"][:200] + "...\n")

if __name__ == "__main__":
    measure_pipeline_latency()