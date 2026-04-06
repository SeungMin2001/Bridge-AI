import argparse
import io
import sys
import time
import numpy as np

try:
    import requests
except ImportError:
    print("requests 라이브러리가 필요합니다: pip install requests")
    sys.exit(1)

try:
    import soundfile as sf
except ImportError:
    sf = None

DENOISE_SERVER_URL = "http://localhost:8002"


def test_health():
    """1. 헬스체크 테스트"""
    print("=" * 60)
    print("[테스트 1] /health 헬스체크")
    print("=" * 60)
    try:
        resp = requests.get(f"{DENOISE_SERVER_URL}/health", timeout=10)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {resp.json()}")

        if resp.status_code == 200 and resp.json().get("model_loaded"):
            print("  ✅ 헬스체크 성공 - 모델 로드 완료")
            return True
        else:
            print("  ❌ 모델이 아직 로드되지 않음")
            return False
    except requests.ConnectionError:
        print(f"  ❌ 서버 연결 실패: {DENOISE_SERVER_URL}")
        print("  → 서버가 실행 중인지 확인하세요.")
        return False


def generate_synthetic_audio(duration=3.0, sample_rate=16000):
    """
    테스트용 합성 오디오 생성 (깨끗한 사인파 + 잡음)
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

    # 깨끗한 음성 시뮬레이션 (여러 사인파 합성)
    clean_signal = (
        0.3 * np.sin(2 * np.pi * 300 * t) +    # 300Hz (기본 음성 주파수)
        0.15 * np.sin(2 * np.pi * 600 * t) +   # 600Hz (하모닉스)
        0.1 * np.sin(2 * np.pi * 1200 * t)     # 1200Hz
    ).astype(np.float32)

    # 잡음 추가 (백색 잡음 + 저주파 험)
    noise = (
        0.1 * np.random.randn(len(t)) +         # 백색 잡음
        0.05 * np.sin(2 * np.pi * 50 * t)       # 50Hz 전기 험
    ).astype(np.float32)

    noisy_signal = clean_signal + noise
    return noisy_signal, clean_signal, sample_rate


def test_denoise_wav(wav_path=None):
    """2. WAV 파일 잡음 제거 테스트"""
    print("\n" + "=" * 60)
    print("[테스트 2] POST /denoise (WAV 파일)")
    print("=" * 60)

    if wav_path:
        print(f"  입력 파일: {wav_path}")
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
    else:
        print("  입력: 합성 노이즈 오디오 (3초, 16kHz)")
        noisy, clean, sr = generate_synthetic_audio(duration=3.0, sample_rate=16000)

        # WAV 바이트로 변환
        buf = io.BytesIO()
        if sf:
            sf.write(buf, noisy, sr, format="WAV", subtype="FLOAT")
        else:
            # soundfile 없으면 scipy 사용
            from scipy.io import wavfile
            buf_temp = io.BytesIO()
            wavfile.write(buf_temp, sr, noisy)
            buf = buf_temp
        buf.seek(0)
        wav_bytes = buf.read()

        noisy_rms = np.sqrt(np.mean(noisy ** 2))
        noise_rms = np.sqrt(np.mean((noisy - clean) ** 2))
        print(f"  노이즈 신호 RMS: {noisy_rms:.4f}")
        print(f"  순수 잡음 RMS: {noise_rms:.4f}")

    # 서버로 전송
    start_time = time.time()
    resp = requests.post(
        f"{DENOISE_SERVER_URL}/denoise",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        timeout=60
    )
    elapsed = time.time() - start_time

    print(f"  Status: {resp.status_code}")
    print(f"  처리 시간: {elapsed:.3f}초")

    if resp.status_code == 200:
        print(f"  응답 크기: {len(resp.content)} bytes")
        print(f"  Content-Type: {resp.headers.get('content-type')}")

        # 결과 WAV 저장
        output_path = "test_output_denoised.wav"
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print(f"  ✅ 잡음 제거 결과 저장됨: {output_path}")

        # 결과 분석
        if sf:
            enhanced, enhanced_sr = sf.read(io.BytesIO(resp.content), dtype="float32")
            enhanced_rms = np.sqrt(np.mean(enhanced ** 2))
            print(f"  결과 SR: {enhanced_sr}Hz, 샘플 수: {len(enhanced)}")
            print(f"  결과 RMS: {enhanced_rms:.4f}")

        return True
    else:
        print(f"  ❌ 잡음 제거 실패: {resp.text}")
        return False


def test_denoise_raw():
    """3. Raw PCM bytes 잡음 제거 테스트"""
    print("\n" + "=" * 60)
    print("[테스트 3] POST /denoise/raw (raw PCM bytes)")
    print("=" * 60)

    sample_rate = 16000
    noisy, clean, sr = generate_synthetic_audio(duration=2.0, sample_rate=sample_rate)

    noisy_rms = np.sqrt(np.mean(noisy ** 2))
    print(f"  입력: 합성 노이즈 (2초, {sample_rate}Hz)")
    print(f"  입력 샘플 수: {len(noisy)}")
    print(f"  입력 바이트 크기: {len(noisy.tobytes())} bytes")
    print(f"  입력 RMS: {noisy_rms:.4f}")

    # raw bytes로 전송
    start_time = time.time()
    resp = requests.post(
        f"{DENOISE_SERVER_URL}/denoise/raw",
        data=noisy.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
        params={"sample_rate": sample_rate},
        timeout=60
    )
    elapsed = time.time() - start_time

    print(f"  Status: {resp.status_code}")
    print(f"  처리 시간: {elapsed:.3f}초")

    if resp.status_code == 200:
        enhanced = np.frombuffer(resp.content, dtype=np.float32)
        enhanced_rms = np.sqrt(np.mean(enhanced ** 2))

        print(f"  결과 샘플 수: {len(enhanced)}")
        print(f"  결과 RMS: {enhanced_rms:.4f}")

        # 잡음 감소량 계산
        if noisy_rms > 0:
            reduction_ratio = (noisy_rms - enhanced_rms) / noisy_rms * 100
            print(f"  RMS 감소율: {reduction_ratio:.1f}%")

        print("  ✅ Raw PCM 잡음 제거 성공")
        return True
    else:
        print(f"  ❌ Raw 잡음 제거 실패: {resp.text}")
        return False


def main():
    global DENOISE_SERVER_URL

    parser = argparse.ArgumentParser(description="DeepFilterNet 서버 테스트")
    parser.add_argument("--file", type=str, help="테스트할 WAV 파일 경로")
    parser.add_argument("--synthetic", action="store_true", help="합성 오디오로 테스트 (기본)")
    parser.add_argument("--url", type=str, default=DENOISE_SERVER_URL, help="서버 URL")
    args = parser.parse_args()

    DENOISE_SERVER_URL = args.url

    print("🔊 DeepFilterNet 잡음 제거 서버 테스트")
    print(f"   서버 URL: {DENOISE_SERVER_URL}")
    print()

    results = []

    # 1. 헬스체크
    if not test_health():
        print("\n⛔ 서버 연결 실패. 테스트를 중단합니다.")
        print("   서버를 먼저 실행해주세요:")
        print("   docker-compose up denoise_server")
        print("   또는")
        print("   cd denoise_server && python server.py")
        sys.exit(1)

    # 2. WAV 파일 테스트
    results.append(("WAV 잡음 제거", test_denoise_wav(args.file)))

    # 3. Raw PCM 테스트
    results.append(("Raw PCM 잡음 제거", test_denoise_raw()))

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} : {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 모든 테스트 통과!")
    else:
        print("\n⚠️  일부 테스트 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
