"""
diart 화자 분리 서버 테스트 스크립트

사용법:
  python test_diart.py                        # 합성 오디오로 테스트
  python test_diart.py --file /path/to/audio.wav  # WAV 파일로 테스트
"""

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

DIART_SERVER_URL = "http://localhost:8003"


def test_health():
    """1. 헬스체크"""
    print("=" * 60)
    print("[테스트 1] /health 헬스체크")
    print("=" * 60)
    try:
        resp = requests.get(f"{DIART_SERVER_URL}/health", timeout=10)
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {resp.json()}")

        if resp.status_code == 200 and resp.json().get("pipeline_loaded"):
            print("  ✅ 파이프라인 로드 완료")
            return True
        else:
            print("  ❌ 파이프라인 미로드")
            return False
    except requests.ConnectionError:
        print(f"  ❌ 서버 연결 실패: {DIART_SERVER_URL}")
        return False


def generate_two_speaker_audio(duration=6.0, sample_rate=16000):
    """2명 화자 시뮬레이션 오디오 (0~3초: 화자A, 3~6초: 화자B)"""
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    half = len(t) // 2

    speaker_a = np.zeros_like(t)
    speaker_a[:half] = (
        0.4 * np.sin(2 * np.pi * 200 * t[:half]) +
        0.2 * np.sin(2 * np.pi * 400 * t[:half])
    )

    speaker_b = np.zeros_like(t)
    speaker_b[half:] = (
        0.4 * np.sin(2 * np.pi * 350 * t[half:]) +
        0.2 * np.sin(2 * np.pi * 700 * t[half:])
    )

    mixed = (speaker_a + speaker_b + 0.02 * np.random.randn(len(t))).astype(np.float32)
    return mixed, sample_rate


def test_diart_wav(wav_path=None):
    """2. WAV 파일 화자 분리"""
    print("\n" + "=" * 60)
    print("[테스트 2] POST /diart (WAV)")
    print("=" * 60)

    if wav_path:
        print(f"  입력 파일: {wav_path}")
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
    else:
        print("  입력: 합성 2인 화자 (6초, 16kHz)")
        audio, sr = generate_two_speaker_audio()
        buf = io.BytesIO()
        if sf:
            sf.write(buf, audio, sr, format="WAV", subtype="FLOAT")
        else:
            from scipy.io import wavfile
            wavfile.write(buf, sr, audio)
        buf.seek(0)
        wav_bytes = buf.read()

    start_time = time.time()
    resp = requests.post(
        f"{DIART_SERVER_URL}/diart",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        timeout=120
    )
    elapsed = time.time() - start_time

    print(f"  Status: {resp.status_code} ({elapsed:.3f}초)")

    if resp.status_code == 200:
        data = resp.json()
        print(f"  화자 수: {data['num_speakers']}, 세그먼트: {len(data['segments'])}")
        for seg in data["segments"]:
            print(f"    [{seg['speaker']}] {seg['start']:.2f}s ~ {seg['end']:.2f}s")
        print("  ✅ WAV 화자 분리 성공")
        return True
    else:
        print(f"  ❌ 실패: {resp.text}")
        return False


def test_diart_raw():
    """3. Raw PCM 화자 분리"""
    print("\n" + "=" * 60)
    print("[테스트 3] POST /diart/raw (PCM bytes)")
    print("=" * 60)

    sample_rate = 16000
    audio, sr = generate_two_speaker_audio(duration=6.0, sample_rate=sample_rate)
    print(f"  입력: 합성 2인 화자 (6초, {sample_rate}Hz, {len(audio.tobytes())} bytes)")

    start_time = time.time()
    resp = requests.post(
        f"{DIART_SERVER_URL}/diart/raw",
        data=audio.tobytes(),
        headers={"Content-Type": "application/octet-stream"},
        params={"sample_rate": sample_rate},
        timeout=120
    )
    elapsed = time.time() - start_time

    print(f"  Status: {resp.status_code} ({elapsed:.3f}초)")

    if resp.status_code == 200:
        data = resp.json()
        print(f"  화자 수: {data['num_speakers']}, 세그먼트: {len(data['segments'])}")
        for seg in data["segments"]:
            print(f"    [{seg['speaker']}] {seg['start']:.2f}s ~ {seg['end']:.2f}s")
        print("  ✅ Raw PCM 화자 분리 성공")
        return True
    else:
        print(f"  ❌ 실패: {resp.text}")
        return False


def main():
    global DIART_SERVER_URL

    parser = argparse.ArgumentParser(description="diart 화자 분리 서버 테스트")
    parser.add_argument("--file", type=str, help="테스트 WAV 파일 경로")
    parser.add_argument("--url", type=str, default=DIART_SERVER_URL, help="서버 URL")
    args = parser.parse_args()

    DIART_SERVER_URL = args.url

    print("🎤 diart 화자 분리 서버 테스트")
    print(f"   서버: {DIART_SERVER_URL}\n")

    if not test_health():
        print("\n⛔ 서버 연결 실패.")
        print("   docker-compose up diart_server")
        sys.exit(1)

    results = [
        ("WAV 화자 분리", test_diart_wav(args.file)),
        ("Raw PCM 화자 분리", test_diart_raw()),
    ]

    print("\n" + "=" * 60)
    print("📋 결과 요약")
    print("=" * 60)
    for name, passed in results:
        print(f"  {'✅ PASS' if passed else '❌ FAIL'} : {name}")

    if all(r[1] for r in results):
        print("\n🎉 모든 테스트 통과!")
    else:
        print("\n⚠️  일부 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
