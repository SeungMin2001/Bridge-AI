"""
실시간 MergePRAG 메모리 주입 클라이언트

역할:
  STT 전사문이 생성될 때마다 텍스트를 버퍼에 모아, llm_server의
  /memory/add API로 전송하여 HyperNetwork K,V를 실시간으로 축적한다.

사용법:
  injector = MemoryInjector(llm_server_url="http://localhost:8001")
  await injector.feed_text(session_id, "오늘은 정규화에 대해 배웁니다")
  stats = injector.get_stats(session_id)
  await injector.close_session(session_id)
"""

import asyncio
import os
import time
from dataclasses import dataclass, field

import httpx

# ── 환경변수 기반 설정 ──
# 메모리 주입 활성화 여부 (기본: true)
MEMORY_INJECT_ENABLED = os.getenv(
    "MEMORY_INJECT_ENABLED", "true"
).strip().lower() in {"1", "true", "yes", "on"}

# 최소 flush 크기 (글자 수). 이 이상 모이면 자동 전송
MIN_FLUSH_CHARS = int(os.getenv("MEMORY_INJECT_MIN_CHARS", "150"))

# 최대 flush 대기 시간 (초). 텍스트가 적어도 이 시간이 지나면 전송
MAX_FLUSH_INTERVAL = float(os.getenv("MEMORY_INJECT_INTERVAL", "10.0"))

# llm_server URL (기본: 같은 머신의 8001 포트)
MEMORY_SERVER_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8001")


@dataclass
class InjectionStats:
    """세션별 메모리 주입 통계를 추적한다."""
    session_id: str
    course_id: str
    total_injections: int = 0          # flush 횟수
    total_passages: int = 0            # 누적 passage 수
    total_chars: int = 0               # 누적 전송 글자 수
    failed_injections: int = 0         # 실패 횟수
    last_inject_time: float | None = None  # 마지막 주입 시각 (epoch)
    created_at: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """통계를 JSON 직렬화 가능한 dict로 반환한다."""
        return {
            "session_id": self.session_id,
            "course_id": self.course_id,
            "total_injections": self.total_injections,
            "total_passages": self.total_passages,
            "total_chars": self.total_chars,
            "failed_injections": self.failed_injections,
            "last_inject_time": self.last_inject_time,
            "created_at": self.created_at,
            "uptime_seconds": round(time.time() - self.created_at, 1),
            "recent_errors": self.errors[-5:],  # 최근 5개만
        }


class MemoryInjector:
    """실시간 전사문 → MergePRAG 메모리 주입 클라이언트.

    각 세션의 전사문 텍스트를 버퍼에 모아 일정 크기에 도달하면
    llm_server의 /memory/add 엔드포인트로 자동 전송한다.
    """

    def __init__(
        self,
        server_url: str = MEMORY_SERVER_URL,
        min_flush_chars: int = MIN_FLUSH_CHARS,
        max_flush_interval: float = MAX_FLUSH_INTERVAL,
    ):
        self.server_url = server_url.rstrip("/")
        self.min_flush_chars = min_flush_chars
        self.max_flush_interval = max_flush_interval
        self._enabled = MEMORY_INJECT_ENABLED

        # 세션별 텍스트 버퍼: {session_id: str}
        self._buffers: dict[str, str] = {}
        # 세션별 course_id 매핑: {session_id: course_id}
        self._course_map: dict[str, str] = {}
        # 세션별 마지막 flush 시각: {session_id: float}
        self._last_flush: dict[str, float] = {}
        # 세션별 통계: {session_id: InjectionStats}
        self._stats: dict[str, InjectionStats] = {}
        # 자동 flush 타이머 태스크: {session_id: asyncio.Task}
        self._timers: dict[str, asyncio.Task] = {}

        print(
            f"[MemoryInjector] 초기화 완료 "
            f"(enabled={self._enabled}, server={self.server_url}, "
            f"min_chars={self.min_flush_chars}, interval={self.max_flush_interval}s)"
        )

    def register_session(self, session_id: str, course_id: str | None = None):
        """세션 시작 시 호출. course_id가 없으면 session_id를 그대로 사용한다."""
        effective_course_id = course_id or session_id
        self._course_map[session_id] = effective_course_id
        self._buffers[session_id] = ""
        self._last_flush[session_id] = time.time()
        self._stats[session_id] = InjectionStats(
            session_id=session_id,
            course_id=effective_course_id,
        )
        print(
            f"[MemoryInjector] 세션 등록: session={session_id}, "
            f"course={effective_course_id}"
        )

    async def feed_text(self, session_id: str, text: str):
        """전사문 텍스트를 버퍼에 추가한다. 조건 충족 시 자동 flush한다."""
        if not self._enabled:
            return
        if not text or not text.strip():
            return
        if session_id not in self._buffers:
            # 미등록 세션이면 자동 등록
            self.register_session(session_id)

        # 버퍼에 텍스트 추가 (공백 구분)
        buf = self._buffers[session_id]
        self._buffers[session_id] = (buf + " " + text).strip() if buf else text.strip()

        current_len = len(self._buffers[session_id])
        elapsed = time.time() - self._last_flush.get(session_id, 0)

        # 조건 충족 시 자동 flush
        if current_len >= self.min_flush_chars or elapsed >= self.max_flush_interval:
            await self.flush(session_id)
        else:
            # 타이머 기반 flush 예약 (텍스트가 적어도 시간이 지나면 전송)
            self._schedule_timer(session_id)

    async def flush(self, session_id: str):
        """버퍼의 텍스트를 llm_server /memory/add로 즉시 전송한다."""
        if not self._enabled:
            return

        text = self._buffers.get(session_id, "").strip()
        if not text:
            return

        # 버퍼 비우기
        self._buffers[session_id] = ""
        self._last_flush[session_id] = time.time()
        # 타이머 취소
        self._cancel_timer(session_id)

        course_id = self._course_map.get(session_id, session_id)
        stats = self._stats.get(session_id)

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                resp = await client.post(
                    f"{self.server_url}/memory/add",
                    json={
                        "course_id": course_id,
                        "passage": text,
                    },
                )
                resp.raise_for_status()
                result = resp.json()

            if stats:
                stats.total_injections += 1
                stats.total_passages = result.get("passage_count", stats.total_passages + 1)
                stats.total_chars += len(text)
                stats.last_inject_time = time.time()

            print(
                f"[MemoryInjector] 주입 성공: session={session_id}, "
                f"chars={len(text)}, passages={result.get('passage_count', '?')}"
            )

        except Exception as e:
            if stats:
                stats.failed_injections += 1
                stats.errors.append(f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {e}")
            print(f"[MemoryInjector] 주입 실패: session={session_id}, error={e}")

    async def close_session(self, session_id: str) -> dict:
        """세션 종료. 남은 버퍼를 flush하고 최종 통계를 반환한다."""
        # 남은 버퍼 전송
        await self.flush(session_id)
        # 타이머 정리
        self._cancel_timer(session_id)

        stats = self.get_stats(session_id)
        print(
            f"[MemoryInjector] 세션 종료: session={session_id}, "
            f"injections={stats.get('total_injections', 0)}, "
            f"chars={stats.get('total_chars', 0)}"
        )

        # 정리 (메모리는 llm_server에 남아있음)
        self._buffers.pop(session_id, None)
        self._last_flush.pop(session_id, None)

        return stats

    def get_stats(self, session_id: str) -> dict:
        """세션별 주입 통계를 반환한다."""
        stats = self._stats.get(session_id)
        if stats:
            return stats.to_dict()
        return {"session_id": session_id, "error": "no stats available"}

    def get_all_stats(self) -> list[dict]:
        """모든 활성 세션의 통계를 반환한다."""
        return [s.to_dict() for s in self._stats.values()]

    def _schedule_timer(self, session_id: str):
        """최대 flush 간격 후 자동 flush하는 타이머를 예약한다."""
        self._cancel_timer(session_id)

        async def _timer_flush():
            await asyncio.sleep(self.max_flush_interval)
            await self.flush(session_id)

        self._timers[session_id] = asyncio.create_task(_timer_flush())

    def _cancel_timer(self, session_id: str):
        """예약된 타이머를 취소한다."""
        task = self._timers.pop(session_id, None)
        if task and not task.done():
            task.cancel()


# ── 모듈 레벨 싱글턴 ──
_injector: MemoryInjector | None = None


def get_injector() -> MemoryInjector:
    """앱 전역에서 사용할 MemoryInjector 싱글턴을 반환한다."""
    global _injector
    if _injector is None:
        _injector = MemoryInjector()
    return _injector
