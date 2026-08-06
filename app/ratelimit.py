"""Per-client token-bucket rate limiter.

Every /api/chat call spends real money (a Claude request) and /api/voice
calls spend Whisper/TTS money, so both are capped per client IP. In-memory:
fine for the single-process deployment this project targets; put a shared
limiter (e.g. nginx limit_req or Redis) in front if you ever scale out.
"""

import threading
import time


class RateLimiter:
    def __init__(self, per_minute: int = 20, burst: int | None = None):
        self.rate = per_minute / 60.0          # tokens added per second
        self.burst = burst or max(5, per_minute // 2)
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, ts)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, ts = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - ts) * self.rate)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            # opportunistic cleanup so the dict can't grow unboundedly
            if len(self._buckets) > 10_000:
                cutoff = now - 600
                self._buckets = {k: v for k, v in self._buckets.items()
                                 if v[1] > cutoff}
            return True
