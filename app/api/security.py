import math
import os
import secrets
import time
from collections import defaultdict, deque
from collections.abc import Callable
from threading import Lock

from fastapi import HTTPException


class RequestGuard:
    """Optional API-key authentication and per-client sliding-window limiting."""

    def __init__(
        self,
        api_key: str | None = None,
        request_limit: int = 60,
        window_seconds: float = 60,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if request_limit < 0:
            raise ValueError("request_limit must not be negative")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        self.api_key = api_key
        self.request_limit = request_limit
        self.window_seconds = window_seconds
        self.clock = clock
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()
        self._last_cleanup = self.clock()

    @classmethod
    def from_environment(cls) -> "RequestGuard":
        raw_limit = os.getenv("SEARCH_RATE_LIMIT_PER_MINUTE", "60")
        try:
            request_limit = int(raw_limit)
        except ValueError as error:
            raise ValueError("SEARCH_RATE_LIMIT_PER_MINUTE must be an integer") from error
        return cls(
            api_key=os.getenv("SEARCH_API_KEY") or None,
            request_limit=request_limit,
        )

    def check(self, provided_api_key: str | None, client_id: str) -> None:
        if self.api_key is not None and (
            provided_api_key is None
            or not secrets.compare_digest(provided_api_key, self.api_key)
        ):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        if self.request_limit == 0:
            return

        now = self.clock()
        cutoff = now - self.window_seconds
        with self._lock:
            if now - self._last_cleanup >= self.window_seconds:
                for identity, bucket in list(self._requests.items()):
                    while bucket and bucket[0] <= cutoff:
                        bucket.popleft()
                    if not bucket:
                        del self._requests[identity]
                self._last_cleanup = now

            requests = self._requests[client_id]
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if len(requests) >= self.request_limit:
                retry_after = max(1, math.ceil(requests[0] + self.window_seconds - now))
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(retry_after)},
                )
            requests.append(now)
