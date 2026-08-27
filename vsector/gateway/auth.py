"""API Gateway - Auth (JWT/OAuth2, API-Key, mTLS stub) + Rate limiting + Circuit breaker."""
from __future__ import annotations

import time
import asyncio
import hashlib
from collections import defaultdict

from fastapi import Header, HTTPException, Depends
from jose import jwt, JWTError  # type: ignore

from ..infra.config import get_settings

settings = get_settings()

# Simple in-memory API-Key store (in prod: DB/Redis)
API_KEYS: dict[str, dict] = {
    "vsector_demo_key": {"tenant": "demo", "quota": 10000},
    "test-key": {"tenant": "test", "quota": 10000},
}

# Rate limiting: token bucket per API key + per namespace
_buckets: dict[str, list[float]] = defaultdict(list)
_lock = asyncio.Lock()

async def rate_limiter(x_api_key: str | None = Header(default=None), x_namespace: str | None = Header(default=None)):
    key = x_api_key or "anonymous"
    now = time.time()
    # token bucket: allow burst, refill per second
    async with _lock:
        window = _buckets[key]
        # keep only last 1s
        _buckets[key] = [t for t in window if now - t < 1]
        quota = API_KEYS.get(key, {}).get("quota", settings.rate_limit_rps)
        if len(_buckets[key]) >= quota:
            raise HTTPException(status_code=429, detail="Rate limit exceeded", headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1"})
        _buckets[key].append(now)
    return key


def verify_api_key(x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)):
    # allow anonymous in dev
    if settings.env == "development":
        return {"sub": "dev"}
    if x_api_key and x_api_key in API_KEYS:
        return {"sub": API_KEYS[x_api_key]["tenant"], "api_key": x_api_key}
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
            return payload
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid JWT")
    raise HTTPException(status_code=401, detail="Missing API key or JWT")


def create_access_token(data: dict) -> str:
    from datetime import datetime, timedelta, timezone
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=settings.jwt_algorithm)


# Circuit breaker stub
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure = 0
        self.state = "CLOSED"  # CLOSED | OPEN | HALF_OPEN

    def record_success(self):
        self.failures = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"

    def allow(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN" and time.time() - self.last_failure > self.recovery_timeout:
            self.state = "HALF_OPEN"
            return True
        return self.state != "OPEN"
