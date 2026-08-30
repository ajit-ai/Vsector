"""CDN edge integration — CloudFront / Cloudflare / Fastly stub.

Uses VSECTOR_CDN_URL for edge cache purge on namespace invalidation.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class CDN:
    def __init__(self, cdn_url: str | None = None):
        self.cdn_url = cdn_url or os.getenv("VSECTOR_CDN_URL") or ""

    @property
    def enabled(self) -> bool:
        return bool(self.cdn_url)

    def purge_namespace(self, namespace: str) -> bool:
        if not self.enabled:
            return False
        try:
            import httpx

            # Generic purge — provider-specific in prod (CloudFront CreateInvalidation, etc.)
            r = httpx.post(f"{self.cdn_url.rstrip('/')}/purge", json={"namespace": namespace}, timeout=2)
            logger.info(f"CDN purge {namespace} -> {r.status_code}")
            return r.status_code < 300
        except Exception as e:
            logger.warning(f"CDN purge failed: {e}")
            return False

    def cache_headers(self, ttl_s: int = 60) -> dict[str, str]:
        if not self.enabled:
            return {}
        return {
            "Cache-Control": f"public, max-age={ttl_s}",
            "CDN-Cache-Control": f"max-age={ttl_s}",
            "Vary": "X-API-Key",
        }
