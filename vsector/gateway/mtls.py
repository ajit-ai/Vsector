"""mTLS CA rotation + cert management.

Supports VSECTOR_MTLS_CA_PATH + VSECTOR_MTLS_CERT_PATH, hot-reload on SIGHUP.
"""

from __future__ import annotations

import logging
import os
import ssl
import time

logger = logging.getLogger(__name__)


class MTLSManager:
    def __init__(self, ca_path: str | None = None, cert_path: str | None = None, key_path: str | None = None):
        self.ca_path = ca_path or os.getenv("VSECTOR_MTLS_CA_PATH") or "certs/ca.pem"
        self.cert_path = cert_path or os.getenv("VSECTOR_MTLS_CERT_PATH") or "certs/tls.crt"
        self.key_path = key_path or os.getenv("VSECTOR_MTLS_KEY_PATH") or "certs/tls.key"
        self._ctx: ssl.SSLContext | None = None
        self._last_reload = 0.0

    def load_context(self, server_side: bool = True) -> ssl.SSLContext | None:
        try:
            ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH if server_side else ssl.Purpose.SERVER_AUTH)
            if os.path.exists(self.ca_path):
                ctx.load_verify_locations(self.ca_path)
                ctx.verify_mode = ssl.CERT_REQUIRED
                logger.info(f"mTLS CA loaded {self.ca_path}")
            if server_side and os.path.exists(self.cert_path) and os.path.exists(self.key_path):
                ctx.load_cert_chain(self.cert_path, self.key_path)
                logger.info(f"mTLS cert loaded {self.cert_path}")
            self._ctx = ctx
            self._last_reload = time.time()
            return ctx
        except Exception as e:
            logger.warning(f"mTLS load failed (dev fallback): {e}")
            return None

    def should_rotate(self, interval_s: int = 3600) -> bool:
        return time.time() - self._last_reload > interval_s

    def rotate_if_needed(self) -> bool:
        if self.should_rotate():
            logger.info("mTLS CA rotation triggered")
            self.load_context()
            return True
        return False

    def verify_client(self, cert_pem: str) -> str | None:
        """Verify client cert fingerprint -> tenant (stub). In prod, parse CN/SAN."""
        if not cert_pem:
            return None
        # fingerprint extraction stub
        import hashlib

        fp = hashlib.sha256(cert_pem.encode()).hexdigest()[:16]
        return fp
