"""S3 cold tier for SegmentStore — boto3 with local fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class S3Tier:
    """S3 cold tier. Uses VSECTOR_S3_BUCKET / AWS creds, falls back to local cold_dir."""

    def __init__(self, bucket: str | None = None, prefix: str | None = None, endpoint_url: str | None = None):
        self.bucket = bucket or os.getenv("VSECTOR_S3_BUCKET") or ""
        self.prefix = prefix or os.getenv("VSECTOR_S3_PREFIX") or "vsector/"
        self.endpoint_url = endpoint_url or os.getenv("VSECTOR_S3_ENDPOINT_URL") or None
        self._client = None
        if self.bucket:
            try:
                import boto3  # type: ignore

                self._client = boto3.client("s3", endpoint_url=self.endpoint_url)
                # probe
                self._client.head_bucket(Bucket=self.bucket)
                logger.info(f"S3Tier bucket {self.bucket} reachable")
            except Exception as e:
                logger.warning(f"S3 not available, using local fallback: {e}")
                self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None and bool(self.bucket)

    def upload(self, local_path: Path, key: str | None = None) -> str | None:
        if not self.enabled or self._client is None:
            return None
        k = key or f"{self.prefix.rstrip('/')}/{local_path.name}"
        try:
            self._client.upload_file(str(local_path), self.bucket, k)  # type: ignore
            logger.info(f"S3 uploaded {local_path} -> s3://{self.bucket}/{k}")
            return f"s3://{self.bucket}/{k}"
        except Exception as e:
            logger.warning(f"S3 upload failed: {e}")
            return None

    def download(self, key: str, dest: Path) -> bool:
        if not self.enabled or self._client is None:
            return False
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self.bucket, key, str(dest))  # type: ignore
            return True
        except Exception as e:
            logger.warning(f"S3 download failed: {e}")
            return False

    def list_cold(self, prefix: str | None = None) -> list[str]:
        if not self.enabled or self._client is None:
            return []
        pfx = prefix or self.prefix
        try:
            resp = self._client.list_objects_v2(Bucket=self.bucket, Prefix=pfx)  # type: ignore
            return [o["Key"] for o in resp.get("Contents", [])]
        except Exception as e:
            logger.warning(f"S3 list failed: {e}")
            return []
